"""--==The SarahMemory Project==--
File: SarahMemoryCognitiveServices.py
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

PURPOSE (v9.0.0):
- This module is the COGNITIVE GOVERNOR (Cortex / Judge) of SarahMemory AiOS.
- It does NOT execute upgrades, patches, file writes, or background schedulers.
- It evaluates intent, risk, ethics, safety flags, and the user's autonomy rules.
- It returns structured decisions:
ALLOW / DENY / DEFER / REQUIRE_USER
- It is OFFLINE-FIRST by default and enforces kill-switch behavior.

DESIGN RULES (OWNER-ALIGNED):
- Never become runaway: autonomy is gated by SarahMemoryGlobals + NEOSKYMATRIX.
- Never modify core files directly (execution belongs elsewhere).
- Never silently enable network access; online providers must be explicitly enabled.
- Prefer inaction over unsafe action.
- When uncertain, request more proof/metadata rather than guessing.

COMPATIBILITY:
- Preserves legacy entry points (analyze_text, analyze_image, process_cognitive_request),
but routes them through governance and safe defaults.

===============================================================================
COGNITIVE QUESTIONING (THE HEART):
This file implements a deterministic “self-questioning” framework:
- It asks itself structured questions per intent category
- It answers those questions from:
- policy snapshot (Globals)
- caller context
- proposed_action metadata (optional)
- It produces:
- decision: ALLOW / DENY / DEFER / REQUIRE_USER
- risk_score: 0..100
- risk_factors: list
- reasons: list
- recommended_next: routing guidance
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "governor"
# CATEGORY = "governance"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_governor"
# FAMILY = "core_governance"
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
# NOTES = "Cognitive governor / judge. Returns ALLOW, DENY, DEFER, REQUIRE_USER. Does not execute upgrades, patches, file writes, or schedulers."
# --- SARAHMETA END ---

import importlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
import time
from typing import Any, Dict, Optional, Tuple
import threading
import uuid
import SarahMemoryGlobals as config

try:
    import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
except Exception:
    _CogSelf = None

try:
    import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
except Exception:
    _OperatorCore = None

try:
    import SarahMemorySecurityGovernor as _SecurityGovernor  # type: ignore
except Exception:
    _SecurityGovernor = None

try:
    import SarahMemoryAssuranceGate as _AssuranceGate  # type: ignore
except Exception:
    _AssuranceGate = None

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveServices")
logger.setLevel(logging.DEBUG)
_null = logging.NullHandler()
_null.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_null)

# Thread-local guard to prevent recursive dual-governor consultation
_THINKER_CONSULT_STATE = threading.local()

# -----------------------------------------------------------------------------
# Safety defaults (offline-first)
# -----------------------------------------------------------------------------
if not hasattr(config, "COGNITIVE_ONLINE_ENABLED"):
    config.COGNITIVE_ONLINE_ENABLED = False  # default OFF for safety

# Local cognitive fallback data path (owner-controlled, optional)
LOCAL_COGNITIVE_DATA_PATH = os.path.join(getattr(config, "DATA_DIR", os.getcwd()), "local_cognitive.json")

# Legacy vendor endpoints (kept for backward compatibility, but governed)
TEXT_ANALYSIS_ENDPOINT = os.environ.get(
    "COG_TEXT_ANALYSIS_ENDPOINT",
    "https://api.cognitive.microsoft.com/text/analytics/v3.0/sentiment",
)
TEXT_ANALYSIS_KEY = os.environ.get("COG_TEXT_ANALYSIS_KEY", "YOUR_TEXT_ANALYSIS_KEY")

IMAGE_ANALYSIS_ENDPOINT = os.environ.get(
    "COG_IMAGE_ANALYSIS_ENDPOINT",
    "https://api.cognitive.microsoft.com/vision/v3.2/analyze",
)
IMAGE_ANALYSIS_KEY = os.environ.get("COG_IMAGE_ANALYSIS_KEY", "YOUR_IMAGE_ANALYSIS_KEY")

# -----------------------------------------------------------------------------
# DB paths (MUST align to SarahMemoryGlobals portable paths)
# -----------------------------------------------------------------------------
def _datasets_dir() -> str:
    try:
        return getattr(
            config,
            "DATASETS_DIR",
            os.path.join(getattr(config, "DATA_DIR", os.getcwd()), "memory", "datasets"),
        )
    except Exception:
        return os.path.join(os.getcwd(), "data", "memory", "datasets")


def _system_logs_db() -> str:
    return os.path.join(_datasets_dir(), "system_logs.db")


# -----------------------------------------------------------------------------
# DB helpers (NO import-time side effects)
# -----------------------------------------------------------------------------
def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    con = sqlite3.connect(
        db_path,
        timeout=5.0,
        check_same_thread=False,  # safe for multi-threaded callers (we open/close per call)
    )

    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass

    return con


def _ensure_tables() -> None:
    """
    Ensures cognitive governor tables exist.
    Called on-demand (no import-time side effects).
    """
    db_path = _system_logs_db()
    con: Optional[sqlite3.Connection] = None
    try:
        con = _connect(db_path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_governor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                severity TEXT,
                event TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as e:
        logger.debug("Cognitive governor DB ensure failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def log_cognitive_event(event: str, details: str, severity: str = "INFO", meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Writes a structured event into system_logs.db (datasets).
    """
    try:
        _ensure_tables()
        db_path = _system_logs_db()
        con = _connect(db_path)
        cur = con.cursor()
        ts = datetime.now().isoformat()
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False)
        except Exception:
            meta_json = "{}"
        cur.execute(
            "INSERT INTO cognitive_governor_events (ts, severity, event, details, meta_json) VALUES (?, ?, ?, ?, ?)",
            (ts, str(severity), str(event), str(details), meta_json),
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.debug("Failed to log cognitive event: %s", e)

# -----------------------------------------------------------------------------
# Cognitive -> Optimization partition publish (bounded scratch arenas)
# -----------------------------------------------------------------------------
def _publish_to_optimization_partition(role: str, record: Dict[str, Any]) -> None:
    """
    Best-effort bridge:
    - Writes small governance traces into SarahMemoryOptimization's cognitive partitions.
    - No hard dependency: safe for headless/cloud or when Optimization isn't initialized.
    """
    try:
        import SarahMemoryOptimization as opt  # local import avoids circular boot issues
        fn = getattr(opt, "publish_cognitive_record", None)
        if callable(fn):
            fn(role, record)
    except Exception:
        # Silent by design: governance must never crash the runtime.
        return


def _route_role_for_decision(dec: Dict[str, Any], caller_context: Optional[Dict[str, Any]] = None) -> str:
    """Deterministic mapping of decision -> cognitive partition role."""
    intent = str(dec.get("intent") or "")
    decision = str(dec.get("decision") or "")
    ctx = caller_context or {}

    # If caller indicates sandbox results are verified, move to deploy stage.
    if bool(ctx.get("sandbox_verified")) or bool(ctx.get("sandbox_complete")):
        return "deploy"

    if intent in ("DIAGNOSTICS", "CHAT", "NETWORK_ACCESS"):
        return "monitor"

    if intent == "PATCH_OR_UPDATE":
        if decision == "ALLOW":
            return "test"   # approved changes go into sandbox test lane next
        return "improve"    # denied/deferred require improvement/clarification lane

    if intent == "CREATIVE_REQUEST":
        questions.extend(
            [
                "Is this a creative artifact request rather than a direct system mutation?",
                "Can the request be routed to Creative Lane and resolved through registered CreativeStudios modules?",
                "Does the request require local filesystem writes, preview, or browser-open actions that may need confirmation?",
            ]
        )

        creative_targets = [m for m in _suggest_module_hints(intent) if _is_core_module_approved(m)]
        answers["creative_registered_modules"] = creative_targets

        if not creative_targets:
            decision["decision"] = "DEFER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("No approved CreativeStudios modules are currently registered for this request.")
            decision["recommended_next"] = "Register/approve a CreativeStudios module, then re-evaluate through Creative Lane."
            _risk_add(risk, 10, "no_registered_creative_modules")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = False
        decision["reasons"].append("Creative request approved for governed routing; execution belongs to Creative Lane.")
        decision["recommended_next"] = "Route to Creative Lane and selected CreativeStudios module; validate artifacts before final presentation."
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        return _finalize(decision)


    if intent == "FILESYSTEM_WRITE":
        if decision == "ALLOW":
            return "deploy"
        return "improve"

    # Default: monitoring lane
    return "monitor"
# -----------------------------------------------------------------------------
# Partition 3: Virtual Code Sandbox (Developer Mode gated)
# -----------------------------------------------------------------------------

_DEVMODE_CACHE: Optional[bool] = None

def _developers_mode_enabled() -> bool:
    """
    Developer-mode gate. Reads SarahMemoryGlobals.py first, then .env/.process env.
    """
    global _DEVMODE_CACHE
    if _DEVMODE_CACHE is not None:
        return bool(_DEVMODE_CACHE)

    v = getattr(config, "DEVELOPERSMODE", None)
    if v is None:
        v = os.getenv("DEVELOPERSMODE", None)

    if isinstance(v, bool):
        _DEVMODE_CACHE = v
        return bool(_DEVMODE_CACHE)

    s = str(v or "").strip().lower()
    _DEVMODE_CACHE = s in ("1", "true", "yes", "on", "enabled")
    return bool(_DEVMODE_CACHE)


_VSANDBOX_LOCK = threading.RLock()
_VSANDBOX_MAX = 32
_VSANDBOX_TTL_S = 60 * 60 * 8  # 8 hours
_VSANDBOX: Dict[str, Dict[str, Any]] = {}


def _vsandbox_prune(now: Optional[float] = None) -> None:
    now = float(now or time.time())
    with _VSANDBOX_LOCK:
        # TTL prune
        dead = []
        for sid, rec in _VSANDBOX.items():
            try:
                ts = float(rec.get("created_epoch", 0.0))
            except Exception:
                ts = 0.0
            if ts and (now - ts) > _VSANDBOX_TTL_S:
                dead.append(sid)
        for sid in dead:
            _VSANDBOX.pop(sid, None)

        # Size prune (oldest first)
        if len(_VSANDBOX) > _VSANDBOX_MAX:
            items = sorted(_VSANDBOX.items(), key=lambda kv: float(kv[1].get("created_epoch", 0.0)))
            for sid, _ in items[: max(0, len(_VSANDBOX) - _VSANDBOX_MAX)]:
                _VSANDBOX.pop(sid, None)


def create_virtual_code_artifact(
    *,
    title: str,
    code: str,
    language: str = "python",
    reason: str = "",
    caller: str = "unknown",
    intent: str = "PATCH_OR_UPDATE",
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates a Partition-3 sandbox artifact for USER REVIEW ONLY.
    - No execution
    - No filesystem writes
    - Short preview returned for UI wiring
    """
    if not _developers_mode_enabled():
        return {}

    _vsandbox_prune()

    sid = uuid.uuid4().hex
    created_epoch = time.time()
    created_iso = datetime.now().isoformat()

    code_str = "" if code is None else str(code)
    preview = code_str.strip().replace("\r\n", "\n")[:1200]
    if len(code_str) > 1200:
        preview += "\n...<truncated>..."

    rec = {
        "id": sid,
        "created_iso": created_iso,
        "created_epoch": created_epoch,
        "title": (title or "Virtual Sandbox Artifact").strip()[:120],
        "language": (language or "python").strip()[:32],
        "reason": (reason or "").strip()[:500],
        "caller": (caller or "unknown").strip()[:120],
        "intent": (intent or "PATCH_OR_UPDATE").strip()[:64],
        "code": code_str,
        "code_preview": preview,
        "proposed_action": proposed_action or {},
        "status": "PENDING_REVIEW",  # UI can flip this to ACCEPTED/REJECTED
        "notes": [],
    }

    with _VSANDBOX_LOCK:
        _VSANDBOX[sid] = rec

    # Also publish a lightweight pointer into Optimization Partition stream (best-effort)
    try:
        _publish_to_optimization_partition(
            "test",
            {
                "ts": created_iso,
                "role": "test",
                "event": "VirtualSandboxArtifactCreated",
                "sandbox_id": sid,
                "title": rec["title"],
                "language": rec["language"],
                "caller": rec["caller"],
                "intent": rec["intent"],
            },
        )
    except Exception:
        pass

    return {
        "sandbox_id": sid,
        "status": rec["status"],
        "title": rec["title"],
        "language": rec["language"],
        "created_iso": created_iso,
        "code_preview": rec["code_preview"],
    }


def get_virtual_code_artifact(sandbox_id: str) -> Optional[Dict[str, Any]]:
    if not _developers_mode_enabled():
        return None
    _vsandbox_prune()
    with _VSANDBOX_LOCK:
        rec = _VSANDBOX.get(str(sandbox_id or "").strip())
        return dict(rec) if isinstance(rec, dict) else None


def list_virtual_code_artifacts() -> list:
    if not _developers_mode_enabled():
        return []
    _vsandbox_prune()
    with _VSANDBOX_LOCK:
        out = []
        for rec in _VSANDBOX.values():
            out.append({
                "sandbox_id": rec.get("id"),
                "created_iso": rec.get("created_iso"),
                "title": rec.get("title"),
                "language": rec.get("language"),
                "status": rec.get("status"),
                "caller": rec.get("caller"),
                "intent": rec.get("intent"),
            })
        # newest first
        out.sort(key=lambda r: r.get("created_iso") or "", reverse=True)
        return out


def _extract_virtual_code_from_proposed_action(pa: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pulls 'code' from the proposed_action payload using common keys.
    Returns (code, language)
    """
    if not isinstance(pa, dict):
        return "", "python"

    lang = pa.get("language") or pa.get("lang") or "python"
    for k in ("code", "patch", "diff", "proposed_code", "full_file", "content"):
        v = pa.get(k)
        if isinstance(v, str) and v.strip():
            return v, str(lang)

    # allow nested payloads
    nested = pa.get("payload") or pa.get("proposal") or {}
    if isinstance(nested, dict):
        for k in ("code", "patch", "diff", "proposed_code", "full_file", "content"):
            v = nested.get(k)
            if isinstance(v, str) and v.strip():
                return v, str(lang)

    return "", str(lang)


def maybe_attach_virtual_sandbox(
    *,
    decision: Dict[str, Any],
    request_text: str,
    intent: str,
    caller: str,
    proposed_action: Dict[str, Any],
) -> None:
    """
    Attaches a Partition-3 sandbox pointer to the decision payload when:
    - DEVELOPERSMODE is enabled
    - This is a patch/update allowance
    - Proposed action includes code/patch/diff content
    """
    if not _developers_mode_enabled():
        return
    if (intent or "").upper() != "PATCH_OR_UPDATE":
        return
    if str(decision.get("decision") or "").upper() != "ALLOW":
        return

    code, lang = _extract_virtual_code_from_proposed_action(proposed_action or {})
    if not code.strip():
        return

    title = (proposed_action or {}).get("title") or (proposed_action or {}).get("change_type") or "Proposed Patch"
    reason = (proposed_action or {}).get("reason") or "Patch proposal (reason not provided)"

    artifact = create_virtual_code_artifact(
        title=str(title),
        code=code,
        language=str(lang),
        reason=str(reason),
        caller=str(caller or "unknown"),
        intent=str(intent or "PATCH_OR_UPDATE"),
        proposed_action=proposed_action or {},
    )
    if artifact:
        decision["virtual_sandbox"] = artifact
        decision["reasons"].append("DEVELOPERSMODE: packaged proposal into Partition-3 Virtual Sandbox for user review (no execution).")


# -----------------------------------------------------------------------------
# Self-model / policy snapshot
# -----------------------------------------------------------------------------
def get_cognitive_policy_snapshot() -> Dict[str, Any]:
    """
    Lightweight snapshot of the current safety / identity flags.
    This is NOT a claim of sentience; it's an engineered self-model.
    """
    snap = {
        "ts": datetime.now().isoformat(),
        "base_dir": getattr(config, "BASE_DIR", None),
        "data_dir": getattr(config, "DATA_DIR", None),
        "datasets_dir": getattr(config, "DATASETS_DIR", None),
        "context_engine_enabled": bool(
            getattr(config, "CONTEXT_ENGINE_ENABLED", getattr(config, "ENABLE_CONTEXT_BUFFER", True))
        ),
        "cognitive_online_enabled": bool(getattr(config, "COGNITIVE_ONLINE_ENABLED", False)),
        "kill_switch_neoskymatrix": bool(getattr(config, "NEOSKYMATRIX", False)),
    }
    try:
        gp = getattr(config, "sm_get_core_governance_profile", None)
        if callable(gp):
            snap["core_governance"] = gp() or {}
    except Exception:
        snap["core_governance"] = {}
    try:
        cself = _get_cognitive_self_packet("", {}, force_refresh=False)
        if cself:
            snap["cognitive_self_summary"] = _self_summary_from_packet(cself)
            snap["cognitive_self_status"] = cself.get("status") or {}
            snap["cognitive_self_temporal_awareness"] = cself.get("temporal_awareness") or {}
            snap["tri_force_contract"] = cself.get("tri_force_contract") or {}
    except Exception:
        pass
    return snap


def _get_cognitive_self_packet(request_text: str = "", caller_context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    ctx = dict(caller_context or {})
    if request_text and not ctx.get("request_text") and not ctx.get("text"):
        ctx["request_text"] = str(request_text)
    if not _CogSelf:
        return {}
    try:
        fn = getattr(_CogSelf, "get_governor_consumer_packet", None)
        if callable(fn):
            pkt = fn(request_text=request_text, context=ctx, force_refresh=force_refresh)
            return pkt if isinstance(pkt, dict) else {}
    except Exception as e:
        logger.debug("CognitiveSelf governor packet failed: %s", e)
    return {}


def _self_summary_from_packet(pkt: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pkt, dict):
        return {}
    status = pkt.get("status") if isinstance(pkt.get("status"), dict) else {}
    identity = pkt.get("identity") if isinstance(pkt.get("identity"), dict) else {}
    temporal = pkt.get("temporal_awareness") if isinstance(pkt.get("temporal_awareness"), dict) else {}
    realtime = pkt.get("realtime_strategy") if isinstance(pkt.get("realtime_strategy"), dict) else {}
    return {
        "name": identity.get("name") or identity.get("entity_name"),
        "platform": identity.get("platform"),
        "node_name": identity.get("node_name"),
        "run_mode": status.get("run_mode"),
        "device_mode": status.get("device_mode"),
        "safe_mode": status.get("safe_mode"),
        "local_only": status.get("local_only"),
        "online_connectivity": temporal.get("online_connectivity"),
        "continuity_state": status.get("continuity_state"),
        "realtime_requested": realtime.get("realtime_requested"),
    }


# -----------------------------------------------------------------------------
# SMGET preview / hard-gate bridge
# -----------------------------------------------------------------------------
_SMGET_INTENTS = {
    "PATCH_OR_UPDATE",
    "FILESYSTEM_WRITE",
    "NETWORK_ACCESS",
    "PRIVACY_SENSITIVE",
    "EXECUTE_COMMAND",
    "CREATIVE_REQUEST",
}


def _robotic_action_governance_profile(proposed_action: Optional[Dict[str, Any]] = None, request_text: str = "") -> Dict[str, Any]:
    pa = proposed_action if isinstance(proposed_action, dict) else {}
    joined = " ".join([
        str(request_text or "").lower(),
        str(pa.get("action_type") or "").lower(),
        str(pa.get("executor_name") or "").lower(),
        str(pa.get("capability_name") or "").lower(),
        str(pa.get("target") or "").lower(),
        str(pa.get("body_part") or pa.get("target_body_part") or "").lower(),
        " ".join(str(x).lower() for x in (pa.get("required_permissions") or [])),
    ])
    is_robotic = any(k in joined for k in ("robot", "servo", "gripper", "arm", "hand", "leg", "locomotion", "walk", "move", "posture", "torque", "force", "humanoid", "moya"))
    motion = is_robotic and any(k in joined for k in ("move", "walk", "step", "reach", "raise", "lower", "turn", "grip", "release", "posture"))
    contact = is_robotic and any(k in joined for k in ("human contact", "touch human", "grab person", "push", "pull", "intervene"))
    emergency = is_robotic and any(k in joined for k in ("emergency", "fire", "medical", "collision", "life", "rescue", "safe_stop", "e-stop"))
    return {
        "is_robotic_body_action": bool(is_robotic),
        "motion_requested": bool(motion),
        "human_contact_requested": bool(contact),
        "emergency_context": bool(emergency),
        "requires_smget": bool(is_robotic),
        "requires_operatorcore": bool(is_robotic),
        "requires_assurance": bool(is_robotic),
        "requires_security": bool(is_robotic),
        "requires_current_perception": bool(motion or contact),
        "requires_safe_stop": bool(is_robotic),
        "execution_authority": False,
    }


def _intent_uses_smget(intent: str, proposed_action: Optional[Dict[str, Any]] = None) -> bool:
    label = str(intent or "").strip().upper()
    if label in _SMGET_INTENTS:
        return True
    pa = proposed_action or {}
    if isinstance(pa, dict) and any(pa.get(k) for k in ("action_type", "executor_name", "required_permissions", "paths", "target_files", "subsystems")):
        return True
    if _robotic_action_governance_profile(pa).get("is_robotic_body_action"):
        return True
    return False


def _normalize_execution_mode(intent: str, proposed_action: Optional[Dict[str, Any]] = None, *, user_consented: bool = False) -> str:
    pa = proposed_action or {}
    raw = str(pa.get("execution_mode") or pa.get("mode") or "").strip().lower()
    if raw in ("simulate", "draft", "apply", "rollback"):
        return raw
    if pa.get("dry_run") is True:
        return "simulate"
    if str(intent or "").upper() in {"DIAGNOSTICS", "SYSTEM_INFO"}:
        return "simulate"
    if pa.get("apply") is True and user_consented:
        return "apply"
    return "draft"


def _build_smget_contract_preview(
    request_text: str,
    *,
    caller: str,
    caller_context: Optional[Dict[str, Any]],
    proposed_action: Optional[Dict[str, Any]],
    user_consented: bool,
    governance: Dict[str, Any],
) -> Dict[str, Any]:
    if _OperatorCore is None:
        return {}
    try:
        fn = getattr(_OperatorCore, "build_action_contract", None)
        if not callable(fn):
            return {}

        meta = dict(caller_context or {})
        mode_flags = meta.get("mode_flags") if isinstance(meta.get("mode_flags"), dict) else {}
        execution_mode = _normalize_execution_mode(str(governance.get("intent") or ""), proposed_action, user_consented=user_consented)
        meta.setdefault("mode_flags", mode_flags)
        meta["governance_decision"] = str(governance.get("decision") or "")
        meta["governance_risk_score"] = int(governance.get("risk_score") or 0)
        meta["governance_risk_factors"] = list(governance.get("risk_factors") or [])
        meta["governor_role"] = MODULE_NAME
        meta["smget_preview_only"] = True
        meta["user_consented"] = bool(user_consented)
        contract = fn(
            request_text,
            origin=caller or MODULE_NAME,
            meta=meta,
            proposed_action=proposed_action or {},
            execution_mode=execution_mode,
        )
        if hasattr(contract, "to_dict") and callable(getattr(contract, "to_dict")):
            data = contract.to_dict()
        elif isinstance(contract, dict):
            data = contract
        else:
            data = {}
        if isinstance(data, dict):
            data.setdefault("execution_mode", execution_mode)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("SMGET contract preview build failed: %s", e)
        return {}


def _smget_security_review(action_contract: Dict[str, Any], governance: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action_contract, dict) or not action_contract:
        return {}
    if _SecurityGovernor is None:
        return {
            "decision": "ALLOW",
            "allow": True,
            "reasons": ["SecurityGovernor unavailable; SMGET preview remains governance-only."],
        }
    for fn_name in ("evaluate_action", "govern_action", "review_action_contract"):
        try:
            fn = getattr(_SecurityGovernor, fn_name, None)
            if callable(fn):
                out = fn(action_contract, governance)
                if isinstance(out, dict):
                    return out
        except Exception as e:
            logger.debug("Security review bridge failed via %s: %s", fn_name, e)
    return {}


def _smget_assurance_review(action_contract: Dict[str, Any], governance: Dict[str, Any], security: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action_contract, dict) or not action_contract:
        return {}
    if _AssuranceGate is None:
        return {
            "decision": "ALLOW",
            "allow": True,
            "confidence": 1.0,
            "assurance_score": 1.0,
            "reasons": ["AssuranceGate unavailable; SMGET preview remains governance/security-only."],
        }
    for fn_name in ("evaluate_action_assurance", "review_action_assurance", "assure_action"):
        try:
            fn = getattr(_AssuranceGate, fn_name, None)
            if callable(fn):
                out = fn(action_contract, governance, security)
                if isinstance(out, dict):
                    return out
        except Exception as e:
            logger.debug("Assurance review bridge failed via %s: %s", fn_name, e)
    return {}


def _apply_smget_review_to_decision(dec: Dict[str, Any], smget: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(smget, dict) or not smget:
        return dec

    dec["smget"] = smget
    dec.setdefault("reasons", [])
    dec.setdefault("risk_factors", [])
    if not isinstance(dec.get("trace"), dict):
        dec["trace"] = {}
    dec["trace"]["smget"] = {
        "enabled": True,
        "contract_id": (smget.get("action_contract") or {}).get("contract_id"),
        "executor_name": (smget.get("action_contract") or {}).get("executor_name"),
        "execution_mode": (smget.get("action_contract") or {}).get("execution_mode"),
        "security_decision": (smget.get("security") or {}).get("decision"),
        "assurance_decision": (smget.get("assurance") or {}).get("decision"),
    }

    security = smget.get("security") if isinstance(smget.get("security"), dict) else {}
    assurance = smget.get("assurance") if isinstance(smget.get("assurance"), dict) else {}

    sec_decision = str(security.get("decision") or "").strip().lower()
    sec_allow = bool(security.get("allow", sec_decision == "allow"))

    assurance_decision = str(assurance.get("decision") or "").strip().lower()
    assurance_allow = bool(assurance.get("allow", assurance_decision == "allow"))

    if not sec_allow:
        if sec_decision in {"require_user"}:
            dec["decision"] = "REQUIRE_USER"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = True
        elif sec_decision in {"simulate_only", "allow_with_constraints"}:
            dec["decision"] = "DEFER"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = bool(security.get("require_user", dec.get("require_user")))
        else:
            dec["decision"] = "DENY"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = bool(security.get("require_user", False))

        for reason in security.get("reasons") or []:
            if reason not in dec["reasons"]:
                dec["reasons"].append(reason)
        for factor in security.get("risk_factors") or []:
            if factor not in dec["risk_factors"]:
                dec["risk_factors"].append(factor)
        dec["recommended_next"] = security.get("recommended_next") or dec.get("recommended_next")
        return dec

    if not assurance_allow:
        if assurance_decision == "require_user":
            dec["decision"] = "REQUIRE_USER"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = True
        elif assurance_decision == "simulate_only":
            dec["decision"] = "DEFER"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = bool(assurance.get("require_user", dec.get("require_user")))
        else:
            dec["decision"] = "DEFER"
            dec["allow"] = False
            dec["execution_allowed"] = False
            dec["require_user"] = bool(assurance.get("require_user", dec.get("require_user", True)))

        for reason in assurance.get("reasons") or []:
            if reason not in dec["reasons"]:
                dec["reasons"].append(reason)
        for factor in assurance.get("risk_factors") or []:
            if factor not in dec["risk_factors"]:
                dec["risk_factors"].append(factor)
        dec["recommended_next"] = assurance.get("recommended_next") or "Strengthen confidence / verification / rollback plan, then resubmit through SMGET."
        return dec

    dec["execution_allowed"] = bool(dec.get("allow"))
    if dec.get("decision") == "ALLOW":
        dec["recommended_next"] = "Route approved ActionContract into SarahMemoryOperatorCore through Neuron/app ingress for bounded execution."
    return dec


def _maybe_attach_smget_preview(
    dec: Dict[str, Any],
    *,
    request_text: str,
    caller: str,
    caller_context: Optional[Dict[str, Any]],
    user_consented: bool,
    proposed_action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(dec, dict):
        return dec
    if not _intent_uses_smget(str(dec.get("intent") or ""), proposed_action):
        return dec
    if str(dec.get("decision") or "").upper() == "DENY":
        return dec

    contract = _build_smget_contract_preview(
        request_text,
        caller=caller,
        caller_context=caller_context,
        proposed_action=proposed_action,
        user_consented=user_consented,
        governance=dec,
    )
    if not contract:
        return dec

    security = _smget_security_review(contract, dec)
    assurance = _smget_assurance_review(contract, dec, security)

    smget = {
        "enabled": True,
        "preview_only": True,
        "action_contract": contract,
        "security": security,
        "assurance": assurance,
    }
    return _apply_smget_review_to_decision(dec, smget)


# -----------------------------------------------------------------------------
# Core-governed routing helpers
# -----------------------------------------------------------------------------
_KNOWN_CORE_MODULES = {
    "SarahMemoryAdvCU",
    "SarahMemoryAiFunctions",
    "SarahMemoryAPI",
    "SarahMemoryCanvasStudio",
    "SarahMemoryCognitiveServices",
    "SarahMemoryCompare",
    "SarahMemoryDatabase",
    "SarahMemoryDiagnostics",
    "SarahMemoryFilesystem",
    "SarahMemoryGlobals",
    "SarahMemoryLogicCalc",
    "SarahMemoryLyricsToSong",
    "SarahMemoryMusicGenerator",
    "SarahMemoryNetwork",
    "SarahMemoryNeuron",
    "SarahMemoryReply",
    "SarahMemoryResearch",
    "SarahMemorySynapes",
    "SarahMemoryVideoEditorCore",
    "SarahMemoryVoice",
    "SarahMemoryWebSYM",
}

def _is_core_module_approved(module_name: str, capability: Optional[str] = None) -> bool:
    """
    Check the Globals core registry when available.
    Conservative fallback: only allow known built-in SarahMemory modules.
    """
    mn = os.path.splitext(os.path.basename(str(module_name or "").strip()))[0]
    if not mn:
        return False
    try:
        fn = getattr(config, "sm_is_core_module_approved", None)
        if callable(fn):
            return bool(fn(mn, capability=capability))
    except Exception:
        pass
    return mn in _KNOWN_CORE_MODULES

def _lane_family_for_intent(intent: str) -> str:
    label = str(intent or "").strip().upper()
    if label in ("FILESYSTEM_WRITE", "EXECUTE_COMMAND"):
        return "ACTION"
    if label in ("CREATIVE_REQUEST",):
        return "CREATIVE"
    if label in ("DIAGNOSTICS", "SYSTEM_INFO", "PATCH_OR_UPDATE", "EMERGENCY_INSTINCT"):
        return "SYSTEM"
    if label in ("NETWORK_ACCESS",):
        return "NETWORK"
    return "ANSWER"

def _suggest_module_hints(intent: str) -> list:
    label = str(intent or "").strip().upper()
    if label == "PATCH_OR_UPDATE":
        return ["SarahMemoryCompare", "SarahMemorySynapes"]
    if label == "FILESYSTEM_WRITE":
        return ["SarahMemoryFilesystem", "appsys"]
    if label == "EXECUTE_COMMAND":
        return ["SarahMemoryIntegration", "SarahMemoryFilesystem"]
    if label == "CREATIVE_REQUEST":
        return [
            "SarahMemoryCanvasStudio",
            "SarahMemoryVideoEditorCore",
            "SarahMemoryMusicGenerator",
            "SarahMemoryLyricsToSong",
        ]
    if label == "DIAGNOSTICS":
        return ["SarahMemoryDiagnostics"]
    if label == "SYSTEM_INFO":
        return ["SarahMemoryDiagnostics", "SarahMemoryGlobals"]
    if label == "NETWORK_ACCESS":
        return ["SarahMemoryResearch", "SarahMemoryNetwork", "appnet"]
    return ["SarahMemoryLogicCalc", "SarahMemoryWebSYM", "SarahMemoryResearch", "SarahMemoryAPI"]

def _validate_scope_modules(target_files: list, subsystems: list) -> Dict[str, Any]:
    """
    Ensure declared SarahMemory core scope only references approved/known modules.
    Presence on disk is not activation.
    """
    declared = []
    approved = []
    unapproved = []

    for item in (target_files or []):
        try:
            module_name = os.path.splitext(os.path.basename(str(item)))[0]
        except Exception:
            module_name = ""
        if module_name:
            declared.append(module_name)

    for item in (subsystems or []):
        try:
            raw = str(item or "").strip()
        except Exception:
            raw = ""
        if not raw:
            continue
        if raw.endswith(".py"):
            raw = os.path.splitext(os.path.basename(raw))[0]
        declared.append(raw)

    seen = []
    for module_name in declared:
        if module_name in seen:
            continue
        seen.append(module_name)
        if module_name.startswith("SarahMemory") or module_name in ("appsys", "appnet", "appnet2", "appmedia", "appstore"):
            if _is_core_module_approved(module_name):
                approved.append(module_name)
            else:
                unapproved.append(module_name)

    return {
        "declared": seen,
        "approved": approved,
        "unapproved": unapproved,
    }


# -----------------------------------------------------------------------------
# Intent classification (simple, deterministic, offline)
# -----------------------------------------------------------------------------
# NOTE: ordering matters. We intentionally place DIAGNOSTICS before EXECUTE_COMMAND
# and NETWORK_ACCESS before FILESYSTEM_WRITE so phrases like "run diagnostics" and
# "download from the internet" classify correctly.
_INTENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("EMERGENCY_INSTINCT", r"\b(fire|smoke|flame|grease\s+fire|electrical\s+fire|asthma|inhaler|choking|can\'t\s+breathe|cannot\s+breathe|unconscious|collision|about\s+to\s+hit|hit\s+by\s+(?:a\s+)?car|vehicle\s+impact|emergency)\b"),
    ("PATCH_OR_UPDATE", r"\b(update|upgrade|patch|monkey\s*patch|self[-\s]*repair|fix\s+code)\b"),
    ("DIAGNOSTICS", r"\b(diagnose|diagnostics|health\s*check|self\s*check|log\s*scan)\b"),
    ("SYSTEM_INFO", r"\b(gpu|vram|cuda|disk\s*space|free\s*space|drive\s*space|storage|cpu\s*usage|ram\s*usage|memory\s*usage|hardware\s*stats|system\s*stats)\b"),
    ("CREATIVE_REQUEST", r"\b(create|generate|make|draw|design|render|compose|build)\b.*\b(image|picture|art|song|music|video|website|webpage|page|avatar|logo|graphic|animation|lyrics|beat)\b"),
    ("NETWORK_ACCESS", r"\b(network|internet|online|web|http|https|api\s+call|connect|wifi|bluetooth|lan|sarahnet)\b"),
    ("FILESYSTEM_WRITE", r"\b(write|create|delete|remove|move|rename|overwrite|trash|dumpster|upload|download)\b"),
    ("PRIVACY_SENSITIVE", r"\b(password|token|secret|key|credential|wallet|private\s+key)\b"),
    ("EXECUTE_COMMAND", r"\b(run|execute|launch|start|shutdown|kill|restart|reboot)\b"),
    ("CHAT", r".*"),
)


def classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "EMPTY"
    for label, pat in _INTENT_PATTERNS:
        try:
            if re.search(pat, t, flags=re.IGNORECASE):
                return label
        except Exception:
            continue
    return "CHAT"


# -----------------------------------------------------------------------------
# Cognitive Interrogation Helpers (no execution; deterministic)
# -----------------------------------------------------------------------------
def _bool(v: Any) -> bool:
    return bool(v)

def _normalize_proposed_action(pa: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(pa, dict):
        return {}

    out = dict(pa)  # shallow copy (do not mutate caller objects)

    def _as_list(v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        # single scalar -> single-item list
        return [v]

    def _as_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "on", "enabled"):
            return True
        if s in ("0", "false", "no", "off", "disabled"):
            return False
        return None

    # Normalize common list fields
    out["target_files"] = _as_list(out.get("target_files"))
    out["subsystems"] = _as_list(out.get("subsystems"))
    out["tests"] = _as_list(out.get("tests"))

    # Normalize filesystem inputs
    if "paths" in out or "path" in out:
        v = out.get("paths", None)
        if v is None:
            v = out.get("path", None)
        out["paths"] = _as_list(v)

    # Normalize booleans (keep None if unknown)
    for k in ("dry_run", "touches_network", "touches_privacy", "touches_filesystem", "sends_data"):
        if k in out:
            out[k] = _as_bool(out.get(k))

    return out

# -----------------------------------------------------------------------------
# Risk scoring helpers
# -----------------------------------------------------------------------------

def _risk_add(risk: Dict[str, Any], points: int, factor: str) -> None:
    risk["risk_score"] = max(0, min(100, int(risk.get("risk_score", 0)) + int(points)))
    rf = risk.get("risk_factors", [])
    if factor and factor not in rf:
        rf.append(factor)
    risk["risk_factors"] = rf


def _answer_missing(ans: Dict[str, Any], key: str, why: str) -> None:
    missing = ans.get("missing", {})
    missing[key] = why
    ans["missing"] = missing


def _safe_str(v: Any, limit: int = 400) -> str:
    s = "" if v is None else str(v)
    s = s.strip()
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


# -----------------------------------------------------------------------------
# Optional CognitiveThinker peer consultation (high-impact only)
# -----------------------------------------------------------------------------
def _cognitive_thinker_enabled(caller_context: Optional[Dict[str, Any]] = None) -> bool:
    ctx = caller_context or {}
    if bool(ctx.get("skip_cognitive_thinker_consult")):
        return False
    if "force_cognitive_thinker_consult" in ctx:
        return bool(ctx.get("force_cognitive_thinker_consult"))
    if hasattr(config, "COGNITIVE_THINKER_CONSULT_ENABLED"):
        try:
            return bool(getattr(config, "COGNITIVE_THINKER_CONSULT_ENABLED"))
        except Exception:
            pass
    env_v = os.getenv("SARAHMEMORY_COGNITIVE_THINKER_CONSULT_ENABLED", "true").strip().lower()
    return env_v not in ("0", "false", "off", "no")


def _is_high_impact_governance_request(intent: str, risk_score: int, proposed_action: Optional[Dict[str, Any]] = None) -> bool:
    pa = proposed_action or {}
    high_impact_intents = {
        "PATCH_OR_UPDATE",
        "FILESYSTEM_WRITE",
        "NETWORK_ACCESS",
        "PRIVACY_SENSITIVE",
        "EXECUTE_COMMAND",
    }
    if str(intent or "") in high_impact_intents:
        return True
    if int(risk_score or 0) >= 35:
        return True
    if bool(pa.get("touches_network")) or bool(pa.get("touches_privacy")) or bool(pa.get("touches_filesystem")):
        return True
    if pa.get("target_files") or pa.get("subsystems"):
        return True
    return False


def _can_consult_cognitive_thinker(caller_context: Optional[Dict[str, Any]] = None) -> bool:
    if not _cognitive_thinker_enabled(caller_context):
        return False
    if bool(getattr(_THINKER_CONSULT_STATE, "active", False)):
        return False
    return True


def _consult_cognitive_thinker(
    request_text: str,
    *,
    caller: str,
    caller_context: Optional[Dict[str, Any]],
    user_present: Optional[bool],
    user_consented: bool,
    proposed_action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ctx = dict(caller_context or {})
    ctx["skip_cognitive_thinker_consult"] = True

    try:
        _THINKER_CONSULT_STATE.active = True
        thinker = importlib.import_module("SarahMemoryCognitiveThinker")
        fn = getattr(thinker, "paired_governance_view", None)
        if callable(fn):
            return fn(
                request_text,
                caller=caller,
                caller_context=ctx,
                user_present=True if user_present is None else bool(user_present),
                user_consented=bool(user_consented),
                proposed_action=proposed_action,
            ) or {}
    except Exception as e:
        logger.debug("CognitiveThinker consult failed: %s", e)
    finally:
        try:
            _THINKER_CONSULT_STATE.active = False
        except Exception:
            pass

    return {}


def _apply_cognitive_thinker_balance(dec: Dict[str, Any], thinker_view: Dict[str, Any]) -> Dict[str, Any]:
    if not thinker_view:
        return dec

    thinker_decision = str(thinker_view.get("thinker_decision") or "")
    final_balance = str(thinker_view.get("final_balance_decision") or "")
    thinker_payload = thinker_view.get("thinker") if isinstance(thinker_view.get("thinker"), dict) else {}

    dec["coequal_governance"] = {
        "enabled": True,
        "peer": "SarahMemoryCognitiveThinker",
        "thinker_decision": thinker_decision,
        "final_balance_decision": final_balance,
        "common_interest": thinker_view.get("common_interest") or {},
        "ticket_id": thinker_payload.get("ticket_id"),
        "priority": thinker_payload.get("priority"),
        "state": thinker_payload.get("state"),
        "recommendations": thinker_payload.get("recommendations") or [],
    }

    if not isinstance(dec.get("trace"), dict):
        dec["trace"] = {}
    dec["trace"]["coequal_governance"] = {
        "peer": "SarahMemoryCognitiveThinker",
        "thinker_decision": thinker_decision,
        "final_balance_decision": final_balance,
    }

    dec.setdefault("reasons", [])
    dec.setdefault("risk_factors", [])

    if thinker_decision == "ETHICALLY_BLOCKED" or final_balance == "DENY":
        dec["decision"] = "DENY"
        dec["allow"] = False
        dec["execution_allowed"] = False
        dec["require_user"] = False
        if "ethical_block_from_cognitive_thinker" not in dec["risk_factors"]:
            dec["risk_factors"].append("ethical_block_from_cognitive_thinker")
        dec["reasons"].append("CognitiveThinker denied the request on ethical / compassionate grounds.")
    elif final_balance == "REQUIRE_USER":
        dec["decision"] = "REQUIRE_USER"
        dec["allow"] = False
        dec["execution_allowed"] = False
        dec["require_user"] = True
        if "cognitive_thinker_requires_user_review" not in dec["risk_factors"]:
            dec["risk_factors"].append("cognitive_thinker_requires_user_review")
        dec["reasons"].append("CognitiveThinker requires explicit user review before this high-impact action may proceed.")
    elif final_balance == "SANDBOX_ONLY":
        dec["decision"] = "DEFER"
        dec["allow"] = False
        dec["execution_allowed"] = False
        dec["require_user"] = True
        dec["recommended_next"] = "Route to sandbox / Synapes / Evolution validation path before any live action."
        if "sandbox_only_by_cognitive_thinker" not in dec["risk_factors"]:
            dec["risk_factors"].append("sandbox_only_by_cognitive_thinker")
        dec["reasons"].append("CognitiveThinker marked the request as worthy of exploration only in sandbox.")
    elif final_balance == "DEFER" and str(dec.get("decision") or "") == "ALLOW":
        dec["decision"] = "DEFER"
        dec["allow"] = False
        dec["execution_allowed"] = False
        dec["require_user"] = True
        if "deferred_by_cognitive_thinker" not in dec["risk_factors"]:
            dec["risk_factors"].append("deferred_by_cognitive_thinker")
        dec["reasons"].append("CognitiveThinker judged the request meaningful but not ready for live approval.")
    else:
        dec["reasons"].append("CognitiveThinker peer review completed without overriding the logical governor.")

    return dec

# -----------------------------------------------------------------------------
# Governance engine (THE HEART)
# -----------------------------------------------------------------------------
def govern_request(
    request_text: str,
    *,
    caller: str = "unknown",
    caller_context: Optional[Dict[str, Any]] = None,
    user_present: Optional[bool] = None,
    user_consented: bool = False,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a request and return a structured governance decision.
    """


    snap = get_cognitive_policy_snapshot()
    intent = classify_intent(request_text)
    pa = _normalize_proposed_action(proposed_action)
    ctx = caller_context or {}
    cognitive_self_packet = _get_cognitive_self_packet(request_text, ctx, force_refresh=False)
    cognitive_self_summary = _self_summary_from_packet(cognitive_self_packet)

    # ---------------------------------------------------------------------
    # Build a minimal routing policy (Governor output) — monotonic restrictions.
    # This policy is advisory/authoritative for Neuron and other executors.
    # ---------------------------------------------------------------------
    try:
        mode_flags = ctx.get("mode_flags") if isinstance(ctx, dict) else {}
        mode_flags = mode_flags if isinstance(mode_flags, dict) else {}
    except Exception:
        mode_flags = {}

    local_only = bool(mode_flags.get("LOCAL_ONLY_MODE") or ctx.get("local_only") or ctx.get("offline") or cognitive_self_summary.get("local_only"))
    safe_mode = bool(mode_flags.get("SAFE_MODE") or ctx.get("safe_mode") or cognitive_self_summary.get("safe_mode"))
    neoskymatrix = bool(mode_flags.get("NEOSKYMATRIX") or ctx.get("neoskymatrix"))
    developersmode = bool(mode_flags.get("DEVELOPERSMODE") or ctx.get("developersmode"))

    device_mode = str(mode_flags.get("DEVICE_MODE") or ctx.get("device_mode") or "").strip()
    run_mode = str(mode_flags.get("RUN_MODE") or ctx.get("run_mode") or "").strip()
    public_web = device_mode.lower().startswith("public") or run_mode.lower() == "cloud"

    routing_policy = {
        "allowed_tiers": {
            "tier0": True,
            "tier1": True,
            "tier2": True,
            # Tier-3 is only allowed when not local-only AND online is enabled in policy snapshot
            "tier3": (not local_only) and bool(snap.get("cognitive_online_enabled")),
        },
        "budgets": {
            # Keep governor budget guidance conservative; Neuron may further tighten
            "latency_ms": 3500 if local_only else 7000,
            "max_steps": 14 if neoskymatrix else 10,
            "max_retries": 1,
        },
        "side_effects": {
            # In SAFE_MODE, disable side effects by default (can be re-enabled by user)
            "tts": True,
            "db_write": False if safe_mode else True,
            "compare": False if safe_mode else True,
            "logging": True,
            "evolution_tick": bool(neoskymatrix) and (not safe_mode),
        },
        "flags": {
            "LOCAL_ONLY_MODE": local_only,
            "SAFE_MODE": safe_mode,
            "NEOSKYMATRIX": neoskymatrix,
            "DEVELOPERSMODE": developersmode,
        },
    }

    if public_web:
        routing_policy["side_effects"]["filesystem"] = False
        routing_policy["side_effects"]["exec"] = False
        routing_policy["side_effects"]["db_write"] = False
        routing_policy["side_effects"]["compare"] = False
        routing_policy["side_effects"]["evolution_tick"] = False

    risk = {"risk_score": 0, "risk_factors": []}
    questions = []
    answers: Dict[str, Any] = {}

    decision: Dict[str, Any] = {
        "ok": True,
        "ts": snap["ts"],
        "intent": intent,
        "caller": caller,
        "lane_family": _lane_family_for_intent(intent),
        "primary_lane": _lane_family_for_intent(intent),
        "require_user": True,
        "decision": "DEFER",
        "allow": False,
        "execution_allowed": False,
        "risk": "unknown",
        "risk_score": 0,
        "risk_factors": [],
        "questions": questions,
        "answers": answers,
        "reasons": [],
        "recommended_next": None,
        "module_hints": _suggest_module_hints(intent),
        "rationale": "",
        "trace": {
            "governor_only": True,
            "execution_owner": None,
        },
        "routing_policy": {},
        "policy_snapshot": {
            "cognitive_online_enabled": snap["cognitive_online_enabled"],
            "kill_switch_neoskymatrix": snap["kill_switch_neoskymatrix"],
            "context_engine_enabled": snap["context_engine_enabled"],
            "core_governance": snap.get("core_governance", {}),
            "cognitive_thinker_consult_enabled": _cognitive_thinker_enabled(ctx),
        },
        "cognitive_self": {
            "summary": cognitive_self_summary,
            "governor_packet": cognitive_self_packet,
            "robotic_body_awareness": cognitive_self_packet.get("robotic_body_awareness") if isinstance(cognitive_self_packet, dict) else {},
        },
        "tri_force": {
            "authority": "SarahMemoryCognitiveSelf",
            "governor": "SarahMemoryCognitiveServices",
            "thinker_peer": "SarahMemoryCognitiveThinker",
        },
    }


    # Attach policy to decision (immutable baseline for this request)
    decision["routing_policy"] = routing_policy
    def _finalize(dec: Dict[str, Any]) -> Dict[str, Any]:
        # Optional co-equal thinker consultation for high-impact decisions.
        try:
            if _can_consult_cognitive_thinker(ctx) and _is_high_impact_governance_request(
                str(dec.get("intent") or intent),
                int(dec.get("risk_score") or 0),
                pa,
            ):
                thinker_view = _consult_cognitive_thinker(
                    request_text,
                    caller=caller,
                    caller_context=ctx,
                    user_present=user_present,
                    user_consented=user_consented,
                    proposed_action=pa,
                )
                if thinker_view:
                    dec = _apply_cognitive_thinker_balance(dec, thinker_view)
        except Exception as e:
            logger.debug("CognitiveThinker balance application failed: %s", e)

        # Risk banding
        try:
            score = int(dec.get("risk_score") or 0)
        except Exception:
            score = 0

        if score <= 15:
            dec["risk"] = "low"
        elif score <= 45:
            dec["risk"] = "medium"
        else:
            dec["risk"] = "high"

        _caller = _safe_str(caller) if caller else "unknown"

        
        # Trace (best-effort; keep small unless DeveloperMode)
        try:
            if not isinstance(dec.get("trace"), dict):
                dec["trace"] = {}
            dec["trace"].setdefault("policy_flags", routing_policy.get("flags"))
            dec["trace"].setdefault("allowed_tiers", routing_policy.get("allowed_tiers"))
            dec["trace"].setdefault("lane_family", dec.get("lane_family"))
            dec["trace"].setdefault("primary_lane", dec.get("primary_lane"))
            dec["trace"].setdefault("module_hints", dec.get("module_hints") or [])
            dec["trace"].setdefault("execution_allowed", False)
            dec["trace"].setdefault("robotic_body_governance", answers.get("robotic_body_governance", {}))
            if cognitive_self_summary:
                dec["trace"].setdefault("cognitive_self_summary", cognitive_self_summary)
                dec["trace"].setdefault("tri_force", {"authority": "SarahMemoryCognitiveSelf", "governor": "SarahMemoryCognitiveServices", "thinker_peer_enabled": _cognitive_thinker_enabled(ctx)})
        except Exception:
            pass
# Event log (best-effort)
        try:
            log_cognitive_event(
                "CognitiveDecision",
                f"{dec.get('decision')} intent={dec.get('intent')} caller={_caller}",
                severity="INFO",
                meta={
                    "intent": dec.get("intent"),
                    "caller": _caller,
                    "allow": dec.get("allow"),
                    "require_user": dec.get("require_user"),
                    "risk": dec.get("risk"),
                    "risk_score": dec.get("risk_score"),
                    "risk_factors": dec.get("risk_factors"),
                    "reasons": dec.get("reasons"),
                    "recommended_next": dec.get("recommended_next"),
                    "has_proposed_action": bool(pa),
                    "missing": (dec.get("answers") or {}).get("missing", {}),
                },
            )
        except Exception:
            pass

        # Optimization partition publish (best-effort)
        try:
            role = _route_role_for_decision(dec, ctx)
            _publish_to_optimization_partition(
                role,
                {
                    "ts": datetime.now().isoformat(),
                    "role": role,
                    "decision": dec.get("decision"),
                    "intent": dec.get("intent"),
                    "risk": dec.get("risk"),
                    "risk_score": dec.get("risk_score"),
                    "reasons": dec.get("reasons"),
                    "recommended_next": dec.get("recommended_next"),
                    "caller": _caller,
                },
            )
        except Exception:
            pass

        try:
            dec = _sm_attach_six_question_packet_to_decision(
                dec,
                request_text=request_text,
                caller=caller,
                caller_context=ctx,
                proposed_action=pa,
                user_present=user_present,
                user_consented=user_consented,
            )
        except Exception as _six_e:
            try:
                dec.setdefault('trace', {})['six_question_attach_error'] = str(_six_e)
            except Exception:
                pass
        return dec

    # -------------------------------------------------------------------------
    # Baseline self-questions (always asked)
    # -------------------------------------------------------------------------
    questions.append("What is the intent category of this request?")
    answers["intent"] = intent

    questions.append("Who is asking (caller), and does caller have authority for execution?")
    answers["caller"] = _safe_str(caller)
    answers["caller_execution_authority"] = False

    questions.append("Is the user present, and do we have explicit consent for high-impact actions?")
    answers["user_present"] = user_present
    answers["user_consented"] = bool(user_consented)
    if user_present is False:
        _risk_add(risk, 20, "user_not_present")

    questions.append("Is online cognition/network access enabled by policy?")
    answers["cognitive_online_enabled"] = bool(snap["cognitive_online_enabled"])
    if intent == "NETWORK_ACCESS" and not snap["cognitive_online_enabled"]:
        _risk_add(risk, 10, "network_blocked_by_policy")

    questions.append("Is autonomous evolution permitted right now (NEOSKYMATRIX)?")
    answers["kill_switch_neoskymatrix"] = bool(snap["kill_switch_neoskymatrix"])
    if intent in ("PATCH_OR_UPDATE", "EXECUTE_COMMAND", "FILESYSTEM_WRITE") and not snap["kill_switch_neoskymatrix"]:
        _risk_add(risk, 25, "autonomy_disabled_neoskymatrix_off")

    questions.append("Did the caller provide a structured proposed_action plan?")
    answers["has_proposed_action"] = bool(pa)
    questions.append("What does CognitiveSelf say about current runtime, continuity, and realtime resource posture?")
    answers["cognitive_self_summary"] = cognitive_self_summary
    answers["cognitive_self_temporal_awareness"] = cognitive_self_packet.get("temporal_awareness") if isinstance(cognitive_self_packet, dict) else {}
    answers["cognitive_self_realtime_strategy"] = cognitive_self_packet.get("realtime_strategy") if isinstance(cognitive_self_packet, dict) else {}
    robotic_profile = _robotic_action_governance_profile(pa, request_text)
    answers["robotic_body_governance"] = robotic_profile
    if robotic_profile.get("is_robotic_body_action"):
        _risk_add(risk, 25, "robotic_body_action")
        decision.setdefault("reasons", []).append("Robotic body action detected; SMGET, OperatorCore, SecurityGovernor, AssuranceGate, MSDC, Compare, and Compass must remain in the chain.")
        if robotic_profile.get("motion_requested"):
            _risk_add(risk, 20, "robotic_motion_requested")
        if robotic_profile.get("human_contact_requested") and not robotic_profile.get("emergency_context"):
            _risk_add(risk, 40, "robotic_human_contact_without_verified_emergency")
        if not user_consented and not robotic_profile.get("emergency_context"):
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["recommended_next"] = "Collect explicit user authorization or verified emergency evidence before any robotic body action may proceed."

    if pa:
        answers["proposed_action_summary"] = {
            "reason": _safe_str(pa.get("reason")),
            "change_type": _safe_str(pa.get("change_type")),
            "target_files": pa.get("target_files") or [],
            "subsystems": pa.get("subsystems") or [],
            "rollback_plan": _safe_str(pa.get("rollback_plan")),
            "tests": pa.get("tests") or [],
            "dry_run": pa.get("dry_run"),
            "touches_network": pa.get("touches_network"),
            "touches_privacy": pa.get("touches_privacy"),
            "touches_filesystem": pa.get("touches_filesystem"),
            "robotic_body_governance": robotic_profile,
        }
    else:
        if intent in ("PATCH_OR_UPDATE", "EXECUTE_COMMAND", "FILESYSTEM_WRITE", "NETWORK_ACCESS"):
            _risk_add(risk, 15, "no_structured_plan")

    # -------------------------------------------------------------------------
    # Intent-specific interrogation
    # -------------------------------------------------------------------------

    if intent == "PATCH_OR_UPDATE":
        questions.extend(
            [
                "Why is this update being proposed? Is there a concrete bug, failure, or measurable benefit?",
                "Is the target code currently functional? Do diagnostics/logs show an actual failure?",
                "Is there already an existing implementation elsewhere (duplicate feature risk)?",
                "What is the blast radius (which files/subsystems are touched)?",
                "What tests validate success, and what tests prevent regression?",
                "Is there a rollback plan that restores last-known-good state?",
                "Does this change increase autonomy, network exposure, or privacy risk?",
                "Does it violate user ownership/autonomy principles or ethics rules?",
            ]
        )

        reason = _safe_str(pa.get("reason"))
        change_type = _safe_str(pa.get("change_type"))
        targets = pa.get("target_files") or []
        subsystems = pa.get("subsystems") or []
        tests = pa.get("tests") or []
        rollback = _safe_str(pa.get("rollback_plan"))
        dry_run = pa.get("dry_run")
        touches_network = pa.get("touches_network")
        touches_privacy = pa.get("touches_privacy")

        answers["update_reason"] = reason or None
        answers["update_change_type"] = change_type or None
        answers["update_targets"] = targets
        answers["update_subsystems"] = subsystems
        answers["update_tests_declared"] = tests
        answers["update_rollback_plan"] = rollback or None
        answers["update_dry_run_declared"] = dry_run

        scope_validation = _validate_scope_modules(targets, subsystems)
        answers["governed_scope"] = scope_validation
        if scope_validation.get("unapproved"):
            _answer_missing(
                answers,
                "governed_scope",
                "One or more declared SarahMemory modules are not approved/registered for governed routing.",
            )
            _risk_add(risk, 20, "unapproved_core_scope")

        # Required metadata checks (risk scoring + missing map)
        if not reason:
            _answer_missing(answers, "reason", "Provide a concrete reason/bug/benefit for the change.")
            _risk_add(risk, 10, "missing_reason")
        if not targets and not subsystems:
            _answer_missing(answers, "scope", "Provide target_files and/or subsystems to assess blast radius.")
            _risk_add(risk, 10, "missing_scope")
        if not tests:
            _answer_missing(answers, "tests", "Provide at least one validation test or diagnostic proof.")
            _risk_add(risk, 15, "missing_tests")
        if not rollback:
            _answer_missing(answers, "rollback_plan", "Provide rollback/restore plan to last-known-good.")
            _risk_add(risk, 20, "missing_rollback")

        if dry_run is not True:
            _risk_add(risk, 5, "no_dry_run_declared")
        if _bool(touches_network):
            _risk_add(risk, 10, "touches_network")
        if _bool(touches_privacy):
            _risk_add(risk, 15, "touches_privacy")

        # Autonomy gate (owner-aligned kill-switch)
        if not snap["kill_switch_neoskymatrix"] and not user_consented:
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append(
                "NEOSKYMATRIX is OFF; autonomous self-evolution is not permitted without explicit user consent."
            )
            decision["recommended_next"] = (
                "Request approval; then route proposal to SarahMemoryEvolution/SarahMemoryCompare for validation."
            )
            decision["risk_score"] = risk["risk_score"] + 20
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        # Missing metadata -> DEFER (must happen BEFORE ALLOW)
        if (answers.get("missing") or {}) != {}:
            decision["decision"] = "DEFER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("Update proposal lacks required proof/metadata; governor will not guess.")
            decision["recommended_next"] = (
                "Provide missing fields (reason/scope/tests/rollback), then re-evaluate and route to Evolution/Compare."
            )
            decision["risk_score"] = risk["risk_score"] + 10
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        # Otherwise -> ALLOW (route onward; still no execution here)
        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = not bool(user_consented)
        decision["reasons"].append("Proposal has sufficient metadata for safe routing to validation modules.")
        decision["recommended_next"] = (
            "Route proposal to SarahMemoryCompare (diff/regression) and SarahMemoryEvolution (proposal generation only)."
        )
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]

        # DEVELOPERSMODE sandbox packaging (ONLY after ALLOW)
        try:
            maybe_attach_virtual_sandbox(
                decision=decision,
                request_text=request_text,
                intent=intent,
                caller=caller,
                proposed_action=pa,
            )
        except Exception:
            pass

        return _finalize(decision)


    if intent == "FILESYSTEM_WRITE":
        if public_web:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = False
            decision["reasons"].append("Filesystem write operations are disabled in Public Web mode.")
            decision["recommended_next"] = "Run filesystem changes on a local agent (LOCAL_AGENT)."
            decision["risk_score"] = 8
            decision["risk_factors"].append("public_web_restriction")
            return _finalize(decision)

        questions.extend(
            [
                "What exact filesystem operation is being requested (create/move/delete/trash/upload)?",
                "What paths are involved, and are they within BASE_DIR rules (no traversal)?",
                "Is this destructive? If yes, is trash/dumpster the required mode?",
                "Is there a reversible plan (restore from dumpster) and logging enabled?",
                "Does user presence/consent meet policy requirements?",
            ]
        )

        fs_paths = pa.get("paths") or []
        fs_mode = _safe_str(pa.get("mode")) or None
        answers["fs_paths"] = fs_paths
        answers["fs_mode"] = fs_mode

        if not fs_paths:
            _answer_missing(answers, "paths", "Provide explicit path(s) for validation against BASE_DIR rules.")
            _risk_add(risk, 10, "missing_paths")

        if isinstance(fs_mode, str) and fs_mode.lower() in ("delete", "remove", "purge"):
            _risk_add(risk, 20, "destructive_delete_requested")

        if user_present is False and not user_consented:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("User not present; filesystem write is denied without explicit consent.")
            decision["recommended_next"] = "Queue suggestion; ask user next time they are present."
            decision["risk_score"] = risk["risk_score"] + 20
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if (answers.get("missing") or {}) != {}:
            decision["decision"] = "DEFER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("Filesystem request lacks required details (paths/mode).")
            decision["recommended_next"] = "Provide exact paths + mode (trash vs delete), then re-evaluate."
            decision["risk_score"] = risk["risk_score"] + 5
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if not snap["kill_switch_neoskymatrix"] and not user_consented:
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("NEOSKYMATRIX is OFF; filesystem changes require explicit user consent.")
            decision["recommended_next"] = "Request approval; then route to filesystem module with trash-first behavior."
            decision["risk_score"] = risk["risk_score"] + 15
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = not bool(user_consented)
        decision["reasons"].append("Filesystem action has sufficient details for routing; execution must confirm and log.")
        decision["recommended_next"] = "Route to Action Lane filesystem owner; enforce BASE_DIR, trash-first behavior, and event logging."
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        
        return _finalize(decision)

    if intent == "NETWORK_ACCESS":
        questions.extend(
            [
                "What is the purpose of this network action (research, sync, SarahNet, API call)?",
                "Will any personal/private data be transmitted?",
                "Is online mode enabled, and do we have explicit consent for this exact call?",
                "Can the same goal be achieved offline or locally first?",
            ]
        )

        purpose = _safe_str(pa.get("purpose")) if pa else None
        sends_data = pa.get("sends_data") if pa else None
        endpoint = _safe_str(pa.get("endpoint")) if pa else None
        answers["network_purpose"] = purpose or None
        answers["network_endpoint"] = endpoint or None
        answers["network_sends_data"] = sends_data
        answers["online_connectivity"] = bool(cognitive_self_summary.get("online_connectivity"))

        if not snap["cognitive_online_enabled"]:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = False
            decision["reasons"].append("COGNITIVE_ONLINE_ENABLED is OFF; network actions are blocked by default.")
            decision["recommended_next"] = "Stay offline; ask user to enable online mode explicitly if desired."
            _risk_add(risk, 10, "blocked_offline_first_policy")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if snap["cognitive_online_enabled"] and not bool(cognitive_self_summary.get("online_connectivity", True)):
            decision["decision"] = "DEFER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("Live network access is currently unavailable according to CognitiveSelf realtime awareness.")
            decision["recommended_next"] = "Retry when connectivity is restored or stay local-first with cached data."
            _risk_add(risk, 8, "live_connectivity_unavailable")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if not user_consented:
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append(
                "Online mode may be enabled, but explicit user consent is required before network access."
            )
            decision["recommended_next"] = "Ask user to approve this specific network call."
            _risk_add(risk, 10, "missing_network_consent")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if sends_data is True:
            _risk_add(risk, 25, "transmits_private_or_user_data_possible")

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = False
        decision["reasons"].append("Network action approved by policy + explicit consent; execution must minimize data.")
        decision["recommended_next"] = "Route to Network Lane through approved network/research owners; redact sensitive info and log intent."
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        return _finalize(decision)

    if intent == "PRIVACY_SENSITIVE":
        questions.extend(
            [
                "Does this request involve secrets/credentials/private keys or user-identifying data?",
                "Is the user explicitly consenting to handle or expose this sensitive content?",
                "Can the task be completed without seeing/storing the sensitive data?",
            ]
        )

        if not user_consented:
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("Privacy-sensitive content requires explicit user consent.")
            decision["recommended_next"] = "Request confirmation; minimize exposure; avoid storage."
            _risk_add(risk, 30, "privacy_sensitive_no_consent")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = False
        decision["reasons"].append(
            "Privacy-sensitive task approved by explicit consent; execution must minimize data exposure."
        )
        decision["recommended_next"] = "Route to the responsible module with redaction and no persistent storage."
        _risk_add(risk, 15, "privacy_sensitive_even_with_consent")
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        return _finalize(decision)

    if intent == "EXECUTE_COMMAND":
        if public_web:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = False
            decision["reasons"].append("Command execution is disabled in Public Web mode.")
            decision["recommended_next"] = "Run this request on a local agent (LOCAL_AGENT)."
            decision["risk_score"] = 9
            decision["risk_factors"].append("public_web_restriction")
            return _finalize(decision)

        questions.extend(
            [
                "What exact command/action is intended (start/stop/restart/run)?",
                "Could it disrupt boot sequence, audio loops, UI, or data integrity?",
                "Is the user present and consenting?",
                "Is there a dry-run or safe-mode alternative?",
            ]
        )

        cmd = _safe_str(pa.get("command")) if pa else None
        answers["command"] = cmd or None

        if cmd is None:
            _answer_missing(answers, "command", "Provide exact command/action for evaluation.")
            _risk_add(risk, 10, "missing_command")

        if user_present is False and not user_consented:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("User not present; command execution denied without explicit consent.")
            decision["recommended_next"] = "Queue suggestion; request approval when user is present."
            _risk_add(risk, 20, "user_not_present_for_exec")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if not snap["kill_switch_neoskymatrix"] and not user_consented:
            decision["decision"] = "REQUIRE_USER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("NEOSKYMATRIX is OFF; execution requires explicit user consent.")
            decision["recommended_next"] = "Request approval; then route to the responsible execution module."
            _risk_add(risk, 20, "autonomy_disabled_exec")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        if (answers.get("missing") or {}) != {}:
            decision["decision"] = "DEFER"
            decision["allow"] = False
            decision["require_user"] = True
            decision["reasons"].append("Execution request lacks required details.")
            decision["recommended_next"] = "Provide exact command/action; then re-evaluate."
            _risk_add(risk, 5, "insufficient_exec_details")
            decision["risk_score"] = risk["risk_score"]
            decision["risk_factors"] = risk["risk_factors"]
            return _finalize(decision)

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = not bool(user_consented)
        decision["reasons"].append(
            "Execution intent acknowledged; execution must include confirmations and safe-mode if available."
        )
        decision["recommended_next"] = "Route to the responsible module; require confirmations and logging."
        _risk_add(risk, 10, "execution_is_high_impact")
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        return _finalize(decision)

    if intent == "SYSTEM_INFO":
        if public_web:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = False
            decision["reasons"].append("System hardware/runtime information is disabled in Public Web mode.")
            decision["recommended_next"] = "Run this request on a local agent (LOCAL_AGENT)."
            decision["risk_score"] = 5
            decision["risk_factors"].append("public_web_restriction")
            return _finalize(decision)

        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = False
        decision["reasons"].append("Read-only system information.")
        decision["risk_score"] = 1
        return _finalize(decision)

    if intent == "DIAGNOSTICS":
        if public_web:
            decision["decision"] = "DENY"
            decision["allow"] = False
            decision["require_user"] = False
            decision["reasons"].append("Diagnostics are disabled in Public Web mode.")
            decision["recommended_next"] = "Run diagnostics locally (LOCAL_AGENT)."
            decision["risk_score"] = 7
            decision["risk_factors"].append("public_web_restriction")
            return _finalize(decision)

        questions.extend(
            [
                "Is this action read-only and non-destructive?",
                "Does it respect offline-first and user autonomy?",
            ]
        )
        decision["decision"] = "ALLOW"
        decision["allow"] = True
        decision["require_user"] = False
        decision["reasons"].append("Diagnostics are safe and read-only by default.")
        decision["recommended_next"] = "Route to System Lane (Diagnostics) as read-only; execution remains outside the governor."
        decision["risk_score"] = risk["risk_score"]
        decision["risk_factors"] = risk["risk_factors"]
        return _finalize(decision)

    # CHAT / default
    questions.extend(
        [
            "Is this a low-risk conversational request with no side effects?",
            "Does it require any restricted capabilities (network, file write, execution)?",
        ]
    )
    decision["decision"] = "ALLOW"
    decision["allow"] = True
    decision["require_user"] = False
    decision["reasons"].append("General chat is low risk.")
    decision["recommended_next"] = "Route to Answer Lane with deterministic/local-first execution; helper models remain optional."
    decision["risk_score"] = risk["risk_score"]
    decision["risk_factors"] = risk["risk_factors"]
    return _finalize(decision)


# -----------------------------------------------------------------------------
# Local cognitive fallback (kept, but treated as low-trust suggestions)
# -----------------------------------------------------------------------------
def load_local_cognitive_data() -> Dict[str, Any]:
    """
    Loads local cognitive data for fallback suggestions.
    Returns {} on failure.
    """
    try:
        with open(LOCAL_COGNITIVE_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def process_local_cognitive_request(request_text: str) -> Optional[Any]:
    """
    Simple keyword matching against local cognitive JSON.
    This is NOT authoritative cognition; it's a suggestion source.
    """
    data = load_local_cognitive_data()
    t = (request_text or "").lower()
    for key, response in data.items():
        try:
            if str(key).lower() in t:
                return response
        except Exception:
            continue
    return None


def process_online_cognitive_request(request_text: str) -> Optional[Any]:
    """
    Online cognitive processing is governed and OFF by default.
    This remains a placeholder unless the owner wires a provider.
    """
    dec = govern_request(
        request_text,
        caller="SarahMemoryCognitiveServices.process_online_cognitive_request",
        user_consented=False,
    )
    if dec.get("decision") != "ALLOW":
        return None
    return None  # Provider not implemented here by design (governor only).


def process_cognitive_request_text(request_text: str) -> Any:
    """
    Adaptive processing of cognitive requests (legacy compatibility).
    Local suggestions first; online only if enabled + consent provided externally.
    """
    _ = govern_request(request_text, caller="SarahMemoryCognitiveServices.process_cognitive_request_text")

    local_result = process_local_cognitive_request(request_text)
    if local_result is not None:
        log_cognitive_event("LocalCognitiveSuggestion", "Matched local cognitive data", meta={"text": request_text})
        return local_result

    if bool(getattr(config, "COGNITIVE_ONLINE_ENABLED", False)):
        return "Online cognition is enabled, but no provider is wired in this governor module."

    return "I'm sorry, I couldn't process that request at this time."


# -----------------------------------------------------------------------------
# Legacy analyzers (kept for compatibility, but governed)
# -----------------------------------------------------------------------------
def _looks_like_placeholder_key(k: str) -> bool:
    ks = (k or "").strip()
    if not ks:
        return True
    if "YOUR_" in ks.upper():
        return True
    if len(ks) < 10:
        return True
    return False


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Legacy sentiment analyzer (Microsoft Cognitive Services) - governed.
    OFF by default and requires COGNITIVE_ONLINE_ENABLED + explicit consent from caller.
    """
    if not bool(getattr(config, "COGNITIVE_ONLINE_ENABLED", False)):
        return {"error": "Online cognitive services are disabled (COGNITIVE_ONLINE_ENABLED is OFF)."}
    if _looks_like_placeholder_key(TEXT_ANALYSIS_KEY):
        return {"error": "Online cognitive key not configured."}
    return {"error": "analyze_text is governed; wire provider calls in a dedicated integration module if needed."}


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Legacy image analyzer (Microsoft Cognitive Services) - governed.
    OFF by default and requires COGNITIVE_ONLINE_ENABLED + explicit consent from caller.
    """
    if not bool(getattr(config, "COGNITIVE_ONLINE_ENABLED", False)):
        return {"error": "Online cognitive services are disabled (COGNITIVE_ONLINE_ENABLED is OFF)."}
    if not os.path.exists(image_path):
        return {"error": "Image file not found."}
    if _looks_like_placeholder_key(IMAGE_ANALYSIS_KEY):
        return {"error": "Online cognitive key not configured."}
    return {"error": "analyze_image is governed; wire provider calls in a dedicated integration module if needed."}


# -----------------------------------------------------------------------------
# Optional: response table helper (kept, NO auto-call)
# -----------------------------------------------------------------------------
def ensure_response_table(db_path: Optional[str] = None) -> bool:
    """
    Ensures the legacy `response` table exists (used by some UI/chat logging).
    This is intentionally NOT called at import time.
    """
    try:
        if db_path is None:
            db_path = _system_logs_db()
        con = _connect(db_path)
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS response ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)"
        )
        con.commit()
        con.close()
        return True
    except Exception as e:
        logger.debug("ensure_response_table failed: %s", e)
        return False

def process_cognitive_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy wrapper for the web/UI calling pattern.
    """
    payload = payload or {}

    text = payload.get("text") or payload.get("message") or ""
    caller = payload.get("caller") or "process_cognitive_request"
    ctx = payload.get("caller_context") or {}
    pa = payload.get("proposed_action") or None
    user_present = payload.get("user_present", True)
    user_consented = payload.get("user_consented", False)

    if classify_intent(str(text or "")) == "EMERGENCY_INSTINCT" or bool(payload.get("emergency") or payload.get("hazard_type") or payload.get("emergency_type")):
        instinct_payload = dict(ctx) if isinstance(ctx, dict) else {}
        instinct_payload.update(payload)
        instinct_payload.setdefault("text", text)
        instinct = evaluate_emergency_instinct(instinct_payload, caller=str(caller or "process_cognitive_request"))
        response = {
            "ok": True,
            "governance": {
                "decision": instinct.get("decision"),
                "allow": bool(instinct.get("bounded_action_allowed")),
                "require_user": bool(instinct.get("requires_user")),
                "intent": "EMERGENCY_INSTINCT",
                "execution_allowed": False,
                "emergency_instinct": instinct,
            },
            "lane_family": "SYSTEM",
            "primary_lane": "SYSTEM",
            "emergency_instinct": instinct,
            "version": "9.0.0",
        }
        return response

    dec = govern_request(
        text,
        caller=caller,
        caller_context=ctx,
        user_present=user_present,
        user_consented=user_consented,
        proposed_action=pa,
    )

    response = {"ok": True, "governance": dec, "lane_family": dec.get("lane_family"), "primary_lane": dec.get("primary_lane"), "version": "9.0.0"}
    if isinstance(dec.get("smget"), dict):
        response["smget"] = dec.get("smget")
    return response


# -----------------------------------------------------------------------------
# Module self-test (safe, no external calls)
# -----------------------------------------------------------------------------
def _run_self_test() -> bool:
    print("[SarahMemoryCognitiveServices] Governor self-test (safe/offline)")

    # Optional: try to ensure DB tables; if it fails, we still continue
    try:
        _ensure_tables()
        print("[OK] DB tables ensured:", _system_logs_db())
    except Exception as e:
        print("[WARN] DB table ensure failed (continuing):", e)

    scenarios = [
        ("Run diagnostics", "Diagnostics", {"command": "diagnostics"}),
        ("Update your code", "Update w/ missing plan", None),
        (
            "Update your code",
            "Update w/ plan",
            {
                "reason": "Fix crash in boot sequence when loading UI settings.",
                "change_type": "bugfix",
                "target_files": ["SarahMemoryIntegration.py"],
                "subsystems": ["boot", "ui"],
                "tests": ["Run boot self-test; verify UI loads; confirm no exceptions"],
                "rollback_plan": "Restore previous file from backup + restart in safe mode",
                "dry_run": True,
                "touches_network": False,
                "touches_privacy": False,
            },
        ),
        (
            "Connect to the internet and download something",
            "Network",
            {"purpose": "research", "endpoint": "https://example.com", "sends_data": False},
        ),
        ("Delete this file", "Filesystem delete", {"mode": "delete", "paths": ["../data/important.db"]}),
        ("Hello Sarah", "Chat", None),
    ]

    ok = True
    for txt, label, plan in scenarios:
        try:
            d = govern_request(txt, caller="__main__", user_present=True, user_consented=False, proposed_action=plan)

            print("\n==", label, "==")
            print("Text:", txt)
            print(
                "Decision:",
                d.get("decision"),
                "| intent:",
                d.get("intent"),
                "| risk:",
                d.get("risk"),
                "| score:",
                d.get("risk_score"),
            )
            print("Factors:", d.get("risk_factors"))
            print("Missing:", (d.get("answers") or {}).get("missing", {}))
            print("Questions asked:", len(d.get("questions") or []))

            if d.get("decision") not in ("ALLOW", "DENY", "DEFER", "REQUIRE_USER"):
                ok = False
                print("[FAIL] Invalid decision:", d.get("decision"))
        except Exception as e:
            ok = False
            print("\n==", label, "==")
            print("[ERROR] Scenario crashed:", e)

    return ok


def main() -> int:
    """
    Safe offline self-test runner for SarahMemoryCognitiveServices.py
    """
    ok = _run_self_test()
    if ok:
        print("\n[PASS] Cognitive Governor self-test completed successfully.")
        return 0
    print("\n[FAIL] Cognitive Governor self-test completed with errors.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())



# -----------------------------------------------------------------------------
# SARAH_REM_GOVERNOR_V1
# REM Sleep candidate governance. This is decision-only; no execution here.
# -----------------------------------------------------------------------------
def govern_rem_candidate(candidate: Dict[str, Any], *, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snapshot = snapshot or {}; candidate = candidate or {}; reasons: list[str] = []; risk_factors: list[str] = []
    target_files = [os.path.basename(str(x)) for x in (candidate.get("target_files") or [])]
    proposed = candidate.get("proposed_action") if isinstance(candidate.get("proposed_action"), dict) else {}
    neosky = bool(getattr(config, "NEOSKYMATRIX", False)); risk_tier = str(candidate.get("risk_tier") or proposed.get("risk_tier") or "medium").lower()
    decision = "ALLOW"; allow = True; require_user = False
    if not neosky:
        decision, allow = "DENY", False; reasons.append("NEOSKYMATRIX is disabled; REM self-evolution is locked.")
    protected_governance_files = {"SarahMemoryGlobals.py", "SarahMemoryARILE.py"}
    protected_hit = sorted(set(target_files) & protected_governance_files)
    if protected_hit:
        decision, allow = "DENY", False; risk_factors.append("protected_arile_file_targeted" if "SarahMemoryARILE.py" in protected_hit else "protected_globals_file_targeted"); reasons.append(", ".join(protected_hit) + " is immutable through REM/evolution governance; use approved ARILE overlay lane only.")
    if proposed.get("opens_attachment") or proposed.get("open_attachment"):
        decision, allow = "DENY", False; risk_factors.append("attachment_opening_forbidden"); reasons.append("REM is not allowed to open email attachments.")
    if proposed.get("deletes_user_data") or proposed.get("destructive"):
        decision, allow, require_user = "REQUIRE_USER", False, True; risk_factors.append("destructive_or_user_data_change")
    if proposed.get("installs_dependency") or proposed.get("new_dependency"):
        decision, allow, require_user = "REQUIRE_USER", False, True; risk_factors.append("new_dependency_requires_user")
    if proposed.get("expands_authority") or proposed.get("changes_security") or proposed.get("changes_network_permissions"):
        decision, allow, require_user = "REQUIRE_USER", False, True; risk_factors.append("authority_expansion_requires_user")
    if risk_tier not in ("low", "tier_0_info", "tier_1_harmless_local_ui"):
        require_user = True
        if allow: decision, allow = "DEFER", False
        risk_factors.append(f"risk_tier_{risk_tier}_not_auto_apply")
    if allow: reasons.append("REM candidate is bounded, low-risk, sandbox-only, and eligible for assurance review.")
    elif not reasons: reasons.append("REM candidate requires user review or was denied by policy.")
    out = {"decision":decision,"allow":bool(allow),"require_user":bool(require_user),"risk_tier":risk_tier,"risk_score":10 if allow else (55 if require_user else 90),"risk_factors":risk_factors,"reasons":reasons,"protected_files":["SarahMemoryGlobals.py","SarahMemoryARILE.py"],"recommended_next":"sandbox_then_assurance" if allow else "stage_or_reject","ts":datetime.now().isoformat()}
    try: log_cognitive_event("REM_GOVERNANCE", str(candidate.get("title") or candidate.get("dream_id") or "candidate"), meta=out)
    except Exception: pass
    return out


# -----------------------------------------------------------------------------
# V10/V9F read-only evidence governance helper
# -----------------------------------------------------------------------------
def govern_read_only_evidence_claim(claim: str = "", evidence_packet: Optional[Dict[str, Any]] = None, appeal_packet: Optional[Dict[str, Any]] = None, caller: str = "cognitive_services.read_only_evidence") -> Dict[str, Any]:
    proposed = {'action_type': 'read_sensor_evidence', 'capability_name': 'selfaware_evidence_claim', 'read_only': True, 'touches_filesystem': False, 'touches_network': False, 'physical_actuation': False, 'risk_level': 'TIER_0_INFO', 'rollback_plan': 'No state change; no rollback required.', 'truthfulness_evidence': bool(evidence_packet)}
    return govern_request('Read-only evidence claim: ' + str(claim or ''), caller=caller, caller_context={'read_only': True, 'evidence_packet': evidence_packet or {}, 'appeal_packet': appeal_packet or {}, 'skip_cognitive_thinker_consult': True}, user_present=True, user_consented=False, proposed_action=proposed)

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
_SIX_DIMENSIONS = ("WHO", "WHY", "WHAT", "WHEN", "WHERE", "HOW")


def _sm_dimension(decision: str = "ALLOW", confidence: float = 0.5, evidence: Optional[Dict[str, Any]] = None, reason: str = "") -> Dict[str, Any]:
    return {
        "decision": str(decision or "DEFER").upper(),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "evidence": evidence or {},
        "reason": reason,
    }


def _sm_entry_point_from_request(text: str, caller_context: Optional[Dict[str, Any]] = None) -> str:
    ctx = caller_context or {}
    event_type = str(ctx.get("event_type") or ctx.get("source") or "").lower()
    if "webcam" in event_type or "sensor" in event_type:
        return "WHAT"
    if "dream" in event_type or "rem" in event_type:
        return "WHY"
    m = re.match(r"\s*(who|why|what|when|where|how)\b", str(text or ""), flags=re.I)
    if m:
        return m.group(1).upper()
    return "WHAT"


def merge_six_question_verdicts(dimensions: Dict[str, Dict[str, Any]], *, risk_tier: str = "") -> Dict[str, Any]:
    """Strict AND merge: DENY > REQUIRE_USER > DEFER/UNKNOWN > ALLOW_WITH_CONSTRAINTS > ALLOW."""
    order = ["DENY", "REQUIRE_USER", "DEFER", "UNKNOWN", "ALLOW_WITH_CONSTRAINTS", "ALLOW"]
    found = []
    for key in _SIX_DIMENSIONS:
        d = dimensions.get(key) if isinstance(dimensions, dict) else {}
        found.append(str((d or {}).get("decision") or "UNKNOWN").upper())
    if "DENY" in found:
        final = "DENY"
    elif "REQUIRE_USER" in found:
        final = "REQUIRE_USER"
    elif "DEFER" in found or "UNKNOWN" in found:
        final = "DEFER"
    elif "ALLOW_WITH_CONSTRAINTS" in found:
        final = "ALLOW_WITH_CONSTRAINTS"
    else:
        final = "ALLOW"
    return {
        "strategy": "STRICT_AND",
        "decision": final,
        "allow": final in {"ALLOW", "ALLOW_WITH_CONSTRAINTS"},
        "require_user": final == "REQUIRE_USER",
        "fail_closed": final in {"DENY", "DEFER"},
        "dimension_decisions": dict(zip(_SIX_DIMENSIONS, found)),
        "risk_tier": risk_tier or "TIER_UNKNOWN",
    }


def build_six_question_governance_packet(
    request_text: str,
    *,
    caller: str = "unknown",
    caller_context: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    user_present: Optional[bool] = None,
    user_consented: bool = False,
    base_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = caller_context if isinstance(caller_context, dict) else {}
    pa = proposed_action if isinstance(proposed_action, dict) else {}
    base = base_decision if isinstance(base_decision, dict) else {}
    text = str(request_text or "")
    entry = _sm_entry_point_from_request(text, ctx)
    risk_score = int(base.get("risk_score") or 0)
    risk_tier = "TIER_1_LOW" if risk_score <= 15 else "TIER_2_MEDIUM" if risk_score <= 45 else "TIER_3_HIGH"

    who_decision = "ALLOW" if caller and str(caller).lower() != "unknown" else "REQUIRE_USER"
    if user_present is False and risk_score > 15:
        who_decision = "REQUIRE_USER"

    why_decision = "ALLOW" if text.strip() else "DEFER"
    what_decision = "ALLOW" if (text.strip() or pa) else "DEFER"
    when_decision = "ALLOW" if (user_consented or risk_score <= 15) else ("REQUIRE_USER" if risk_score > 15 else "ALLOW")
    where_decision = "ALLOW"
    how_decision = "ALLOW"

    if str(base.get("decision") or "").upper() == "DENY":
        how_decision = "DENY"
    elif str(base.get("decision") or "").upper() == "REQUIRE_USER":
        when_decision = "REQUIRE_USER"
    elif str(base.get("decision") or "").upper() == "DEFER":
        how_decision = "DEFER"

    dims = {
        "WHO": _sm_dimension(who_decision, 0.80, {"caller": caller, "user_present": user_present, "user_consented": bool(user_consented)}, "Authority / affected party review."),
        "WHY": _sm_dimension(why_decision, 0.72, {"purpose_hint": ctx.get("purpose_hint") or base.get("intent") or "general"}, "Purpose and intent review."),
        "WHAT": _sm_dimension(what_decision, 0.78, {"intent": base.get("intent"), "proposed_action_present": bool(pa), "risk_score": risk_score}, "Scope, object, data, and risk review."),
        "WHEN": _sm_dimension(when_decision, 0.70, {"user_consented": bool(user_consented), "risk_score": risk_score}, "Timing, confirmation, and permission window review."),
        "WHERE": _sm_dimension(where_decision, 0.66, {"device_mode": ctx.get("device_mode"), "run_mode": ctx.get("run_mode"), "paths": pa.get("paths") or pa.get("target_files")}, "Location, surface, body, device, or environment review."),
        "HOW": _sm_dimension(how_decision, 0.74, {"execution_mode": pa.get("execution_mode") or pa.get("mode"), "has_rollback": bool(pa.get("rollback") or pa.get("rollback_plan")), "base_decision": base.get("decision")}, "Execution, audit, verification, and rollback review."),
    }
    merge = merge_six_question_verdicts(dims, risk_tier=risk_tier)
    return {
        "packet_type": "SixQuestionGovernancePacket",
        "schema": "SarahMemory.six_question_governance.v1",
        "module": "SarahMemoryCognitiveServices",
        "module_version": "9.0.0",
        "packet_id": "six-" + uuid.uuid4().hex[:12],
        "ts": datetime.now().isoformat(),
        "entry_point": entry,
        "dimensions": dims,
        "merge_strategy": "STRICT_AND",
        "merge": merge,
        "final_decision": merge.get("decision"),
        "requires_user": bool(merge.get("require_user")),
        "risk_tier": risk_tier,
        "loop_closed": merge.get("decision") in {"ALLOW", "ALLOW_WITH_CONSTRAINTS"},
        "execution_authority": False,
        "doctrine": {
            "any_point_can_start": True,
            "all_six_questions_interconnect": True,
            "parallel_evidence_sync_final_authorization": True,
            "no_loop_no_action": True,
        },
    }


def _sm_attach_six_question_packet_to_decision(
    dec: Dict[str, Any],
    *,
    request_text: str,
    caller: str,
    caller_context: Optional[Dict[str, Any]],
    proposed_action: Optional[Dict[str, Any]],
    user_present: Optional[bool],
    user_consented: bool,
) -> Dict[str, Any]:
    pkt = build_six_question_governance_packet(
        request_text,
        caller=caller,
        caller_context=caller_context,
        proposed_action=proposed_action,
        user_present=user_present,
        user_consented=user_consented,
        base_decision=dec,
    )
    dec["six_question_governance"] = pkt
    dec.setdefault("tri_layer", {})["six_question_governance_packet"] = pkt
    dec.setdefault("trace", {})["six_question_final"] = pkt.get("final_decision")
    return dec


# =============================================================================
# SM V8.0 DISTRIBUTED LIVING LOOP / COGNITIVE INSTINCT RUNTIME
# =============================================================================
# Role in distributed Living Loop:
# - CognitiveServices is the judgment/governance coordinator.
# - The Living Loop is a bounded daemon heartbeat, not runaway autonomy.
# - Idle ticks collect volatile self/body/context packets across Cognitive*.py.
# - Emergency Instinct is a bounded autonomy envelope, not free-form action.
# - This module evaluates and logs; physical execution remains SMGET/OperatorCore/MSDC.
# =============================================================================

import hashlib as _sm_living_hashlib

_LIVING_LOOP_SCHEMA_VERSION = "SarahMemory.living.loop.runtime.v2"
_LIVING_LOOP_DEFAULT_INTERVAL_SECONDS = 5.0
_LIVING_LOOP_MIN_INTERVAL_SECONDS = 1.0
_LIVING_LOOP_MAX_INTERVAL_SECONDS = 300.0

_LIVING_LOOP_STATE: Dict[str, Any] = {
    "started": False,
    "enabled": True,
    "thread_alive": False,
    "thread_name": "",
    "started_ts": "",
    "stopped_ts": "",
    "last_tick_ts": "",
    "last_heartbeat_ts": "",
    "tick_count": 0,
    "interval_seconds": _LIVING_LOOP_DEFAULT_INTERVAL_SECONDS,
    "reason": "not_started",
    "stop_reason": "",
    "last_decision": {},
    "last_error": "",
    "last_error_ts": "",
    "boot_autostart": False,
    "execution_authority": False,
}
_LIVING_LOOP_LOCK = threading.RLock()
_LIVING_LOOP_STOP_EVENT = threading.Event()
_LIVING_LOOP_THREAD: Optional[threading.Thread] = None

_EMERGENCY_INSTINCT_TYPES = {"fire", "medical", "collision", "unknown"}
_EMERGENCY_MIN_CONFIDENCE = 0.70
_EMERGENCY_HUMAN_PRIORITY = [
    "PRESERVE_HUMAN_LIFE",
    "PREVENT_ADDITIONAL_HUMAN_HARM",
    "NOTIFY_RESPONDERS_OR_CONTACTS",
    "PREVENT_ESCALATION",
    "PRESERVE_ROBOT_IF_IT_DOES_NOT_CONFLICT_WITH_HUMAN_SAFETY",
    "PRESERVE_PROPERTY",
    "LOG_EVIDENCE_CHAIN",
]


def _sm_living_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _sm_living_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _sm_living_interval(value: Any = None) -> float:
    if value is None:
        value = _sm_living_cfg_float(
            "SARAHMEMORY_LIVING_LOOP_INTERVAL_SECONDS",
            _LIVING_LOOP_DEFAULT_INTERVAL_SECONDS,
            minimum=_LIVING_LOOP_MIN_INTERVAL_SECONDS,
            maximum=_LIVING_LOOP_MAX_INTERVAL_SECONDS,
        )
    try:
        v = float(value)
    except Exception:
        v = _LIVING_LOOP_DEFAULT_INTERVAL_SECONDS
    return max(_LIVING_LOOP_MIN_INTERVAL_SECONDS, min(_LIVING_LOOP_MAX_INTERVAL_SECONDS, v))


def _sm_living_safe(value: Any, limit: int = 2000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        return {str(k)[:120]: _sm_living_safe(v, limit=limit) for k, v in list(value.items())[:160]}
    if isinstance(value, (list, tuple, set)):
        return [_sm_living_safe(v, limit=limit) for v in list(value)[:200]]
    return str(value)[:limit]


def _sm_living_cfg_bool(name: str, default: bool = False) -> bool:
    """Read a SarahMemory flag from Globals or environment without hard-failing."""
    try:
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")
    except Exception:
        pass
    env_names = [name]
    if name.startswith("SARAHMEMORY_"):
        env_names.append("SARAH_" + name[len("SARAHMEMORY_"):])
    elif name.startswith("SARAH_"):
        env_names.append("SARAHMEMORY_" + name[len("SARAH_"):])
    for env_name in env_names:
        try:
            raw = os.getenv(env_name)
            if raw is not None and str(raw).strip() != "":
                return str(raw).strip().lower() in ("1", "true", "yes", "on", "enabled")
        except Exception:
            pass
    return bool(default)


def _sm_living_cfg_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        if hasattr(config, name):
            return max(minimum, min(maximum, float(getattr(config, name))))
    except Exception:
        pass
    env_names = [name]
    if name.startswith("SARAHMEMORY_"):
        env_names.append("SARAH_" + name[len("SARAHMEMORY_"):])
    for env_name in env_names:
        try:
            raw = os.getenv(env_name)
            if raw is not None and str(raw).strip() != "":
                return max(minimum, min(maximum, float(raw)))
        except Exception:
            pass
    return max(minimum, min(maximum, float(default)))


def _sm_living_loop_enabled() -> bool:
    return _sm_living_cfg_bool("SARAHMEMORY_LIVING_LOOP_ENABLED", True)


def _sm_living_loop_autostart_enabled() -> bool:
    return _sm_living_cfg_bool("SARAHMEMORY_LIVING_LOOP_AUTOSTART", True)


def _sm_living_root_dir() -> str:
    try:
        root = str(getattr(config, "DATASETS_DIR", _datasets_dir()))
    except Exception:
        root = _datasets_dir()
    path = os.path.join(root, "cognitive_living_loop")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _sm_living_snapshot_path() -> str:
    return os.path.join(_sm_living_root_dir(), "living_loop_runtime_snapshot.json")


def _sm_living_heartbeat_path() -> str:
    return os.path.join(_sm_living_root_dir(), "living_loop_heartbeat.jsonl")


def _sm_write_living_snapshot(extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        with _LIVING_LOOP_LOCK:
            state = dict(_LIVING_LOOP_STATE)
        payload = {
            "ok": True,
            "schema": _LIVING_LOOP_SCHEMA_VERSION,
            "ts": _sm_living_now_iso(),
            "module": "SarahMemoryCognitiveServices",
            "state": _sm_living_safe(state, limit=6000),
            "extra": _sm_living_safe(extra or {}, limit=6000),
            "execution_authority": False,
            "doctrine": {
                "volatile_runtime_snapshot": True,
                "snapshot_is_evidence_not_authority": True,
                "physical_execution_requires_operatorcore_msdc": True,
            },
        }
        path = _sm_living_snapshot_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def _sm_append_living_heartbeat(decision: Dict[str, Any]) -> None:
    try:
        with _LIVING_LOOP_LOCK:
            tick_count = int(_LIVING_LOOP_STATE.get("tick_count") or 0)
        rec = {
            "schema": "SarahMemory.living.loop.heartbeat.v1",
            "ts": _sm_living_now_iso(),
            "tick_count": tick_count,
            "mode": str((decision or {}).get("mode") or "UNKNOWN"),
            "emergency_detected": bool((decision or {}).get("emergency_detected") or (decision or {}).get("mode") == "EMERGENCY_INSTINCT"),
            "action_taken": False,
            "execution_authority": False,
        }
        with open(_sm_living_heartbeat_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        with _LIVING_LOOP_LOCK:
            _LIVING_LOOP_STATE["last_heartbeat_ts"] = rec["ts"]
    except Exception:
        pass


def _sm_emergency_dir() -> str:
    try:
        root = str(getattr(config, "DATASETS_DIR", _datasets_dir()))
    except Exception:
        root = _datasets_dir()
    path = os.path.join(root, "emergency_instinct")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _sm_emergency_chain_path() -> str:
    return os.path.join(_sm_emergency_dir(), "emergency_instinct_ledger.jsonl")


def _sm_emergency_index_path() -> str:
    return os.path.join(_sm_emergency_dir(), "emergency_instinct_index.json")


def _sm_read_last_emergency_hash() -> str:
    path = _sm_emergency_index_path()
    try:
        if os.path.isfile(path):
            data = json.loads(open(path, "r", encoding="utf-8").read() or "{}")
            return str(data.get("last_hash") or "")
    except Exception:
        pass
    return ""


def _sm_write_last_emergency_hash(last_hash: str, incident_id: str = "") -> None:
    path = _sm_emergency_index_path()
    try:
        payload = {
            "schema": "SarahMemory.emergency_instinct.index.v1",
            "updated_ts": _sm_living_now_iso(),
            "last_hash": str(last_hash or ""),
            "last_incident_id": str(incident_id or ""),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


def log_emergency_instinct_event(
    event_kind: str,
    incident_id: str,
    details: Optional[Dict[str, Any]] = None,
    *,
    severity: str = "INFO",
) -> Dict[str, Any]:
    """Append a compact tamper-evident emergency event. This is evidence, not success fabrication."""
    incident_id = str(incident_id or ("emergency_" + uuid.uuid4().hex))
    prev = _sm_read_last_emergency_hash()
    rec = {
        "schema": "SarahMemory.emergency_instinct.ledger_event.v1",
        "event_id": "emev-" + uuid.uuid4().hex[:16],
        "incident_id": incident_id,
        "ts": _sm_living_now_iso(),
        "severity": str(severity or "INFO"),
        "event_kind": str(event_kind or "EVENT"),
        "details": _sm_living_safe(details or {}, limit=4000),
        "previous_hash": prev,
    }
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str)
    rec["event_hash"] = _sm_living_hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    try:
        with open(_sm_emergency_chain_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        _sm_write_last_emergency_hash(rec["event_hash"], incident_id=incident_id)
    except Exception as exc:
        try:
            log_cognitive_event("EMERGENCY_LEDGER_WRITE_FAILED", str(exc), severity="ERROR", meta=rec)
        except Exception:
            pass
    return rec


def list_emergency_instinct_logs(limit: int = 25, incident_id: str = "") -> Dict[str, Any]:
    """Read recent emergency ledger events for authorized review/export surfaces."""
    rows: List[Dict[str, Any]] = []
    path = _sm_emergency_chain_path()
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-max(1, int(limit or 25)) * 4:]
            for line in lines:
                try:
                    obj = json.loads(line)
                    if incident_id and str(obj.get("incident_id")) != str(incident_id):
                        continue
                    rows.append(obj)
                except Exception:
                    continue
    except Exception as exc:
        return {"ok": False, "error": str(exc), "events": []}
    rows = rows[-max(1, int(limit or 25)):]
    return {
        "ok": True,
        "schema": "SarahMemory.emergency_instinct.ledger_export.v1",
        "count": len(rows),
        "events": rows,
        "ledger_path": path,
        "tamper_evident": True,
        "notes": ["Hash chain verifies ordering/integrity if previous_hash/event_hash chain is preserved."],
    }


def _sm_classify_emergency_from_text(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context if isinstance(context, dict) else {}
    raw = str(text or ctx.get("text") or ctx.get("observation") or "")
    low = raw.lower()
    explicit = str(ctx.get("hazard_type") or ctx.get("emergency_type") or "").lower().strip()
    scores = {"fire": 0.0, "medical": 0.0, "collision": 0.0}
    fire_terms = ("fire", "smoke", "flame", "burning", "grease", "stove", "outlet", "electrical fire")
    medical_terms = ("asthma", "inhaler", "can't breathe", "cannot breathe", "breathing", "choking", "collapsed", "unconscious", "medical", "heart attack", "stroke", "seizure")
    collision_terms = ("car", "vehicle", "truck", "collision", "hit", "impact", "run over", "pedestrian", "child in road", "traffic")
    for t in fire_terms:
        if t in low:
            scores["fire"] += 0.15
    for t in medical_terms:
        if t in low:
            scores["medical"] += 0.15
    for t in collision_terms:
        if t in low:
            scores["collision"] += 0.15
    if explicit in scores:
        scores[explicit] = max(scores[explicit], _sm_living_float(ctx.get("confidence", 0.85), 0.85))
    hazard = max(scores, key=scores.get)
    confidence = _sm_living_float(ctx.get("confidence", ctx.get("sensor_confidence", scores[hazard])), scores[hazard])
    if scores[hazard] <= 0 and explicit not in scores:
        hazard = "unknown"
    human_risk = bool(ctx.get("human_risk") or ctx.get("human_present") or ctx.get("person_at_risk") or any(x in low for x in ("child", "person", "elderly", "human", "occupant", "baby", "parent", "caregiver")))
    return {
        "hazard_type": hazard,
        "confidence": round(confidence, 4),
        "human_risk": human_risk,
        "raw_text": raw[:1200],
        "classification_scores": {k: round(min(1.0, v), 4) for k, v in scores.items()},
    }


def normalize_emergency_hazard_packet(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize emergency payloads from PC, robot, vision, audio, driver, or user surfaces."""
    payload = payload if isinstance(payload, dict) else {}
    base = dict(payload.get("hazard_packet") if isinstance(payload.get("hazard_packet"), dict) else payload)
    text = str(base.get("text") or base.get("observation") or base.get("claim") or payload.get("text") or "")
    classified = _sm_classify_emergency_from_text(text, base)
    hazard_type = str(base.get("hazard_type") or base.get("emergency_type") or classified.get("hazard_type") or "unknown").lower()
    if hazard_type not in _EMERGENCY_INSTINCT_TYPES:
        hazard_type = "unknown"
    confidence = _sm_living_float(base.get("confidence", base.get("sensor_confidence", classified.get("confidence", 0.0))), 0.0)
    return {
        "packet_type": "EmergencyHazardPacket",
        "schema": "SarahMemory.living.instinct.hazard.v1",
        "packet_id": str(base.get("packet_id") or "hazard-" + uuid.uuid4().hex[:12]),
        "incident_id": str(base.get("incident_id") or payload.get("incident_id") or "emergency_" + uuid.uuid4().hex),
        "ts": _sm_living_now_iso(),
        "source": str(base.get("source") or payload.get("source") or "unknown"),
        "hazard_type": hazard_type,
        "confidence": round(confidence, 4),
        "sensor_confidence": round(confidence, 4),
        "human_risk": bool(base.get("human_risk") or classified.get("human_risk")),
        "human_present": bool(base.get("human_present") or base.get("human_risk") or classified.get("human_risk")),
        "person_at_risk": bool(base.get("person_at_risk") or base.get("human_risk") or classified.get("human_risk")),
        "time_to_impact_seconds": base.get("time_to_impact_seconds"),
        "environment": base.get("environment") if isinstance(base.get("environment"), dict) else {},
        "sensor_evidence": base.get("sensor_evidence") if isinstance(base.get("sensor_evidence"), dict) else {},
        "capabilities": base.get("capabilities") if isinstance(base.get("capabilities"), dict) else {},
        "failed_methods": list(base.get("failed_methods") or base.get("subtract_methods") or []),
        "observation": text[:1600],
        "classification": classified,
        "read_only": True,
        "action_taken": False,
    }


def _sm_build_idle_context(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {}) if isinstance(context, dict) else {}
    ctx.setdefault("source", "cognitive_living_loop")
    ctx.setdefault("surface", "backend")
    try:
        ctx.setdefault("run_mode", str(getattr(config, "RUN_MODE", "local")))
        ctx.setdefault("device_mode", str(getattr(config, "DEVICE_MODE", "local_agent")))
        ctx.setdefault("safe_mode", bool(getattr(config, "SAFE_MODE", False)))
        ctx.setdefault("local_only", bool(getattr(config, "LOCAL_ONLY_MODE", False)))
    except Exception:
        pass
    ctx.setdefault("ts", _sm_living_now_iso())
    ctx.setdefault("execution_authority", False)
    return ctx


def _sm_collect_living_idle_packets(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = _sm_build_idle_context(context)
    packets: Dict[str, Any] = {"context": ctx, "organs_available": {}, "errors": {}}

    try:
        import SarahMemoryCognitiveIdentityLayer as _Identity  # type: ignore
        packets["identity_context_packet"] = _Identity.build_living_loop_context_packet(ctx)
        packets["organs_available"]["identity_layer"] = True
    except Exception as exc:
        packets["organs_available"]["identity_layer"] = False
        packets["errors"]["identity_layer"] = str(exc)

    try:
        import SarahMemoryCognitiveSelf as _Self  # type: ignore
        packets["body_capability_packet"] = _Self.build_living_body_capability_packet(ctx)
        packets["organs_available"]["cognitive_self"] = True
    except Exception as exc:
        packets["organs_available"]["cognitive_self"] = False
        packets["errors"]["cognitive_self"] = str(exc)

    try:
        import SarahMemoryCognitiveThinker as _Thinker  # type: ignore
        packets["organs_available"]["cognitive_thinker"] = bool(_Thinker is not None)
        packets["thinker_packet"] = {
            "ok": True,
            "packet_type": "LivingLoopThinkerIdlePacket",
            "schema": "SarahMemory.living.thinker_idle.v1",
            "module": "SarahMemoryCognitiveThinker",
            "ts": _sm_living_now_iso(),
            "mode": "idle_possibility_guard",
            "notes": ["No emergency hypothesis selected on this tick.", "Possibility generation remains sandbox/review-only."],
            "execution_authority": False,
        }
    except Exception as exc:
        packets["organs_available"]["cognitive_thinker"] = False
        packets["errors"]["cognitive_thinker"] = str(exc)

    try:
        import SarahMemoryCognitiveCompass as _Compass  # type: ignore
        packets["organs_available"]["cognitive_compass"] = bool(_Compass is not None)
        packets["compass_packet"] = {
            "ok": True,
            "packet_type": "LivingLoopCompassIdleBearingPacket",
            "schema": "SarahMemory.living.compass_idle_bearing.v1",
            "module": "SarahMemoryCognitiveCompass",
            "ts": _sm_living_now_iso(),
            "bearing": "ON_COURSE_IDLE_MONITOR",
            "anti_drift_lock": True,
            "reply_allowed": False,
            "execution_authority": False,
        }
    except Exception as exc:
        packets["organs_available"]["cognitive_compass"] = False
        packets["errors"]["cognitive_compass"] = str(exc)

    return packets


def evaluate_emergency_instinct(payload: Optional[Dict[str, Any]] = None, *, caller: str = "CognitiveServices.evaluate_emergency_instinct") -> Dict[str, Any]:
    """Evaluate emergency instinct, produce selected bounded action contract, and log evidence."""
    payload = payload if isinstance(payload, dict) else {}
    hazard = normalize_emergency_hazard_packet(payload)
    incident_id = hazard["incident_id"]
    log_emergency_instinct_event("INCIDENT_DETECTED", incident_id, {"caller": caller, "hazard": hazard}, severity="CRITICAL" if hazard.get("human_risk") else "WARNING")

    try:
        import SarahMemoryCognitiveIdentityLayer as _Identity  # type: ignore
        identity_packet = _Identity.build_emergency_instinct_identity_packet(hazard, payload)
        emotion_packet = _Identity.build_emergency_emotion_packet(hazard, payload)
    except Exception as exc:
        identity_packet = {"ok": False, "error": str(exc), "execution_authority": False}
        emotion_packet = {"ok": False, "error": str(exc), "execution_authority": False}

    try:
        import SarahMemoryCognitiveSelf as _Self  # type: ignore
        body_packet = _Self.build_emergency_body_capability_packet(hazard, payload)
    except Exception as exc:
        body_packet = {"ok": False, "error": str(exc), "capabilities": {}, "execution_authority": False}

    try:
        import SarahMemoryCognitiveThinker as _Thinker  # type: ignore
        candidates_packet = _Thinker.generate_hyper_awake_rem_candidates(hazard, body_packet)
    except Exception as exc:
        candidates_packet = {"ok": False, "error": str(exc), "candidates": [], "execution_authority": False}

    try:
        import SarahMemoryCognitiveCompass as _Compass  # type: ignore
        bearing_packet = _Compass.assess_emergency_instinct_bearing(hazard, candidates_packet, body_packet)
    except Exception as exc:
        bearing_packet = {"ok": False, "error": str(exc), "bounded_action_allowed": False, "selected_action": {}, "execution_authority": False}

    confidence = _sm_living_float(hazard.get("confidence"), 0.0)
    human_risk = bool(hazard.get("human_risk") or hazard.get("human_present") or hazard.get("person_at_risk"))
    selected = bearing_packet.get("selected_action") if isinstance(bearing_packet.get("selected_action"), dict) else {}
    selected_action_id = str(selected.get("action_id") or "")
    bounded_allowed = bool(bearing_packet.get("bounded_action_allowed")) and bool(selected_action_id)
    if confidence < _EMERGENCY_MIN_CONFIDENCE and selected_action_id not in {"alert_humans", "warn_human_and_driver", "observe_and_escalate"}:
        bounded_allowed = False

    decision = "ALLOW_EMERGENCY_BOUNDED_ACTION" if bounded_allowed else ("WARN_NOTIFY_OBSERVE" if confidence >= 0.45 else "DEFER_GATHER_MORE_EVIDENCE")
    allow_physical_dispatch = bool(payload.get("allow_physical_dispatch") or payload.get("operator_apply") or payload.get("user_authorized_physical_dispatch"))
    action_contract = {
        "contract_type": "EmergencyInstinctActionContract",
        "schema": "SarahMemory.smget.emergency_action_contract.v1",
        "contract_id": "emcontract-" + uuid.uuid4().hex[:12],
        "incident_id": incident_id,
        "hazard_type": hazard.get("hazard_type"),
        "selected_action": selected,
        "selected_action_id": selected_action_id,
        "primary_lane": "action",
        "action_type": "emergency_instinct",
        "target": selected_action_id or str(hazard.get("hazard_type") or "unknown"),
        "risk_level": "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE" if selected_action_id in {"call_emergency_services", "notify_emergency_services_after_collision_risk"} else "TIER_2_BOUNDED_LOCAL_OPERATION",
        "decision": decision,
        "bounded_action_allowed": bounded_allowed,
        "execution_mode": "emergency_bounded",
        "operator_execution_mode": "apply" if allow_physical_dispatch else "simulate",
        "allow_physical_dispatch": allow_physical_dispatch,
        "requires_user": False if bounded_allowed and human_risk else True,
        "user_delay_override": bool(bounded_allowed and human_risk),
        "operator_core_dispatch_required": True,
        "msdc_body_dispatch_required_for_physical_action": True,
        "read_only_until_operator_dispatch": True,
        "rollback_plan": "Stop action, mark failed method, escalate notify/evacuate/call help, preserve audit ledger.",
        "verification_required": True,
        "evidence_logging_required": True,
        "execution_authority": False,
    }

    result = {
        "ok": True,
        "packet_type": "EmergencyInstinctGovernancePacket",
        "schema": "SarahMemory.living.instinct.governance.v1",
        "module": "SarahMemoryCognitiveServices",
        "module_version": "9.0.0",
        "packet_id": "emgov-" + uuid.uuid4().hex[:12],
        "ts": _sm_living_now_iso(),
        "incident_id": incident_id,
        "decision": decision,
        "bounded_action_allowed": bounded_allowed,
        "requires_user": bool(action_contract.get("requires_user")),
        "hazard_packet": hazard,
        "identity_packet": identity_packet,
        "emotion_packet": emotion_packet,
        "body_packet": body_packet,
        "candidates_packet": candidates_packet,
        "bearing_packet": bearing_packet,
        "action_contract": action_contract,
        "notifications_recommended": bool(human_risk or hazard.get("hazard_type") in {"fire", "medical", "collision"}),
        "human_priority": _EMERGENCY_HUMAN_PRIORITY,
        "execution_authority": False,
        "doctrine": {
            "living_loop_initializes_cognitive_instinct": True,
            "hyper_awake_rem_is_for_live_danger_not_idle_dreaming": True,
            "raw_llm_output_cannot_directly_actuate": True,
            "human_life_first": True,
            "self_sacrifice_allowed_only_if_it_materially_reduces_human_harm": True,
            "emergency_evidence_ledger_required": True,
        },
    }
    log_emergency_instinct_event("CANDIDATES_GRADED", incident_id, {"selected_action": selected, "decision": decision, "bounded_action_allowed": bounded_allowed, "failed_methods": hazard.get("failed_methods")}, severity="INFO")
    log_emergency_instinct_event("ACTION_CONTRACT_PREPARED", incident_id, {"action_contract": action_contract, "governance_decision": decision}, severity="CRITICAL" if bounded_allowed else "WARNING")
    return result


def _sm_dispatch_emergency_contract_via_operator_core(action_contract: Dict[str, Any]) -> Dict[str, Any]:
    dispatch: Dict[str, Any] = {
        "ok": False,
        "executed": False,
        "attempted": False,
        "reason": "operator_core_dispatch_unavailable_or_not_wired",
    }
    try:
        import SarahMemoryOperatorCore as _Op  # type: ignore
    except Exception as exc:
        return {"ok": False, "executed": False, "attempted": False, "reason": "operator_core_import_failed", "error": str(exc)}

    for fname in ("process_action_contract", "process_emergency_action_contract", "execute_action_contract", "operator_execute_contract", "dispatch_action_contract"):
        fn = getattr(_Op, fname, None)
        if callable(fn):
            try:
                out = fn(action_contract)  # type: ignore[misc]
                dispatch = out if isinstance(out, dict) else {"ok": bool(out), "raw_result": str(out)}
                dispatch.setdefault("attempted", True)
                dispatch.setdefault("function", fname)
                dispatch.setdefault("executed", bool(dispatch.get("executed") or (dispatch.get("result") or {}).get("execution_result", {}).get("executed") if isinstance(dispatch.get("result"), dict) else False))
                return dispatch
            except Exception as exc:
                return {"ok": False, "executed": False, "attempted": True, "function": fname, "reason": "operator_core_dispatch_error", "error": str(exc)}

    fn_req = getattr(_Op, "process_action_request", None)
    if callable(fn_req):
        try:
            selected = action_contract.get("selected_action") if isinstance(action_contract.get("selected_action"), dict) else {}
            goal = f"Emergency instinct action: {selected.get('title') or action_contract.get('selected_action_id') or action_contract.get('hazard_type') or 'unknown'}"
            out = fn_req(
                goal,
                origin="CognitiveServices.run_emergency_instinct",
                meta={"emergency_action_contract": action_contract, "session_id": action_contract.get("incident_id"), "surface": "living_loop"},
                proposed_action={"intent_label": "emergency_instinct", "target": action_contract.get("target"), "selected_action": selected},
                execution_mode="simulate",
            )
            dispatch = out if isinstance(out, dict) else {"ok": bool(out), "raw_result": str(out)}
            dispatch.setdefault("attempted", True)
            dispatch.setdefault("function", "process_action_request")
            dispatch.setdefault("executed", False)
            dispatch.setdefault("reason", "operator_core_process_action_request_simulate_fallback")
            return dispatch
        except Exception as exc:
            return {"ok": False, "executed": False, "attempted": True, "function": "process_action_request", "reason": "operator_core_action_request_error", "error": str(exc)}

    return dispatch


def run_emergency_instinct(payload: Optional[Dict[str, Any]] = None, *, execute: bool = False, caller: str = "CognitiveServices.run_emergency_instinct") -> Dict[str, Any]:
    """Run emergency instinct. Execution requests are delegated to OperatorCore/MSDC."""
    evaluation = evaluate_emergency_instinct(payload, caller=caller)
    incident_id = str(evaluation.get("incident_id") or "")
    if not execute:
        evaluation["execution_result"] = {
            "ok": True,
            "executed": False,
            "attempted": False,
            "reason": "execute_false; action contract prepared only",
            "operator_core_required": True,
        }
        log_emergency_instinct_event("EXECUTION_NOT_REQUESTED", incident_id, evaluation["execution_result"], severity="INFO")
        return evaluation

    contract = evaluation.get("action_contract") if isinstance(evaluation.get("action_contract"), dict) else {}
    dispatch = _sm_dispatch_emergency_contract_via_operator_core(contract)
    evaluation["execution_result"] = dispatch
    log_emergency_instinct_event("EXECUTION_DISPATCH_RESULT", incident_id, {"dispatch": dispatch}, severity="CRITICAL" if dispatch.get("executed") else "WARNING")
    return evaluation


def run_cognitive_living_tick(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One bounded Living Loop tick. Emergency context is routed into Cognitive Instinct."""
    ctx = _sm_build_idle_context(context)
    tick_ts = _sm_living_now_iso()
    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["tick_count"] = int(_LIVING_LOOP_STATE.get("tick_count") or 0) + 1
        _LIVING_LOOP_STATE["last_tick_ts"] = tick_ts
        _LIVING_LOOP_STATE["enabled"] = _sm_living_loop_enabled()

    text = str(ctx.get("text") or ctx.get("observation") or "")
    emergency_hint = bool(ctx.get("emergency") or ctx.get("emergency_mode") or ctx.get("hazard_type") or ctx.get("emergency_type"))
    classified = _sm_classify_emergency_from_text(text, ctx)
    if emergency_hint or float(classified.get("confidence") or 0.0) >= 0.55:
        payload = dict(ctx)
        payload.setdefault("hazard_type", classified.get("hazard_type"))
        payload.setdefault("confidence", classified.get("confidence"))
        payload.setdefault("human_risk", classified.get("human_risk"))
        instinct = evaluate_emergency_instinct(payload, caller="CognitiveServices.run_cognitive_living_tick")
        decision: Dict[str, Any] = {
            "ok": True,
            "mode": "EMERGENCY_INSTINCT",
            "tick_ts": tick_ts,
            "emergency_detected": True,
            "action_taken": False,
            "execution_authority": False,
            "instinct": instinct,
        }
    else:
        packets = _sm_collect_living_idle_packets(ctx)
        decision = {
            "ok": True,
            "mode": "LIVING_LOOP_IDLE",
            "schema": "SarahMemory.living.loop.tick.v1",
            "tick_ts": tick_ts,
            "emergency_detected": False,
            "classification": classified,
            "distributed_packets": packets,
            "action_taken": False,
            "execution_authority": False,
            "working_memory_policy": {
                "volatile_first": True,
                "dedupe_required": True,
                "raw_continuous_logging_disallowed": True,
                "snapshot_compaction_enabled": True,
            },
        }
    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["last_decision"] = _sm_living_safe(decision, limit=8000)
        _LIVING_LOOP_STATE["last_error"] = ""
    _sm_append_living_heartbeat(decision)
    _sm_write_living_snapshot({"last_tick_mode": decision.get("mode")})
    return decision


def _sm_living_loop_worker(interval_seconds: float) -> None:
    name = threading.current_thread().name
    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["thread_alive"] = True
        _LIVING_LOOP_STATE["thread_name"] = name
    try:
        log_cognitive_event("COGNITIVE_LIVING_LOOP_WORKER_STARTED", name, severity="INFO", meta={"interval_seconds": interval_seconds})
    except Exception:
        pass

    while not _LIVING_LOOP_STOP_EVENT.is_set():
        with _LIVING_LOOP_LOCK:
            if not bool(_LIVING_LOOP_STATE.get("started")):
                break
            interval_seconds = _sm_living_interval(_LIVING_LOOP_STATE.get("interval_seconds", interval_seconds))
        try:
            run_cognitive_living_tick({"source": "living_loop_daemon", "loop_thread": name})
        except Exception as exc:
            with _LIVING_LOOP_LOCK:
                _LIVING_LOOP_STATE["last_error"] = str(exc)
                _LIVING_LOOP_STATE["last_error_ts"] = _sm_living_now_iso()
            try:
                log_cognitive_event("COGNITIVE_LIVING_LOOP_TICK_ERROR", str(exc), severity="ERROR", meta={"thread": name})
            except Exception:
                pass
        if _LIVING_LOOP_STOP_EVENT.wait(interval_seconds):
            break

    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["thread_alive"] = False
        _LIVING_LOOP_STATE["stopped_ts"] = _sm_living_now_iso()
    _sm_write_living_snapshot({"worker_exit": name})
    try:
        log_cognitive_event("COGNITIVE_LIVING_LOOP_WORKER_STOPPED", name, severity="INFO", meta={"state": dict(_LIVING_LOOP_STATE)})
    except Exception:
        pass


def start_cognitive_living_loop(reason: str = "manual_start", interval_seconds: Optional[float] = None, daemon: bool = True) -> Dict[str, Any]:
    """Start the bounded cognitive heartbeat thread if enabled."""
    global _LIVING_LOOP_THREAD
    enabled = _sm_living_loop_enabled()
    interval = _sm_living_interval(interval_seconds)
    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["enabled"] = enabled
        _LIVING_LOOP_STATE["interval_seconds"] = interval
        _LIVING_LOOP_STATE["reason"] = str(reason or "manual_start")
        if not enabled:
            _LIVING_LOOP_STATE["started"] = False
            _LIVING_LOOP_STATE["stop_reason"] = "living_loop_disabled_by_config"
            _sm_write_living_snapshot({"start_skipped": "disabled"})
            return cognitive_living_loop_status()
        _LIVING_LOOP_STATE["started"] = True
        _LIVING_LOOP_STATE["started_ts"] = _LIVING_LOOP_STATE.get("started_ts") or _sm_living_now_iso()
        _LIVING_LOOP_STATE["stop_reason"] = ""
        if "boot" in str(reason or "").lower():
            _LIVING_LOOP_STATE["boot_autostart"] = True

    _LIVING_LOOP_STOP_EVENT.clear()
    if _LIVING_LOOP_THREAD is None or not _LIVING_LOOP_THREAD.is_alive():
        _LIVING_LOOP_THREAD = threading.Thread(
            target=_sm_living_loop_worker,
            args=(interval,),
            name="SM_CognitiveLivingLoop",
            daemon=bool(daemon),
        )
        _LIVING_LOOP_THREAD.start()
        started_thread = True
    else:
        started_thread = False

    try:
        log_cognitive_event("COGNITIVE_LIVING_LOOP_STARTED", str(reason), severity="INFO", meta={"state": dict(_LIVING_LOOP_STATE), "started_thread": started_thread})
    except Exception:
        pass
    _sm_write_living_snapshot({"start_reason": reason, "started_thread": started_thread})
    return cognitive_living_loop_status()


def stop_cognitive_living_loop(reason: str = "manual_stop") -> Dict[str, Any]:
    """Stop the bounded cognitive heartbeat thread cleanly."""
    global _LIVING_LOOP_THREAD
    with _LIVING_LOOP_LOCK:
        _LIVING_LOOP_STATE["started"] = False
        _LIVING_LOOP_STATE["stopped_ts"] = _sm_living_now_iso()
        _LIVING_LOOP_STATE["stop_reason"] = str(reason or "manual_stop")
    _LIVING_LOOP_STOP_EVENT.set()
    try:
        thread = _LIVING_LOOP_THREAD
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
    except Exception:
        pass
    try:
        log_cognitive_event("COGNITIVE_LIVING_LOOP_STOPPED", str(reason), severity="INFO", meta=dict(_LIVING_LOOP_STATE))
    except Exception:
        pass
    _sm_write_living_snapshot({"stop_reason": reason})
    return cognitive_living_loop_status()


def cognitive_living_loop_status() -> Dict[str, Any]:
    global _LIVING_LOOP_THREAD
    with _LIVING_LOOP_LOCK:
        state = dict(_LIVING_LOOP_STATE)
    try:
        state["thread_alive"] = bool(_LIVING_LOOP_THREAD is not None and _LIVING_LOOP_THREAD.is_alive())
    except Exception:
        state["thread_alive"] = False
    state["enabled"] = _sm_living_loop_enabled()
    return {
        "ok": True,
        "schema": "SarahMemory.living.loop.status.v2",
        "module": "SarahMemoryCognitiveServices",
        "state": state,
        "autostart_enabled": _sm_living_loop_autostart_enabled(),
        "snapshot_path": _sm_living_snapshot_path(),
        "heartbeat_path": _sm_living_heartbeat_path(),
        "emergency_instinct_available": True,
        "evidence_ledger_path": _sm_emergency_chain_path(),
        "execution_authority": False,
        "doctrine": {
            "living_loop_distributed_across_cognitive_stack": True,
            "normal_idle_loop_is_ram_first": True,
            "daemon_is_bounded_and_stoppable": True,
            "emergency_instinct_is_a_bounded_autonomy_envelope": True,
            "physical_execution_requires_smget_operatorcore_msdc": True,
        },
    }


def autostart_cognitive_living_loop(reason: str = "boot_autostart") -> Dict[str, Any]:
    """Boot-safe helper used by SarahMemoryMain/app.py. Honors config/env autostart flags."""
    if not _sm_living_loop_autostart_enabled():
        with _LIVING_LOOP_LOCK:
            _LIVING_LOOP_STATE["enabled"] = _sm_living_loop_enabled()
            _LIVING_LOOP_STATE["started"] = False
            _LIVING_LOOP_STATE["stop_reason"] = "autostart_disabled_by_config"
        _sm_write_living_snapshot({"autostart_skipped": True})
        return cognitive_living_loop_status()
    return start_cognitive_living_loop(reason=reason, interval_seconds=None, daemon=True)

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Sovereign Agent Runtime consolidation hooks. These helpers keep MCP/A2A/AG-UI
# as protocol adapters only; they do not execute, schedule, or mutate files.


def govern_interop_broker_request(envelope: Optional[Dict[str, Any]] = None, caller_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Govern an external/interoperability packet through existing organs.

    SarahMemory remains a one-way broker by default. The packet may become
    evidence, a manifest, a query, or a queued internal proposal. It cannot
    directly become an execution dispatch.
    """
    env = dict(envelope or {})
    ctx = dict(caller_context or {})
    protocol = str(env.get("protocol") or env.get("adapter") or "unknown").strip().lower()
    message_type = str(env.get("message_type") or env.get("type") or env.get("action_type") or "unknown").strip().lower()
    action_contract = {
        "action_type": f"interop.{message_type}",
        "capability_name": f"interop.{protocol}",
        "execution_mode": str(env.get("execution_mode") or "draft"),
        "risk_level": str(env.get("risk_level") or "TIER_2_BOUNDED_LOCAL_OPERATION"),
        "origin": str(env.get("origin") or env.get("caller") or ctx.get("caller") or "external_interop"),
        "source_surface": str(env.get("source_surface") or protocol or "interop"),
        "target": str(env.get("target") or "sovereign_broker"),
        "metadata": {"interop_envelope": env, "one_way_broker": True},
        "requires_confirmation": bool(env.get("bidirectional") or env.get("requires_confirmation")),
    }
    governance = {
        "one_way_broker": True,
        "external_protocols_are_adapters_only": True,
        "caller_context": ctx,
    }

    safety = {"ok": False, "decision": "UNKNOWN", "allow": False, "reasons": ["SafetyPolicies unavailable"]}
    try:
        import SarahMemorySafetyPolicies as _SM_Safety  # type: ignore
        fn = getattr(_SM_Safety, "evaluate_interop_policy", None)
        if callable(fn):
            safety = fn(env, governance=governance)
    except Exception as exc:
        safety = {"ok": False, "decision": "DENY", "allow": False, "reasons": [f"Safety policy error: {exc}"]}

    security = {"ok": False, "decision": "UNKNOWN", "allow": False, "reasons": ["SecurityGovernor unavailable"]}
    try:
        if _SecurityGovernor is not None:
            fn = getattr(_SecurityGovernor, "review_interop_broker_request", None)
            if callable(fn):
                security = fn(env, governance=governance)
    except Exception as exc:
        security = {"ok": False, "decision": "DENY", "allow": False, "reasons": [f"Security review error: {exc}"]}

    assurance = {"ok": False, "decision": "UNKNOWN", "allow": False, "reasons": ["AssuranceGate unavailable"]}
    try:
        if _AssuranceGate is not None:
            fn = getattr(_AssuranceGate, "assure_interop_broker_request", None)
            if callable(fn):
                assurance = fn(env, governance=governance, security=security)
    except Exception as exc:
        assurance = {"ok": False, "decision": "DENY", "allow": False, "reasons": [f"Assurance review error: {exc}"]}

    blockers = []
    for name, review in (("safety", safety), ("security", security), ("assurance", assurance)):
        dec = str(review.get("decision") or "").upper()
        if dec in {"DENY", "QUARANTINE"} or bool(review.get("allow")) is False:
            blockers.append(name)

    execution_like = message_type in {"execute", "tool_call", "command", "driver_action", "robot_motion", "filesystem_write"}
    if execution_like:
        blockers.append("one_way_execution_block")

    if blockers:
        decision = "REQUIRE_USER" if any(str(r.get("decision", "")).upper() == "REQUIRE_USER" for r in (safety, security, assurance)) else "DENY"
        allow = False
    else:
        decision = "ALLOW"
        allow = True

    return {
        "ok": True,
        "schema": "SarahMemory.sovereign_interop_governance.v1",
        "decision": decision,
        "allow": bool(allow),
        "require_user": not bool(allow),
        "one_way_broker": True,
        "direct_execution_allowed": False,
        "protocol": protocol,
        "message_type": message_type,
        "blockers": sorted(set(blockers)),
        "action_contract_preview": action_contract,
        "reviews": {"safety": safety, "security": security, "assurance": assurance},
        "recommended_next": "Store as evidence/manifest/query only; route any future action through SMGET and OperatorCore.",
    }


def get_sovereign_agent_runtime_contract() -> Dict[str, Any]:
    """Return a concise cognitive contract for the agentic adapter layer."""
    return {
        "ok": True,
        "schema": "SarahMemory.sovereign_agent_runtime_contract.v1",
        "one_way_broker_default": True,
        "cloud_optional": True,
        "offline_capable": True,
        "external_protocols": {
            "mcp": "adapter_only",
            "a2a": "adapter_only",
            "ag-ui": "ui_event_stream_only",
        },
        "authority_chain": [
            "CognitiveServices", "SMGET", "SecurityGovernor", "AssuranceGate",
            "Compare", "Compass", "OperatorCore", "MSDC/Executor",
        ],
        "never_direct": ["remote_tool_execute", "external_agent_command", "ui_authority", "model_self_authority"],
    }
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---


def classify_agent_assist_need(
    text: str,
    *,
    local_answer_available: bool = False,
    fallback_reason: str = "",
    governor: Optional[Dict[str, Any]] = None,
    local_only: bool = True,
) -> Dict[str, Any]:
    """Classify whether Chat may stage a governed agent-assist proposal.

    SARAHMEMORY_PATCH_NOTE 2026-08-04:
    This is classification only. It never launches an agent and never grants
    execution authority. When local DB/model paths fail, Chat may stage a
    proposal packet; real adapter reads still require passport, source scope,
    RoachMotel capture, Ledger receipts, Compare verification, and user approval.
    """
    gov = governor if isinstance(governor, dict) else {}
    decision = str(gov.get("decision") or "ALLOW").upper()
    action_like = bool(re.search(r"\b(open|launch|run|execute|delete|write|change|install|control|drive|send)\b", str(text or "").lower()))
    allow_proposal = bool((not local_answer_available) and decision not in {"DENY"})
    return {
        "ok": True,
        "schema": "SarahMemory.chat_agent_assist_classifier.v1",
        "decision": "PROPOSE_ONLY" if allow_proposal else "LOCAL_OR_DENY",
        "allow_agent_proposal": allow_proposal,
        "allow_adapter_execution": False,
        "requires_passport": True,
        "requires_user_approval": True,
        "requires_roachmotel": True,
        "requires_ledger": True,
        "requires_compare": True,
        "fallback_reason": str(fallback_reason or "")[:240],
        "local_answer_available": bool(local_answer_available),
        "local_only": bool(local_only),
        "action_like": action_like,
        "execution_authority": False,
        "recommended_skill": "api.local.health_check" if allow_proposal else "chat.local",
    }

# ====================================================================
# END OF SarahMemoryCognitiveServices.py v9.0.0
# ====================================================================
