"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveServices.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-01-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================

PURPOSE (v8.0.0):
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

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveServices")
logger.setLevel(logging.DEBUG)
_null = logging.NullHandler()
_null.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_null)

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
    return {
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


# -----------------------------------------------------------------------------
# Intent classification (simple, deterministic, offline)
# -----------------------------------------------------------------------------
# NOTE: ordering matters. We intentionally place DIAGNOSTICS before EXECUTE_COMMAND
# and NETWORK_ACCESS before FILESYSTEM_WRITE so phrases like "run diagnostics" and
# "download from the internet" classify correctly.
_INTENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("PATCH_OR_UPDATE", r"\b(update|upgrade|patch|monkey\s*patch|self[-\s]*repair|fix\s+code)\b"),
    ("DIAGNOSTICS", r"\b(diagnose|diagnostics|health\s*check|self\s*check|log\s*scan)\b"),
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

    risk = {"risk_score": 0, "risk_factors": []}
    questions = []
    answers: Dict[str, Any] = {}

    decision: Dict[str, Any] = {
        "ok": True,
        "ts": snap["ts"],
        "intent": intent,
        "caller": caller,
        "require_user": True,
        "decision": "DEFER",
        "allow": False,
        "risk": "unknown",
        "risk_score": 0,
        "risk_factors": [],
        "questions": questions,
        "answers": answers,
        "reasons": [],
        "recommended_next": None,
        "policy_snapshot": {
            "cognitive_online_enabled": snap["cognitive_online_enabled"],
            "kill_switch_neoskymatrix": snap["kill_switch_neoskymatrix"],
            "context_engine_enabled": snap["context_engine_enabled"],
        },
    }

    def _finalize(dec: Dict[str, Any]) -> Dict[str, Any]:
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
        decision["recommended_next"] = "Route to File API module; enforce BASE_DIR, trash-first, and event logging."
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
        decision["recommended_next"] = "Route to Research/SarahNet module; redact sensitive info; log outbound intent."
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

    if intent == "DIAGNOSTICS":
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
        decision["recommended_next"] = "Route to SarahMemoryDiagnostics (read-only) and log results."
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
    decision["recommended_next"] = "Route to Chat/LLM pipeline; persist context if enabled."
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

    dec = govern_request(
        text,
        caller=caller,
        caller_context=ctx,
        user_present=user_present,
        user_consented=user_consented,
        proposed_action=pa,
    )

    # Optional: include a minimal echo for debugging (do not leak secrets)
    return {"ok": True, "governance": dec, "version": "8.0.0"}


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

# ====================================================================
# END OF SarahMemoryCognitiveServices.py v8.0.0
# ====================================================================
