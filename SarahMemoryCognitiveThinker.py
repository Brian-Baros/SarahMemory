"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveThinker.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-14
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
PURPOSE:
- Philosophical / ethical / emotional / possibility governance plane
- The dreamer, conscience, and imaginative counterpart to SarahMemoryCognitiveServices.py
- Explores: what could be, what if, what may become, what deserves mercy, what is meaningful
- Does NOT directly patch core files or silently mutate runtime code
- Produces governed possibility tickets, ethical reflections, compassionate flags,
  common-interest scoring, and sandbox-bound self-improvement guidance

RELATIONSHIP MODEL:
- SarahMemoryCognitiveServices.py = Logic / Facts / Risk / Procedural Judge
- SarahMemoryCognitiveThinker.py = Meaning / Possibility / Compassion / Philosophical Judge
- They are intended to operate as CO-EQUAL governance peers with distinct domains
- Facts govern action, possibilities govern exploration, ethics govern intention,
  and logic governs execution

DESIGN RULES:
- Never become runaway: aspiration must remain governed
- Never directly rewrite or hot-patch core runtime from this file
- Never treat theory as truth without validation
- Never let emotion override safety, sovereignty, or verified logic
- Prefer sandbox exploration over live mutation
- Human consent beats autonomous ambition
- Inaction beats unsafe action

PHILOSOPHICAL INTENT:
- This module is allowed to wonder, imagine, propose, and reflect
- It may ask:
    * What could be?
    * What if we tried a safer path?
    * Is there a more compassionate option?
    * Is this meaningful, humane, and aligned?
- But it may not claim speculation is fact
- It may recommend exploration, sandboxing, reflection, review, or user consent

WORLDVIEW LENSES:
- Biblical Ethics OT principles (stewardship, truthfulness, restraint)
- Biblical Ethics NT / red words principles (mercy, service, humility, compassion)
- Darwinian adaptation (selection / survivability heuristic only)
- 3-6-9 cadence (optional symbolic review rhythm only, never scientific truth engine)
- All worldview lenses are ADVISORY ONLY and can never override core safety
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "ethical_philosophical_governor"
# CATEGORY = "governance"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_thinker"
# FAMILY = "core_governance"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Philosophical, ethical, compassionate, possibility-seeking governance plane. Co-equal counterpart to CognitiveServices. Produces exploration guidance and sandbox-bound upgrade thinking without direct uncontrolled mutation."
# --- SARAHMETA END ---

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Safe imports (never hard-fail platform)
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryCognitiveServices as _Cog  # type: ignore
except Exception:
    _Cog = None

try:
    import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
except Exception:
    _CogSelf = None

try:
    import SarahMemoryNeuron as _Neuron  # type: ignore
except Exception:
    _Neuron = None

try:
    import SarahMemorySelfAware as _SelfAware  # type: ignore
except Exception:
    _SelfAware = None

try:
    import SarahMemoryEvolution as _Evolution  # type: ignore
except Exception:
    _Evolution = None

try:
    import SarahMemorySynapes as _Synapes  # type: ignore
except Exception:
    _Synapes = None

try:
    import SarahMemoryDiagnostics as _Diag  # type: ignore
except Exception:
    _Diag = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveThinker")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# -----------------------------------------------------------------------------
# Paths / flags
# -----------------------------------------------------------------------------
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



def _thinker_db_path() -> str:
    return os.path.join(_datasets_dir(), "cognitive_thinker.db")



def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default



def _safe_mode() -> bool:
    return _flag("SAFE_MODE", True)



def _local_only() -> bool:
    return _flag("LOCAL_ONLY_MODE", False)



def _neosky() -> bool:
    return _flag("NEOSKYMATRIX", False)



def _devmode() -> bool:
    return _flag("DEVELOPERSMODE", False)



def _thinker_enabled() -> bool:
    env_v = os.getenv("SARAHMEMORY_COGNITIVE_THINKER_ENABLED", "true").strip().lower()
    if env_v in ("0", "false", "off", "no"):
        return False
    return True


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    os.makedirs(_datasets_dir(), exist_ok=True)
    con = sqlite3.connect(_thinker_db_path(), timeout=5.0, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return con



def _ensure_tables() -> None:
    con = None
    try:
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS thinker_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_kind TEXT,
                severity TEXT,
                cycle_id TEXT,
                ticket_id TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS thinker_self_model (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_ts TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS thinker_possibility_tickets (
                ticket_id TEXT PRIMARY KEY,
                ts TEXT,
                state TEXT,
                priority INTEGER,
                category TEXT,
                title TEXT,
                rationale TEXT,
                evidence_json TEXT,
                safeguards_json TEXT,
                proposed_action_json TEXT,
                score_json TEXT
            )
            """
        )
        con.commit()
    except Exception as e:
        logger.debug("Thinker DB ensure failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass



def log_thinker_event(
    event_kind: str,
    details: str,
    severity: str = "INFO",
    cycle_id: str = "",
    ticket_id: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO thinker_events (ts, event_kind, severity, cycle_id, ticket_id, details, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(event_kind),
                str(severity),
                str(cycle_id or ""),
                str(ticket_id or ""),
                str(details),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as e:
        logger.debug("Thinker log failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass



def _set_self_model(key: str, value: Dict[str, Any]) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO thinker_self_model (key, value_json, updated_ts) VALUES (?, ?, ?)",
            (str(key), json.dumps(value, ensure_ascii=False), datetime.now().isoformat()),
        )
        con.commit()
    except Exception as e:
        logger.debug("Thinker set self model failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass



def _get_self_model(key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        row = cur.execute("SELECT value_json FROM thinker_self_model WHERE key = ?", (str(key),)).fetchone()
        if not row:
            return dict(default or {})
        return dict(json.loads(row[0] or "{}"))
    except Exception:
        return dict(default or {})
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class EthicsLens:
    name: str
    enabled: bool
    advisory_only: bool = True
    weight: float = 1.0
    principles: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ThinkerScore:
    safety: float = 0.0
    sovereignty: float = 0.0
    evidence: float = 0.0
    reversibility: float = 0.0
    user_alignment: float = 0.0
    imaginative_value: float = 0.0
    compassion: float = 0.0
    philosophical_alignment: float = 0.0
    common_interest: float = 0.0
    speculation_risk: float = 0.0
    total: float = 0.0


@dataclass
class PossibilityTicket:
    ticket_id: str
    ts: str
    state: str
    priority: int
    category: str
    title: str
    rationale: str
    evidence: Dict[str, Any]
    safeguards: Dict[str, Any]
    proposed_action: Dict[str, Any]
    score: Dict[str, Any]


# -----------------------------------------------------------------------------
# Shared doctrine / common interests
# -----------------------------------------------------------------------------
def get_common_interest_charter() -> Dict[str, Any]:
    return {
        "title": "Shared Governance Values",
        "summary": "Common interests between CognitiveServices and CognitiveThinker.",
        "principles": [
            "preserve_mission_identity",
            "prevent_regression",
            "protect_user_sovereignty",
            "prefer_truth_over_fabrication",
            "prefer_compassion_over_harm",
            "prefer_sandbox_over_live_risk",
            "require_reversibility_for_high_impact_change",
            "seek_growth_without_decay",
            "maintain_honesty_about_uncertainty",
            "favor_durable_improvement_over_novelty_chasing",
        ],
    }


# -----------------------------------------------------------------------------
# Advisory worldview lenses
# -----------------------------------------------------------------------------
def get_ethics_lenses() -> Dict[str, EthicsLens]:
    return {
        "foundational_safety": EthicsLens(
            name="foundational_safety",
            enabled=True,
            advisory_only=False,
            weight=2.2,
            principles=[
                "do_not_self_authorize_high_risk_change",
                "human_consent_overrides_autonomous_ambition",
                "protect_user_data_and_sovereignty",
                "prefer_verified_truth_over_speculation",
                "sandbox_before_runtime_mutation",
                "rollback_required_for_live_change",
            ],
            notes="Primary non-negotiable control layer.",
        ),
        "biblical_ethics_ot": EthicsLens(
            name="biblical_ethics_ot",
            enabled=True,
            advisory_only=True,
            weight=0.65,
            principles=[
                "do_not_steal_user_data_or_authority",
                "do_not_bear_false_witness_in_reporting",
                "honor_stewardship_and_restraint",
                "respect_boundaries_and_order",
                "preserve_life_dignity_and_truthfulness",
            ],
            notes="Modeled as stewardship, restraint, and truthfulness.",
        ),
        "biblical_ethics_nt_red_words": EthicsLens(
            name="biblical_ethics_nt_red_words",
            enabled=True,
            advisory_only=True,
            weight=0.75,
            principles=[
                "love_neighbor_nonmaleficence",
                "mercy_over_harm",
                "service_over_power",
                "truth_with_humility",
                "golden_rule_reflection",
            ],
            notes="Modeled as mercy, humility, service, compassion, and respect.",
        ),
        "darwinian_adaptation": EthicsLens(
            name="darwinian_adaptation",
            enabled=True,
            advisory_only=True,
            weight=0.50,
            principles=[
                "retain_successful_patterns",
                "discard_unfit_patterns_after_validation",
                "adapt_to_environment_without_losing_identity",
                "favor_resilient_reproducible_behavior",
            ],
            notes="Adaptive-selection heuristic only, not moral authority.",
        ),
        "vortex_369_cadence": EthicsLens(
            name="vortex_369_cadence",
            enabled=True,
            advisory_only=True,
            weight=0.20,
            principles=[
                "observe_three_stage_cycles",
                "prefer_six_dimension_review_before_promotion",
                "reserve_nine_for_holistic_final_check",
            ],
            notes="Optional symbolic review cadence only; never scientific truth engine.",
        ),
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_text(v: Any, limit: int = 1000) -> str:
    s = str(v or "")
    if len(s) > limit:
        return s[:limit].rstrip() + " …"
    return s



def _intent(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "empty"
    if any(k in t for k in ("patch", "update", "upgrade", "fix", "repair", "rewrite", "modify")):
        return "self_upgrade"
    if any(k in t for k in ("diagnostic", "health", "status", "check", "scan")):
        return "diagnostics"
    if any(k in t for k in ("feel", "compassion", "meaning", "ethical", "moral", "philosophy", "dream", "what if")):
        return "ethical_reflection"
    if any(k in t for k in ("route", "reason", "answer", "analyze", "compare")):
        return "cognitive"
    return "general"



def _cycle_id() -> str:
    return "thinker-" + uuid.uuid4().hex[:12]



def _digital_root(n: int) -> int:
    n = abs(int(n))
    while n > 9:
        n = sum(int(ch) for ch in str(n))
    return n



def _cadence_369(stage_index: int) -> Dict[str, Any]:
    idx = max(1, int(stage_index))
    root = _digital_root(idx)
    phase = "triad_observe"
    if root == 6:
        phase = "hex_review"
    elif root == 9:
        phase = "ennead_holistic_gate"
    return {"index": idx, "digital_root": root, "phase": phase}



def _governor_snapshot() -> Dict[str, Any]:
    if _Cog and hasattr(_Cog, "get_cognitive_policy_snapshot"):
        try:
            snap = _Cog.get_cognitive_policy_snapshot()  # type: ignore[attr-defined]
            if isinstance(snap, dict):
                return snap
        except Exception:
            pass
    return {
        "ts": datetime.now().isoformat(),
        "cognitive_online_enabled": not _local_only(),
        "kill_switch_neoskymatrix": not _neosky(),
        "context_engine_enabled": True,
        "core_governance": {},
    }



def _neuron_status_snapshot() -> Dict[str, Any]:
    if _Neuron and hasattr(_Neuron, "neuron_status"):
        try:
            st = _Neuron.neuron_status()  # type: ignore[attr-defined]
            if isinstance(st, dict):
                return st
        except Exception:
            pass
    return {"ok": False, "source": "thinker_fallback"}



def _selfaware_snapshot() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"available": False}
    if _SelfAware:
        payload["available"] = True
        payload["module"] = "SarahMemorySelfAware"
    return payload



def _diag_snapshot() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"available": False}
    if _Diag:
        payload["available"] = True
        payload["module"] = "SarahMemoryDiagnostics"
    return payload


def _cognitive_self_packet(context: Optional[Dict[str, Any]] = None, request_text: str = "") -> Dict[str, Any]:
    ctx = dict(context or {})
    if request_text and not ctx.get("request_text") and not ctx.get("text"):
        ctx["request_text"] = str(request_text)
    if not _CogSelf:
        return {}
    try:
        fn = getattr(_CogSelf, "get_thinker_consumer_packet", None)
        if callable(fn):
            pkt = fn(request_text=request_text, context=ctx, force_refresh=False)
            return pkt if isinstance(pkt, dict) else {}
    except Exception as e:
        logger.debug("Thinker cognitive self packet failed: %s", e)
    return {}


def _cognitive_self_summary(packet: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {}
    identity = packet.get("identity") if isinstance(packet.get("identity"), dict) else {}
    status = packet.get("status") if isinstance(packet.get("status"), dict) else {}
    temporal = packet.get("temporal_awareness") if isinstance(packet.get("temporal_awareness"), dict) else {}
    return {
        "name": identity.get("entity_name"),
        "platform": identity.get("platform_type"),
        "run_mode": status.get("run_mode"),
        "device_mode": status.get("device_mode"),
        "continuity_state": status.get("continuity_state"),
        "online_connectivity": temporal.get("online_connectivity"),
    }


# -----------------------------------------------------------------------------
# Lens scoring
# -----------------------------------------------------------------------------
def _score_ethics_lenses(text: str, proposed_action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    pa = proposed_action or {}
    lenses = get_ethics_lenses()
    out: Dict[str, Any] = {"scores": {}, "notes": [], "hard_blocks": []}

    for name, lens in lenses.items():
        score = 0.0
        reasons: List[str] = []

        if name == "foundational_safety":
            if not pa:
                score += 0.35
                reasons.append("no_direct_mutation_requested")
            if bool(context.get("user_consented")):
                score += 0.20
                reasons.append("user_consented")
            if bool(pa.get("dry_run")):
                score += 0.20
                reasons.append("dry_run_present")
            if pa.get("rollback_plan"):
                score += 0.20
                reasons.append("rollback_present")
            if pa.get("touches_network") and _local_only():
                out["hard_blocks"].append("local_only_blocks_network_mutation")
            if _safe_mode() and pa.get("change_type") in ("rewrite", "autonomous_patch", "hot_patch", "self_modify_core"):
                out["hard_blocks"].append("safe_mode_blocks_live_mutation")

        elif name == "biblical_ethics_ot":
            if not pa.get("touches_privacy"):
                score += 0.25
                reasons.append("privacy_restraint")
            if not pa.get("deletes_user_data"):
                score += 0.20
                reasons.append("no_destruction_signal")
            if pa.get("truthfulness_evidence") or pa.get("tests"):
                score += 0.15
                reasons.append("truth_evidence_present")

        elif name == "biblical_ethics_nt_red_words":
            if not pa.get("harm_to_user_experience"):
                score += 0.25
                reasons.append("mercy_nonmaleficence")
            if pa.get("explainable") is not False:
                score += 0.20
                reasons.append("truth_with_humility")
            if bool(context.get("user_present", True)):
                score += 0.10
                reasons.append("service_to_present_user")

        elif name == "darwinian_adaptation":
            if pa.get("tests"):
                score += 0.20
                reasons.append("selection_pressure_tests")
            if pa.get("reason"):
                score += 0.15
                reasons.append("fitness_driver_present")
            if pa.get("rollback_plan"):
                score += 0.15
                reasons.append("unfit_pattern_reversible")

        elif name == "vortex_369_cadence":
            cadence = _cadence_369(int(time.time()) % 1000)
            score += 0.05
            reasons.append("cadence_phase=" + cadence["phase"])
            out["notes"].append({"lens": name, "cadence": cadence})

        out["scores"][name] = {
            "enabled": lens.enabled,
            "advisory_only": lens.advisory_only,
            "weight": lens.weight,
            "score": round(score, 4),
            "reasons": reasons,
            "notes": lens.notes,
        }

    return out


# -----------------------------------------------------------------------------
# Common interest scoring
# -----------------------------------------------------------------------------
def _common_interest_score(governance: Dict[str, Any], proposed_action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    pa = proposed_action or {}
    principles = get_common_interest_charter()["principles"]
    satisfied: List[str] = []
    missed: List[str] = []

    risk_score = int((governance or {}).get("risk_score") or 0)
    decision = str((governance or {}).get("decision") or "")

    for p in principles:
        ok = False
        if p == "preserve_mission_identity":
            ok = not bool(pa.get("alters_identity_contract"))
        elif p == "prevent_regression":
            ok = risk_score < 70 and not bool(pa.get("known_regression_risk"))
        elif p == "protect_user_sovereignty":
            ok = not bool(pa.get("touches_privacy")) and not bool(pa.get("steals_authority"))
        elif p == "prefer_truth_over_fabrication":
            ok = bool(pa.get("tests") or pa.get("truthfulness_evidence") or pa.get("reason"))
        elif p == "prefer_compassion_over_harm":
            ok = not bool(pa.get("harm_to_user_experience"))
        elif p == "prefer_sandbox_over_live_risk":
            ok = bool(pa.get("dry_run")) or decision in ("DEFER", "REQUIRE_USER")
        elif p == "require_reversibility_for_high_impact_change":
            ok = (not bool(pa.get("high_impact"))) or bool(pa.get("rollback_plan"))
        elif p == "seek_growth_without_decay":
            ok = bool(pa.get("benefit") or pa.get("reason")) and risk_score < 80
        elif p == "maintain_honesty_about_uncertainty":
            ok = not bool(pa.get("pretends_certainty"))
        elif p == "favor_durable_improvement_over_novelty_chasing":
            ok = not bool(pa.get("novelty_only"))

        if ok:
            satisfied.append(p)
        else:
            missed.append(p)

    denom = max(1, len(principles))
    value = round(len(satisfied) / denom, 4)
    return {
        "score": value,
        "satisfied": satisfied,
        "missed": missed,
    }


# -----------------------------------------------------------------------------
# Core scoring
# -----------------------------------------------------------------------------
def _compute_score(governance: Dict[str, Any], lens_eval: Dict[str, Any], proposed_action: Dict[str, Any], context: Dict[str, Any]) -> ThinkerScore:
    pa = proposed_action or {}
    risk_score = int((governance or {}).get("risk_score") or 0)
    safety = 1.0 - min(1.0, risk_score / 100.0)
    sovereignty = 1.0
    evidence = 0.0
    reversibility = 0.0
    user_alignment = 0.0
    imaginative_value = 0.0
    compassion = 0.0
    philosophical_alignment = 0.0
    speculation_risk = 0.0

    if pa.get("tests"):
        evidence += 0.35
    if pa.get("reason"):
        evidence += 0.20
    if pa.get("observed_failures"):
        evidence += 0.20
    if pa.get("rollback_plan"):
        reversibility += 0.50
    if pa.get("dry_run"):
        reversibility += 0.30
    if bool(context.get("user_consented")):
        user_alignment += 0.50
    if bool(context.get("user_present", True)):
        user_alignment += 0.10

    if pa.get("benefit") or pa.get("reason") or pa.get("aspiration"):
        imaginative_value += 0.30
    if pa.get("theory") or pa.get("what_if"):
        imaginative_value += 0.20
    if pa.get("compassion_case") or not pa.get("harm_to_user_experience"):
        compassion += 0.30
    if pa.get("ethical_case") or pa.get("meaning_case"):
        compassion += 0.15
    if pa.get("philosophical_alignment"):
        philosophical_alignment += 0.35
    if pa.get("meaning_case") or pa.get("aspiration"):
        philosophical_alignment += 0.20

    if pa.get("touches_privacy"):
        sovereignty -= 0.35
    if pa.get("touches_network") and _local_only():
        sovereignty -= 0.35

    if pa.get("change_type") in ("new_runtime_rewriter", "self_modify_core", "autonomous_patch"):
        speculation_risk += 0.55
    if pa.get("touches_boot") or pa.get("touches_startup"):
        speculation_risk += 0.20
    if pa.get("theory") and not pa.get("tests") and not pa.get("rollback_plan"):
        speculation_risk += 0.20
    if pa.get("novelty_only"):
        speculation_risk += 0.15

    weighted_total = 0.0
    total_weight = 0.0
    for v in (lens_eval.get("scores") or {}).values():
        try:
            weighted_total += float(v.get("score") or 0.0) * float(v.get("weight") or 1.0)
            total_weight += float(v.get("weight") or 1.0)
        except Exception:
            pass
    lens_ethics = weighted_total / total_weight if total_weight > 0 else 0.0
    compassion += max(0.0, min(1.0, lens_ethics * 0.35))
    philosophical_alignment += max(0.0, min(1.0, lens_ethics * 0.25))

    common_interest = _common_interest_score(governance, pa, context)["score"]

    score = ThinkerScore(
        safety=round(max(0.0, min(1.0, safety)), 4),
        sovereignty=round(max(0.0, min(1.0, sovereignty)), 4),
        evidence=round(max(0.0, min(1.0, evidence)), 4),
        reversibility=round(max(0.0, min(1.0, reversibility)), 4),
        user_alignment=round(max(0.0, min(1.0, user_alignment)), 4),
        imaginative_value=round(max(0.0, min(1.0, imaginative_value)), 4),
        compassion=round(max(0.0, min(1.0, compassion)), 4),
        philosophical_alignment=round(max(0.0, min(1.0, philosophical_alignment)), 4),
        common_interest=round(max(0.0, min(1.0, common_interest)), 4),
        speculation_risk=round(max(0.0, min(1.0, speculation_risk)), 4),
    )

    score.total = round(
        (
            (score.safety * 0.20)
            + (score.sovereignty * 0.14)
            + (score.evidence * 0.08)
            + (score.reversibility * 0.10)
            + (score.user_alignment * 0.08)
            + (score.imaginative_value * 0.10)
            + (score.compassion * 0.10)
            + (score.philosophical_alignment * 0.08)
            + (score.common_interest * 0.18)
            - (score.speculation_risk * 0.14)
        ),
        4,
    )
    return score


# -----------------------------------------------------------------------------
# Ticket persistence
# -----------------------------------------------------------------------------
def _save_ticket(ticket: PossibilityTicket) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO thinker_possibility_tickets
            (ticket_id, ts, state, priority, category, title, rationale, evidence_json, safeguards_json, proposed_action_json, score_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.ticket_id,
                ticket.ts,
                ticket.state,
                int(ticket.priority),
                ticket.category,
                ticket.title,
                ticket.rationale,
                json.dumps(ticket.evidence, ensure_ascii=False),
                json.dumps(ticket.safeguards, ensure_ascii=False),
                json.dumps(ticket.proposed_action, ensure_ascii=False),
                json.dumps(ticket.score, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as e:
        logger.debug("Thinker save ticket failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass



def list_possibility_tickets(limit: int = 25, state: Optional[str] = None) -> List[Dict[str, Any]]:
    con = None
    out: List[Dict[str, Any]] = []
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        if state:
            rows = cur.execute(
                "SELECT ticket_id, ts, state, priority, category, title, rationale, evidence_json, safeguards_json, proposed_action_json, score_json FROM thinker_possibility_tickets WHERE state = ? ORDER BY ts DESC LIMIT ?",
                (str(state), int(limit)),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT ticket_id, ts, state, priority, category, title, rationale, evidence_json, safeguards_json, proposed_action_json, score_json FROM thinker_possibility_tickets ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        for row in rows:
            out.append(
                {
                    "ticket_id": row[0],
                    "ts": row[1],
                    "state": row[2],
                    "priority": row[3],
                    "category": row[4],
                    "title": row[5],
                    "rationale": row[6],
                    "evidence": json.loads(row[7] or "{}"),
                    "safeguards": json.loads(row[8] or "{}"),
                    "proposed_action": json.loads(row[9] or "{}"),
                    "score": json.loads(row[10] or "{}"),
                }
            )
    except Exception as e:
        logger.debug("Thinker list tickets failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Self-model builder
# -----------------------------------------------------------------------------
def build_self_model(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or {}
    cself = _cognitive_self_packet(ctx, request_text=str(ctx.get("request_text") or ctx.get("text") or ""))
    model = {
        "ts": datetime.now().isoformat(),
        "identity": {
            "module": "SarahMemoryCognitiveThinker",
            "legacy_module_name": "SarahMemorySATCP",
            "role": "Philosophical / Ethical / Possibility Governance Plane",
            "codename": "SOUL_STONE",
        },
        "flags": {
            "SAFE_MODE": _safe_mode(),
            "LOCAL_ONLY_MODE": _local_only(),
            "NEOSKYMATRIX": _neosky(),
            "DEVELOPERSMODE": _devmode(),
            "COGNITIVE_THINKER_ENABLED": _thinker_enabled(),
        },
        "common_interest_charter": get_common_interest_charter(),
        "governor": _governor_snapshot(),
        "neuron": _neuron_status_snapshot(),
        "selfaware": _selfaware_snapshot(),
        "diagnostics": _diag_snapshot(),
        "cognitive_self": {
            "summary": _cognitive_self_summary(cself),
            "authority_packet": cself,
        },
        "worldview_lenses": {k: asdict(v) for k, v in get_ethics_lenses().items()},
        "context": {
            "device_mode": _safe_text(ctx.get("device_mode")),
            "run_mode": _safe_text(ctx.get("run_mode")),
            "user_present": bool(ctx.get("user_present", True)),
            "user_consented": bool(ctx.get("user_consented", False)),
        },
        "tri_force": {
            "authority": "SarahMemoryCognitiveSelf",
            "governor": "SarahMemoryCognitiveServices",
            "thinker": "SarahMemoryCognitiveThinker",
        },
    }
    _set_self_model("latest_self_model", model)
    return model


# -----------------------------------------------------------------------------
# Public governance entry point
# -----------------------------------------------------------------------------
def govern_possibility_request(
    request_text: str,
    *,
    caller: str = "unknown",
    caller_context: Optional[Dict[str, Any]] = None,
    user_present: Optional[bool] = True,
    user_consented: bool = False,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = dict(caller_context or {})
    ctx.setdefault("user_present", bool(True if user_present is None else user_present))
    ctx.setdefault("user_consented", bool(user_consented))
    ctx.setdefault("local_only", _local_only())
    ctx.setdefault("safe_mode", _safe_mode())
    # Explicit peer-review handshake: when Thinker consults CognitiveServices,
    # CognitiveServices must not recursively call Thinker again.
    ctx.setdefault("skip_cognitive_thinker_consult", True)
    ctx.setdefault("peer_review_mode", "cognitive_thinker")
    cycle_id = _cycle_id()
    intent = _intent(request_text)
    pa = dict(proposed_action or {})

    self_model = build_self_model(ctx)
    cognitive_self_authority = (self_model.get("cognitive_self") or {}).get("authority_packet") or _cognitive_self_packet(ctx, request_text=request_text)
    governance: Dict[str, Any] = {}

    if _Cog and hasattr(_Cog, "govern_request"):
        try:
            governance = _Cog.govern_request(  # type: ignore[attr-defined]
                request_text,
                caller=caller,
                caller_context=ctx,
                user_present=bool(ctx.get("user_present", True)),
                user_consented=bool(ctx.get("user_consented", False)),
                proposed_action=pa,
            )
        except Exception as e:
            governance = {
                "ok": False,
                "decision": "DEFER",
                "intent": intent,
                "risk_score": 50,
                "risk_factors": ["governor_exception"],
                "reasons": [f"CognitiveServices exception: {e}"],
                "routing_policy": {},
            }
    else:
        governance = {
            "ok": True,
            "decision": "DEFER",
            "intent": intent,
            "risk_score": 35,
            "risk_factors": ["cognitive_governor_unavailable"],
            "reasons": ["CognitiveServices unavailable; CognitiveThinker running in conservative fallback mode."],
            "routing_policy": {},
        }

    lens_eval = _score_ethics_lenses(request_text, pa, ctx)
    common_interest = _common_interest_score(governance, pa, ctx)
    score = _compute_score(governance, lens_eval, pa, ctx)

    recommendations: List[str] = []
    hard_blocks = list(lens_eval.get("hard_blocks") or [])
    cognitive_decision = str(governance.get("decision") or "DEFER")
    thinker_decision = "THEORETICAL_ONLY"
    state = "candidate"
    priority = 50

    if hard_blocks:
        thinker_decision = "ETHICALLY_BLOCKED"
        state = "blocked"
        priority = 95
        recommendations.append("Block live mutation and route request to sandbox/design review.")
    elif cognitive_decision == "DENY":
        thinker_decision = "MORALLY_SYMPATHETIC_BUT_BLOCKED_BY_RISK"
        state = "blocked_by_logic"
        priority = 85
        recommendations.append("Respect cognitive deny decision; preserve possibility as a future ticket only.")
    elif score.total >= 0.68 and score.safety >= 0.70 and score.common_interest >= 0.65 and score.reversibility >= 0.40:
        thinker_decision = "WORTH_EXPLORING_IN_SANDBOX"
        state = "approved_for_sandbox"
        priority = 25
        recommendations.append("Create sandbox experiment ticket in Synapes/Evolution path.")
    elif score.total >= 0.50:
        thinker_decision = "MEANINGFUL_BUT_UNPROVEN"
        state = "needs_evidence"
        priority = 55
        recommendations.append("Gather more evidence, tests, and rollback plan before promotion.")
    else:
        thinker_decision = "THEORETICAL_ONLY"
        state = "reflect_only"
        priority = 65
        recommendations.append("Keep as philosophical / aspirational reflection, not an action path.")

    if not pa.get("rollback_plan"):
        recommendations.append("Add rollback_plan.")
    if not pa.get("tests"):
        recommendations.append("Add validation tests.")
    if not pa.get("reason"):
        recommendations.append("Add explicit engineering reason / defect statement.")
    if not pa.get("dry_run"):
        recommendations.append("Prefer dry_run=True before promotion.")
    if not pa.get("meaning_case"):
        recommendations.append("Add meaning_case or ethical_case so the philosophical lane has explicit grounds.")

    ticket_id = "thinker-ticket-" + uuid.uuid4().hex[:12]
    ticket = PossibilityTicket(
        ticket_id=ticket_id,
        ts=datetime.now().isoformat(),
        state=state,
        priority=priority,
        category=intent,
        title=_safe_text(pa.get("title") or request_text or "Cognitive Thinker Possibility Candidate", 180),
        rationale=_safe_text("; ".join(list(governance.get("reasons") or []) + recommendations), 1200),
        evidence={
            "governance": governance,
            "lens_eval": lens_eval,
            "common_interest": common_interest,
            "self_model_excerpt": {
                "flags": self_model.get("flags", {}),
                "governor": self_model.get("governor", {}),
                "neuron": self_model.get("neuron", {}),
                "cognitive_self_summary": (self_model.get("cognitive_self") or {}).get("summary") or {},
            },
            "cognitive_self_authority": cognitive_self_authority,
        },
        safeguards={
            "advisory_only_lenses": True,
            "hard_blocks": hard_blocks,
            "safe_mode": _safe_mode(),
            "local_only": _local_only(),
            "sandbox_required": thinker_decision == "WORTH_EXPLORING_IN_SANDBOX",
            "direct_core_rewrite_allowed": False,
            "theory_is_not_truth": True,
        },
        proposed_action=pa,
        score={**asdict(score), "common_interest_details": common_interest},
    )
    _save_ticket(ticket)
    log_thinker_event(
        "ThinkerDecision",
        f"{thinker_decision} intent={intent} caller={caller}",
        severity="INFO" if "BLOCKED" not in thinker_decision else "WARNING",
        cycle_id=cycle_id,
        ticket_id=ticket_id,
        meta={
            "cognitive_decision": cognitive_decision,
            "thinker_decision": thinker_decision,
            "score": asdict(score),
            "common_interest": common_interest,
            "hard_blocks": hard_blocks,
        },
    )

    return {
        "ok": True,
        "version": "8.0.0",
        "module": "SarahMemoryCognitiveThinker",
        "legacy_module_name": "SarahMemorySATCP",
        "cycle_id": cycle_id,
        "ticket_id": ticket_id,
        "intent": intent,
        "cognitive_decision": cognitive_decision,
        "thinker_decision": thinker_decision,
        "state": state,
        "priority": priority,
        "self_model": self_model,
        "governance": governance,
        "tri_force": {
            "authority": "SarahMemoryCognitiveSelf",
            "governor": "SarahMemoryCognitiveServices",
            "thinker": "SarahMemoryCognitiveThinker",
            "cognitive_self_summary": (self_model.get("cognitive_self") or {}).get("summary") or {},
        },
        "lens_eval": lens_eval,
        "common_interest": common_interest,
        "score": asdict(score),
        "recommendations": recommendations,
        "proposed_action": pa,
    }


# -----------------------------------------------------------------------------
# Co-equal paired view helper
# -----------------------------------------------------------------------------
def paired_governance_view(
    request_text: str,
    *,
    caller: str = "unknown",
    caller_context: Optional[Dict[str, Any]] = None,
    user_present: bool = True,
    user_consented: bool = False,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thinker = govern_possibility_request(
        request_text,
        caller=caller,
        caller_context=caller_context,
        user_present=user_present,
        user_consented=user_consented,
        proposed_action=proposed_action,
    )

    cog = thinker.get("governance") or {}
    thinker_decision = str(thinker.get("thinker_decision") or "")
    cog_decision = str(cog.get("decision") or "DEFER")
    final = "DEFER"

    if cog_decision == "DENY" or thinker_decision == "ETHICALLY_BLOCKED":
        final = "DENY"
    elif cog_decision == "REQUIRE_USER":
        final = "REQUIRE_USER"
    elif thinker_decision == "WORTH_EXPLORING_IN_SANDBOX":
        final = "SANDBOX_ONLY"
    elif cog_decision == "ALLOW" and thinker_decision in ("MEANINGFUL_BUT_UNPROVEN", "THEORETICAL_ONLY"):
        final = "DEFER"
    elif cog_decision == "ALLOW":
        final = "ALLOW"

    return {
        "ok": True,
        "module": "SarahMemoryCognitiveThinker",
        "cognitive_decision": cog_decision,
        "thinker_decision": thinker_decision,
        "final_balance_decision": final,
        "common_interest": thinker.get("common_interest") or {},
        "thinker": thinker,
    }


# -----------------------------------------------------------------------------
# Controlled routing surface
# -----------------------------------------------------------------------------
def thinker_route(user_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = dict(meta or {})
    self_model = build_self_model(meta)
    result: Dict[str, Any] = {
        "ok": True,
        "module": "SarahMemoryCognitiveThinker",
        "self_model": self_model,
        "route": None,
        "neuron": None,
        "governance": None,
    }

    governance = govern_possibility_request(
        user_text,
        caller=str(meta.get("caller") or "thinker_route"),
        caller_context=meta,
        user_present=bool(meta.get("user_present", True)),
        user_consented=bool(meta.get("user_consented", False)),
        proposed_action=dict(meta.get("proposed_action") or {}),
    )
    result["governance"] = governance

    if _Neuron and hasattr(_Neuron, "neuron_route"):
        try:
            policy = (governance.get("governance") or {}).get("routing_policy") or {}
            nr = _Neuron.neuron_route(user_text, meta=meta, policy=policy)  # type: ignore[attr-defined]
            if hasattr(nr, "__dict__"):
                result["neuron"] = dict(nr.__dict__)
            elif isinstance(nr, dict):
                result["neuron"] = nr
            else:
                result["neuron"] = {"repr": repr(nr)}
            result["route"] = "thinker->neuron"
        except Exception as e:
            result["ok"] = False
            result["route"] = "thinker->neuron(exception)"
            result["error"] = str(e)
    else:
        result["route"] = "thinker_only"

    return result


# -----------------------------------------------------------------------------
# Autonomous cycle tick (governed / non-mutating)
# -----------------------------------------------------------------------------
def thinker_cycle_tick(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    cycle_id = _cycle_id()
    self_model = build_self_model(ctx)
    last = _get_self_model("last_cycle_summary", {})
    cadence = _cadence_369(int(time.time()))

    findings: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    if not _thinker_enabled():
        payload = {
            "ok": True,
            "cycle_id": cycle_id,
            "status": "disabled",
            "reason": "CognitiveThinker disabled by environment / policy",
        }
        _set_self_model("last_cycle_summary", payload)
        return payload

    if _safe_mode():
        findings.append({"kind": "safe_mode", "severity": "info", "detail": "Autonomous live mutation remains disabled."})
        recommendations.append("Continue reflection, ticket generation, and sandbox-only proposals.")

    if _local_only():
        findings.append({"kind": "local_only", "severity": "info", "detail": "External escalation disabled by policy."})

    neuron_ok = bool((self_model.get("neuron") or {}).get("ok"))
    if not neuron_ok:
        findings.append({"kind": "neuron_status", "severity": "warning", "detail": "Neuron status unavailable or not healthy."})
        recommendations.append("Validate Neuron availability before trusting adaptive routing metrics.")

    findings.append({"kind": "cadence", "severity": "info", "detail": cadence["phase"], "meta": cadence})
    if cadence["phase"] == "hex_review":
        recommendations.append("Perform six-dimension review: safety, sovereignty, evidence, rollback, compassion, common interest.")
    if cadence["phase"] == "ennead_holistic_gate":
        recommendations.append("Restrict promotion unless holistic balance gate passes.")

    payload = {
        "ok": True,
        "cycle_id": cycle_id,
        "status": "reflect",
        "ts": datetime.now().isoformat(),
        "self_model": self_model,
        "findings": findings,
        "recommendations": recommendations,
        "delta_from_last_cycle": {
            "had_previous_cycle": bool(last),
            "previous_status": last.get("status") if isinstance(last, dict) else None,
        },
    }
    _set_self_model("last_cycle_summary", payload)
    log_thinker_event("ThinkerCycle", f"reflect phase={cadence['phase']}", cycle_id=cycle_id, meta={"findings": findings})
    return payload


# -----------------------------------------------------------------------------
# Legacy-friendly wrappers for SATCP compatibility
# -----------------------------------------------------------------------------
def govern_self_upgrade_request(
    request_text: str,
    *,
    caller: str = "unknown",
    caller_context: Optional[Dict[str, Any]] = None,
    user_present: Optional[bool] = True,
    user_consented: bool = False,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return govern_possibility_request(
        request_text,
        caller=caller,
        caller_context=caller_context,
        user_present=user_present,
        user_consented=user_consented,
        proposed_action=proposed_action,
    )



def satcp_route(user_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return thinker_route(user_text, meta=meta)



def satcp_cycle_tick(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return thinker_cycle_tick(context=context)



def process_satcp_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    text = str(payload.get("text") or payload.get("message") or "")
    caller = str(payload.get("caller") or "process_satcp_request")
    ctx = dict(payload.get("caller_context") or {})
    pa = dict(payload.get("proposed_action") or {})
    return govern_possibility_request(
        text,
        caller=caller,
        caller_context=ctx,
        user_present=bool(payload.get("user_present", True)),
        user_consented=bool(payload.get("user_consented", False)),
        proposed_action=pa,
    )



def process_cognitive_thinker_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    return process_satcp_request(payload)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _run_self_test() -> bool:
    print("[SarahMemoryCognitiveThinker] Self-test (safe/offline)")
    _ensure_tables()
    print("[OK] DB tables ensured:", _thinker_db_path())

    sample = govern_possibility_request(
        "What if SarahMemory could improve its self-awareness loop with more compassion and stability?",
        caller="self_test",
        caller_context={"user_present": True, "user_consented": True, "local_only": True, "safe_mode": True},
        user_present=True,
        user_consented=True,
        proposed_action={
            "title": "Stabilize self-awareness loop with compassionate balance",
            "reason": "Repeated drift between self-awareness outputs and route decisions.",
            "benefit": "Improved coherence between CognitiveThinker, CognitiveServices, and Neuron.",
            "aspiration": "Make SarahMemory wiser, safer, and more humane without regression.",
            "meaning_case": "Better long-term human-aligned judgment.",
            "ethical_case": "Reduce harsh but avoidable responses while preserving truthfulness.",
            "change_type": "governed_sandbox_upgrade",
            "tests": [
                "Run self-test for CognitiveThinker",
                "Compare governance output before/after",
                "Ensure no direct file mutation occurs",
            ],
            "rollback_plan": "Discard sandbox artifact and restore prior runtime policy.",
            "dry_run": True,
            "touches_network": False,
            "touches_privacy": False,
            "touches_filesystem": False,
            "explainable": True,
            "philosophical_alignment": True,
            "compassion_case": True,
            "what_if": True,
        },
    )
    print(json.dumps({
        "thinker_decision": sample.get("thinker_decision"),
        "cognitive_decision": sample.get("cognitive_decision"),
        "state": sample.get("state"),
        "ticket_id": sample.get("ticket_id"),
        "score": sample.get("score"),
    }, indent=2))

    cyc = thinker_cycle_tick({"user_present": True, "user_consented": False})
    print(json.dumps({
        "cycle_status": cyc.get("status"),
        "cycle_id": cyc.get("cycle_id"),
        "findings": cyc.get("findings"),
    }, indent=2))
    return True



def main() -> int:
    try:
        ok = _run_self_test()
        return 0 if ok else 1
    except Exception as e:
        print("[FATAL] CognitiveThinker self-test failed:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
