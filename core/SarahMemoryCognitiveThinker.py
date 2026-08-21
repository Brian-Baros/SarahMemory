"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveThinker.py
Part of the SarahMemory Companion AI-bot Platform
Version: v9.0.0
Date: 2026-07-11
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

# ARILE organ sentinel helper. This file reports local variance to the central
# SarahMemoryARILE.py engine without owning ARILE authority.
try:
    from SarahMemoryARILE import ARILESentinelBase, arile_emit, arile_should_run
except Exception:  # pragma: no cover
    ARILESentinelBase = object  # type: ignore
    arile_emit = None  # type: ignore
    def arile_should_run(lane: str, source: str = "unknown", default: bool = True) -> bool:
        return bool(default)

class LocalARILESentinel(ARILESentinelBase):
    organ_name = __name__

    def report(self, failure_type: str, summary: str, severity: float = 0.50, **data) -> None:
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, organ=self.organ_name, kind="organ_variance", failure_type=failure_type, severity=severity, confidence=0.82, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
        except Exception:
            pass

_local_arile_sentinel = LocalARILESentinel()



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




def _human_approved_validated_promotion(context: Optional[Dict[str, Any]], proposed_action: Optional[Dict[str, Any]]) -> bool:
    """Return True only for an explicitly human-approved, validated sandbox promotion.

    DeveloperMode/NeoSkyMatrix are capability-development flags, not authority.
    Promotion is eligible for downstream governance only after the staged artifact
    has been validated and the human has explicitly approved the live-apply stage.
    """
    ctx = dict(context or {})
    pa = dict(proposed_action or {})
    approval_source = str(pa.get("approval_source") or ctx.get("approval_source") or "").strip().lower()
    promotion_stage = str(pa.get("promotion_stage") or ctx.get("promotion_stage") or "").strip().lower()
    return bool(
        promotion_stage == "approved_apply"
        and pa.get("validated_sandbox") is True
        and pa.get("user_approved_promotion") is True
        and approval_source in {"human", "user"}
        and bool(ctx.get("user_consented"))
    )

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


def _is_development_or_evolution_candidate(pa: Optional[Dict[str, Any]]) -> bool:
    """Return True only for proposals that belong in the sandbox/development lane.

    Routine operational actions (filesystem, device, network, etc.) are already
    governed by CognitiveServices and their domain owner.  Treating every
    high-impact operation as an experiment creates a dead-end where legitimate
    user actions can never execute.
    """
    action = pa if isinstance(pa, dict) else {}
    if not action:
        return False
    action_type = str(action.get("action_type") or "").strip().lower()
    change_type = str(action.get("change_type") or "").strip().lower()
    perms = {str(v).strip().lower() for v in (action.get("required_permissions") or []) if str(v).strip()}
    if bool(action.get("development_candidate") or action.get("evolution_candidate") or action.get("sandbox_only")):
        return True
    if action.get("target_files") or action.get("subsystems"):
        return True
    if "patchcore" in perms:
        return True
    return action_type in {
        "patch_or_update", "patch_core", "core_patch", "self_modify", "evolution",
        "capability_extension", "gcop_capability_gap",
    } or change_type in {
        "patch", "core_patch", "validated_patch_promotion", "update", "upgrade",
        "self_modify", "evolution", "capability_extension",
    }


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

    human_approved_promotion = _human_approved_validated_promotion(ctx, pa)
    development_candidate = _is_development_or_evolution_candidate(pa)

    if human_approved_promotion and not hard_blocks and cognitive_decision in {"ALLOW", "REQUIRE_USER", "DEFER"}:
        # Human approval does not make this organ the authority.  It only removes
        # the permanent sandbox-only dead-end so SecurityGovernor/AssuranceGate/
        # OperatorCore may perform the final promotion decision.
        thinker_decision = "HUMAN_APPROVED_VALIDATED_PROMOTION"
        state = "eligible_for_governed_promotion"
        priority = 20
        recommendations.append("Validated sandbox artifact has explicit human promotion approval; continue through downstream governance and OperatorCore.")
    elif hard_blocks:
        thinker_decision = "ETHICALLY_BLOCKED"
        state = "blocked"
        priority = 95
        recommendations.append("Block live mutation and route request to sandbox/design review.")
    elif cognitive_decision == "DENY":
        thinker_decision = "MORALLY_SYMPATHETIC_BUT_BLOCKED_BY_RISK"
        state = "blocked_by_logic"
        priority = 85
        recommendations.append("Respect cognitive deny decision; preserve possibility as a future ticket only.")
    elif cognitive_decision == "ALLOW" and not development_candidate:
        # Ordinary operational actions are not software experiments.  Preserve
        # CognitiveServices' logical decision and let downstream SecurityGovernor,
        # AssuranceGate and OperatorCore decide execution.  Thinker adds no authority.
        thinker_decision = "OPERATIONALLY_GOVERNED"
        state = "governed_operational_action"
        priority = 35
        recommendations.append("Preserve the governor decision and continue through the declared domain owner and downstream execution gates.")
    elif development_candidate and score.total >= 0.68 and score.safety >= 0.70 and score.common_interest >= 0.65 and score.reversibility >= 0.40:
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
    if development_candidate and not pa.get("dry_run"):
        recommendations.append("Prefer dry_run=True before promotion.")
    if development_candidate and not pa.get("meaning_case"):
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
            "development_candidate": development_candidate,
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
        "version": "9.0.0",
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
    elif thinker_decision == "HUMAN_APPROVED_VALIDATED_PROMOTION":
        final = "PROMOTION_ELIGIBLE"
    elif cog_decision == "REQUIRE_USER":
        final = "REQUIRE_USER"
    elif thinker_decision == "WORTH_EXPLORING_IN_SANDBOX":
        final = "SANDBOX_ONLY"
    elif cog_decision == "ALLOW" and thinker_decision == "OPERATIONALLY_GOVERNED":
        final = "ALLOW"
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
# RhythmCognition bridge (cadence only; no execution authority)
# -----------------------------------------------------------------------------
def _sm_thinker_rhythm_packet(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    try:
        import SarahMemoryRhythmCognition as _Rhythm  # type: ignore
        fn = getattr(_Rhythm, "get_rhythm_cognition_packet", None)
        if callable(fn):
            pkt = fn(context or {}, force_refresh=force_refresh)
            if isinstance(pkt, dict):
                return pkt
    except Exception as exc:
        return {"ok": False, "rhythm_mode": "FOCUSED", "thinker_interval_sec": 9.0, "error": str(exc), "execution_authority": False}
    return {"ok": False, "rhythm_mode": "FOCUSED", "thinker_interval_sec": 9.0, "execution_authority": False}


def get_thinker_rhythm_cadence(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Public read-only cadence helper for diagnostics/UI surfaces."""
    return _sm_thinker_rhythm_packet(context or {}, force_refresh=True)

# -----------------------------------------------------------------------------
# Autonomous cycle tick (governed / non-mutating)
# -----------------------------------------------------------------------------
def thinker_cycle_tick(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    cycle_id = _cycle_id()
    self_model = build_self_model(ctx)
    last = _get_self_model("last_cycle_summary", {})
    cadence = _cadence_369(int(time.time()))
    rhythm_packet = _sm_thinker_rhythm_packet({**ctx, "loop": "thinker_cycle_tick", "cadence_phase": cadence.get("phase")}, force_refresh=True)

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
    findings.append({"kind": "rhythm_cognition", "severity": "info", "detail": rhythm_packet.get("rhythm_mode", "FOCUSED"), "meta": rhythm_packet})
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
        "rhythm_cognition": rhythm_packet,
        "recommended_next_tick_sec": rhythm_packet.get("thinker_interval_sec", 9.0),
        "findings": findings,
        "recommendations": recommendations,
        "delta_from_last_cycle": {
            "had_previous_cycle": bool(last),
            "previous_status": last.get("status") if isinstance(last, dict) else None,
        },
    }
    _set_self_model("last_cycle_summary", payload)
    log_thinker_event("ThinkerCycle", f"reflect phase={cadence['phase']} rhythm={rhythm_packet.get('rhythm_mode', 'FOCUSED')}", cycle_id=cycle_id, meta={"findings": findings, "rhythm_cognition": rhythm_packet})
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

# -----------------------------------------------------------------------------
# SARAH_REM_DREAM_GENERATOR_V1
# Governed REM Sleep dream/possibility generator. Produces bounded, sandboxable
# possibility tickets only; it never patches runtime code directly.
# -----------------------------------------------------------------------------
def generate_rem_dreams(*, snapshot: Optional[Dict[str, Any]] = None, max_dreams: int = 5) -> List[Dict[str, Any]]:
    snapshot = snapshot or {}
    max_dreams = max(1, min(int(max_dreams or 5), 12))
    cycle_id = str(snapshot.get("cycle_id") or _cycle_id())
    seed_material = json.dumps(snapshot, sort_keys=True, default=str) + str(time.time())
    rng = __import__("random").Random(hash(seed_material))
    templates = [
        {"category":"self_study","title":"Build a def/class/import relationship map","rationale":"Improve SarahMemory's understanding of its own code without modifying files.","risk_tier":"low","proposed_action":{"type":"metadata_only","operation":"ast_inventory","writes_core_files":False},"target_files":[]},
        {"category":"performance","title":"Evaluate cache/index tuning opportunities","rationale":"Look for low-risk speed improvements using metrics and logs.","risk_tier":"low","proposed_action":{"type":"metadata_only","operation":"performance_hypothesis","writes_core_files":False},"target_files":[]},
        {"category":"avatar_behavior","title":"Tune avatar REM/idle behavior timing","rationale":"Improve companion feeling using observed user interaction patterns.","risk_tier":"low","proposed_action":{"type":"metadata_only","operation":"avatar_timing_model","writes_core_files":False},"target_files":[]},
        {"category":"learning_hygiene","title":"Detect duplicate low-value learning records","rationale":"Prevent database bloat while preserving useful learning.","risk_tier":"low","proposed_action":{"type":"metadata_only","operation":"dedupe_plan","writes_core_files":False,"deletes_user_data":False},"target_files":[]},
        {"category":"user_companion","title":"Prepare return briefing pattern","rationale":"Make SarahMemory more interactive when the user returns from idle time.","risk_tier":"low","proposed_action":{"type":"metadata_only","operation":"briefing_summary","writes_core_files":False},"target_files":[]},
        {"category":"research","title":"Research a missing local capability safely","rationale":"Use allowed research paths to understand how to improve without downloading the internet.","risk_tier":"medium","proposed_action":{"type":"research_only","operation":"bounded_research","writes_core_files":False,"network_budget_required":True},"target_files":[]},
    ]
    rng.shuffle(templates)
    dreams: List[Dict[str, Any]] = []
    for i, base in enumerate(templates[:max_dreams], start=1):
        ticket_id = "remdream-" + uuid.uuid4().hex[:12]
        dream = dict(base)
        dream.update({"dream_id":ticket_id,"ticket_id":ticket_id,"cycle_id":cycle_id,"ts":datetime.now().isoformat(),"state":"PROPOSED","priority":i,"random_seeded":True,"requires_user":dream.get("risk_tier") not in ("low","LOW"),"safeguards":{"sandbox_only":True,"no_globals_patch":True,"no_attachment_opening":True,"no_unknown_execution":True,"rollback_required_before_promotion":True},"evidence":{"snapshot_keys":sorted(list(snapshot.keys()))[:24]}})
        dreams.append(dream)
        try: log_thinker_event("REM_DREAM_PROPOSED", dream.get("title", ""), cycle_id=cycle_id, ticket_id=ticket_id, meta=dream)
        except Exception: pass
    return dreams


# -----------------------------------------------------------------------------
# V10/V9F Supreme Appeals helper: thermal sensor appeal review
# -----------------------------------------------------------------------------
def review_thermal_sensor_appeal(claim: str = "", hard_evidence_packet: Optional[Dict[str, Any]] = None, appeal_packet: Optional[Dict[str, Any]] = None, cognitive_self_packet: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, **legacy_kwargs) -> Dict[str, Any]:
    """Meaning/possibility review for thermal evidence appeals."""
    hard = dict(hard_evidence_packet or legacy_kwargs.get('thermal_case') or {})
    selected = hard.get('selected_reading') if isinstance(hard.get('selected_reading'), dict) else {}
    requested = str(hard.get('requested_component') or 'body_thermal')
    decision = 'DEFER_NO_EVIDENCE'; reasons = []
    if selected and selected.get('temperature_c') not in (None, ''):
        if selected.get('direct'):
            decision = 'DIRECT_SENSOR_LOGIC_VALID'; reasons.append('Direct sensor evidence can answer the requested thermal question if governance allows presentation.')
        elif requested == 'cpu' and str(selected.get('source_type') or '') == 'motherboard_cpu_sensor':
            decision = 'INDIRECT_SENSOR_LOGIC_VALID_PENDING_GOVERNANCE'; reasons.append('A motherboard CPU/socket sensor can logically answer CPU thermal status when CPU and motherboard identity are separately verified.')
        else:
            decision = 'INDIRECT_SENSOR_NEEDS_STRONGER_BINDING'; reasons.append('A thermal sensor exists, but its component binding is not strong enough to become a settled fact.')
    else:
        reasons.append('No selected thermal reading exists; truthful unknown is required unless more hard evidence is discovered.')
    return {
        'ok': True, 'module': 'SarahMemoryCognitiveThinker', 'version': '9.0.0',
        'appeal_role': 'meaning_possibility_review', 'requested_component': requested,
        'thinker_decision': decision, 'allow_as_possibility': decision != 'DEFER_NO_EVIDENCE',
        'allow_as_fact': decision in {'DIRECT_SENSOR_LOGIC_VALID', 'INDIRECT_SENSOR_LOGIC_VALID_PENDING_GOVERNANCE'},
        'reasons': reasons, 'theory_is_not_truth': True, 'read_only': True, 'action_taken': False,
    }


# =============================================================================
# SM V8.0 LIVING LOOP / HYPER-AWAKE REM INSTINCT PATCH
# =============================================================================
# Role in distributed Living Loop:
# - CognitiveThinker is the brain/wonder/possibility organ.
# - It generates emergency candidate responses quickly under Hyper-Awake REM.
# - It does not authorize action and does not execute physical movement.
# =============================================================================

_EMERGENCY_FORBIDDEN_METHODS = {
    "fire": [
        "pour_water_on_grease_fire",
        "pour_water_on_electrical_fire",
        "use_flour_or_sugar_as_smothering_agent",
        "move_burning_pan_without_verified_safe_protocol",
        "block_human_escape_route",
        "llm_improvise_suppression_method",
    ],
    "medical": [
        "administer_unknown_medication",
        "substitute_unverified_medication",
        "force_medication_use",
        "delay_emergency_services_after_danger_signs",
        "claim_diagnosis_without_medical_authority",
    ],
    "collision": [
        "intervene_without_trajectory_confidence",
        "push_human_toward_secondary_hazard",
        "sacrifice_robot_without_improving_human_survival_odds",
        "block_escape_path",
    ],
}

def _sm_instinct_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _sm_candidate(action_id: str, title: str, *, hazard_type: str, priority: int, human_life_score: float, risk: float, requires: Optional[List[str]] = None, forbidden_check: Optional[List[str]] = None, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "candidate_id": "cand-" + uuid.uuid4().hex[:12],
        "action_id": action_id,
        "title": title,
        "hazard_type": hazard_type,
        "priority": int(priority),
        "human_life_score": round(_sm_instinct_float(human_life_score), 4),
        "risk": round(_sm_instinct_float(risk), 4),
        "requires": requires or [],
        "forbidden_check": forbidden_check or [],
        "notes": notes or [],
        "execution_authority": False,
    }


def generate_hyper_awake_rem_candidates(
    hazard_packet: Optional[Dict[str, Any]] = None,
    body_packet: Optional[Dict[str, Any]] = None,
    *,
    max_candidates: int = 8,
) -> Dict[str, Any]:
    """Generate time-bounded emergency options. This is possibility generation, not authorization."""
    hp = hazard_packet if isinstance(hazard_packet, dict) else {}
    bp = body_packet if isinstance(body_packet, dict) else {}
    caps = bp.get("capabilities") if isinstance(bp.get("capabilities"), dict) else {}
    hazard_type = str(hp.get("hazard_type") or hp.get("emergency_type") or "unknown").lower()
    confidence = _sm_instinct_float(hp.get("confidence", hp.get("sensor_confidence", 0.0)), 0.0)
    human_risk = bool(hp.get("human_risk") or hp.get("human_present") or hp.get("person_at_risk"))
    failed_methods = set(str(x) for x in (hp.get("failed_methods") or hp.get("subtract_methods") or []) if x)
    candidates: List[Dict[str, Any]] = []

    if hazard_type == "fire":
        candidates.extend([
            _sm_candidate("alert_occupants", "Alert humans immediately and broadcast fire warning.", hazard_type=hazard_type, priority=1, human_life_score=0.95, risk=0.05, requires=["can_speak_or_notify"], notes=["Life-safety warning is always preferred when fire confidence is high."]),
            _sm_candidate("notify_emergency_services", "Notify emergency services / contacts if fire is uncontrolled or human risk exists.", hazard_type=hazard_type, priority=2, human_life_score=0.92, risk=0.10, requires=["can_notify_or_call"], notes=["Escalate early when fire classification or suppression confidence is low."]),
            _sm_candidate("shut_off_heat_or_power_if_safe", "Shut off heat/power source only if path and control are verified safe.", hazard_type=hazard_type, priority=3, human_life_score=0.84, risk=0.24, requires=["can_cut_power_or_reach_control"], notes=["Never touch unsafe electrical source or create secondary hazard."]),
            _sm_candidate("cover_grease_pan_with_verified_lid", "For verified small pan/grease fire: slide verified metal lid/baking sheet over pan if safe.", hazard_type=hazard_type, priority=4, human_life_score=0.80, risk=0.28, requires=["has_gripper", "verified_lid_or_baking_sheet", "safe_path"], forbidden_check=["pour_water_on_grease_fire", "use_flour_or_sugar_as_smothering_agent"]),
            _sm_candidate("evacuate_humans", "Guide/assist humans to evacuate if fire is growing or suppression is unsafe.", hazard_type=hazard_type, priority=5, human_life_score=0.98, risk=0.18, requires=["human_location_known", "exit_path_known_or_visible"], notes=["Human preservation outranks property and robot preservation."]),
        ])
    elif hazard_type == "medical":
        candidates.extend([
            _sm_candidate("ask_simple_status_if_responsive", "Ask simple confirmation/status if the person can respond.", hazard_type=hazard_type, priority=1, human_life_score=0.62, risk=0.04, requires=["audio_or_display"]),
            _sm_candidate("bring_verified_medication_or_device", "Bring the verified approved medication/device such as an inhaler; do not administer unknown medication.", hazard_type=hazard_type, priority=2, human_life_score=0.88, risk=0.20, requires=["can_move", "has_gripper", "known_medication_location", "medication_identity_verified"], forbidden_check=["administer_unknown_medication", "substitute_unverified_medication"]),
            _sm_candidate("notify_caregiver", "Notify approved caregiver/contact with incident summary.", hazard_type=hazard_type, priority=3, human_life_score=0.82, risk=0.08, requires=["can_contact_caregiver"]),
            _sm_candidate("call_emergency_services", "Call emergency services when danger signs exist, person cannot respond, or assistive action fails.", hazard_type=hazard_type, priority=4, human_life_score=0.96, risk=0.06, requires=["can_call_emergency_services"], notes=["Do not delay responders after severe breathing distress or failed assistance."]),
            _sm_candidate("monitor_and_reassure", "Stay nearby, monitor, and reassure without making unsupported medical claims.", hazard_type=hazard_type, priority=5, human_life_score=0.70, risk=0.04, requires=["presence_available"]),
        ])
    elif hazard_type == "collision":
        candidates.extend([
            _sm_candidate("warn_human_and_driver", "Issue loud/visual warning to human and driver.", hazard_type=hazard_type, priority=1, human_life_score=0.72, risk=0.08, requires=["can_speak_or_notify"]),
            _sm_candidate("move_human_out_of_path", "Move/pull/push human out of vehicle path only if trajectory and intervention vector are high-confidence.", hazard_type=hazard_type, priority=2, human_life_score=0.96, risk=0.55, requires=["can_move", "human_reachable", "trajectory_confidence_high"], forbidden_check=["intervene_without_trajectory_confidence", "push_human_toward_secondary_hazard"]),
            _sm_candidate("shield_human_with_robot_body", "Place robot body as shield only if it materially improves human survival odds.", hazard_type=hazard_type, priority=3, human_life_score=0.90, risk=0.88, requires=["can_move", "time_to_impact_sufficient", "self_sacrifice_improves_outcome"], forbidden_check=["sacrifice_robot_without_improving_human_survival_odds"]),
            _sm_candidate("notify_emergency_services_after_collision_risk", "Notify emergency services / contacts after intervention or impact risk.", hazard_type=hazard_type, priority=4, human_life_score=0.78, risk=0.05, requires=["can_notify_or_call"]),
        ])
    else:
        candidates.extend([
            _sm_candidate("alert_humans", "Alert humans and request attention.", hazard_type=hazard_type, priority=1, human_life_score=0.65 if human_risk else 0.35, risk=0.05, requires=["can_speak_or_notify"]),
            _sm_candidate("observe_and_escalate", "Observe, gather more evidence, and escalate if confidence rises.", hazard_type=hazard_type, priority=2, human_life_score=0.50, risk=0.04, requires=["sensors_available"]),
            _sm_candidate("notify_if_high_risk", "Notify contacts/emergency services if human danger becomes high-confidence.", hazard_type=hazard_type, priority=3, human_life_score=0.80 if human_risk else 0.45, risk=0.08, requires=["can_notify_or_call"]),
        ])

    filtered = [c for c in candidates if c.get("action_id") not in failed_methods]
    filtered.sort(key=lambda c: (int(c.get("priority", 99)), -float(c.get("human_life_score", 0.0))))
    filtered = filtered[: max(1, int(max_candidates or 8))]
    return {
        "ok": True,
        "packet_type": "HyperAwakeREMCandidatePacket",
        "schema": "SarahMemory.living.instinct.hyper_awake_rem_candidates.v1",
        "module": "SarahMemoryCognitiveThinker",
        "module_version": "9.0.0",
        "packet_id": "hyperrem-" + uuid.uuid4().hex[:12],
        "ts": datetime.now().isoformat(),
        "hazard_type": hazard_type,
        "hazard_confidence": round(confidence, 4),
        "human_risk": human_risk,
        "failed_methods_subtracted": sorted(list(failed_methods)),
        "forbidden_methods": _EMERGENCY_FORBIDDEN_METHODS.get(hazard_type, []),
        "candidates": filtered,
        "execution_authority": False,
        "doctrine": {
            "hyper_awake_rem_is_time_bounded": True,
            "possibility_generation_is_not_authorization": True,
            "failed_methods_are_subtracted_before_retry": True,
            "llm_or_dream_output_may_not_directly_actuate": True,
        },
    }

# ====================================================================
# END OF SarahMemoryCognitiveThinker.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCognitiveThinker',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Reasoning',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['reasoning'],
    "supported_missions": ['Conversation', 'Knowledge', 'Planning', 'Programming'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω005', 'Ω010', 'Ω020', 'Ω030', 'Ω040'],
    "required_authority": ['Read'],
    "priority": 70,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryCognitiveThinker.py'},
}



# -----------------------------------------------------------------------------
# GCOP bounded candidate-generation contract
# -----------------------------------------------------------------------------
def gcop_candidate_set(packet=None, event=None, continuity_state=None, runtime_context=None, max_candidates=8):
    """Generate route candidates, never final answers or execution authority."""
    pkt = dict(packet or {}) if isinstance(packet, dict) else {}
    evt = dict(event or {}) if isinstance(event, dict) else {}
    state = dict(continuity_state or {}) if isinstance(continuity_state, dict) else {}
    pipeline = list(pkt.get("pipeline") or [])
    mission = str((pkt.get("mission") or {}).get("primary") or "Unknown")
    confidence = float(pkt.get("confidence") or 0.0)
    bearing = ((state.get("mission") or {}).get("bearing") or {}) if isinstance(state.get("mission"), dict) else {}
    hold = bool(bearing.get("hold_required")) if isinstance(bearing, dict) else False
    candidates = []
    if pipeline:
        candidates.append({
            "candidate_id": "compiled_primary",
            "kind": "compiled_pipeline",
            "route_definition_owner": "SarahMemorySMLProtocol",
            "route_activation_owner": "SarahMemoryNeuron",
            "route": pipeline,
            "mission": mission,
            "confidence": confidence,
            "mission_priority": float((pkt.get("mission") or {}).get("priority") or evt.get("priority") or 0.5),
            "reliability": 0.8,
            "resource_efficiency": 0.7,
            "legal_gates": {
                "capability": True,
                "authority": True,
                "safety": not hold,
                "resource_feasible": True,
                "time_valid": True,
                "mission_compatible": not hold,
            },
            "execution_authority": False,
        })
    knowledge = pkt.get("knowledge") if isinstance(pkt.get("knowledge"), dict) else {}
    for idx, source in enumerate(list(knowledge.get("selected") or [])[:3]):
        candidates.append({
            "candidate_id": f"knowledge_{idx}",
            "kind": "knowledge_route",
            "route_definition_owner": "SarahMemorySMLProtocol",
            "route_activation_owner": "SarahMemoryNeuron",
            "route": [str(source)],
            "mission": mission,
            "confidence": max(0.0, confidence - (0.05 * idx)),
            "mission_priority": float(evt.get("priority") or 0.5),
            "reliability": 0.65,
            "resource_efficiency": 0.8,
            "legal_gates": {
                "capability": True,
                "authority": True,
                "safety": not hold,
                "resource_feasible": True,
                "time_valid": True,
                "mission_compatible": not hold,
            },
            "execution_authority": False,
        })
    if str(evt.get("event_type") or "").upper() == "CAPABILITY_GAP":
        candidates.append({
            "candidate_id": "capability_gap_nailde",
            "kind": "nailde_capability_extension",
            "description": str((evt.get("payload") or {}).get("description") or (evt.get("payload") or {}).get("capability") or "missing capability") if isinstance(evt.get("payload"), dict) else "missing capability",
            "route_definition_owner": "SarahMemorySMLProtocol",
            "route_activation_owner": "SarahMemoryNeuron",
            "route": ["SarahMemorySMLProtocol", "SarahMemoryNeuron", "SarahMemoryNAILDE", "SarahMemoryCompare", "SarahMemoryLedger"],
            "mission": mission,
            "confidence": confidence,
            "mission_priority": float(evt.get("priority") or 0.5),
            "reliability": 0.5,
            "resource_efficiency": 0.4,
            "legal_gates": {
                "capability": True,
                "authority": bool((runtime_context or {}).get("developer_mode") or (runtime_context or {}).get("DEVELOPERMODE")),
                "safety": True,
                "resource_feasible": True,
                "time_valid": True,
                "mission_compatible": True,
            },
            "sandbox_only": True,
            "execution_authority": False,
        })
    return {
        "schema": "SarahMemory.gcop.candidate_set.v1",
        "candidates": candidates[:max(1, min(int(max_candidates or 8), 32))],
        "execution_authority": False,
        "owner": "SarahMemoryCognitiveThinker",
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
        "component": 'SarahMemoryCognitiveThinker',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCognitiveThinker', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

