"""--==The SarahMemory Project==--
File: SarahMemoryAssuranceGate.py
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

- Assurance and confidence hard-gate for SarahMemory AiOS / SMGET.
- Converts "the system thinks this is okay" into "the system has enough assurance to proceed."
- Blocks low-confidence, weakly-specified, or under-verified action requests before real execution.
- Enforces the doctrine that risky execution requires stronger confidence, stronger verification planning,
and stronger rollback readiness.
- Closes the "no hallucinated actuation" gap by refusing high-risk action flows that lack adequate
evidence, structure, or confirmation.

CORE DOCTRINE:
- No hallucinated actuation.
- Riskier actions require stronger assurance.
- Apply-mode execution must not rely on vague or low-evidence intent.
- Verification readiness matters as much as confidence.
- Rollback readiness matters for non-trivial apply-mode operations.
- Fail closed when assurance is insufficient.

RELATIONSHIP MODEL:
- SarahMemoryCognitiveServices.py  -> logical/risk governance
- SarahMemorySecurityGovernor.py   -> trust / sovereignty / integrity hard gate
- SarahMemorySafetyPolicies.py     -> constitutional safety rules and runtime policy snapshot
- SarahMemoryAssuranceGate.py      -> assurance thresholding / confidence / readiness hard gate
- SarahMemoryOperatorCore.py       -> lifecycle / dispatch / verify / rollback orchestration
- Executors / Drivers              -> perform bounded work only after assurance passes
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "assurance_gate"
# CATEGORY = "runtime_assurance_and_confidence"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "assurance_gate"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "assurance_gate"
# FAMILY = "smget"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "SMGET assurance gate. Evaluates confidence, verification readiness, rollback readiness, and execution risk before real action dispatch."
# --- SARAHMETA END ---

import json
import logging
import re
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None

try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryAssuranceGate")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryAssuranceGate"
MODULE_VERSION = "9.1.0a"
_DB_NAME = "assurance_gate.db"
_JSON_NAME = "assurance_gate_snapshot.json"

MODE_SIMULATE = "simulate"
MODE_DRAFT = "draft"
MODE_APPLY = "apply"
MODE_ROLLBACK = "rollback"

RISK_TIER_0 = "TIER_0_INFO"
RISK_TIER_1 = "TIER_1_HARMLESS_LOCAL_UI"
RISK_TIER_2 = "TIER_2_BOUNDED_LOCAL_OPERATION"
RISK_TIER_3 = "TIER_3_PRIVILEGED_SYSTEM"
RISK_TIER_4 = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

DECISION_ALLOW = "ALLOW"
DECISION_REQUIRE_USER = "REQUIRE_USER"
DECISION_SIMULATE_ONLY = "SIMULATE_ONLY"
DECISION_DENY = "DENY"

TRUST_TIER_CORE = "core"
TRUST_TIER_FIRST_PARTY = "first_party"
TRUST_TIER_VERIFIED_THIRD_PARTY = "verified_third_party"
TRUST_TIER_UNVERIFIED = "unverified"
TRUST_TIER_UNKNOWN = "unknown"
TRUST_TIER_QUARANTINED = "quarantined"

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AssuranceSignal:
    name: str
    value: float
    source: str
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssuranceReview:
    review_id: str
    ts: str
    decision: str
    allow: bool
    require_user: bool
    action_type: str
    capability_name: str
    execution_mode: str
    risk_level: str
    confidence: float
    assurance_score: float
    threshold: float
    verification_ready: bool
    rollback_ready: bool
    reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# v9.1.0a: AssuranceGate validates evidence; it never mints approval authority.
def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}

def _strict_true(value: Any) -> bool:
    return value is True or (isinstance(value, int) and not isinstance(value,bool) and value==1)

def _positive_finite(value: Any) -> bool:
    try:
        n=float(value)
        return math.isfinite(n) and n>0
    except (TypeError,ValueError,OverflowError):
        return False

def _valid_numeric_signal(value: Any, percentage: bool=False) -> bool:
    if isinstance(value,bool) or value is None:
        return False
    try:
        n=float(value)
        return math.isfinite(n) and 0 <= n <= (100 if percentage else 100)
    except (TypeError,ValueError,OverflowError):
        return False

def _authoritative_allow(record: Dict[str,Any]) -> bool:
    return record.get("allow") is True and _safe_lower(record.get("decision")) == "allow"

def _redact_bound(value: Any, *, depth: int=0) -> Any:
    """Bound and redact audit metadata. Original review remains available in memory."""
    if depth>6: return "[DEPTH_LIMIT]"
    if isinstance(value,dict):
        out={}
        for i,(k,v) in enumerate(value.items()):
            if i>=96: out["_truncated"]="[FIELD_LIMIT]";break
            key=str(k)[:100]
            if any(x in key.lower() for x in ("password","secret","token","api_key","apikey","authorization","cookie","private_key","credential","bearer","session_key","lease_id","approval_receipt","return_nonce")):
                out[key]="[REDACTED]"
            else:out[key]=_redact_bound(v,depth=depth+1)
        return out
    if isinstance(value,(list,tuple)):
        return [_redact_bound(v,depth=depth+1) for v in value[:96]]
    if isinstance(value,str):
        import re
        t=value[:1500]
        t=re.sub(r"(?i)\b(bearer\s+)[A-Za-z0-9._~-]+",r"\1[REDACTED]",t)
        t=re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{12,})\b","[REDACTED]",t)
        return t
    if isinstance(value,float) and not math.isfinite(value):return None
    if value is None or isinstance(value,(int,float,bool)):return value
    return str(type(value).__name__)

_DB_SCHEMA_READY=False
_DB_SCHEMA_LOCK=threading.RLock()

def _audit_log_review(review: AssuranceReview) -> bool:
    """Record a final assurance review with bounded retention; return durable outcome."""
    global _DB_SCHEMA_READY
    con=None
    try:
        with _DB_SCHEMA_LOCK:
            if not _DB_SCHEMA_READY:
                _ensure_tables()
                # _ensure_tables is best-effort, actual insert will validate success.
                _DB_SCHEMA_READY=True
        con=_connect_db()
        if con is None:return False
        safe=_redact_bound(review.to_dict())
        encoded=json.dumps(safe,ensure_ascii=False,sort_keys=True,default=str)
        if len(encoded)>65536:
            safe = {"review_id":review.review_id,"decision":review.decision,"execution_mode":review.execution_mode,
                    "meta":{"audit_truncated":True},"execution_authority":False}
            encoded=json.dumps(safe,ensure_ascii=False,sort_keys=True)
        with con:
            con.execute("INSERT INTO assurance_events (ts,review_id,decision,action_type,capability_name,execution_mode,risk_level,confidence,assurance_score,threshold,details,meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (review.ts,review.review_id,review.decision,_redact_bound(review.action_type),_redact_bound(review.capability_name),
                 review.execution_mode,review.risk_level,review.confidence,review.assurance_score,
                 review.threshold,_redact_bound("; ".join(review.reasons[:4])[:1024]),encoded))
            # Retain a bounded number of recent reviews, incrementally, no full table scans.
            cap=max(100,min(50000,int(os.getenv("SARAH_ASSURANCE_MAX_AUDIT_ROWS","10000") or 10000)))
            con.execute("DELETE FROM assurance_events WHERE id <= (SELECT COALESCE(MAX(id),0)-? FROM assurance_events)",(cap,))
        return True
    except Exception as exc:
        logger.warning("Assurance audit unavailable: %s",type(exc).__name__)
        with _DB_SCHEMA_LOCK:_DB_SCHEMA_READY=False
        return False
    finally:
        if con is not None:
            try:con.close()
            except Exception:pass

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


def _settings_dir() -> str:
    try:
        return str(getattr(config, "SETTINGS_DIR", os.path.join(_data_dir(), "settings")))
    except Exception:
        return os.path.join(_data_dir(), "settings")


def _assurance_db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _snapshot_path() -> str:
    return os.path.join(_settings_dir(), _JSON_NAME)


def _ensure_dirs() -> None:
    try:
        os.makedirs(_datasets_dir(), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(_settings_dir(), exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect_db() -> Optional[sqlite3.Connection]:
    try:
        _ensure_dirs()
        con = sqlite3.connect(_assurance_db_path(), timeout=5.0, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return con
    except Exception:
        return None


def _ensure_tables() -> None:
    con = None
    try:
        con = _connect_db()
        if con is None:
            return
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assurance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                review_id TEXT,
                decision TEXT,
                action_type TEXT,
                capability_name TEXT,
                execution_mode TEXT,
                risk_level TEXT,
                confidence REAL,
                assurance_score REAL,
                threshold REAL,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as exc:
        logger.debug("AssuranceGate DB ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(review: AssuranceReview) -> None:
    """Backward-compatible logging entry; records only redacted bounded evidence."""
    _audit_log_review(review)


def _write_snapshot(review: AssuranceReview) -> None:
    import tempfile
    path = _snapshot_path()
    tmp = None
    try:
        _ensure_dirs()
        fd, tmp = tempfile.mkstemp(prefix=".assurance_",suffix=".tmp",dir=os.path.dirname(path))
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f:
                json.dump(_redact_bound(review.to_dict()),f,indent=2,ensure_ascii=False,allow_nan=False)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
            tmp=None
        finally:
            if tmp and os.path.exists(tmp):os.unlink(tmp)
    except Exception as exc:
        logger.warning("Assurance snapshot unavailable: %s",type(exc).__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return "{}"


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("non-finite confidence")
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError, OverflowError):
        try:
            d = float(default)
            return max(0.0, min(1.0, d)) if math.isfinite(d) else 0.0
        except Exception:
            return 0.0


def _risk_weight(risk_level: str) -> float:
    mapping = {
        RISK_TIER_0: 0.10,
        RISK_TIER_1: 0.18,
        RISK_TIER_2: 0.35,
        RISK_TIER_3: 0.62,
        RISK_TIER_4: 0.82,
    }
    return float(mapping.get(str(risk_level or RISK_TIER_0), 0.35))


def _trust_bonus(trust_tier: str) -> float:
    mapping = {
        TRUST_TIER_CORE: 0.18,
        TRUST_TIER_FIRST_PARTY: 0.12,
        TRUST_TIER_VERIFIED_THIRD_PARTY: 0.06,
        TRUST_TIER_UNVERIFIED: -0.04,
        TRUST_TIER_UNKNOWN: -0.08,
        TRUST_TIER_QUARANTINED: -0.25,
    }
    return float(mapping.get(str(trust_tier or TRUST_TIER_UNKNOWN), -0.08))


def _mode_bonus(execution_mode: str) -> float:
    mapping = {
        MODE_SIMULATE: 0.10,
        MODE_DRAFT: 0.05,
        MODE_APPLY: -0.02,
        MODE_ROLLBACK: 0.02,
    }
    return float(mapping.get(str(execution_mode or MODE_DRAFT), 0.0))


def _threshold_for(execution_mode: str, risk_level: str) -> float:
    mode = str(execution_mode or MODE_DRAFT)
    risk = str(risk_level or RISK_TIER_0)
    table = {
        MODE_SIMULATE: {
            RISK_TIER_0: 0.10,
            RISK_TIER_1: 0.15,
            RISK_TIER_2: 0.25,
            RISK_TIER_3: 0.35,
            RISK_TIER_4: 0.45,
        },
        MODE_DRAFT: {
            RISK_TIER_0: 0.20,
            RISK_TIER_1: 0.28,
            RISK_TIER_2: 0.42,
            RISK_TIER_3: 0.58,
            RISK_TIER_4: 0.70,
        },
        MODE_APPLY: {
            RISK_TIER_0: 0.45,
            RISK_TIER_1: 0.55,
            RISK_TIER_2: 0.70,
            RISK_TIER_3: 0.84,
            RISK_TIER_4: 0.92,
        },
        MODE_ROLLBACK: {
            RISK_TIER_0: 0.30,
            RISK_TIER_1: 0.38,
            RISK_TIER_2: 0.52,
            RISK_TIER_3: 0.60,
            RISK_TIER_4: 0.70,
        },
    }
    return float(table.get(mode, table[MODE_DRAFT]).get(risk, 0.58))


def _now_iso() -> str:
    return datetime.now().isoformat()


def _policy_snapshot() -> Dict[str, Any]:
    # Safety defaults are intentionally conservative when Globals/Policies fail.
    default = {
        "safe_mode": bool(getattr(config, "SAFE_MODE", True)),
        "local_only": bool(getattr(config, "LOCAL_ONLY_MODE", True)),
        "neosky_armed": bool(getattr(config, "NEOSKYMATRIX", False) and getattr(config, "DEVELOPERSMODE", False)),
        "require_rollback_for_apply": True,
        "require_verification_before_success": True,
        "require_confirmation_for_privileged": True,
        "require_confirmation_for_network_exposure": True,
        "require_confirmation_for_physical_control": True,
        "fail_closed_on_policy_error": True,
        "allow_model_direct_authority": False,
        "riskier_actions_require_stronger_assurance": True,
    }
    if _SafetyPolicies is None:
        return default
    for fn_name in ("get_runtime_policy_snapshot", "get_policy_snapshot", "get_security_policy_snapshot"):
        try:
            fn = getattr(_SafetyPolicies, fn_name, None)
            if callable(fn):
                out = fn()
                if isinstance(out, dict):
                    merged = dict(default)
                    merged.update(out)
                    # An optional policy snapshot cannot independently disable
                    # constitutional assurance invariants.
                    for key in ("require_rollback_for_apply", "require_verification_before_success",
                                "require_confirmation_for_privileged", "require_confirmation_for_network_exposure",
                                "require_confirmation_for_physical_control", "fail_closed_on_policy_error",
                                "riskier_actions_require_stronger_assurance"):
                        merged[key] = True
                    merged["allow_model_direct_authority"] = False
                    merged["safe_mode"] = default["safe_mode"] or _strict_true(out.get("safe_mode"))
                    merged["local_only"] = default["local_only"] or _strict_true(out.get("local_only"))
                    return merged
        except Exception as exc:
            logger.debug("SafetyPolicies.%s failed: %s", fn_name, type(exc).__name__)
    return default


def _extract_confidence_signals(action_contract: Dict[str, Any], governance: Dict[str, Any], security: Dict[str, Any]) -> List[AssuranceSignal]:
    contract = dict(action_contract or {})
    meta = _as_mapping(contract.get("metadata"))
    signals: List[AssuranceSignal] = []
    def _maybe_add(name: str, value: Any, source: str, weight: float = 1.0, *, scale_from_pct: bool = False) -> None:
        if value is None or isinstance(value, bool) or str(value).strip() == "":
            return
        try:
            fv = float(value)
            if not math.isfinite(fv):
                return
            if scale_from_pct or fv > 1.0:
                fv = fv / 100.0
            if not 0.0 <= fv <= 1.0:
                return
            signals.append(AssuranceSignal(name=name, value=fv, source=source, weight=weight))
        except (ValueError, TypeError, OverflowError):
            return
    # Caller-provided model scores are supplementary, not approvals.
    for key, weight in (("confidence", 1.0), ("intent_confidence", 1.0),
                        ("advcu_confidence", 1.1), ("neuron_confidence", 1.1),
                        ("plan_confidence", 0.9), ("executor_confidence", 0.9),
                        ("verification_confidence", 1.2)):
        _maybe_add(key, meta.get(key), "contract.metadata", weight)
        _maybe_add(key, contract.get(key), "contract", weight)
    _maybe_add("governance_confidence", governance.get("confidence"), "governance", 1.1)
    _maybe_add("security_confidence", security.get("confidence"), "security", 1.0)
    # Missing risk score is UNKNOWN, never zero-risk or perfect confidence.
    for origin, rec in (("governance", governance), ("security", security)):
        if "risk_score" in rec:
            try:
                risk = float(rec["risk_score"])
                if math.isfinite(risk) and 0.0 <= risk <= 100.0:
                    _maybe_add(origin + "_risk_inverse", (100.0-risk)/100.0, origin, 0.8)
            except (TypeError, ValueError, OverflowError):
                pass
    return signals


def _weighted_confidence(signals: List[AssuranceSignal]) -> float:
    total_weight = total = 0.0
    for sig in signals[:64]:
        try:
            w = float(sig.weight)
            v = float(sig.value)
            if not math.isfinite(w) or not math.isfinite(v) or w <= 0 or not 0 <= v <= 1:
                continue
            total_weight += w
            total += w * v
        except (TypeError, ValueError, OverflowError):
            continue
    return _clamp01(total / total_weight) if total_weight > 0 else 0.0


def _verification_readiness(action_contract: Dict[str, Any]) -> Tuple[float, bool, List[str]]:
    contract = dict(action_contract or {})
    missing: List[str] = []
    checks = contract.get("verification_checks")
    def meaningful(x: Any) -> bool:
        if isinstance(x, str):
            return len(x.strip()) >= 3
        if isinstance(x, dict):
            return any(isinstance(x.get(k), str) and len(x[k].strip()) >= 3 for k in ("name", "check", "assertion", "expected", "predicate"))
        return False
    verified_checks = isinstance(checks, list) and 0 < len(checks) <= 128 and all(meaningful(x) for x in checks)
    expected = contract.get("expected_result")
    expected_ok = isinstance(expected, str) and len(expected.strip()) >= 5
    pre = contract.get("preconditions")
    pre_ok = isinstance(pre, list) and 0 < len(pre) <= 128 and all(meaningful(x) for x in pre)
    target_ok = bool(str(contract.get("target") or contract.get("target_ref") or "").strip())
    capability_ok = bool(str(contract.get("capability_name") or "").strip())
    score = (0.55 if verified_checks else 0) + (0.25 if expected_ok else 0) + (0.10 if pre_ok else 0) + (0.05 if target_ok else 0) + (0.05 if capability_ok else 0)
    for ok,key in ((verified_checks,"verification_checks"),(expected_ok,"expected_result"),(pre_ok,"preconditions"),(target_ok,"target"),(capability_ok,"capability_name")):
        if not ok: missing.append(key)
    # Every apply path needs an actual verification check AND expected result.
    return _clamp01(score), bool(verified_checks and expected_ok and score >= 0.80), missing


def _rollback_readiness(action_contract: Dict[str, Any]) -> Tuple[float, bool, List[str]]:
    contract = dict(action_contract or {})
    mode = _safe_lower(contract.get("execution_mode")) or MODE_DRAFT
    risk = str(contract.get("risk_level") or RISK_TIER_0)
    plan = contract.get("rollback_plan")
    def meaningful(x: Any) -> bool:
        return isinstance(x, str) and len(x.strip()) >= 3 or isinstance(x, dict) and any(isinstance(x.get(k), str) and len(x[k].strip()) >= 3 for k in ("name", "action", "step", "operation"))
    exists = isinstance(plan, list) and 0 < len(plan) <= 128 and all(meaningful(x) for x in plan)
    if mode not in {MODE_APPLY, MODE_ROLLBACK}:
        return 1.0, True, []
    if mode == MODE_APPLY and risk in {RISK_TIER_0, RISK_TIER_1}:
        return (1.0 if exists else 0.80), True, []
    return (1.0, True, []) if exists else (0.0, False, ["rollback_plan"])


def _evidence_freshness_profile(action_contract: Dict[str, Any]) -> Dict[str, Any]:
    contract = dict(action_contract or {})
    meta = _as_mapping(contract.get("metadata"))
    age = contract.get("evidence_age_seconds", meta.get("evidence_age_seconds"))
    lifetime = contract.get("evidence_useful_lifetime_seconds", meta.get("evidence_useful_lifetime_seconds", meta.get("freshness_tau_seconds")))
    ts = contract.get("evidence_timestamp", meta.get("evidence_timestamp", meta.get("current_perception_timestamp", meta.get("telemetry_timestamp"))))
    try:
        if ts is not None:
            observed_ts = float(ts) if isinstance(ts, (float, int)) else datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
            if not math.isfinite(observed_ts) or observed_ts > time.time()+2.0:
                return {"known": False, "score": None, "fresh": False, "error": "invalid_or_future_timestamp"}
            calculated_age = time.time() - observed_ts
            if age is None: age = calculated_age
        if age is None or lifetime is None:
            return {"known": False, "score": None, "fresh": False, "age_seconds": age, "useful_lifetime_seconds": lifetime}
        age_f, tau = float(age), float(lifetime)
        if not math.isfinite(age_f) or not math.isfinite(tau) or age_f < 0 or tau <= 0:
            raise ValueError("invalid age or lifetime")
        # A caller-supplied age does not override an older timestamp.
        if ts is not None and age_f + 2 < calculated_age:
            return {"known": False, "score": None, "fresh": False, "error": "age_timestamp_mismatch"}
        score = math.exp(-age_f/tau)
        minimum = _clamp01(contract.get("minimum_freshness",meta.get("minimum_freshness",0.5)),0.5)
        return {"known": True, "score": score, "fresh": score >= minimum, "minimum": minimum,
                "age_seconds": age_f, "useful_lifetime_seconds": tau}
    except (TypeError, ValueError, OverflowError) as exc:
        return {"known": False, "score": None, "fresh": False, "error": type(exc).__name__}


def _controller_continuity_profile(action_contract: Dict[str, Any]) -> Dict[str, Any]:
    contract = dict(action_contract or {})
    meta = _as_mapping(contract.get("metadata"))
    controller = str(contract.get("controller_identity") or meta.get("controller_identity") or meta.get("controller") or "").strip()
    owner = str(contract.get("current_control_owner") or meta.get("current_control_owner") or "").strip()
    lease_id = str(contract.get("authority_lease_id") or meta.get("authority_lease_id") or "").strip()
    status = _safe_lower(contract.get("authority_lease_status") or meta.get("authority_lease_status"))
    expiry = contract.get("authority_lease_expires_ts", meta.get("authority_lease_expires_ts"))
    try:
        expiry_f = float(expiry)
        valid_expiry = math.isfinite(expiry_f) and expiry_f > time.time()
    except (TypeError, ValueError, OverflowError):
        valid_expiry = False
    known = bool(controller or owner or lease_id)
    valid = bool(controller and owner and lease_id and controller == owner and status == "active" and valid_expiry)
    return {"known": known, "valid": valid, "controller_identity": controller, "current_control_owner": owner,
            "authority_lease_id": lease_id, "authority_lease_status": status or "unverified",
            "expired": not valid_expiry, "owner_mismatch": bool(controller and owner and controller != owner),
            "execution_authority": False}


def _has_explicit_confirmation(action_contract: Dict[str, Any], governance: Dict[str, Any], security: Dict[str, Any]) -> bool:
    # Never accept self-attested booleans on an action contract/metadata.
    # Governance and SecurityGovernor are the only accepted *input sources*;
    # actual receipt validation stays with those owners.
    for owner in (governance, security):
        if _authoritative_allow(owner) and any(_strict_true(owner.get(k)) for k in (
                "operator_approval_verified", "user_confirmation_verified", "approval_receipt_verified",
                "verified_operator_receipt")):
            return True
    return False


def _trust_tier(security: Dict[str, Any]) -> str:
    caller = security.get("caller_identity") or {}
    if isinstance(caller, dict):
        tier = str(caller.get("trust_tier") or "").strip()
        if tier:
            return tier
    return TRUST_TIER_UNKNOWN


def _robotic_body_profile(action_contract: Dict[str, Any]) -> Dict[str, Any]:
    contract = dict(action_contract or {})
    meta = _as_mapping(contract.get("metadata"))
    permissions = contract.get("required_permissions") if isinstance(contract.get("required_permissions"), list) else []
    joined = " ".join([_safe_lower(contract.get(k)) for k in ("action_type","capability_name","executor_name","target","normalized_text")] + [_safe_lower(p) for p in permissions])
    risk = str(contract.get("risk_level") or "")
    is_robotic = risk.startswith("TIER_ROBOT_") or bool(re.search(r"\b(robot|robotic|servo|gripper|locomotion|torque|actuator|physical body|robot arm|human contact)\b",joined))
    motion = is_robotic and (risk not in ("TIER_ROBOT_OBSERVE_ONLY", "TIER_ROBOT_FACE_EXPRESSION") or bool(re.search(r"\b(move|walk|step|reach|raise|lower|turn|grip|release|posture|actuate|rotate)\b",joined)))
    env = _as_mapping(contract.get("safety_envelope"))
    return {"is_robotic": is_robotic, "motion_requested": motion,
            "current_perception_fresh": _strict_true(contract.get("current_perception_fresh")) or _strict_true(meta.get("current_perception_fresh")),
            "safe_stop_available": _strict_true(contract.get("safe_stop_available")) or _strict_true(meta.get("safe_stop_available")),
            "force_speed_limits_present": any(_positive_finite(env.get(k)) for k in ("max_force_n","max_speed_mps","max_torque_nm")),
            "body_part_driver_verified": _strict_true(contract.get("body_part_driver_verified")) or _strict_true(meta.get("body_part_driver_verified"))}


def _seems_hazardous(action_contract: Dict[str, Any]) -> bool:
    contract = dict(action_contract or {})
    joined = " ".join([
        _safe_lower(contract.get("action_type")),
        _safe_lower(contract.get("capability_name")),
        _safe_lower(contract.get("target")),
        _safe_lower(contract.get("normalized_text")),
        " ".join([_safe_lower(x) for x in (contract.get("required_permissions") or [])]),
    ])
    keywords = (
        "delete", "wipe", "format", "firewall", "driver.control", "camera", "microphone", "plc",
        "cnc", "machine", "serial", "usb", "remote", "public", "ingress", "port", "terminal.execute",
        "admin", "service", "boot", "network",
    )
    return any(k in joined for k in keywords)


def _compare_hint(action_contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if _Compare is None:
        return None
    contract = dict(action_contract or {})
    text = str(contract.get("normalized_text") or "").strip()
    expected = str(contract.get("expected_result") or "").strip()
    if not text or not expected:
        return None
    try:
        fn = getattr(_Compare, "evaluate_similarity", None)
        if callable(fn):
            score = fn(text, expected)
            if score is not None:
                try:
                    score_f = float(score)
                    if score_f > 1.0:
                        score_f = score_f / 100.0
                    return {"name": "compare_similarity", "score": _clamp01(score_f)}
                except Exception:
                    return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Main assurance evaluation
# ---------------------------------------------------------------------------
def evaluate_action_assurance(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Non-authorizing, fail-closed assurance review. No input alone grants execution.

    `governance` and `security` must be results from existing trusted organs;
    this function cannot independently authenticate their transport origin.
    """
    contract = _as_mapping(action_contract)
    gov, sec = _as_mapping(governance), _as_mapping(security)
    mode = _safe_lower(contract.get("execution_mode")) or MODE_DRAFT
    risk = str(contract.get("risk_level") or "")
    action_type = str(contract.get("action_type") or "unknown")[:180]
    capability = str(contract.get("capability_name") or "unknown")[:180]
    policy = _policy_snapshot()
    robot = _robotic_body_profile(contract)
    freshness = _evidence_freshness_profile(contract)
    controller = _controller_continuity_profile(contract)
    signals = _extract_confidence_signals(contract,gov,sec)
    if freshness.get("known"):
        signals.append(AssuranceSignal("evidence_freshness",_clamp01(freshness.get("score")),"telemetry",0.8, freshness))
    if controller.get("known"):
        signals.append(AssuranceSignal("controller_continuity",1.0 if controller.get("valid") else 0.0,"action_contract",0.8,controller))
    hint = _compare_hint(contract)
    if hint:
        signals.append(AssuranceSignal("compare_similarity",_clamp01(hint.get("score")),"compare",0.6))
    conf = _weighted_confidence(signals)
    v_score,v_ready,v_missing = _verification_readiness(contract)
    b_score,b_ready,b_missing = _rollback_readiness(contract)
    trust = _trust_tier(sec)
    known_risks = {RISK_TIER_0,RISK_TIER_1,RISK_TIER_2,RISK_TIER_3,RISK_TIER_4}
    known_robot_risks = {"TIER_ROBOT_OBSERVE_ONLY","TIER_ROBOT_FACE_EXPRESSION","TIER_ROBOT_HEAD_MOVEMENT",
        "TIER_ROBOT_POSTURE","TIER_ROBOT_ARM_LOW_FORCE","TIER_ROBOT_GRIP_OBJECT","TIER_ROBOT_LOCOMOTION",
        "TIER_ROBOT_HUMAN_CONTACT","TIER_ROBOT_EMERGENCY_INTERVENTION","TIER_ROBOT_EMERGENCY_STOP"}
    mode_valid = mode in {MODE_SIMULATE, MODE_DRAFT, MODE_APPLY, MODE_ROLLBACK}
    risk_valid = risk in known_risks | known_robot_risks
    effective_risk = RISK_TIER_4 if risk in known_robot_risks else risk
    threshold = _threshold_for(mode if mode_valid else MODE_APPLY, effective_risk if risk_valid else RISK_TIER_4)
    if robot["is_robotic"]:
        threshold = max(threshold, 0.90 if mode == MODE_APPLY else 0.72)
    score = _clamp01(conf + v_score * 0.30 + b_score * 0.10 + _trust_bonus(trust) + _mode_bonus(mode) - _risk_weight(effective_risk) * 0.12)
    review = AssuranceReview(review_id="asg_"+uuid.uuid4().hex,ts=_now_iso(),decision=DECISION_DENY,
        allow=False,require_user=False,action_type=action_type,capability_name=capability,
        execution_mode=mode,risk_level=risk,confidence=conf,assurance_score=score,threshold=threshold,
        verification_ready=v_ready,rollback_ready=b_ready,policy_snapshot=policy,
        meta={"verification_score":v_score,"rollback_score":b_score,"trust_tier":trust,
              "risk_penalty":_risk_weight(effective_risk),"robotic_body_profile":robot,
              "evidence_freshness":freshness,"controller_continuity":controller,
              "execution_authority":False})
    hard = []; user = []; simulation = []
    def fail(key: str, reason: str, missing: Optional[str] = None) -> None:
        if key not in review.risk_factors: review.risk_factors.append(key)
        hard.append(reason)
        if missing and missing not in review.missing_requirements: review.missing_requirements.append(missing)
    # Explicit upstream DENY always wins and is never overwritten.
    for name,owner in (("governance",gov),("security",sec)):
        if owner.get("allow") is False or _safe_lower(owner.get("decision")) in {"deny","defer","quarantine","block"}:
            fail(name+"_block_present", name+" owner blocked this request")
    if mode in {MODE_APPLY,MODE_ROLLBACK} and (config is None or _SafetyPolicies is None):
        fail("safety_policies_unavailable","Critical runtime policy dependencies are missing")
    if not mode_valid: fail("invalid_execution_mode","Execution mode is not recognized","execution_mode")
    if not risk_valid: fail("invalid_risk_tier","Risk tier is not recognized","risk_level")
    if not action_type.strip() or action_type=="unknown": fail("action_unspecified","Action type is missing","action_type")
    if not capability.strip() or capability=="unknown": fail("capability_unspecified","Capability is missing","capability_name")
    if trust==TRUST_TIER_QUARANTINED: fail("quarantined_trust_tier","Quarantined caller cannot be assured")
    if mode in {MODE_APPLY,MODE_ROLLBACK}:
        if not _authoritative_allow(gov): fail("governance_approval_unverified","Positive governance review is absent or ambiguous","governance_allow")
        if not _authoritative_allow(sec): fail("security_approval_unverified","Positive security review is absent or ambiguous","security_allow")
        if not v_ready: fail("verification_plan_insufficient","Executable action requires meaningful checks and an expected result","verification_checks")
        if not b_ready: fail("rollback_plan_insufficient","Executable action lacks a usable recovery plan","rollback_plan")
        if not any(sig.name not in {"governance_risk_inverse","security_risk_inverse","controller_continuity"} for sig in signals):
            fail("confidence_evidence_missing","No independent confidence/freshness signals are available","confidence_evidence")
    elif not _authoritative_allow(gov) or not _authoritative_allow(sec):
        simulation.append("An unverified governance chain supports observation/draft only")
    # Invalid numeric fields are hard errors, not a high-confidence absence.
    for origin,rec in (("contract",contract),("metadata",_as_mapping(contract.get("metadata"))),
                       ("governance",gov),("security",sec)):
        for key in (("confidence","intent_confidence","advcu_confidence","neuron_confidence","plan_confidence","executor_confidence","verification_confidence") if origin in {"contract","metadata"} else ("confidence","risk_score")):
            if key in rec and not _valid_numeric_signal(rec[key], percentage=(key=="risk_score")):
                fail("invalid_"+origin+"_"+key,"Malformed or non-finite confidence/risk evidence",key)
    confirmed = _has_explicit_confirmation(contract,gov,sec)
    needs_confirm = _strict_true(contract.get("requires_confirmation"))
    needs_confirm |= mode == MODE_APPLY and effective_risk==RISK_TIER_3 and policy["require_confirmation_for_privileged"]
    needs_confirm |= mode == MODE_APPLY and effective_risk==RISK_TIER_4 and policy["require_confirmation_for_network_exposure"]
    needs_confirm |= mode == MODE_APPLY and robot["is_robotic"] and policy["require_confirmation_for_physical_control"]
    if needs_confirm and not confirmed:
        user.append("Confirmation from an approved governance owner is required")
        review.risk_factors.append("verified_confirmation_missing")
        review.missing_requirements.append("verified_operator_approval")
    meta = _as_mapping(contract.get("metadata"))
    if _strict_true(contract.get("requires_fresh_evidence")) or _strict_true(meta.get("requires_fresh_evidence")) or robot["motion_requested"]:
        if not freshness.get("known") or not freshness.get("fresh"):
            fail("evidence_stale_or_unverified","Fresh perception/evidence witness is absent or stale","fresh_evidence_witness")
    if controller.get("known") and not controller["valid"]:
        fail("controller_continuity_failed","Controller lease is incomplete, expired or mismatched","verified_authority_lease")
    if _strict_true(contract.get("authority_lease_required")) or _strict_true(meta.get("authority_lease_required")) or robot["motion_requested"]:
        if not controller["valid"] or not (_strict_true(sec.get("authority_lease_verified")) or _strict_true(gov.get("authority_lease_verified"))):
            fail("authority_lease_unverified","Control lease lacks verified current ownership","verified_authority_lease")
    if _strict_true(contract.get("safe_stop_required")) or _strict_true(meta.get("safe_stop_required")) or robot["motion_requested"]:
        if not (_strict_true(contract.get("safe_stop_available")) or _strict_true(meta.get("safe_stop_available"))):
            fail("safe_stop_unavailable","Required safe-stop witness is absent","safe_stop_available")
    if robot["is_robotic"]:
        review.constraints.update({"smget_required":True,"operatorcore_required":True,
                                   "security_required":True,"safe_stop_required":True})
        if mode==MODE_APPLY:
            if not robot["safe_stop_available"] or not _strict_true(sec.get("safe_stop_verified")):
                fail("robotic_safe_stop_unverified","Physical safe stop is not verified by SecurityGovernor","safe_stop_verified")
            if robot["motion_requested"] and not robot["force_speed_limits_present"]:
                fail("robotic_limits_missing","Physical motion requires force/speed/torque limits","safety_envelope")
            if not robot["body_part_driver_verified"] or not _strict_true(sec.get("body_part_driver_verified")):
                fail("robotic_driver_unverified","Physical body driver requires independent verification","body_part_driver_verified")
            if robot["motion_requested"] and not _strict_true(sec.get("perception_witness_verified")):
                fail("robotic_perception_unverified","Physical motion requires verified fresh perception","perception_witness_verified")
    smugcc_check=meta.get("smugcc_validation")
    if isinstance(smugcc_check,dict) and smugcc_check.get("ok") is not True:
        fail("smugcc_contract_invalid","SMUGCC envelope validation failed","smugcc_validation")
    if policy["safe_mode"] and mode==MODE_APPLY and effective_risk in {RISK_TIER_3,RISK_TIER_4}:
        fail("safe_mode_high_risk_apply","Safe Mode prohibits high-risk apply")
    if policy["local_only"] and mode==MODE_APPLY and ("network" in _safe_lower(capability) or "remote" in _safe_lower(action_type)) and effective_risk==RISK_TIER_4:
        fail("local_only_remote_action","Local-only mode blocks remote apply")
    if trust in {TRUST_TIER_UNVERIFIED,TRUST_TIER_UNKNOWN} and mode==MODE_APPLY and effective_risk in {RISK_TIER_3,RISK_TIER_4}:
        fail("untrusted_high_risk","High-risk apply requires verified caller identity")
    if _seems_hazardous(contract) and conf<0.55 and mode==MODE_APPLY:
        fail("hazardous_low_confidence","Hazardous action lacks independent confidence evidence")
    if score<threshold:
        review.risk_factors.append("assurance_below_threshold")
        (hard if mode in {MODE_APPLY,MODE_ROLLBACK} else simulation).append("Assurance score below risk-specific threshold")
    # Decision lattice is monotonic: DENY > REQUIRE_USER > SIMULATE_ONLY > ALLOW.
    if hard:
        review.decision=DECISION_DENY
        review.reasons.extend(hard)
    elif user:
        review.decision=DECISION_REQUIRE_USER; review.require_user=True
        review.reasons.extend(user)
    elif simulation:
        review.decision=DECISION_SIMULATE_ONLY; review.reasons.extend(simulation)
        review.constraints["allowed_execution_mode"]=MODE_SIMULATE
    else:
        review.decision=DECISION_ALLOW; review.allow=True
        review.constraints.update({"verify_before_success":True,"audit_required":True})
        if mode in {MODE_APPLY,MODE_ROLLBACK}:
            review.constraints["postcondition_verification_required"]=True
        review.reasons.append("Assurance prerequisites and current authorization evidence satisfied; downstream execution authority remains required")
    if not review.allow and mode!=MODE_APPLY:
        review.constraints.setdefault("allowed_execution_mode",MODE_SIMULATE)
    review.signals=[asdict(x) for x in signals[:64]]
    # Persistence cannot be the authority. However missing audit durability must
    # prevent apply/rollback from returning ALLOW.
    audit_ok = _audit_log_review(review)
    review.meta["audit_persisted"] = bool(audit_ok)
    if not audit_ok and review.allow and mode in {MODE_APPLY,MODE_ROLLBACK}:
        review.allow=False; review.decision=DECISION_DENY
        review.risk_factors.append("assurance_audit_unavailable")
        review.reasons.append("Assurance audit storage failed: execution cannot be assured")
    _write_snapshot(review)
    result = review.to_dict()
    result["execution_authority"] = False
    return result


# ---------------------------------------------------------------------------
# Compatibility aliases for OperatorCore
# ---------------------------------------------------------------------------
def review_action_assurance(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action_assurance(action_contract, governance, security)


def review_smugcc_assurance(envelope: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        import importlib
        smugcc = importlib.import_module("SarahMemorySMUGCC")
        validation = smugcc.validate_smugcc_envelope(envelope)
        contract = smugcc.smugcc_to_action_contract_dict(envelope)
        # SMUGCC's external vocabulary is low/medium/high/critical, not SMGET's
        # internal risk tiers. Normalize only inside this validated adapter.
        risk_aliases={"low":RISK_TIER_2,"medium":RISK_TIER_3,
                      "high":RISK_TIER_4,"critical":RISK_TIER_4}
        supplied_risk=_safe_lower(contract.get("risk_level"))
        if supplied_risk in risk_aliases:
            contract["risk_level"]=risk_aliases[supplied_risk]
        metadata = _as_mapping(contract.get("metadata"))
        metadata["smugcc_validation"] = validation
        contract["metadata"] = metadata
        review = evaluate_action_assurance(contract,governance,security)
        review["smugcc_contract"] = _redact_bound(contract)
        review["execution_authority"] = False
        return review
    except Exception as exc:
        return {"allow":False,"decision":DECISION_DENY,"execution_authority":False,
                "verification_ready":False,"rollback_ready":False,
                "reasons":["SMUGCC validation or contract conversion unavailable"],"error_type":type(exc).__name__}



def assure_action(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action_assurance(action_contract, governance, security)


# ---------------------------------------------------------------------------
# Optional helpers for diagnostics / later wiring
# ---------------------------------------------------------------------------
def get_assurance_snapshot() -> Dict[str, Any]:
    try:
        path = _snapshot_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}



def get_assurance_profile() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "thresholds": {
            MODE_SIMULATE: {
                RISK_TIER_0: 0.10,
                RISK_TIER_1: 0.15,
                RISK_TIER_2: 0.25,
                RISK_TIER_3: 0.35,
                RISK_TIER_4: 0.45,
            },
            MODE_DRAFT: {
                RISK_TIER_0: 0.20,
                RISK_TIER_1: 0.28,
                RISK_TIER_2: 0.42,
                RISK_TIER_3: 0.58,
                RISK_TIER_4: 0.70,
            },
            MODE_APPLY: {
                RISK_TIER_0: 0.45,
                RISK_TIER_1: 0.55,
                RISK_TIER_2: 0.70,
                RISK_TIER_3: 0.84,
                RISK_TIER_4: 0.92,
            },
            MODE_ROLLBACK: {
                RISK_TIER_0: 0.30,
                RISK_TIER_1: 0.38,
                RISK_TIER_2: 0.52,
                RISK_TIER_3: 0.60,
                RISK_TIER_4: 0.70,
            },
        },
        "signals_considered": [
            "confidence",
            "intent_confidence",
            "advcu_confidence",
            "neuron_confidence",
            "plan_confidence",
            "executor_confidence",
            "verification_confidence",
            "governance_confidence",
            "governance_risk_inverse",
            "security_confidence",
            "security_risk_inverse",
        ],
        "doctrine": {
            "no_hallucinated_actuation": True,
            "verification_required_for_apply": True,
            "rollback_required_for_nontrivial_apply": True,
            "riskier_actions_require_stronger_assurance": True,
            "robotic_motion_requires_fresh_perception": True,
            "robotic_apply_requires_safe_stop": True,
            "robotic_apply_requires_force_speed_limits": True,
            "robotic_apply_requires_verified_body_driver": True,
            "fail_closed_on_low_assurance": True,
        },
    }


if __name__ == "__main__":
    demo_contract = {
        "action_type": "open_app_and_write_summary",
        "capability_name": "software.word.write",
        "execution_mode": MODE_APPLY,
        "risk_level": RISK_TIER_2,
        "requires_confirmation": False,
        "expected_result": "Word is opened and a summary is written into a document.",
        "verification_checks": [
            {"name": "word_running", "required": True},
            {"name": "document_written", "required": True},
        ],
        "rollback_plan": [
            {"name": "close_unsaved_document", "required": False},
        ],
        "metadata": {
            "advcu_confidence": 0.86,
            "neuron_confidence": 0.82,
        },
    }
    demo_governance = {"decision": "ALLOW", "allow": True, "risk_score": 30}
    demo_security = {
        "decision": "ALLOW",
        "allow": True,
        "risk_score": 30,
        "caller_identity": {"caller_id": "core:demo", "trust_tier": TRUST_TIER_CORE},
    }
    print(json.dumps(evaluate_action_assurance(demo_contract, demo_governance, demo_security), indent=2, ensure_ascii=False))

# -----------------------------------------------------------------------------
# SARAH_REM_ASSURANCE_V1
# REM Sleep assurance gate. Requires sandbox pass + rollback/promotion readiness.
# -----------------------------------------------------------------------------
def evaluate_rem_candidate_assurance(candidate: Dict[str, Any], *, snapshot: Optional[Dict[str, Any]] = None, sandbox: Optional[Dict[str, Any]] = None, governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assure a REM candidate for non-executing staging; never authorize source mutation."""
    c, sb, gov = _as_mapping(candidate), _as_mapping(sandbox), _as_mapping(governance)
    proposed = _as_mapping(c.get("proposed_action"))
    targets = c.get("target_files") if isinstance(c.get("target_files"),list) else []
    names = [str(x).replace("\\","/").split("/")[-1].lower() for x in targets]
    protected = {"sarahmemoryglobals.py","sarahmemoryarile.py","sarahmemorysecuritygovernor.py",
                 "sarahmemoryassurancegate.py","sarahmemorycompare.py","sarahmemorycognitivecompass.py",
                 "sarahmemorycognitiveservices.py","sarahmemoryoperatorcore.py","sarahmemorymsdc.py"}
    hits = sorted(set(names)&protected)
    reasons=[]; missing=[]; factors=[]
    if not c or not proposed:
        missing.append("candidate_proposed_action")
    if not _authoritative_allow(gov):
        missing.append("governance_allow")
    if sb.get("passed") is not True or sb.get("sandboxed") is False:
        missing.append("verified_sandbox_pass")
    # A sandbox boolean alone is not an independently authenticated receipt.
    sandbox_evidence = bool(sb.get("audit_id") or sb.get("receipt_id") or sb.get("evidence_hash") or sb.get("test_results"))
    if not sandbox_evidence:
        missing.append("sandbox_evidence_receipt")
    if hits:
        factors.append("protected_core_file_targeted");reasons.append("Protected security/core file cannot be changed through autonomous REM")
    proposed_type = str(proposed.get("type") or "").strip().lower()
    passive = proposed_type in {"metadata_only","research_only","analysis_only"}
    rollback_ready=bool(sb.get("rollback_ready")) or passive
    if not passive and not rollback_ready:
        missing.append("rollback_ready")
    allow=not (missing or factors)
    if allow: reasons.append("Candidate eligible for bounded staging/evidence review only; no execution authority granted")
    else: reasons.append("REM candidate lacks independently reviewable assurance prerequisites")
    return {"review_id":"rem-assure-"+uuid.uuid4().hex[:12],"ts":_now_iso(),
        "decision":DECISION_ALLOW if allow else DECISION_DENY,"allow":allow,
        "execution_authority":False,"confidence":0.0,"assurance_score":0.0,
        "threshold":0.75,"verification_ready":bool(sb.get("passed") is True and sandbox_evidence),
        "rollback_ready":rollback_ready,"reasons":reasons,"risk_factors":factors,
        "missing_requirements":missing,"protected_files":sorted(protected),
        "constraints":{"promotion_authority":False,"staging_only":True}}

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def assure_tri_layer_packet(packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assurance review for packet completeness; does not approve execution."""
    pkt = packet if isinstance(packet, dict) else {}
    missing = []
    for k in ("language_context_packet", "emotion_affect_packet", "identity_packet"):
        if not isinstance(pkt.get(k), dict):
            missing.append(k)
    return {
        "ok": not missing,
        "decision": "PACKET_COMPLETE" if not missing else "PACKET_INCOMPLETE",
        "missing": missing,
        "execution_authority": False,
    }

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Assurance checks for sovereign adapter/broker packets. No execution.


def assure_interop_broker_request(envelope: Optional[Dict[str, Any]] = None, governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Passive interoperability intake review; never approves execution or external authority."""
    env,gov,sec=_as_mapping(envelope),_as_mapping(governance),_as_mapping(security)
    kind=_safe_lower(env.get("message_type") or env.get("type") or env.get("action_type")) or "unknown"
    protocol=_safe_lower(env.get("protocol") or env.get("adapter")) or "unknown"
    remote=_strict_true(env.get("remote")) or _strict_true(env.get("is_remote")) or bool(env.get("remote_origin"))
    bidirectional=_strict_true(env.get("bidirectional")) or _safe_lower(env.get("direction"))=="bidirectional"
    payload_ok=isinstance(env.get("payload"),dict) or bool(env.get("content") or env.get("manifest") or env.get("schema"))
    missing=[];reasons=[]
    if protocol not in {"mcp","a2a","ag-ui","local"}:missing.append("recognized_protocol")
    if kind=="unknown":missing.append("message_type")
    if not payload_ok:missing.append("payload_or_manifest")
    denied = (gov.get("allow") is False or _safe_lower(gov.get("decision")) in {"deny","defer","quarantine"}
              or sec.get("allow") is False or _safe_lower(sec.get("decision")) in {"deny","quarantine"})
    execution_like=kind in {"execute","tool_call","command","driver_action","robot_motion","filesystem_write"}
    score=_clamp01((0.45 if payload_ok else 0.0)+(0.10 if kind!="unknown" else 0)+(0.10 if protocol in {"mcp","a2a","ag-ui","local"} else 0)+(0.05 if not remote else 0))
    threshold=0.70 if remote else 0.60
    if denied:
        decision=DECISION_DENY;reasons.append("Upstream governance/security rejected broker evidence")
    elif execution_like:
        decision=DECISION_SIMULATE_ONLY;reasons.append("Execution-like message is not passive evidence")
    elif bidirectional:
        decision=DECISION_REQUIRE_USER;reasons.append("Bidirectional interop requires governance review")
    elif missing or score<threshold:
        decision=DECISION_REQUIRE_USER;reasons.append("Interop evidence is incomplete or below threshold")
    elif remote and not (_authoritative_allow(gov) and _authoritative_allow(sec)):
        decision=DECISION_REQUIRE_USER;reasons.append("Remote evidence requires explicit governance and security review")
    else:
        decision=DECISION_ALLOW;reasons.append("Passive one-way evidence intake only; no execution authority")
    allow=decision==DECISION_ALLOW
    return {"ok":allow,"review_id":uuid.uuid4().hex,"decision":decision,"allow":allow,
        "require_user":decision==DECISION_REQUIRE_USER,"protocol":protocol,"message_type":kind,
        "risk_level":str(env.get("risk_level") or RISK_TIER_2),"assurance_score":score,"threshold":threshold,
        "verification_ready":bool(payload_ok and kind!="unknown"),"rollback_ready":True,
        "missing_requirements":missing,"reasons":reasons,"execution_authority":False,
        "constraints":{"direct_execution_allowed":False,"passive_evidence_only":True,"one_way_broker":True}}
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemoryAssuranceGate.py v9.1.0a
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAssuranceGate',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Governance',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['governance', 'governance_validation'],
    "supported_missions": ['Conversation', 'Execution', 'Governance', 'Security'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω060'],
    "required_authority": ['Read', 'Research'],
    "priority": 90,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAssuranceGate.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Report observed readiness without fabricating flawless operational metrics."""
    try:
        db_exists=os.path.isfile(_assurance_db_path())
        policy_ok=_SafetyPolicies is not None
        return {"status":"DEGRADED" if not policy_ok else "UNVERIFIED",
            "availability":1.0,"integrity":None,"performance":None,"reliability":None,
            "confidence":0.0,"latency_ms":None,"stability":None,"compatibility":None,
            "notes":["Availability measures import only, not execution assurance",
                     "audit_database_present="+str(db_exists),"safety_policy_module_present="+str(policy_ok)]}
    except Exception:
        return {"status":"UNKNOWN","availability":None,"integrity":None,"performance":None,
                "reliability":None,"confidence":0.0,"latency_ms":None,"stability":None,"compatibility":None,"notes":["health_check_failed"]}


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAssuranceGate',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAssuranceGate', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

# --- SML ASSURANCE SPECIALIZATION START ---
def sml_assurance_review(packet, note="Assurance reviewed SML packet"):
    """Record assurance validation in the SML packet."""
    from SarahMemorySMLProtocol import sml_touch_packet
    return sml_touch_packet(packet, organ="SarahMemoryAssuranceGate", action="assurance", omega="Ω050", note=note)
# --- SML ASSURANCE SPECIALIZATION END ---
