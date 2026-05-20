"""--==The SarahMemory Project==--
File: SarahMemoryMSDC.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

Motor, Servo, Device Controller (MSDC)
======================================
Purpose:
- Brainstem / motor-function and device-manager organ for SarahMemory AiOS.
- Maps body parts to governed driver packages without granting itself authority.
- Provides court-grade read-only witness packets for devices/body parts.
- Dispatches bounded device operations only when an upstream governance/SMGET
  contract or explicit user authority is present.

Doctrine:
- Cognition decides. SMGET authorizes. OperatorCore contracts. MSDC moves.
- MSDC never self-authorizes physical/device actions.
- Discovery is not activation. Driver presence is not runtime authority.
- User control remains final authority, especially for cameras, servos, robots,
  vehicles, forklifts, and other physical devices.
"""
from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "motor_device_controller"
# CATEGORY = "device_manager_and_motor_function"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "msdc"
# HARDWARE_DOMAIN = "camera_usb_servo_motor_device_driver"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "motor_servo_device_controller"
# FAMILY = "smget"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Brainstem motor-function/device-manager organ. Maps body parts to governed drivers and dispatches only approved operations. No self-authorization."
# --- SARAHMETA END ---

import base64
import importlib.util
import json
import logging
import os
import time
import traceback
import uuid
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    import SarahMemoryCognitiveServices as _CogServices  # type: ignore
except Exception:
    _CogServices = None

try:
    import SarahMemorySecurityGovernor as _SecurityGovernor  # type: ignore
except Exception:
    _SecurityGovernor = None

try:
    import SarahMemoryAssuranceGate as _AssuranceGate  # type: ignore
except Exception:
    _AssuranceGate = None

try:
    import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
except Exception:
    _OperatorCore = None

try:
    import SarahMemoryTrustRegistry as _TrustRegistry  # type: ignore
except Exception:
    _TrustRegistry = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None

try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None

logger = logging.getLogger("SarahMemoryMSDC")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False

_VISION_CACHE_LOCK = threading.RLock()
_VISION_OBSERVATION_DEDUPE: Dict[str, float] = {}

MODULE_NAME = "SarahMemoryMSDC"
MODULE_VERSION = "8.0.0"
CAMERA_DRIVER_ID = "com.softdev0.camera.uvc.usb"
USBHOST_DRIVER_ID = "com.softdev0.boot.usbhost"

READ_ONLY_ACTIONS = {
    "status", "get_status", "discover", "discover_devices", "list_devices", "scan",
    "select_device", "probe", "probe_capabilities", "capabilities", "enumerate_controls",
    "list_controls", "controls", "get_control", "get_property", "get_config", "ping",
    "describe_capabilities", "probe_backends", "court_witness", "body_map",
}

PRIVACY_SENSITIVE_ACTIONS = {
    "open_stream", "open", "start_capture", "read_frame", "grab_frame", "frame_info",
    "save_snapshot", "snapshot", "capture_photo", "capture_frame_b64", "snapshot_b64",
}

STATE_CHANGING_ACTIONS = {
    "open_stream", "open", "start_capture", "close_stream", "close", "stop_capture",
    "save_snapshot", "snapshot", "capture_photo", "start_recording", "record_video_start",
    "record_frame", "write_record_frame", "stop_recording", "record_video_stop",
    "start_audio_recording", "record_audio_start", "stop_audio_recording", "record_audio_stop",
    "set_control", "set_property", "ptz", "ptz_action", "vendor_passthrough", "extension_unit",
    "xu", "update_config", "set_config", "safe_stop", "stop",
}

@dataclass
class DeviceCapabilityRecord:
    body_part: str
    device_class: str
    driver_id: str
    transport: str = "unknown"
    risk_tier: str = "TIER_2_BOUNDED_LOCAL_OPERATION"
    privacy_sensitive: bool = False
    physical_safety_sensitive: bool = False
    user_control_required: bool = True
    driver_present: bool = False
    manifest_valid: bool = False
    actions: List[str] = field(default_factory=list)
    status: str = "unknown"
    evidence: Dict[str, Any] = field(default_factory=dict)


def _base_dir() -> Path:
    try:
        return Path(str(getattr(config, "BASE_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return Path.cwd().resolve()


def _data_dir() -> Path:
    try:
        return Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return (_base_dir() / "data").resolve()


def _drivers_dir() -> Path:
    try:
        return Path(str(getattr(config, "DRIVERS_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return (_data_dir() / "drivers").resolve()


def _boot_drivers_dir() -> Path:
    return (_data_dir() / "boot" / "drivers").resolve()


def _registry_dir() -> Path:
    return (_data_dir() / "registry").resolve()


def _body_map_path() -> Path:
    return _registry_dir() / "body_map.json"


def _vision_policy_path() -> Path:
    return _registry_dir() / "vision_policy.json"


def _vision_memory_dir() -> Path:
    return (_data_dir() / "vision" / "msdc_cache").resolve()


def _vision_observations_dir() -> Path:
    return (_vision_memory_dir() / "observations").resolve()


def _vision_patterns_path() -> Path:
    return _vision_memory_dir() / "visual_patterns.json"


def _vision_snapshots_dir() -> Path:
    return (_vision_memory_dir() / "snapshots").resolve()


def _safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        if path.exists() and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return deepcopy(default)


def _safe_json_write(path: Path, payload: Any) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning("Failed to write %s: %s", path, exc)
        return False


def _driver_root_for(driver_id: str) -> Path:
    driver_id = str(driver_id or "").strip()
    if driver_id.startswith("com.softdev0.boot."):
        return _boot_drivers_dir() / driver_id
    return _drivers_dir() / driver_id


def _read_driver_manifest(driver_id: str) -> Dict[str, Any]:
    path = _driver_root_for(driver_id) / "manifest.json"
    data = _safe_json_load(path, default={})
    return data if isinstance(data, dict) else {}


def _read_driver_config(driver_id: str) -> Dict[str, Any]:
    root = _driver_root_for(driver_id)
    config_data = _safe_json_load(root / "config.json", default={})
    if isinstance(config_data, dict) and config_data:
        return config_data
    defaults = _safe_json_load(root / "defaults.json", default={})
    return defaults if isinstance(defaults, dict) else {}


def _driver_file(driver_id: str) -> Path:
    return _driver_root_for(driver_id) / "driver.py"


def _load_driver_module(driver_id: str) -> Tuple[Optional[Any], Optional[str]]:
    path = _driver_file(driver_id)
    if not path.exists():
        return None, f"driver_file_missing:{path}"
    try:
        module_name = f"_sarahmemory_msdc_driver_{driver_id.replace('.', '_')}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if not spec or not spec.loader:
            return None, "spec_load_failed"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod, None
    except Exception as exc:
        return None, f"driver_import_failed:{exc}"


def _action_list_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    raw = manifest.get("actions_callable") or manifest.get("supported_actions") or manifest.get("actions") or []
    if isinstance(raw, dict):
        raw = list(raw.keys())
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _action_risk(action_id: str) -> str:
    action = str(action_id or "").strip().lower()
    if action in READ_ONLY_ACTIONS:
        return "TIER_0_INFO"
    if action in PRIVACY_SENSITIVE_ACTIONS:
        return "TIER_2_BOUNDED_LOCAL_OPERATION_PRIVACY"
    if action in STATE_CHANGING_ACTIONS:
        if action in {"ptz", "ptz_action", "set_control", "set_property", "vendor_passthrough", "extension_unit", "xu"}:
            return "TIER_3_DEVICE_CONTROL"
        return "TIER_2_BOUNDED_LOCAL_OPERATION"
    return "TIER_2_BOUNDED_LOCAL_OPERATION"


def _has_governance_authority(action_id: str, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Return whether MSDC may dispatch a non-read-only action.

    This is not a substitute for CognitiveServices/SMGET. It is a local hard stop
    to prevent accidental execution when no upstream contract is visible.
    """
    action = str(action_id or "").strip().lower()
    context = context if isinstance(context, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    if action in READ_ONLY_ACTIONS:
        return True, "read_only"
    if bool(payload.get("user_authorized")) or bool(payload.get("user_confirmed")) or bool(payload.get("user_consent")):
        return True, "explicit_user_authority"
    contract = context.get("action_contract") or payload.get("action_contract") or context.get("operator_contract") or payload.get("operator_contract")
    if isinstance(contract, dict):
        decision = str(contract.get("decision") or contract.get("authorization") or contract.get("status") or "").upper()
        if decision in {"ALLOW", "AUTHORIZED", "APPROVED", "VERIFIED"}:
            return True, "operator_contract_authorized"
    gov = context.get("governor") or payload.get("governor")
    if isinstance(gov, dict):
        decision = str(gov.get("decision") or "").upper()
        if decision == "ALLOW" and bool(gov.get("user_present", True)):
            return True, "governor_allow_user_present"
    return False, "missing_user_or_operator_authority"


def _call_governance_probe(action_id: str, driver_id: str, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort governance trace. Does not override the local hard stop."""
    trace: Dict[str, Any] = {
        "cognitive_services_available": _CogServices is not None,
        "security_governor_available": _SecurityGovernor is not None,
        "assurance_gate_available": _AssuranceGate is not None,
        "operator_core_available": _OperatorCore is not None,
        "trust_registry_available": _TrustRegistry is not None,
        "safety_policies_available": _SafetyPolicies is not None,
        "compare_available": _Compare is not None,
    }
    try:
        if _CogServices is not None and hasattr(_CogServices, "govern_request"):
            req = f"MSDC driver action {driver_id}:{action_id}"
            gov = _CogServices.govern_request(  # type: ignore[attr-defined]
                req,
                caller="SarahMemoryMSDC",
                caller_context={"driver_id": driver_id, "action_id": action_id, "payload_keys": sorted(payload.keys()), "context": context},
                user_present=bool(payload.get("user_authorized") or payload.get("user_confirmed") or context.get("user_present", False)),
                user_consented=bool(payload.get("user_authorized") or payload.get("user_confirmed") or payload.get("user_consent")),
                proposed_action={"driver_id": driver_id, "action_id": action_id, "risk_tier": _action_risk(action_id)},
            )
            if isinstance(gov, dict):
                trace["cognitive_services"] = {
                    "decision": gov.get("decision"),
                    "allow": bool(gov.get("allow")),
                    "require_user": bool(gov.get("require_user")),
                    "reasons": gov.get("reasons") if isinstance(gov.get("reasons"), list) else [],
                }
    except Exception as exc:
        trace["cognitive_services_error"] = str(exc)
    return trace




def _cfg_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default)) if config is not None else bool(default)
    except Exception:
        return bool(default)


def _cfg_int(name: str, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        value = int(getattr(config, name, default)) if config is not None else int(default)
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _utc_now() -> datetime:
    return datetime.utcnow()


def _utc_now_iso() -> str:
    return _utc_now().isoformat() + "Z"


def _iso_to_epoch(value: Any) -> float:
    try:
        text = str(value or "").strip()
        if not text:
            return 0.0
        if text.endswith("Z"):
            text = text[:-1]
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def _canonical_json_hash(payload: Any) -> str:
    try:
        import hashlib
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return str(uuid.uuid4())


def _limit_text(value: Any, limit: int = 512) -> str:
    try:
        text = str(value or "")
        return text[:limit]
    except Exception:
        return ""


def msdc_get_vision_governance_profile() -> Dict[str, Any]:
    """Return the constitutional vision-retention and pattern-learning profile.

    This is read-only policy plumbing. SarahMemoryGlobals.py owns the flags.
    MSDC only reports and obeys them.
    """
    retention_days = _cfg_int("VISION_TEMP_CACHE_DAYS", 7, minimum=1, maximum=30)
    profile = {
        "ok": True,
        "source": MODULE_NAME,
        "schema": "SarahMemoryMSDC.vision_governance.v1",
        "generated_at": _utc_now_iso(),
        "flags": {
            "VISION_OBSERVATION_LOGGING_ENABLED": _cfg_bool("VISION_OBSERVATION_LOGGING_ENABLED", False),
            "VISION_TEMP_CACHE_ENABLED": _cfg_bool("VISION_TEMP_CACHE_ENABLED", False),
            "VISION_TEMP_CACHE_DAYS": retention_days,
            "VISION_PATTERN_LEARNING_ENABLED": _cfg_bool("VISION_PATTERN_LEARNING_ENABLED", False),
            "VISION_PATTERN_HEALTH_WATCH_ENABLED": _cfg_bool("VISION_PATTERN_HEALTH_WATCH_ENABLED", False),
            "VISION_RAW_FRAME_RETENTION_ENABLED": _cfg_bool("VISION_RAW_FRAME_RETENTION_ENABLED", False),
            "VISION_APPROVED_LEARNING_REQUIRED": _cfg_bool("VISION_APPROVED_LEARNING_REQUIRED", True),
        },
        "capabilities": {
            "temporary_visual_observation_cache": _cfg_bool("VISION_TEMP_CACHE_ENABLED", False),
            "derived_pattern_detection": _cfg_bool("VISION_PATTERN_LEARNING_ENABLED", False),
            "health_watch_signal_only": _cfg_bool("VISION_PATTERN_HEALTH_WATCH_ENABLED", False),
            "raw_frame_retention": _cfg_bool("VISION_RAW_FRAME_RETENTION_ENABLED", False),
            "approved_learning_gate": _cfg_bool("VISION_APPROVED_LEARNING_REQUIRED", True),
            "permanent_observation_logging": _cfg_bool("VISION_OBSERVATION_LOGGING_ENABLED", False),
        },
        "retention": {
            "temporary_cache_days": retention_days,
            "raw_frames_retained_by_default": _cfg_bool("VISION_RAW_FRAME_RETENTION_ENABLED", False),
            "observation_metadata_retained": _cfg_bool("VISION_TEMP_CACHE_ENABLED", False),
            "frame_spam_allowed": False,
        },
        "paths": {
            "vision_memory_dir": str(_vision_memory_dir()),
            "observations_dir": str(_vision_observations_dir()),
            "patterns_path": str(_vision_patterns_path()),
            "snapshots_dir": str(_vision_snapshots_dir()),
        },
        "limits": {
            "no_medical_diagnosis": True,
            "no_hidden_identity_enrollment": True,
            "no_frame_by_frame_learning": True,
            "user_remains_final_authority": True,
        },
    }
    return profile


def _observation_ledger_path(dt: Optional[datetime] = None) -> Path:
    d = dt or _utc_now()
    return _vision_observations_dir() / f"visual_observations_{d.strftime('%Y%m%d')}.jsonl"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return True
    except Exception as exc:
        logger.warning("MSDC visual observation append failed: %s", exc)
        return False


def _iter_observation_paths(days: Optional[int] = None) -> List[Path]:
    retention = days or _cfg_int("VISION_TEMP_CACHE_DAYS", 7, minimum=1, maximum=30)
    root = _vision_observations_dir()
    if not root.exists():
        return []
    cutoff = time.time() - (float(retention) * 86400.0)
    out: List[Path] = []
    try:
        for path in sorted(root.glob("visual_observations_*.jsonl")):
            try:
                if path.stat().st_mtime >= cutoff:
                    out.append(path)
            except Exception:
                pass
    except Exception:
        pass
    return out


def _extract_observation_features(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stable pattern features from a SOBJE/FaceRec/appvision packet.

    This intentionally avoids storing raw images. Callers may pass richer packets;
    MSDC keeps only lightweight, bounded metadata for pattern awareness.
    """
    obs = observation if isinstance(observation, dict) else {}
    details = obs.get("details") if isinstance(obs.get("details"), dict) else {}
    scene = obs.get("scene") if isinstance(obs.get("scene"), dict) else {}

    def first_value(keys: List[str]) -> str:
        for key in keys:
            for src in (obs, details, scene):
                try:
                    val = src.get(key) if isinstance(src, dict) else None
                except Exception:
                    val = None
                if val not in (None, "", [], {}):
                    return _limit_text(val, 128).strip().lower()
        return ""

    labels: List[str] = []
    for key in ("objects", "labels", "detected_objects", "visible_objects"):
        raw = obs.get(key) if isinstance(obs, dict) else None
        if raw is None and isinstance(details, dict):
            raw = details.get(key)
        if isinstance(raw, list):
            for item in raw[:20]:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("name") or item.get("class")
                else:
                    label = item
                if label:
                    labels.append(_limit_text(label, 64).strip().lower())
        elif isinstance(raw, str):
            labels.extend([x.strip().lower() for x in raw.split(",") if x.strip()][:20])

    features = {
        "person_label": first_value(["person", "person_label", "identity", "known_person", "user_label"]),
        "shirt_color": first_value(["shirt_color", "upper_garment_color", "clothing_color", "dominant_clothing_color"]),
        "hat_state": first_value(["hat", "wearing_hat", "headwear", "headwear_state"]),
        "expression_state": first_value(["expression", "expression_state", "facial_expression", "fer_state", "mood_visible"]),
        "posture_state": first_value(["posture", "posture_state", "body_language", "visible_posture"]),
        "scene_label": first_value(["scene", "scene_label", "location_context"]),
        "held_object": first_value(["held_object", "object_in_hand", "hand_object"]),
        "object_labels": sorted(set([x for x in labels if x]))[:20],
    }
    return features


def msdc_record_visual_observation(
    observation: Dict[str, Any],
    *,
    source: str = "appvision",
    approved: bool = False,
    user_label: str = "",
    raw_frame_path: str = "",
    min_repeat_interval_seconds: int = 300,
) -> Dict[str, Any]:
    """Record one governed visual observation into the temporary MSDC cache.

    This is not frame logging. It stores lightweight derived metadata only, unless
    Globals explicitly allows raw frame retention and the caller supplies a path.
    """
    profile = msdc_get_vision_governance_profile()
    flags = profile.get("flags", {}) if isinstance(profile.get("flags"), dict) else {}
    if not bool(flags.get("VISION_TEMP_CACHE_ENABLED", False)):
        return {"ok": False, "skipped": True, "reason": "VISION_TEMP_CACHE_ENABLED_FALSE", "source": MODULE_NAME}

    if not isinstance(observation, dict):
        return {"ok": False, "error": "observation_must_be_dict", "source": MODULE_NAME}

    now = _utc_now()
    features = _extract_observation_features(observation)
    confidence = observation.get("confidence", observation.get("score", None))
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except Exception:
        confidence_value = None

    raw_retention_allowed = bool(flags.get("VISION_RAW_FRAME_RETENTION_ENABLED", False))
    approved_required = bool(flags.get("VISION_APPROVED_LEARNING_REQUIRED", True))
    permanent_logging_allowed = bool(flags.get("VISION_OBSERVATION_LOGGING_ENABLED", False))

    compact_observation = {
        "observation_id": str(uuid.uuid4()),
        "ts": now.isoformat() + "Z",
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "hour": now.hour,
        "source": _limit_text(source, 96),
        "features": features,
        "confidence": confidence_value,
        "approved": bool(approved),
        "user_label": _limit_text(user_label, 128),
        "learning_status": "approved_candidate" if approved else "temporary_observation_only",
        "permanent_logging_allowed": permanent_logging_allowed,
        "approved_learning_required": approved_required,
        "raw_frame_retained": bool(raw_retention_allowed and raw_frame_path),
        "raw_frame_path": _limit_text(raw_frame_path, 512) if raw_retention_allowed and raw_frame_path else "",
        "diagnostic_note": "metadata_only_no_frame_spam",
    }

    # Dedupe/throttle repeated same-feature observations inside one runtime.
    dedupe_material = {
        "weekday": compact_observation["weekday"],
        "hour": compact_observation["hour"],
        "source": compact_observation["source"],
        "features": features,
    }
    dedupe_key = _canonical_json_hash(dedupe_material)
    with _VISION_CACHE_LOCK:
        last_ts = float(_VISION_OBSERVATION_DEDUPE.get(dedupe_key, 0.0) or 0.0)
        now_epoch = time.time()
        if last_ts and (now_epoch - last_ts) < max(30, int(min_repeat_interval_seconds)):
            return {
                "ok": True,
                "skipped": True,
                "reason": "duplicate_observation_throttled",
                "dedupe_key": dedupe_key,
                "source": MODULE_NAME,
                "features": features,
            }
        _VISION_OBSERVATION_DEDUPE[dedupe_key] = now_epoch
        if len(_VISION_OBSERVATION_DEDUPE) > 5000:
            for k in list(_VISION_OBSERVATION_DEDUPE.keys())[:1000]:
                _VISION_OBSERVATION_DEDUPE.pop(k, None)

        msdc_prune_visual_cache()
        ok = _append_jsonl(_observation_ledger_path(now), compact_observation)

    return {
        "ok": bool(ok),
        "stored": bool(ok),
        "observation_id": compact_observation["observation_id"],
        "cache_scope": "temporary_visual_observation_cache",
        "retention_days": int(flags.get("VISION_TEMP_CACHE_DAYS", 7) or 7),
        "features": features,
        "raw_frame_retained": compact_observation["raw_frame_retained"],
        "source": MODULE_NAME,
    }


def msdc_load_visual_observations(days: Optional[int] = None, max_records: int = 1000) -> Dict[str, Any]:
    """Load recent temporary visual observations from JSONL ledgers."""
    records: List[Dict[str, Any]] = []
    retention = days or _cfg_int("VISION_TEMP_CACHE_DAYS", 7, minimum=1, maximum=30)
    cutoff = time.time() - (float(retention) * 86400.0)
    for path in _iter_observation_paths(retention):
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= max(1, int(max_records)):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if _iso_to_epoch(row.get("ts")) >= cutoff:
                        records.append(row)
        except Exception:
            continue
        if len(records) >= max(1, int(max_records)):
            break
    records.sort(key=lambda r: str(r.get("ts") or ""))
    return {"ok": True, "days": retention, "count": len(records), "records": records[-max(1, int(max_records)):]}


def msdc_prune_visual_cache(days: Optional[int] = None) -> Dict[str, Any]:
    """Delete expired temporary visual-observation cache artifacts."""
    retention = days or _cfg_int("VISION_TEMP_CACHE_DAYS", 7, minimum=1, maximum=30)
    cutoff = time.time() - (float(retention) * 86400.0)
    removed: List[str] = []
    for root in (_vision_observations_dir(), _vision_snapshots_dir()):
        try:
            if not root.exists():
                continue
            for path in root.glob("*"):
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed.append(str(path))
                except Exception:
                    pass
        except Exception:
            pass
    return {"ok": True, "retention_days": retention, "removed_count": len(removed), "removed": removed[:100], "source": MODULE_NAME}


def _pattern_key_values(record: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    weekday = str(record.get("weekday") or "unknown")
    out: List[Tuple[str, str, str]] = []
    for feature_key in (
        "person_label", "shirt_color", "hat_state", "expression_state",
        "posture_state", "scene_label", "held_object",
    ):
        value = str(features.get(feature_key) or "").strip().lower()
        if value:
            out.append((weekday, feature_key, value))
    for label in (features.get("object_labels") or [])[:20]:
        value = str(label or "").strip().lower()
        if value:
            out.append((weekday, "object_label", value))
    return out


def msdc_detect_visual_patterns(days: Optional[int] = None, min_count: int = 2) -> Dict[str, Any]:
    """Detect routine visual patterns and deviations from temporary observations.

    No medical diagnosis is produced. Health watch remains a gated signal layer.
    """
    profile = msdc_get_vision_governance_profile()
    flags = profile.get("flags", {}) if isinstance(profile.get("flags"), dict) else {}
    if not bool(flags.get("VISION_PATTERN_LEARNING_ENABLED", False)):
        return {"ok": False, "skipped": True, "reason": "VISION_PATTERN_LEARNING_ENABLED_FALSE", "source": MODULE_NAME}

    loaded = msdc_load_visual_observations(days=days, max_records=2500)
    records = loaded.get("records") if isinstance(loaded.get("records"), list) else []
    counts: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for weekday, feature_key, value in _pattern_key_values(rec):
            k = f"{weekday}::{feature_key}::{value}"
            slot = counts.setdefault(k, {"weekday": weekday, "feature": feature_key, "value": value, "count": 0, "examples": []})
            slot["count"] += 1
            if len(slot["examples"]) < 3:
                slot["examples"].append({"ts": rec.get("ts"), "observation_id": rec.get("observation_id")})

    patterns = [v for v in counts.values() if int(v.get("count", 0)) >= max(2, int(min_count))]
    patterns.sort(key=lambda x: int(x.get("count", 0)), reverse=True)

    deviations: List[Dict[str, Any]] = []
    if records:
        latest = records[-1]
        latest_values = {(w, f): value for (w, f, value) in _pattern_key_values(latest)}
        for pattern in patterns:
            key = (str(pattern.get("weekday")), str(pattern.get("feature")))
            if key in latest_values and latest_values[key] != str(pattern.get("value")):
                deviations.append({
                    "kind": "routine_deviation",
                    "weekday": pattern.get("weekday"),
                    "feature": pattern.get("feature"),
                    "expected_pattern": pattern.get("value"),
                    "observed_latest": latest_values[key],
                    "pattern_count": pattern.get("count"),
                    "latest_observation_id": latest.get("observation_id"),
                    "severity": "low" if int(pattern.get("count", 0)) < 4 else "medium",
                    "action": "ask_user_if_relevant_no_diagnosis",
                })

    result = {
        "ok": True,
        "source": MODULE_NAME,
        "schema": "SarahMemoryMSDC.visual_patterns.v1",
        "generated_at": _utc_now_iso(),
        "days": int(loaded.get("days") or flags.get("VISION_TEMP_CACHE_DAYS") or 7),
        "record_count": int(loaded.get("count") or 0),
        "patterns": patterns[:100],
        "deviations": deviations[:50],
        "health_watch_enabled": bool(flags.get("VISION_PATTERN_HEALTH_WATCH_ENABLED", False)),
        "diagnostic_limits": {
            "does_not_diagnose_health": True,
            "may_flag_observable_deviation": True,
            "requires_user_or_governance_before_escalation": True,
        },
    }
    _safe_json_write(_vision_patterns_path(), result)
    return result


def msdc_visual_pattern_witness() -> Dict[str, Any]:
    """Court-safe witness for visual pattern awareness readiness."""
    profile = msdc_get_vision_governance_profile()
    patterns = msdc_detect_visual_patterns() if profile.get("flags", {}).get("VISION_PATTERN_LEARNING_ENABLED") else {"ok": False, "skipped": True}
    return {
        "ok": True,
        "source_family": "SarahMemoryMSDC",
        "evidence_class": "visual_pattern_awareness_witness",
        "verified": bool(profile.get("flags", {}).get("VISION_TEMP_CACHE_ENABLED")),
        "governance_profile": profile,
        "pattern_summary": {
            "ok": bool(patterns.get("ok")),
            "record_count": patterns.get("record_count", 0),
            "pattern_count": len(patterns.get("patterns", []) or []),
            "deviation_count": len(patterns.get("deviations", []) or []),
            "patterns_path": str(_vision_patterns_path()),
        },
        "limits": {
            "no_medical_diagnosis": True,
            "no_frame_spam": True,
            "no_hidden_identity_enrollment": True,
            "user_final_authority": True,
        },
    }

def msdc_driver_present(driver_id: str) -> bool:
    root = _driver_root_for(driver_id)
    return bool(root.exists() and (root / "driver.py").exists() and (root / "manifest.json").exists())


def msdc_get_device_capability(body_part: str = "eyes") -> DeviceCapabilityRecord:
    body_part = str(body_part or "eyes").strip().lower()
    if body_part in {"eye", "eyes", "camera", "webcam", "vision"}:
        manifest = _read_driver_manifest(CAMERA_DRIVER_ID)
        actions = _action_list_from_manifest(manifest)
        present = msdc_driver_present(CAMERA_DRIVER_ID)
        return DeviceCapabilityRecord(
            body_part="eyes",
            device_class="camera_vision",
            driver_id=CAMERA_DRIVER_ID,
            transport=str((manifest.get("platform") or {}).get("transport") or "usb") if isinstance(manifest.get("platform"), dict) else "usb",
            risk_tier="TIER_2_BOUNDED_LOCAL_OPERATION_PRIVACY",
            privacy_sensitive=True,
            physical_safety_sensitive=False,
            user_control_required=True,
            driver_present=present,
            manifest_valid=bool(manifest.get("id") == CAMERA_DRIVER_ID and actions),
            actions=actions,
            status="driver_available" if present else "driver_missing",
            evidence={
                "manifest_name": manifest.get("name"),
                "manifest_version": manifest.get("version"),
                "permissions": manifest.get("permissions") if isinstance(manifest.get("permissions"), list) else [],
                "device_classes": (manifest.get("platform") or {}).get("device_classes") if isinstance(manifest.get("platform"), dict) else [],
                "backend_support": (manifest.get("platform") or {}).get("backend_support") if isinstance(manifest.get("platform"), dict) else [],
                "vision_governance": msdc_get_vision_governance_profile(),
            },
        )
    return DeviceCapabilityRecord(
        body_part=body_part,
        device_class="unknown",
        driver_id="",
        status="unmapped_body_part",
        user_control_required=True,
    )


def msdc_map_body(force_refresh: bool = False, persist: bool = True) -> Dict[str, Any]:
    eyes = asdict(msdc_get_device_capability("eyes"))
    usb_manifest = _read_driver_manifest(USBHOST_DRIVER_ID)
    usbhost = {
        "driver_id": USBHOST_DRIVER_ID,
        "driver_present": msdc_driver_present(USBHOST_DRIVER_ID),
        "manifest_valid": bool(usb_manifest.get("id") == USBHOST_DRIVER_ID),
        "actions": _action_list_from_manifest(usb_manifest),
        "role": "usb_host_nerve_root_support_witness",
        "support_only": True,
    }
    body_map = {
        "ok": True,
        "schema": "SarahMemoryMSDC.body_map.v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "limits": {
            "msdc_self_authorizes": False,
            "requires_smget_for_actions": True,
            "discovery_is_activation": False,
            "user_control_required_for_camera": True,
            "vision_pattern_awareness_supported": True,
            "vision_observations_are_metadata_not_frame_spam": True,
        },
        "body_parts": {"eyes": eyes},
        "support_buses": {"usb_host": usbhost},
        "vision_memory": msdc_get_vision_governance_profile(),
    }
    if persist:
        _safe_json_write(_body_map_path(), body_map)
    return body_map


def msdc_court_witness(body_part: str = "eyes", include_probe: bool = False) -> Dict[str, Any]:
    body_map = msdc_map_body(persist=True)
    eyes = body_map.get("body_parts", {}).get("eyes", {})
    witness: Dict[str, Any] = {
        "ok": True,
        "source_family": "SarahMemoryMSDC",
        "evidence_class": "motor_device_manager_witness",
        "verified": bool(eyes.get("driver_present") and eyes.get("manifest_valid")),
        "capability_class": "vision_environment",
        "body_part": body_part or "eyes",
        "body_part_record": eyes,
        "body_map_path": str(_body_map_path()),
        "limits": {
            "camera_opened": False,
            "driver_session_started": False,
            "read_only_fact_check": True,
            "msdc_self_authorizes": False,
        },
    }
    witness["visual_pattern_witness"] = msdc_visual_pattern_witness()
    if include_probe:
        witness["driver_discovery"] = msdc_driver_action(CAMERA_DRIVER_ID, "discover_devices", payload={}, context={"read_only_probe": True})
    return witness


def msdc_driver_action(driver_id: str, action_id: str, payload: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    driver_id = str(driver_id or "").strip()
    action_id = str(action_id or "").strip()
    payload = payload if isinstance(payload, dict) else {}
    context = context if isinstance(context, dict) else {}
    if not driver_id or not action_id:
        return {"ok": False, "error": "missing_driver_or_action", "source": MODULE_NAME}

    authorized, authority_reason = _has_governance_authority(action_id, context=context, payload=payload)
    gov_trace = _call_governance_probe(action_id, driver_id, payload, context)
    if not authorized:
        return {
            "ok": False,
            "error": "msdc_authority_required",
            "reason": authority_reason,
            "driver_id": driver_id,
            "action_id": action_id,
            "risk_tier": _action_risk(action_id),
            "governance_trace": gov_trace,
            "source": MODULE_NAME,
        }

    manifest = _read_driver_manifest(driver_id)
    actions = set(_action_list_from_manifest(manifest))
    action_alias_ok = action_id in actions or action_id in READ_ONLY_ACTIONS or action_id in STATE_CHANGING_ACTIONS
    if actions and not action_alias_ok:
        return {
            "ok": False,
            "error": "action_not_declared_by_driver_manifest",
            "driver_id": driver_id,
            "action_id": action_id,
            "declared_actions": sorted(actions),
            "source": MODULE_NAME,
        }

    mod, err = _load_driver_module(driver_id)
    if mod is None:
        return {"ok": False, "error": err or "driver_load_failed", "driver_id": driver_id, "action_id": action_id, "source": MODULE_NAME}

    try:
        cfg = _read_driver_config(driver_id)
        if hasattr(mod, "driver_init"):
            try:
                mod.driver_init(context={"caller": MODULE_NAME, "driver_id": driver_id}, config=cfg)  # type: ignore[attr-defined]
            except TypeError:
                try:
                    mod.driver_init(config=cfg)  # type: ignore[attr-defined]
                except TypeError:
                    mod.driver_init(cfg)  # type: ignore[attr-defined]
        if not hasattr(mod, "driver_action"):
            return {"ok": False, "error": "driver_action_missing", "driver_id": driver_id, "source": MODULE_NAME}
        try:
            result = mod.driver_action(action_id=action_id, context={"caller": MODULE_NAME, **context}, payload=payload)  # type: ignore[attr-defined]
        except TypeError:
            result = mod.driver_action(action_id, context={"caller": MODULE_NAME, **context}, payload=payload)  # type: ignore[attr-defined]
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}
        result.setdefault("driver_id", driver_id)
        result.setdefault("action_id", action_id)
        result.setdefault("source", MODULE_NAME)
        result.setdefault("risk_tier", _action_risk(action_id))
        result.setdefault("authority", authority_reason)
        return result
    except Exception as exc:
        return {
            "ok": False,
            "error": "driver_action_exception",
            "detail": str(exc),
            "traceback": traceback.format_exc(limit=4),
            "driver_id": driver_id,
            "action_id": action_id,
            "source": MODULE_NAME,
        }


def msdc_camera_discover() -> Dict[str, Any]:
    return msdc_driver_action(CAMERA_DRIVER_ID, "discover_devices", payload={}, context={"read_only_probe": True})


def msdc_camera_probe() -> Dict[str, Any]:
    return msdc_driver_action(CAMERA_DRIVER_ID, "probe_capabilities", payload={}, context={"read_only_probe": True})


def msdc_camera_open(user_authorized: bool = False, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(payload or {})
    p["user_authorized"] = bool(user_authorized or p.get("user_authorized") or p.get("user_confirmed"))
    return msdc_driver_action(CAMERA_DRIVER_ID, "open_stream", payload=p, context={"body_part": "eyes"})


def msdc_camera_close(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(payload or {})
    # Closing/stopping is safety-positive; allow explicit safe stop even without a new user prompt.
    p.setdefault("user_authorized", True)
    return msdc_driver_action(CAMERA_DRIVER_ID, "close_stream", payload=p, context={"body_part": "eyes", "safety_positive": True})


def msdc_camera_capture_b64(user_authorized: bool = False, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture a still frame through the camera driver and return base64.

    Uses existing driver save_snapshot action to avoid changing the camera driver
    contract. The generic driver intentionally does not serialize raw frames.
    """
    p = dict(payload or {})
    p["user_authorized"] = bool(user_authorized or p.get("user_authorized") or p.get("user_confirmed"))
    snap = msdc_driver_action(CAMERA_DRIVER_ID, "save_snapshot", payload=p, context={"body_part": "eyes", "capture_kind": "snapshot_b64"})
    if not snap.get("ok"):
        return snap
    path = str(snap.get("path") or "").strip()
    if not path or not os.path.exists(path):
        snap["ok"] = False
        snap["error"] = "snapshot_file_missing"
        return snap
    try:
        blob = Path(path).read_bytes()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        snap["image_b64"] = base64.b64encode(blob).decode("ascii")
        snap["data_url"] = f"data:{mime};base64,{snap['image_b64']}"
        snap["mime"] = mime
        snap["body_part"] = "eyes"
        return snap
    except Exception as exc:
        snap["ok"] = False
        snap["error"] = "snapshot_encode_failed"
        snap["detail"] = str(exc)
        return snap


def msdc_status() -> Dict[str, Any]:
    return {
        "ok": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "globals_available": config is not None,
        "data_dir": str(_data_dir()),
        "drivers_dir": str(_drivers_dir()),
        "boot_drivers_dir": str(_boot_drivers_dir()),
        "body_map_path": str(_body_map_path()),
        "vision_policy_path": str(_vision_policy_path()),
        "vision_memory_dir": str(_vision_memory_dir()),
        "vision_governance": msdc_get_vision_governance_profile(),
        "visual_pattern_witness": msdc_visual_pattern_witness(),
        "body_map": msdc_map_body(persist=False),
    }


# Backward-compatible aliases for possible future callers.
def get_device_manager_status() -> Dict[str, Any]:
    return msdc_status()


def get_body_map(force_refresh: bool = False) -> Dict[str, Any]:
    return msdc_map_body(force_refresh=force_refresh, persist=True)


def get_court_witness(body_part: str = "eyes") -> Dict[str, Any]:
    return msdc_court_witness(body_part=body_part, include_probe=False)


def get_vision_governance_profile() -> Dict[str, Any]:
    return msdc_get_vision_governance_profile()


def get_visual_pattern_witness() -> Dict[str, Any]:
    return msdc_visual_pattern_witness()


def record_visual_observation(observation: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return msdc_record_visual_observation(observation, **kwargs)


def detect_visual_patterns(days: Optional[int] = None, min_count: int = 2) -> Dict[str, Any]:
    return msdc_detect_visual_patterns(days=days, min_count=min_count)


if __name__ == "__main__":
    print(json.dumps(msdc_status(), indent=2, default=str))
