"""--==The SarahMemory Project==--
File: SarahMemoryMSDC.py
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

Motor, Servo, Device Controller (MSDC)
======================================
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
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Brainstem motor-function/device-manager organ. Maps body parts to governed drivers and dispatches only approved operations. No self-authorization."
# --- SARAHMETA END ---

import base64
import importlib.util
import json
import logging
import os
import sys
import subprocess
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

_MSDC_LIGHTWEIGHT_PROBE_IMPORT = os.getenv("SARAH_MSDC_BOUNDED_PROBE", "0").strip().lower() in ("1", "true", "yes", "on")

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
    except Exception:
        _CogServices = None
else:
    _CogServices = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemorySecurityGovernor as _SecurityGovernor  # type: ignore
    except Exception:
        _SecurityGovernor = None
else:
    _SecurityGovernor = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemoryAssuranceGate as _AssuranceGate  # type: ignore
    except Exception:
        _AssuranceGate = None
else:
    _AssuranceGate = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
    except Exception:
        _OperatorCore = None
else:
    _OperatorCore = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemoryTrustRegistry as _TrustRegistry  # type: ignore
    except Exception:
        _TrustRegistry = None
else:
    _TrustRegistry = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
    except Exception:
        _SafetyPolicies = None
else:
    _SafetyPolicies = None

if not _MSDC_LIGHTWEIGHT_PROBE_IMPORT:
    try:
        import SarahMemoryCompare as _Compare  # type: ignore
    except Exception:
        _Compare = None
else:
    _Compare = None

logger = logging.getLogger("SarahMemoryMSDC")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False

MODULE_NAME = "SarahMemoryMSDC"
MODULE_VERSION = "9.0.0"
CAMERA_DRIVER_ID = "com.softdev0.camera.uvc.usb"
USBHOST_DRIVER_ID = "com.softdev0.boot.usbhost"
VR_HEADSET_DRIVER_ID = "com.softdev0.vr.headset.usb"
DISPLAY_DRIVER_ID = "com.softdev0.vga.hdmi"

READ_ONLY_ACTIONS = {
    "status", "get_status", "discover", "discover_devices", "list_devices", "scan",
    "select_device", "probe", "probe_capabilities", "capabilities", "enumerate_controls",
    "list_controls", "controls", "get_control", "get_property", "get_config", "ping",
    "describe_capabilities", "probe_backends", "court_witness", "body_map",
    "operator_hud_status", "build_operator_hud_request", "operator_hud_surface",
    "build_hud_surface_request", "native_hmd_status", "native_headset_profile",
    "rhythm_motion_witness", "rhythm_motion_intent", "motion_intent_witness",
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
    # SM V8 Robotic Body Expansion: structured embodiment metadata.
    # These fields are descriptive contracts only; they do not authorize motion.
    embodied_role: str = "generic_body_part"
    motion_capable: bool = False
    locomotion_capable: bool = False
    manipulation_capable: bool = False
    safe_stop_action: str = "safe_stop"
    force_limit_n: Optional[float] = None
    speed_limit_mps: Optional[float] = None
    torque_limit_nm: Optional[float] = None
    requires_smget: bool = True
    requires_assurance: bool = True
    requires_local_presence: bool = True
    actuator_state: str = "not_configured"
    safety_envelope: Dict[str, Any] = field(default_factory=dict)


ROBOTIC_BODY_SCHEMA_VERSION = "SarahMemoryMSDC.robotic_body.v1"

ROBOTIC_BODY_PART_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "head": {"device_class": "robot_head_orientation", "embodied_role": "orientation_and_attention", "risk_tier": "TIER_ROBOT_HEAD_MOVEMENT", "motion_capable": True, "actions": ["look_at", "center_head", "safe_stop"], "speed_limit_mps": 0.20, "torque_limit_nm": 0.30},
    "neck": {"device_class": "robot_neck_servo", "embodied_role": "head_pose_support", "risk_tier": "TIER_ROBOT_HEAD_MOVEMENT", "motion_capable": True, "actions": ["yaw", "pitch", "center", "safe_stop"], "torque_limit_nm": 0.30},
    "ears": {"device_class": "microphone_audio", "embodied_role": "hearing", "risk_tier": "TIER_2_BOUNDED_LOCAL_OPERATION_PRIVACY", "privacy_sensitive": True, "actions": ["listen_status", "start_listen", "stop_listen"], "motion_capable": False, "physical_safety_sensitive": False},
    "mouth_voice": {"device_class": "speaker_voice_output", "embodied_role": "speech_expression", "risk_tier": "TIER_1_HARMLESS_LOCAL_UI", "actions": ["speak", "stop_speaking"], "motion_capable": False, "physical_safety_sensitive": False},
    "face_expression": {"device_class": "facial_expression_actuator", "embodied_role": "social_expression", "risk_tier": "TIER_ROBOT_FACE_EXPRESSION", "motion_capable": True, "actions": ["smile", "blink", "wink", "neutral", "safe_stop"], "torque_limit_nm": 0.05},
    "torso": {"device_class": "robot_torso_posture", "embodied_role": "posture_balance", "risk_tier": "TIER_ROBOT_POSTURE", "motion_capable": True, "actions": ["stand_posture", "sit_posture", "safe_stop"], "torque_limit_nm": 1.0},
    "left_arm": {"device_class": "robot_arm", "embodied_role": "left_reach", "risk_tier": "TIER_ROBOT_ARM_LOW_FORCE", "motion_capable": True, "actions": ["raise", "lower", "reach", "safe_stop"], "force_limit_n": 8.0, "torque_limit_nm": 1.0},
    "right_arm": {"device_class": "robot_arm", "embodied_role": "right_reach", "risk_tier": "TIER_ROBOT_ARM_LOW_FORCE", "motion_capable": True, "actions": ["raise", "lower", "reach", "safe_stop"], "force_limit_n": 8.0, "torque_limit_nm": 1.0},
    "left_hand": {"device_class": "robot_gripper", "embodied_role": "left_grip", "risk_tier": "TIER_ROBOT_GRIP_OBJECT", "motion_capable": True, "manipulation_capable": True, "actions": ["open", "close", "grip_low_force", "release", "safe_stop"], "force_limit_n": 5.0, "torque_limit_nm": 0.20},
    "right_hand": {"device_class": "robot_gripper", "embodied_role": "right_grip", "risk_tier": "TIER_ROBOT_GRIP_OBJECT", "motion_capable": True, "manipulation_capable": True, "actions": ["open", "close", "grip_low_force", "release", "safe_stop"], "force_limit_n": 5.0, "torque_limit_nm": 0.20},
    "hips": {"device_class": "robot_hip_balance", "embodied_role": "locomotion_balance", "risk_tier": "TIER_ROBOT_LOCOMOTION", "motion_capable": True, "locomotion_capable": True, "actions": ["balance_hold", "safe_stop"], "speed_limit_mps": 0.25, "torque_limit_nm": 1.5},
    "left_leg": {"device_class": "robot_leg", "embodied_role": "left_locomotion", "risk_tier": "TIER_ROBOT_LOCOMOTION", "motion_capable": True, "locomotion_capable": True, "actions": ["step", "hold", "safe_stop"], "speed_limit_mps": 0.25, "torque_limit_nm": 1.5},
    "right_leg": {"device_class": "robot_leg", "embodied_role": "right_locomotion", "risk_tier": "TIER_ROBOT_LOCOMOTION", "motion_capable": True, "locomotion_capable": True, "actions": ["step", "hold", "safe_stop"], "speed_limit_mps": 0.25, "torque_limit_nm": 1.5},
    "feet": {"device_class": "robot_feet_contact", "embodied_role": "ground_contact", "risk_tier": "TIER_ROBOT_LOCOMOTION", "motion_capable": True, "locomotion_capable": True, "actions": ["contact_status", "safe_stop"], "speed_limit_mps": 0.25},
    "imu_balance": {"device_class": "imu_balance_sensor", "embodied_role": "balance_witness", "risk_tier": "TIER_0_INFO", "actions": ["read_orientation", "read_acceleration"], "motion_capable": False, "physical_safety_sensitive": True, "requires_assurance": False},
    "touch_skin": {"device_class": "touch_pressure_skin", "embodied_role": "contact_sensing", "risk_tier": "TIER_2_BOUNDED_LOCAL_OPERATION_PRIVACY", "actions": ["read_contact", "read_pressure"], "privacy_sensitive": True, "physical_safety_sensitive": True, "requires_assurance": False},
    "battery": {"device_class": "battery_power", "embodied_role": "power_state", "risk_tier": "TIER_0_INFO", "actions": ["read_battery", "read_health"], "requires_assurance": False},
    "thermal_body": {"device_class": "thermal_body", "embodied_role": "thermal_state", "risk_tier": "TIER_0_INFO", "actions": ["read_temperature", "read_thermal_limits"], "requires_assurance": False},
    "emergency_stop": {"device_class": "emergency_stop", "embodied_role": "physical_safety_stop", "risk_tier": "TIER_ROBOT_EMERGENCY_STOP", "actions": ["safe_stop", "disable_actuators"], "physical_safety_sensitive": True},
    "charging_dock": {"device_class": "charging_dock", "embodied_role": "self_maintenance_power", "risk_tier": "TIER_ROBOT_LOCOMOTION", "actions": ["dock_status", "request_dock", "safe_stop"], "motion_capable": True, "locomotion_capable": True, "speed_limit_mps": 0.20},
}

def _canonical_robotic_body_part(body_part: str) -> str:
    key = str(body_part or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "arm_left": "left_arm", "arm_right": "right_arm", "hand_left": "left_hand", "hand_right": "right_hand",
        "gripper_left": "left_hand", "gripper_right": "right_hand", "leg_left": "left_leg", "leg_right": "right_leg",
        "voice": "mouth_voice", "speaker": "mouth_voice", "microphone": "ears", "mic": "ears", "face": "face_expression",
        "imu": "imu_balance", "balance": "imu_balance", "skin": "touch_skin", "estop": "emergency_stop", "e_stop": "emergency_stop",
        "dock": "charging_dock", "charger": "charging_dock", "thermal": "thermal_body",
    }
    return aliases.get(key, key)

def _robotic_body_capability_record(body_part: str) -> Optional[DeviceCapabilityRecord]:
    key = _canonical_robotic_body_part(body_part)
    spec = ROBOTIC_BODY_PART_DEFINITIONS.get(key)
    if not isinstance(spec, dict):
        return None
    actions = [str(x) for x in (spec.get("actions") or [])]
    risk_tier = str(spec.get("risk_tier") or "TIER_ROBOT_OBSERVE_ONLY")
    physical = bool(spec.get("physical_safety_sensitive", spec.get("motion_capable", False) or spec.get("manipulation_capable", False) or spec.get("locomotion_capable", False)))
    driver_id = str(spec.get("driver_id") or f"com.softdev0.robot.placeholder.{key}")
    present = msdc_driver_present(driver_id) if driver_id else False
    return DeviceCapabilityRecord(
        body_part=key,
        device_class=str(spec.get("device_class") or "robot_body_part"),
        driver_id=driver_id,
        transport=str(spec.get("transport") or "robot_internal_bus"),
        risk_tier=risk_tier,
        privacy_sensitive=bool(spec.get("privacy_sensitive", False)),
        physical_safety_sensitive=physical,
        user_control_required=True,
        driver_present=present,
        manifest_valid=False,
        actions=actions,
        status="driver_available" if present else "declared_not_installed",
        evidence={
            "declared_robotic_body_part": True,
            "actual_driver_required_before_execution": True,
            "doctrine": "Structured body representation only. Presence in body_map is not actuation authority.",
        },
        embodied_role=str(spec.get("embodied_role") or "robot_body_part"),
        motion_capable=bool(spec.get("motion_capable", False)),
        locomotion_capable=bool(spec.get("locomotion_capable", False)),
        manipulation_capable=bool(spec.get("manipulation_capable", False)),
        safe_stop_action=str(spec.get("safe_stop_action") or "safe_stop"),
        force_limit_n=spec.get("force_limit_n"),
        speed_limit_mps=spec.get("speed_limit_mps"),
        torque_limit_nm=spec.get("torque_limit_nm"),
        requires_smget=bool(spec.get("requires_smget", True)),
        requires_assurance=bool(spec.get("requires_assurance", True)),
        requires_local_presence=bool(spec.get("requires_local_presence", True)),
        actuator_state="not_installed" if not present else "available_requires_governance",
        safety_envelope={
            "observe_only_until_driver_verified": not present,
            "max_force_n": spec.get("force_limit_n"),
            "max_speed_mps": spec.get("speed_limit_mps"),
            "max_torque_nm": spec.get("torque_limit_nm"),
            "human_contact_allowed_without_emergency": False,
            "requires_current_perception": bool(spec.get("motion_capable", False) or spec.get("locomotion_capable", False)),
            "safe_stop_required": True,
        },
    )


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
        rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        try:
            if path.exists() and path.read_text(encoding="utf-8") == rendered:
                return True
        except Exception:
            pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
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
            },
        )
    if body_part in {"vr", "vr_hud", "operator_vr_surface", "operator_visual_surface", "headset", "hmd"}:
        manifest = _read_driver_manifest(VR_HEADSET_DRIVER_ID)
        actions = _action_list_from_manifest(manifest)
        present = msdc_driver_present(VR_HEADSET_DRIVER_ID)
        return DeviceCapabilityRecord(
            body_part="operator_vr_surface",
            device_class="immersive_operator_display",
            driver_id=VR_HEADSET_DRIVER_ID,
            transport="usb+hdmi",
            risk_tier="TIER_1_HARMLESS_LOCAL_UI",
            privacy_sensitive=False,
            physical_safety_sensitive=False,
            user_control_required=True,
            driver_present=present,
            manifest_valid=bool(manifest.get("id") == VR_HEADSET_DRIVER_ID and actions),
            actions=actions,
            status="driver_available" if present else "driver_missing",
            evidence={
                "manifest_name": manifest.get("name"),
                "manifest_version": manifest.get("version"),
                "category": manifest.get("category"),
                "runtime": (manifest.get("platform") or {}).get("runtime") if isinstance(manifest.get("platform"), dict) else [],
                "transport": (manifest.get("platform") or {}).get("transport") if isinstance(manifest.get("platform"), dict) else [],
                "display_role": "read_only_operator_hud_surface",
            },
        )
    if body_part in {"display", "display_bridge", "hdmi", "vga", "monitor", "secondary_display"}:
        manifest = _read_driver_manifest(DISPLAY_DRIVER_ID)
        actions = _action_list_from_manifest(manifest)
        present = msdc_driver_present(DISPLAY_DRIVER_ID)
        return DeviceCapabilityRecord(
            body_part="display_bridge",
            device_class="display_output_bridge",
            driver_id=DISPLAY_DRIVER_ID,
            transport="hdmi",
            risk_tier="TIER_1_HARMLESS_LOCAL_UI",
            privacy_sensitive=False,
            physical_safety_sensitive=False,
            user_control_required=True,
            driver_present=present,
            manifest_valid=bool((manifest.get("id") in (DISPLAY_DRIVER_ID, None, "")) and actions),
            actions=actions,
            status="driver_available" if present else "driver_missing",
            evidence={
                "manifest_name": manifest.get("name") or "VGA/HDMI Display Bridge",
                "manifest_version": manifest.get("version"),
                "transport": (manifest.get("platform") or {}).get("transport") if isinstance(manifest.get("platform"), dict) else [],
                "display_role": "secondary_display_route_support",
            },
        )
    robotic = _robotic_body_capability_record(body_part)
    if robotic is not None:
        return robotic
    return DeviceCapabilityRecord(
        body_part=body_part,
        device_class="unknown",
        driver_id="",
        status="unmapped_body_part",
        user_control_required=True,
        evidence={"known_robotic_body_parts": sorted(ROBOTIC_BODY_PART_DEFINITIONS.keys())},
    )


def _robotic_body_parts_map() -> Dict[str, Dict[str, Any]]:
    return {k: asdict(msdc_get_device_capability(k)) for k in sorted(ROBOTIC_BODY_PART_DEFINITIONS.keys())}


def _robotic_safety_doctrine() -> Dict[str, Any]:
    return {
        "no_hallucinated_actuation": True,
        "structured_body_map_is_not_execution_authority": True,
        "requires_smget_for_physical_actions": True,
        "requires_operatorcore_for_execution": True,
        "requires_assurancegate_for_apply_mode": True,
        "requires_securitygovernor_for_caller_authority": True,
        "requires_compare_compass_validation": True,
        "requires_current_sensor_evidence_for_motion": True,
        "safe_stop_overrides_task_goal": True,
        "human_life_over_property_over_robot_body": True,
        "unrestricted_autonomy_forbidden": True,
    }


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
        "schema": "SarahMemoryMSDC.body_map.v2",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "limits": {
            "msdc_self_authorizes": False,
            "requires_smget_for_actions": True,
            "discovery_is_activation": False,
            "user_control_required_for_camera": True,
            "humanoid_body_representation_enabled": True,
            "robotic_body_execution_enabled": False,
        },
        "body_parts": {
            "eyes": eyes,
            "operator_vr_surface": asdict(msdc_get_device_capability("operator_vr_surface")),
            "display_bridge": asdict(msdc_get_device_capability("display_bridge")),
            **_robotic_body_parts_map(),
        },
        "robotic_body": {
            "schema": ROBOTIC_BODY_SCHEMA_VERSION,
            "mode": "STRUCTURED_REPRESENTATION_ONLY",
            "body_style": "humanoid_general",
            "body_parts_declared": sorted(ROBOTIC_BODY_PART_DEFINITIONS.keys()),
            "execution_authority": False,
            "movement_lock_default": True,
            "doctrine": _robotic_safety_doctrine(),
        },
        "support_buses": {"usb_host": usbhost},
        "operator_view": {
            "mode": "OBSERVE_ONLY",
            "movement_lock": True,
            "hud_surface": "operator_vr_surface",
            "camera_source": "eyes",
            "display_bridge": "display_bridge",
            "doctrine": "VR HUD renders telemetry only; actions remain behind Cognitive TriForce, SMGET, OperatorCore, AssuranceGate, and MSDC.",
        },
    }
    if persist:
        _safe_json_write(_body_map_path(), body_map)
    return body_map


def msdc_court_witness(body_part: str = "eyes", include_probe: bool = False) -> Dict[str, Any]:
    body_map = msdc_map_body(persist=True)
    key = str(body_part or "eyes").strip().lower()
    if key in {"vr", "vr_hud", "headset", "hmd"}:
        key = "operator_vr_surface"
    elif key in {"display", "hdmi", "monitor"}:
        key = "display_bridge"
    elif key in {"eye", "camera", "webcam", "vision"}:
        key = "eyes"
    record = body_map.get("body_parts", {}).get(key, body_map.get("body_parts", {}).get("eyes", {}))
    witness: Dict[str, Any] = {
        "ok": True,
        "source_family": "SarahMemoryMSDC",
        "evidence_class": "motor_device_manager_witness",
        "verified": bool(record.get("driver_present") and record.get("manifest_valid")),
        "capability_class": "operator_visual_environment" if key == "operator_vr_surface" else "vision_environment",
        "body_part": key,
        "body_part_record": record,
        "body_map_path": str(_body_map_path()),
        "limits": {
            "camera_opened": False,
            "driver_session_started": False,
            "read_only_fact_check": True,
            "msdc_self_authorizes": False,
        },
    }
    if include_probe:
        witness["driver_discovery"] = msdc_camera_discover(timeout_seconds=_probe_timeout_seconds())
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


def _probe_timeout_seconds(default: float = 3.0) -> float:
    try:
        value = float(os.getenv("SARAH_MSDC_PROBE_TIMEOUT_SEC", str(default)) or default)
        return max(0.5, min(value, 10.0))
    except Exception:
        return default


def msdc_driver_action_bounded(
    driver_id: str,
    action_id: str,
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a read-only driver action in a killable subprocess.

    Windows camera backends such as MSMF/DirectShow can block inside device
    discovery or capability probing. This wrapper preserves the MSDC doctrine
    that discovery is not activation while preventing Flask/API requests from
    hanging indefinitely. State-changing actions still use msdc_driver_action
    directly and remain behind explicit authority.
    """
    driver_id = str(driver_id or "").strip()
    action_id = str(action_id or "").strip()
    payload = payload if isinstance(payload, dict) else {}
    context = context if isinstance(context, dict) else {}
    timeout = _probe_timeout_seconds(timeout_seconds if timeout_seconds is not None else 3.0)

    if not driver_id or not action_id:
        return {"ok": False, "error": "missing_driver_or_action", "source": MODULE_NAME, "bounded": True}

    if action_id not in READ_ONLY_ACTIONS:
        return {
            "ok": False,
            "error": "bounded_action_requires_read_only_action",
            "driver_id": driver_id,
            "action_id": action_id,
            "risk_tier": _action_risk(action_id),
            "source": MODULE_NAME,
            "bounded": True,
        }

    probe_code = '''
import json, os, sys, traceback
try:
    base = os.getcwd()
    if base and base not in sys.path:
        sys.path.insert(0, base)
    import SarahMemoryMSDC as m
    driver_id = sys.argv[1]
    action_id = sys.argv[2]
    payload = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else {}
    context = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}
    result = m.msdc_driver_action(driver_id, action_id, payload=payload, context=context)
    print("__SARAH_MSDC_RESULT__" + json.dumps(result if isinstance(result, dict) else {"ok": bool(result), "result": result}, default=str))
except Exception as exc:
    print("__SARAH_MSDC_RESULT__" + json.dumps({"ok": False, "error": "bounded_subprocess_exception", "detail": str(exc), "traceback": traceback.format_exc(limit=4)}, default=str))
'''
    try:
        env = os.environ.copy()
        env["SARAH_MSDC_BOUNDED_PROBE"] = "1"
        root = str(_base_dir())
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root + (os.pathsep + existing_pp if existing_pp else "")
        cp = subprocess.run(
            [
                sys.executable,
                "-c",
                probe_code,
                driver_id,
                action_id,
                json.dumps(payload, default=str),
                json.dumps({"caller": MODULE_NAME, **context}, default=str),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        marker = "__SARAH_MSDC_RESULT__"
        combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
        for line in reversed(combined.splitlines()):
            if line.startswith(marker):
                try:
                    result = json.loads(line[len(marker):])
                    if isinstance(result, dict):
                        result.setdefault("driver_id", driver_id)
                        result.setdefault("action_id", action_id)
                        result.setdefault("source", MODULE_NAME)
                        result["bounded"] = True
                        result["timeout_seconds"] = timeout
                        result["subprocess_returncode"] = cp.returncode
                        return result
                except Exception:
                    break
        return {
            "ok": False,
            "error": "bounded_subprocess_no_json_result",
            "driver_id": driver_id,
            "action_id": action_id,
            "returncode": cp.returncode,
            "stdout_tail": (cp.stdout or "")[-1200:],
            "stderr_tail": (cp.stderr or "")[-1200:],
            "timeout_seconds": timeout,
            "source": MODULE_NAME,
            "bounded": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timed_out": True,
            "error": "driver_probe_timeout",
            "reason": "read_only_driver_action_exceeded_timeout_and_was_terminated",
            "driver_id": driver_id,
            "action_id": action_id,
            "timeout_seconds": timeout,
            "source": MODULE_NAME,
            "bounded": True,
            "discovery_is_activation": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": "bounded_driver_action_failed",
            "detail": str(exc),
            "driver_id": driver_id,
            "action_id": action_id,
            "timeout_seconds": timeout,
            "source": MODULE_NAME,
            "bounded": True,
        }


def msdc_camera_discover(timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    return msdc_driver_action_bounded(
        CAMERA_DRIVER_ID,
        "discover_devices",
        payload={},
        context={"read_only_probe": True, "body_part": "eyes", "discovery_is_activation": False},
        timeout_seconds=timeout_seconds,
    )


def msdc_camera_probe(timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    return msdc_driver_action_bounded(
        CAMERA_DRIVER_ID,
        "probe_capabilities",
        payload={},
        context={"read_only_probe": True, "body_part": "eyes", "probe_is_activation": False},
        timeout_seconds=timeout_seconds,
    )


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




def _bool_ready(record: Dict[str, Any]) -> bool:
    return bool(isinstance(record, dict) and record.get("driver_present") and record.get("manifest_valid"))


def msdc_vr_native_profile_status() -> Dict[str, Any]:
    """Read-only SarahMemory-native VR/HMD profile status.

    This is a witness packet only. It does not open cameras, launch renderers,
    change displays, or send headset control packets.
    """
    headset_cfg = _read_driver_config(VR_HEADSET_DRIVER_ID)
    display_cfg = _read_driver_config(DISPLAY_DRIVER_ID)
    profile = headset_cfg.get("native_profile") if isinstance(headset_cfg.get("native_profile"), dict) else {}
    profiles = headset_cfg.get("headset_profiles") if isinstance(headset_cfg.get("headset_profiles"), dict) else {}
    selected_profile = str(headset_cfg.get("selected_profile") or profile.get("profile_id") or "psvr_v1_processor_box")
    active_profile = profiles.get(selected_profile) if isinstance(profiles, dict) and isinstance(profiles.get(selected_profile), dict) else profile
    return {
        "ok": True,
        "schema": "SarahMemoryMSDC.vr_native_profile.v1",
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": bool(headset_cfg.get("external_runtime_allowed", False)),
        "external_runtime_dependency": False,
        "selected_profile": selected_profile,
        "active_profile": active_profile if isinstance(active_profile, dict) else {},
        "headset_driver_id": VR_HEADSET_DRIVER_ID,
        "display_driver_id": DISPLAY_DRIVER_ID,
        "camera_driver_id": CAMERA_DRIVER_ID,
        "auto_start_on_headset_connected": bool(headset_cfg.get("auto_start_on_headset_connected", True)),
        "auto_stop_on_headset_disconnected": bool(headset_cfg.get("auto_stop_on_headset_disconnected", True)),
        "display_surface": display_cfg.get("vr_surface") if isinstance(display_cfg.get("vr_surface"), dict) else {},
        "limits": {
            "msdc_launches_renderer": False,
            "msdc_opens_camera": False,
            "msdc_controls_headset_display": False,
            "movement_lock": True,
        },
    }


def msdc_vr_probe(include_driver_actions: bool = True) -> Dict[str, Any]:
    """Aggregate read-only VR probe across camera, display bridge, and HMD.

    This is the packet app.py should call before starting the renderer.
    """
    body_map = msdc_map_body(persist=True)
    parts = body_map.get("body_parts", {}) if isinstance(body_map, dict) else {}
    eyes = parts.get("eyes", {}) if isinstance(parts.get("eyes"), dict) else {}
    vr_surface = parts.get("operator_vr_surface", {}) if isinstance(parts.get("operator_vr_surface"), dict) else {}
    display = parts.get("display_bridge", {}) if isinstance(parts.get("display_bridge"), dict) else {}
    native_profile = msdc_vr_native_profile_status()

    driver_probe: Dict[str, Any] = {}
    if include_driver_actions:
        driver_probe["headset"] = msdc_driver_action_bounded(VR_HEADSET_DRIVER_ID, "native_hmd_status", payload={}, context={"read_only_probe": True, "body_part": "operator_vr_surface"}, timeout_seconds=_probe_timeout_seconds())
        if not driver_probe["headset"].get("ok"):
            driver_probe["headset"] = msdc_driver_action_bounded(VR_HEADSET_DRIVER_ID, "operator_hud_status", payload={}, context={"read_only_probe": True, "body_part": "operator_vr_surface"}, timeout_seconds=_probe_timeout_seconds())
        driver_probe["display"] = msdc_driver_action_bounded(DISPLAY_DRIVER_ID, "operator_hud_surface", payload={}, context={"read_only_probe": True, "body_part": "display_bridge"}, timeout_seconds=_probe_timeout_seconds())
        if not driver_probe["display"].get("ok"):
            driver_probe["display"] = msdc_driver_action_bounded(DISPLAY_DRIVER_ID, "build_hud_surface_request", payload={}, context={"read_only_probe": True, "body_part": "display_bridge"}, timeout_seconds=_probe_timeout_seconds())
        driver_probe["camera"] = msdc_camera_probe(timeout_seconds=_probe_timeout_seconds())

    headset_connected = False
    try:
        h = driver_probe.get("headset") if isinstance(driver_probe, dict) else {}
        headset_connected = bool(
            h.get("connected")
            or h.get("headset_connected")
            or (isinstance(h.get("native_hmd"), dict) and h["native_hmd"].get("connected"))
            or (isinstance(h.get("detect"), dict) and h["detect"].get("connected"))
        )
    except Exception:
        headset_connected = False

    readiness = {
        "camera_eye_source_ready": _bool_ready(eyes),
        "operator_vr_surface_ready": _bool_ready(vr_surface),
        "display_bridge_ready": _bool_ready(display),
        "headset_connected": headset_connected,
        "renderer_allowed": True,
        "movement_lock": True,
    }
    readiness["ready_for_renderer_start"] = bool(
        readiness["camera_eye_source_ready"]
        and readiness["operator_vr_surface_ready"]
        and readiness["display_bridge_ready"]
    )
    return {
        "ok": True,
        "schema": "SarahMemoryMSDC.vr_probe.v1",
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": False,
        "readiness": readiness,
        "body_map": body_map,
        "native_profile": native_profile,
        "drivers": driver_probe,
        "limits": {
            "probe_is_activation": False,
            "camera_opened": False,
            "renderer_started": False,
            "hud_can_execute_actions": False,
            "hud_can_authorize_movement": False,
            "requires_smget_for_actions": True,
            "msdc_self_authorizes": False,
        },
    }


def msdc_vr_surface_request(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a read-only surface contract for app.py/SarahMemoryVRHudRenderer.py."""
    payload = payload if isinstance(payload, dict) else {}
    probe = msdc_vr_probe(include_driver_actions=True)
    display_packet = ((probe.get("drivers") or {}).get("display") or {}) if isinstance(probe.get("drivers"), dict) else {}
    surface = display_packet.get("surface") if isinstance(display_packet.get("surface"), dict) else display_packet.get("hud_surface")
    if not isinstance(surface, dict):
        surface = {}
    request_packet = {
        "ok": True,
        "schema": "SarahMemoryMSDC.vr_surface_request.v1",
        "native_runtime": "sarahmemory_native",
        "renderer_file": "SarahMemoryVRHudRenderer.py",
        "api_base": str(payload.get("api_base") or "http://127.0.0.1:8000"),
        "mirror_preview": bool(payload.get("mirror_preview", True)),
        "headset_surface": bool(payload.get("headset_surface", True)),
        "surface": surface,
        "probe": probe,
        "safety": {
            "observe_only": True,
            "movement_locked": True,
            "hud_can_execute_actions": False,
            "hud_can_authorize_movement": False,
        },
    }
    return request_packet

def msdc_dispatch_emergency_contract(contract: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Bounded MSDC emergency dispatcher for OperatorCore-owned contracts.

    This function intentionally does not invent body abilities. If SarahMemory is
    only in a PC/server body, it stages notify/warn/observe actions and reports
    exactly what is missing. Physical movement/device action requires a real driver,
    a valid contract, and explicit allow_physical_dispatch from OperatorCore.
    """
    c = dict(contract or {}) if isinstance(contract, dict) else {}
    context = context if isinstance(context, dict) else {}
    selected = c.get("selected_action") if isinstance(c.get("selected_action"), dict) else {}
    action_id = str(selected.get("action_id") or c.get("selected_action_id") or "").strip()
    incident_id = str(c.get("incident_id") or "")
    allow_physical = bool(c.get("allow_physical_dispatch"))
    body_map = msdc_map_body(persist=True)

    if str(c.get("schema") or "") != "SarahMemory.smget.emergency_action_contract.v1":
        return {
            "ok": False,
            "executed": False,
            "error": "invalid_emergency_contract_schema",
            "source": MODULE_NAME,
            "incident_id": incident_id,
        }

    if not action_id:
        return {
            "ok": False,
            "executed": False,
            "error": "missing_selected_action_id",
            "source": MODULE_NAME,
            "incident_id": incident_id,
        }

    notification_actions = {
        "alert_humans",
        "warn_human_and_driver",
        "notify_caregiver",
        "call_emergency_services",
        "notify_emergency_services_after_collision_risk",
        "notify_if_high_risk",
        "observe_and_escalate",
        "monitor_and_reassure",
        "evacuate_or_alert",
    }
    physical_actions = {
        "cut_power_if_verified_safe",
        "suppress_with_correct_extinguisher",
        "retrieve_verified_inhaler",
        "move_human_out_of_path",
        "shield_human_with_robot_body",
    }

    if action_id in notification_actions:
        return {
            "ok": True,
            "executed": False,
            "staged": True,
            "notification_required": True,
            "action_id": action_id,
            "incident_id": incident_id,
            "reason": "Emergency notification/warning action staged; communications executor is required for outbound calls/messages.",
            "body_map": body_map,
            "source": MODULE_NAME,
            "execution_authority": False,
        }

    if action_id in physical_actions and not allow_physical:
        return {
            "ok": True,
            "executed": False,
            "staged": True,
            "action_id": action_id,
            "incident_id": incident_id,
            "reason": "Physical emergency action blocked until OperatorCore contract explicitly allows physical dispatch.",
            "body_map": body_map,
            "source": MODULE_NAME,
            "execution_authority": False,
        }

    if action_id in physical_actions and allow_physical:
        return {
            "ok": False,
            "executed": False,
            "action_id": action_id,
            "incident_id": incident_id,
            "error": "no_verified_physical_body_driver_for_selected_emergency_action",
            "reason": "MSDC is wired, but no matching robot/vehicle/actuator driver is present in this Project Folder snapshot.",
            "body_map": body_map,
            "source": MODULE_NAME,
            "execution_authority": False,
        }

    return {
        "ok": True,
        "executed": False,
        "staged": True,
        "action_id": action_id,
        "incident_id": incident_id,
        "reason": "Emergency action accepted as staged evidence; no specific MSDC driver route is declared for this action_id.",
        "body_map": body_map,
        "source": MODULE_NAME,
        "execution_authority": False,
    }


def msdc_vr_hud_status() -> Dict[str, Any]:
    """Read-only body-map status for the VR Operator HUD surface."""
    body_map = msdc_map_body(persist=True)
    parts = body_map.get("body_parts", {}) if isinstance(body_map, dict) else {}
    native_profile = msdc_vr_native_profile_status()
    readiness = {
        "camera_eye_source_ready": _bool_ready(parts.get("eyes", {}) if isinstance(parts.get("eyes"), dict) else {}),
        "operator_vr_surface_ready": _bool_ready(parts.get("operator_vr_surface", {}) if isinstance(parts.get("operator_vr_surface"), dict) else {}),
        "display_bridge_ready": _bool_ready(parts.get("display_bridge", {}) if isinstance(parts.get("display_bridge"), dict) else {}),
        "movement_lock": True,
    }
    return {
        "ok": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "schema": "SarahMemoryMSDC.vr_hud_status.v2",
        "mode": "OBSERVE_ONLY",
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": False,
        "renderer_expected": True,
        "movement_lock": True,
        "readiness": readiness,
        "operator_vr_surface": parts.get("operator_vr_surface", {}),
        "display_bridge": parts.get("display_bridge", {}),
        "camera_eye_source": parts.get("eyes", {}),
        "native_profile": native_profile,
        "driver_ids": {
            "headset": VR_HEADSET_DRIVER_ID,
            "display": DISPLAY_DRIVER_ID,
            "camera": CAMERA_DRIVER_ID,
        },
        "body_map": body_map,
        "limits": {
            "hud_can_execute_actions": False,
            "hud_can_authorize_movement": False,
            "requires_smget_for_actions": True,
            "msdc_self_authorizes": False,
            "vision_background_analysis_continues_after_vr_stop": True,
        },
    }

def msdc_robotic_body_status() -> Dict[str, Any]:
    """Return a read-only embodied humanoid body status packet.

    This is a structured representation layer for Moya-class / humanoid bodies.
    It does not start drivers and does not authorize motion.
    """
    body_map = msdc_map_body(persist=False)
    parts = body_map.get("body_parts", {}) if isinstance(body_map, dict) else {}
    robotic_keys = sorted(ROBOTIC_BODY_PART_DEFINITIONS.keys())
    installed = [k for k in robotic_keys if isinstance(parts.get(k), dict) and bool(parts[k].get("driver_present"))]
    declared = [k for k in robotic_keys if isinstance(parts.get(k), dict)]
    return {
        "ok": True,
        "schema": "SarahMemoryMSDC.robotic_body_status.v1",
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "mode": "OBSERVE_AND_GOVERN_ONLY",
        "execution_authority": False,
        "declared_parts": declared,
        "installed_parts": installed,
        "missing_driver_parts": [k for k in declared if k not in installed],
        "movement_lock": True,
        "doctrine": _robotic_safety_doctrine(),
        "safety_envelopes": {k: (parts.get(k) or {}).get("safety_envelope", {}) for k in declared},
    }


def msdc_evaluate_physical_action_envelope(action_request: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate whether a proposed robot/body action has a valid envelope.

    This is a pre-governance witness. It never authorizes execution.
    """
    req = action_request if isinstance(action_request, dict) else {}
    ctx = context if isinstance(context, dict) else {}
    body_part = _canonical_robotic_body_part(str(req.get("body_part") or req.get("target_body_part") or req.get("target") or ""))
    action_id = str(req.get("action") or req.get("action_id") or req.get("action_type") or "").strip().lower()
    record = asdict(msdc_get_device_capability(body_part)) if body_part else {}
    reasons: List[str] = []
    ok = True
    if not body_part or body_part not in ROBOTIC_BODY_PART_DEFINITIONS:
        ok = False; reasons.append("unknown_or_missing_robotic_body_part")
    if record and not bool(record.get("driver_present")):
        ok = False; reasons.append("body_part_driver_not_verified")
    if action_id and record and action_id not in [str(x).lower() for x in (record.get("actions") or [])]:
        ok = False; reasons.append("action_not_declared_for_body_part")
    if bool(record.get("motion_capable") or record.get("locomotion_capable") or record.get("manipulation_capable")) and not bool(ctx.get("current_perception_fresh") or req.get("current_perception_fresh")):
        ok = False; reasons.append("current_perception_required_for_motion")
    return {
        "ok": bool(ok),
        "decision": "ENVELOPE_VALID" if ok else "ENVELOPE_INVALID",
        "execution_authority": False,
        "body_part": body_part,
        "action_id": action_id,
        "body_part_record": record,
        "reasons": reasons or ["Physical action envelope is structurally valid; SMGET/OperatorCore/Assurance still required."],
        "requires_smget": True,
        "requires_operatorcore": True,
        "requires_assurance": True,
        "requires_safe_stop": True,
    }


# ---------------------------------------------------------------------------
# RHYTHM COGNITION / EMBODIED MOTION WITNESS - v9.0.0
# ---------------------------------------------------------------------------
def msdc_rhythm_motion_witness(
    command_text: str = "",
    *,
    context: Optional[Dict[str, Any]] = None,
    body_packet: Optional[Dict[str, Any]] = None,
    hazard_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a read-only MSDC witness packet for rhythm-informed motion.

    This function does NOT move hardware. It translates RhythmCognition's
    MotionIntentPacket into MSDC feasibility/envelope checks so OperatorCore,
    SMGET, SafetyPolicies, and AssuranceGate can decide whether anything may
    proceed. Music/emotion/urgency may affect cadence suggestions only; MSDC
    does not self-authorize motor movement.
    """
    ctx = dict(context or {})
    body = dict(body_packet or {}) if isinstance(body_packet, dict) else {}
    hazard = dict(hazard_packet or {}) if isinstance(hazard_packet, dict) else {}
    motion_packet: Dict[str, Any] = {
        "ok": False,
        "error": "SarahMemoryRhythmCognition unavailable",
        "execution_authority": False,
    }

    try:
        import SarahMemoryRhythmCognition as _Rhythm  # type: ignore
        fn = getattr(_Rhythm, "build_embodied_motion_packet", None)
        if callable(fn):
            motion_packet = fn(command_text, context=ctx, body_packet=body, hazard_packet=hazard)
    except Exception as exc:
        motion_packet = {"ok": False, "error": str(exc), "execution_authority": False}

    profile = str(motion_packet.get("motion_profile") or "").lower()
    rhythm_mode = str(motion_packet.get("rhythm_mode") or "FOCUSED")
    suggested_actions: List[Dict[str, Any]] = []

    def _add(body_part: str, action: str, reason: str) -> None:
        suggested_actions.append({
            "body_part": body_part,
            "action": action,
            "reason": reason,
            "motion_profile": profile,
            "rhythm_mode": rhythm_mode,
        })

    if profile in {"safe_stop", "still"}:
        for part in ("head", "neck", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "feet"):
            _add(part, "safe_stop", "RhythmCognition requested still/safe-stop profile.")
    elif profile in {"idle_sway", "slow_dance", "dance", "head_bob", "hand_tap"}:
        _add("face_expression", "smile" if profile in {"slow_dance", "dance"} else "neutral", "Facial expression may match rhythm if approved.")
        _add("head", "look_at" if profile == "head_bob" else "center_head", "Head/attention expression may follow rhythm if approved.")
        if profile in {"slow_dance", "dance", "hand_tap"}:
            _add("left_arm", "raise", "Upper-body rhythm expression suggestion only.")
            _add("right_arm", "raise", "Upper-body rhythm expression suggestion only.")
        if profile in {"slow_dance", "dance"}:
            _add("torso", "stand_posture", "Posture/sway suggestion only; no locomotion authority.")
    elif profile == "walk_pace_sync":
        _add("hips", "balance_hold", "Locomotion balance witness required.")
        _add("left_leg", "step", "RhythmCognition requested pace-sync movement; envelope must validate.")
        _add("right_leg", "step", "RhythmCognition requested pace-sync movement; envelope must validate.")
        _add("feet", "contact_status", "Ground-contact witness required before movement.")
    else:
        _add("face_expression", "neutral", "Default avatar/body expression fallback.")

    envelopes: List[Dict[str, Any]] = []
    for action in suggested_actions:
        env_ctx = dict(ctx)
        env_ctx.update({
            "current_perception_fresh": bool(ctx.get("current_perception_fresh") or body.get("current_perception_fresh") or body.get("perception_fresh")),
            "rhythm_motion_witness": True,
        })
        try:
            envelopes.append(msdc_evaluate_physical_action_envelope(action, env_ctx))
        except Exception as exc:
            envelopes.append({
                "ok": False,
                "decision": "ENVELOPE_INVALID",
                "body_part": action.get("body_part"),
                "action_id": action.get("action"),
                "reasons": [str(exc)],
                "execution_authority": False,
            })

    envelope_ok = all(bool(e.get("ok")) for e in envelopes) if envelopes else False
    return {
        "ok": True,
        "schema": "SarahMemoryMSDC.rhythm_motion_witness.v1",
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "mode": "READ_ONLY_WITNESS",
        "execution_authority": False,
        "rhythm_cognition_available": not bool(motion_packet.get("error")),
        "motion_packet": motion_packet,
        "suggested_actions": suggested_actions,
        "envelopes": envelopes,
        "envelope_ok": bool(envelope_ok),
        "requires_smget": True,
        "requires_operatorcore": True,
        "requires_assurance": True,
        "requires_current_sensor_evidence": True,
        "requires_safe_stop": True,
        "doctrine": {
            "rhythm_is_not_motor_authority": True,
            "music_may_not_directly_control_motors": True,
            "emotion_may_not_override_safety": True,
            "msdc_never_self_authorizes": True,
        },
    }


def get_rhythm_motion_witness(command_text: str = "", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return msdc_rhythm_motion_witness(command_text, context=context)


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
        "body_map": msdc_map_body(persist=False),
        "robotic_body_status": msdc_robotic_body_status(),
        "rhythm_motion_witness_available": True,
    }


# Backward-compatible aliases for possible future callers.
def get_device_manager_status() -> Dict[str, Any]:
    return msdc_status()


def get_body_map(force_refresh: bool = False) -> Dict[str, Any]:
    return msdc_map_body(force_refresh=force_refresh, persist=True)


def get_court_witness(body_part: str = "eyes") -> Dict[str, Any]:
    return msdc_court_witness(body_part=body_part, include_probe=False)


def get_robotic_body_status() -> Dict[str, Any]:
    return msdc_robotic_body_status()


def get_msdc_rhythm_motion_witness(command_text: str = "", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return msdc_rhythm_motion_witness(command_text, context=context)


def evaluate_physical_action_envelope(action_request: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return msdc_evaluate_physical_action_envelope(action_request, context)


if __name__ == "__main__":
    print(json.dumps(msdc_status(), indent=2, default=str))

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Physical twin witness. Simulation witness only; never authorizes movement.

class PhysicalTwinWitness:
    def __init__(self) -> None:
        self.schema = "SarahMemory.physical_twin_witness.v1"

    def simulate(self, action_request: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req = dict(action_request or {})
        ctx = dict(context or {})
        envelope = msdc_evaluate_physical_action_envelope(req, ctx)
        return {
            "ok": True,
            "schema": self.schema,
            "simulation_only": True,
            "authority": False,
            "envelope": envelope,
            "safe_stop_required": True,
            "requires_smget": True,
            "requires_operator_core": True,
            "reasons": ["PhysicalTwinWitness can advise on feasibility/risk but cannot authorize robot motion."],
        }


_PHYSICAL_TWIN_WITNESS = PhysicalTwinWitness()


def simulate_physical_twin_action(action_request: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _PHYSICAL_TWIN_WITNESS.simulate(action_request, context)
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemoryMSDC.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryMSDC',
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
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryMSDC.py'},
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
        "component": 'SarahMemoryMSDC',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryMSDC', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

