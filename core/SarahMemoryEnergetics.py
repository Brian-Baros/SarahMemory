"""--==The SarahMemory Project==--
File: SarahMemoryEnergetics.py
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

SarahMemory Energetics Organ v9.0.0

Governed survival/metabolic organ for computational, network, robotic, vehicle,
aerial, mobile, edge, and future embodied machine bodies.

CORE DOCTRINE:
- This organ does not create energy.
- This organ does not directly execute actions.
- This organ does not overclock, underclock, mutate voltage, tune firmware, or
  silently alter BIOS/UEFI/driver clock behavior.
- It observes, estimates, scores, and recommends bounded power/rhythm/device
  modes for governance review.
- Active operational safety outranks scheduled diagnostics, evolution, sync,
  scans, and non-critical background processing.
- Reserve is multi-domain: energy, thermal, compute, network, storage, sensor,
  actuator, stability, and emergency-action margin.

INTEGRATION:
- SarahMemoryLogicCalc.py proves physics/math facts.
- SarahMemoryRhythmCognition.py consumes cadence/power-mode recommendations.
- SarahMemoryNetwork.py consumes network-device power-state verdicts.
- SarahMemoryDiagnostics.py and SarahMemoryEvolution.py must preflight scheduled
  phases before running heavy checks or self-repair cycles.
- SafetyPolicies / OperatorCore / SMGET remain execution authorities.

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "energetics_organ"
# CATEGORY = "survival_metabolism_power_governance"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "energetics"
# HARDWARE_DOMAIN = "compute_network_robotics_vehicle_aeronautics_optional"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "energetics"
# FAMILY = "core_survival"
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
# NOTES = "Governed metabolic survival organ for reserve surplus, power mode, device duty-cycle, environment physics, and no-clock-mutation policy. Advisory only; no execution authority."
# --- SARAHMETA END ---

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

try:
    from SarahMemoryLogicCalc import LogicCalc as _LogicCalc  # type: ignore
except Exception:  # pragma: no cover
    _LogicCalc = None  # type: ignore

MODULE_NAME = "SarahMemoryEnergetics"
MODULE_VERSION = "9.0.0"

POWER_OFF = "OFF"
POWER_LOW = "LOW_POWER"
POWER_READY = "READY"
POWER_ACTIVE = "ACTIVE"
POWER_FULL = "FULL_POWER"
POWER_RECOVERY = "RECOVERY"

DECISION_ALLOW = "ALLOW"
DECISION_ALLOW_WITH_CONSTRAINTS = "ALLOW_WITH_CONSTRAINTS"
DECISION_DEFER = "DEFER"
DECISION_DENY = "DENY"
DECISION_REDUCE_MODE = "REDUCE_MODE"

CLOCK_MUTATION_FORBIDDEN = {
    "overclock", "underclock", "clock_mutation", "cpu_clock", "gpu_clock",
    "vram_clock", "voltage_change", "bios_tuning", "uefi_tuning",
    "firmware_clock", "fan_curve_override", "power_limit_override",
}

SCHEDULED_PHASES = {
    "diagnostics", "diagnostic", "self_check", "hardware_diagnostics",
    "network_diagnostics", "sync_diagnostics", "security_diagnostics",
    "evolution", "self_evolution", "repair", "rem", "dl", "sync",
    "network_scan", "device_scan", "background_learning",
}

BODY_DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "pc_edge": {
        "critical_organs": ["thermal_monitor", "storage_integrity", "user_io", "security_sentinel"],
        "reserve_keys": ["compute", "thermal", "memory", "storage", "network"],
        "full_power_requires": ["thermal_margin", "compute_margin"],
        "safe_low_power": ["batch_writes", "lower_camera_fps", "defer_sync", "prefer_cached_context"],
    },
    "server": {
        "critical_organs": ["network_presence", "storage_integrity", "security_sentinel"],
        "reserve_keys": ["compute", "thermal", "memory", "storage", "network"],
        "full_power_requires": ["thermal_margin", "storage_margin", "network_need"],
    },
    "ground_vehicle": {
        "critical_organs": ["steering", "braking", "traction", "emergency_comms", "hazard_sensors"],
        "reserve_keys": ["traction", "braking", "thermal", "battery", "network", "sensor"],
        "full_power_requires": ["braking_reserve", "traction_margin", "hazard_clearance"],
    },
    "wheeled_robot": {
        "critical_organs": ["balance", "traction", "safe_stop", "proximity", "emergency_comms"],
        "reserve_keys": ["motor_torque", "battery", "thermal", "sensor", "stability"],
    },
    "tracked_robot": {
        "critical_organs": ["track_drive", "terrain_contact", "safe_stop", "thermal_monitor"],
        "reserve_keys": ["track_torque", "terrain_resistance", "battery", "thermal"],
    },
    "biped_robot": {
        "critical_organs": ["balance", "center_of_mass", "foot_contact", "recovery_step", "proximity"],
        "reserve_keys": ["stability", "joint_torque", "sensor", "battery", "thermal"],
    },
    "quadruped_robot": {
        "critical_organs": ["stance_polygon", "load_distribution", "terrain_contact", "safe_stop"],
        "reserve_keys": ["stability", "joint_torque", "sensor", "battery", "thermal"],
    },
    "drone_multirotor": {
        "critical_organs": ["stabilization", "altitude_hold", "landing_reserve", "return_path", "emergency_comms"],
        "reserve_keys": ["battery", "thrust_margin", "thermal", "network", "sensor", "landing_reserve"],
        "never_spend": ["controlled_landing_reserve", "stabilization_reserve", "emergency_comms_reserve"],
    },
    "fixed_wing_aircraft": {
        "critical_organs": ["airspeed", "lift", "control_surfaces", "return_path", "emergency_comms"],
        "reserve_keys": ["fuel_or_battery", "airspeed_margin", "altitude_margin", "thermal", "sensor"],
    },
    "marine_surface": {
        "critical_organs": ["buoyancy", "propulsion", "navigation", "comms", "bilge_or_water_state"],
        "reserve_keys": ["battery_or_fuel", "propulsion", "weather", "network", "sensor"],
    },
    "industrial_machine": {
        "critical_organs": ["safe_stop", "guarding", "thermal_monitor", "operator_presence"],
        "reserve_keys": ["thermal", "load", "duty_cycle", "safety_interlock"],
    },
}


# Body profile alias map. SarahMemoryGlobals.py uses constitutional body-domain labels
# such as ``robotic_body`` and ``aerial_drone_body``. Energetics keeps more detailed
# engineering profile names internally. This alias layer prevents a body-domain mismatch
# from falling back silently to PC math when the organism is actually embodied.
BODY_PROFILE_ALIASES: Dict[str, str] = {
    "pc": "pc_edge",
    "desktop": "pc_edge",
    "laptop": "pc_edge",
    "pc_edge": "pc_edge",
    "server": "server",
    "cloud": "server",
    "headless_node": "server",
    "robot": "wheeled_robot",
    "robotic": "wheeled_robot",
    "robotic_body": "wheeled_robot",
    "vehicle": "ground_vehicle",
    "vehicle_body": "ground_vehicle",
    "drone": "drone_multirotor",
    "uav": "drone_multirotor",
    "aerial_drone_body": "drone_multirotor",
    "marine_body": "marine_surface",
    "industrial_machine_body": "industrial_machine",
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_float(value: Any, default: float = 0.0, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "allow", "allowed"}


def _safe_lower(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _env_float(name: str, default: float) -> float:
    try:
        return _safe_float(os.environ.get(name), default)
    except Exception:
        return default


@dataclass
class EnergyVerdict:
    ok: bool
    decision: str
    reason: str
    allowed_power_mode: str
    requested_power_mode: str
    reserve_status: str
    risk_level: str
    energy_score: float
    confidence: float
    action_type: str
    body_domain: str
    phase: str = ""
    diagnostics_mode: str = "NORMAL"
    requires_smget: bool = True
    requires_operator_core: bool = True
    requires_safety_policy: bool = True
    execution_authority: bool = False
    clock_mutation_allowed: bool = False
    no_clock_mutation_rule: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    external_state: Dict[str, Any] = field(default_factory=dict)
    logiccalc_proofs: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SarahMemoryEnergetics:
    """Governed metabolic advisor. No direct execution authority."""

    def __init__(self) -> None:
        self.module = MODULE_NAME
        self.version = MODULE_VERSION
        self.body_profiles = dict(BODY_DOMAIN_PROFILES)
        self.default_body_domain = str(getattr(config, "BODY_DOMAIN", "pc_edge") if config else "pc_edge")
        self.min_battery_reserve_pct = _env_float("SARAH_ENERGETICS_MIN_BATTERY_RESERVE_PCT", 18.0)
        self.min_emergency_reserve_pct = _env_float("SARAH_ENERGETICS_EMERGENCY_RESERVE_PCT", 25.0)
        self.max_cpu_pressure = _env_float("SARAH_ENERGETICS_MAX_CPU_PCT", 92.0)
        self.max_memory_pressure = _env_float("SARAH_ENERGETICS_MAX_MEMORY_PCT", 94.0)
        self.max_disk_pressure = _env_float("SARAH_ENERGETICS_MAX_DISK_PCT", 96.0)
        self.full_power_cpu_soft_cap = _env_float("SARAH_ENERGETICS_FULL_POWER_CPU_SOFT_CAP", 78.0)
        self.full_power_memory_soft_cap = _env_float("SARAH_ENERGETICS_FULL_POWER_MEMORY_SOFT_CAP", 82.0)

    # ------------------------------------------------------------------
    # Snapshot / profile
    # ------------------------------------------------------------------
    def body_profile(self, body_domain: Optional[str] = None) -> Dict[str, Any]:
        # Normalize constitutional body labels into internal engineering profiles.
        # If the profile is unknown, Energetics still returns the PC profile for status math,
        # but live authority is separately blocked by the constitutional body-awareness checklist.
        domain = _safe_lower(body_domain or self.default_body_domain).replace("-", "_") or "pc_edge"
        domain = BODY_PROFILE_ALIASES.get(domain, domain)
        return dict(self.body_profiles.get(domain) or self.body_profiles["pc_edge"])

    def _constitution_status(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read SarahMemoryGlobals hazardous-energy constitution.

        Energetics is not allowed to self-certify. This method delegates to the
        constitution file so ENERGETICS=True, NEOSKYMATRIX=True, or DEVELOPERSMODE=True
        cannot independently arm this organ. Missing Globals helpers are treated as a
        fail-closed condition.
        """
        try:
            if config is not None:
                fn = getattr(config, "sm_hazardous_energy_status", None)
                if callable(fn):
                    status = fn(context if isinstance(context, dict) else {})
                    if isinstance(status, dict):
                        return status
        except Exception as exc:
            return {
                "ok": False,
                "authority": "constitution_exception_fail_closed",
                "live_authority_allowed": False,
                "dry_run_allowed": False,
                "lockout_active": True,
                "blockers": [f"constitution_status_exception:{exc}"],
                "body_awareness": {"ok": False, "body_domain": "unknown"},
            }
        return {
            "ok": False,
            "authority": "constitution_unavailable_fail_closed",
            "live_authority_allowed": False,
            "dry_run_allowed": False,
            "lockout_active": True,
            "blockers": ["SarahMemoryGlobals hazardous-energy helpers unavailable"],
            "body_awareness": {"ok": False, "body_domain": "unknown"},
        }

    def _lockout_verdict(
        self,
        *,
        action_type: str,
        phase: str,
        requested_power_mode: str,
        constitution: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a governed no-authority verdict when the constitution blocks Energetics.

        This is the primary industrial safety gate: even when the module imports and the
        ENERGETICS flag is later set True, no preflight recommendation may influence live
        diagnostics, evolution, network, drivers, MSDC, or embodied hardware until the
        full SarahMemoryGlobals checklist clears.
        """
        ctx = dict(context or {}) if isinstance(context, dict) else {}
        body = constitution.get("body_awareness") if isinstance(constitution.get("body_awareness"), dict) else {}
        body_domain = str(body.get("body_domain") or ctx.get("body_domain") or self.default_body_domain or "unknown")
        blockers = constitution.get("blockers") if isinstance(constitution.get("blockers"), list) else []
        reason = "Hazardous-energy constitution blocks live Energetics authority."
        if blockers:
            reason += " Blockers: " + "; ".join(str(x) for x in blockers[:6])
        return EnergyVerdict(
            ok=False,
            decision=DECISION_DENY,
            reason=reason,
            allowed_power_mode=POWER_LOW,
            requested_power_mode=str(requested_power_mode or POWER_ACTIVE).strip().upper(),
            reserve_status="CONSTITUTION_LOCKOUT",
            risk_level="HIGH",
            energy_score=1.0,
            confidence=0.99,
            action_type=str(action_type or "unknown"),
            body_domain=body_domain,
            phase=str(phase or "hazardous_energy_preflight"),
            diagnostics_mode="DENIED_OR_LIGHTWEIGHT_ONLY",
            requires_smget=True,
            requires_operator_core=True,
            requires_safety_policy=True,
            execution_authority=False,
            clock_mutation_allowed=False,
            no_clock_mutation_rule=True,
            constraints={
                "hazardous_energy_lockout_active": True,
                "energetics_live_authority_allowed": False,
                "dry_run_only": True,
                "body_awareness_required": True,
                "no_clock_mutation_rule": True,
                "clock_mutation_allowed": False,
                "software_workload_shaping_only": True,
            },
            recommendations={
                "allowed_scope": "status_and_math_dry_run_only",
                "device_power_mode": POWER_LOW,
                "rhythm_mode": "LOW_POWER",
                "network_mode": POWER_LOW,
                "diagnostics_mode": "LIGHTWEIGHT_ONLY",
                "evolution_allowed": False,
                "embodied_use_allowed": False,
            },
            reasons=[reason],
            internal_state={"constitution_authority": constitution.get("authority"), "constitution_ok": constitution.get("ok")},
            external_state=ctx.get("external_state") if isinstance(ctx.get("external_state"), dict) else {},
            meta={
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "constitution": constitution,
                "doctrine": "Energetics cannot certify itself; SarahMemoryGlobals hazardous-energy preflight is authoritative.",
            },
        ).to_dict()

    def runtime_snapshot(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {}) if isinstance(context, dict) else {}
        cpu = _safe_float(ctx.get("cpu_percent") or ctx.get("cpu_pressure"), 0.0, 0.0, 100.0)
        mem = _safe_float(ctx.get("memory_percent") or ctx.get("memory_pressure"), 0.0, 0.0, 100.0)
        disk = _safe_float(ctx.get("disk_percent") or ctx.get("disk_pressure"), 0.0, 0.0, 100.0)
        battery_percent = ctx.get("battery_percent")
        plugged = ctx.get("power_plugged")
        temps: Dict[str, Any] = {}

        if _HAS_PSUTIL and psutil is not None:
            try:
                cpu = max(cpu, float(psutil.cpu_percent(interval=None)))
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                mem = max(mem, float(vm.percent))
            except Exception:
                pass
            try:
                root = str(getattr(config, "BASE_DIR", os.getcwd()) if config else os.getcwd())
                du = psutil.disk_usage(root)
                disk = max(disk, float(du.percent))
            except Exception:
                pass
            try:
                bat = psutil.sensors_battery()
                if bat is not None:
                    battery_percent = _safe_float(getattr(bat, "percent", None), battery_percent if battery_percent is not None else -1.0, 0.0, 100.0)
                    plugged = bool(getattr(bat, "power_plugged", False))
            except Exception:
                pass
            try:
                st = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
                if isinstance(st, dict):
                    for name, entries in st.items():
                        vals = []
                        for entry in entries or []:
                            cur = getattr(entry, "current", None)
                            if cur is not None:
                                vals.append(_safe_float(cur, 0.0))
                        if vals:
                            temps[str(name)] = {"max_c": max(vals), "count": len(vals)}
            except Exception:
                pass

        battery_known = battery_percent is not None and _safe_float(battery_percent, -1.0) >= 0.0
        battery_pct = _safe_float(battery_percent, -1.0, -1.0, 100.0)
        thermal_state = str(ctx.get("thermal_state") or "unknown").upper()
        temp_max = 0.0
        for rec in temps.values():
            if isinstance(rec, dict):
                temp_max = max(temp_max, _safe_float(rec.get("max_c"), 0.0))
        if thermal_state == "UNKNOWN":
            if temp_max >= 88.0:
                thermal_state = "CRITICAL"
            elif temp_max >= 78.0:
                thermal_state = "HOT"
            elif temp_max > 0.0:
                thermal_state = "NORMAL"

        pressure = max(cpu, mem, disk)
        if pressure >= 95.0 or thermal_state == "CRITICAL":
            reserve_status = "CRITICAL"
        elif pressure >= 88.0 or thermal_state == "HOT":
            reserve_status = "LOW"
        elif battery_known and (not _safe_bool(plugged, False)) and battery_pct < self.min_battery_reserve_pct:
            reserve_status = "LOW"
        elif pressure >= 72.0:
            reserve_status = "GUARDED"
        else:
            reserve_status = "SUFFICIENT"

        return {
            "ok": True,
            "ts": _now_iso(),
            "source": MODULE_NAME,
            "cpu_percent": round(cpu, 2),
            "memory_percent": round(mem, 2),
            "disk_percent": round(disk, 2),
            "battery_percent": round(battery_pct, 2) if battery_known else None,
            "battery_known": bool(battery_known),
            "power_plugged": bool(plugged) if plugged is not None else None,
            "thermal_state": thermal_state,
            "temperature_summary": temps,
            "reserve_status": reserve_status,
            "psutil_available": bool(_HAS_PSUTIL),
            "no_clock_mutation_rule": True,
            "clock_mutation_allowed": False,
        }

    # ------------------------------------------------------------------
    # Core preflight
    # ------------------------------------------------------------------
    def preflight_action(
        self,
        action_type: str,
        *,
        phase: str = "",
        body_domain: Optional[str] = None,
        requested_power_mode: str = POWER_ACTIVE,
        active_task: Optional[Dict[str, Any]] = None,
        internal_state: Optional[Dict[str, Any]] = None,
        external_state: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = dict(context or {}) if isinstance(context, dict) else {}

        # Constitution gate first, before telemetry, routing, diagnostics, or any recommendation.
        # Energetics may be imported for status/math-only inspection, but live preflight authority requires
        # SarahMemoryGlobals.sm_energetics_preflight_checklist() to clear. This also enforces dual-court
        # body awareness: if the current running body cannot be identified and agreed upon, this organ
        # remains non-functional for live decisions.
        requested_early = str(requested_power_mode or ctx.get("requested_power_mode") or POWER_ACTIVE).strip().upper()
        if requested_early not in {POWER_OFF, POWER_LOW, POWER_READY, POWER_ACTIVE, POWER_FULL, POWER_RECOVERY}:
            requested_early = POWER_ACTIVE
        constitution_context = dict(ctx)
        constitution_context.update({
            "action_type": action_type,
            "phase": phase or ctx.get("phase") or action_type,
            "requested_power_mode": requested_early,
            "body_domain": body_domain or ctx.get("body_domain"),
        })
        constitution = self._constitution_status(constitution_context)
        if bool(constitution.get("lockout_active", True)) or not bool(constitution.get("live_authority_allowed", False)):
            return self._lockout_verdict(
                action_type=str(action_type or ctx.get("action_type") or "unknown"),
                phase=str(phase or ctx.get("phase") or action_type or "hazardous_energy_preflight"),
                requested_power_mode=requested_early,
                constitution=constitution,
                context=constitution_context,
            )

        constraints = dict(constraints or {}) if isinstance(constraints, dict) else {}
        active_task = dict(active_task or ctx.get("active_task") or {}) if isinstance(active_task or ctx.get("active_task"), dict) else {}
        external_state = dict(external_state or ctx.get("external_state") or {}) if isinstance(external_state or ctx.get("external_state"), dict) else {}
        internal_seed = dict(internal_state or ctx.get("internal_state") or {}) if isinstance(internal_state or ctx.get("internal_state"), dict) else {}
        internal = self.runtime_snapshot(internal_seed)

        body = _safe_lower(body_domain or ctx.get("body_domain") or self.default_body_domain).replace("-", "_") or "pc_edge"
        requested = str(requested_power_mode or ctx.get("requested_power_mode") or POWER_ACTIVE).strip().upper()
        if requested not in {POWER_OFF, POWER_LOW, POWER_READY, POWER_ACTIVE, POWER_FULL, POWER_RECOVERY}:
            requested = POWER_ACTIVE
        action = _safe_lower(action_type or ctx.get("action_type") or "unknown")
        phase_s = _safe_lower(phase or ctx.get("phase") or action)
        phase_key = phase_s.replace(" ", "_")
        emergency = bool(ctx.get("emergency") or ctx.get("safety_critical") or constraints.get("safety_critical") or active_task.get("emergency"))
        safety_critical = emergency or action in {"emergency", "emergency_comms", "safe_stop", "stabilize", "controlled_landing"}
        scheduled = bool(ctx.get("scheduled") or phase_key in SCHEDULED_PHASES or action in SCHEDULED_PHASES)
        risk_level = str(ctx.get("risk_level") or active_task.get("risk_level") or constraints.get("risk_level") or "TIER_1").upper()

        reasons: List[str] = []
        allowed_power_mode = requested
        decision = DECISION_ALLOW
        ok = True
        diagnostics_mode = "NORMAL"
        reserve_status = str(internal.get("reserve_status") or "UNKNOWN")
        cpu = _safe_float(internal.get("cpu_percent"), 0.0)
        mem = _safe_float(internal.get("memory_percent"), 0.0)
        disk = _safe_float(internal.get("disk_percent"), 0.0)
        battery_pct = internal.get("battery_percent")
        battery_known = bool(internal.get("battery_known"))
        plugged = internal.get("power_plugged")

        text_blob = json.dumps({"action": action, "phase": phase_key, "constraints": constraints, "context": ctx}, sort_keys=True, default=str).lower()
        if any(k in text_blob for k in CLOCK_MUTATION_FORBIDDEN):
            return EnergyVerdict(
                ok=False,
                decision=DECISION_DENY,
                reason="ENERGETICS_CLOCK_INTEGRITY_RULE blocked clock/voltage/firmware mutation.",
                allowed_power_mode=POWER_READY,
                requested_power_mode=requested,
                reserve_status=reserve_status,
                risk_level="HIGH",
                energy_score=1.0,
                confidence=0.96,
                action_type=action,
                body_domain=body,
                phase=phase_key,
                diagnostics_mode="DENIED",
                constraints={"no_clock_mutation_rule": True, "clock_mutation_allowed": False},
                recommendations={"use": "software workload shaping only"},
                reasons=["CPU/GPU/VRAM clocks, voltage, BIOS/UEFI, firmware, and unsafe fan curve mutation are forbidden."],
                internal_state=internal,
                external_state=external_state,
                meta={"module": MODULE_NAME, "version": MODULE_VERSION},
            ).to_dict()

        safe_to_interrupt = _safe_bool(active_task.get("safe_to_interrupt"), True)
        active_task_value = _safe_lower(active_task.get("operational_value") or active_task.get("value") or "")
        physical_active = bool(active_task.get("physical_control") or active_task.get("object_being_carried") or active_task.get("motion_active") or active_task.get("in_flight") or active_task.get("vehicle_moving"))
        hazard_external = bool(external_state.get("human_nearby") or external_state.get("hazard_detected") or external_state.get("terrain_unstable") or external_state.get("low_visibility") or external_state.get("high_wind"))

        if scheduled and active_task and not safe_to_interrupt and not safety_critical:
            ok = False
            decision = DECISION_DEFER
            diagnostics_mode = "DEFERRED"
            allowed_power_mode = POWER_LOW if requested in {POWER_ACTIVE, POWER_FULL} else requested
            reasons.append("Scheduled phase deferred because active task is not safe to interrupt.")

        if physical_active and scheduled and not safety_critical:
            ok = False
            decision = DECISION_DEFER
            diagnostics_mode = "LIGHTWEIGHT_ONLY"
            allowed_power_mode = POWER_READY if requested == POWER_FULL else POWER_LOW
            reasons.append("Active embodied operation detected; non-critical scheduled work must not interrupt physical control.")

        if hazard_external and scheduled and not safety_critical:
            ok = False
            decision = DECISION_DEFER
            diagnostics_mode = "SAFETY_ONLY"
            allowed_power_mode = POWER_READY
            reasons.append("External hazard context detected; scheduled maintenance/evolution must defer or run safety-only checks.")

        if reserve_status in {"CRITICAL", "LOW"} and not safety_critical:
            ok = False
            decision = DECISION_DEFER if action in {"evolution", "self_evolution", "repair"} or "evolution" in phase_key else DECISION_REDUCE_MODE
            diagnostics_mode = "LIGHTWEIGHT_ONLY"
            allowed_power_mode = POWER_LOW
            reasons.append(f"Reserve status is {reserve_status}; non-critical work must reduce mode or defer.")

        if requested == POWER_FULL and not safety_critical:
            if cpu >= self.full_power_cpu_soft_cap or mem >= self.full_power_memory_soft_cap:
                decision = DECISION_REDUCE_MODE if ok else decision
                allowed_power_mode = POWER_ACTIVE if reserve_status not in {"LOW", "CRITICAL"} else POWER_LOW
                diagnostics_mode = "NORMAL" if allowed_power_mode == POWER_ACTIVE else "LIGHTWEIGHT_ONLY"
                reasons.append("FULL_POWER request reduced because compute/memory pressure is above soft cap.")
            if battery_known and plugged is False and _safe_float(battery_pct, 0.0) < self.min_emergency_reserve_pct:
                ok = False
                decision = DECISION_DEFER
                allowed_power_mode = POWER_LOW
                diagnostics_mode = "LIGHTWEIGHT_ONLY"
                reasons.append("Battery reserve is below emergency surplus; FULL_POWER denied for non-critical work.")

        if disk >= self.max_disk_pressure and action in {"diagnostics", "evolution", "sync", "network_scan", "background_learning"}:
            ok = False
            decision = DECISION_DEFER
            allowed_power_mode = POWER_LOW
            diagnostics_mode = "DEFERRED"
            reasons.append("Disk pressure is too high for write-heavy scheduled work.")

        if action in {"evolution", "self_evolution", "repair"} and (reserve_status != "SUFFICIENT" or physical_active or hazard_external) and not safety_critical:
            ok = False
            decision = DECISION_DEFER
            diagnostics_mode = "DEFERRED"
            allowed_power_mode = POWER_LOW
            reasons.append("Evolution requires surplus reserve and stable body/environment state.")

        if not reasons:
            reasons.append("Energetics found sufficient reserve for the requested bounded action mode.")

        pressure_score = max(cpu / 100.0, mem / 100.0, disk / 100.0)
        battery_penalty = 0.0
        if battery_known and plugged is False:
            battery_penalty = max(0.0, (self.min_emergency_reserve_pct - _safe_float(battery_pct, 100.0)) / 100.0)
        hazard_penalty = 0.18 if hazard_external else 0.0
        physical_penalty = 0.22 if physical_active and scheduled else 0.0
        energy_score = round(max(0.0, min(1.0, pressure_score + battery_penalty + hazard_penalty + physical_penalty)), 4)

        recommendations = {
            "rhythm_mode": self._rhythm_recommendation(decision, allowed_power_mode, safety_critical),
            "device_power_mode": allowed_power_mode,
            "diagnostics_mode": diagnostics_mode,
            "network_mode": allowed_power_mode if action.startswith("network") or "network" in phase_key else None,
            "prefer_low_power_over_off": True,
            "do_not_mutate_clocks": True,
            "active_task_policy": "finish_task_or_safe_pause_before_scheduled_phase" if active_task else "no_active_task_reported",
        }
        constraints_out = {
            "no_clock_mutation_rule": True,
            "clock_mutation_allowed": False,
            "software_workload_shaping_only": True,
            "execution_authority": False,
            "operator_core_required": True,
            "safety_policy_required": True,
            "smget_required": True,
            "scheduled_phases_are_proposals_not_commands": True,
        }
        constraints_out.update(constraints)

        return EnergyVerdict(
            ok=bool(ok),
            decision=decision if ok or decision in {DECISION_DEFER, DECISION_DENY, DECISION_REDUCE_MODE} else DECISION_DEFER,
            reason=reasons[0],
            allowed_power_mode=allowed_power_mode,
            requested_power_mode=requested,
            reserve_status=reserve_status,
            risk_level=risk_level,
            energy_score=energy_score,
            confidence=0.86,
            action_type=action,
            body_domain=body,
            phase=phase_key,
            diagnostics_mode=diagnostics_mode,
            constraints=constraints_out,
            recommendations=recommendations,
            reasons=reasons,
            internal_state=internal,
            external_state=external_state,
            meta={
                "module": MODULE_NAME,
                "version": MODULE_VERSION,
                "profile": self.body_profile(body),
                "constitution": constitution,
                "scheduled": scheduled,
                "safety_critical": safety_critical,
                "active_task_value": active_task_value,
            },
        ).to_dict()

    def _rhythm_recommendation(self, decision: str, allowed_power_mode: str, safety_critical: bool) -> str:
        if safety_critical:
            return "EMERGENCY"
        if decision == DECISION_DEFER:
            return "RECOVERY"
        if allowed_power_mode == POWER_LOW:
            return "LOW_POWER"
        if allowed_power_mode == POWER_READY:
            return "READY"
        if allowed_power_mode == POWER_FULL:
            return "HIGH_LOAD"
        return "NORMAL"

    # ------------------------------------------------------------------
    # Specialized preflights
    # ------------------------------------------------------------------
    def preflight_scheduled_phase(self, phase: str, **kwargs: Any) -> Dict[str, Any]:
        return self.preflight_action(str(phase or "scheduled_phase"), phase=phase, requested_power_mode=kwargs.pop("requested_power_mode", POWER_ACTIVE), context={**kwargs, "scheduled": True})

    def preflight_diagnostics_phase(self, phase: str = "diagnostics", *, requested_power_mode: str = POWER_ACTIVE, **kwargs: Any) -> Dict[str, Any]:
        return self.preflight_action("diagnostics", phase=phase, requested_power_mode=requested_power_mode, context={**kwargs, "scheduled": True})

    def preflight_evolution_cycle(self, *, autonomous: bool = False, weekly_gate: bool = False, **kwargs: Any) -> Dict[str, Any]:
        ctx = {**kwargs, "scheduled": True, "autonomous": autonomous, "weekly_gate": weekly_gate}
        return self.preflight_action("evolution", phase="self_evolution", requested_power_mode=POWER_FULL if autonomous else POWER_ACTIVE, context=ctx)

    def preflight_network_action(self, action_type: str, *, requested_power_mode: str = POWER_ACTIVE, device_type: str = "network", **kwargs: Any) -> Dict[str, Any]:
        ctx = {**kwargs, "device_type": device_type, "scheduled": action_type in {"scan", "wifi_scan", "bluetooth_scan", "heartbeat", "sync"}}
        return self.preflight_action("network_" + str(action_type or "action"), phase="network", requested_power_mode=requested_power_mode, context=ctx)

    def recommend_device_power_mode(self, device_type: str, requested_power_mode: str = POWER_ACTIVE, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {}) if isinstance(context, dict) else {}
        verdict = self.preflight_action(
            f"device_power_state:{device_type}",
            phase="device_power",
            requested_power_mode=requested_power_mode,
            context=ctx,
        )
        verdict.setdefault("device_type", str(device_type or "device"))
        return verdict

    # ------------------------------------------------------------------
    # Physics / scenario proof bridge
    # ------------------------------------------------------------------
    def calculate_physics_cost(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic LogicCalc-backed proof packets for common energy scenarios."""
        sc = dict(scenario or {}) if isinstance(scenario, dict) else {}
        # Physics cost calculation remains dry-run only when lockout is active. It proves math;
        # it never authorizes motion, device power changes, diagnostics, or evolution.
        constitution = self._constitution_status({**sc, "action_type": "energetics_physics_cost", "phase": "math_dry_run"})
        proofs: List[Dict[str, Any]] = []
        task = sc.get("task") if isinstance(sc.get("task"), dict) else sc
        env = sc.get("environment_state") if isinstance(sc.get("environment_state"), dict) else sc
        mass = _safe_float(task.get("mass_kg") or task.get("mass"), 0.0)
        g = _safe_float(env.get("gravity_m_s2") or env.get("g"), 9.80665)
        height = _safe_float(task.get("height_delta_m") or task.get("height_m") or task.get("h"), 0.0)
        duration = _safe_float(task.get("duration_s") or task.get("t"), 0.0)
        velocity = _safe_float(task.get("velocity_m_s") or task.get("v"), 0.0)
        voltage = _safe_float(task.get("voltage_v") or task.get("V"), 0.0)
        current = _safe_float(task.get("current_a") or task.get("I"), 0.0)

        def _solve(name: str, known: Dict[str, Any]) -> None:
            if _LogicCalc is None:
                proofs.append({"ok": False, "formula": name, "reason": "LogicCalc unavailable", "known": known})
                return
            try:
                fn = getattr(_LogicCalc, "solve_formula_by_name", None)
                if callable(fn):
                    proofs.append(fn(name, known))
                else:
                    proofs.append({"ok": False, "formula": name, "reason": "solve_formula_by_name unavailable", "known": known})
            except Exception as exc:
                proofs.append({"ok": False, "formula": name, "reason": str(exc), "known": known})

        if mass and g:
            _solve("Gravity Force", {"m": mass, "g": g})
        if mass and g and height:
            _solve("Gravitational Potential Energy", {"m": mass, "g": g, "h": abs(height)})
        if mass and velocity:
            _solve("Kinetic Energy", {"m": mass, "v": velocity})
        if voltage and current:
            _solve("Electrical Power", {"V": voltage, "I": current})
        if voltage and current and duration:
            _solve("Electrical Energy", {"V": voltage, "I": current, "t": duration})

        useful = 0.0
        input_e = 0.0
        for p in proofs:
            val = p.get("value") if isinstance(p, dict) else None
            if isinstance(val, dict):
                if "PE" in val:
                    useful += abs(_safe_float(val.get("PE"), 0.0))
                if "E" in val:
                    input_e += abs(_safe_float(val.get("E"), 0.0))
        waste = max(0.0, input_e - useful) if input_e else 0.0
        return {
            "ok": True,
            "scenario_id": sc.get("scenario_id") or "scenario_" + uuid.uuid4().hex[:10],
            "logiccalc_available": _LogicCalc is not None,
            "constitution": constitution,
            "dry_run_only": True,
            "execution_authority": False,
            "proofs": proofs,
            "estimated_useful_work_j": round(useful, 4),
            "estimated_input_energy_j": round(input_e, 4),
            "estimated_waste_j": round(waste, 4),
            "doctrine": "Energetics estimates only; SMGET/SafetyPolicies/OperatorCore govern execution.",
        }

    def status(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Status is allowed in dry-run because operators need to see why the organ is blocked.
        # It is not an arming signal and carries no execution authority.
        constitution = self._constitution_status(context if isinstance(context, dict) else {})
        snap = self.runtime_snapshot(context)
        return {
            "ok": True,
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "snapshot": snap,
            "constitution": constitution,
            "authority": constitution.get("authority", "unknown"),
            "live_authority_allowed": bool(constitution.get("live_authority_allowed", False)),
            "dry_run_allowed": bool(constitution.get("dry_run_allowed", False)),
            "lockout_active": bool(constitution.get("lockout_active", True)),
            "body_domain": (constitution.get("body_awareness") or {}).get("body_domain", self.default_body_domain) if isinstance(constitution.get("body_awareness"), dict) else self.default_body_domain,
            "body_profiles": sorted(self.body_profiles.keys()),
            "power_states": [POWER_OFF, POWER_LOW, POWER_READY, POWER_ACTIVE, POWER_FULL, POWER_RECOVERY],
            "clock_integrity_rule": {
                "overclocking_allowed": False,
                "underclocking_allowed": False,
                "voltage_mutation_allowed": False,
                "firmware_clock_mutation_allowed": False,
                "software_workload_shaping_allowed": True,
            },
        }


Energetics = SarahMemoryEnergetics()


def get_energetics_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return Energetics.status(context)


def preflight_action(action_type: str, **kwargs: Any) -> Dict[str, Any]:
    return Energetics.preflight_action(action_type, **kwargs)


def preflight_scheduled_phase(phase: str, **kwargs: Any) -> Dict[str, Any]:
    return Energetics.preflight_scheduled_phase(phase, **kwargs)


def preflight_diagnostics_phase(phase: str = "diagnostics", **kwargs: Any) -> Dict[str, Any]:
    return Energetics.preflight_diagnostics_phase(phase, **kwargs)


def preflight_evolution_cycle(**kwargs: Any) -> Dict[str, Any]:
    return Energetics.preflight_evolution_cycle(**kwargs)


def preflight_network_action(action_type: str, **kwargs: Any) -> Dict[str, Any]:
    return Energetics.preflight_network_action(action_type, **kwargs)


def recommend_device_power_mode(device_type: str, requested_power_mode: str = POWER_ACTIVE, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return Energetics.recommend_device_power_mode(device_type, requested_power_mode, context=context)


def calculate_physics_cost(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return Energetics.calculate_physics_cost(scenario)


# ====================================================================
# END OF SarahMemoryEnergetics.py v9.0.0
# ====================================================================
# END OF LINE

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryEnergetics',
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
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryEnergetics.py'},
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
        "component": 'SarahMemoryEnergetics',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryEnergetics', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

