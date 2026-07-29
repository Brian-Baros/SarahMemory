"""
--==The SarahMemory Project==--
Driver: com.softdev0.vr.headset.usb
Purpose: SarahMemory-native universal HMD/VR headset witness bridge.

This driver is not an OpenXR/SteamVR/OpenVR/Monado launcher. It provides
read-only native HMD discovery/profile evidence for SarahMemoryMSDC and app.py.
PSVR v1 processor-box USB+HDMI is the first proof profile; Meta/high-end devices
remain adapter targets behind the same SarahMemory contract.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.vr.headset.usb"
DRIVER_NAME = "SarahMemory Native Universal HMD Bridge Driver"
VERSION = "3.0.0"

_DEFAULTS: Dict[str, Any] = {'enabled': True, 'autoload': False, 'runtime_preference': ['sarahmemory_native'], 'external_runtime_allowed': False, 'allow_runtime_launch': False, 'allow_scene_requests': False, 'allow_secondary_display': True, 'allow_pose_tracking': False, 'allow_eye_tracking': False, 'allow_hand_tracking': False, 'allow_passthrough': False, 'allow_microphone': False, 'allow_vendor_tools': False, 'selected_profile': 'psvr_v1_processor_box', 'auto_start_on_headset_connected': True, 'auto_stop_on_headset_disconnected': True, 'poll_interval_ms': 1000, 'history_limit': 200, 'native_profile': {'profile_id': 'psvr_v1_processor_box', 'vendor_family': 'sony_psvr', 'transport': ['usb', 'hdmi'], 'display_mode': 'secondary_surface', 'width': 1920, 'height': 1080, 'refresh_hz': 60, 'render_mode': 'mono_mirror', 'stereo_split': False, 'lens_correction': False, 'requires_external_runtime': False}, 'headset_profiles': {'psvr_v1_processor_box': {'profile_id': 'psvr_v1_processor_box', 'vendor_family': 'sony_psvr', 'transport': ['usb', 'hdmi'], 'display_mode': 'secondary_surface', 'width': 1920, 'height': 1080, 'refresh_hz': 60, 'render_mode': 'mono_mirror', 'stereo_split': False, 'lens_correction': False, 'requires_external_runtime': False, 'usb_match_keywords': ['playstation vr', 'psvr', 'sony', 'morpheus']}, 'generic_hdmi_hmd': {'profile_id': 'generic_hdmi_hmd', 'vendor_family': 'generic', 'transport': ['usb', 'hdmi', 'displayport'], 'display_mode': 'secondary_surface', 'width': 1920, 'height': 1080, 'refresh_hz': 60, 'render_mode': 'mono_mirror', 'stereo_split': False, 'lens_correction': False, 'requires_external_runtime': False}, 'meta_quest_generic': {'profile_id': 'meta_quest_generic', 'vendor_family': 'meta', 'transport': ['usb_c', 'wifi'], 'display_mode': 'encoded_stream_or_link_surface_future', 'width': 1920, 'height': 1080, 'refresh_hz': 72, 'render_mode': 'mono_mirror', 'stereo_split': False, 'lens_correction': False, 'requires_external_runtime': False, 'status': 'adapter_future'}}}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "ready",
    "error": None,
    "config": deepcopy(_DEFAULTS),
    "history": [],
    "last_detect": {},
}


def _ts() -> float:
    return time.time()


def _push(kind: str, detail: Any) -> None:
    _SESSION.setdefault("history", []).append({"ts": _ts(), "kind": kind, "detail": _json_safe(detail)})
    limit = int((_SESSION.get("config") or {}).get("history_limit", 200) or 200)
    _SESSION["history"] = _SESSION["history"][-limit:]


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: List[str], timeout: float = 8.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        # merge one level deep for profiles
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                m = dict(merged[k])
                m.update(v)
                merged[k] = m
            else:
                merged[k] = v
    merged["runtime_preference"] = ["sarahmemory_native"]
    merged["external_runtime_allowed"] = False
    merged["allow_runtime_launch"] = False
    return merged


def _probe_backends() -> Dict[str, Any]:
    return {
        "ok": True,
        "platform": platform.system(),
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": False,
        "tools": {
            "powershell": _which("powershell") or _which("pwsh"),
            "lsusb": _which("lsusb"),
            "system_profiler": _which("system_profiler"),
        },
        "note": "External XR runtimes are intentionally not used by the SarahMemory native HMD path.",
    }


def _candidate_keywords() -> List[str]:
    cfg = _SESSION.get("config") or {}
    selected = str(cfg.get("selected_profile") or "psvr_v1_processor_box")
    profiles = cfg.get("headset_profiles") if isinstance(cfg.get("headset_profiles"), dict) else {}
    profile = profiles.get(selected) if isinstance(profiles.get(selected), dict) else {}
    base = ["playstation vr", "psvr", "sony", "morpheus", "oculus", "quest", "meta", "vive", "valve", "index", "hmd", "vr headset"]
    extra = profile.get("usb_match_keywords") if isinstance(profile.get("usb_match_keywords"), list) else []
    return sorted(set(str(x).lower() for x in base + extra if str(x).strip()))


def _classify_candidate(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("playstation vr", "psvr", "morpheus", "sony")):
        return "psvr_v1_processor_box"
    if any(k in t for k in ("quest", "oculus", "meta")):
        return "meta_quest_generic"
    if any(k in t for k in ("vive", "index", "hmd", "vr")):
        return "generic_hdmi_hmd"
    return "unknown_hmd"


def _windows_usb_devices() -> List[Dict[str, Any]]:
    ps = _which("powershell") or _which("pwsh")
    if not ps:
        return []
    script = "Get-CimInstance Win32_PnPEntity | Where-Object {$_.Name -match 'VR|PSVR|PlayStation|Sony|Oculus|Meta|Quest|Vive|Index|HMD'} | Select-Object Name,PNPDeviceID,Manufacturer,Status | ConvertTo-Json -Depth 4"
    res = _run([ps, "-NoProfile", "-Command", script], timeout=12.0)
    if not res.get("ok"):
        return []
    try:
        data = json.loads(res.get("stdout") or "[]")
    except Exception:
        data = []
    if isinstance(data, dict):
        data = [data]
    out: List[Dict[str, Any]] = []
    for item in data or []:
        if isinstance(item, dict):
            raw = " ".join(str(item.get(k) or "") for k in ("Name", "Manufacturer", "PNPDeviceID", "Status"))
            out.append({
                "name": item.get("Name"),
                "manufacturer": item.get("Manufacturer"),
                "device_id": item.get("PNPDeviceID"),
                "status": item.get("Status"),
                "profile_hint": _classify_candidate(raw),
                "raw": raw,
            })
    return out


def _linux_usb_devices() -> List[Dict[str, Any]]:
    if not _which("lsusb"):
        return []
    res = _run(["lsusb"], timeout=8.0)
    if not res.get("ok"):
        return []
    out = []
    kws = _candidate_keywords()
    for line in (res.get("stdout") or "").splitlines():
        low = line.lower()
        if any(k in low for k in kws):
            out.append({"name": line.strip(), "status": "OK", "profile_hint": _classify_candidate(line), "raw": line.strip()})
    return out


def _mac_usb_devices() -> List[Dict[str, Any]]:
    if not _which("system_profiler"):
        return []
    res = _run(["system_profiler", "SPUSBDataType", "-json"], timeout=20.0)
    if not res.get("ok"):
        return []
    text = res.get("stdout") or ""
    kws = _candidate_keywords()
    if not any(k in text.lower() for k in kws):
        return []
    # Keep compact: return matched lines as evidence, not full profiler dump.
    lines = [ln.strip() for ln in text.splitlines() if any(k in ln.lower() for k in kws)]
    return [{"name": ln, "status": "OK", "profile_hint": _classify_candidate(ln), "raw": ln} for ln in lines[:20]]


def _detect_headsets() -> Dict[str, Any]:
    sysname = platform.system()
    if sysname == "Windows":
        devices = _windows_usb_devices()
    elif sysname == "Linux":
        devices = _linux_usb_devices()
    elif sysname == "Darwin":
        devices = _mac_usb_devices()
    else:
        devices = []
    connected = bool(devices)
    selected = str((_SESSION.get("config") or {}).get("selected_profile") or "psvr_v1_processor_box")
    if devices:
        selected = str(devices[0].get("profile_hint") or selected)
    payload = {
        "ok": True,
        "connected": connected,
        "headset_connected": connected,
        "native_runtime": "sarahmemory_native",
        "selected_profile": selected,
        "devices": devices,
        "count": len(devices),
        "external_runtime_allowed": False,
        "source": DRIVER_ID,
    }
    _SESSION["last_detect"] = payload
    _push("detect", payload)
    return payload


def _active_profile() -> Dict[str, Any]:
    cfg = _SESSION.get("config") or {}
    profiles = cfg.get("headset_profiles") if isinstance(cfg.get("headset_profiles"), dict) else {}
    selected = str(cfg.get("selected_profile") or "psvr_v1_processor_box")
    profile = profiles.get(selected) if isinstance(profiles.get(selected), dict) else cfg.get("native_profile")
    if not isinstance(profile, dict):
        profile = deepcopy(_DEFAULTS["native_profile"])
    out = deepcopy(profile)
    out.setdefault("profile_id", selected)
    out["requires_external_runtime"] = False
    return out


def driver_init(context: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _SESSION["config"] = _merge_config(config)
    _SESSION["started_ts"] = _ts()
    _SESSION["instance_id"] = "DRV-%s-%s" % (DRIVER_ID, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    _SESSION["status"] = "ready"
    _push("init", {"context": context or {}, "native_runtime": "sarahmemory_native"})
    return {"ok": True, "driver_id": DRIVER_ID, "version": VERSION, "native_runtime": "sarahmemory_native"}


def driver_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "driver_id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "status": _SESSION.get("status"),
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": False,
        "session": {"instance_id": _SESSION.get("instance_id"), "started_ts": _SESSION.get("started_ts")},
        "last_detect": deepcopy(_SESSION.get("last_detect") or {}),
        "config": deepcopy(_SESSION.get("config") or {}),
        "history_tail": deepcopy((_SESSION.get("history") or [])[-20:]),
    }


def driver_validate(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _merge_config(config)
    errors = []
    if cfg.get("external_runtime_allowed"):
        errors.append("external_runtime_allowed_must_remain_false_for_native_path")
    if cfg.get("runtime_preference") != ["sarahmemory_native"]:
        errors.append("runtime_preference_must_be_sarahmemory_native")
    return {"ok": not errors, "errors": errors, "config": cfg}


def driver_action(action_id: str, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    action = str(action_id or "").strip().lower()
    payload = payload if isinstance(payload, dict) else {}
    try:
        if action == "ping":
            return {"ok": True, "driver_id": DRIVER_ID, "ts": _ts()}
        if action in {"probe", "probe_backends"}:
            return _probe_backends()
        if action in {"discover", "discover_devices", "detect", "detect_headsets"}:
            return _detect_headsets()
        if action in {"native_hmd_status", "operator_hud_status"}:
            detect = _detect_headsets()
            return {
                "ok": True,
                "driver_id": DRIVER_ID,
                "native_hmd": detect,
                "connected": bool(detect.get("connected")),
                "headset_connected": bool(detect.get("connected")),
                "active_profile": _active_profile(),
                "native_runtime": "sarahmemory_native",
                "external_runtime_allowed": False,
                "mode": "OBSERVE_ONLY",
                "movement_authority": False,
                "telemetry_surface": True,
            }
        if action == "native_headset_profile":
            return {"ok": True, "driver_id": DRIVER_ID, "active_profile": _active_profile(), "profiles": deepcopy((_SESSION.get("config") or {}).get("headset_profiles") or {})}
        if action == "build_operator_hud_request":
            detect = _detect_headsets()
            return {
                "ok": True,
                "driver_id": DRIVER_ID,
                "request": {
                    "type": "sarahmemory_native_vr_hud_request",
                    "profile": _active_profile(),
                    "connected": bool(detect.get("connected")),
                    "display_role": "operator_vr_surface",
                    "movement_lock": True,
                    "requires_external_runtime": False,
                },
                "detect": detect,
            }
        if action == "get_config":
            return {"ok": True, "config": deepcopy(_SESSION.get("config") or {})}
        if action == "set_config":
            cfg = deepcopy(_SESSION.get("config") or {})
            patch = payload.get("config") if isinstance(payload.get("config"), dict) else payload
            cfg.update(patch)
            valid = driver_validate(cfg)
            if not valid.get("ok"):
                return {"ok": False, "error": "validation_failed", "details": valid}
            _SESSION["config"] = valid["config"]
            _push("set_config", patch)
            return {"ok": True, "config": deepcopy(_SESSION["config"]), "validate": valid}
        if action == "describe_capabilities":
            return {"ok": True, "capabilities": {"sarahmemory_native_hmd": True, "psvr_profile": True, "universal_profiles": True, "external_xr_runtime_dependency": False, "movement_authority": False}}
        if action in {"safe_stop", "stop"}:
            _SESSION["status"] = "ready"
            _push("safe_stop", payload)
            return {"ok": True, "stopped": True, "note": "No external runtime was started; native HUD process is managed by app.py."}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        return {"ok": False, "error": str(exc), "action_id": action}
    return {"ok": False, "error": "action_not_supported", "action_id": action}


def driver_shutdown() -> Dict[str, Any]:
    _SESSION["status"] = "shutdown"
    _push("shutdown", {})
    return {"ok": True, "driver_id": DRIVER_ID, "shutdown": True}
