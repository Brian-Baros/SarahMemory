"""
--==The SarahMemory Project==--
 File: ./data/drivers/com.softdev0.[DRIVERNAME].[TYPE]/driver.py
 Purpose: SarahMemory AiOS governed boot-layer bridge driver.
 Part of the SarahMemory Companion AI-bot Platform
 Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
 www.linkedin.com/in/brian-baros-29962a176
 https://www.facebook.com/bbaros
 brian.baros@sarahmemory.com
 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
 https://www.sarahmemory.com
 https://api.sarahmemory.com
 https://ai.sarahmemory.com
 https://store.sarahmemory.com
 ==============================================================================================
"""
# com.softdev0.camera.uvc.usb/driver.py
# Purpose:
#   Governed USB/UVC camera driver package for SarahMemory AiOS.
# Design goals:
#   - Discover as many USB/UVC camera devices and controls as the host exposes.
#   - Degrade gracefully when optional backends/tools are absent.
#   - Keep one stable driver contract: init/status/action/shutdown.
#   - Support generic imaging, PTZ, optics, lighting, audio, and vendor-extension control paths.
# Notes:
#   - Actual supported features depend on OS drivers, the device firmware, and installed backend tools.
#   - This package does not assume every camera supports every feature; it exposes and routes what is available.

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_DRIVER_ID = "com.softdev0.camera.uvc.usb"
_DRIVER_NAME = "USB/UVC Camera Driver"
_DRIVER_VERSION = "2.0.0"

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # optional

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "config": {},
    "active_device": None,
    "active_backend": None,
    "capture_open": False,
    "audio_recording": False,
    "video_recording": False,
    "last_frame_ts": None,
    "last_frame_shape": None,
    "last_snapshot_path": None,
    "last_record_path": None,
    "last_audio_path": None,
    "last_result": None,
    "capabilities": {},
    "discovery_cache": [],
}

_STATE: Dict[str, Any] = {
    "capture": None,
    "writer": None,
    "record_process": None,
    "audio_process": None,
    "stream_thread": None,
    "stream_stop": threading.Event(),
    "stream_lock": threading.RLock(),
}

_GENERIC_CONTROL_ALIASES: Dict[str, List[str]] = {
    "brightness": ["brightness"],
    "contrast": ["contrast"],
    "saturation": ["saturation"],
    "sharpness": ["sharpness", "sharpness_auto"],
    "hue": ["hue"],
    "gamma": ["gamma"],
    "gain": ["gain"],
    "backlight": ["backlight_compensation", "backlight"],
    "white_balance_auto": ["auto_white_balance", "white_balance_auto"],
    "white_balance_temperature": ["white_balance_temperature", "white_balance"],
    "exposure_auto": ["auto_exposure", "exposure_auto"],
    "exposure": ["exposure_absolute", "exposure", "exposure_time_absolute"],
    "focus_auto": ["focus_auto", "autofocus", "continuous_focus"],
    "focus": ["focus_absolute", "focus", "focus_manual"],
    "zoom": ["zoom_absolute", "zoom"],
    "zoom_relative": ["zoom_relative"],
    "zoom_continuous": ["zoom_continuous"],
    "pan": ["pan_absolute", "pan"],
    "tilt": ["tilt_absolute", "tilt"],
    "pan_relative": ["pan_relative"],
    "tilt_relative": ["tilt_relative"],
    "pan_speed": ["pan_speed"],
    "tilt_speed": ["tilt_speed"],
    "roll": ["roll_absolute", "roll"],
    "iris": ["iris_absolute", "iris"],
    "power_line_frequency": ["power_line_frequency", "powerline_frequency"],
    "mirror": ["horizontal_flip", "mirror"],
    "flip": ["vertical_flip", "flip"],
    "led": ["led1_mode", "led_mode", "indicator_led", "torch", "fill_light"],
    "roi": ["roi", "region_of_interest"],
    "night_vision": ["night_mode", "night_vision", "ir_cut", "infrared_mode"],
    "thermal_palette": ["thermal_palette", "palette"],
    "temperature_unit": ["temperature_unit"],
}

_CV2_PROPS: Dict[str, Any] = {}
if cv2 is not None:
    _CV2_PROPS = {
        "brightness": getattr(cv2, "CAP_PROP_BRIGHTNESS", None),
        "contrast": getattr(cv2, "CAP_PROP_CONTRAST", None),
        "saturation": getattr(cv2, "CAP_PROP_SATURATION", None),
        "hue": getattr(cv2, "CAP_PROP_HUE", None),
        "gain": getattr(cv2, "CAP_PROP_GAIN", None),
        "exposure": getattr(cv2, "CAP_PROP_EXPOSURE", None),
        "gamma": getattr(cv2, "CAP_PROP_GAMMA", None),
        "sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", None),
        "focus": getattr(cv2, "CAP_PROP_FOCUS", None),
        "focus_auto": getattr(cv2, "CAP_PROP_AUTOFOCUS", None),
        "zoom": getattr(cv2, "CAP_PROP_ZOOM", None),
        "white_balance_temperature": getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None),
        "white_balance_auto": getattr(cv2, "CAP_PROP_AUTO_WB", None),
        "backlight": getattr(cv2, "CAP_PROP_BACKLIGHT", None),
        "pan": getattr(cv2, "CAP_PROP_PAN", None),
        "tilt": getattr(cv2, "CAP_PROP_TILT", None),
        "roll": getattr(cv2, "CAP_PROP_ROLL", None),
        "iris": getattr(cv2, "CAP_PROP_IRIS", None),
        "fps": getattr(cv2, "CAP_PROP_FPS", None),
        "width": getattr(cv2, "CAP_PROP_FRAME_WIDTH", None),
        "height": getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None),
        "convert_rgb": getattr(cv2, "CAP_PROP_CONVERT_RGB", None),
        "fourcc": getattr(cv2, "CAP_PROP_FOURCC", None),
        "buffercount": getattr(cv2, "CAP_PROP_BUFFERSIZE", None),
    }

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "device_index": 0,
    "device_path": "",
    "device_name": "",
    "vendor_id": "",
    "product_id": "",
    "preferred_backend": "auto",
    "backend_priority": ["opencv", "v4l2", "ffmpeg", "directshow", "avfoundation", "gstreamer"],
    "width": 1280,
    "height": 720,
    "fps": 30,
    "pixel_format": "MJPG",
    "fourcc": "MJPG",
    "buffer_size": 2,
    "open_timeout_sec": 10,
    "read_timeout_sec": 5,
    "frame_warmup_count": 4,
    "stream_preview": False,
    "mirror": False,
    "flip": False,
    "rotation_degrees": 0,
    "convert_rgb": True,
    "save_dir": "./data/captures/camera",
    "record_dir": "./data/captures/camera",
    "audio_dir": "./data/captures/audio",
    "snapshot_format": "jpg",
    "video_codec": "mp4v",
    "record_audio": True,
    "audio_backend": "auto",
    "audio_input_name": "",
    "audio_sample_rate": 48000,
    "audio_channels": 2,
    "audio_container": "wav",
    "ffmpeg_path": "ffmpeg",
    "ffprobe_path": "ffprobe",
    "v4l2_ctl_path": "v4l2-ctl",
    "media_ctl_path": "media-ctl",
    "lsusb_path": "lsusb",
    "pw_record_path": "pw-record",
    "arecord_path": "arecord",
    "powershell_path": "powershell",
    "enable_vendor_passthrough": True,
    "enable_audio_enumeration": True,
    "enable_ext_controls": True,
    "allow_unsafe_vendor_controls": False,
    "control_cache_ttl_sec": 5,
    "default_ptz_step": 3600,
    "default_focus_step": 5,
    "default_zoom_step": 1,
    "default_exposure_step": 1,
    "default_gain_step": 1,
    "default_pan_speed": 1,
    "default_tilt_speed": 1,
    "vendor_extension_hints": {
        "thermal": ["thermal_palette", "temperature_unit", "hotspot_tracking"],
        "infrared": ["night_vision", "ir_cut", "led"],
        "ptz": ["pan", "tilt", "zoom", "pan_speed", "tilt_speed"],
        "lighting": ["led", "backlight", "night_vision"],
        "audio": ["audio_input_name", "record_audio", "audio_sample_rate", "audio_channels"],
    },
}

def _now_ts() -> float:
    return time.time()

def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )

def _append_note(note: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append("[%s] %s" % (_utc_now(), note))
    if len(notes) > 100:
        _SESSION["notes"] = notes[-100:]

def _tool_exists(cmd: str) -> bool:
    if not cmd:
        return False
    return shutil.which(cmd) is not None or os.path.exists(cmd)

def _run(cmd: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}

def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(str(value), 0)
    except Exception:
        return default

def _norm_hex(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, int):
        return "0x%04x" % value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return text
    try:
        return "0x%04x" % int(text, 16)
    except Exception:
        return text

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    masked = deepcopy(cfg)
    for key in ("vendor_payload_b64",):
        if key in masked and masked[key]:
            masked[key] = "***"
    return masked

def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "usb",
        "session": deepcopy(_SESSION),
        "runtime": {
            "platform": platform.system(),
            "python": platform.python_version(),
            "cv2": bool(cv2 is not None),
            "tools": {
                "ffmpeg": _tool_exists(_SESSION.get("config", {}).get("ffmpeg_path", "ffmpeg")),
                "ffprobe": _tool_exists(_SESSION.get("config", {}).get("ffprobe_path", "ffprobe")),
                "v4l2-ctl": _tool_exists(_SESSION.get("config", {}).get("v4l2_ctl_path", "v4l2-ctl")),
                "media-ctl": _tool_exists(_SESSION.get("config", {}).get("media_ctl_path", "media-ctl")),
                "lsusb": _tool_exists(_SESSION.get("config", {}).get("lsusb_path", "lsusb")),
                "pw-record": _tool_exists(_SESSION.get("config", {}).get("pw_record_path", "pw-record")),
                "arecord": _tool_exists(_SESSION.get("config", {}).get("arecord_path", "arecord")),
            },
        },
    }

def driver_validate(config: dict) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    for key in ("enabled", "autoload", "stream_preview", "mirror", "flip", "record_audio",
                "enable_vendor_passthrough", "enable_audio_enumeration", "enable_ext_controls",
                "allow_unsafe_vendor_controls", "convert_rgb"):
        if key in config and not isinstance(config.get(key), bool):
            errors.append("%s must be a boolean" % key)
    for key in ("device_index", "width", "height", "fps", "buffer_size", "open_timeout_sec",
                "read_timeout_sec", "frame_warmup_count", "rotation_degrees",
                "audio_sample_rate", "audio_channels", "control_cache_ttl_sec",
                "default_ptz_step", "default_focus_step", "default_zoom_step",
                "default_exposure_step", "default_gain_step", "default_pan_speed", "default_tilt_speed"):
        if key in config and not isinstance(config.get(key), int):
            errors.append("%s must be an integer" % key)
    if "backend_priority" in config and not isinstance(config.get("backend_priority"), list):
        errors.append("backend_priority must be an array")
    if cv2 is None:
        warnings.append("opencv-python/cv2 not available; live frame capture via OpenCV is unavailable")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

def _config() -> Dict[str, Any]:
    return _SESSION.get("config", {})

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _is_linux() -> bool:
    return platform.system().lower() == "linux"

def _is_windows() -> bool:
    return platform.system().lower() == "windows"

def _is_macos() -> bool:
    return platform.system().lower() == "darwin"

def _list_v4l2_devices(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    tool = cfg.get("v4l2_ctl_path", "v4l2-ctl")
    if not _is_linux() or not _tool_exists(tool):
        return results
    data = _run([tool, "--list-devices"], timeout=10)
    if not data.get("ok"):
        return results
    current: Optional[Dict[str, Any]] = None
    for raw_line in data.get("stdout", "").splitlines():
        line = raw_line.rstrip()
        if not line:
            current = None
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            if current:
                results.append(current)
            current = {"name": line.rstrip(":"), "paths": []}
        else:
            if current is None:
                continue
            dev = line.strip()
            current.setdefault("paths", []).append(dev)
    if current:
        results.append(current)
    return results

def _list_lsusb(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    tool = cfg.get("lsusb_path", "lsusb")
    if not _is_linux() or not _tool_exists(tool):
        return results
    data = _run([tool], timeout=10)
    if not data.get("ok"):
        return results
    for line in data.get("stdout", "").splitlines():
        # Example: Bus 001 Device 004: ID 046d:0825 Logitech, Inc. Webcam C270
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "Bus" and parts[2].startswith("Device"):
            vidpid = parts[5]
            if ":" in vidpid:
                vid, pid = vidpid.split(":", 1)
            else:
                vid, pid = "", ""
            results.append({
                "bus": parts[1],
                "device": parts[3].rstrip(":"),
                "vendor_id": _norm_hex(vid),
                "product_id": _norm_hex(pid),
                "name": " ".join(parts[6:]).strip(),
                "raw": line,
            })
    return results

def _list_windows_cameras(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not _is_windows():
        return []
    ps = cfg.get("powershell_path", "powershell")
    if not _tool_exists(ps):
        return []
    script = (
        "Get-PnpDevice -Class Camera,Image | "
        "Select-Object FriendlyName,InstanceId,Status,Class | ConvertTo-Json -Depth 3"
    )
    data = _run([ps, "-NoProfile", "-Command", script], timeout=15)
    if not data.get("ok") or not data.get("stdout", "").strip():
        return []
    try:
        obj = json.loads(data["stdout"])
        items = obj if isinstance(obj, list) else [obj]
        out = []
        for item in items:
            out.append({
                "name": item.get("FriendlyName"),
                "instance_id": item.get("InstanceId"),
                "status": item.get("Status"),
                "class": item.get("Class"),
                "backend": "windows-pnp",
            })
        return out
    except Exception:
        return []

def _list_avfoundation_cameras() -> List[Dict[str, Any]]:
    # Placeholder for macOS. Enumerating AVFoundation devices robustly requires PyObjC or ffmpeg probing.
    return []

def _cv2_backend_constant(name: str) -> Any:
    if cv2 is None:
        return None
    mapping = {
        "auto": None,
        "opencv": None,
        "v4l2": getattr(cv2, "CAP_V4L2", None),
        "directshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
        "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
        "any": getattr(cv2, "CAP_ANY", None),
    }
    return mapping.get((name or "auto").lower())

def _discover_cv2_indices(cfg: Dict[str, Any], max_indices: int = 8) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if cv2 is None:
        return results
    backends = cfg.get("backend_priority") or ["opencv", "v4l2", "directshow", "avfoundation", "ffmpeg"]
    seen: set = set()
    for idx in range(max_indices):
        device_entry = {"index": idx, "backends": [], "name": "Camera %s" % idx}
        opened = False
        for backend_name in backends:
            backend_const = _cv2_backend_constant(backend_name)
            try:
                cap = cv2.VideoCapture(idx, backend_const) if backend_const is not None else cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    opened = True
                    width = int(cap.get(_CV2_PROPS.get("width")) or 0) if _CV2_PROPS.get("width") is not None else 0
                    height = int(cap.get(_CV2_PROPS.get("height")) or 0) if _CV2_PROPS.get("height") is not None else 0
                    fps = float(cap.get(_CV2_PROPS.get("fps")) or 0) if _CV2_PROPS.get("fps") is not None else 0
                    device_entry["backends"].append({
                        "backend": backend_name,
                        "width": width,
                        "height": height,
                        "fps": fps,
                    })
                    cap.release()
            except Exception:
                pass
        if opened and idx not in seen:
            results.append(device_entry)
            seen.add(idx)
    return results

def _parse_v4l2_ctrls_text(text: str) -> Dict[str, Dict[str, Any]]:
    controls: Dict[str, Dict[str, Any]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        key_part = left.strip().split()
        name = key_part[0].strip()
        entry: Dict[str, Any] = {"raw": line, "name": name}
        for token in right.strip().split():
            if "=" in token:
                k, v = token.split("=", 1)
                entry[k] = v
        controls[name] = entry
    return controls

def _get_v4l2_controls(device_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tool = cfg.get("v4l2_ctl_path", "v4l2-ctl")
    if not (_is_linux() and device_path and _tool_exists(tool)):
        return {}
    data = _run([tool, "-d", device_path, "-L"], timeout=15)
    if not data.get("ok"):
        return {}
    return _parse_v4l2_ctrls_text(data.get("stdout", ""))

def _get_ffprobe_streams(source: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tool = cfg.get("ffprobe_path", "ffprobe")
    if not _tool_exists(tool) or not source:
        return {"ok": False, "error": "ffprobe unavailable or no source"}
    data = _run([
        tool, "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", source
    ], timeout=15)
    if not data.get("ok"):
        return {"ok": False, "details": data}
    try:
        return {"ok": True, "probe": json.loads(data.get("stdout", "{}") or "{}")}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "details": data}

def _list_audio_inputs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cfg.get("enable_audio_enumeration", True):
        return []
    results: List[Dict[str, Any]] = []
    ffmpeg = cfg.get("ffmpeg_path", "ffmpeg")
    if _is_windows() and _tool_exists(ffmpeg):
        data = _run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], timeout=20)
        text = (data.get("stderr") or "") + "\n" + (data.get("stdout") or "")
        section = None
        for line in text.splitlines():
            lower = line.lower()
            if "directshow video devices" in lower:
                section = "video"
            elif "directshow audio devices" in lower:
                section = "audio"
            elif section == "audio" and '"' in line:
                start = line.find('"')
                end = line.rfind('"')
                if start >= 0 and end > start:
                    results.append({"name": line[start + 1:end], "backend": "dshow"})
    elif _is_linux():
        for tool_name, parser in (
            (cfg.get("arecord_path", "arecord"), "arecord"),
            (cfg.get("pw_record_path", "pw-record"), "pipewire"),
        ):
            if _tool_exists(tool_name):
                if parser == "arecord":
                    data = _run([tool_name, "-l"], timeout=10)
                    for line in data.get("stdout", "").splitlines():
                        if line.strip().startswith("card "):
                            results.append({"name": line.strip(), "backend": "alsa"})
                else:
                    results.append({"name": "default", "backend": "pipewire"})
    return results

def _discover_devices() -> List[Dict[str, Any]]:
    cfg = _config()
    devices: List[Dict[str, Any]] = []
    cv2_devices = _discover_cv2_indices(cfg)
    v4l2_devices = _list_v4l2_devices(cfg)
    lsusb_devices = _list_lsusb(cfg)
    windows_devices = _list_windows_cameras(cfg)
    mac_devices = _list_avfoundation_cameras()

    for item in cv2_devices:
        devices.append({
            "type": "camera",
            "discovered_by": ["opencv"],
            "index": item.get("index"),
            "name": item.get("name"),
            "backends": item.get("backends", []),
        })

    for group in v4l2_devices:
        entry = {
            "type": "camera",
            "discovered_by": ["v4l2"],
            "name": group.get("name"),
            "paths": group.get("paths", []),
            "path": (group.get("paths") or [None])[0],
            "controls": {},
        }
        if entry["path"]:
            entry["controls"] = _get_v4l2_controls(entry["path"], cfg)
        devices.append(entry)

    for item in lsusb_devices:
        devices.append({
            "type": "usb",
            "discovered_by": ["lsusb"],
            **item,
        })

    for item in windows_devices:
        devices.append({
            "type": "camera",
            "discovered_by": ["windows-pnp"],
            **item,
        })

    for item in mac_devices:
        devices.append({
            "type": "camera",
            "discovered_by": ["avfoundation"],
            **item,
        })

    _SESSION["discovery_cache"] = deepcopy(devices)
    return devices

def _score_device(device: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    score = 0
    if cfg.get("device_path") and cfg.get("device_path") in (device.get("path"), *(device.get("paths") or [])):
        score += 100
    if cfg.get("device_name") and cfg.get("device_name").lower() in str(device.get("name", "")).lower():
        score += 60
    if cfg.get("vendor_id") and cfg.get("vendor_id").lower() == str(device.get("vendor_id", "")).lower():
        score += 40
    if cfg.get("product_id") and cfg.get("product_id").lower() == str(device.get("product_id", "")).lower():
        score += 30
    if device.get("index") == cfg.get("device_index"):
        score += 20
    if device.get("type") == "camera":
        score += 10
    return score

def _select_device(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _deep_merge(_config(), payload.get("config", {}))
    devices = _discover_devices()
    if not devices:
        return {"ok": False, "error": "no_devices_found"}
    ranked = sorted(devices, key=lambda d: _score_device(d, cfg), reverse=True)
    selected = ranked[0]
    _SESSION["active_device"] = deepcopy(selected)
    return {"ok": True, "device": selected, "devices": ranked}

def _apply_cv2_property(cap: Any, key: str, value: Any) -> Tuple[bool, str]:
    prop_id = _CV2_PROPS.get(key)
    if prop_id is None or cap is None:
        return False, "cv2_property_unavailable"
    try:
        ok = bool(cap.set(prop_id, float(value)))
        return ok, "cv2_set_ok" if ok else "cv2_set_rejected"
    except Exception as exc:
        return False, str(exc)

def _open_capture(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _deep_merge(_config(), payload.get("config", {}))
    with _STATE["stream_lock"]:
        if _STATE["capture"] is not None:
            return {"ok": True, "already_open": True, "device": deepcopy(_SESSION.get("active_device"))}
        sel = _select_device({"config": cfg})
        if not sel.get("ok"):
            _SESSION["error"] = sel.get("error")
            return sel
        device = sel["device"]
        if cv2 is None:
            return {"ok": False, "error": "cv2_unavailable"}
        backend_name = (cfg.get("preferred_backend") or "auto").lower()
        backend_const = _cv2_backend_constant(backend_name)
        source: Any = device.get("index", cfg.get("device_index", 0))
        if device.get("path") and _is_linux():
            source = device.get("path")
        cap = cv2.VideoCapture(source, backend_const) if backend_const is not None else cv2.VideoCapture(source)
        if cap is None or not cap.isOpened():
            return {"ok": False, "error": "camera_open_failed", "source": source, "backend": backend_name}
        # Apply base configuration
        for key, cfg_key in (("width", "width"), ("height", "height"), ("fps", "fps"), ("convert_rgb", "convert_rgb"), ("buffercount", "buffer_size")):
            if _CV2_PROPS.get(key) is not None and cfg.get(cfg_key) is not None:
                try:
                    cap.set(_CV2_PROPS[key], float(cfg[cfg_key]) if isinstance(cfg[cfg_key], (int, float, bool)) else cfg[cfg_key])
                except Exception:
                    pass
        if cfg.get("fourcc") and _CV2_PROPS.get("fourcc") is not None:
            try:
                fourcc = cv2.VideoWriter_fourcc(*str(cfg["fourcc"])[:4])
                cap.set(_CV2_PROPS["fourcc"], fourcc)
            except Exception:
                pass
        # Warm-up frames
        warmup = max(0, int(cfg.get("frame_warmup_count", 0)))
        for _ in range(warmup):
            try:
                cap.read()
            except Exception:
                break
        _STATE["capture"] = cap
        _SESSION["capture_open"] = True
        _SESSION["active_backend"] = backend_name
        _SESSION["active_device"] = deepcopy(device)
        _SESSION["status"] = "stream_ready"
        _SESSION["error"] = None
        _append_note("Capture opened on source=%s backend=%s" % (source, backend_name))
        return {"ok": True, "device": deepcopy(device), "backend": backend_name, "source": source}

def _close_capture() -> Dict[str, Any]:
    with _STATE["stream_lock"]:
        cap = _STATE.get("capture")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _STATE["capture"] = None
        _SESSION["capture_open"] = False
        if _SESSION["status"] != "stopped":
            _SESSION["status"] = "ready"
        _append_note("Capture closed")
        return {"ok": True}

def _read_frame(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    if _STATE.get("capture") is None:
        opened = _open_capture(payload)
        if not opened.get("ok"):
            return opened
    cap = _STATE.get("capture")
    if cap is None:
        return {"ok": False, "error": "capture_unavailable"}
    ok, frame = cap.read()
    if not ok or frame is None:
        return {"ok": False, "error": "frame_read_failed"}
    # Mirror/flip are applied logically for snapshots/preview metadata; raw frame remains camera-native unless requested.
    if _config().get("mirror"):
        try:
            frame = cv2.flip(frame, 1)
        except Exception:
            pass
    if _config().get("flip"):
        try:
            frame = cv2.flip(frame, 0)
        except Exception:
            pass
    rot = int(_config().get("rotation_degrees") or 0)
    if cv2 is not None and rot in (90, 180, 270):
        rotate_code = {
            90: getattr(cv2, "ROTATE_90_CLOCKWISE", None),
            180: getattr(cv2, "ROTATE_180", None),
            270: getattr(cv2, "ROTATE_90_COUNTERCLOCKWISE", None),
        }.get(rot)
        if rotate_code is not None:
            try:
                frame = cv2.rotate(frame, rotate_code)
            except Exception:
                pass
    _SESSION["last_frame_ts"] = _now_ts()
    try:
        _SESSION["last_frame_shape"] = list(frame.shape)
    except Exception:
        _SESSION["last_frame_shape"] = None
    return {"ok": True, "frame": frame, "shape": _SESSION["last_frame_shape"], "ts": _SESSION["last_frame_ts"]}

def _save_snapshot(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    if cv2 is None:
        return {"ok": False, "error": "cv2_unavailable"}
    frame_result = _read_frame(payload)
    if not frame_result.get("ok"):
        return frame_result
    frame = frame_result["frame"]
    cfg = _config()
    out_dir = _ensure_dir(payload.get("save_dir") or cfg.get("save_dir") or "./data/captures/camera")
    fmt = str(payload.get("format") or cfg.get("snapshot_format") or "jpg").lower()
    if fmt not in ("jpg", "jpeg", "png", "bmp", "tiff", "webp"):
        fmt = "jpg"
    out_path = payload.get("path") or os.path.join(out_dir, "snapshot_%s.%s" % (datetime.utcnow().strftime("%Y%m%dT%H%M%S"), fmt))
    ok = cv2.imwrite(out_path, frame)
    if not ok:
        return {"ok": False, "error": "snapshot_write_failed", "path": out_path}
    _SESSION["last_snapshot_path"] = out_path
    _append_note("Snapshot saved to %s" % out_path)
    return {"ok": True, "path": out_path, "shape": frame_result.get("shape")}

def _start_video_recording(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _config()
    if _SESSION.get("video_recording"):
        return {"ok": True, "already_recording": True, "path": _SESSION.get("last_record_path")}
    if cv2 is None:
        return {"ok": False, "error": "cv2_unavailable"}
    open_result = _open_capture(payload)
    if not open_result.get("ok"):
        return open_result
    frame_info = _read_frame(payload)
    if not frame_info.get("ok"):
        return frame_info
    frame = frame_info["frame"]
    height, width = frame.shape[:2]
    codec = str(payload.get("codec") or cfg.get("video_codec") or "mp4v")[:4]
    out_dir = _ensure_dir(payload.get("record_dir") or cfg.get("record_dir") or "./data/captures/camera")
    out_path = payload.get("path") or os.path.join(out_dir, "record_%s.mp4" % datetime.utcnow().strftime("%Y%m%dT%H%M%S"))
    try:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), float(cfg.get("fps", 30)), (width, height))
        if not writer.isOpened():
            return {"ok": False, "error": "video_writer_open_failed", "path": out_path}
        _STATE["writer"] = writer
        _SESSION["video_recording"] = True
        _SESSION["last_record_path"] = out_path
        _append_note("Video recording started: %s" % out_path)
        return {"ok": True, "path": out_path, "codec": codec, "size": [width, height]}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": out_path}

def _write_record_frame(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _SESSION.get("video_recording"):
        return {"ok": False, "error": "video_not_recording"}
    frame_info = _read_frame(payload)
    if not frame_info.get("ok"):
        return frame_info
    writer = _STATE.get("writer")
    if writer is None:
        return {"ok": False, "error": "writer_unavailable"}
    try:
        writer.write(frame_info["frame"])
        return {"ok": True, "ts": frame_info.get("ts"), "shape": frame_info.get("shape")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

def _stop_video_recording() -> Dict[str, Any]:
    writer = _STATE.get("writer")
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    _STATE["writer"] = None
    was = bool(_SESSION.get("video_recording"))
    _SESSION["video_recording"] = False
    if was:
        _append_note("Video recording stopped")
    return {"ok": True, "path": _SESSION.get("last_record_path"), "was_recording": was}

def _audio_record_cmd(out_path: str, cfg: Dict[str, Any], seconds: Optional[int] = None) -> Optional[List[str]]:
    ffmpeg = cfg.get("ffmpeg_path", "ffmpeg")
    if _tool_exists(ffmpeg):
        if _is_windows():
            input_name = cfg.get("audio_input_name") or "audio="
            cmd = [ffmpeg, "-y"]
            if seconds:
                cmd += ["-t", str(seconds)]
            cmd += ["-f", "dshow", "-i", "audio=%s" % input_name, out_path]
            return cmd
        elif _is_linux():
            if _tool_exists(cfg.get("pw_record_path", "pw-record")):
                cmd = [cfg.get("pw_record_path", "pw-record"), "--target", cfg.get("audio_input_name") or "0"]
                if seconds:
                    cmd += ["--duration", str(seconds)]
                cmd += [out_path]
                return cmd
            elif _tool_exists(cfg.get("arecord_path", "arecord")):
                cmd = [cfg.get("arecord_path", "arecord"), "-f", "cd", "-r", str(cfg.get("audio_sample_rate", 48000)), "-c", str(cfg.get("audio_channels", 2))]
                if seconds:
                    cmd += ["-d", str(seconds)]
                cmd += [out_path]
                return cmd
    return None

def _start_audio_recording(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _config()
    if _SESSION.get("audio_recording"):
        return {"ok": True, "already_recording": True, "path": _SESSION.get("last_audio_path")}
    out_dir = _ensure_dir(payload.get("audio_dir") or cfg.get("audio_dir") or "./data/captures/audio")
    ext = str(cfg.get("audio_container") or "wav").lower()
    out_path = payload.get("path") or os.path.join(out_dir, "audio_%s.%s" % (datetime.utcnow().strftime("%Y%m%dT%H%M%S"), ext))
    cmd = _audio_record_cmd(out_path, cfg)
    if not cmd:
        return {"ok": False, "error": "audio_backend_unavailable"}
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _STATE["audio_process"] = proc
        _SESSION["audio_recording"] = True
        _SESSION["last_audio_path"] = out_path
        _append_note("Audio recording started: %s" % out_path)
        return {"ok": True, "path": out_path, "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "path": out_path}

def _stop_audio_recording() -> Dict[str, Any]:
    proc = _STATE.get("audio_process")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    _STATE["audio_process"] = None
    was = bool(_SESSION.get("audio_recording"))
    _SESSION["audio_recording"] = False
    if was:
        _append_note("Audio recording stopped")
    return {"ok": True, "path": _SESSION.get("last_audio_path"), "was_recording": was}

def _enumerate_controls(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    sel = _select_device(payload)
    if not sel.get("ok"):
        return sel
    device = sel["device"]
    cfg = _config()
    controls: Dict[str, Any] = {}
    if device.get("path") and _is_linux():
        controls = _get_v4l2_controls(device["path"], cfg)
    control_aliases = {}
    for generic, aliases in _GENERIC_CONTROL_ALIASES.items():
        present = [a for a in aliases if a in controls]
        if present:
            control_aliases[generic] = present
    result = {
        "ok": True,
        "device": device,
        "controls": controls,
        "generic_aliases": control_aliases,
        "vendor_extensions_possible": bool(cfg.get("enable_ext_controls", True)),
    }
    _SESSION["capabilities"] = {
        "controls": list(controls.keys()),
        "generic_aliases": control_aliases,
    }
    return result

def _find_v4l2_control_name(generic_name: str, controls: Dict[str, Any]) -> Optional[str]:
    aliases = _GENERIC_CONTROL_ALIASES.get(generic_name, [generic_name])
    for alias in aliases:
        if alias in controls:
            return alias
    if generic_name in controls:
        return generic_name
    return None

def _set_v4l2_control(device_path: str, control_name: str, value: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tool = cfg.get("v4l2_ctl_path", "v4l2-ctl")
    if not (_is_linux() and device_path and _tool_exists(tool)):
        return {"ok": False, "error": "v4l2_ctl_unavailable"}
    cmd = [tool, "-d", device_path, "-c", "%s=%s" % (control_name, value)]
    data = _run(cmd, timeout=10)
    return {"ok": data.get("ok"), "details": data, "control": control_name, "value": value}

def _set_control(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    control = str(payload.get("control") or payload.get("name") or "").strip()
    value = payload.get("value")
    if not control:
        return {"ok": False, "error": "missing_control"}
    sel = _select_device(payload)
    if not sel.get("ok"):
        return sel
    device = sel["device"]
    controls = _enumerate_controls(payload).get("controls", {})
    # OpenCV path first for common properties if capture is open or openable.
    if control in _CV2_PROPS and cv2 is not None:
        open_result = _open_capture(payload)
        if open_result.get("ok"):
            ok, msg = _apply_cv2_property(_STATE.get("capture"), control, value)
            if ok:
                _append_note("Set control via OpenCV: %s=%s" % (control, value))
                return {"ok": True, "path": "opencv", "control": control, "value": value, "message": msg}
    # Linux v4l2 path
    if device.get("path") and _is_linux():
        ctrl_name = _find_v4l2_control_name(control, controls) or control
        result = _set_v4l2_control(device["path"], ctrl_name, value, _config())
        if result.get("ok"):
            _append_note("Set control via V4L2: %s=%s" % (ctrl_name, value))
        return result
    return {"ok": False, "error": "control_path_unavailable", "control": control, "value": value}

def _get_control(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    control = str(payload.get("control") or payload.get("name") or "").strip()
    if not control:
        return {"ok": False, "error": "missing_control"}
    enum_result = _enumerate_controls(payload)
    if not enum_result.get("ok"):
        return enum_result
    controls = enum_result.get("controls", {})
    ctrl_name = _find_v4l2_control_name(control, controls) or control
    if ctrl_name in controls:
        return {"ok": True, "control": ctrl_name, "info": controls[ctrl_name]}
    if control in _CV2_PROPS and _STATE.get("capture") is not None:
        cap = _STATE["capture"]
        prop_id = _CV2_PROPS.get(control)
        try:
            return {"ok": True, "control": control, "value": cap.get(prop_id)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": "control_not_found", "control": control}

def _ptz_action(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    action = str(payload.get("ptz_action") or payload.get("action") or "").strip().lower()
    step = payload.get("step", _config().get("default_ptz_step", 3600))
    if action in ("pan_left", "left"):
        return _set_control({"control": "pan_relative", "value": -abs(int(step))})
    if action in ("pan_right", "right"):
        return _set_control({"control": "pan_relative", "value": abs(int(step))})
    if action in ("tilt_up", "up"):
        return _set_control({"control": "tilt_relative", "value": abs(int(step))})
    if action in ("tilt_down", "down"):
        return _set_control({"control": "tilt_relative", "value": -abs(int(step))})
    if action in ("center", "home", "reset"):
        out = []
        for ctl in ("pan_reset", "tilt_reset"):
            out.append(_set_control({"control": ctl, "value": 1}))
        return {"ok": any(item.get("ok") for item in out), "results": out}
    if action in ("stop",):
        out = [
            _set_control({"control": "pan_speed", "value": 0}),
            _set_control({"control": "tilt_speed", "value": 0}),
            _set_control({"control": "zoom_continuous", "value": 0}),
        ]
        return {"ok": any(item.get("ok") for item in out), "results": out}
    return {"ok": False, "error": "unsupported_ptz_action", "ptz_action": action}

def _vendor_passthrough(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _config()
    if not cfg.get("enable_vendor_passthrough", True):
        return {"ok": False, "error": "vendor_passthrough_disabled"}
    device_path = payload.get("device_path") or (_SESSION.get("active_device") or {}).get("path")
    if _is_linux() and device_path and _tool_exists(cfg.get("v4l2_ctl_path", "v4l2-ctl")):
        # Generic passthrough for control names and ext-ctrls via v4l2-ctl.
        if payload.get("ext_controls"):
            cmd = [cfg.get("v4l2_ctl_path", "v4l2-ctl"), "-d", device_path, "--set-ctrl", str(payload["ext_controls"])]
            data = _run(cmd, timeout=10)
            return {"ok": data.get("ok"), "details": data}
        if payload.get("command"):
            cmd = [cfg.get("v4l2_ctl_path", "v4l2-ctl"), "-d", device_path] + list(payload["command"])
            data = _run(cmd, timeout=15)
            return {"ok": data.get("ok"), "details": data}
    return {"ok": False, "error": "vendor_passthrough_unavailable"}

def _probe_capabilities(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    sel = _select_device(payload)
    if not sel.get("ok"):
        return sel
    device = sel["device"]
    cfg = _config()
    controls_result = _enumerate_controls(payload)
    ffprobe_result = {"ok": False, "error": "probe_skipped"}
    source = device.get("path") or payload.get("source")
    if source:
        ffprobe_result = _get_ffprobe_streams(source, cfg)
    result = {
        "ok": True,
        "device": device,
        "controls": controls_result.get("controls", {}),
        "generic_aliases": controls_result.get("generic_aliases", {}),
        "audio_inputs": _list_audio_inputs(cfg),
        "probe": ffprobe_result,
        "backends": {
            "cv2": bool(cv2 is not None),
            "ffmpeg": _tool_exists(cfg.get("ffmpeg_path", "ffmpeg")),
            "v4l2": _tool_exists(cfg.get("v4l2_ctl_path", "v4l2-ctl")),
            "media_ctl": _tool_exists(cfg.get("media_ctl_path", "media-ctl")),
        },
    }
    _SESSION["capabilities"] = deepcopy(result)
    return result

def _safe_stop() -> Dict[str, Any]:
    _STATE["stream_stop"].set()
    _stop_video_recording()
    _stop_audio_recording()
    _close_capture()
    _SESSION["status"] = "ready"
    _append_note("Safe stop completed")
    return {"ok": True}

def driver_init(context: Optional[dict] = None, config: Optional[dict] = None) -> Dict[str, Any]:
    config = config or {}
    merged = _deep_merge(_DEFAULT_CONFIG, config)
    v = driver_validate(merged)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        _SESSION["config"] = _mask_config(merged)
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = _now_ts()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = []
    _SESSION["config"] = merged
    _SESSION["active_device"] = None
    _SESSION["active_backend"] = None
    _SESSION["capture_open"] = False
    _SESSION["audio_recording"] = False
    _SESSION["video_recording"] = False
    _append_note("Driver initialized")
    return {
        "ok": True,
        "instance_id": _SESSION["instance_id"],
        "validate": v,
        "info": driver_info(),
        "discovery": _discover_devices(),
    }

def driver_shutdown(context: Optional[dict] = None) -> bool:
    _safe_stop()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _append_note("Shutdown completed")
    return True

def driver_status(context: Optional[dict] = None) -> Dict[str, Any]:
    status = {
        "ok": True,
        "session": deepcopy(_SESSION),
        "runtime": driver_info().get("runtime"),
        "devices": deepcopy(_SESSION.get("discovery_cache") or []),
    }
    return status

def driver_action(action_id: str, context: Optional[dict] = None, payload: Optional[dict] = None) -> Dict[str, Any]:
    payload = payload or {}
    action = (action_id or "").strip().lower()

    if action == "ping":
        return {"ok": True, "pong": True, "ts": _now_ts()}
    if action in ("discover", "discover_devices", "list_devices", "scan"):
        return {"ok": True, "devices": _discover_devices(), "audio_inputs": _list_audio_inputs(_config())}
    if action in ("select_device",):
        return _select_device(payload)
    if action in ("probe_capabilities", "capabilities", "probe"):
        return _probe_capabilities(payload)
    if action in ("enumerate_controls", "list_controls", "controls"):
        return _enumerate_controls(payload)
    if action in ("open_stream", "open", "start_capture"):
        return _open_capture(payload)
    if action in ("close_stream", "close", "stop_capture"):
        return _close_capture()
    if action in ("read_frame", "grab_frame", "frame_info"):
        result = _read_frame(payload)
        if result.get("ok"):
            # Do not serialize the raw frame in action return.
            result = {k: v for k, v in result.items() if k != "frame"}
        return result
    if action in ("save_snapshot", "snapshot", "capture_photo"):
        return _save_snapshot(payload)
    if action in ("start_recording", "record_video_start"):
        return _start_video_recording(payload)
    if action in ("record_frame", "write_record_frame"):
        return _write_record_frame(payload)
    if action in ("stop_recording", "record_video_stop"):
        return _stop_video_recording()
    if action in ("start_audio_recording", "record_audio_start"):
        return _start_audio_recording(payload)
    if action in ("stop_audio_recording", "record_audio_stop"):
        return _stop_audio_recording()
    if action in ("set_control", "set_property"):
        return _set_control(payload)
    if action in ("get_control", "get_property"):
        return _get_control(payload)
    if action in ("ptz", "ptz_action"):
        return _ptz_action(payload)
    if action in ("vendor_passthrough", "extension_unit", "xu"):
        return _vendor_passthrough(payload)
    if action in ("get_config",):
        return {"ok": True, "config": _mask_config(_config())}
    if action in ("update_config", "set_config"):
        merged = _deep_merge(_config(), payload.get("config", payload))
        v = driver_validate(merged)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        _SESSION["config"] = merged
        _append_note("Config updated")
        return {"ok": True, "config": _mask_config(merged), "validate": v}
    if action in ("safe_stop", "stop"):
        return _safe_stop()

    return {"ok": False, "error": "action_not_supported", "action": action}
