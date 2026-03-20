"""
--==The SarahMemory Project==--
 File: ./data/boot/drivers/com.softdev0.boot.storagecore
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

from __future__ import annotations
import copy
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVER_ID = 'com.softdev0.boot.storagecore'
DRIVER_NAME = 'SarahMemory Boot Storage Core'
SUPPORTED_ACTIONS = ['probe_backends', 'describe_capabilities', 'list_block_devices', 'list_partitions', 'find_install_targets', 'find_install_media', 'get_smart_summary', 'get_nvme_summary', 'mount_device', 'unmount_device', 'safe_stop', 'get_config', 'set_config']
DEFAULT_CONFIG = {'enabled': True, 'autoload': True, 'allow_mount_ops': False, 'require_mount_confirmation': True, 'allow_smartctl': True, 'allow_nvme_cli': True, 'history_limit': 300}

_STATE: Dict[str, Any] = {
    "initialized": False,
    "config": copy.deepcopy(DEFAULT_CONFIG),
    "started_at": None,
    "history": [],
    "notes": [],
    "last_result": None,
}

def _now() -> float:
    return time.time()

def _push_history(action: str, payload: Optional[Dict[str, Any]], result: Dict[str, Any]) -> None:
    item = {
        "ts": _now(),
        "action": action,
        "payload": payload or {},
        "ok": bool(result.get("ok")),
    }
    _STATE["history"].append(item)
    limit = int(_STATE["config"].get("history_limit", 200) or 200)
    if len(_STATE["history"]) > limit:
        _STATE["history"] = _STATE["history"][-limit:]
    _STATE["last_result"] = result

def _run(cmd: List[str], timeout: int = 8) -> Dict[str, Any]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "cmd": cmd,
        }
    except FileNotFoundError:
        return {"ok": False, "error": "tool_not_found", "cmd": cmd}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "cmd": cmd}

def _which(name: str) -> Optional[str]:
    return shutil.which(name)

def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None

def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dst)
    for k, v in (src or {}).items():
        out[k] = v
    return out

def _linux() -> bool:
    return platform.system().lower() == "linux"

def _windows() -> bool:
    return platform.system().lower() == "windows"

def _mac() -> bool:
    return platform.system().lower() == "darwin"

def _probe_backends() -> Dict[str, Any]:
    tools = {
        "dmidecode": _which("dmidecode"),
        "efibootmgr": _which("efibootmgr"),
        "lspci": _which("lspci"),
        "lsusb": _which("lsusb"),
        "lsblk": _which("lsblk"),
        "blkid": _which("blkid"),
        "smartctl": _which("smartctl"),
        "nvme": _which("nvme"),
        "mount": _which("mount"),
        "umount": _which("umount"),
        "xrandr": _which("xrandr"),
        "ip": _which("ip"),
        "iw": _which("iw"),
        "pactl": _which("pactl"),
        "wpctl": _which("wpctl"),
        "aplay": _which("aplay"),
        "lsmod": _which("lsmod"),
        "fwupdmgr": _which("fwupdmgr"),
    }
    return {
        "ok": True,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "tools": tools,
        "efi_present": Path("/sys/firmware/efi").exists(),
    }

def _describe_capabilities() -> Dict[str, Any]:
    return {
        "ok": True,
        "driver_id": DRIVER_ID,
        "driver_name": DRIVER_NAME,
        "supported_actions": SUPPORTED_ACTIONS,
        "config_keys": sorted(DEFAULT_CONFIG.keys()),
        "system": platform.system(),
        "machine": platform.machine(),
    }

def _get_firmware_info() -> Dict[str, Any]:
    data = {
        "vendor": _read_text("/sys/class/dmi/id/bios_vendor"),
        "version": _read_text("/sys/class/dmi/id/bios_version"),
        "date": _read_text("/sys/class/dmi/id/bios_date"),
        "board_vendor": _read_text("/sys/class/dmi/id/board_vendor"),
        "board_name": _read_text("/sys/class/dmi/id/board_name"),
        "board_version": _read_text("/sys/class/dmi/id/board_version"),
        "product_name": _read_text("/sys/class/dmi/id/product_name"),
        "product_family": _read_text("/sys/class/dmi/id/product_family"),
        "product_uuid": _read_text("/sys/class/dmi/id/product_uuid"),
        "efi_present": Path("/sys/firmware/efi").exists(),
    }
    return {"ok": True, "firmware": data}

def _get_cpu_info() -> Dict[str, Any]:
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "architecture": platform.architecture(),
        "uname": platform.uname()._asdict(),
    }
    if _linux():
        cpuinfo = _read_text("/proc/cpuinfo") or ""
        info["cpuinfo_excerpt"] = cpuinfo[:4000]
    return {"ok": True, "cpu": info}

def _list_pci() -> Dict[str, Any]:
    if _which("lspci"):
        res = _run(["lspci", "-nn"])
        if res.get("ok"):
            return {"ok": True, "devices": [ln for ln in res["stdout"].splitlines() if ln.strip()]}
    return {"ok": True, "devices": []}

def _list_usb() -> Dict[str, Any]:
    if _which("lsusb"):
        res = _run(["lsusb"])
        if res.get("ok"):
            return {"ok": True, "devices": [ln for ln in res["stdout"].splitlines() if ln.strip()]}
    return {"ok": True, "devices": []}

def _list_block_devices() -> Dict[str, Any]:
    if _which("lsblk"):
        res = _run(["lsblk", "-J", "-o", "NAME,KNAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,TRAN,SERIAL"])
        if res.get("ok"):
            try:
                return {"ok": True, "blockdevices": json.loads(res["stdout"]).get("blockdevices", [])}
            except Exception:
                return {"ok": True, "raw": res["stdout"]}
    return {"ok": True, "blockdevices": []}

def _list_filesystems() -> Dict[str, Any]:
    fs = []
    for p in ("/proc/filesystems",):
        txt = _read_text(p)
        if txt:
            fs = [ln.split()[-1] for ln in txt.splitlines() if ln.strip()]
            break
    return {"ok": True, "filesystems": fs}

def _list_displays() -> Dict[str, Any]:
    displays = []
    if _which("xrandr"):
        res = _run(["xrandr", "--query"])
        if res.get("ok"):
            displays = [ln for ln in res["stdout"].splitlines() if " connected" in ln or " disconnected" in ln]
    return {"ok": True, "displays": displays}

def _list_input_devices() -> Dict[str, Any]:
    devices = []
    if _linux():
        base = Path("/sys/class/input")
        if base.exists():
            for p in sorted(base.iterdir()):
                if p.name.startswith("event") or p.name.startswith("mouse"):
                    devices.append(p.name)
    return {"ok": True, "devices": devices}

def _list_network() -> Dict[str, Any]:
    ifaces = []
    if _which("ip"):
        res = _run(["ip", "-j", "addr"])
        if res.get("ok"):
            try:
                ifaces = json.loads(res["stdout"])
            except Exception:
                pass
    return {"ok": True, "interfaces": ifaces}

def _list_audio() -> Dict[str, Any]:
    routes = {"outputs": [], "inputs": []}
    if _which("pactl"):
        sinks = _run(["pactl", "list", "short", "sinks"])
        sources = _run(["pactl", "list", "short", "sources"])
        if sinks.get("ok"):
            routes["outputs"] = [ln for ln in sinks["stdout"].splitlines() if ln.strip()]
        if sources.get("ok"):
            routes["inputs"] = [ln for ln in sources["stdout"].splitlines() if ln.strip()]
    return {"ok": True, "audio": routes}

def _safe_mount(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["config"].get("allow_mount_ops"):
        return {"ok": False, "error": "mount_ops_disabled"}
    if _STATE["config"].get("require_mount_confirmation", False) and not payload.get("confirm"):
        return {"ok": False, "error": "mount_confirmation_required"}
    device = payload.get("device")
    target = payload.get("target")
    fstype = payload.get("fstype")
    if not device or not target:
        return {"ok": False, "error": "missing_device_or_target"}
    cmd = ["mount"]
    if fstype:
        cmd += ["-t", str(fstype)]
    cmd += [str(device), str(target)]
    return _run(cmd, timeout=15)

def _safe_umount(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["config"].get("allow_mount_ops"):
        return {"ok": False, "error": "mount_ops_disabled"}
    if _STATE["config"].get("require_mount_confirmation", False) and not payload.get("confirm"):
        return {"ok": False, "error": "mount_confirmation_required"}
    target = payload.get("target") or payload.get("device")
    if not target:
        return {"ok": False, "error": "missing_target"}
    return _run(["umount", str(target)], timeout=15)

def _smart_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["config"].get("allow_smartctl", True):
        return {"ok": False, "error": "smartctl_disabled"}
    dev = payload.get("device")
    if not dev:
        return {"ok": False, "error": "missing_device"}
    return _run(["smartctl", "-H", "-A", str(dev)], timeout=15)

def _nvme_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["config"].get("allow_nvme_cli", True):
        return {"ok": False, "error": "nvme_cli_disabled"}
    dev = payload.get("device")
    if not dev:
        return {"ok": False, "error": "missing_device"}
    return _run(["nvme", "smart-log", str(dev)], timeout=15)

def _uefi_vars() -> Dict[str, Any]:
    root = Path("/sys/firmware/efi/efivars")
    if not root.exists():
        return {"ok": False, "error": "efivars_unavailable"}
    return {"ok": True, "variables": sorted([p.name for p in root.iterdir()])[:1000]}

def _read_uefi_variable(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("name")
    if not name:
        return {"ok": False, "error": "missing_name"}
    path = Path("/sys/firmware/efi/efivars") / name
    if not path.exists():
        return {"ok": False, "error": "variable_not_found"}
    try:
        data = path.read_bytes()
        return {"ok": True, "name": name, "size": len(data), "hex": data[:128].hex()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

def _write_uefi_variable(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["config"].get("allow_uefi_writes", False):
        return {"ok": False, "error": "uefi_writes_disabled"}
    if _STATE["config"].get("require_write_confirmation", True) and not payload.get("confirm"):
        return {"ok": False, "error": "write_confirmation_required"}
    return {"ok": False, "error": "not_performed_in_generic_boot_driver", "reason": "guarded_planning_only"}

def _simple_passthrough(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_arch_info":
        return {"ok": True, "arch": {"machine": platform.machine(), "system": platform.system(), "platform": platform.platform()}}
    if name == "get_virtualization_info":
        txt = (_read_text("/proc/cpuinfo") or "").lower()
        return {"ok": True, "virtualization": {"vmx": "vmx" in txt, "svm": "svm" in txt}}
    if name == "get_topology":
        return {"ok": True, "topology": {"cpu_count": os.cpu_count()}}
    if name == "find_install_media":
        blk = _list_block_devices().get("blockdevices", [])
        hits = [d for d in blk if str(d.get("tran","")).lower() == "usb" or d.get("type") in ("rom","disk")]
        return {"ok": True, "candidates": hits}
    if name == "find_install_targets":
        blk = _list_block_devices().get("blockdevices", [])
        hits = [d for d in blk if d.get("type") == "disk"]
        return {"ok": True, "targets": hits}
    if name == "list_partitions":
        blk = _list_block_devices().get("blockdevices", [])
        parts = []
        def walk(devs):
            for d in devs:
                if d.get("type") == "part":
                    parts.append(d)
                walk(d.get("children") or [])
        walk(blk)
        return {"ok": True, "partitions": parts}
    if name == "check_mount_support":
        return _list_filesystems()
    if name == "get_framebuffer_info":
        info = {"efi_present": Path("/sys/firmware/efi").exists()}
        fb = Path("/sys/class/graphics")
        info["graphics_entries"] = sorted([p.name for p in fb.iterdir()]) if fb.exists() else []
        return {"ok": True, "framebuffer": info}
    if name == "list_modes":
        return _list_displays()
    if name == "set_mode":
        if not _STATE["config"].get("allow_mode_switch", False):
            return {"ok": False, "error": "mode_switch_disabled"}
        if _STATE["config"].get("require_mode_confirmation", True) and not payload.get("confirm"):
            return {"ok": False, "error": "mode_confirmation_required"}
        return {"ok": False, "error": "generic_mode_switch_not_executed"}
    if name == "get_active_display":
        return _list_displays()
    if name == "find_primary_keyboard":
        devs = _list_input_devices().get("devices", [])
        cand = [d for d in devs if d.startswith("event")]
        return {"ok": True, "device": cand[0] if cand else None}
    if name == "find_primary_pointer":
        devs = _list_input_devices().get("devices", [])
        cand = [d for d in devs if d.startswith("mouse")]
        return {"ok": True, "device": cand[0] if cand else None}
    if name == "find_touch_devices":
        return {"ok": True, "devices": []}
    if name == "get_link_state" or name == "get_ip_summary" or name == "find_boot_network_candidates":
        return _list_network()
    if name == "get_default_audio_routes":
        return _list_audio()
    if name == "test_tone":
        if not _STATE["config"].get("allow_test_tone", False):
            return {"ok": False, "error": "test_tone_disabled"}
        if _STATE["config"].get("require_tone_confirmation", True) and not payload.get("confirm"):
            return {"ok": False, "error": "tone_confirmation_required"}
        return {"ok": False, "error": "generic_test_tone_not_emitted"}
    if name == "validate_target":
        target = payload.get("target")
        return {"ok": True, "target": target, "validated": bool(target)}
    if name == "generate_partition_plan":
        return {"ok": True, "plan": {"scheme": "gpt", "boot": "efi", "root": "primary", "vault": "separate_optional"}}
    if name == "generate_format_plan":
        return {"ok": True, "plan": {"efi": "fat32", "root": "ext4", "vault": "encrypted"}}
    if name == "generate_install_plan":
        return {"ok": True, "plan": {"bootloader": "uefi_preferred", "root_image": "sarahmemory_aios", "recovery": True}}
    if name == "stage_vault_plan":
        return {"ok": True, "vault_plan": {"mount_point": "/mnt/sarahmemory_vault", "encrypted": True}}
    if name == "generate_safe_mode_profile":
        return {"ok": True, "profile": {"graphics": "safe", "network": "off_optional", "drivers": "minimal"}}
    if name == "generate_recovery_profile":
        return {"ok": True, "profile": {"shell": True, "diagnostics": True, "rollback": "planned"}}
    if name == "collect_recovery_snapshot":
        return {"ok": True, "snapshot": {"firmware": _get_firmware_info().get("firmware"), "cpu": _get_cpu_info().get("cpu")}}
    if name == "plan_rollback":
        return {"ok": True, "rollback_plan": {"strategy": "boot_entry_fallback_then_root_snapshot"}}
    return {"ok": False, "error": "unsupported_action"}

def driver_init(config: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for src in (config, context, payload, kwargs):
        if isinstance(src, dict):
            merged = _merge(merged, src)
    _STATE["config"] = merged
    _STATE["initialized"] = True
    _STATE["started_at"] = _now()
    return {"ok": True, "driver_id": DRIVER_ID, "config": copy.deepcopy(_STATE["config"])}

def driver_status(context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "driver_id": DRIVER_ID,
        "initialized": _STATE["initialized"],
        "started_at": _STATE["started_at"],
        "config": copy.deepcopy(_STATE["config"]),
        "history_count": len(_STATE["history"]),
        "last_result": _STATE["last_result"],
    }

def driver_shutdown(context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    _STATE["initialized"] = False
    return {"ok": True, "driver_id": DRIVER_ID, "shutdown": True}

def driver_action(action: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    act = action or (payload or {}).get("action") or kwargs.get("action")
    body = {}
    if isinstance(payload, dict):
        body.update(payload)
    if isinstance(context, dict):
        for k, v in context.items():
            body.setdefault(k, v)
    if isinstance(config, dict):
        _STATE["config"] = _merge(_STATE["config"], config)
    result: Dict[str, Any]
    if act == "probe_backends":
        result = _probe_backends()
    elif act == "describe_capabilities":
        result = _describe_capabilities()
    elif act in ("get_firmware_info", "get_smbios_info"):
        result = _get_firmware_info()
    elif act == "get_acpi_snapshot":
        result = {"ok": True, "acpi": {"efi_present": Path('/sys/firmware/efi').exists(), "power_state_hint": "S0_or_runtime"}}
    elif act == "list_uefi_variables":
        result = _uefi_vars()
    elif act == "read_uefi_variable":
        result = _read_uefi_variable(body)
    elif act == "write_uefi_variable":
        result = _write_uefi_variable(body)
    elif act == "get_boot_entries":
        result = _run(["efibootmgr", "-v"]) if _which("efibootmgr") else {"ok": False, "error": "efibootmgr_not_found"}
    elif act == "set_boot_next":
        if not _STATE["config"].get("allow_uefi_writes", False):
            result = {"ok": False, "error": "uefi_writes_disabled"}
        elif _STATE["config"].get("require_write_confirmation", True) and not body.get("confirm"):
            result = {"ok": False, "error": "write_confirmation_required"}
        else:
            result = {"ok": False, "error": "generic_set_boot_next_not_executed"}
    elif act == "get_power_state":
        result = {"ok": True, "power_state": "runtime"}
    elif act == "get_cpu_info":
        result = _get_cpu_info()
    elif act == "list_devices":
        # package-specific fallback based on driver id prefix
        if "pci" in DRIVER_ID:
            result = _list_pci()
        elif "usbhost" in DRIVER_ID:
            result = _list_usb()
        elif "displaycore" in DRIVER_ID:
            result = _list_displays()
        elif "inputcore" in DRIVER_ID:
            result = _list_input_devices()
        elif "networkcore" in DRIVER_ID:
            result = _list_network()
        elif "audiocore" in DRIVER_ID:
            result = _list_audio()
        else:
            result = {"ok": True, "devices": []}
    elif act == "list_block_devices":
        result = _list_block_devices()
    elif act == "list_supported_filesystems":
        result = _list_filesystems()
    elif act == "mount_device":
        result = _safe_mount(body)
    elif act == "unmount_device":
        result = _safe_umount(body)
    elif act == "get_smart_summary":
        result = _smart_summary(body)
    elif act == "get_nvme_summary":
        result = _nvme_summary(body)
    elif act == "safe_stop":
        result = {"ok": True, "safe_stop": True}
    elif act == "get_config":
        result = {"ok": True, "config": copy.deepcopy(_STATE["config"])}
    elif act == "set_config":
        _STATE["config"] = _merge(_STATE["config"], body)
        result = {"ok": True, "config": copy.deepcopy(_STATE["config"])}
    else:
        result = _simple_passthrough(str(act), body)
    _push_history(str(act), body, result)
    return result
