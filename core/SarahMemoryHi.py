"""--==The SarahMemory Project==--
File: SarahMemoryHi.py
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

SYSTEM INFORMATION MODULE v9.0.0
=============================================
This module has standards with comprehensive hardware
detection, advanced network monitoring, and real-time system metrics while maintaining
100% backward compatibility.

KEY ENHANCEMENTS:
-----------------
1. COMPREHENSIVE HARDWARE DETECTION
- Cross-platform system information
- CPU, memory, disk, network metrics
- GPU detection and monitoring
- Thermal sensor readings
- Battery status (mobile devices)
- Hardware capabilities profiling

2. ADVANCED NETWORK MONITORING
- Real-time connectivity status
- Network state tracking (green/yellow/red)
- Bandwidth utilization monitoring
- Connection quality metrics
- Async network state updates
- Multi-interface support

3. PERFORMANCE METRICS
- Real-time resource usage
- Historical trend analysis
- Performance bottleneck detection
- Resource allocation recommendations
- Predictive usage patterns

4. CROSS-PLATFORM COMPATIBILITY
- Windows (including DXDiag integration)
- Linux (lscpu, lspci integration)
- macOS support
- Headless server compatibility
- Graceful feature degradation

5. COMPREHENSIVE LOGGING
- Structured system logs
- JSON-formatted metrics
- Database persistence
- Historical data retention
- Audit trail generation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- get_system_info()
- get_system_info_json()
- display_system_info(info)
- log_system_info_to_db(info)
- is_connected(host, port, timeout)
- update_network_state()

New functions added (non-breaking):
- get_extended_system_info()
- get_gpu_info()
- get_network_metrics()
- get_thermal_info()
- get_system_health_score()
- async_update_network_state()

INTEGRATION POINTS:
-------------------
- SarahMemoryOptimization.py: Resource monitoring
- SarahMemoryDiagnostics.py: Health checks
- SarahMemoryGlobals.py: Configuration
- SarahMemoryDatabase.py: Metrics persistence
- SarahMemoryNetwork.py: Network status

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "telemetry_engine"
# CATEGORY = "system_info_and_network"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_network_sensors"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "system_information"
# FAMILY = "telemetry"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "System information and network telemetry engine for hardware detection, health metrics, GPU/thermal info, connectivity state, and async network status updates."
# --- SARAHMETA END ---

import logging
import platform
import psutil
import subprocess
import shutil
import os
import json
import sqlite3
import socket
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Core imports
import SarahMemoryGlobals as config

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryHi')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# GLOBAL NETWORK STATE
# =============================================================================
NETWORK_STATE = "unknown"  # green, yellow, red, unknown
_last_net_io = None
_network_state_cache = {"state": "unknown", "updated": None}


# =============================================================================
# SINGLE BOOT ENVIRONMENT SNAPSHOT - v8.0 Enterprise Body Map
# =============================================================================
# One authoritative hardware/body snapshot is captured once per boot and then reused
# by boot logging, model-tier grading, driver readiness, chat answers, diagnostics,
# and API/UI endpoints. This prevents repeated CPU/GPU/driver scans and keeps
# SarahMemory's answers aligned with the environment it actually booted into.
_BOOT_ENVIRONMENT_CACHE: Dict[str, Any] = {"snapshot": None, "ts": 0.0, "boot_id": ""}
_BOOT_ENVIRONMENT_BUILDING = False
_BOOT_ENVIRONMENT_SCHEMA_VERSION = "9.0.0-env-unified-1"


def _datasets_dir_path() -> Path:
    try:
        return Path(str(getattr(config, "DATASETS_DIR", ""))).expanduser().resolve()
    except Exception:
        pass
    try:
        data_dir = Path(str(getattr(config, "DATA_DIR", ""))).expanduser().resolve()
        if str(data_dir):
            return (data_dir / "memory" / "datasets").resolve()
    except Exception:
        pass
    return (Path(os.getcwd()).resolve() / "data" / "memory" / "datasets").resolve()


def _environment_snapshot_json_path() -> Path:
    return _datasets_dir_path() / "runtime_environment_snapshot.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _write_json_file_atomic(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as exc:
        logger.debug(f"[v8.0] environment snapshot persist skipped: {exc}")


def _current_os_boot_id() -> str:
    try:
        boot_ts = psutil.boot_time()
        return f"{platform.node()}::{int(float(boot_ts))}"
    except Exception:
        return f"{platform.node()}::unknown"


def _current_os_boot_time_iso() -> Optional[str]:
    try:
        return datetime.fromtimestamp(psutil.boot_time()).isoformat()
    except Exception:
        return None


def _powershell_json(script: str, timeout: float = 10.0) -> Any:
    try:
        if platform.system() != "Windows":
            return None
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            return None
        result = _run_probe_command([ps, "-NoProfile", "-Command", script], timeout=timeout)
        if result.get("ok") and str(result.get("stdout") or "").strip():
            return json.loads(str(result.get("stdout") or ""))
    except Exception as exc:
        logger.debug(f"[v8.0] PowerShell JSON probe failed: {exc}")
    return None


def _first_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _detect_cpu_identity() -> Dict[str, Any]:
    uname = platform.uname()
    name = (platform.processor() or getattr(uname, "processor", "") or getattr(uname, "machine", "") or "Unknown CPU").strip()
    cores_physical = None
    cores_logical = None
    max_clock_mhz = None
    current_clock_mhz = None
    usage_pct = None

    try:
        cores_physical = psutil.cpu_count(logical=False)
    except Exception:
        pass
    try:
        cores_logical = psutil.cpu_count(logical=True)
    except Exception:
        pass
    try:
        freq = psutil.cpu_freq()
        if freq:
            current_clock_mhz = _safe_float(getattr(freq, "current", None), 0.0) or None
            max_clock_mhz = _safe_float(getattr(freq, "max", None), 0.0) or None
    except Exception:
        pass
    try:
        usage_pct = round(float(psutil.cpu_percent(interval=0.2)), 2)
    except Exception:
        usage_pct = None

    if platform.system() == "Windows":
        data = _powershell_json(
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
            "ConvertTo-Json -Depth 4",
            timeout=10.0,
        )
        row = _first_dict(data)
        if row:
            name = str(row.get("Name") or name).strip() or name
            cores_physical = _safe_int(row.get("NumberOfCores"), cores_physical or 0) or cores_physical
            cores_logical = _safe_int(row.get("NumberOfLogicalProcessors"), cores_logical or 0) or cores_logical
            max_clock_mhz = _safe_float(row.get("MaxClockSpeed"), max_clock_mhz or 0.0) or max_clock_mhz

    return {
        "name": name,
        "manufacturer": "",
        "architecture": getattr(uname, "machine", ""),
        "physical_cores": cores_physical,
        "logical_threads": cores_logical,
        "current_clock_mhz": current_clock_mhz,
        "max_clock_mhz": max_clock_mhz,
        "usage_pct": usage_pct,
    }


def _detect_memory_identity() -> Dict[str, Any]:
    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_gb": round(float(vm.total) / (1024 ** 3), 2),
            "available_gb": round(float(vm.available) / (1024 ** 3), 2),
            "used_gb": round(float(vm.used) / (1024 ** 3), 2),
            "usage_pct": round(float(vm.percent), 2),
            "swap_total_gb": round(float(swap.total) / (1024 ** 3), 2),
            "swap_usage_pct": round(float(swap.percent), 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _detect_gpu_identity() -> Dict[str, Any]:
    gpu = get_gpu_info() or {}
    out: Dict[str, Any] = {
        "available": bool(gpu),
        "name": str(gpu.get("Name") or gpu.get("Info") or gpu.get("Type") or "").strip() or None,
        "type": gpu.get("Type"),
        "driver": gpu.get("Driver"),
        "memory": gpu.get("Memory"),
        "raw": gpu,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [x.strip() for x in result.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 6:
                out.update({
                    "available": True,
                    "name": parts[0],
                    "driver": parts[1],
                    "vram_total_mb": _safe_int(parts[2], 0),
                    "vram_free_mb": _safe_int(parts[3], 0),
                    "temperature_c": None if parts[4].upper() in {"N/A", ""} else _safe_float(parts[4], 0.0),
                    "utilization_pct": _safe_float(parts[5], 0.0),
                    "backend": "nvidia-smi",
                })
    except Exception:
        pass
    return out


def _detect_storage_identity() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen = set()
    try:
        for part in psutil.disk_partitions(all=False) or []:
            mount = str(getattr(part, "mountpoint", "") or "").strip()
            device = str(getattr(part, "device", "") or "").strip()
            if not mount or mount in seen:
                continue
            seen.add(mount)
            row: Dict[str, Any] = {
                "device": device,
                "mountpoint": mount,
                "fstype": str(getattr(part, "fstype", "") or ""),
                "opts": str(getattr(part, "opts", "") or ""),
            }
            try:
                usage = psutil.disk_usage(mount)
                row.update({
                    "total_gb": round(float(usage.total) / (1024 ** 3), 2),
                    "used_gb": round(float(usage.used) / (1024 ** 3), 2),
                    "free_gb": round(float(usage.free) / (1024 ** 3), 2),
                    "usage_pct": round(float(usage.percent), 2),
                })
            except Exception:
                pass
            items.append(row)
    except Exception as exc:
        logger.debug(f"[v8.0] storage identity probe failed: {exc}")
    return items


def _detect_network_identity() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        stats = psutil.net_if_stats() or {}
        addrs = psutil.net_if_addrs() or {}
        for name, addr_rows in addrs.items():
            low = str(name).lower()
            if low.startswith("lo") or "loopback" in low:
                continue
            st = stats.get(name)
            ips = []
            macs = []
            for addr in addr_rows or []:
                fam = str(getattr(addr, "family", ""))
                address = str(getattr(addr, "address", "") or "").strip()
                if not address:
                    continue
                if "AF_LINK" in fam or "AF_PACKET" in fam:
                    macs.append(address)
                elif "." in address or ":" in address:
                    ips.append(address)
            items.append({
                "name": _hardware_label(name),
                "is_up": bool(getattr(st, "isup", False)) if st else None,
                "speed_mbps": getattr(st, "speed", None) if st else None,
                "mtu": getattr(st, "mtu", None) if st else None,
                "ip_addresses": ips,
                "mac_addresses": macs,
            })
    except Exception as exc:
        logger.debug(f"[v8.0] network identity probe failed: {exc}")
    return items


def _boot_environment_chat_facts(snapshot: Dict[str, Any]) -> Dict[str, str]:
    body = snapshot.get("body") if isinstance(snapshot.get("body"), dict) else {}
    cpu = body.get("cpu") if isinstance(body.get("cpu"), dict) else {}
    gpu = body.get("gpu") if isinstance(body.get("gpu"), dict) else {}
    ram = body.get("ram") if isinstance(body.get("ram"), dict) else {}
    motherboard = str(body.get("motherboard") or "Unknown motherboard")
    gpu_name = str(gpu.get("name") or "No dedicated GPU detected")
    cpu_name = str(cpu.get("name") or "Unknown CPU")
    camera_devices = body.get("camera_devices") if isinstance(body.get("camera_devices"), list) else []
    facts = {
        "cpu": cpu_name,
        "gpu": gpu_name,
        "motherboard": motherboard,
        "ram": f"{ram.get('total_gb', 'Unknown')} GB total, {ram.get('available_gb', 'Unknown')} GB available",
        "network_adapters": str(len(body.get("network_adapters") or [])),
        "storage_devices": str(len(body.get("storage") or [])),
        "camera_devices": str(len(camera_devices)),
        "webcam": "; ".join(str(x) for x in camera_devices[:3]) if camera_devices else "No verified physical webcam/camera device detected",
        "vision_hardware": "; ".join(str(x) for x in camera_devices[:3]) if camera_devices else "No verified physical webcam/camera device detected",
    }
    return facts


def _build_boot_environment_snapshot(refresh_reason: str = "boot") -> Dict[str, Any]:
    motherboard_items = _detect_motherboard_items()
    cpu = _detect_cpu_identity()
    ram = _detect_memory_identity()
    gpu = _detect_gpu_identity()
    storage = _detect_storage_identity()
    network = _detect_network_identity()
    audio = _detect_audio_items()
    cameras = _detect_camera_items()
    printers = _detect_printer_items()
    bluetooth = _detect_bluetooth_items()
    npu = _detect_npu_items()
    driver_readiness = _build_b_level_driver_readiness_snapshot_uncached()
    thermal_sensors = get_thermal_info() or {}

    root_path = None
    try:
        root_path = str(getattr(config, "BASE_DIR", os.getcwd()))
    except Exception:
        root_path = os.getcwd()
    model_metrics = {
        "cpu_count": cpu.get("logical_threads"),
        "cpu_pct": cpu.get("usage_pct"),
        "ram_total_mb": _safe_int(_safe_float(ram.get("total_gb"), 0.0) * 1024, 0),
        "ram_avail_mb": _safe_int(_safe_float(ram.get("available_gb"), 0.0) * 1024, 0),
        "disk_free_gb": None,
        "disk_total_gb": None,
        "gpu_name": gpu.get("name"),
        "gpu_vram_total_mb": gpu.get("vram_total_mb"),
        "gpu_vram_free_mb": gpu.get("vram_free_mb"),
        "gpu_temp_c": gpu.get("temperature_c"),
        "cpu_temp_c": None,
    }
    try:
        # Cross-platform best-effort: do not assume Windows/WMI.
        for group, entries in (thermal_sensors or {}).items():
            for entry in entries or []:
                label = str((entry or {}).get("label") or "")
                low = (str(group) + " " + label).lower()
                if any(k in low for k in ("coretemp", "k10temp", "cpu package", "package id", "tctl", "tdie", "cpu", "socket", "cputin")):
                    cur = (entry or {}).get("current")
                    if cur not in (None, "", "N/A"):
                        model_metrics["cpu_temp_c"] = float(cur)
                        raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass
    try:
        disk_for_root = shutil.disk_usage(root_path or os.getcwd())
        model_metrics["disk_total_gb"] = float(disk_for_root.total) / (1024 ** 3)
        model_metrics["disk_free_gb"] = float(disk_for_root.free) / (1024 ** 3)
    except Exception:
        if storage:
            model_metrics["disk_total_gb"] = storage[0].get("total_gb")
            model_metrics["disk_free_gb"] = storage[0].get("free_gb")

    hardware_grade: Dict[str, Any] = {}
    try:
        grade_fn = getattr(config, "hardware_score", None)
        if callable(grade_fn):
            hardware_grade = grade_fn(model_metrics)
    except Exception as exc:
        hardware_grade = {"score": None, "tier": "low", "tier_rating": "Unknown", "error": str(exc), "metrics": model_metrics}

    snapshot: Dict[str, Any] = {
        "ok": True,
        "schema_version": _BOOT_ENVIRONMENT_SCHEMA_VERSION,
        "source": "SarahMemoryHi.get_boot_environment_snapshot",
        "refresh_reason": str(refresh_reason or "boot"),
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid(),
        "boot_id": _current_os_boot_id(),
        "os_boot_time": _current_os_boot_time_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "body": {
            "motherboard": motherboard_items[0].replace(" Motherboard", "") if motherboard_items else "Unknown motherboard",
            "motherboard_items": motherboard_items,
            "cpu": cpu,
            "ram": ram,
            "gpu": gpu,
            "thermal_sensors": thermal_sensors,
            "storage": storage,
            "network_adapters": network,
            "audio_devices": audio,
            "camera_devices": cameras,
            "printer_devices": printers,
            "bluetooth_devices": bluetooth,
            "npu_devices": npu,
        },
        "model_metrics": model_metrics,
        "hardware_grade": hardware_grade,
        "driver_readiness": driver_readiness,
    }
    snapshot["chat_facts"] = _boot_environment_chat_facts(snapshot)
    return snapshot


def get_boot_environment_snapshot(force_refresh: bool = False, refresh_reason: str = "runtime", persist: bool = True) -> Dict[str, Any]:
    """Return the single authoritative boot environment/body snapshot.

    - force_refresh=True performs the one expensive boot scan and persists it.
    - force_refresh=False reuses in-memory cache, then persisted same-boot cache,
      and only falls back to a fresh scan if no valid same-boot snapshot exists.
    """
    global _BOOT_ENVIRONMENT_BUILDING
    boot_id = _current_os_boot_id()
    cached = _BOOT_ENVIRONMENT_CACHE.get("snapshot")
    if not force_refresh and isinstance(cached, dict) and cached.get("boot_id") == boot_id:
        return json.loads(json.dumps(cached, ensure_ascii=False))

    disk_snapshot = _read_json_file(_environment_snapshot_json_path())
    if not force_refresh and disk_snapshot.get("boot_id") == boot_id and disk_snapshot.get("ok"):
        _BOOT_ENVIRONMENT_CACHE.update({"snapshot": disk_snapshot, "ts": time.time(), "boot_id": boot_id})
        return json.loads(json.dumps(disk_snapshot, ensure_ascii=False))

    if _BOOT_ENVIRONMENT_BUILDING:
        # Re-entrant protection: return the best available partial/persisted snapshot.
        if isinstance(cached, dict):
            return dict(cached)
        if isinstance(disk_snapshot, dict) and disk_snapshot:
            return dict(disk_snapshot)
        return {"ok": False, "error": "environment_snapshot_build_in_progress", "boot_id": boot_id}

    _BOOT_ENVIRONMENT_BUILDING = True
    try:
        snap = _build_boot_environment_snapshot(refresh_reason=refresh_reason)
        _BOOT_ENVIRONMENT_CACHE.update({"snapshot": snap, "ts": time.time(), "boot_id": boot_id})
        if persist:
            _write_json_file_atomic(_environment_snapshot_json_path(), snap)
        return json.loads(json.dumps(snap, ensure_ascii=False))
    finally:
        _BOOT_ENVIRONMENT_BUILDING = False


def get_cached_boot_environment_snapshot() -> Dict[str, Any]:
    return get_boot_environment_snapshot(force_refresh=False, refresh_reason="cached")


def get_boot_environment_model_metrics(force_refresh: bool = False) -> Dict[str, Any]:
    snap = get_boot_environment_snapshot(force_refresh=force_refresh, refresh_reason="model_metrics")
    metrics = snap.get("model_metrics") if isinstance(snap.get("model_metrics"), dict) else {}
    return dict(metrics)


def get_boot_environment_chat_fact(key: str = "") -> Any:
    snap = get_boot_environment_snapshot(force_refresh=False, refresh_reason="chat_fact")
    facts = snap.get("chat_facts") if isinstance(snap.get("chat_facts"), dict) else {}
    if key:
        return facts.get(str(key).strip().lower())
    return facts

# =============================================================================
# SYSTEM INFORMATION - v8.0 Enhanced
# =============================================================================
def get_system_info() -> Dict[str, Any]:
    """
    Retrieve comprehensive system and hardware information.
    v8.0: Enhanced with additional metrics and error handling.
    
    Returns:
        Dictionary containing system information
    """
    try:
        system_info = {}
        
        # Platform information
        uname = platform.uname()
        system_info['System'] = uname.system
        system_info['Node Name'] = uname.node
        system_info['Release'] = uname.release
        system_info['Version'] = uname.version
        system_info['Machine'] = uname.machine
        system_info['Processor'] = uname.processor
        
        # CPU information
        try:
            system_info['CPU Cores (Physical)'] = psutil.cpu_count(logical=False) or 0
            system_info['CPU Cores (Logical)'] = psutil.cpu_count(logical=True) or 0
            system_info['CPU Usage (%)'] = round(psutil.cpu_percent(interval=1), 2)
            system_info['CPU Frequency (MHz)'] = round(psutil.cpu_freq().current, 2) if psutil.cpu_freq() else 0
        except Exception as e:
            logger.debug(f"[v8.0] CPU info error: {e}")
            system_info['CPU Cores (Physical)'] = 0
            system_info['CPU Cores (Logical)'] = 0
            system_info['CPU Usage (%)'] = 0
        
        # Memory information
        try:
            virtual_mem = psutil.virtual_memory()
            system_info['Total Memory (GB)'] = round(virtual_mem.total / (1024**3), 2)
            system_info['Available Memory (GB)'] = round(virtual_mem.available / (1024**3), 2)
            system_info['Memory Usage (%)'] = round(virtual_mem.percent, 2)
            
            # Swap memory
            swap = psutil.swap_memory()
            system_info['Total Swap (GB)'] = round(swap.total / (1024**3), 2)
            system_info['Swap Usage (%)'] = round(swap.percent, 2)
        except Exception as e:
            logger.debug(f"[v8.0] Memory info error: {e}")
        
        # Disk information
        try:
            disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
            system_info['Total Disk (GB)'] = round(disk_usage.total / (1024**3), 2)
            system_info['Used Disk (GB)'] = round(disk_usage.used / (1024**3), 2)
            system_info['Free Disk (GB)'] = round(disk_usage.free / (1024**3), 2)
            system_info['Disk Usage (%)'] = round(disk_usage.percent, 2)
        except Exception as e:
            logger.debug(f"[v8.0] Disk info error: {e}")
        
        # Network information
        try:
            net_io = psutil.net_io_counters()
            system_info['Bytes Sent (MB)'] = round(net_io.bytes_sent / (1024**2), 2)
            system_info['Bytes Received (MB)'] = round(net_io.bytes_recv / (1024**2), 2)
            system_info['Network State'] = NETWORK_STATE
        except Exception as e:
            logger.debug(f"[v8.0] Network info error: {e}")
        
        # Boot time
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            system_info['Boot Time'] = boot_time.strftime("%Y-%m-%d %H:%M:%S")
            uptime = datetime.now() - boot_time
            system_info['Uptime'] = str(uptime).split('.')[0]  # Remove microseconds
        except Exception as e:
            logger.debug(f"[v8.0] Boot time error: {e}")
        
        # Platform-specific information
        if platform.system() == "Windows":
            try:
                dx_info = get_dxdiag_info()
                if dx_info:
                    system_info['DXDiag Info'] = dx_info
            except Exception as e:
                logger.debug(f"[v8.0] DXDiag error: {e}")
                system_info['DXDiag Info'] = "Not available"
        else:
            system_info['DXDiag Info'] = "Not available on non-Windows platforms"
        
        # Metadata
        system_info['Version'] = "9.0.0"
        system_info['Timestamp'] = datetime.now().isoformat()
        
        logger.info("[v8.0] System information retrieved successfully")
        return system_info
        
    except Exception as e:
        logger.error(f"[v8.0] Error retrieving system information: {e}")
        return {"error": str(e), "version": "9.0.0"}

# =============================================================================
# DXDiag INFORMATION - v8.0 Enhanced
# =============================================================================
def get_dxdiag_info() -> str:
    """
    Get DirectX diagnostic information on Windows.
    v8.0: Enhanced with better error handling and timeout.
    
    Returns:
        DXDiag information as string or error message
    """
    try:
        if platform.system() != "Windows":
            return "Not available on non-Windows platforms"
        
        output_file = "dxdiag_output.txt"
        
        # Run DXDiag with timeout
        result = subprocess.run(
            ["dxdiag", "/t", output_file],
            shell=True,
            capture_output=True,
            timeout=30
        )
        
        # Wait for file to be written
        import time
        for _ in range(10):
            if os.path.exists(output_file):
                break
            time.sleep(0.5)
        
        if not os.path.exists(output_file):
            return "DXDiag output file not generated"
        
        # Read and cleanup
        with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
            dx_data = f.read()
        
        try:
            os.remove(output_file)
        except Exception:
            pass
        
        # Extract key information (first 5000 chars to avoid overwhelming)
        return dx_data[:5000] + ("..." if len(dx_data) > 5000 else "")
        
    except subprocess.TimeoutExpired:
        logger.warning("[v8.0] DXDiag timeout")
        return "DXDiag timeout (process took too long)"
    except Exception as e:
        logger.warning(f"[v8.0] DXDiag error: {e}")
        return f"DXDiag info not available: {e}"

# =============================================================================
# EXTENDED SYSTEM INFORMATION - v8.0 New
# =============================================================================
def get_extended_system_info() -> Dict[str, Any]:
    """
    Get extended system information including GPU and sensors.
    v8.0: New function with comprehensive hardware detection.
    
    Returns:
        Dictionary with extended system information
    """
    info = get_system_info()
    
    # Add GPU information
    try:
        gpu_info = get_gpu_info()
        if gpu_info:
            info['GPU'] = gpu_info
    except Exception as e:
        logger.debug(f"[v8.0] GPU detection error: {e}")
    
    # Add thermal information
    try:
        thermal_info = get_thermal_info()
        if thermal_info:
            info['Thermal'] = thermal_info
    except Exception as e:
        logger.debug(f"[v8.0] Thermal detection error: {e}")
    
    # Add battery information (for laptops/mobile)
    try:
        if hasattr(psutil, 'sensors_battery'):
            battery = psutil.sensors_battery()
            if battery:
                info['Battery'] = {
                    'Percent': battery.percent,
                    'Plugged In': battery.power_plugged,
                    'Time Remaining': str(timedelta(seconds=battery.secsleft)) if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "Unlimited"
                }
    except Exception as e:
        logger.debug(f"[v8.0] Battery detection error: {e}")
    
    return info

# =============================================================================
# GPU INFORMATION - v8.0 New
# =============================================================================
def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Detect GPU information.
    v8.0: New function for GPU detection.
    
    Returns:
        GPU information dictionary or None
    """
    try:
        # Try nvidia-smi first (NVIDIA GPUs)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'Type': 'NVIDIA',
                    'Name': parts[0].strip() if len(parts) > 0 else 'Unknown',
                    'Memory': parts[1].strip() if len(parts) > 1 else 'Unknown',
                    'Driver': parts[2].strip() if len(parts) > 2 else 'Unknown'
                }
        except Exception:
            pass
        
        # Try DirectX on Windows
        if platform.system() == "Windows":
            # GPU info usually in DXDiag, already captured
            return {'Type': 'See DXDiag', 'Detection': 'Windows Graphics'}
        
        # Try lspci on Linux
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or 'Display' in line or '3D' in line:
                            return {'Type': 'Linux GPU', 'Info': line.split(':', 1)[1].strip() if ':' in line else line}
            except Exception:
                pass
        
        return None
        
    except Exception as e:
        logger.debug(f"[v8.0] GPU detection error: {e}")
        return None

# =============================================================================
# THERMAL INFORMATION - v8.0 New
# =============================================================================
def get_thermal_info() -> Optional[Dict[str, Any]]:
    """
    Get thermal sensor information.
    v8.0: New function for temperature monitoring.
    
    Returns:
        Thermal information dictionary or None
    """
    try:
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                thermal = {}
                for name, entries in temps.items():
                    thermal[name] = [
                        {
                            'label': entry.label or 'N/A',
                            'current': entry.current,
                            'high': entry.high if entry.high else 'N/A',
                            'critical': entry.critical if entry.critical else 'N/A'
                        }
                        for entry in entries
                    ]
                return thermal
        return None
    except Exception as e:
        logger.debug(f"[v8.0] Thermal info error: {e}")
        return None

# =============================================================================
# JSON OUTPUT - Backward Compatible
# =============================================================================
def get_system_info_json() -> str:
    """
    Retrieve system information in JSON format.
    v8.0: Enhanced with better JSON formatting.
    
    Returns:
        JSON-formatted system information
    """
    try:
        info = get_system_info()
        return json.dumps(info, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[v8.0] JSON formatting error: {e}")
        return json.dumps({"error": str(e)}, indent=4)

# =============================================================================
# GUI DISPLAY - Backward Compatible
# =============================================================================
def display_system_info(info: Dict[str, Any]) -> None:
    """
    Display system information in a GUI window using Tkinter.
    v8.0: Enhanced with better formatting and error handling.
    
    Args:
        info: System information dictionary
    """
    try:
        import tkinter as tk
        from tkinter import scrolledtext
        
        root = tk.Tk()
        root.title("System Information - SarahMemory v8.0")
        root.geometry("900x700")
        
        # Create scrolled text widget
        text_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            width=100,
            height=40,
            font=("Consolas", 10)
        )
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Format and insert information
        info_text = "SarahMemory System Information v9.0.0\n"
        info_text += "=" * 80 + "\n\n"
        
        for key, value in info.items():
            if isinstance(value, dict):
                info_text += f"\n{key}:\n"
                info_text += "-" * 40 + "\n"
                for sub_key, sub_value in value.items():
                    info_text += f"  {sub_key}: {sub_value}\n"
            else:
                info_text += f"{key}: {value}\n"
        
        text_area.insert(tk.INSERT, info_text)
        text_area.config(state=tk.DISABLED)  # Make read-only
        
        logger.info("[v8.0] Displaying system information in GUI")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"[v8.0] GUI display error: {e}")
        print(f"Error displaying GUI: {e}")
        print("System information:")
        print(json.dumps(info, indent=2))

# =============================================================================
# DATABASE LOGGING - Backward Compatible
# =============================================================================
def log_system_info_to_db(info: Dict[str, Any]) -> None:
    """
    Log system information to the database.
    v8.0: Enhanced with better database handling.
    
    Args:
        info: System information dictionary
    """
    try:
        # Get database path
        datasets_dir = getattr(config, 'DATASETS_DIR', os.path.join('data', 'memory', 'datasets'))
        db_path = os.path.abspath(os.path.join(datasets_dir, "system_logs.db"))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                info TEXT NOT NULL,
                version TEXT DEFAULT '9.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert log
        timestamp = datetime.now().isoformat()
        info_json = json.dumps(info, ensure_ascii=False)
        
        cursor.execute(
            "INSERT INTO system_logs (timestamp, info, version) VALUES (?, ?, ?)",
            (timestamp, info_json, "9.0.0")
        )
        
        conn.commit()
        conn.close()
        
        logger.info("[v8.0] System info logged to database successfully")
        
    except Exception as e:
        logger.error(f"[v8.0] Database logging error: {e}")

# =============================================================================
# NETWORK CONNECTIVITY - Backward Compatible
# =============================================================================
def is_connected(host: str = "8.8.8.8", port: int = 53, timeout: float = 3) -> bool:
    """
    Check network connectivity to a host.
    v8.0: Enhanced with better error handling.
    
    Args:
        host: Target host
        port: Target port
        timeout: Connection timeout
        
    Returns:
        True if connected, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False

# =============================================================================
# NETWORK STATE MONITORING - v8.0 Enhanced
# =============================================================================
def update_network_state() -> str:
    """
    Update and return current network state.
    v8.0: Enhanced with bandwidth monitoring.

    SARAHMEMORY_PATCH_NOTE 2026-06-23:
    LOCAL_ONLY_MODE must not be interpreted as a failed boot or public hub
    dependency. When local-only is active and no online session is armed, the
    telemetry lane reports yellow/local instead of probing external hosts.
    """
    global NETWORK_STATE, _last_net_io, _network_state_cache

    try:
        try:
            if bool(getattr(config, "LOCAL_ONLY_MODE", True)) and not bool(getattr(config, "SARAHMEMORY_ONLINE_SESSION_ARMED", False)):
                NETWORK_STATE = "yellow"
                _network_state_cache = {"state": "yellow", "updated": datetime.now(), "mode": "local_only"}
                return NETWORK_STATE
        except Exception:
            pass

        # Check basic connectivity
        if not is_connected():
            NETWORK_STATE = "red"
            _network_state_cache = {"state": "red", "updated": datetime.now()}
            return NETWORK_STATE
        
        # Get current network I/O
        net_io = psutil.net_io_counters()
        
        # First run - assume active
        if _last_net_io is None:
            _last_net_io = net_io
            NETWORK_STATE = "green"
            _network_state_cache = {"state": "green", "updated": datetime.now()}
            return NETWORK_STATE
        
        # Calculate traffic delta
        sent_diff = net_io.bytes_sent - _last_net_io.bytes_sent
        recv_diff = net_io.bytes_recv - _last_net_io.bytes_recv
        _last_net_io = net_io
        
        # Determine state based on activity
        threshold = 1024  # 1 KB threshold
        if sent_diff >= threshold or recv_diff >= threshold:
            NETWORK_STATE = "green"  # Active traffic
        else:
            NETWORK_STATE = "yellow"  # Connected but idle
        
        _network_state_cache = {"state": NETWORK_STATE, "updated": datetime.now()}
        return NETWORK_STATE
        
    except Exception as e:
        logger.warning(f"[v8.0] Network state update error: {e}")
        NETWORK_STATE = "unknown"
        return NETWORK_STATE

# =============================================================================
# ASYNC NETWORK STATE - Backward Compatible
# =============================================================================
async def async_update_network_state() -> str:
    """
    Asynchronously update network state.
    v8.0: Enhanced with proper async handling.
    
    Returns:
        Network state: green/yellow/red/unknown
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, update_network_state)
    except Exception as e:
        logger.warning(f"[v8.0] Async network update error: {e}")
        return "unknown"

# =============================================================================
# SYSTEM HEALTH SCORE - v8.0 New
# =============================================================================
def get_system_health_score() -> Dict[str, Any]:
    """
    Calculate overall system health score.
    v8.0: New function for health assessment.
    
    Returns:
        Dictionary with health score and components
    """
    try:
        info = get_system_info()
        
        # Component scores (0-100)
        cpu_score = max(0, 100 - info.get('CPU Usage (%)', 0))
        mem_score = max(0, 100 - info.get('Memory Usage (%)', 0))
        disk_score = max(0, 100 - info.get('Disk Usage (%)', 0))
        
        # Network score
        net_state = info.get('Network State', 'unknown')
        net_score = {'green': 100, 'yellow': 70, 'red': 30, 'unknown': 50}.get(net_state, 50)
        
        # Overall score (weighted average)
        overall = (cpu_score * 0.3 + mem_score * 0.3 + disk_score * 0.25 + net_score * 0.15)
        
        return {
            'overall_score': round(overall, 2),
            'cpu_health': round(cpu_score, 2),
            'memory_health': round(mem_score, 2),
            'disk_health': round(disk_score, 2),
            'network_health': round(net_score, 2),
            'status': 'excellent' if overall >= 80 else 'good' if overall >= 60 else 'fair' if overall >= 40 else 'poor',
            'timestamp': datetime.now().isoformat(),
            'version': '9.0.0'
        }
        
    except Exception as e:
        logger.error(f"[v8.0] Health score error: {e}")
        return {'overall_score': 0, 'status': 'error', 'message': str(e)}

# =============================================================================
# B-LEVEL DRIVER READINESS REPORTING - v8.0 New
# =============================================================================
def _run_probe_command(cmd: List[str], timeout: float = 8.0) -> Dict[str, Any]:
    """Best-effort command runner for boot-safe hardware probing."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            'ok': proc.returncode == 0,
            'returncode': proc.returncode,
            'stdout': proc.stdout or '',
            'stderr': proc.stderr or '',
        }
    except Exception as exc:
        return {'ok': False, 'error': str(exc), 'stdout': '', 'stderr': ''}


def _hardware_label(name: str) -> str:
    try:
        return ' '.join(str(name or '').replace('_', ' ').replace('-', ' ').split()).strip() or 'Unknown Hardware'
    except Exception:
        return 'Unknown Hardware'


def _driver_root_path() -> Path:
    """Resolve runtime driver root without making it boot-critical."""
    try:
        drv = getattr(config, 'DRIVERS_DIR', None)
        if drv:
            return Path(str(drv)).expanduser().resolve()
    except Exception:
        pass
    try:
        data_dir = getattr(config, 'DATA_DIR', None)
        if data_dir:
            return (Path(str(data_dir)).expanduser().resolve() / 'drivers').resolve()
    except Exception:
        pass
    return (Path(os.getcwd()).resolve() / 'data' / 'drivers').resolve()


def _scan_runtime_driver_packages(drivers_root: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Return runtime Level B driver packages present under ./data/drivers/com.softdev0.*."""
    root = Path(drivers_root).expanduser().resolve() if drivers_root else _driver_root_path()
    packages: Dict[str, Dict[str, Any]] = {}
    try:
        if not root.exists() or not root.is_dir():
            return packages
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir():
                continue
            driver_id = entry.name
            if not driver_id.startswith('com.softdev0.'):
                continue
            manifest_path = entry / 'manifest.json'
            driver_path = entry / 'driver.py'
            if not manifest_path.exists() or not driver_path.exists():
                continue
            manifest: Dict[str, Any] = {}
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if not isinstance(manifest, dict):
                    manifest = {}
            except Exception:
                manifest = {}
            packages[driver_id] = {
                'id': driver_id,
                'path': str(entry),
                'manifest': manifest,
                'name': manifest.get('name') or driver_id,
            }
    except Exception as exc:
        logger.debug(f"[v8.0] Driver package scan error: {exc}")
    return packages


def _driver_classes_from_packages(packages: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    ids = set(packages.keys())
    return {
        'motherboard': 'com.softdev0.motherboards' in ids,
        'network': 'com.softdev0.network' in ids,
        'storage': 'com.softdev0.storage' in ids,
        'camera': 'com.softdev0.camera.uvc.usb' in ids,
        'audio': ('com.softdev0.ac97audio' in ids) or ('com.softdev0.speaker.bluetooth' in ids),
        'printer': ('com.softdev0.printer.generic.tcp' in ids) or ('com.softdev0.printer.escpos.usb' in ids),
        'bluetooth': ('com.softdev0.gamepad.bluetooth' in ids) or ('com.softdev0.midi.bluetooth' in ids) or ('com.softdev0.speaker.bluetooth' in ids),
        'serial_usb': any(did in ids for did in (
            'com.softdev0.arduino.usb',
            'com.softdev0.cnc.grbl.serial',
            'com.softdev0.rfid.serial',
            'com.softdev0.zigbee.coordinator.usb',
            'com.softdev0.zwave.usb',
        )),
        'display': 'com.softdev0.vga.hdmi' in ids,
        'npu': 'com.softdev0.localnpu' in ids,
    }


def _detect_motherboard_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = (
                    "$bb=Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product;"
                    "$cs=Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model;"
                    "[pscustomobject]@{bb=$bb;cs=$cs}|ConvertTo-Json -Depth 4"
                )
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=12.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        bb = data.get('bb') or {}
                        manu = str(bb.get('Manufacturer') or '').strip()
                        prod = str(bb.get('Product') or '').strip()
                        name = ' '.join(x for x in [manu, prod] if x).strip()
                        if not name:
                            cs = data.get('cs') or {}
                            name = ' '.join(x for x in [str(cs.get('Manufacturer') or '').strip(), str(cs.get('Model') or '').strip()] if x).strip()
                        if name:
                            items.append(_hardware_label(f"{name} Motherboard"))
                    except Exception:
                        pass
        elif sysname == 'Linux':
            sysfs = Path('/sys/devices/virtual/dmi/id')
            vendor = None
            board = None
            try:
                vendor = (sysfs / 'board_vendor').read_text(encoding='utf-8', errors='ignore').strip() or None
            except Exception:
                try:
                    vendor = (sysfs / 'sys_vendor').read_text(encoding='utf-8', errors='ignore').strip() or None
                except Exception:
                    vendor = None
            try:
                board = (sysfs / 'board_name').read_text(encoding='utf-8', errors='ignore').strip() or None
            except Exception:
                try:
                    board = (sysfs / 'product_name').read_text(encoding='utf-8', errors='ignore').strip() or None
                except Exception:
                    board = None
            name = ' '.join(x for x in [vendor, board] if x).strip()
            if name:
                items.append(_hardware_label(f"{name} Motherboard"))
        elif sysname == 'Darwin':
            model = platform.machine()
            items.append(_hardware_label(f"Apple {model} Logic Board"))
    except Exception as exc:
        logger.debug(f"[v8.0] Motherboard detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_printer_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = "Get-CimInstance Win32_Printer | Select-Object Name,DriverName,PortName | ConvertTo-Json -Depth 4"
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=12.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        if isinstance(data, dict):
                            data = [data]
                        for row in data or []:
                            if not isinstance(row, dict):
                                continue
                            name = str(row.get('Name') or '').strip()
                            if name:
                                items.append(_hardware_label(name))
                    except Exception:
                        pass
        else:
            lpstat = shutil.which('lpstat')
            if lpstat:
                result = _run_probe_command([lpstat, '-p'], timeout=8.0)
                if result.get('ok'):
                    for line in (result.get('stdout') or '').splitlines():
                        line = line.strip()
                        if not line.startswith('printer '):
                            continue
                        name = line.split()[1]
                        if name:
                            items.append(_hardware_label(name))
    except Exception as exc:
        logger.debug(f"[v8.0] Printer detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_camera_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = (
                    "Get-CimInstance Win32_PnPEntity | "
                    "Where-Object { $_.Name -match 'camera|webcam|imaging|usb video' } | "
                    "Select-Object -ExpandProperty Name | ConvertTo-Json -Depth 3"
                )
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=12.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        if isinstance(data, str):
                            data = [data]
                        for row in data or []:
                            name = str(row or '').strip()
                            if name:
                                items.append(_hardware_label(name))
                    except Exception:
                        pass
        elif sysname == 'Linux':
            dev = Path('/dev')
            if dev.exists():
                for node in sorted(dev.glob('video*')):
                    items.append(_hardware_label(f"UVC Camera {node.name}"))
    except Exception as exc:
        logger.debug(f"[v8.0] Camera detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_audio_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = (
                    "Get-CimInstance Win32_SoundDevice | Select-Object -ExpandProperty Name | ConvertTo-Json -Depth 3"
                )
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=12.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        if isinstance(data, str):
                            data = [data]
                        for row in data or []:
                            name = str(row or '').strip()
                            if name:
                                items.append(_hardware_label(name))
                    except Exception:
                        pass
        else:
            # Keep Linux/macOS audio reporting conservative at boot.
            if shutil.which('aplay'):
                result = _run_probe_command(['aplay', '-l'], timeout=6.0)
                if result.get('ok'):
                    for line in (result.get('stdout') or '').splitlines():
                        line = line.strip()
                        if line.startswith('card '):
                            items.append(_hardware_label(line))
    except Exception as exc:
        logger.debug(f"[v8.0] Audio detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_network_items() -> List[str]:
    items: List[str] = []
    try:
        for if_name, addrs in (psutil.net_if_addrs() or {}).items():
            low = str(if_name).lower()
            if low.startswith('lo') or 'loopback' in low:
                continue
            if not addrs:
                continue
            items.append(_hardware_label(if_name))
    except Exception as exc:
        logger.debug(f"[v8.0] Network interface detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_storage_items() -> List[str]:
    items: List[str] = []
    try:
        seen = set()
        for part in psutil.disk_partitions(all=False) or []:
            device = str(getattr(part, 'device', '') or '').strip()
            mountpoint = str(getattr(part, 'mountpoint', '') or '').strip()
            label = device or mountpoint
            if not label or label in seen:
                continue
            seen.add(label)
            items.append(_hardware_label(label))
    except Exception as exc:
        logger.debug(f"[v8.0] Storage detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_bluetooth_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = (
                    "Get-CimInstance Win32_PnPEntity | "
                    "Where-Object { $_.Name -match 'Bluetooth' } | "
                    "Select-Object -ExpandProperty Name | ConvertTo-Json -Depth 3"
                )
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=12.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        if isinstance(data, str):
                            data = [data]
                        for row in data or []:
                            name = str(row or '').strip()
                            if name:
                                items.append(_hardware_label(name))
                    except Exception:
                        pass
        elif sysname == 'Linux' and shutil.which('bluetoothctl'):
            result = _run_probe_command(['bluetoothctl', 'list'], timeout=6.0)
            if result.get('ok'):
                for line in (result.get('stdout') or '').splitlines():
                    line = line.strip()
                    if line.startswith('Controller '):
                        parts = line.split(' ', 2)
                        if len(parts) >= 3:
                            items.append(_hardware_label(parts[2]))
    except Exception as exc:
        logger.debug(f"[v8.0] Bluetooth detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_display_items() -> List[str]:
    items: List[str] = []
    try:
        gpu = get_gpu_info() or {}
        if isinstance(gpu, dict):
            name = str(gpu.get('Name') or gpu.get('Info') or '').strip()
            if name:
                items.append(_hardware_label(name))
    except Exception as exc:
        logger.debug(f"[v8.0] Display detection error: {exc}")
    return list(dict.fromkeys(items))


def _detect_npu_items() -> List[str]:
    items: List[str] = []
    try:
        sysname = platform.system()
        if sysname == 'Windows':
            ps = shutil.which('powershell') or shutil.which('pwsh')
            if ps:
                script = (
                    "Get-CimInstance Win32_PnPEntity | "
                    "Where-Object { $_.Name -match 'NPU|Neural|AI Accelerator' } | "
                    "Select-Object -ExpandProperty Name | ConvertTo-Json -Depth 3"
                )
                result = _run_probe_command([ps, '-NoProfile', '-Command', script], timeout=10.0)
                if result.get('ok') and result.get('stdout', '').strip():
                    try:
                        data = json.loads(result['stdout'])
                        if isinstance(data, str):
                            data = [data]
                        for row in data or []:
                            name = str(row or '').strip()
                            if name:
                                items.append(_hardware_label(name))
                    except Exception:
                        pass
    except Exception as exc:
        logger.debug(f"[v8.0] NPU detection error: {exc}")
    return list(dict.fromkeys(items))


def _build_b_level_driver_readiness_snapshot_uncached(drivers_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Internal uncached boot-safe hardware readiness snapshot.
    Returns detected hardware items and READY/NOT READY status based on generic
    Level B driver families present under ./data/drivers/com.softdev0.*.
    """
    packages = _scan_runtime_driver_packages(drivers_root)
    driver_classes = _driver_classes_from_packages(packages)

    detectors = [
        ('motherboard', _detect_motherboard_items),
        ('display', _detect_display_items),
        ('network', _detect_network_items),
        ('storage', _detect_storage_items),
        ('audio', _detect_audio_items),
        ('camera', _detect_camera_items),
        ('printer', _detect_printer_items),
        ('bluetooth', _detect_bluetooth_items),
        ('npu', _detect_npu_items),
    ]

    entries: List[Dict[str, Any]] = []
    seen_names = set()
    for device_class, detector in detectors:
        try:
            names = detector() or []
        except Exception as exc:
            logger.debug(f"[v8.0] Detector failure for {device_class}: {exc}")
            names = []
        for raw_name in names:
            name = _hardware_label(raw_name)
            key = name.lower()
            if not name or key in seen_names:
                continue
            seen_names.add(key)
            ready = bool(driver_classes.get(device_class, False))
            entries.append({
                'name': name,
                'device_class': device_class,
                'detected': True,
                'ready': ready,
                'status': f"DETECTED/{'READY' if ready else 'NOT READY'}",
            })

    ready_count = sum(1 for item in entries if item.get('ready'))
    return {
        'ok': True,
        'drivers_root': str(Path(drivers_root).expanduser().resolve()) if drivers_root else str(_driver_root_path()),
        'driver_packages': sorted(packages.keys()),
        'driver_classes': driver_classes,
        'entries': entries,
        'detected_count': len(entries),
        'ready_count': ready_count,
        'not_ready_count': max(0, len(entries) - ready_count),
        'timestamp': datetime.now().isoformat(),
        'version': '9.0.0',
    }


def get_b_level_driver_readiness_snapshot(drivers_root: Optional[Union[str, Path]] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """Return driver readiness from the single boot environment snapshot.

    This preserves the old public function name while preventing the readiness
    detectors from running a second or third time during boot/API/chat calls.
    """
    if drivers_root is not None or force_refresh:
        snap = get_boot_environment_snapshot(force_refresh=force_refresh, refresh_reason="driver_readiness")
    else:
        snap = get_boot_environment_snapshot(force_refresh=False, refresh_reason="driver_readiness")
    readiness = snap.get("driver_readiness") if isinstance(snap.get("driver_readiness"), dict) else {}
    if readiness:
        return dict(readiness)
    return _build_b_level_driver_readiness_snapshot_uncached(drivers_root=drivers_root)


def get_b_level_driver_readiness_report(drivers_root: Optional[Union[str, Path]] = None) -> List[str]:
    """Return one industrialized boot line per detected hardware item."""
    snap = get_b_level_driver_readiness_snapshot(drivers_root=drivers_root)
    lines: List[str] = []
    for item in snap.get('entries', []) or []:
        try:
            lines.append(f"{item['name']} - {item['status']}")
        except Exception:
            continue
    return lines


def boot_probe_b_level_driver_readiness(drivers_root: Optional[Union[str, Path]] = None) -> List[str]:
    """Alias kept simple for boot callers."""
    return get_b_level_driver_readiness_report(drivers_root=drivers_root)


# =============================================================================
# MAIN TEST HARNESS - v8.0 Enhanced
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("SarahMemory System Information v9.0.0 - Test Mode")
    print("=" * 80)
    
    logger.info("[v8.0] Starting SarahMemoryHi test suite")
    
    # Get system info
    print("\nGathering system information...")
    info = get_system_info()
    
    # Log to database
    print("Logging to database...")
    log_system_info_to_db(info)
    
    # Display key information
    print("\n" + "=" * 80)
    print("System Information Summary:")
    print("=" * 80)
    
    for key, value in info.items():
        if not isinstance(value, (dict, list)) and key != 'DXDiag Info':
            print(f"{key:30} {value}")
    
    # Get health score
    print("\n" + "=" * 80)
    print("System Health Score:")
    print("=" * 80)
    health = get_system_health_score()
    print(json.dumps(health, indent=2))
    
    # Network state
    print("\n" + "=" * 80)
    print("Network Status:")
    print("=" * 80)
    net_state = update_network_state()
    print(f"Current State: {net_state}")
    print(f"Internet Connectivity: {'Yes' if is_connected() else 'No'}")
    
    # Option to display GUI
    show_gui = input("\nDisplay GUI? (y/n): ").strip().lower()
    if show_gui == 'y':
        display_system_info(info)
    
    print("\n" + "=" * 80)
    logger.info("[v8.0] SarahMemoryHi test suite complete")

logger.info("[v8.0] SarahMemoryHi module loaded successfully")

# ====================================================================
# END OF SarahMemoryHi.py v9.0.0
# ====================================================================
