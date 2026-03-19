# --== The SarahMemory Project ==--
# Driver: com.softdev0.motherboards
# Purpose: Governed motherboard / firmware / chipset bridge for SMBIOS/DMI/firmware detection and safe diagnostics.
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.motherboards"
DRIVER_VERSION = "2.0.0"

SESSION: Dict[str, Any] = {
    "started_at": None,
    "connected": False,
    "last_result": None,
    "notes": [],
    "history": [],
}
CONFIG: Dict[str, Any] = {}

def _now() -> float:
    return time.time()

def _note(msg: str) -> None:
    SESSION["notes"].append({"ts": _now(), "msg": msg})
    SESSION["notes"] = SESSION["notes"][-100:]

def _push_history(action: str, ok: bool, detail: Any = None) -> None:
    SESSION["history"].append({"ts": _now(), "action": action, "ok": bool(ok), "detail": detail})
    SESSION["history"] = SESSION["history"][-int(CONFIG.get("history_limit", 200)):]

def _result(ok: bool, action: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {"ok": bool(ok), "action": action, "driver_id": DRIVER_ID, **kwargs}
    SESSION["last_result"] = payload
    _push_history(action, ok, kwargs)
    return payload

def _which(name: str) -> Optional[str]:
    return shutil.which(name)

def _run(cmd: List[str], timeout: float = 12.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}

def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return None

def _probe_backends() -> Dict[str, Any]:
    return {
        "platform": platform.system(),
        "backends": {
            "dmidecode": _which("dmidecode"),
            "lspci": _which("lspci"),
            "fwupdmgr": _which("fwupdmgr"),
            "vcgencmd": _which("vcgencmd"),
            "raspi_config": _which("raspi-config"),
            "powershell": _which("powershell") or _which("pwsh"),
            "wmic": _which("wmic"),
            "system_profiler": _which("system_profiler"),
        }
    }

def _linux_inventory() -> Dict[str, Any]:
    sysfs = "/sys/devices/virtual/dmi/id"
    inventory = {
        "board_vendor": _read_file(os.path.join(sysfs, "board_vendor")),
        "board_name": _read_file(os.path.join(sysfs, "board_name")),
        "board_version": _read_file(os.path.join(sysfs, "board_version")),
        "bios_vendor": _read_file(os.path.join(sysfs, "bios_vendor")),
        "bios_version": _read_file(os.path.join(sysfs, "bios_version")),
        "bios_date": _read_file(os.path.join(sysfs, "bios_date")),
        "product_name": _read_file(os.path.join(sysfs, "product_name")),
        "product_version": _read_file(os.path.join(sysfs, "product_version")),
        "sys_vendor": _read_file(os.path.join(sysfs, "sys_vendor")),
    }
    if _which("lspci"):
        res = _run(["lspci"], timeout=10.0)
        inventory["pci_devices"] = [ln.strip() for ln in res.get("stdout", "").splitlines() if ln.strip()]
    if _which("vcgencmd"):
        res = _run(["vcgencmd", "version"], timeout=5.0)
        if res.get("ok"):
            inventory["raspberry_pi_firmware"] = res.get("stdout", "").strip()
    return inventory

def _windows_inventory() -> Dict[str, Any]:
    ps = _which("powershell") or _which("pwsh")
    if not ps:
        return {}
    script = "; ".join([
        "$bb=Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,SerialNumber,Version",
        "$bios=Get-CimInstance Win32_BIOS | Select-Object Manufacturer,SMBIOSBIOSVersion,ReleaseDate,Version",
        "$cs=Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemType",
        '[pscustomobject]@{baseboard=$bb; bios=$bios; system=$cs} | ConvertTo-Json -Depth 5'
    ])
    res = _run([ps, "-NoProfile", "-Command", script], timeout=15.0)
    try:
        return json.loads(res.get("stdout", "") or "{}")
    except Exception:
        return {}

def _mac_inventory() -> Dict[str, Any]:
    if not _which("system_profiler"):
        return {}
    res = _run(["system_profiler", "SPHardwareDataType", "-json"], timeout=15.0)
    try:
        data = json.loads(res.get("stdout", "") or "{}")
        return {"hardware": data.get("SPHardwareDataType", [])}
    except Exception:
        return {}

def _inventory() -> Dict[str, Any]:
    sysname = platform.system()
    if sysname == "Linux":
        return _linux_inventory()
    if sysname == "Windows":
        return _windows_inventory()
    if sysname == "Darwin":
        return _mac_inventory()
    return {}

def _firmware_info() -> Dict[str, Any]:
    info = _inventory()
    secure_boot = None
    if platform.system() == "Linux":
        secure_boot = os.path.exists("/sys/firmware/efi")
    return {"inventory": info, "uefi_present": secure_boot, "platform": platform.platform()}

def driver_init(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CONFIG
    CONFIG = dict(config or {})
    SESSION["started_at"] = _now()
    SESSION["connected"] = True
    _note("motherboard bridge initialized")
    return _result(True, "init", config=CONFIG, backends=_probe_backends())

def driver_status() -> Dict[str, Any]:
    return _result(True, "status", connected=SESSION["connected"], firmware=_firmware_info(), notes=SESSION["notes"][-20:])

def driver_shutdown() -> Dict[str, Any]:
    SESSION["connected"] = False
    _note("motherboard bridge shutdown")
    return _result(True, "shutdown")

def driver_action(action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    act = (action or "").strip().lower()
    if act == "ping":
        return _result(True, act, pong=True)
    if act == "probe_backends":
        return _result(True, act, **_probe_backends())
    if act in {"inventory", "detect", "describe_board"}:
        return _result(True, act, data=_inventory())
    if act in {"firmware_info", "bios_info", "uefi_info"}:
        return _result(True, act, **_firmware_info())
    if act == "get_config":
        return _result(True, act, config=CONFIG)
    if act == "set_config":
        CONFIG.update(payload)
        return _result(True, act, config=CONFIG)
    if act == "dump_dmidecode":
        if not CONFIG.get("allow_privileged_queries", False):
            return _result(False, act, error="privileged_queries_not_allowed")
        if not _which("dmidecode"):
            return _result(False, act, error="dmidecode_not_available")
        res = _run(["dmidecode", "-t", "baseboard"], timeout=20.0)
        return _result(bool(res.get("ok")), act, response=res)
    if act == "fwupd_get_devices":
        if not _which("fwupdmgr"):
            return _result(False, act, error="fwupdmgr_not_available")
        res = _run(["fwupdmgr", "get-devices", "--json"], timeout=20.0)
        return _result(bool(res.get("ok")), act, response=res)
    if act == "safe_stop":
        return _result(True, act, note="motherboard bridge has no active motion path")
    return _result(False, act, error="action_not_supported")
