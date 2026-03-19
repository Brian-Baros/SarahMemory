from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.motherboards"
DRIVER_VERSION = "3.0.0"

SESSION: Dict[str, Any] = {
    "started_at": None,
    "connected": False,
    "last_result": None,
    "notes": [],
    "history": [],
    "listener": {"active": False, "thread": None},
}
CONFIG: Dict[str, Any] = {}


def _now() -> float:
    return time.time()


def _note(msg: str) -> None:
    SESSION["notes"].append({"ts": _now(), "msg": msg})
    SESSION["notes"] = SESSION["notes"][-200:]


def _push_history(action: str, ok: bool, detail: Any = None) -> None:
    SESSION["history"].append({"ts": _now(), "action": action, "ok": bool(ok), "detail": detail})
    SESSION["history"] = SESSION["history"][-int(CONFIG.get("history_limit", 300) or 300):]


def _result(ok: bool, action: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {"ok": bool(ok), "action": action, "driver_id": DRIVER_ID, "driver_version": DRIVER_VERSION, **kwargs}
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


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
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
            "efibootmgr": _which("efibootmgr"),
        },
    }


def _linux_smbios_identity() -> Dict[str, Any]:
    sysfs = Path('/sys/devices/virtual/dmi/id')
    keys = {
        'board_vendor': 'board_vendor',
        'board_name': 'board_name',
        'board_version': 'board_version',
        'board_serial': 'board_serial',
        'board_asset_tag': 'board_asset_tag',
        'bios_vendor': 'bios_vendor',
        'bios_version': 'bios_version',
        'bios_date': 'bios_date',
        'product_name': 'product_name',
        'product_version': 'product_version',
        'product_serial': 'product_serial',
        'product_uuid': 'product_uuid',
        'product_sku': 'product_sku',
        'sys_vendor': 'sys_vendor',
        'chassis_vendor': 'chassis_vendor',
        'chassis_type': 'chassis_type',
        'chassis_serial': 'chassis_serial',
    }
    out = {k: _read_file(str(sysfs / v)) for k, v in keys.items()}
    out['uefi_present'] = Path('/sys/firmware/efi').exists()
    return out


def _linux_acpi_snapshot() -> Dict[str, Any]:
    acpi = {
        'button_lid_state': _read_file('/proc/acpi/button/lid/LID/state'),
        'power_supply_state': _read_file('/sys/power/state'),
        'wakealarm': _read_file('/sys/class/rtc/rtc0/wakealarm'),
        'mem_sleep': _read_file('/sys/power/mem_sleep'),
    }
    for button in ('power', 'sleep', 'lid'):
        p = Path('/proc/acpi/button') / button
        if p.exists():
            acpi[f'{button}_paths'] = [str(x) for x in p.rglob('*') if x.is_file()]
    return acpi


def _windows_inventory() -> Dict[str, Any]:
    ps = _which('powershell') or _which('pwsh')
    if not ps:
        return {}
    script = "; ".join([
        "$bb=Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,SerialNumber,Version",
        "$bios=Get-CimInstance Win32_BIOS | Select-Object Manufacturer,SMBIOSBIOSVersion,ReleaseDate,Version,SerialNumber",
        "$cs=Get-CimInstance Win32_ComputerSystemProduct | Select-Object UUID,Vendor,Name,Version,IdentifyingNumber",
        "$sys=Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemType",
        '[pscustomobject]@{baseboard=$bb; bios=$bios; computer_system_product=$cs; system=$sys} | ConvertTo-Json -Depth 6'
    ])
    res = _run([ps, '-NoProfile', '-Command', script], timeout=20.0)
    try:
        return json.loads(res.get('stdout', '') or '{}')
    except Exception:
        return {'raw': res}


def _mac_inventory() -> Dict[str, Any]:
    if not _which('system_profiler'):
        return {}
    res = _run(['system_profiler', 'SPHardwareDataType', '-json'], timeout=20.0)
    try:
        return json.loads(res.get('stdout', '') or '{}')
    except Exception:
        return {'raw': res}


def _raspi_inventory() -> Dict[str, Any]:
    out = {}
    if _which('vcgencmd'):
        ver = _run(['vcgencmd', 'version'], timeout=5.0)
        if ver.get('ok'):
            out['firmware_version'] = ver.get('stdout', '').strip()
        otp = _run(['vcgencmd', 'otp_dump'], timeout=10.0)
        if otp.get('ok'):
            out['otp_dump'] = otp.get('stdout', '').splitlines()[:32]
    for path in ('/proc/device-tree/model', '/proc/device-tree/serial-number', '/sys/firmware/devicetree/base/model'):
        val = _read_file(path)
        if val:
            out[path] = val
    return out


def _inventory() -> Dict[str, Any]:
    sysname = platform.system()
    payload: Dict[str, Any] = {'platform': sysname, 'platform_release': platform.release(), 'hostname': platform.node()}
    if sysname == 'Linux':
        payload['smbios'] = _linux_smbios_identity()
        payload['acpi'] = _linux_acpi_snapshot()
        if _which('lspci'):
            r = _run(['lspci'], timeout=10.0)
            payload['pci_devices'] = [ln.strip() for ln in r.get('stdout', '').splitlines() if ln.strip()]
        if any('raspberry' in str(v).lower() for v in payload.get('smbios', {}).values() if v):
            payload['raspberry_pi'] = _raspi_inventory()
    elif sysname == 'Windows':
        payload['windows'] = _windows_inventory()
    elif sysname == 'Darwin':
        payload['mac'] = _mac_inventory()
    return payload


def _identity_packet() -> Dict[str, Any]:
    inv = _inventory()
    if platform.system() == 'Linux':
        smbios = inv.get('smbios', {})
        return {
            'serial_number': smbios.get('product_serial') or smbios.get('board_serial'),
            'uuid': smbios.get('product_uuid'),
            'bios_version': smbios.get('bios_version'),
            'bios_date': smbios.get('bios_date'),
            'manufacturer': smbios.get('sys_vendor') or smbios.get('board_vendor'),
            'model': smbios.get('product_name') or smbios.get('board_name'),
            'raw': inv,
        }
    win = inv.get('windows', {}) if isinstance(inv, dict) else {}
    bios = win.get('bios', {}) if isinstance(win, dict) else {}
    cs = win.get('computer_system_product', {}) if isinstance(win, dict) else {}
    bb = win.get('baseboard', {}) if isinstance(win, dict) else {}
    return {
        'serial_number': cs.get('IdentifyingNumber') or bios.get('SerialNumber') or bb.get('SerialNumber'),
        'uuid': cs.get('UUID'),
        'bios_version': bios.get('SMBIOSBIOSVersion') or bios.get('Version'),
        'bios_date': bios.get('ReleaseDate'),
        'manufacturer': cs.get('Vendor') or bb.get('Manufacturer'),
        'model': cs.get('Name') or bb.get('Product'),
        'raw': inv,
    }


def _uefi_paths() -> Dict[str, Any]:
    out: Dict[str, Any] = {'linux_efivars': [], 'efi_present': Path('/sys/firmware/efi').exists()}
    efivarfs = Path('/sys/firmware/efi/efivars')
    if efivarfs.exists():
        out['linux_efivars'] = sorted([p.name for p in efivarfs.iterdir()][:200])
    return out


def _read_uefi_variable(name: str) -> Dict[str, Any]:
    efivarfs = Path('/sys/firmware/efi/efivars')
    if not efivarfs.exists():
        return {'ok': False, 'error': 'efivarfs_not_present'}
    matches = [p for p in efivarfs.iterdir() if p.name.startswith(name)]
    if not matches:
        return {'ok': False, 'error': 'variable_not_found'}
    data = _read_bytes(str(matches[0]))
    if data is None:
        return {'ok': False, 'error': 'read_failed'}
    attrs = int.from_bytes(data[:4], 'little', signed=False) if len(data) >= 4 else None
    return {'ok': True, 'name': matches[0].name, 'attributes': attrs, 'data_hex': data[4:].hex()}


def _write_uefi_variable(name: str, data_hex: str) -> Dict[str, Any]:
    if not CONFIG.get('allow_uefi_write', False):
        return {'ok': False, 'error': 'uefi_write_not_allowed'}
    if not CONFIG.get('confirm_uefi_write', False):
        return {'ok': False, 'error': 'uefi_write_confirmation_required'}
    efivarfs = Path('/sys/firmware/efi/efivars')
    if not efivarfs.exists():
        return {'ok': False, 'error': 'efivarfs_not_present'}
    matches = [p for p in efivarfs.iterdir() if p.name.startswith(name)]
    if not matches:
        return {'ok': False, 'error': 'variable_not_found'}
    try:
        raw = bytes.fromhex(str(data_hex or '').strip())
    except Exception:
        return {'ok': False, 'error': 'invalid_hex'}
    try:
        path = matches[0]
        current = _read_bytes(str(path)) or b''
        attrs = current[:4] if len(current) >= 4 else (7).to_bytes(4, 'little')
        with open(path, 'wb') as f:
            f.write(attrs + raw)
        return {'ok': True, 'name': path.name, 'written_bytes': len(raw)}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}


def _boot_order_read() -> Dict[str, Any]:
    if _which('efibootmgr'):
        r = _run(['efibootmgr'], timeout=10.0)
        return {'ok': bool(r.get('ok')), 'response': r}
    return {'ok': False, 'error': 'efibootmgr_not_available'}


def _boot_order_write(order: str) -> Dict[str, Any]:
    if not CONFIG.get('allow_uefi_write', False):
        return {'ok': False, 'error': 'uefi_write_not_allowed'}
    if not CONFIG.get('confirm_uefi_write', False):
        return {'ok': False, 'error': 'uefi_write_confirmation_required'}
    if not _which('efibootmgr'):
        return {'ok': False, 'error': 'efibootmgr_not_available'}
    r = _run(['efibootmgr', '-o', str(order)], timeout=15.0)
    return {'ok': bool(r.get('ok')), 'response': r}


def _acpi_listener_loop() -> None:
    interval = float(CONFIG.get('acpi_poll_interval', 2.0) or 2.0)
    prev = None
    while SESSION.get('listener', {}).get('active'):
        snap = _linux_acpi_snapshot() if platform.system() == 'Linux' else {'platform': platform.system()}
        if snap != prev:
            _note(f"acpi_state_update: {json.dumps(snap, ensure_ascii=False)[:600]}")
            prev = snap
        time.sleep(max(0.5, interval))


def _listener_start() -> Dict[str, Any]:
    if SESSION['listener'].get('active'):
        return {'ok': True, 'already_active': True}
    if platform.system() != 'Linux':
        return {'ok': False, 'error': 'acpi_listener_linux_only'}
    SESSION['listener']['active'] = True
    t = threading.Thread(target=_acpi_listener_loop, name='SM-Mobo-ACPI', daemon=True)
    SESSION['listener']['thread'] = t
    t.start()
    return {'ok': True, 'started': True}


def _listener_stop() -> Dict[str, Any]:
    SESSION['listener']['active'] = False
    return {'ok': True, 'stopped': True}


def driver_init(context: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CONFIG
    CONFIG = dict(config or {})
    SESSION['started_at'] = _now()
    SESSION['connected'] = True
    _note('motherboard bridge initialized')
    return _result(True, 'init', config=CONFIG, backends=_probe_backends(), identity=_identity_packet())


def driver_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _result(True, 'status', connected=SESSION['connected'], identity=_identity_packet(), uefi=_uefi_paths(), listener_active=SESSION['listener'].get('active', False), notes=SESSION['notes'][-20:])


def driver_shutdown(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _listener_stop()
    SESSION['connected'] = False
    _note('motherboard bridge shutdown')
    return _result(True, 'shutdown')


def driver_action(action: Optional[str] = None, action_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
    body = dict(payload or {})
    act = (action_id or action or '').strip().lower()
    if act == 'ping':
        return _result(True, act, pong=True)
    if act == 'probe_backends':
        return _result(True, act, **_probe_backends())
    if act in {'inventory', 'detect', 'describe_board'}:
        return _result(True, act, data=_inventory())
    if act in {'identity', 'identity_packet', 'smbios_identity'}:
        return _result(True, act, data=_identity_packet())
    if act in {'firmware_info', 'bios_info', 'uefi_info'}:
        return _result(True, act, inventory=_inventory(), uefi=_uefi_paths())
    if act == 'uefi_read_variable':
        return _result(True, act, **_read_uefi_variable(str(body.get('name') or 'BootOrder')))
    if act == 'uefi_write_variable':
        res = _write_uefi_variable(str(body.get('name') or ''), str(body.get('data_hex') or ''))
        return _result(bool(res.get('ok')), act, **res)
    if act == 'boot_order_read':
        res = _boot_order_read()
        return _result(bool(res.get('ok')), act, **res)
    if act == 'boot_order_write':
        res = _boot_order_write(str(body.get('order') or ''))
        return _result(bool(res.get('ok')), act, **res)
    if act == 'acpi_listener_start':
        res = _listener_start()
        return _result(bool(res.get('ok')), act, **res)
    if act == 'acpi_listener_stop':
        res = _listener_stop()
        return _result(bool(res.get('ok')), act, **res)
    if act == 'acpi_snapshot':
        return _result(True, act, data=_linux_acpi_snapshot() if platform.system() == 'Linux' else {'platform': platform.system()})
    if act == 'get_config':
        return _result(True, act, config=CONFIG)
    if act == 'set_config':
        CONFIG.update(body)
        return _result(True, act, config=CONFIG)
    if act == 'dump_dmidecode':
        if not CONFIG.get('allow_privileged_queries', False):
            return _result(False, act, error='privileged_queries_not_allowed')
        if not _which('dmidecode'):
            return _result(False, act, error='dmidecode_not_available')
        res = _run(['dmidecode', '-t', 'baseboard'], timeout=20.0)
        return _result(bool(res.get('ok')), act, response=res)
    if act == 'fwupd_get_devices':
        if not _which('fwupdmgr'):
            return _result(False, act, error='fwupdmgr_not_available')
        res = _run(['fwupdmgr', 'get-devices', '--json'], timeout=20.0)
        return _result(bool(res.get('ok')), act, response=res)
    if act == 'safe_stop':
        _listener_stop()
        return _result(True, act, note='motherboard bridge has no active motion path')
    return _result(False, act, error='action_not_supported')
