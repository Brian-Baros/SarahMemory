from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION = {"connected": False, "notes": [], "history": [], "buffer": [], "config": {}, "vault_mounts": []}
DRIVER_ID = "com.softdev0.storage"
DRIVER_KIND = "storage"
DRIVER_VERSION = "3.0.0"
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")
CONFIG_PATH = Path(__file__).with_name("config.json")


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _note(msg):
    SESSION["notes"].append(f"[{_now()}] {msg}")
    SESSION["notes"] = SESSION["notes"][-100:]


def _hist(action, ok, **extra):
    row = {"ts": _now(), "action": action, "ok": bool(ok)}
    row.update(extra)
    SESSION["history"].append(row)
    limit = int(SESSION["config"].get("history_limit", 300) or 300)
    SESSION["history"] = SESSION["history"][-limit:]
    return row


def _run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "cmd": cmd}
    except Exception as e:
        return {"ok": False, "error": str(e), "cmd": cmd}


def _probe_tools(candidates):
    return {c: shutil.which(c) for c in candidates}


def _platform_summary():
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()}


def _lsblk_devices():
    if not shutil.which("lsblk"):
        return []
    r = _run(["lsblk", "-J", "-b", "-o", "NAME,KNAME,TYPE,SIZE,MODEL,VENDOR,TRAN,SERIAL,MOUNTPOINT,FSTYPE,RM,RO,UUID,PATH"])
    if not r.get("ok"):
        return []
    try:
        return json.loads(r["stdout"]).get("blockdevices", [])
    except Exception:
        return []


def _linux_mounts():
    out = []
    try:
        with open('/proc/mounts', 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) >= 3:
                    out.append({'device': parts[0], 'mountpoint': parts[1], 'fstype': parts[2], 'options': parts[3] if len(parts) > 3 else ''})
    except Exception:
        pass
    return out


def _storage_inventory():
    devices = _lsblk_devices()
    mounts = _linux_mounts() if platform.system() == 'Linux' else []
    return {
        'platform': _platform_summary(),
        'devices': devices,
        'mounts': mounts,
        'tools': _probe_tools(['lsblk', 'blkid', 'smartctl', 'nvme', 'mount', 'umount', 'udisksctl', 'diskutil', 'wmic', 'powershell']),
    }


def _smart_summary(device: Optional[str] = None) -> Dict[str, Any]:
    if not shutil.which('smartctl'):
        return {'ok': False, 'error': 'smartctl_not_available'}
    cmd = ['smartctl', '-H', '-A']
    if device:
        cmd.append(str(device))
        res = _run(cmd, timeout=20)
        return {'ok': bool(res.get('ok')), 'device': device, 'result': res}
    results = []
    for dev in _lsblk_devices():
        path = dev.get('path') or ('/dev/' + str(dev.get('kname') or dev.get('name') or ''))
        if dev.get('type') == 'disk':
            results.append({'device': path, 'result': _run(cmd + [path], timeout=20)})
    return {'ok': True, 'devices': results}


def _nvme_summary(device: Optional[str] = None) -> Dict[str, Any]:
    if not shutil.which('nvme'):
        return {'ok': False, 'error': 'nvme_cli_not_available'}
    if device:
        res = _run(['nvme', 'smart-log', str(device)], timeout=20)
        return {'ok': bool(res.get('ok')), 'device': device, 'result': res}
    results = []
    for dev in _lsblk_devices():
        path = dev.get('path') or ('/dev/' + str(dev.get('kname') or dev.get('name') or ''))
        if 'nvme' in str(path):
            results.append({'device': path, 'result': _run(['nvme', 'smart-log', path], timeout=20)})
    return {'ok': True, 'devices': results}


def _vault_module():
    try:
        import SarahMemoryVault as vault  # type: ignore
        return vault
    except Exception:
        return None


def _vault_status() -> Dict[str, Any]:
    vault = _vault_module()
    if not vault:
        return {'available': False, 'error': 'SarahMemoryVault_unavailable'}
    info = {'available': True}
    try:
        if hasattr(vault, 'get_vault_mount_targets'):
            info['targets'] = vault.get_vault_mount_targets()  # type: ignore[attr-defined]
        if hasattr(vault, 'get_vault_mount_state'):
            info['mount_state'] = vault.get_vault_mount_state()  # type: ignore[attr-defined]
    except Exception as exc:
        info['error'] = str(exc)
    return info


def _vault_register_mount(device: str, mountpoint: str, state: str = 'mounted') -> Dict[str, Any]:
    vault = _vault_module()
    if not vault:
        return {'ok': False, 'error': 'SarahMemoryVault_unavailable'}
    try:
        if hasattr(vault, 'register_vault_mount_state'):
            vault.register_vault_mount_state(device=device, mountpoint=mountpoint, state=state)  # type: ignore[attr-defined]
            return {'ok': True, 'device': device, 'mountpoint': mountpoint, 'state': state}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}
    return {'ok': False, 'error': 'vault_mount_hook_missing'}


def _mount_path(device: str, mountpoint: str) -> Dict[str, Any]:
    if not SESSION['config'].get('allow_mount_management', False):
        return {'ok': False, 'error': 'mount_management_not_allowed'}
    if platform.system() == 'Linux':
        Path(mountpoint).mkdir(parents=True, exist_ok=True)
        if shutil.which('mount'):
            res = _run(['mount', device, mountpoint], timeout=15)
            if res.get('ok'):
                _vault_register_mount(device, mountpoint, 'mounted')
            return {'ok': bool(res.get('ok')), 'response': res}
    return {'ok': False, 'error': 'mount_not_supported_on_this_host'}


def _unmount_path(target: str) -> Dict[str, Any]:
    if not SESSION['config'].get('allow_mount_management', False):
        return {'ok': False, 'error': 'mount_management_not_allowed'}
    if platform.system() == 'Linux' and shutil.which('umount'):
        res = _run(['umount', target], timeout=15)
        if res.get('ok'):
            _vault_register_mount(target, target, 'unmounted')
        return {'ok': bool(res.get('ok')), 'response': res}
    return {'ok': False, 'error': 'umount_not_supported_on_this_host'}


def _auto_mount_vault() -> Dict[str, Any]:
    status = _vault_status()
    targets = status.get('targets') or []
    results = []
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        device = str(entry.get('device') or '')
        mountpoint = str(entry.get('mountpoint') or '')
        if device and mountpoint:
            results.append({'device': device, 'mountpoint': mountpoint, 'result': _mount_path(device, mountpoint)})
    return {'ok': True, 'results': results}


def driver_init(context=None, config=None, payload=None):
    SESSION['config'] = _load_json(CONFIG_PATH) or _load_json(DEFAULTS_PATH)
    if isinstance(config, dict):
        SESSION['config'].update(config)
    _save_json(CONFIG_PATH, SESSION['config'])
    SESSION['connected'] = True
    _note('storage bridge initialized')
    _hist('init', True)
    return {'ok': True, 'driver_id': DRIVER_ID, 'driver_version': DRIVER_VERSION, 'config': SESSION['config'], 'inventory': _storage_inventory(), 'vault': _vault_status()}


def driver_status(context=None):
    return {'ok': True, 'connected': SESSION['connected'], 'driver_id': DRIVER_ID, 'driver_version': DRIVER_VERSION, 'inventory': _storage_inventory(), 'vault': _vault_status(), 'notes': SESSION['notes'][-20:], 'history_tail': SESSION['history'][-20:]}


def driver_validate(context=None, config=None, payload=None):
    cfg = dict(SESSION.get('config') or {})
    if isinstance(config, dict):
        cfg.update(config)
    warnings = []
    if cfg.get('allow_mount_management') and platform.system() != 'Linux':
        warnings.append('mount_management_enabled_but_current_host_is_not_linux')
    if cfg.get('vault_auto_mount') and not _vault_module():
        warnings.append('vault_auto_mount_enabled_but_vault_module_unavailable')
    return {'ok': True, 'warnings': warnings, 'tools': _probe_tools(['lsblk', 'smartctl', 'nvme', 'mount', 'umount'])}


def driver_action(action=None, action_id=None, context=None, payload=None, **kwargs):
    action = (action_id or action or '').strip().lower()
    body = dict(payload or {})
    if action == 'ping':
        return {'ok': True, 'pong': True}
    if action in ('probe_backends', 'discover_storage', 'list_volumes', 'inventory'):
        data = _storage_inventory()
        _hist(action, True, count=len(data.get('devices', [])))
        return {'ok': True, **data}
    if action in ('smart_probe', 'smart_summary'):
        res = _smart_summary(body.get('device'))
        _hist(action, bool(res.get('ok')), device=body.get('device'))
        return res
    if action in ('nvme_probe', 'nvme_summary'):
        res = _nvme_summary(body.get('device'))
        _hist(action, bool(res.get('ok')), device=body.get('device'))
        return res
    if action == 'vault_status':
        return {'ok': True, **_vault_status()}
    if action == 'vault_register_mount':
        res = _vault_register_mount(str(body.get('device') or ''), str(body.get('mountpoint') or ''), str(body.get('state') or 'mounted'))
        _hist(action, bool(res.get('ok')), **res)
        return res
    if action == 'vault_auto_mount':
        res = _auto_mount_vault()
        _hist(action, True)
        return res
    if action == 'mount':
        res = _mount_path(str(body.get('device') or ''), str(body.get('mountpoint') or ''))
        _hist(action, bool(res.get('ok')), device=body.get('device'), mountpoint=body.get('mountpoint'))
        return res
    if action == 'unmount':
        res = _unmount_path(str(body.get('target') or body.get('mountpoint') or body.get('device') or ''))
        _hist(action, bool(res.get('ok')), target=body.get('target') or body.get('mountpoint') or body.get('device'))
        return res
    if action == 'get_config':
        return {'ok': True, 'config': SESSION['config']}
    if action == 'set_config':
        if isinstance(body, dict):
            SESSION['config'].update(body)
            _save_json(CONFIG_PATH, SESSION['config'])
        return {'ok': True, 'config': SESSION['config']}
    if action == 'safe_stop':
        _hist(action, True)
        return {'ok': True, 'note': 'storage bridge has no active motion path'}
    _hist(action, False, error='not_implemented')
    return {'ok': False, 'error': 'not_implemented', 'action': action}


def driver_shutdown(context=None):
    SESSION['connected'] = False
    _hist('shutdown', True)
    return {'ok': True, 'driver_id': DRIVER_ID}
