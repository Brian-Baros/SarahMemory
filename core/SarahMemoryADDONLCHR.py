"""--==The SarahMemory Project==--
File: SarahMemoryADDONLCHR.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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

ADDON LAUNCHER ENGINE
======================================================

--------
This module serves as the launching platform for MODS and Additional Modules that an END-UUSER may want to add
to the SarahMemory Platform, The Ai should be able to Learn and adapt to other Modules and
to user preferences, patterns, timing, and operational habits. Making SarahMemory the dynamic functional foundation.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A-CANDIDATE"
# ROLE = "addon_manager"
# CATEGORY = "addon_operations"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "classic_ui"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "addon_launcher"
# FAMILY = "addons"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "enterprise_candidate"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Governed classic-UI add-on launcher with TrustRegistry binding, bounded integrity checks, fail-closed execution review, and auditable lifecycle control."
# --- SARAHMETA END ---

import hashlib
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import weakref
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import SarahMemoryGlobals as config
from SarahMemoryGlobals import ADDONS_DIR, BASE_DIR, log_gui_event

logger = logging.getLogger("SarahMemoryADDONLCHR")

# Limits are deliberately modest on the MASTER PC. Review any increase before use.
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_INTEGRITY_FILES = 256
_MAX_INTEGRITY_BYTES = 64 * 1024 * 1024
_MAX_INTEGRITY_DEPTH = 6
_MAX_BUNDLE_CHILDREN = 16
_MAX_RUNNING = 8
_MAX_LOG_BYTES = 1024 * 1024
_MAX_ERROR_OUTPUT = 1024
_SHUTDOWN_GRACE_SECONDS = 2.0
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_INSTANCES = weakref.WeakSet()
# One runtime registry per imported CORE organ. Classic GUI may recreate a launcher.
_RUNTIME_STATE = {"processes": [], "manifest_records": {}, "process_meta": {},
                  "log_threads": {}, "lock": threading.RLock()}

# No untrusted value is interpolated into Python source. Values are arguments.
_SUBPROCESS_BOOTSTRAP = """\
import importlib
import json
import os
import sys
module_name, function_name, addon_root, raw_context = sys.argv[1:5]
sys.path.insert(0, addon_root)
module = importlib.import_module(module_name)
actual = os.path.realpath(getattr(module, "__file__", ""))
root = os.path.realpath(addon_root)
if not actual or os.path.commonpath([root, actual]) != root:
    raise RuntimeError("Add-on module resolved outside approved directory")
entry = getattr(module, function_name, None)
if not callable(entry):
    raise RuntimeError("Manifest entrypoint is not callable")
entry(json.loads(raw_context))
"""


def _safe_audit(event: str, detail: str) -> bool:
    """Attempt existing project audit sink; a failed launch audit denies execution."""
    try:
        log_gui_event(event, detail[:512])
        return True
    except Exception as exc:
        logger.error("Add-on audit unavailable (%s): %s", event, type(exc).__name__)
        return False


def _contained(root: str, candidate: str, *, allow_root: bool = True) -> str:
    """Canonicalize and reject traversal and symlinks at every path component."""
    base = os.path.abspath(root)
    raw = os.path.abspath(candidate)
    try:
        if os.path.commonpath((base, raw)) != base:
            raise ValueError("Path escapes the add-on boundary")
    except ValueError:
        raise ValueError("Invalid add-on path") from None
    relative = os.path.relpath(raw, base)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise ValueError("Add-on traversal blocked")
    if not allow_root and relative == '.':
        raise ValueError("Expected an add-on child path")
    current = base
    if os.path.islink(current):
        raise ValueError("Symlinked add-on storage not permitted")
    if relative != '.':
        for part in relative.split(os.sep):
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise ValueError("Symlinked add-on path not permitted")
    real_base = os.path.realpath(base)
    real = os.path.realpath(raw)
    if os.path.commonpath((real_base, real)) != real_base:
        raise ValueError("Add-on path resolves outside approved directory")
    return raw


def _addon_root(name: str) -> str:
    if not isinstance(name, str) or not name or name in {'.', '..'} or '/' in name or '\\' in name or '\x00' in name:
        raise ValueError("Invalid add-on folder name")
    path = _contained(ADDONS_DIR, os.path.join(ADDONS_DIR, name), allow_root=False)
    if not os.path.isdir(path):
        raise FileNotFoundError("Add-on directory not found")
    return path


def _manifest_data(root: str) -> Dict[str, Any]:
    path = _contained(ADDONS_DIR, os.path.join(root, 'manifest.json'))
    if not os.path.isfile(path) or os.path.getsize(path) > _MAX_MANIFEST_BYTES:
        raise ValueError("Manifest missing or exceeds size budget")
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    normalized = dict(data)
    if 'addon_id' not in normalized and 'id' in normalized:
        normalized['addon_id'] = normalized['id']
    if 'entrypoint' not in normalized and 'entry' in normalized:
        normalized['entrypoint'] = normalized['entry']
    addon_id = normalized.get('addon_id') or os.path.basename(root)
    if not isinstance(addon_id, str) or not _ID_RE.fullmatch(addon_id):
        raise ValueError("Invalid add-on identifier")
    normalized['addon_id'] = addon_id
    if 'execution' in normalized and not isinstance(normalized['execution'], dict):
        raise ValueError("Invalid execution policy")
    if 'security' in normalized and not isinstance(normalized['security'], dict):
        raise ValueError("Invalid security metadata")
    permissions = normalized.get('permissions', normalized.get('permission', normalized.get('capabilities', [])))
    if isinstance(permissions, str):
        permissions = [permissions]
    if not isinstance(permissions, list) or len(permissions) > 64 or any(not isinstance(p, str) or not p or len(p)>128 for p in permissions):
        raise ValueError("Invalid add-on permissions")
    normalized['permissions'] = sorted(set(permissions))
    if str(normalized.get('type') or '').lower() == 'addon_bundle':
        includes = normalized.get('includes')
        if not isinstance(includes, list) or not 0 < len(includes) <= _MAX_BUNDLE_CHILDREN:
            raise ValueError("Invalid bundle includes")
        for entry in includes:
            if not isinstance(entry, dict) or not isinstance(entry.get('path'), str) or not entry['path']:
                raise ValueError("Invalid bundle child")
    else:
        entry = normalized.get('entrypoint')
        if not isinstance(entry, dict):
            raise ValueError("Entrypoint requires module and callable")
        module = entry.get('module')
        callable_name = entry.get('callable')
        if not isinstance(module, str) or not _MODULE_RE.fullmatch(module):
            raise ValueError("Entrypoint module contains invalid characters")
        if not isinstance(callable_name, str) or not _NAME_RE.fullmatch(callable_name):
            raise ValueError("Entrypoint callable contains invalid characters")
        module_path = module.replace('.', os.sep)
        candidates = (os.path.join(root, module_path + '.py'), os.path.join(root, module_path, '__init__.py'))
        if not any(os.path.isfile(_contained(root, candidate)) for candidate in candidates):
            raise ValueError("Entrypoint must resolve to Python source inside add-on directory")
    return normalized


def _source_fingerprint(root: str) -> str:
    """Hash a bounded directory tree, excluding only launcher logs and Python caches."""
    root = _contained(ADDONS_DIR, root)
    if not os.path.isdir(root):
        raise ValueError("Cannot fingerprint missing add-on")
    count, total = 0, 0
    digest = hashlib.sha256()
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        if depth > _MAX_INTEGRITY_DEPTH:
            raise ValueError("Add-on tree exceeds integrity depth limit")
        with os.scandir(directory) as entries:
            children = sorted(list(entries), key=lambda e: e.name)
        for item in children:
            if item.is_symlink():
                raise ValueError("Symlinks are not permitted in executable add-on trees")
            if item.is_dir(follow_symlinks=False):
                if item.name == '__pycache__':
                    continue
                if depth >= _MAX_INTEGRITY_DEPTH:
                    raise ValueError("Add-on tree exceeds integrity depth limit")
                pending.append((item.path, depth + 1))
                continue
            if not item.is_file(follow_symlinks=False):
                raise ValueError("Non-regular add-on content is unsupported")
            if item.name.endswith(('_launch.log', '_launch.log.1')) or item.name in {'manifest_launch.log', 'manifest_launch.log.1'}:
                continue
            if item.name.endswith(('.pyc', '.pyo')):
                continue
            count += 1
            if count > _MAX_INTEGRITY_FILES:
                raise ValueError("Add-on exceeds integrity file-count budget")
            size = item.stat(follow_symlinks=False).st_size
            total += size
            if total > _MAX_INTEGRITY_BYTES:
                raise ValueError("Add-on exceeds integrity byte budget")
            relative = os.path.relpath(item.path, root).replace(os.sep, '/')
            digest.update(relative.encode('utf-8') + b'\0' + str(size).encode('ascii') + b'\0')
            with open(item.path, 'rb') as stream:
                remaining = size
                while remaining:
                    buf = stream.read(min(65536, remaining))
                    if not buf:
                        raise ValueError("Source changed while hashing")
                    digest.update(buf)
                    remaining -= len(buf)
            if item.stat(follow_symlinks=False).st_size != size:
                raise ValueError("Source changed while hashing")
    if count == 0:
        raise ValueError("Empty add-on directory")
    return digest.hexdigest()


def _trust_review(addon_id: str, fingerprint: str, root: str, permissions: List[str]) -> Dict[str, Any]:
    """Read the existing TrustRegistry. Never auto-register or self-approve add-ons."""
    try:
        from SarahMemoryTrustRegistry import lookup_subject
        record = lookup_subject(addon_id)
    except Exception as exc:
        raise PermissionError("TrustRegistry unavailable: launch denied") from exc
    if not isinstance(record, dict):
        raise PermissionError("Add-on requires independent TrustRegistry approval")
    if str(record.get('subject_kind', '')).lower() != 'addon':
        raise PermissionError("Trust record belongs to a different subject kind")
    if str(record.get('status', '')).lower() != 'active' or not record.get('approved') or not record.get('trusted') or record.get('quarantined'):
        raise PermissionError("Add-on trust is not active and approved")
    if str(record.get('trust_tier', '')).lower() not in {'first_party', 'verified_third_party', 'core'}:
        raise PermissionError("Add-on trust tier is insufficient")
    metadata = record.get('metadata') if isinstance(record.get('metadata'), dict) else {}
    pinned = str(metadata.get('addon_sha256') or '').lower()
    if not re.fullmatch(r'[0-9a-f]{64}', pinned) or pinned != fingerprint:
        raise PermissionError("Approved add-on fingerprint is absent or has changed")
    declared_path = str(record.get('manifest_path') or '')
    if declared_path and os.path.realpath(declared_path) not in {os.path.realpath(root), os.path.realpath(os.path.join(root, 'manifest.json'))}:
        raise PermissionError("Approved add-on path does not match selected directory")
    granted = set(record.get('permissions') or [])
    if not set(permissions).issubset(granted):
        raise PermissionError("Requested add-on permissions exceed granted scope")
    return record


def _govern_launch(addon_id: str, root: str, fingerprint: str, record: Dict[str, Any], confirmed: bool, inprocess: bool) -> None:
    """Invoke verified existing SecurityGovernor and AssuranceGate entrypoints."""
    if not confirmed:
        raise PermissionError("Explicit local user confirmation is required")
    contract = {
        'action_type': 'addon.launch', 'capability_name': 'addon.launch',
        'executor_name': addon_id, 'target': addon_id, 'target_path': root,
        'source_surface': 'gui', 'execution_mode': 'apply',
        'risk_level': 'TIER_3_PRIVILEGED_SYSTEM',  # arbitrary Python is privileged even in a subprocess
        'confirmed': True, 'user_confirmed': True,
        'requires_confirmation': True,
        'preconditions': ['TrustRegistry approval', 'pinned source integrity', 'local consent'],
        'verification_checks': ['process or in-process entrypoint launches', 'lifecycle state observed'],
        'expected_result': 'approved add-on starts and its lifecycle remains observable',
        'rollback_plan': ['stop the approved add-on process', 'disable the trust record if required'],
        'metadata': {'addon_id': addon_id, 'caller_id': addon_id, 'caller_kind': 'addon',
                     'source_sha256': fingerprint, 'publisher': str(record.get('publisher') or '')},
    }
    try:
        from SarahMemorySecurityGovernor import evaluate_action
        security = evaluate_action(contract, {'source': 'addon_launcher', 'user_confirmed': True})
    except Exception as exc:
        raise PermissionError("SecurityGovernor unavailable: launch denied") from exc
    if not isinstance(security, dict) or security.get('allow') is not True:
        raise PermissionError("SecurityGovernor denied add-on execution")
    try:
        from SarahMemoryAssuranceGate import evaluate_action_assurance
        assurance = evaluate_action_assurance(contract, {'allow': True, 'user_confirmed': True}, security)
    except Exception as exc:
        raise PermissionError("AssuranceGate unavailable: launch denied") from exc
    if not isinstance(assurance, dict) or assurance.get('allow') is not True:
        raise PermissionError("AssuranceGate denied add-on execution")


def _minimal_child_env() -> Dict[str, str]:
    """Use a minimal host environment; never forward tokens or secret overrides."""
    allow = {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP',
             'TMPDIR', 'HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH',
             'VIRTUAL_ENV', 'LANG', 'LC_ALL', 'LC_CTYPE', 'DISPLAY',
             'XDG_RUNTIME_DIR', 'TZ', 'APPDATA', 'LOCALAPPDATA'}
    env = {name: value for name, value in os.environ.items() if name.upper() in allow}
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


class AddonLauncher:
    """Existing Tk launcher interface; execution is governed and deny-by-default."""

    def __init__(self, parent):
        self.parent = parent
        self.selected_addon = tk.StringVar(master=parent)
        self.selected_file = tk.StringVar(master=parent)
        self.addons_window = None
        self.launch_window = None
        self.dropdown = None
        self.running_addons = _RUNTIME_STATE['processes']  # compatibility: list of tuples
        self.loaded_manifest_addons = _RUNTIME_STATE['manifest_records']
        self._process_meta = _RUNTIME_STATE['process_meta']
        self._log_threads = _RUNTIME_STATE['log_threads']
        self._display_folder_map: Dict[str, str] = {}
        self._lock = _RUNTIME_STATE['lock']
        self._last_errors: List[str] = []
        _INSTANCES.add(self)

    def _failure(self, label: str, error: Exception) -> None:
        message = f"{label}: {error}"
        self._last_errors.append(message[:300])
        self._last_errors = self._last_errors[-10:]
        _safe_audit('Addon Operation Denied', message)
        try:
            messagebox.showerror('Add-on Operation Denied', str(error)[:500])
        except Exception:
            logger.warning("Add-on action denied (%s)", type(error).__name__)

    def open_addons(self):
        if self.addons_window is not None:
            try:
                if self.addons_window.winfo_exists():
                    self.addons_window.lift()
                    return
            except Exception:
                pass
        self.addons_window = tk.Toplevel(self.parent)
        self.addons_window.title('Add-ons (Governed)')
        self.addons_window.geometry('460x410')
        self.addons_window.protocol('WM_DELETE_WINDOW', self.close_addons)
        ttk.Button(self.addons_window, text='Refresh List', command=self.refresh_addon_list).pack(pady=5)
        ttk.Label(self.addons_window, text='Select Add-on Folder').pack()
        ttk.Label(self.addons_window, text='Execution requires approved trust, pinned source, and confirmation.').pack(pady=5)
        self.refresh_addon_list()
        ttk.Button(self.addons_window, text='Inspect Selected (Read-only)', command=self.inspect_selected_addon).pack(pady=5)
        ttk.Button(self.addons_window, text='LOAD', command=self.addon1_LOADOPTIONS).pack(pady=10)
        ttk.Button(self.addons_window, text='Shutdown all Add-ons', command=self.addon3_SHUTDOWNADDONS).pack(pady=10)

    def close_addons(self):
        """Close UI only, retaining existing explicitly launched process semantics."""
        for attr in ('launch_window', 'addons_window'):
            window = getattr(self, attr, None)
            try:
                if window is not None:
                    window.destroy()
            except Exception:
                pass
            setattr(self, attr, None)
        self.dropdown = None

    def _reap_finished(self):
        with self._lock:
            alive = []
            for proc, logfile, folder in self.running_addons:
                if proc.poll() is None:
                    alive.append((proc, logfile, folder))
                    continue
                meta = self._process_meta.pop(proc.pid, {})
                worker = self._log_threads.pop(proc.pid, None)
                if worker is not None:
                    worker.join(timeout=0.2)
                else:
                    try:
                        logfile.close()
                    except Exception:
                        pass
                addon_id = meta.get('addon_id')
                if addon_id and addon_id in self.loaded_manifest_addons:
                    self.loaded_manifest_addons.pop(addon_id, None)
                _safe_audit('Addon Process Exited', f"addon={addon_id or folder} pid={proc.pid} code={proc.returncode}")
            self.running_addons[:] = alive

    def refresh_addon_list(self):
        try:
            self._reap_finished()
            # Discovery is bounded and does not follow symlinks.
            base = os.path.abspath(ADDONS_DIR)
            if not os.path.isdir(base):
                self.addon_subdirs = []
            else:
                with os.scandir(base) as scan:
                    entries = sorted(list(scan), key=lambda item: item.name.lower())
                if len(entries) > 512:
                    raise ValueError('Too many add-on directory entries for bounded discovery')
                self.addon_subdirs = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
        except Exception as exc:
            self.addon_subdirs = []
            self._failure('Add-on discovery failed', exc)
        previous = self._addon_folder_from_display(self.selected_addon.get())
        if self.dropdown is not None:
            try:
                self.dropdown.destroy()
            except Exception:
                pass
            self.dropdown = None
        self._display_folder_map = {}
        options = []
        for folder in self.addon_subdirs:
            tags = []
            try:
                if self._has_manifest(_addon_root(folder)):
                    tags.append('Manifest')
            except Exception:
                tags.append('Invalid')
            if self._is_manifest_loaded(folder):
                tags.append('Loaded')
            if self.is_addon_running(folder):
                tags.append('Running')
            label = folder + (f" ({', '.join(tags)})" if tags else '')
            options.append(label)
            self._display_folder_map[label] = folder
        if not options:
            self.selected_addon.set('')
            if self.addons_window is not None:
                ttk.Label(self.addons_window, text='No add-ons found.').pack()
            return
        preferred = next((label for label, folder in self._display_folder_map.items() if folder == previous), options[0])
        self.selected_addon.set(preferred)
        if self.addons_window is not None:
            self.dropdown = ttk.OptionMenu(self.addons_window, self.selected_addon, preferred, *options)
            self.dropdown.pack(pady=10)

    def is_addon_running(self, folder_name):
        self._reap_finished()
        with self._lock:
            return any(folder == folder_name and proc.poll() is None for proc, _, folder in self.running_addons)

    def addon1_LOADOPTIONS(self):
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            addon_path = _addon_root(folder)
            with os.scandir(addon_path) as scan:
                files = sorted(e.name for e in scan if e.is_file(follow_symlinks=False) and e.name.lower().endswith('.py'))
            self.available_py = files[:256]
            has_manifest = self._has_manifest(addon_path)
        except Exception as exc:
            self._failure('Cannot open add-on', exc)
            return
        if self.launch_window is not None:
            try:
                self.launch_window.destroy()
            except Exception:
                pass
        self.launch_window = tk.Toplevel(self.addons_window)
        self.launch_window.title(f'{folder} Add-on Options')
        self.launch_window.geometry('400x270')
        if has_manifest:
            ttk.Label(self.launch_window, text='Manifest detected (manifest.json).').pack(pady=6)
            ttk.Button(self.launch_window, text='LAUNCH (Manifest Add-on)', command=self.addon2_LAUNCH).pack(pady=6)
            if self.available_py:
                ttk.Separator(self.launch_window, orient='horizontal').pack(fill='x', pady=6)
                ttk.Label(self.launch_window, text='Legacy scripts in this folder:').pack()
                self.selected_file.set(self.available_py[0])
                self.dropdown_file = ttk.OptionMenu(self.launch_window, self.selected_file, self.available_py[0], *self.available_py)
                self.dropdown_file.pack(pady=8)
                ttk.Button(self.launch_window, text='LAUNCH (Legacy Script)', command=self._launch_selected_legacy).pack(pady=6)
        else:
            ttk.Label(self.launch_window, text='Choose a Python file:').pack(pady=6)
            if self.available_py:
                self.selected_file.set(self.available_py[0])
                self.dropdown_file = ttk.OptionMenu(self.launch_window, self.selected_file, self.available_py[0], *self.available_py)
                self.dropdown_file.pack(pady=10)
                ttk.Button(self.launch_window, text='LAUNCH', command=self.addon2_LAUNCH).pack(pady=10)
            else:
                ttk.Label(self.launch_window, text='No .py files found.').pack(pady=10)

    def _confirm(self, addon_id: str, fingerprint: str, mode: str) -> bool:
        return bool(messagebox.askyesno('Approve Add-on Execution',
            f'Add-on: {addon_id}\nExecution: {mode}\nSHA-256: {fingerprint[:16]}…\n\n'
            'Approved add-on Python code can access your host resources. '
            'Subprocess mode is NOT a sandbox. Continue?'))

    def inspect_selected_addon(self):
        """Read-only provenance preview for operator review; never grants trust."""
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            root = _addon_root(folder)
            manifest = self._read_manifest(root) if self._has_manifest(root) else None
            if self._has_manifest(root) and manifest is None:
                raise ValueError('Malformed manifest: cannot approve')
            fingerprint = _source_fingerprint(root)
            addon_id = str(manifest['addon_id']) if manifest else f'legacy:{folder}:<selected-script>.py'
            permissions = manifest.get('permissions', []) if manifest else []
            entry = manifest.get('entrypoint', {}) if manifest else {}
            detail = (
                f'ID: {addon_id}\nFolder: {folder}\n'
                f'Type: {manifest.get("type", "manifest") if manifest else "legacy"}\n'
                f'Entrypoint: {entry.get("module", "not specified")}\n'
                f'Requested permissions: {", ".join(permissions)[:256] or "none declared"}\n'
                f'SHA-256 (directory):\n{fingerprint}\n\n'
                'Review-only. No trust is granted or add-on code executed.\n'
                'The source digest must be independently approved in TrustRegistry.'
            )
            messagebox.showinfo('Add-on Source Review', detail)
            return {'addon_id': addon_id, 'sha256': fingerprint,
                    'permissions': list(permissions), 'execution_authority': False}
        except Exception as exc:
            self._failure('Inspection failed', exc)
            return None

    def addon2_LAUNCH(self):
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            root = _addon_root(folder)
            if self._has_manifest(root):
                self._launch_manifest_addon(root)
            else:
                filename = (self.selected_file.get() or '').strip()
                self._launch_legacy_script(os.path.join(root, filename), folder, filename)
            self.refresh_addon_list()
        except Exception as exc:
            self._failure('Launch failed', exc)

    def addon3_SHUTDOWNADDONS(self):
        if not messagebox.askyesno('Confirm Shutdown', 'Stop all locally tracked add-ons?'):
            return
        failures = []
        self._reap_finished()
        with self._lock:
            records = list(self.running_addons)
        for proc, logfile, folder in records:
            if not self._stop_process(proc):
                failures.append(f'{folder} pid={proc.pid}')
        with self._lock:
            for addon_id, rec in list(self.loaded_manifest_addons.items()):
                if rec.get('mode') != 'inprocess':
                    continue
                hook = (rec.get('hooks') or {}).get('shutdown')
                if callable(hook):
                    try:
                        hook(rec.get('context') or {})
                    except Exception as exc:
                        failures.append(f'{addon_id}: {type(exc).__name__}')
                        continue
                self.loaded_manifest_addons.pop(addon_id, None)
        self._reap_finished()
        if failures:
            _safe_audit('Addon Shutdown Incomplete', '; '.join(failures))
            messagebox.showwarning('Shutdown Incomplete', 'Some add-ons could not be confirmed stopped: ' + ', '.join(failures)[:300])
        else:
            _safe_audit('Addon Shutdown', 'All tracked add-ons stopped or exited')
            messagebox.showinfo('Shutdown', 'All tracked add-ons have stopped.')
        self.refresh_addon_list()

    @staticmethod
    def _stop_process(proc: subprocess.Popen) -> bool:
        if proc.poll() is not None:
            return True
        if os.name == 'nt':
            # Windows terminate-tree request first. Never claim descendant isolation.
            try:
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=4, check=False)
                proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
                return True
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
            return True
        except subprocess.TimeoutExpired:
            pass
        except (OSError, ProcessLookupError):
            if proc.poll() is not None:
                return True
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                # taskkill /T is necessary for descendants; process termination alone is insufficient.
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4, check=False)
                if proc.poll() is None:
                    proc.kill()
            proc.wait(timeout=2)
            return True
        except Exception:
            return proc.poll() is not None

    def _addon_folder_from_display(self, display_value: str) -> str:
        """Never strip arbitrary suffixes from real folder names."""
        value = (display_value or '').strip()
        return self._display_folder_map.get(value, value)

    def _has_manifest(self, addon_root: str) -> bool:
        try:
            return os.path.isfile(_contained(ADDONS_DIR, os.path.join(addon_root, 'manifest.json')))
        except Exception:
            return False

    def _is_manifest_loaded(self, folder_name: str) -> bool:
        try:
            root = _addon_root(folder_name)
            manifest = self._read_manifest(root)
            return bool(manifest and manifest['addon_id'] in self.loaded_manifest_addons)
        except Exception:
            return False

    def _read_manifest(self, addon_root: str):
        try:
            return _manifest_data(addon_root)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError, UnicodeError) as exc:
            _safe_audit('Addon Manifest Invalid', f'{os.path.basename(addon_root)}: {type(exc).__name__}')
            return None

    def _addon_exec_mode_from_manifest(self, manifest: dict) -> str:
        """Never let a manifest's own trusted flag elevate it to in-process execution."""
        if bool(getattr(config, 'SAFE_MODE', False)):
            return 'subprocess'
        execution = manifest.get('execution') if isinstance(manifest.get('execution'), dict) else {}
        mode = str(execution.get('mode') or 'auto').lower()
        if mode not in {'auto', 'inprocess', 'subprocess'}:
            raise ValueError('Unknown execution mode')
        if mode == 'inprocess':
            if not bool(getattr(config, 'ALLOW_TRUSTED_INPROCESS_ADDONS', False)):
                raise PermissionError('In-process add-ons disabled; request subprocess mode')
            return 'inprocess'
        return 'subprocess'

    def _launch_selected_legacy(self):
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            filename = (self.selected_file.get() or '').strip()
            root = _addon_root(folder)
            self._launch_legacy_script(os.path.join(root, filename), folder, filename)
            self.refresh_addon_list()
        except Exception as exc:
            self._failure('Legacy launch failed', exc)

    def _review(self, root: str, addon_id: str, permissions: List[str], mode: str) -> str:
        """Trust, content pin, direct user approval and two existing policy gates."""
        fingerprint = _source_fingerprint(root)
        record = _trust_review(addon_id, fingerprint, root, permissions)
        if mode == 'inprocess' and str(record.get('trust_tier')).lower() not in {'core', 'first_party'}:
            raise PermissionError('In-process execution requires independently trusted first-party or core classification')
        if not self._confirm(addon_id, fingerprint, mode):
            _safe_audit('Addon Launch Rejected', addon_id)
            raise PermissionError('User declined launch')
        _govern_launch(addon_id, root, fingerprint, record, True, mode == 'inprocess')
        # Confirm the same on-disk content is present after approvals.
        if _source_fingerprint(root) != fingerprint:
            raise PermissionError('Add-on changed during authorization')
        if not _safe_audit('Addon Launch Authorized', f'addon={addon_id} sha256={fingerprint[:20]} mode={mode}'):
            raise PermissionError('Audit unavailable: launch denied')
        return fingerprint

    def _open_log(self, root: str, filename: str):
        path = _contained(root, os.path.join(root, filename), allow_root=False)
        if os.path.lexists(path) and not os.path.isfile(path):
            raise ValueError('Invalid add-on log path')
        if os.path.exists(path) and os.path.getsize(path) >= _MAX_LOG_BYTES:
            rotated = _contained(root, path + '.1')
            if os.path.islink(rotated):
                raise ValueError('Log rotation target is symlinked')
            os.replace(path, rotated)
        return open(path, 'ab')

    def _start_process(self, args: List[str], root: str, folder: str, addon_id: str, logname: str):
        self._reap_finished()
        with self._lock:
            if len(self.running_addons) >= _MAX_RUNNING:
                raise RuntimeError('Concurrent add-on limit reached')
            if any(rec.get('addon_id') == addon_id for rec in self._process_meta.values()):
                raise RuntimeError('Add-on already running')
        logfile = self._open_log(root, logname)
        options: Dict[str, Any] = {'cwd': root, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT,
                                  'stdin': subprocess.DEVNULL, 'shell': False, 'env': _minimal_child_env(),
                                  'bufsize': 0}
        if os.name == 'nt':
            options['creationflags'] = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        else:
            options['start_new_session'] = True
        try:
            proc = subprocess.Popen(args, **options)
        except Exception:
            logfile.close()
            raise
        def drain_output():
            # Drain continuously to avoid deadlocks, but never store beyond budget.
            # A misbehaving add-on may still consume CPU/other resources: not a sandbox.
            written = logfile.tell()
            truncated = False
            try:
                if proc.stdout is not None:
                    while True:
                        chunk = proc.stdout.read(16384)
                        if not chunk:
                            break
                        room = max(0, _MAX_LOG_BYTES - written)
                        if room:
                            logfile.write(chunk[:room])
                            logfile.flush()
                            written += len(chunk[:room])
                        if len(chunk) > room:
                            truncated = True
            except Exception as exc:
                logger.warning('Add-on output collection interrupted: %s', type(exc).__name__)
            finally:
                if truncated:
                    _safe_audit('Addon Log Truncated', f'addon={addon_id} pid={proc.pid}')
                try:
                    if proc.stdout is not None:
                        proc.stdout.close()
                    logfile.close()
                except Exception:
                    pass
        worker = threading.Thread(target=drain_output, name='SarahAddonLog-' + str(proc.pid), daemon=True)
        with self._lock:
            self.running_addons.append((proc, logfile, folder))
            self._process_meta[proc.pid] = {'addon_id': addon_id, 'root': root, 'started': time.time()}
            self._log_threads[proc.pid] = worker
        worker.start()
        # Don't claim success if the process exited synchronously.
        time.sleep(0.05)
        if proc.poll() is not None:
            self._reap_finished()
            error_text = ''
            try:
                with open(os.path.join(root, logname), 'rb') as stream:
                    error_text = stream.read(_MAX_ERROR_OUTPUT).decode('utf-8', 'replace')
            except Exception:
                pass
            _safe_audit('Addon Process Failed', f'addon={addon_id} exit={proc.returncode}')
            raise RuntimeError(f'Add-on exited immediately (exit={proc.returncode}): {error_text[:300]}')
        if not _safe_audit('Addon Process Started', f'addon={addon_id} pid={proc.pid}'):
            self._stop_process(proc)
            self._reap_finished()
            raise PermissionError('Audit failed after spawn; process terminated')
        return proc

    def _launch_legacy_script(self, full_path: str, folder: str, display_name: str = ''):
        root = _addon_root(folder)
        file = _contained(root, full_path, allow_root=False)
        if not os.path.isfile(file) or not file.endswith('.py') or os.path.dirname(file) != root:
            raise ValueError('Legacy script must be a regular top-level Python source')
        addon_id = f'legacy:{folder}:{os.path.basename(file)}'
        self._review(root, addon_id, [], 'subprocess')
        proc = self._start_process([sys.executable, '-B', file], root, folder, addon_id,
                                   os.path.basename(file) + '_launch.log')
        messagebox.showinfo('Add-on Launched', f'{display_name or os.path.basename(file)} started (pid {proc.pid}).')
        return proc

    def _launch_manifest_addon(self, addon_root: str):
        root = _contained(ADDONS_DIR, addon_root, allow_root=False)
        manifest = self._read_manifest(root)
        if not manifest:
            raise ValueError('manifest.json missing or invalid')
        if str(manifest.get('type') or '').lower() == 'addon_bundle':
            return self._launch_manifest_bundle(root, manifest)
        mode = self._addon_exec_mode_from_manifest(manifest)
        return self._launch_manifest_inprocess(root, manifest) if mode == 'inprocess' else self._launch_manifest_subprocess(root, manifest)

    def _launch_manifest_bundle(self, bundle_root: str, manifest: dict):
        root = _contained(ADDONS_DIR, bundle_root, allow_root=False)
        bundle_id = manifest['addon_id']
        includes = manifest['includes']
        # Verify every path and manifest before the first executable operation.
        children = []
        seen = {bundle_id}
        for item in includes:
            rel = item['path']
            if os.path.isabs(rel):
                raise ValueError('Absolute bundle includes denied')
            child_root = _contained(root, os.path.join(root, rel), allow_root=False)
            if not os.path.isdir(child_root):
                raise ValueError('Missing bundle child directory')
            child = self._read_manifest(child_root)
            if not child or child['addon_id'] in seen or str(child.get('type') or '').lower() == 'addon_bundle':
                raise ValueError('Invalid, duplicate, or nested bundle child')
            seen.add(child['addon_id'])
            children.append((child_root, child))
        self._review(root, bundle_id, manifest.get('permissions', []), 'bundle')
        started = []
        try:
            for child_root, child in children:
                mode = self._addon_exec_mode_from_manifest(child)
                if mode == 'inprocess':
                    self._launch_manifest_inprocess(child_root, child)
                else:
                    self._launch_manifest_subprocess(child_root, child)
                started.append(child['addon_id'])
        except Exception as exc:
            # Roll back only children launched by this bundle, never unrelated processes.
            failed_rollbacks = []
            for addon_id in reversed(started):
                if not self._stop_addon_id(addon_id):
                    failed_rollbacks.append(addon_id)
            _safe_audit('Addon Bundle Failed', f'bundle={bundle_id} failed={type(exc).__name__} rollback_incomplete={failed_rollbacks}')
            if failed_rollbacks:
                raise RuntimeError(f'Bundle failed and rollback incomplete for {failed_rollbacks}') from exc
            raise
        self.loaded_manifest_addons[bundle_id] = {
            'manifest': manifest, 'hooks': {}, 'context': {'addon_path': root},
            'mode': 'bundle', 'children': started,
        }
        if not _safe_audit('Addon Bundle Loaded', f'bundle={bundle_id} children={len(started)}'):
            for addon_id in reversed(started):
                self._stop_addon_id(addon_id)
            self.loaded_manifest_addons.pop(bundle_id, None)
            raise PermissionError('Bundle audit failed; rollback requested')
        messagebox.showinfo('Add-on Bundle Loaded', f'{bundle_id}: {len(started)} children launched.')
        return started

    def _stop_addon_id(self, addon_id: str) -> bool:
        found = False
        for proc, _, _ in list(self.running_addons):
            meta = self._process_meta.get(proc.pid, {})
            if meta.get('addon_id') == addon_id:
                found = True
                if not self._stop_process(proc):
                    return False
        entry = self.loaded_manifest_addons.get(addon_id)
        if entry and entry.get('mode') == 'inprocess':
            found = True
            hook = (entry.get('hooks') or {}).get('shutdown')
            try:
                if callable(hook):
                    hook(entry.get('context') or {})
            except Exception:
                return False
            self.loaded_manifest_addons.pop(addon_id, None)
        self._reap_finished()
        return found

    def _context(self, root: str, manifest: dict) -> dict:
        return {
            'platform_version': getattr(config, 'PROJECT_VERSION', '9.0.0'),
            'addon_path': root,
            # This is a capability request description, not an execution grant.
            'permissions': manifest.get('permissions', []),
            'run_mode': getattr(config, 'RUN_MODE', 'local'),
            'device_mode': getattr(config, 'DEVICE_MODE', 'local_agent'),
            'data_dir': getattr(config, 'DATA_DIR', os.path.join(BASE_DIR, 'data')),
            'execution_authority': False,
        }

    def _launch_manifest_inprocess(self, addon_root: str, manifest: dict):
        root = _contained(ADDONS_DIR, addon_root, allow_root=False)
        if self._addon_exec_mode_from_manifest(manifest) != 'inprocess':
            raise PermissionError('In-process execution disabled')
        addon_id = manifest['addon_id']
        self._review(root, addon_id, manifest.get('permissions', []), 'inprocess')
        if addon_id in self.loaded_manifest_addons:
            raise RuntimeError('Add-on already loaded')
        module_name = manifest['entrypoint']['module']
        if '.' in module_name:
            raise PermissionError('In-process package entrypoints are disabled; use subprocess')
        path = _contained(root, os.path.join(root, module_name + '.py'))
        if not os.path.isfile(path):
            raise ValueError('In-process entrypoint must be a single local Python file')
        unique_name = '_sarah_addon_' + hashlib.sha256((root + addon_id).encode('utf-8')).hexdigest()[:24]
        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise ImportError('Unable to load approved add-on module')
        module = importlib.util.module_from_spec(spec)
        # No sys.path modification and no global module-name collision.
        spec.loader.exec_module(module)
        entry = getattr(module, manifest['entrypoint']['callable'], None)
        if not callable(entry):
            raise TypeError('Manifest callable not available')
        context = self._context(root, manifest)
        hooks = entry(context) or {}
        if not isinstance(hooks, dict):
            raise TypeError('In-process add-on must return a lifecycle hook mapping')
        self.loaded_manifest_addons[addon_id] = {'manifest': manifest, 'hooks': hooks,
                                                  'context': context, 'mode': 'inprocess'}
        _safe_audit('Addon InProcess Loaded', addon_id)
        messagebox.showinfo('Add-on Loaded', f'{addon_id} loaded in-process (cannot be forcibly isolated).')
        return hooks

    def _launch_manifest_subprocess(self, addon_root: str, manifest: dict):
        root = _contained(ADDONS_DIR, addon_root, allow_root=False)
        addon_id = manifest['addon_id']
        self._review(root, addon_id, manifest.get('permissions', []), 'subprocess')
        ctx = self._context(root, manifest)
        proc = self._start_process(
            [sys.executable, '-I', '-c', _SUBPROCESS_BOOTSTRAP,
             manifest['entrypoint']['module'], manifest['entrypoint']['callable'],
             root, json.dumps(ctx, separators=(',', ':'))],
            root, os.path.basename(root), addon_id, 'manifest_launch.log',
        )
        # A running child is not an imported in-process module; keep record truthful.
        _safe_audit('Addon Manifest Launched', f'addon={addon_id} pid={proc.pid}')
        messagebox.showinfo('Add-on Launched', f'{addon_id} launched as subprocess (pid {proc.pid}).')
        return proc


# --- SML ORGAN ADAPTER START ---
# Non-executing protocol adapter. Observing health never scans or launches add-ons.
SML_ORGAN_METADATA = {
    'name': 'SarahMemoryADDONLCHR',
    'version': 'v9.1.0a-enterprise-candidate',
    'category': 'addon_operations',
    'protocol_version': 'SML/1.0',
    'packet_version': 1,
    'omega_registry_version': 'Ω/1.0',
    'capabilities': ['addon.discovery', 'addon.manifest', 'addon.governed_launch', 'addon.shutdown'],
    'supported_missions': ['Conversation'],
    'supported_omega': ['Ω001'],
    'required_authority': ['Read'],
    'priority': 40,
    'trust_level': 'source_integrated',
    'internal_only': True,
    'metadata': {'sml_adapter': 'generic_non_executing', 'source_file': 'SarahMemoryADDONLCHR.py',
                 'execution_authority': False},
}


def sml_get_metadata():
    """Return detached metadata to avoid accidental caller mutation."""
    return json.loads(json.dumps(SML_ORGAN_METADATA))


def sml_health():
    """Report observable adapter conditions, without claiming verified runtime security."""
    available = os.path.isdir(ADDONS_DIR)
    launched = sum(1 for proc, _, _ in list(_RUNTIME_STATE['processes']) if proc.poll() is None)
    return {
        'status': 'Available' if available else 'Degraded',
        'availability': 1.0 if available else 0.0,
        'integrity': 0.0,  # Not verified by a read-only health observation.
        'performance': 0.0,
        'reliability': 0.0,
        'confidence': 0.5 if available else 0.0,
        'latency_ms': 0.0,
        'stability': 0.0,
        'compatibility': 0.0,
        'notes': ['Adapter importable; governance requires per-launch validation',
                  'Active tracked subprocesses: ' + str(launched),
                  'Subprocess execution does not provide OS sandbox isolation'],
    }


def sml_diagnostics():
    """Read-only bounded operational diagnostic, not an authorization decision."""
    health = sml_health()
    return {'status': 'OK' if health['availability'] else 'DEGRADED',
            'component': 'SarahMemoryADDONLCHR', 'sml_adapter': True,
            'metadata': sml_get_metadata(), 'health': health, 'execution_authority': False}


def sml_receive_packet(packet, *, action='observe', note='', updates=None):
    """Preserve the existing non-executing SML bridge."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryADDONLCHR', action=action,
                                note=note or 'organ observed packet', updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---
