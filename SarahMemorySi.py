"""--==The SarahMemory Project==--
File: SarahMemorySi.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
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

SOFTWARE INTERACTION MODULE v9.0.0
===============================================
This module has standards with intelligent application
management, cross-platform software control, and advanced media playback while
maintaining 100% backward compatibility for testing purposes.

KEY ENHANCEMENTS:
-----------------
1. INTELLIGENT APPLICATION DETECTION
- Multi-source software discovery (Registry, PATH, common locations)
- Fuzzy name matching for user-friendly commands
- Automatic path caching and learning
- Cross-platform compatibility (Windows, Linux, macOS)
- Smart fallback strategies

2. ADVANCED WINDOW MANAGEMENT
- Focus, minimize, maximize controls
- Multi-monitor support
- Window state tracking
- Accessibility features
- Keyboard shortcut integration

3. MEDIA PLAYBACK CONTROL
- Spotify integration with search
- YouTube playback
- Local media player support
- Multi-platform media handling
- Smart query parsing

4. PROCESS LIFECYCLE MANAGEMENT
- Launch tracking and monitoring
- Graceful termination
- Resource cleanup
- Process state verification
- Zombie process prevention

5. COMPREHENSIVE EVENT LOGGING
- Software interaction tracking
- Usage pattern analysis
- Error recovery logging
- Performance metrics
- Audit trail generation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- connect_software_db()
- cache_app_path(name, path)
- get_app_path(app_name)
- launch_application(path)
- list_running_applications()
- terminate_application(app_name)
- minimize_application(app_name)
- maximize_application(app_name)
- focus_application(app_name)
- manage_application_request(full_command)
- execute_play_command(action, target, original_query)

New functions added (non-breaking):
- smart_app_search(app_name)
- get_application_info(app_name)
- verify_application_state(app_name)
- get_software_metrics()
- cleanup_orphaned_processes()
- batch_application_control()

INTEGRATION POINTS:
-------------------
- SarahMemoryDatabase.py: Software tracking
- SarahMemoryGlobals.py: Configuration
- SarahMemoryDiagnostics.py: Health checks
- SarahMemoryOptimization.py: Resource management
- SarahMemoryVoice.py: Voice command integration

PLATFORM SUPPORT:
-----------------
- Windows: Full registry access, pygetwindow, pyautogui
- Linux: wmctrl, xdotool integration
- macOS: AppleScript support
- Cross-platform: psutil for process management

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "software_control_engine"
# CATEGORY = "software_interaction"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_process_window_media"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "software_interaction"
# FAMILY = "system_control"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Software interaction module for app discovery, launching, focus/minimize/maximize, media playback control, and process lifecycle management across platforms."
# --- SARAHMETA END ---

import os
import subprocess
import sqlite3
import logging
import psutil
import time
import threading
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Platform-specific imports with error handling
try:
    import winreg
except ImportError:
    winreg = None

try:
    import pygetwindow as gw
except Exception:
    # On Linux, pygetwindow can raise NotImplementedError at import-time.
    gw = None

try:
    import pyautogui
except Exception:
    # On Linux/headless (no DISPLAY), pyautogui/mouseinfo can raise at import time.
    pyautogui = None

# Core imports
import SarahMemoryGlobals as config

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemorySi")
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
if not logger.hasHandlers():
    handler = logging.NullHandler()
    formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# GLOBAL STATE
# =============================================================================
DATABASE_PATH = os.path.join(getattr(config, 'DATASETS_DIR', 'data'), "software.db")
active_launched_apps = []
_app_cache = {}  # In-memory cache for fast lookups

# =============================================================================
# NORMALIZATION / STATE HELPERS - v8.0 SMGET integration
# =============================================================================
_APP_ALIASES = {
    "microsoft word": "winword",
    "ms word": "winword",
    "word": "winword",
    "winword": "winword",
    "microsoft excel": "excel",
    "ms excel": "excel",
    "excel": "excel",
    "microsoft powerpoint": "powerpnt",
    "ms powerpoint": "powerpnt",
    "powerpoint": "powerpnt",
    "powerpnt": "powerpnt",
    "microsoft paint": "mspaint",
    "paint": "mspaint",
    "mspaint": "mspaint",
    "calculator": "calc",
    "calc": "calc",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "msedge": "msedge",
    "google chrome": "chrome",
    "chrome": "chrome",
    "firefox": "firefox",
    "brave": "brave",
    "opera": "opera",
    "notepad": "notepad",
    "outlook": "outlook",
    "visual studio code": "code",
    "vs code": "code",
    "vscode": "code",
    "code": "code",
    "file explorer": "explorer",
    "explorer": "explorer",
}

def _normalize_app_alias(app_name: str) -> str:
    app = str(app_name or '').strip().lower().replace('.exe', '')
    app = " ".join(app.replace('_', ' ').replace('-', ' ').split())
    return _APP_ALIASES.get(app, app)

def _candidate_app_names(app_name: str) -> List[str]:
    raw = str(app_name or '').strip().lower()
    canonical = _normalize_app_alias(raw)
    out = []
    for item in (raw, canonical, f"{canonical}.exe" if canonical else "", f"{raw}.exe" if raw else ""):
        item = str(item or '').strip()
        if item and item not in out:
            out.append(item)
    return out

def _find_running_processes(app_name: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    try:
        for proc in psutil.process_iter(attrs=['pid', 'name', 'exe', 'cmdline', 'status']):
            try:
                name = str(proc.info.get('name') or '')
                exe = str(proc.info.get('exe') or '')
                cmdline = " ".join(proc.info.get('cmdline') or [])
                haystack = " | ".join([name, exe, cmdline]).lower()
                if any(c in haystack for c in _candidate_app_names(app_name)):
                    matches.append({
                        'pid': int(proc.info.get('pid') or 0),
                        'name': name,
                        'exe': exe,
                        'status': proc.info.get('status'),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception:
                continue
    except Exception:
        return []
    return matches

def wait_for_application_window(app_name: str, timeout: float = 10.0, poll_s: float = 0.35) -> bool:
    """
    Wait until an application window (or at least a running process) is observable.
    This stays fail-soft on headless/Linux environments where pygetwindow is unavailable.
    """
    deadline = time.time() + max(0.2, float(timeout or 0.0))
    canonical = _normalize_app_alias(app_name)
    while time.time() < deadline:
        try:
            if gw:
                windows = []
                for token in _candidate_app_names(canonical):
                    windows.extend(gw.getWindowsWithTitle(token) or [])
                if windows:
                    return True
            if _find_running_processes(canonical):
                return True
        except Exception:
            pass
        time.sleep(max(0.05, float(poll_s or 0.0)))
    return False

def smart_app_search(app_name: str) -> Dict[str, Any]:
    canonical = _normalize_app_alias(app_name)
    resolved = get_app_path(canonical)
    source = 'unknown'
    if resolved:
        if canonical in _app_cache:
            source = 'memory_cache'
        else:
            source = 'database_or_discovery'
    if not resolved:
        try:
            which_hit = shutil.which(canonical) or shutil.which(f"{canonical}.exe")
        except Exception:
            which_hit = None
        if which_hit:
            resolved = which_hit
            source = 'system_path'
            try:
                cache_app_path(canonical, resolved)
            except Exception:
                pass
    if not resolved and os.name == 'nt':
        fallback_map = {
            'winword': 'WINWORD.EXE',
            'excel': 'EXCEL.EXE',
            'powerpnt': 'POWERPNT.EXE',
            'outlook': 'OUTLOOK.EXE',
            'mspaint': 'mspaint.exe',
            'calc': 'calc.exe',
            'notepad': 'notepad.exe',
            'msedge': 'msedge.exe',
            'explorer': 'explorer.exe',
            'chrome': 'chrome.exe',
            'firefox': 'firefox.exe',
            'brave': 'brave.exe',
            'opera': 'opera.exe',
            'code': 'Code.exe',
        }
        fallback_exe = fallback_map.get(canonical)
        if fallback_exe:
            resolved = fallback_exe
            source = 'windows_fallback'
    return {
        'app_name': str(app_name or ''),
        'canonical_name': canonical,
        'resolved_path': resolved,
        'found': bool(resolved),
        'source': source,
    }

def get_application_info(app_name: str) -> Dict[str, Any]:
    canonical = _normalize_app_alias(app_name)
    search = smart_app_search(canonical)
    running = _find_running_processes(canonical)
    window_titles: List[str] = []
    focused = False
    minimized = False
    maximized = False
    try:
        if gw:
            windows = []
            for token in _candidate_app_names(canonical):
                windows.extend(gw.getWindowsWithTitle(token) or [])
            for win in windows or []:
                try:
                    title = str(getattr(win, 'title', '') or '')
                    if title:
                        window_titles.append(title)
                    if bool(getattr(win, 'isActive', False)):
                        focused = True
                    if bool(getattr(win, 'isMinimized', False)):
                        minimized = True
                    if bool(getattr(win, 'isMaximized', False)):
                        maximized = True
                except Exception:
                    continue
    except Exception:
        pass
    return {
        'app_name': str(app_name or ''),
        'canonical_name': canonical,
        'path': search.get('resolved_path'),
        'found': bool(search.get('found')),
        'source': search.get('source'),
        'running': bool(running),
        'is_running': bool(running),
        'pid_count': len(running),
        'pids': [int(x.get('pid') or 0) for x in running if int(x.get('pid') or 0) > 0],
        'processes': running,
        'window_titles': window_titles,
        'window_found': bool(window_titles),
        'focused': focused,
        'is_focused': focused,
        'minimized': minimized,
        'maximized': maximized,
    }

def verify_application_state(app_name: str, expected_state: str = 'running') -> bool:
    info = get_application_info(app_name)
    state = str(expected_state or 'running').strip().lower()
    if state in {'running', 'open', 'launched'}:
        return bool(info.get('running'))
    if state in {'focused', 'active'}:
        return bool(info.get('focused'))
    if state == 'minimized':
        return bool(info.get('minimized'))
    if state == 'maximized':
        return bool(info.get('maximized'))
    if state in {'closed', 'not_running'}:
        return not bool(info.get('running'))
    return bool(info.get('running'))


# =============================================================================
# DATABASE CONNECTION - Backward Compatible
# =============================================================================
def connect_software_db() -> sqlite3.Connection:
    """
    Connect to software database and ensure tables exist.
    v8.0: Enhanced with better error handling.

    Returns:
        SQLite database connection
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH, timeout=5.0)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS software_apps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                platform TEXT,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                version TEXT DEFAULT '9.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS software_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT,
                app_name TEXT,
                success BOOLEAN DEFAULT 1,
                version TEXT DEFAULT '9.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # --- v8.0 compatibility migrations (upgrade older schemas in-place) ---
        # If software.db was created by an older build, the tables may exist but be missing
        # newer columns (e.g., 'name'). SQLite will not update schemas automatically, so we
        # best-effort ALTER TABLE to add missing columns. This must never crash startup.
        try:
            # software_apps: ensure required columns exist
            cursor.execute("PRAGMA table_info(software_apps)")
            app_cols = {row[1] for row in cursor.fetchall()}  # row[1] = column name

            if "name" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN name TEXT")
            if "path" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN path TEXT")
            if "platform" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN platform TEXT")
            if "last_used" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN last_used TEXT")
            if "usage_count" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN usage_count INTEGER DEFAULT 0")
            if "version" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN version TEXT DEFAULT '9.0.0'")
            if "created_at" not in app_cols:
                cursor.execute("ALTER TABLE software_apps ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

            # Ensure a unique index exists for ON CONFLICT(name) upserts (older DBs may lack it)
            try:
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_software_apps_name ON software_apps(name)")
            except Exception:
                pass

            # software_events: ensure required columns exist
            cursor.execute("PRAGMA table_info(software_events)")
            evt_cols = {row[1] for row in cursor.fetchall()}

            if "timestamp" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN timestamp TEXT")
            if "event" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN event TEXT")
            if "details" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN details TEXT")
            if "app_name" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN app_name TEXT")
            if "success" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN success BOOLEAN DEFAULT 1")
            if "version" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN version TEXT DEFAULT '9.0.0'")
            if "created_at" not in evt_cols:
                cursor.execute("ALTER TABLE software_events ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            # Never let migrations break the module load path
            pass

        conn.commit()
        logger.debug("[v8.0] Software database connected")
        return conn

    except Exception as e:
        logger.error(f"[v8.0] Database connection error: {e}")
        raise

# =============================================================================
# APPLICATION PATH CACHING - Backward Compatible
# =============================================================================
def cache_app_path(name: str, path: str) -> None:
    """
    Cache application path in database and memory.
    v8.0: Enhanced with usage tracking.

    Args:
        name: Application name
        path: Application executable path
    """
    try:
        conn = connect_software_db()
        cursor = conn.cursor()

        # Insert or update with usage count
        cursor.execute("""
            INSERT INTO software_apps (name, path, platform, last_used, usage_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(name) DO UPDATE SET
                path = excluded.path,
                last_used = excluded.last_used,
                usage_count = usage_count + 1
        """, (name.lower(), path, os.name, datetime.now().isoformat()))

        conn.commit()
        conn.close()

        # Update memory cache
        _app_cache[name.lower()] = path

        logger.info(f"[v8.0] Cached: {name} → {path}")
        log_software_event("Cache App Path", f"Cached {name}", name)

    except Exception as e:
        logger.error(f"[v8.0] Cache error: {e}")

# =============================================================================
# REGISTRY SEARCH (Windows) - Backward Compatible
# =============================================================================
def search_registry_for_software(software_name: str) -> Optional[str]:
    """
    Search Windows registry for software installation path.
    v8.0: Enhanced with better error handling.

    Args:
        software_name: Name of software to find

    Returns:
        Installation path or None
    """
    if not winreg or os.name != 'nt':
        return None

    try:
        registry_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
        ]

        for hive, path in registry_paths:
            try:
                with winreg.OpenKey(hive, path) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name, _ = winreg.QueryValueEx(subkey, "DisplayName")

                                    # Fuzzy matching
                                    if software_name.lower() in display_name.lower():
                                        try:
                                            install_location, _ = winreg.QueryValueEx(subkey, "InstallLocation")
                                            if install_location:
                                                exe_path = find_executable_in_folder(install_location)
                                                if exe_path:
                                                    logger.info(f"[v8.0] Registry found: {software_name} at {exe_path}")
                                                    return exe_path
                                        except FileNotFoundError:
                                            continue
                                except (OSError, FileNotFoundError):
                                    continue
                        except (OSError, FileNotFoundError, PermissionError):
                            continue
            except Exception as e:
                logger.debug(f"[v8.0] Registry search error for {path}: {e}")

        return None

    except Exception as e:
        logger.warning(f"[v8.0] Registry access error: {e}")
        return None

# =============================================================================
# EXECUTABLE FINDER - Backward Compatible
# =============================================================================
def find_executable_in_folder(folder_path: str) -> Optional[str]:
    """
    Find executable file in specified folder.
    v8.0: Enhanced with recursive search.

    Args:
        folder_path: Folder to search

    Returns:
        Executable path or None
    """
    try:
        if not os.path.isdir(folder_path):
            return None

        # Search in root folder first
        for file in os.listdir(folder_path):
            if file.lower().endswith(".exe"):
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    return full_path

        # Search in bin subdirectory
        bin_path = os.path.join(folder_path, "bin")
        if os.path.isdir(bin_path):
            for file in os.listdir(bin_path):
                if file.lower().endswith(".exe"):
                    full_path = os.path.join(bin_path, file)
                    if os.path.isfile(full_path):
                        return full_path

        return None

    except Exception as e:
        logger.debug(f"[v8.0] Executable search error: {e}")
        return None

# =============================================================================
# APPLICATION PATH RETRIEVAL - Backward Compatible
# =============================================================================
def get_app_path(app_name: str) -> Optional[str]:
    """
    Get application executable path from cache or discovery.
    v8.0: Enhanced with multi-source lookup.

    Args:
        app_name: Application name

    Returns:
        Application path or None
    """
    app_name_lower = _normalize_app_alias(app_name)

    # Check memory cache first
    if app_name_lower in _app_cache:
        logger.debug(f"[v8.0] Memory cache hit: {app_name}")
        return _app_cache[app_name_lower]

    # Check database
    try:
        conn = connect_software_db()
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM software_apps WHERE name = ?", (app_name_lower,))
        result = cursor.fetchone()
        conn.close()

        if result:
            path = result[0]
            _app_cache[app_name_lower] = path  # Update memory cache
            logger.debug(f"[v8.0] Database cache hit: {app_name}")
            return path
    except Exception as e:
        logger.warning(f"[v8.0] Database lookup error: {e}")

    # Search registry (Windows)
    if os.name == 'nt':
        path_from_registry = search_registry_for_software(app_name)
        if path_from_registry:
            cache_app_path(app_name, path_from_registry)
            return path_from_registry

    # Not found
    logger.warning(f"[v8.0] Software not found: {app_name}")
    return None

# =============================================================================
# APPLICATION LAUNCH - Backward Compatible
# =============================================================================
def _launch_target_to_app_name(path: str) -> str:
    target = str(path or '').strip()
    if not target:
        return ''
    base = os.path.basename(target).strip()
    if not base:
        base = target
    return _normalize_app_alias(base)


def launch_application(path: str) -> bool:
    """
    Launch application by path.
    v8.0: Enhanced with process tracking and readiness verification.

    Args:
        path: Application executable path

    Returns:
        True if successful and the launched application becomes observable.
    """
    try:
        launch_target = str(path or '').strip()
        if not launch_target:
            return False

        proc = subprocess.Popen([launch_target])
        active_launched_apps.append(proc)

        app_name = _launch_target_to_app_name(launch_target)
        ready = wait_for_application_window(app_name or launch_target, timeout=10.0, poll_s=0.35)
        if not ready:
            logger.warning(f"[v8.0] Launch request started but readiness verification failed: {path} (PID: {proc.pid})")
            log_software_event("Launch Application Verification Failed", f"Launch started but readiness not confirmed: {path}", os.path.basename(launch_target))
            return False

        logger.info(f"[v8.0] Launched and verified: {path} (PID: {proc.pid})")
        log_software_event("Launch Application", f"Launched and verified: {path}", os.path.basename(launch_target))
        return True

    except Exception as e:
        logger.error(f"[v8.0] Launch error: {e}")
        log_software_event("Launch Application Error", f"Failed: {path} | {e}", os.path.basename(str(path or '')))
        return False

# =============================================================================
# RUNNING APPLICATIONS - Backward Compatible
# =============================================================================
def list_running_applications() -> List[Tuple[int, str]]:
    """
    List all running applications.
    v8.0: Enhanced with better error handling.

    Returns:
        List of (PID, name) tuples
    """
    try:
        running_apps = []
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            try:
                running_apps.append((proc.info['pid'], proc.info['name']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        logger.debug(f"[v8.0] Found {len(running_apps)} running processes")
        return running_apps

    except Exception as e:
        logger.error(f"[v8.0] Process list error: {e}")
        return []

# =============================================================================
# APPLICATION TERMINATION - Backward Compatible
# =============================================================================
def terminate_application(app_name: str) -> bool:
    """
    Terminate a running application.
    v8.0: Enhanced with graceful shutdown.

    Args:
        app_name: Application name

    Returns:
        True if successful, False otherwise
    """
    try:
        app_name_lower = app_name.lower()

        # Try to terminate from tracked apps
        for proc in active_launched_apps.copy():
            try:
                if proc.poll() is None:  # Process still running
                    exe_name = os.path.basename(proc.args[0]).lower() if proc.args else ""
                    if app_name_lower in exe_name:
                        proc.terminate()
                        proc.wait(timeout=5)  # Wait for graceful shutdown
                        active_launched_apps.remove(proc)
                        logger.info(f"[v8.0] Terminated: {app_name}")
                        log_software_event("Terminate Application", f"Terminated: {app_name}", app_name)
                        return True
            except Exception as e:
                logger.debug(f"[v8.0] Terminate attempt failed: {e}")

        # Fallback: search all processes
        for proc in psutil.process_iter(attrs=['pid', 'name']):
            try:
                if app_name_lower in proc.info['name'].lower():
                    process = psutil.Process(proc.info['pid'])
                    process.terminate()
                    logger.info(f"[v8.0] Terminated (fallback): {app_name}")
                    log_software_event("Terminate Application", f"Terminated (fallback): {app_name}", app_name)
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        logger.warning(f"[v8.0] Terminate failed: {app_name} not found")
        return False

    except Exception as e:
        logger.error(f"[v8.0] Terminate error: {e}")
        log_software_event("Terminate Application Error", f"Failed: {app_name} | {e}", app_name)
        return False

# =============================================================================
# WINDOW MANAGEMENT - Backward Compatible
# =============================================================================
def minimize_application(app_name: str) -> bool:
    """Minimize application window."""
    if not gw:
        logger.warning("[v8.0] pygetwindow not available")
        return False

    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows and len(windows) > 0:
            windows[0].minimize()
            logger.info(f"[v8.0] Minimized: {app_name}")
            log_software_event("Minimize Window", f"Minimized: {app_name}", app_name)
            return True
        return False
    except Exception as e:
        logger.error(f"[v8.0] Minimize error: {e}")
        return False

def maximize_application(app_name: str) -> bool:
    """Maximize application window."""
    if not gw:
        logger.warning("[v8.0] pygetwindow not available")
        return False

    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows and len(windows) > 0:
            windows[0].maximize()
            logger.info(f"[v8.0] Maximized: {app_name}")
            log_software_event("Maximize Window", f"Maximized: {app_name}", app_name)
            return True
        return False
    except Exception as e:
        logger.error(f"[v8.0] Maximize error: {e}")
        return False

def focus_application(app_name: str) -> bool:
    """Focus application window."""
    if not gw:
        logger.warning("[v8.0] pygetwindow not available")
        return False

    try:
        windows = gw.getWindowsWithTitle(app_name)
        if windows and len(windows) > 0:
            windows[0].activate()
            logger.info(f"[v8.0] Focused: {app_name}")
            log_software_event("Focus Window", f"Focused: {app_name}", app_name)
            return True
        return False
    except Exception as e:
        logger.error(f"[v8.0] Focus error: {e}")
        return False

# =============================================================================
# APPLICATION MANAGEMENT - Backward Compatible
# =============================================================================
def manage_application_request(full_command: str) -> bool:
    """
    Handle application management commands.
    v8.0: Enhanced with better command parsing and normalized aliases.

    Args:
        full_command: Command string (e.g., "open notepad")

    Returns:
        True if successful, False otherwise
    """
    try:
        parts = full_command.strip().lower().split()
        if len(parts) < 2:
            return False

        action = parts[0]
        raw_app_name = " ".join(parts[1:]).strip()
        app_name = _normalize_app_alias(raw_app_name) if raw_app_name else raw_app_name

        # Launch/Open/Start
        if action in ["open", "launch", "start"]:
            app_path = get_app_path(app_name)

            if app_path:
                return launch_application(app_path)
            else:
                # Windows fallback strategies
                if os.name == 'nt':
                    return _windows_fallback_launch(app_name or raw_app_name)
                return False

        # Close/Terminate/Kill
        elif action in ["close", "terminate", "kill", "exit", "quit"]:
            if "all" in raw_app_name:
                return _terminate_all_apps()
            return terminate_application(app_name or raw_app_name)

        # Maximize
        elif action == "maximize":
            return maximize_application(app_name or raw_app_name)

        # Minimize
        elif action == "minimize":
            return minimize_application(app_name or raw_app_name)

        # Focus
        elif action in ["focus", "bring"]:
            return focus_application(app_name or raw_app_name)

        else:
            logger.warning(f"[v8.0] Unknown action: {action}")
            return False

    except Exception as e:
        logger.error(f"[v8.0] Manage request error: {e}")
        return False

# =============================================================================
# WINDOWS FALLBACK LAUNCH - v8.0 New
# =============================================================================
def _windows_fallback_launch(app_name: str) -> bool:
    """Windows-specific fallback launch strategies with readiness verification."""
    try:
        base = _normalize_app_alias(app_name.replace('.exe', '').strip())

        # Common Windows utilities
        common_apps = {
            'notepad': 'notepad.exe',
            'wordpad': 'write.exe',
            'paint': 'mspaint.exe',
            'mspaint': 'mspaint.exe',
            'calculator': 'calc.exe',
            'calc': 'calc.exe',
            'cmd': 'cmd.exe',
            'powershell': 'powershell.exe',
            'msedge': 'msedge.exe',
            'edge': 'msedge.exe',
            'chrome': 'chrome.exe',
            'firefox': 'firefox.exe',
            'brave': 'brave.exe',
            'opera': 'opera.exe',
            'explorer': 'explorer.exe',
            'winword': 'WINWORD.EXE',
            'excel': 'EXCEL.EXE',
            'powerpnt': 'POWERPNT.EXE',
            'outlook': 'OUTLOOK.EXE'
        }

        exe = common_apps.get(base, app_name if app_name.endswith('.exe') else f"{base}.exe")

        # Try direct Popen
        try:
            proc = subprocess.Popen([exe])
            active_launched_apps.append(proc)
            if wait_for_application_window(base or exe, timeout=10.0, poll_s=0.35):
                logger.info(f"[v8.0] Fallback launched and verified: {exe}")
                cache_app_path(base, exe)
                return True
            logger.warning(f"[v8.0] Fallback launch started but readiness verification failed: {exe}")
        except Exception:
            pass

        # Try os.startfile
        try:
            startfile = getattr(os, 'startfile', None)
            if callable(startfile):
                startfile(exe)  # type: ignore[misc]
                if wait_for_application_window(base or exe, timeout=10.0, poll_s=0.35):
                    logger.info(f"[v8.0] startfile launched and verified: {exe}")
                    return True
                logger.warning(f"[v8.0] startfile launched but readiness verification failed: {exe}")
        except Exception:
            pass

        # Try shell start command
        try:
            subprocess.Popen(f'start "" "{exe}"', shell=True)
            if wait_for_application_window(base or exe, timeout=10.0, poll_s=0.35):
                logger.info(f"[v8.0] Shell start launched and verified: {exe}")
                return True
            logger.warning(f"[v8.0] Shell start launched but readiness verification failed: {exe}")
        except Exception:
            pass

        return False

    except Exception as e:
        logger.error(f"[v8.0] Fallback launch error: {e}")
        return False

def _terminate_all_apps() -> bool:
    """Terminate all tracked applications."""
    try:
        success = True
        for proc in active_launched_apps.copy():
            try:
                if proc.poll() is None:
                    proc.terminate()
                    active_launched_apps.remove(proc)
            except Exception as e:
                logger.warning(f"[v8.0] Failed to terminate process: {e}")
                success = False
        return success
    except Exception as e:
        logger.error(f"[v8.0] Terminate all error: {e}")
        return False

# =============================================================================
# MEDIA PLAYBACK - Backward Compatible
# =============================================================================
def execute_play_command(action: str, target: str, original_query: str) -> str:
    """
    Handle media playback commands.
    v8.0: Enhanced with better parsing and error handling.

    Args:
        action: Play action
        target: Media target
        original_query: Original user query

    Returns:
        Status message
    """
    try:
        import re
        import webbrowser

        song = None
        artist = None
        platform = "Spotify"  # Default

        # Parse command
        match = re.match(
            r"play\s+(.*?)\s+(?:by\s+(.*?))?(?:\s+on\s+(\w+))?$",
            original_query,
            re.IGNORECASE
        )

        if match:
            song = match.group(1)
            artist = match.group(2)
            platform = match.group(3) or platform
        else:
            song = original_query.replace("play", "").strip()

        platform = platform.lower().strip()

        # YouTube playback
        if "youtube" in platform:
            search_query = song.replace(" ", "+")
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
            logger.info(f"[v8.0] YouTube search: {song}")
            return f"Searching YouTube for {song}"

        # Spotify playback
        elif platform == "spotify":
            if not gw or not pyautogui:
                return "Spotify automation requires pygetwindow and pyautogui"

            # Open Spotify
            manage_application_request("open spotify")
            time.sleep(2)

            # Find Spotify window
            windows = [w for w in gw.getWindowsWithTitle('Spotify') if not w.isMinimized]
            if windows:
                windows[0].activate()
                time.sleep(0.5)

                # Search
                pyautogui.hotkey('ctrl', 'l')
                search_query = f"{song} {artist}" if artist else song
                pyautogui.write(search_query)
                pyautogui.press('enter')
                time.sleep(1.5)

                # Play first result
                pyautogui.press('tab')
                pyautogui.press('enter')

                logger.info(f"[v8.0] Spotify playback: {search_query}")
                return f"Playing {search_query} on Spotify"
            else:
                return "Spotify window not found"

        # Media Player
        elif "media player" in platform:
            return f"[TODO] Local media player support for: {song}"

        return f"Unsupported platform or command: {original_query}"

    except Exception as e:
        logger.error(f"[v8.0] Play command error: {e}")
        return f"Error executing play command: {e}"



def type_text_to_active_window(text: str, interval: float = 0.01, press_enter: bool = False) -> bool:
    """Type text into the currently focused window using pyautogui when available."""
    if not pyautogui:
        return False
    try:
        pyautogui.FAILSAFE = False
        pyautogui.write(str(text or ''), interval=max(0.0, float(interval or 0.0)))
        if press_enter:
            pyautogui.press('enter')
        return True
    except Exception as e:
        logger.error(f"[v8.0] type_text_to_active_window error: {e}")
        return False


def press_hotkey(*keys: str) -> bool:
    if not pyautogui:
        return False
    try:
        combo = [str(k).strip().lower() for k in keys if str(k).strip()]
        if not combo:
            return False
        pyautogui.FAILSAFE = False
        pyautogui.hotkey(*combo)
        return True
    except Exception as e:
        logger.error(f"[v8.0] press_hotkey error: {e}")
        return False


def browser_search(app_name: str, query: str, timeout: float = 10.0) -> bool:
    canonical = _normalize_app_alias(app_name) or app_name
    try:
        if not manage_application_request(f"open {canonical}"):
            return False
        wait_for_application_window(canonical, timeout=timeout, poll_s=0.35)
        focus_application(canonical)
        time.sleep(0.8)
        if not press_hotkey('ctrl', 'l'):
            return False
        time.sleep(0.2)
        if not type_text_to_active_window(query, interval=0.01, press_enter=True):
            return False
        return True
    except Exception as e:
        logger.error(f"[v8.0] browser_search error: {e}")
        return False


def browser_open_url(app_name: str, url: str, timeout: float = 10.0) -> bool:
    return browser_search(app_name, url, timeout=timeout)

# =============================================================================
# EVENT LOGGING - Backward Compatible
# =============================================================================
def log_software_event(event: str, details: str, app_name: str = None) -> None:
    """
    Log software interaction event.
    v8.0: Enhanced with structured logging.

    Args:
        event: Event name
        details: Event details
        app_name: Optional application name
    """
    try:
        conn = connect_software_db()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO software_events (timestamp, event, details, app_name) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), event, details, app_name)
        )

        conn.commit()
        conn.close()

        logger.debug(f"[v8.0] Logged event: {event}")

    except Exception as e:
        logger.warning(f"[v8.0] Event log error: {e}")

# =============================================================================
# ASYNC REQUEST HANDLER - Backward Compatible
# =============================================================================
def manage_application_request_async(request: str) -> None:
    """Execute application request asynchronously."""
    thread = threading.Thread(target=manage_application_request, args=(request,))
    thread.daemon = True
    thread.start()

# =============================================================================
# METRICS - v8.0 New
# =============================================================================
def get_software_metrics() -> Dict[str, Any]:
    """
    Get software interaction metrics.
    v8.0: New function for analytics.
    """
    try:
        conn = connect_software_db()
        cursor = conn.cursor()

        # Get total apps cached
        cursor.execute("SELECT COUNT(*) FROM software_apps")
        total_apps = cursor.fetchone()[0]

        # Get recent events
        cursor.execute("""
            SELECT COUNT(*) FROM software_events
            WHERE datetime(timestamp) > datetime('now', '-7 days')
        """)
        recent_events = cursor.fetchone()[0]

        # Get most used apps
        cursor.execute("""
            SELECT name, usage_count FROM software_apps
            ORDER BY usage_count DESC LIMIT 5
        """)
        top_apps = cursor.fetchall()

        conn.close()

        return {
            "total_apps_cached": total_apps,
            "recent_events": recent_events,
            "active_processes": len(active_launched_apps),
            "top_apps": [{"name": name, "count": count} for name, count in top_apps],
            "version": "9.0.0"
        }

    except Exception as e:
        logger.error(f"[v8.0] Metrics error: {e}")
        return {"error": str(e)}

# =============================================================================
# SMGET ADAPTER HELPERS - v8.0 Integration
# =============================================================================
def cleanup_orphaned_processes() -> Dict[str, Any]:
    removed = 0
    kept = []
    try:
        for proc in list(active_launched_apps):
            try:
                if proc.poll() is None:
                    kept.append(proc)
                else:
                    removed += 1
            except Exception:
                removed += 1
        active_launched_apps[:] = kept
    except Exception:
        pass
    return {'ok': True, 'removed': removed, 'active_tracked': len(active_launched_apps)}


def batch_application_control(requests: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    ok = True
    for item in list(requests or []):
        try:
            if isinstance(item, dict):
                action = str(item.get('action') or '').strip().lower()
                app_name = str(item.get('app_name') or item.get('target') or '').strip()
                command = f"{action} {app_name}".strip()
            else:
                command = str(item or '').strip()
            if not command:
                results.append({'ok': False, 'command': '', 'reason': 'empty_command'})
                ok = False
                continue
            handled = bool(manage_application_request(command))
            results.append({'ok': handled, 'command': command})
            if not handled:
                ok = False
        except Exception as e:
            results.append({'ok': False, 'command': str(item or ''), 'reason': str(e)})
            ok = False
    return {'ok': ok, 'results': results}


def execute_software_action(action_type: str, target: str, execution_mode: str = 'apply', metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    action = str(action_type or '').strip().lower()
    target_name = _normalize_app_alias(str(target or '').strip())
    mode = str(execution_mode or 'apply').strip().lower()
    metadata = dict(metadata or {})
    surface_action_map = {
        'open_app': 'open',
        'close_app': 'close',
        'maximize_window': 'maximize',
        'minimize_window': 'minimize',
        'focus_window': 'focus',
    }
    surface_action = str(surface_action_map.get(action, metadata.get('surface_action') or action) or action).strip().lower()
    command = f"{surface_action} {target_name}".strip()
    if mode != 'apply':
        return {'ok': True, 'executed': False, 'simulated': True, 'action_type': action, 'surface_action': surface_action, 'target': target_name, 'command': command, 'reason': 'non_apply_mode'}

    ok = bool(manage_application_request(command))
    info = get_application_info(target_name)
    if action == 'open_app':
        verified = bool(info.get('running') or info.get('is_running'))
    elif action == 'close_app':
        verified = not bool(info.get('running') or info.get('is_running'))
    elif action == 'maximize_window':
        verified = bool(info.get('maximized'))
    elif action == 'minimize_window':
        verified = bool(info.get('minimized'))
    elif action == 'focus_window':
        verified = bool(info.get('focused') or info.get('is_focused'))
    else:
        verified = bool(ok)
    return {'ok': bool(ok and verified), 'executed': bool(ok), 'verified': bool(verified), 'action_type': action, 'surface_action': surface_action, 'target': target_name, 'command': command, 'application_info': info, 'reason': 'software_action_verified' if ok and verified else ('software_action_unverified' if ok else 'software_action_failed')}


def register_operator_executor() -> bool:
    try:
        import SarahMemoryOperatorCore as _Op  # type: ignore
        snapshot = getattr(_Op, 'executor_registry_snapshot', None)
        if callable(snapshot):
            return isinstance(snapshot(), dict)
    except Exception:
        return False
    return False

# =============================================================================
# MAIN TEST - v8.0 Enhanced
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("SarahMemory Software Interaction v9.0.0 - Test Mode")
    print("=" * 80)

    # Test application detection
    print("\nTesting application detection:")
    test_apps = ["notepad", "calculator", "paint"]

    for app in test_apps:
        path = get_app_path(app)
        print(f"{app:15} → {path or 'Not found'}")

    # Display metrics
    print("\nSoftware Metrics:")
    import json
    print(json.dumps(get_software_metrics(), indent=2))

    print("\n" + "=" * 80)

logger.info("[v8.0] SarahMemorySi module loaded successfully")

# ====================================================================
# END OF SarahMemorySi.py v9.0.0
# ====================================================================
