"""--==The SarahMemory Project==--
File: SarahMemoryIntegration.py
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
SarahMemory v9.0 - Integration & Main Menu System
Integration with Enhanced Features
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "integration_orchestrator"
# CATEGORY = "runtime_integration"
# USER_FACING = True
# UI_EXPOSURE = "family_shell_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "integration"
# FAMILY = "runtime"
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
# NOTES = "Main integration and runtime orchestration layer for menu flow, GUI launch, action ticket execution, bootstrap sequencing, runtime service startup, and controlled system coordination."
# --- SARAHMETA END ---

import logging
import os
import json
import subprocess
import re
import sys
import time
import threading
import asyncio
import hashlib
from ftplib import FTP, error_temp, error_perm, error_proto, all_errors
from tqdm import tqdm

# SARAHMEMORY_PATCH_NOTE: GUI import is optional at module import time.
# Missing PyQt5 or WebEngine must not kill the entire offline core boot.
# The menu will report GUI unavailability and allow safe shutdown instead.
try:
    from SarahMemoryGUI import run_gui  # type: ignore
    _GUI_IMPORT_ERROR = None
except Exception as _sm_gui_import_error:  # pragma: no cover
    run_gui = None  # type: ignore
    _GUI_IMPORT_ERROR = _sm_gui_import_error
from SarahMemoryVoice import synthesize_voice, speak_text_async, shutdown_tts
from SarahMemoryDiagnostics import run_self_check
import SarahMemoryGlobals as config

# =============================================================================
# FILESYSTEM ACTION EXECUTOR (Kernel "do it" lane)
# =============================================================================
try:
    import SarahMemoryFilesystem as _FS  # type: ignore
except Exception:
    _FS = None

_FS_BACKUP_MGR = None
_FS_SCANNER = None

def execute_action_ticket(ticket: dict, *, confirm: bool = False) -> dict:
    """Execute a normalized action ticket locally using SarahMemoryFilesystem.

    Governance:
    - Respects SAFE_MODE and LOCAL_ONLY_MODE gates.
    - If ticket requires confirmation, caller must pass confirm=True.
    - Never calls network/API.

    Returns: {ok, action, result, error, needs_confirm}
    """
    if not isinstance(ticket, dict):
        return {"ok": False, "error": "ticket must be a dict", "action": None}

    action = str(ticket.get("action") or "").strip().lower()
    args = ticket.get("args") if isinstance(ticket.get("args"), dict) else {}
    requires_confirm = bool(ticket.get("requires_confirm", True))
    safety_level = str(ticket.get("safety_level") or "medium").strip().lower()

    # Tighten confirmation in SAFE_MODE for anything non-trivial
    try:
        if bool(getattr(config, "SAFE_MODE", True)) and safety_level in ("medium", "high"):
            requires_confirm = True
    except Exception:
        pass

    if requires_confirm and not confirm:
        return {"ok": False, "action": action, "needs_confirm": True, "error": "confirmation_required"}

    if _FS is None:
        return {"ok": False, "action": action, "error": "SarahMemoryFilesystem not available"}

    global _FS_BACKUP_MGR, _FS_SCANNER
    try:
        if _FS_BACKUP_MGR is None:
            _FS_BACKUP_MGR = _FS.BackupManager()
    except Exception:
        _FS_BACKUP_MGR = None

    try:
        if _FS_SCANNER is None:
            _FS_SCANNER = _FS.FileScanner()
    except Exception:
        _FS_SCANNER = None

    try:
        if action in ("file_copy", "copy"):
            ok = bool(_FS.FileOperations.safe_copy(args.get("source", ""), args.get("destination", ""), verify=bool(args.get("verify", True))))
            return {"ok": ok, "action": action, "result": {}}

        if action in ("file_move", "move"):
            ok = bool(_FS.FileOperations.safe_move(args.get("source", ""), args.get("destination", ""), overwrite=bool(args.get("overwrite", False))))
            return {"ok": ok, "action": action, "result": {}}

        if action in ("file_rename", "rename"):
            ok = bool(_FS.FileOperations.safe_rename(args.get("old_path", ""), args.get("new_path", "")))
            return {"ok": ok, "action": action, "result": {}}

        if action in ("file_delete", "delete"):
            ok = bool(_FS.FileOperations.safe_delete(args.get("file_path", ""), secure=bool(args.get("secure", False))))
            return {"ok": ok, "action": action, "result": {}}

        if action in ("file_attrs", "set_attributes", "file_attributes"):
            ok = bool(_FS.FileOperations.set_file_attributes(
                args.get("file_path", ""),
                readonly=args.get("readonly", None),
                hidden=args.get("hidden", None),
                system=args.get("system", None),
            ))
            return {"ok": ok, "action": action, "result": {}}

        if action in ("backup_full", "backup_create_full"):
            if _FS_BACKUP_MGR is None:
                return {"ok": False, "action": action, "error": "BackupManager unavailable"}
            path = _FS_BACKUP_MGR.create_full_backup(source_dir=args.get("source_dir", None), destination=args.get("destination", None))
            return {"ok": bool(path), "action": action, "result": {"backup_path": path}}

        if action in ("backup_incremental", "backup_create_incremental"):
            if _FS_BACKUP_MGR is None:
                return {"ok": False, "action": action, "error": "BackupManager unavailable"}
            path = _FS_BACKUP_MGR.create_incremental_backup(source_dir=args.get("source_dir", None), base_backup=args.get("base_backup", None))
            return {"ok": bool(path), "action": action, "result": {"backup_path": path}}

        if action in ("backup_restore", "restore_backup"):
            if _FS_BACKUP_MGR is None:
                return {"ok": False, "action": action, "error": "BackupManager unavailable"}
            ok = _FS_BACKUP_MGR.restore_backup(args.get("backup_path", ""), destination=args.get("destination", None), verify_checksum=bool(args.get("verify_checksum", True)))
            return {"ok": bool(ok), "action": action, "result": {}}

        if action in ("backup_rotate", "rotate_backups"):
            if _FS_BACKUP_MGR is None:
                return {"ok": False, "action": action, "error": "BackupManager unavailable"}
            _FS_BACKUP_MGR.rotate_old_backups(max_count=int(args.get("max_count", 50)), max_age_days=int(args.get("max_age_days", 30)))
            return {"ok": True, "action": action, "result": {}}

        if action in ("scan_file", "file_scan"):
            if _FS_SCANNER is None:
                return {"ok": False, "action": action, "error": "FileScanner unavailable"}
            data = _FS_SCANNER.scan_file(args.get("file_path", ""), quarantine_on_threat=bool(args.get("quarantine_on_threat", True)))
            return {"ok": True, "action": action, "result": data}

        if action in ("scan_dir", "scan_directory", "dir_scan"):
            if _FS_SCANNER is None:
                return {"ok": False, "action": action, "error": "FileScanner unavailable"}
            data = _FS_SCANNER.scan_directory(args.get("directory", ""), recursive=bool(args.get("recursive", True)), quarantine_on_threat=bool(args.get("quarantine_on_threat", True)))
            return {"ok": True, "action": action, "result": {"items": data}}

        if action in ("quarantine_restore", "restore_quarantine"):
            if _FS_SCANNER is None:
                return {"ok": False, "action": action, "error": "FileScanner unavailable"}
            ok = _FS_SCANNER.restore_from_quarantine(args.get("quarantine_path", ""), restore_path=args.get("restore_path", None))
            return {"ok": bool(ok), "action": action, "result": {}}

        return {"ok": False, "action": action, "error": f"unknown_action:{action}"}

    except Exception as e:
        return {"ok": False, "action": action, "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# CONTEXT BUFFER INITIALIZATION
# =============================================================================
if config.ENABLE_CONTEXT_BUFFER:
    import SarahMemoryAiFunctions as context
    # Runtime optimization: avoid database/context writes during import unless explicitly requested.
    if str(os.getenv("SARAH_CONTEXT_INIT_ON_IMPORT", "0")).strip().lower() in ("1", "true", "yes", "on"):
        try:
            context.init_context_history()
        except Exception:
            pass

# =============================================================================
# LOGGER SETUP - v9.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemoryIntegration")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", False)) else logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - v9.0 - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.propagate = False

# =============================================================================
# GLOBAL STATE
# =============================================================================
terminate_flag = threading.Event()

_shutdown_started = False
_shutdown_lock = threading.RLock()


def _cfg_bool(name: str, default: bool = False) -> bool:
    """Read a boolean setting from config or environment without making shutdown fragile."""
    try:
        value = getattr(config, name, default)
    except Exception:
        value = default
    try:
        env_val = os.getenv(name, None)
        if env_val is None:
            env_val = os.getenv(f"SARAH_{name}", None)
        if env_val is not None and str(env_val).strip() != "":
            value = env_val
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _runtime_pid_paths() -> list:
    paths = []
    try:
        data_dir = getattr(config, "DATA_DIR", os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data"))
    except Exception:
        data_dir = os.path.join(os.getcwd(), "data")
    for name in ("sarahmemory.pid", "local_api.pid"):
        try:
            paths.append(os.path.join(data_dir, name))
        except Exception:
            pass
    return paths


def _clear_runtime_state_files() -> None:
    """Mark runtime state offline and remove PID marker files."""
    try:
        data_dir = getattr(config, "DATA_DIR", os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data"))
        os.makedirs(data_dir, exist_ok=True)
        state_file = str(getattr(config, "SERVER_STATE_PATH", os.path.join(getattr(config, "SETTINGS_DIR", os.path.join(data_dir, "settings")), "server_state.json")))
        state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}
        now_ts = time.time()
        notes = state.get("notes") if isinstance(state.get("notes"), list) else []
        notes = (notes + [f"{time.strftime('%Y-%m-%d %H:%M:%S')} shutdown:integration"])[-20:]
        state.update({
            "ok": True,
            "ts": now_ts,
            "source": "SarahMemoryIntegration",
            "notes": notes,
            "main_running": False,
            "main_pid": None,
            "api_running": False,
            "api_pid": None,
            "MAIN_RUNNING": False,
            "MAIN_PID": None,
            "API_RUNNING": False,
            "API_PID": None,
        })
        tmp = state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, state_file)
    except Exception:
        pass

    for path in _runtime_pid_paths():
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _call_main_process_cleanup() -> bool:
    """Call SarahMemoryMain/__main__ lifecycle cleanup when available."""
    for module_name in ("SarahMemoryMain", "__main__"):
        try:
            mod = sys.modules.get(module_name)
            if mod is None:
                continue
            cleanup = getattr(mod, "main_process_cleanup", None)
            if callable(cleanup):
                cleanup(reason="integration_shutdown")
                return True
            stop_api = getattr(mod, "stop_local_api_server", None)
            if callable(stop_api):
                stop_api(timeout=4.0)
                return True
        except Exception:
            pass
    return False


def _kill_api_port_fallback(port: int) -> None:
    """Best-effort port/PID cleanup for the local Flask API child process."""
    try:
        pids = set()
        for raw in (
            os.environ.get("SARAHMEMORY_LOCAL_API_PID"),
            os.environ.get("SARAHMEMORY_API_PID"),
        ):
            if raw and str(raw).strip().isdigit():
                pids.add(int(str(raw).strip()))

        for path in _runtime_pid_paths():
            try:
                if path.endswith("local_api.pid") and os.path.exists(path):
                    raw = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
                    if raw.isdigit():
                        pids.add(int(raw))
            except Exception:
                pass

        if os.name == "nt":
            try:
                out = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], text=True, errors="ignore")
            except Exception:
                out = ""
            needle = f":{int(port)}"
            for line in out.splitlines():
                if needle in line and "LISTENING" in line.upper():
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pids.add(int(parts[-1]))

            for pid in sorted(pids):
                if pid <= 0 or pid == os.getpid():
                    continue
                try:
                    info = subprocess.check_output(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        text=True,
                        errors="ignore",
                    ).strip()
                    image = info.split(",")[0].strip('"').lower() if info else ""
                    if image and ("python" in image or "wsgi" in image or "gunicorn" in image):
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=3.0,
                        )
                except Exception:
                    pass
        else:
            import signal
            for pid in sorted(pids):
                if pid <= 0 or pid == os.getpid():
                    continue
                try:
                    os.killpg(pid, signal.SIGTERM)
                except Exception:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
    except Exception:
        pass
    finally:
        try:
            os.environ.pop("SARAHMEMORY_LOCAL_API_PID", None)
            os.environ.pop("SARAHMEMORY_API_PID", None)
        except Exception:
            pass


def _join_sarahmemory_threads(timeout_each: float = 0.75) -> None:
    """Give non-daemon SarahMemory threads a small window to exit after terminate_flag is set."""
    try:
        current = threading.current_thread()
        for th in list(threading.enumerate()):
            try:
                if th is current or not th.is_alive():
                    continue
                name = str(getattr(th, "name", "") or "")
                if not (name.startswith("SM_") or name.startswith("SarahMemory")):
                    continue
                th.join(timeout=timeout_each)
            except Exception:
                pass
    except Exception:
        pass


# =============================================================================
# RUNTIME SERVICES (POST-API) - v9.0
# =============================================================================
# Start SelfAware/Synapes/Diagnostics ONLY after the local API is up.
_runtime_services_started = False
_runtime_services_lock = threading.Lock()

def _start_runtime_services_once() -> None:
    """Start post-API runtime services exactly once (best-effort, non-blocking)."""
    global _runtime_services_started

    # Idempotency guard
    try:
        with _runtime_services_lock:
            if _runtime_services_started:
                return
            _runtime_services_started = True
    except Exception:
        if _runtime_services_started:
            return
        _runtime_services_started = True

    # Wait briefly for local API readiness (do NOT hard-fail)
    try:
        host = getattr(config, "DEFAULT_HOST", "127.0.0.1")
        port = int(getattr(config, "DEFAULT_PORT", 8000))
    except Exception:
        host, port = "127.0.0.1", 8000

    status_url = f"http://{host}:{port}/api/status"
    api_ok = False
    try:
        import requests  # type: ignore
        for _ in range(40):  # ~10s @ 0.25s
            if terminate_flag.is_set():
                break
            try:
                r = requests.get(status_url, timeout=0.75)
                if getattr(r, "status_code", 0) == 200:
                    api_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.25)
    except Exception:
        api_ok = False

    if api_ok:
        logger.info("[v9.0][RUNTIME] Local API is ready. Starting post-API runtime services...")
    else:
        logger.warning("[v9.0][RUNTIME] Local API not confirmed ready. Starting runtime services in degraded mode...")

    # Diagnostics / Self-check (post-API) - optional background thread.
    # Normal boot already runs initialization checks; keep this off by default to avoid duplicate DB/file reads.
    if _cfg_bool("RUNTIME_START_SELF_CHECK", False):
        def _diag_worker():
            try:
                run_self_check()
            except Exception as e:
                logger.debug("[v9.0][RUNTIME] run_self_check failed: %s", e)

        try:
            threading.Thread(target=_diag_worker, name="SM_RuntimeDiagnostics", daemon=True).start()
        except Exception:
            pass
    else:
        logger.info("[v9.0][RUNTIME] Runtime self-check skipped by optimized runtime policy.")

    # Synapses: awareness tick + background training dispatcher (if available)
    try:
        import SarahMemorySynapes as syn  # type: ignore

        if hasattr(syn, "start_training_dispatcher_background"):
            if _cfg_bool("SYNAPES_TRAINING_DISPATCHER_ON_BOOT", False):
                try:
                    syn.start_training_dispatcher_background()
                    logger.info("[v9.0][RUNTIME] Synapes training dispatcher started.")
                except Exception as e:
                    logger.debug("[v9.0][RUNTIME] start_training_dispatcher_background failed: %s", e)
            else:
                logger.info("[v9.0][RUNTIME] Synapes training dispatcher skipped by optimized runtime policy.")

        if hasattr(syn, "synapes_awareness_tick") and _cfg_bool("SYNAPES_AWARENESS_ON_BOOT", False):
            try:
                interval = float(os.getenv("SARAH_SYNAPES_AWARENESS_INTERVAL_SEC", str(getattr(config, "SYNAPES_AWARENESS_INTERVAL_SEC", 900.0))))
            except Exception:
                interval = 900.0
            interval = max(300.0, interval)

            def _syn_awareness_loop():
                try:
                    while not terminate_flag.is_set():
                        try:
                            syn.synapes_awareness_tick(enqueue_job=False, max_rows_per_table=25, mode="background")
                        except TypeError:
                            break
                        except Exception:
                            pass
                        time.sleep(interval)
                except Exception:
                    pass

            try:
                threading.Thread(target=_syn_awareness_loop, name="SM_SynapesAwareness", daemon=True).start()
                logger.info("[v9.0][RUNTIME] Synapes awareness loop started at %.1fs interval.", interval)
            except Exception as e:
                logger.debug("[v9.0][RUNTIME] Synapes awareness thread failed: %s", e)
        else:
            logger.info("[v9.0][RUNTIME] Synapes awareness loop skipped by optimized runtime policy.")

    except Exception as e:
        logger.debug("[v9.0][RUNTIME] Synapes module not available: %s", e)

    # SelfAware runtime loop - governed standby by default.
    try:
        # SARAHMEMORY_PATCH_NOTE 2026-06-23:
        # Runtime integration must not start SelfAware merely because developer
        # flags exist. External agents, stale configs, or copied env vars must not
        # turn the local UI into an autonomous executor. Startup requires the new
        # explicit autonomy master flag plus the SelfAware-specific autostart flag.
        neosky = bool(getattr(config, "NEOSKYMATRIX", False))
        dev = bool(getattr(config, "DEVELOPERSMODE", False))
        autonomy_master = bool(getattr(config, "SARAHMEMORY_AUTONOMOUS_STARTUP_ENABLED", False))
        selfaware_auto = bool(getattr(config, "SARAHMEMORY_SELFAWARE_AUTOSTART_ENABLED", False))

        if neosky and dev and autonomy_master and selfaware_auto:
            try:
                import SarahMemorySelfAware as sma  # type: ignore
                if hasattr(sma, "run_autonomous_loop"):
                    threading.Thread(target=sma.run_autonomous_loop, name="SM_SelfAware", daemon=True).start()
                    logger.warning("[v9.0][RUNTIME][GOV] SelfAware started after explicit governed autostart flags.")
            except Exception as e:
                logger.debug("[v9.0][RUNTIME] SelfAware start failed: %s", e)
        else:
            logger.info("[v9.0][RUNTIME][GOV] SelfAware held in governed standby; no silent autonomous startup.")
    except Exception:
        pass



# =============================================================================
# v9.0 BOOTSTRAP STARTUP SEQUENCE
# =============================================================================
def bootstrap_startup():
    """
    v9.0 Enhanced: Run updater, dataset sync, and kick off vector warmup before menu.
    Includes better error handling and progress reporting.
    """
    logger.info("[v9.0][BOOTSTRAP] Starting bootstrap sequence...")
    
    try:
        # =====================================================================
        # REMOTE DATASET SYNC
        # =====================================================================
        try:
            host = getattr(config, "FTP_HOST", "")
            user = getattr(config, "FTP_USER", "")
            pw = getattr(config, "FTP_PASS", "")
            remote = getattr(config, "FTP_REMOTE_DATASETS_DIR", 
                           "/public_html/api/data/memory/datasets")
            local = getattr(config, "DATASETS_DIR", 
                          os.path.join(os.getcwd(), "data", "memory", "datasets"))
            
            if host and user and pw:
                try:
                    logger.info("[v9.0][BOOTSTRAP] Attempting remote dataset sync...")
                    sync_dataset_bidirectional(local, host, user, pw, remote)
                    logger.info("[v9.0][BOOTSTRAP] Remote dataset sync completed")
                except Exception as se:
                    logger.warning(f"[v9.0][BOOTSTRAP] Dataset sync skipped: {se}")
        except Exception as e:
            logger.debug(f"[v9.0][BOOTSTRAP] No FTP sync: {e}")
        
        # =====================================================================
        # VECTOR WARMUP
        # =====================================================================
        try:
            import SarahMemoryResearch as research
            if getattr(config, "LOCAL_DATA_ENABLED", True) and _cfg_bool("BOOT_VECTOR_WARMUP", False):
                logger.info("[v9.0][BOOTSTRAP] Warming up vector search...")
                _ = research.get_research_data("warmup")
                logger.info("[v9.0][BOOTSTRAP] Vector warmup completed")
            else:
                logger.info("[v9.0][BOOTSTRAP] Vector warmup skipped by optimized runtime policy.")
        except Exception as ve:
            logger.debug(f"[v9.0][BOOTSTRAP] Vector warmup skipped: {ve}")
    
    except Exception as e:
        logger.warning(f"[v9.0][BOOTSTRAP] bootstrap_startup error: {e}")


# =============================================================================
# FILE HASH UTILITIES
# =============================================================================
def hash_file(filepath):
    """
    v9.0: Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        str: Hexadecimal hash digest
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()


def upload_with_progress(ftp, filepath, filename):
    """
    v9.0: Upload file to FTP server with progress bar.
    
    Args:
        ftp: FTP connection object
        filepath: Local file path
        filename: Remote filename
    """
    filesize = os.path.getsize(filepath)
    bar = tqdm(total=filesize, unit='B', unit_scale=True, 
               desc=f"[v9.0] Uploading {filename}", ncols=80)
    start = time.time()
    
    try:
        with open(filepath, 'rb') as f:
            def callback(chunk):
                bar.update(len(chunk))
            ftp.storbinary(f"STOR {filename}", f, 1024, callback=callback)
    except Exception as e:
        bar.close()
        logger.error(f"[v9.0][UPLOAD ERROR] {filename} failed: {type(e).__name__}: {e}")
        return
    
    bar.close()
    elapsed = time.time() - start
    logger.info(f"[v9.0][UPLOAD COMPLETE] {filename} ({filesize} bytes) in {elapsed:.2f} sec")


# =============================================================================
# BIDIRECTIONAL DATASET SYNCHRONIZATION
# =============================================================================
def sync_dataset_bidirectional():
    """
    v9.0 Enhanced: Bidirectional dataset sync with FTP server.
    
    Features:
    - Respects SAFE_MODE / LOCAL_ONLY and 'is_offline()' if present
    - Uses FTP settings from SarahMemoryGlobals.py
    - Skips sync when FTP_BACKUP_SCHEDULE is "never" or not due yet
    - Compares hashes and uploads/downloads changed files only
    - Writes a .last_ftp_backup.txt stamp on success
    """
    logger.info("[v9.0][SYNC] Starting bidirectional dataset synchronization...")
    
    # =========================================================================
    # SAFETY CHECKS
    # =========================================================================
    offline = False
    safe_mode = False
    
    try:
        if hasattr(config, "is_offline"):
            offline = config.is_offline()
    except Exception:
        offline = False
    
    try:
        safe_mode = getattr(config, "SAFE_MODE", False) or getattr(config, "LOCAL_ONLY_MODE", False)
    except Exception:
        safe_mode = False
    
    if safe_mode or offline:
        logger.info("[v9.0][SYNC] Skipping dataset sync due to safe mode or offline status.")
        return

    # =========================================================================
    # PATH AND CONFIG SETUP
    # =========================================================================
    data_dir = getattr(config, "DATA_DIR", os.path.join(os.getcwd(), "data"))
    local_dir = getattr(config, "DATASETS_DIR", 
                       os.path.join(os.getcwd(), "data", "memory", "datasets"))
    os.makedirs(local_dir, exist_ok=True)

    ftp_host = getattr(config, "FTP_HOST", "ftp.sarahmemory.com")
    ftp_port = int(getattr(config, "FTP_PORT", 21))
    ftp_user = getattr(config, "FTP_USER", "anonymous")
    ftp_pass = getattr(config, "FTP_PASS", "")
    remote_dir = (getattr(config, "FTP_REMOTE_DIR", None) or 
                 getattr(config, "FTP_REMOTE_DATASETS_DIR", 
                        "/public_html/api/data/memory/datasets"))

    # =========================================================================
    # SCHEDULE GATE
    # =========================================================================
    schedule_kind = getattr(config, "FTP_BACKUP_SCHEDULE", "weekly")
    
    try:
        from datetime import datetime, timedelta
        days = getattr(config, "schedule_to_days", lambda k: 7)(schedule_kind)
        stamp_file = os.path.join(local_dir, ".last_ftp_backup.txt")
        
        if days == 0:  # "never"
            logger.info(f"[v9.0][SYNC] FTP backup skipped (policy: {schedule_kind}).")
            return
        
        should_run = True
        try:
            with open(stamp_file, "r", encoding="utf-8") as sf:
                last = datetime.fromisoformat(sf.read().strip())
            should_run = (datetime.now() - last).days >= days
        except Exception:
            should_run = True
        
        if not should_run:
            logger.info(f"[v9.0][SYNC] FTP backup skipped (next run not due; policy: {schedule_kind}).")
            return
    
    except Exception as _sched_e:
        logger.warning(f"[v9.0][SYNC] Schedule gate error (continuing): {_sched_e}")

    # =========================================================================
    # FTP SYNC
    # =========================================================================
    try:
        logger.info(f"[v9.0][SYNC] Connecting to FTP server at {ftp_host}:{ftp_port}...")
        ftp = FTP(timeout=45)
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_user, ftp_pass)
        
        # Change to remote directory (create if needed)
        try:
            ftp.cwd(remote_dir)
        except error_perm:
            # Try to create the remote path if missing
            parts = [p for p in remote_dir.split("/") if p]
            cur = ""
            for p in parts:
                cur += "/" + p
                try:
                    ftp.cwd(cur)
                except error_perm:
                    try:
                        ftp.mkd(cur)
                        ftp.cwd(cur)
                    except error_perm:
                        pass
            ftp.cwd(remote_dir)

        # Hash helper
        def _hash_file(fp):
            import hashlib
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for b in iter(lambda: f.read(65536), b""):
                    h.update(b)
            return h.hexdigest()

        # Collect file lists
        try:
            remote_list = ftp.nlst()
        except Exception:
            remote_list = []
        
        local_list = [f for f in os.listdir(local_dir) 
                     if os.path.isfile(os.path.join(local_dir, f))]
        combined = sorted(set(remote_list + local_list))

        tmp_dl = os.path.join(local_dir, ".tmp_download")
        if os.path.exists(tmp_dl):
            try:
                os.remove(tmp_dl)
            except Exception:
                pass

        # Sync each file
        for fname in combined:
            if fname in [".", "..", ".ftpquota", ".htaccess", ".last_ftp_backup.txt"]:
                continue
            
            lp = os.path.join(local_dir, fname)
            local_exists = os.path.isfile(lp)
            remote_exists = fname in remote_list

            local_hash = _hash_file(lp) if local_exists else None
            remote_hash = None
            
            if remote_exists:
                try:
                    with open(tmp_dl, "wb") as f:
                        ftp.retrbinary(f"RETR {fname}", f.write)
                    remote_hash = _hash_file(tmp_dl)
                except error_perm as ep:
                    logger.warning(f"[v9.0][FTP RETR SKIP] {fname}: {ep}")
                    remote_hash = None

            # Compare and sync
            if local_hash != remote_hash:
                if not local_exists and remote_exists:
                    logger.info(f"[v9.0][DOWNLOAD] {fname} missing locally. Downloading...")
                    try:
                        with open(lp, "wb") as wf, open(tmp_dl, "rb") as rf:
                            wf.write(rf.read())
                    except Exception as e:
                        logger.error(f"[v9.0][DOWNLOAD ERROR] {fname}: {e}")
                
                elif local_exists:
                    logger.info(f"[v9.0][UPLOAD] Updating remote: {fname}")
                    size = os.path.getsize(lp)
                    bar = tqdm(total=size, unit='B', unit_scale=True, 
                              desc=f"[v9.0] Uploading {fname}", ncols=80)
                    try:
                        with open(lp, "rb") as rf:
                            def _cb(chunk):
                                bar.update(len(chunk))
                            ftp.storbinary(f"STOR {fname}", rf, 1024, callback=_cb)
                    except Exception as e:
                        logger.error(f"[v9.0][UPLOAD ERROR] {fname}: {e}")
                    finally:
                        bar.close()
                else:
                    logger.info(f"[v9.0][SYNC] Skipping {fname} (no local or remote?)")
            else:
                logger.info(f"[v9.0][MATCH] {fname} already synced.")

            # Cleanup temp
            try:
                if os.path.exists(tmp_dl):
                    os.remove(tmp_dl)
            except Exception:
                pass

        ftp.quit()
        
        # Write success stamp
        try:
            from datetime import datetime
            with open(os.path.join(local_dir, ".last_ftp_backup.txt"), "w", encoding="utf-8") as sf:
                sf.write(datetime.now().isoformat())
        except Exception as _e:
            logger.warning(f"[v9.0][SYNC] Could not write last backup stamp: {_e}")
        
        logger.info("[v9.0][SYNC COMPLETE] Bi-directional dataset sync finished.")

    except all_errors as e:
        logger.warning(f"[v9.0][SYNC ERROR] Dataset sync failed: {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"[v9.0][SYNC ERROR] {type(e).__name__}: {e}")


# =============================================================================
# LOOP DETECTION
# =============================================================================
def detect_loop(response):
    """
    v9.0: Detect if the AI is repeating the same response (loop detection).
    
    Args:
        response: The response to check
    
    Returns:
        bool: True if loop detected, False otherwise
    """
    if not config.ENABLE_CONTEXT_BUFFER:
        return False
    
    recent_responses = [entry.get('final_response', '') for entry in context.get_context()]
    count = recent_responses.count(response)
    threshold = config.LOOP_DETECTION_THRESHOLD + (len(recent_responses) // 10)
    
    return count >= threshold


# =============================================================================
# VOICE CHAT THREAD
# =============================================================================
def run_voice_chat():
    """
    v9.0 Enhanced: Voice chat loop with ambient noise calibration and context awareness.
    """
    try:
        logger.info("[v9.0] Starting voice chat thread with ambient noise calibration...")
        time.sleep(1.5)
        
        while not terminate_flag.is_set():
            logger.info("[v9.0] Listening for voice input...")
            result = context.get_voice_input()
            
            if result is None or result == "":
                logger.warning("[v9.0] No speech detected or not understood. Retrying...")
                continue
            
            logger.info(f"[v9.0] Voice input recognized: {result}")
            
            # Classify intent
            intent = context.classify_intent(result)
            logger.info(f"[v9.0] Intent classified as: {intent}")
            
            # Get personality response
            personality_response = context.integrate_with_personality(result)
            logger.info(f"[v9.0] Personality response: {personality_response}")
            
            final_response = personality_response
            
            # Loop detection
            if detect_loop(final_response):
                logger.warning("[v9.0] Loop detected. Modifying response.")
                final_response += " (Additional details available on request.)"
            
            # Add to context
            if config.ENABLE_CONTEXT_BUFFER:
                context.add_to_context({
                    "user_input": result,
                    "intent": intent,
                    "final_response": final_response,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            
            # Speak response
            synthesize_voice(final_response)
    
    except Exception as e:
        logger.error(f"[v9.0] Voice chat error: {e}")
    finally:
        logger.info("[v9.0] Voice chat loop terminated.")


# =============================================================================
# GUI LAUNCHER
# =============================================================================
def launch_gui():
    """
    v9.0 Enhanced: Launch the main GUI with voice chat integration.
    """
    voice_thread = None
    try:
        logger.info("[v9.0] Launching main GUI...")
        # Ensure runtime services are active before GUI mainloop
        try:
            _start_runtime_services_once()
        except Exception:
            pass
        synthesize_voice("Loading Main GUI interface, Please Wait.")

        # Daemonize the voice thread so a blocked microphone read never prevents process exit.
        voice_thread = threading.Thread(target=run_voice_chat, name="SM_VoiceChat", daemon=True)
        voice_thread.start()

        if run_gui is None:
            print("\n[v9.0] GUI unavailable; install PyQt5/PyQtWebEngine or use web/headless mode.")
            logger.error(f"[v9.0] GUI unavailable: {_GUI_IMPORT_ERROR}")
            return
        run_gui()
        logger.info("[v9.0] GUI closed; returning to integration menu.")

    except Exception as e:
        logger.error(f"[v9.0] GUI Launch Error: {e}")
    finally:
        terminate_flag.set()
        try:
            shutdown_tts()
        except Exception:
            pass
        if voice_thread is not None:
            try:
                voice_thread.join(timeout=2.0)
            except Exception:
                pass


# =============================================================================
# SHUTDOWN SEQUENCE
# =============================================================================
def shutdown_sequence(exit_code: int = 0):
    """
    v9.0 Enhanced: Clean shutdown with one authoritative lifecycle gate.

    This function is intentionally idempotent:
    - GUI close may call it.
    - menu option 2 may call it.
    - Ctrl+C / exception cleanup may call into Main separately.
    - duplicate calls should not hang or double-kill.
    """
    global _shutdown_started

    try:
        with _shutdown_lock:
            if _shutdown_started:
                raise SystemExit(exit_code)
            _shutdown_started = True
    except SystemExit:
        raise
    except Exception:
        if _shutdown_started:
            raise SystemExit(exit_code)
        _shutdown_started = True

    terminate_flag.set()

    print("\n" + "═" * 78)
    print("  SARAHMEMORY v9.0 - SHUTTING DOWN")
    print("═" * 78)

    # Voice confirmation is best-effort only. Shutdown must never wait on TTS.
    try:
        speak_text_async("Shutting down. Have a great day!")
        time.sleep(min(0.35, float(getattr(config, "SHUTDOWN_TTS_GRACE_SECONDS", 0.25) or 0.25)))
    except Exception:
        pass

    logger.info("[v9.0] Initiating safe shutdown procedures.")

    # Stop voice/TTS early so a TTS engine thread cannot keep python.exe alive.
    try:
        shutdown_tts()
        print("  ✓ TTS shutdown requested")
    except Exception as e:
        logger.debug("[v9.0] TTS shutdown skipped: %s", e)

    # Central module cleanup: shared frames, OpenCV windows, context, etc.
    try:
        import SarahMemoryInitialization as init
        init.safe_shutdown()
        print("  ✓ Initialization cleanup completed")
    except Exception as e:
        logger.warning(f"[v9.0] Initialization safe shutdown hook failed: {e}")

    # Stop local API child process through Main when possible.
    called = False
    try:
        called = _call_main_process_cleanup()
        if called:
            print("  ✓ Main/API lifecycle cleanup completed")
    except Exception as e:
        logger.debug("[v9.0] Main lifecycle cleanup failed: %s", e)

    # Fallback collection is reserved for a missing lifecycle owner. Hard PID
    # termination is never the normal shutdown path.
    try:
        port_val = os.environ.get("PORT") or str(getattr(config, "DEFAULT_PORT", "8000"))
        port = int(port_val)
    except Exception:
        port = 8000

    if not called and _cfg_bool("SARAH_ALLOW_HARD_PROCESS_KILL_FALLBACK", False):
        try:
            _kill_api_port_fallback(port)
            print("  ✓ Local API fallback cleanup completed")
        except Exception:
            pass

    # Mark state offline and remove stale pid markers so WebUI health is not poisoned on relaunch.
    try:
        _clear_runtime_state_files()
        print("  ✓ Runtime state and PID markers cleared")
    except Exception:
        pass

    # Give known SarahMemory threads a brief chance to see terminate_flag and exit.
    try:
        _join_sarahmemory_threads(timeout_each=0.75)
    except Exception:
        pass

    print("\n  ✓ Shutdown complete. Thank you for using SarahMemory!")
    print("  ✓ Visit https://www.sarahmemory.com for updates\n")

    try:
        logger.info("[v9.0] Safe shutdown completed successfully.")
    except Exception:
        pass

    # Flush before final process exit.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    try:
        logging.shutdown()
    except Exception:
        pass

    # Desktop runtime default: force the current process to exit only after cleanup.
    # This replaces the external pytaskkill.bat dependency without killing unrelated Python jobs.
    if _cfg_bool("SARAH_FORCE_PROCESS_EXIT_ON_SHUTDOWN", False):
        os._exit(int(exit_code))

    raise SystemExit(exit_code)


# =============================================================================
# MAIN MENU - v9.0 World-Class
# =============================================================================
def main_menu():
    """
    v9.0 Enhanced: Integration main menu with optional bypass.
    
    Respects config.SM_INT_MAIN_MENU:
      - True  => show menu as normal
      - False => bypass menu, announce via TTS, auto-launch GUI, then shut down on close
    
    The two synthesized voice lines should run in both modes.
    """
    # Normalize flag to boolean
    try:
        flag_raw = getattr(config, "SM_INT_MAIN_MENU", "True")
        flag_str = str(flag_raw).strip().lower()
        show_menu = flag_str in ("true", "1", "yes", "y", "on")
    except Exception:
        show_menu = True

    while not terminate_flag.is_set():
        if show_menu:
            # =================================================================
            # STANDARD INTERACTIVE MENU PATH
            # =================================================================
            print("\n" + "═" * 78)
            print("  SARAHMEMORY v9.0 - INTEGRATION MENU")
            print("═" * 78)
            
            try:
                synthesize_voice("...,Main Menu,....")
            except Exception:
                logger.debug("[v9.0][TTS] Main Menu prompt failed silently.")
            
            print("\n  1. Launch Main AI-Bot Text/Voice GUI")
            print("  2. Safe Shutdown and Exit")
            print("\n" + "═" * 78)

            choice = input("\n[v9.0] Enter your choice (1-2): ").strip()
            
            if choice == "1":
                try:
                    synthesize_voice("Now Loading GUI interface, Please Wait")
                except Exception:
                    logger.debug("[v9.0][TTS] Loading GUI prompt failed silently.")
                
                print("\n[v9.0] Launching Chat GUI...")

                # Start post-API runtime services before entering GUI mainloop
                try:
                    _start_runtime_services_once()
                except Exception:
                    pass
                
                try:
                    if run_gui is None:
                        print("\n[v9.0] GUI unavailable; install PyQt5/PyQtWebEngine or use web/headless mode.")
                        logger.error(f"[v9.0] GUI unavailable: {_GUI_IMPORT_ERROR}")
                    else:
                        run_gui()
                except Exception as e:
                    logger.error(f"[v9.0] GUI exited with error: {e}")
                finally:
                    logger.info("[v9.0] Returning to integration menu.")
            
            elif choice == "2":
                logger.info("[v9.0] Initiating safe shutdown and exit.")
                shutdown_sequence()
            
            else:
                try:
                    synthesize_voice("Invalid Choice., try again")
                except Exception:
                    logger.debug("[v9.0][TTS] Invalid choice prompt failed silently.")
                
                print("\n[v9.0] ✗ Invalid choice. Please select a valid option (1-2).")
        
        else:
            # =================================================================
            # BYPASS MENU PATH
            # =================================================================
            try:
                synthesize_voice("Now Loading GUI interface, Please Wait")
            except Exception:
                logger.debug("[v9.0][TTS] Loading GUI prompt (bypass mode) failed silently.")
            
            print("\n[v9.0] Launching Chat GUI (bypass mode)...")

            # Start post-API runtime services before entering GUI mainloop
            try:
                _start_runtime_services_once()
            except Exception:
                pass
            
            try:
                if run_gui is None:
                    print("\n[v9.0] GUI unavailable; install PyQt5/PyQtWebEngine or use web/headless mode.")
                    logger.error(f"[v9.0] GUI unavailable: {_GUI_IMPORT_ERROR}")
                else:
                    run_gui()
            except Exception as e:
                logger.error(f"[v9.0] GUI exited with error: {e}")
            finally:
                logger.info("[v9.0] GUI closed (bypass mode). Proceeding to shutdown.")
            
            # In bypass mode, shut down immediately after GUI closes
            logger.info("[v9.0] Initiating safe shutdown (bypass mode).")
            shutdown_sequence()
            break


# =============================================================================
# BACKWARD-COMPATIBILITY WRAPPER
# =============================================================================
def integration_menu():
    """
    v9.0: Backward-compatible wrapper. Delegates to main_menu().
    """
    return main_menu()


# =============================================================================
# ASYNC SELF-CHECK
# =============================================================================
async def run_self_check_async():
    """
    v9.0: Asynchronous self-check runner.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_self_check)


# =============================================================================
# API WAIT HELPER
# =============================================================================
def _sm_wait_for_api_ready(url="http://127.0.0.1:8765/health", retries=30, delay=0.5):
    """
    v9.0: Wait for the API server to be ready.
    
    Args:
        url: Health check endpoint
        retries: Maximum number of retries
        delay: Delay between retries in seconds
    
    Returns:
        bool: True if API is ready, False otherwise
    """
    try:
        import time
        import requests
        
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=0.5)
                if r.ok:
                    try:
                        logger_inst = globals().get("logger", None)
                        if logger_inst:
                            logger_inst.info("[v9.0][BOOT] Local API server is ready.")
                    except Exception:
                        pass
                    return True
            except Exception:
                time.sleep(delay)
        
        try:
            logger_inst = globals().get("logger", None)
            if logger_inst:
                logger_inst.warning("[v9.0][TIMEOUT] Local API server did not respond in time.")
        except Exception:
            pass
    
    except Exception:
        pass
    
    return False


# =============================================================================
# DATABASE SCHEMA VALIDATION
# =============================================================================
def _ensure_response_table(db_path=None):
    """
    v9.0: Ensure the response table exists in the database.
    """
    try:
        import sqlite3
        import logging
        
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config:
                pass
        
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            datasets_dir = getattr(config, "DATASETS_DIR", 
                                 os.path.join(base, "data", "memory", "datasets"))
            db_path = os.path.join(datasets_dir, "system_logs.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        con.commit()
        con.close()
        
        logging.debug("[v9.0][DB] Ensured table `response` in %s", db_path)
    
    except Exception as e:
        try:
            import logging
            logging.warning("[v9.0][DB] Ensure `response` failed: %s", e)
        except Exception:
            pass


# Response table schema creation is available on demand. Avoid import-time DB writes by default.
if _cfg_bool("INTEGRATION_ENSURE_RESPONSE_TABLE_ON_IMPORT", False):
    try:
        _ensure_response_table()
    except Exception:
        pass


# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================
if __name__ == "__main__":
    logger.info("[v9.0] Starting SarahMemory AI Bot.")

    # In standalone mode, try to start runtime services once the local API is reachable.
    try:
        _start_runtime_services_once()
    except Exception:
        pass

    # Optional dataset sync (network-dependent); disabled by default for anti-thrash boot.
    if _cfg_bool("INTEGRATION_SYNC_DATASETS_ON_STANDALONE_START", False):
        try:
            sync_dataset_bidirectional()
        except Exception:
            pass

    main_menu()

# ====================================================================
# END OF SarahMemoryIntegration.py v9.0.0
# ====================================================================
