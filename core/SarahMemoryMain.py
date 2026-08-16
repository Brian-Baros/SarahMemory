"""--==The SarahMemory Project==--
File: SarahMemoryMain.py
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
SarahMemory v9.0 - The First True AI Operating System (AiOS)
World-Class Bootup Sequence with Full Media Integration
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "main_entrypoint"
# CATEGORY = "boot_and_runtime_start"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_filesystem_network"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "main"
# FAMILY = "boot_sequence"
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
# NOTES = "Primary boot entrypoint launched by python SarahMemoryMain.py. Writes main heartbeat, starts services, gates autonomous lab-mode services, and orchestrates startup."
# --- SARAHMETA END ---

# =============================================================================
# CRITICAL IMPORTS - Database Functions
# =============================================================================
try:
    from SarahMemoryDatabase import ask_index_prompt
except Exception:
    ask_index_prompt = None

try:
    from SarahMemoryDatabase import embed_and_store_dataset_sentences
except Exception:
    embed_and_store_dataset_sentences = None

try:
    from SarahMemoryDatabase import vector_search
except Exception:
    vector_search = None

try:
    from SarahMemoryDatabase import vector_search_qa_cache
except Exception:
    vector_search_qa_cache = None

# =============================================================================
# ENVIRONMENT AND CONFIGURATION
# =============================================================================
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception as e:
    print(f"[WARN] python-dotenv unavailable or failed, .env not loaded: {e}")

import os
import atexit
import logging
from logging.handlers import RotatingFileHandler
import datetime
import sys
import subprocess
import time
import json
import warnings
import importlib.util
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
import platform
import signal
import threading
import socket
import sqlite3
import uuid
import SarahMemoryGlobals as config

# =============================================================================
# [v9.0] MAIN PROCESS HEARTBEAT / PID MARKER
# -----------------------------------------------------------------------------
# The local WebUI checks /api/health -> main_running by reading DATA_DIR/sarahmemory.pid.
# When SarahMemoryMain is launched directly (python SarahMemoryMain.py), we must write
# our PID so app.py can detect that the full desktop stack is alive.
# Also refresh server_state.json with a lightweight heartbeat so the file timestamp moves.
# =============================================================================
try:
    _data_dir = getattr(config, "DATA_DIR", None) or os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data")
    os.makedirs(_data_dir, exist_ok=True)

    _pid = int(os.getpid())
    _pid_file = os.path.join(_data_dir, "sarahmemory.pid")
    _pid_tmp = _pid_file + f".{_pid}.tmp"
    with open(_pid_tmp, "w", encoding="utf-8") as _f:
        _f.write(str(_pid))
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(_pid_tmp, _pid_file)

    _state_file = str(getattr(config, "SERVER_STATE_PATH", os.path.join(getattr(config, "SETTINGS_DIR", os.path.join(_data_dir, "settings")), "server_state.json")))
    os.makedirs(os.path.dirname(_state_file), exist_ok=True)
    _state: dict = {}
    try:
        if os.path.exists(_state_file):
            with open(_state_file, "r", encoding="utf-8") as _f:
                try:
                    _state = json.load(_f)
                except Exception:
                    _state = {}
            if not isinstance(_state, dict):
                _state = {}
    except Exception:
        _state = {}

    # Dual-schema update:
    # - Legacy/internal keys (UPPERCASE) used by some desktop modules
    # - API/UI keys (lowercase) expected by /api/health + diagnostics tooling
    _now = float(time.time())
    _state.update({
        "ok": True,
        "boot_instance_id": f"main-{_pid}-{int(_now * 1000)}",
        "ts": _now,
        "notes": _state.get("notes") if isinstance(_state.get("notes"), list) else [],
        "source": "SarahMemoryMain",
        "main_running": True,
        "main_pid": _pid,
        "main_last_seen_ts": _now,

        "MAIN_RUNNING": True,
        "MAIN_PID": _pid,
        "MAIN_LAST_SEEN_TS": _now,
    })

    _tmp = _state_file + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(_state, _f, indent=2, sort_keys=True)
        _f.flush()
        os.fsync(_f.fileno())
    os.replace(_tmp, _state_file)
except Exception:
    # Never block boot if filesystem permissions are weird.
    pass


# =============================================================================
# CROSS-PLATFORM COMPATIBILITY
# =============================================================================
# Safe optional imports for cross-platform compatibility
if platform.system() == "Windows":
    try:
        import pyautogui
        import pygetwindow
        import pyscreeze
        import mouseinfo
    except Exception as e:
        print(f"[WARN] Windows UI helpers unavailable: {e}")
else:
    # On Linux / PythonAnywhere, skip desktop UI modules
    pyautogui = None
    pygetwindow = None
    pyscreeze = None
    mouseinfo = None

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

# =============================================================================
# LOGGING CONFIGURATION - v9.0 Enhanced
# =============================================================================
log_filename = os.path.join(config.LOGS_DIR, "System.log")

# Centralized logging: write bounded INFO+ to System.log, only show ERROR+ on console.
# Runtime optimization: avoid unbounded System.log growth on the C: NVMe drive.
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)

try:
    _debug_mode = bool(getattr(config, "DEBUG_MODE", False))
except Exception:
    _debug_mode = False

root.setLevel(logging.DEBUG if _debug_mode else logging.INFO)
os.makedirs(config.LOGS_DIR, exist_ok=True)

try:
    _max_log_bytes = int(getattr(config, "SM_SYSTEM_LOG_MAX_BYTES", int(os.getenv("SARAH_SYSTEM_LOG_MAX_BYTES", "5242880"))) or 5242880)
except Exception:
    _max_log_bytes = 5242880
try:
    _backup_count = int(getattr(config, "SM_SYSTEM_LOG_BACKUP_COUNT", int(os.getenv("SARAH_SYSTEM_LOG_BACKUP_COUNT", "5"))) or 5)
except Exception:
    _backup_count = 5

file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=max(262144, _max_log_bytes),
    backupCount=max(1, _backup_count),
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG if _debug_mode else logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - v9.0 - %(levelname)s - %(name)s - %(message)s")
)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

root.addHandler(file_handler)
root.addHandler(console_handler)
logger = logging.getLogger("SarahMemoryMain")

# =============================================================================
# PROCESS LIFECYCLE / CLEAN SHUTDOWN CONTROL - v9.0
# -----------------------------------------------------------------------------
# SarahMemoryMain owns the local API child process and runtime PID/state files.
# Integration/GUI shutdown calls back into these helpers so a closed GUI does not
# leave python.exe / app.py running and does not require pytaskkill.bat.
# =============================================================================
_LOCAL_API_PROCESS = None
_LOCAL_API_LOG_HANDLE = None
_LOCAL_API_OWNED_PIDS = set()
_RUNTIME_INSTANCE_ID = uuid.uuid4().hex
_MAIN_CLEANUP_STARTED = False
_MAIN_CLEANUP_LOCK = threading.RLock()
_MAIN_SHUTDOWN_EVENT = threading.Event()
_SHUTDOWN_WATCHER_THREAD = None


def _sm_boot_flag(name: str, default: bool = False) -> bool:
    """Read a boot/runtime flag from SarahMemoryGlobals or environment."""
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


def _sm_data_dir() -> str:
    try:
        data_dir = getattr(config, "DATA_DIR", None) or os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data")
    except Exception:
        data_dir = os.path.join(os.getcwd(), "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception:
        pass
    return data_dir


def _sm_runtime_state_path() -> str:
    try:
        return str(getattr(config, "SERVER_STATE_PATH"))
    except Exception:
        return os.path.join(getattr(config, "SETTINGS_DIR", os.path.join(_sm_data_dir(), "settings")), "server_state.json")


def _sm_main_pid_path() -> str:
    return os.path.join(_sm_data_dir(), "sarahmemory.pid")


def _sm_api_pid_path() -> str:
    return os.path.join(_sm_data_dir(), "local_api.pid")


def _sm_atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(str(text))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sm_read_pid(path: str) -> int:
    try:
        raw = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
        return int(raw) if raw.isdigit() else 0
    except Exception:
        return 0


def _sm_pid_alive(pid: int) -> bool:
    if int(pid or 0) <= 0:
        return False
    try:
        if platform.system() == "Windows":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False


def _sm_process_command_line(pid: int) -> str:
    try:
        import psutil  # type: ignore
        return " ".join(psutil.Process(int(pid)).cmdline())
    except Exception:
        pass
    if platform.system() != "Windows":
        try:
            return open(f"/proc/{int(pid)}/cmdline", "rb").read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            return ""
    try:
        completed = subprocess.run(
            ["wmic", "process", "where", f"processid={int(pid)}", "get", "CommandLine", "/value"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        return completed.stdout or ""
    except Exception:
        return ""


def _sm_pid_is_sarah_api(pid: int, api_script: str = "") -> bool:
    if not _sm_pid_alive(pid):
        return False
    command_line = _sm_process_command_line(pid).lower().replace("\\", "/")
    if not command_line:
        return False
    expected = str(api_script or "api/server/app.py").lower().replace("\\", "/")
    return (expected in command_line) or ("api/server/app.py" in command_line)


def _sm_probe_api_health(host: str, port: int, timeout: float = 1.25) -> tuple[bool, dict]:
    if requests is None:
        return False, {"error": "requests_unavailable"}
    try:
        response = requests.get(f"http://{host}:{int(port)}/api/health", timeout=max(0.25, float(timeout)))
        data = response.json() if response.headers.get("content-type", "").lower().startswith("application/json") else {}
        if response.status_code != 200 or not isinstance(data, dict):
            return False, {"status_code": response.status_code}
        is_sarah = bool(data.get("running")) and str(data.get("version") or "").startswith("9")
        return is_sarah, data
    except Exception as exc:
        return False, {"error": str(exc)}


def _sm_port_accepting(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.1, float(timeout))):
            return True
    except Exception:
        return False


def _sm_checkpoint_runtime_databases(mode: str = "PASSIVE") -> dict:
    requested = str(mode or "PASSIVE").strip().upper()
    if requested not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
        requested = "PASSIVE"
    dataset_dir = str(getattr(config, "DATASETS_DIR", os.path.join(_sm_data_dir(), "memory", "datasets")))
    names = (
        "context_history.db", "neuron_axis.db", "cognitive_compass.db",
        "functions.db", "system_logs.db", "user_profile.db", "ai_learning.db",
    )
    limit = max(1, min(int(getattr(config, "DB_SHUTDOWN_CHECKPOINT_LIMIT", 16) or 16), len(names)))
    results = {}
    for name in names[:limit]:
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            con = sqlite3.connect(path, timeout=2.0)
            try:
                con.execute("PRAGMA busy_timeout=2000")
                row = con.execute(f"PRAGMA wal_checkpoint({requested})").fetchone()
                results[name] = {"ok": True, "result": list(row or [])}
            finally:
                con.close()
        except Exception as exc:
            results[name] = {"ok": False, "error": str(exc)}
    return {"ok": all(item.get("ok") for item in results.values()) if results else True, "mode": requested, "databases": results}


def _sm_write_runtime_state(*, main_running: bool, api_running: bool = False, reason: str = "") -> None:
    """Best-effort persisted lifecycle state for WebUI/API health checks."""
    try:
        state_file = _sm_runtime_state_path()
        state = {}
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}

        now_ts = float(time.time())
        notes = state.get("notes") if isinstance(state.get("notes"), list) else []
        if reason:
            notes = (notes + [f"{time.strftime('%Y-%m-%d %H:%M:%S')} shutdown:{reason}"])[-20:]

        api_pid = None
        try:
            proc = globals().get("_LOCAL_API_PROCESS")
            if proc is not None and getattr(proc, "poll", lambda: 1)() is None:
                api_pid = int(getattr(proc, "pid", 0) or 0) or None
        except Exception:
            api_pid = None
        if api_pid is None and api_running:
            try:
                api_pid = int(os.environ.get("SARAHMEMORY_LOCAL_API_PID") or 0) or None
            except Exception:
                api_pid = None

        state.update({
            "ok": True,
            "ts": now_ts,
            "source": "SarahMemoryMain",
            "runtime_instance_id": _RUNTIME_INSTANCE_ID,
            "notes": notes,
            "main_running": bool(main_running),
            "main_pid": int(os.getpid()) if main_running else None,
            "main_last_seen_ts": now_ts,
            "api_running": bool(api_running),
            "api_pid": api_pid if api_running else None,
            "api_last_seen_ts": now_ts if api_running else state.get("api_last_seen_ts"),

            "MAIN_RUNNING": bool(main_running),
            "MAIN_PID": int(os.getpid()) if main_running else None,
            "MAIN_LAST_SEEN_TS": now_ts,
            "API_RUNNING": bool(api_running),
            "API_PID": api_pid if api_running else None,
        })

        _sm_atomic_write_text(state_file, json.dumps(state, indent=2, sort_keys=True))
    except Exception:
        pass


def _sm_remove_runtime_pid_files() -> None:
    main_path = _sm_main_pid_path()
    api_path = _sm_api_pid_path()
    try:
        if _sm_read_pid(main_path) in {0, os.getpid()}:
            os.remove(main_path) if os.path.exists(main_path) else None
    except Exception:
        pass
    try:
        api_pid = _sm_read_pid(api_path)
        if api_pid == 0 or api_pid in _LOCAL_API_OWNED_PIDS or not _sm_pid_alive(api_pid):
            os.remove(api_path) if os.path.exists(api_path) else None
    except Exception:
        pass


def _sm_windows_kill_pid(pid: int) -> None:
    try:
        if pid <= 0 or pid == os.getpid():
            return
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
    except Exception:
        pass


def _sm_posix_kill_pid(pid: int) -> None:
    try:
        if pid <= 0 or pid == os.getpid():
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass


def stop_local_api_server(timeout: float = 5.0) -> bool:
    """Stop only API processes owned or explicitly adopted by this runtime."""
    global _LOCAL_API_PROCESS, _LOCAL_API_LOG_HANDLE
    stopped_any = False
    proc = _LOCAL_API_PROCESS
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=max(0.5, float(timeout)))
                stopped_any = True
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=2.0)
                    stopped_any = True
            except Exception:
                pass

    api_script = os.path.join(str(getattr(config, "API_DIR", "")), "server", "app.py")
    for pid in sorted(set(_LOCAL_API_OWNED_PIDS)):
        if pid <= 0 or pid == os.getpid() or not _sm_pid_alive(pid):
            continue
        if proc is not None and int(getattr(proc, "pid", 0) or 0) == pid:
            continue
        if not _sm_pid_is_sarah_api(pid, api_script):
            logger.warning("[SHUTDOWN] Refused to terminate unverified PID %s from local_api.pid", pid)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + max(0.5, float(timeout))
            while _sm_pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _sm_pid_alive(pid) and _sm_boot_flag("SARAH_ALLOW_HARD_PROCESS_KILL_FALLBACK", False):
                if platform.system() == "Windows":
                    _sm_windows_kill_pid(pid)
                else:
                    _sm_posix_kill_pid(pid)
            stopped_any = stopped_any or not _sm_pid_alive(pid)
        except Exception:
            pass

    try:
        if _LOCAL_API_LOG_HANDLE not in (None, subprocess.DEVNULL):
            _LOCAL_API_LOG_HANDLE.flush()
            _LOCAL_API_LOG_HANDLE.close()
    except Exception:
        pass
    _LOCAL_API_LOG_HANDLE = None
    _LOCAL_API_PROCESS = None
    _LOCAL_API_OWNED_PIDS.clear()
    os.environ.pop("SARAHMEMORY_LOCAL_API_PID", None)
    os.environ.pop("SARAHMEMORY_API_PID", None)
    return stopped_any

def main_process_cleanup(reason: str = "shutdown") -> None:
    """Idempotent process cleanup for normal close, Ctrl+C, exceptions, and atexit."""
    global _MAIN_CLEANUP_STARTED

    try:
        with _MAIN_CLEANUP_LOCK:
            if _MAIN_CLEANUP_STARTED:
                return
            _MAIN_CLEANUP_STARTED = True
    except Exception:
        if _MAIN_CLEANUP_STARTED:
            return
        _MAIN_CLEANUP_STARTED = True

    _MAIN_SHUTDOWN_EVENT.set()
    try:
        logger.info("[v9.0][SHUTDOWN] Main cleanup started: %s", reason)
    except Exception:
        pass

    try:
        import SarahMemoryVoice as _Voice  # type: ignore
        if hasattr(_Voice, "shutdown_tts"):
            _Voice.shutdown_tts()
    except Exception:
        pass

    try:
        import SarahMemoryInitialization as _Initialization  # type: ignore
        if hasattr(_Initialization, "safe_shutdown"):
            _Initialization.safe_shutdown()
    except Exception:
        pass

    # COGNITIVE_LIVING_LOOP_STOP_ON_MAIN_CLEANUP
    try:
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        if hasattr(_CogServices, "stop_cognitive_living_loop"):
            _CogServices.stop_cognitive_living_loop(reason=f"main_cleanup:{reason}")
    except Exception:
        pass

    try:
        import SarahMemoryAiFunctions as _AiFunctions  # type: ignore
        if hasattr(_AiFunctions, "shutdown_advanced_agent"):
            _AiFunctions.shutdown_advanced_agent(timeout=2.5)
    except Exception:
        pass

    try:
        import SarahMemoryNeuron as _Neuron  # type: ignore
        if hasattr(_Neuron, "stop_neuron_background"):
            _Neuron.stop_neuron_background()
    except Exception:
        pass

    try:
        _sm_checkpoint_runtime_databases("PASSIVE")
    except Exception:
        pass

    try:
        stop_local_api_server(timeout=4.0)
    except Exception:
        pass

    try:
        _sm_write_runtime_state(main_running=False, api_running=False, reason=reason)
    except Exception:
        pass

    try:
        _sm_remove_runtime_pid_files()
    except Exception:
        pass

    try:
        logging.shutdown()
    except Exception:
        pass


try:
    atexit.register(main_process_cleanup, reason="atexit")
except Exception:
    pass


def _main_signal_handler(signum, _frame) -> None:
    try:
        main_process_cleanup(reason=f"signal:{signum}")
    finally:
        raise SystemExit(0)


def _shutdown_request_watcher() -> None:
    """Observe the API/UI shutdown request without drive polling pressure."""
    poll_seconds = max(0.5, min(5.0, float(getattr(config, "SHUTDOWN_STATE_POLL_SECONDS", 1.5) or 1.5)))
    while not _MAIN_SHUTDOWN_EVENT.wait(poll_seconds):
        try:
            path = _sm_runtime_state_path()
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                state = json.load(fh)
            if isinstance(state, dict) and bool(state.get("shutdown_requested")):
                main_process_cleanup(reason=str(state.get("shutdown_reason") or "api_shutdown_request"))
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception:
                    pass
                return
        except Exception:
            continue


def _start_shutdown_watcher() -> None:
    global _SHUTDOWN_WATCHER_THREAD
    if _SHUTDOWN_WATCHER_THREAD is not None and _SHUTDOWN_WATCHER_THREAD.is_alive():
        return
    _SHUTDOWN_WATCHER_THREAD = threading.Thread(target=_shutdown_request_watcher, name="SM_ShutdownWatcher", daemon=True)
    _SHUTDOWN_WATCHER_THREAD.start()


try:
    signal.signal(signal.SIGINT, _main_signal_handler)
    signal.signal(signal.SIGTERM, _main_signal_handler)
except Exception:
    pass
try:
    _start_shutdown_watcher()
except Exception:
    pass


# =============================================================================
# OPTIONAL AUTONOMOUS SERVICES - v9.0 governed startup overlay
# -----------------------------------------------------------------------------
# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# NEOSKYMATRIX + DEVELOPERSMODE remain developer/lab signals, but they must not
# silently start SelfAware, Evolution, or other autonomous background loops during
# ordinary local boot. This keeps the system balanced, auditable, local-first, and
# non-hijackable. Explicit startup flags are required in addition to the older
# developer flags. This preserves existing governance while adding a necessary
# preflight leash.
# =============================================================================
try:
    # Synapses bootstrap may ensure model directories only. It is filesystem-local
    # and does not authorize autonomous reasoning/action loops.
    import SarahMemorySynapes as _SYN  # type: ignore
    if hasattr(_SYN, "ensure_sarahmemory_model_dirs"):
        _SYN.ensure_sarahmemory_model_dirs()  # type: ignore
        logger.info("[v9.0] Synapses bootstrap complete (model dirs ensured).")
except Exception as _e:
    logger.debug(f"[v9.0] Synapses bootstrap skipped/failed: {type(_e).__name__}: {_e}")

try:
    _neosky = bool(getattr(config, "NEOSKYMATRIX", False)) or str(os.getenv("NEOSKYMATRIX", "")).strip().lower() in ("1","true","yes","on","enabled")
    _dev = bool(getattr(config, "DEVELOPERSMODE", False)) or str(os.getenv("DEVELOPERSMODE", "")).strip().lower() in ("1","true","yes","on","enabled")
    _autonomy_master = bool(getattr(config, "SARAHMEMORY_AUTONOMOUS_STARTUP_ENABLED", False))
    _selfaware_auto = bool(getattr(config, "SARAHMEMORY_SELFAWARE_AUTOSTART_ENABLED", False))
    _evolution_auto = bool(getattr(config, "SARAHMEMORY_EVOLUTION_AUTOSTART_ENABLED", False))

    if not (_neosky and _dev and _autonomy_master):
        logger.info("[v9.0][GOV] Autonomous startup loops held in SAFE STANDBY. NEOSKYMATRIX/DEVELOPERSMODE do not self-authorize boot execution.")
    else:
        import threading

        if _selfaware_auto:
            try:
                import SarahMemorySelfAware as _SMA  # type: ignore
                if hasattr(_SMA, "run_autonomous_loop"):
                    _t = threading.Thread(target=_SMA.run_autonomous_loop, name="SM_SelfAware", daemon=True)
                    _t.start()
                    logger.warning("[v9.0][GOV] SelfAware autonomous loop started after explicit startup preflight flags.")
            except Exception as _e:
                logger.exception(f"[v9.0] SelfAware start failed: {type(_e).__name__}: {_e}")
        else:
            logger.info("[v9.0][GOV] SelfAware autostart disabled; available on explicit governed request only.")

        if _evolution_auto:
            try:
                import SarahMemoryEvolution as _EVO  # type: ignore
                if hasattr(_EVO, "evolve_once"):
                    _t2 = threading.Thread(
                        target=lambda: _EVO.evolve_once(autonomous=True, weekly_gate=True),  # type: ignore
                        name="SM_Evolution",
                        daemon=True
                    )
                    _t2.start()
                    logger.warning("[v9.0][GOV] Evolution cycle scheduled after explicit startup preflight flags.")
            except Exception as _e:
                logger.exception(f"[v9.0] Evolution start failed: {type(_e).__name__}: {_e}")
        else:
            logger.info("[v9.0][GOV] Evolution autostart disabled; available on explicit governed request only.")
except Exception as _e:
    logger.debug(f"[v9.0] Autonomous services gate evaluation skipped: {type(_e).__name__}: {_e}")


# =============================================================================
# API SERVER MANAGEMENT - v9.0 Enhanced
# =============================================================================
def start_local_api_server() -> bool:
    """Launch or safely adopt the single local SarahMemory API instance."""
    global _LOCAL_API_PROCESS, _LOCAL_API_LOG_HANDLE
    try:
        if _LOCAL_API_PROCESS is not None and _LOCAL_API_PROCESS.poll() is None:
            return True
        base_dir = str(getattr(config, "BASE_DIR", os.getcwd()))
        api_dir = str(getattr(config, "API_DIR", os.path.join(base_dir, "api")))
        candidates = (
            os.path.join(api_dir, "server", "app.py"),
            os.path.join(base_dir, "api", "server", "app.py"),
            os.path.abspath(os.path.join("api", "server", "app.py")),
        )
        api_server_script = next((path for path in candidates if os.path.isfile(path)), "")
        if not api_server_script:
            logger.warning("[BOOT] API server script not found. Skipping API server startup.")
            return False

        host = str(getattr(config, "SARAHMEMORY_LOCAL_API_BIND_HOST", "127.0.0.1") or "127.0.0.1")
        probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        port = int(getattr(config, "DEFAULT_PORT", 8000) or 8000)
        stale_pid = _sm_read_pid(_sm_api_pid_path())
        if stale_pid and not _sm_pid_alive(stale_pid):
            try:
                os.remove(_sm_api_pid_path())
            except Exception:
                pass
            stale_pid = 0

        healthy, health = _sm_probe_api_health(probe_host, port, timeout=1.0)
        if healthy:
            if stale_pid and _sm_pid_is_sarah_api(stale_pid, api_server_script):
                _LOCAL_API_OWNED_PIDS.add(stale_pid)
                os.environ["SARAHMEMORY_LOCAL_API_PID"] = str(stale_pid)
                os.environ["SARAHMEMORY_API_PID"] = str(stale_pid)
                _sm_write_runtime_state(main_running=True, api_running=True, reason="api_adopted")
                logger.info("[BOOT] Adopted healthy SarahMemory API pid=%s on port %s.", stale_pid, port)
                return True
            logger.info("[BOOT] Healthy SarahMemory API already serves port %s; no duplicate process launched.", port)
            _sm_write_runtime_state(main_running=True, api_running=True, reason="api_already_healthy")
            return True

        if _sm_port_accepting(probe_host, port):
            logger.error("[BOOT] Port %s is occupied but does not identify as SarahMemory API. Startup denied.", port)
            return False
        if stale_pid and _sm_pid_alive(stale_pid):
            if not _sm_pid_is_sarah_api(stale_pid, api_server_script):
                logger.error("[BOOT] local_api.pid points to a live non-SarahMemory process (%s). Startup denied.", stale_pid)
                return False
            logger.warning("[BOOT] Prior SarahMemory API pid=%s is alive but unhealthy; leaving it untouched for explicit recovery.", stale_pid)
            return False

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        child_env = os.environ.copy()
        child_env["PORT"] = str(port)
        child_env["SARAHMEMORY_API_HOST"] = host
        child_env["SARAH_LOCAL_ONLY_MODE"] = "true" if bool(getattr(config, "LOCAL_ONLY_MODE", True)) else "false"
        child_env["SARAHMEMORY_PARENT_PID"] = str(os.getpid())
        child_env["SARAHMEMORY_RUNTIME_INSTANCE_ID"] = _RUNTIME_INSTANCE_ID
        logs_dir = os.path.join(base_dir, "data", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        api_log_path = os.path.join(logs_dir, "local_api_server.log")
        try:
            _LOCAL_API_LOG_HANDLE = open(api_log_path, "a", encoding="utf-8", buffering=1)
            _LOCAL_API_LOG_HANDLE.write("\n--- SarahMemory local API boot attempt %s instance=%s ---\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), _RUNTIME_INSTANCE_ID))
        except Exception:
            _LOCAL_API_LOG_HANDLE = subprocess.DEVNULL

        proc = subprocess.Popen(
            [sys.executable, api_server_script],
            cwd=base_dir,
            stdout=_LOCAL_API_LOG_HANDLE,
            stderr=_LOCAL_API_LOG_HANDLE,
            creationflags=creationflags,
            start_new_session=(platform.system() != "Windows"),
            env=child_env,
        )
        _LOCAL_API_PROCESS = proc
        pid = int(getattr(proc, "pid", 0) or 0)
        if pid <= 0:
            raise RuntimeError("API child did not return a valid PID")
        _LOCAL_API_OWNED_PIDS.add(pid)
        os.environ["SARAHMEMORY_LOCAL_API_PID"] = str(pid)
        os.environ["SARAHMEMORY_API_PID"] = str(pid)
        _sm_atomic_write_text(_sm_api_pid_path(), str(pid))
        _sm_write_runtime_state(main_running=True, api_running=True, reason="api_started")
        logger.info("[BOOT][v9.0] Local API server launched (pid=%s port=%s).", pid, port)
        return True
    except Exception as exc:
        logger.error("[BOOT ERROR][v9.0] Failed to launch local API server: %s", exc)
        return False


def wait_for_api_server(timeout=30):
    """
    Enhanced v9.0: Check if the local API server is online before launching integration.
    Includes retry logic and better error handling.
    
    Args:
        timeout: Maximum seconds to wait for server response
    
    Returns:
        bool: True if server is ready, False otherwise
    """
    logger.info("[v9.0] Waiting for local API server to respond...")
    if requests is None:
        logger.warning("[v9.0] requests is unavailable; skipping local API readiness probe.")
        return False
    url_health = f"http://{config.DEFAULT_HOST}:{config.DEFAULT_PORT}/api/health"
    url_status = f"http://{config.DEFAULT_HOST}:{config.DEFAULT_PORT}/api/status"

    for attempt in range(timeout):
        try:
            for _url in (url_health, url_status):
                response = requests.get(_url, timeout=2)
                if response.status_code == 200:
                    # /api/health returns JSON; accept even if body parsing fails
                    logger.info(f"[READY][v9.0] Local API server is online via {_url} (attempt {attempt + 1}/{timeout}).")
                    return True
        except Exception:
            pass
        
        # Visual progress indicator
        if attempt < timeout - 1:
            print(f"[v9.0] API Server startup... {attempt + 1}/{timeout}", end='\r')
            time.sleep(1)
    
    logger.warning(f"[TIMEOUT][v9.0] Local API server did not respond within {timeout} seconds.")
    return False
# =============================================================================
# OPTIONAL: API server keepalive / status probes
# =============================================================================
def check_api_server_health(url="http://127.0.0.1:5000/api/health", timeout=1.25):
    if requests is None:
        return False, {"error": "requests_unavailable"}
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        return False, {"status_code": r.status_code}
    except Exception as e:
        return False, {"error": str(e)}

# =============================================================================
# v9.0 BOOTUP BANNER - World-Class Visual Identity
# =============================================================================
def display_v8_banner():
    """
    Display the SarahMemory v9.0 bootup banner with visual flair.
    Cross-platform compatible.
    """
    banner = """
───────────────────────────────────────────────────────────────────────────────
                             S A R A H M E M O R Y   A i O S
                    THE FIRST FULL AI-DRIVEN OPERATING SYSTEM
                                   Version 9.0.0
───────────────────────────────────────────────────────────────────────────────


    🌟 World-Class Features:
       • Self-Updating Intelligence        • Advanced Media Creation
       • Multi-Platform Support            • Voice & Sound Synthesis
       • Distributed Mesh Network          • Blockchain Integration
       • Autonomous Learning               • Professional Content Studio

    📡 Network Hubs:
       • www.sarahmemory.com    - E-Commerce & Distribution
       • api.sarahmemory.com    - Network Hub & AI Ranking
       • ai.sarahmemory.com     - Web/Mobile Interface

    © 2025-2026 Brian Lee Baros | SOFTDEV0 LLC
    ═══════════════════════════════════════════════════════════════════════════
"""
    print(banner)
    logger.info("[v9.0] SarahMemory AiOS v9.0.0 Bootup Initiated")

# =============================================================================
# v9.0 MAIN EXECUTION BLOCK
# =============================================================================
try:
    # Display v9.0 banner
    display_v8_banner()
    
    logger.info("[v9.0] Starting SarahMemory AI Bot Main Launcher...")
    logger.info(f"[v9.0] Platform: {platform.system()} {platform.release()}")
    logger.info(f"[v9.0] Python: {platform.python_version()}")
    logger.info(f"[v9.0] Run Mode: {config.RUN_MODE}")
    logger.info(f"[v9.0] Device Mode: {config.DEVICE_MODE}")

    # ==========================================================================
    # PHASE 1: EARLY UPDATER HOOK (v9.0 Enhanced)
    # ==========================================================================
    # Before importing heavy modules, run the updater to apply any minimal fixes.
    # If no internet connectivity or errors occur, the updater will skip silently
    # without blocking startup. This ensures the latest code updates are applied
    # when available.
    print("[v9.0][PHASE 1] Checking for system updates...")
    if _sm_boot_flag("BOOT_RUN_UPDATER_ON_STARTUP", False):
        try:
            from SarahMemoryUpdater import run_updater
            run_updater(invoked_by_main=True)
            logger.info("[v9.0][PHASE 1] Update check completed successfully")
        except Exception as e:
            # Never block boot if anything goes wrong here
            print(f"[v9.0][Updater] Skipped due to error: {e}")
            logger.warning(f"[v9.0][Updater] Skipped due to error: {e}")
    else:
        print("[v9.0][Updater] Skipped (BOOT_RUN_UPDATER_ON_STARTUP disabled for optimized boot).")
        logger.info("[v9.0][PHASE 1] Updater skipped by optimized runtime policy.")

    # ==========================================================================
    # PHASE 2: CORE MODULE INITIALIZATION
    # ==========================================================================
    print("[v9.0][PHASE 2] Initializing core modules...")
    import SarahMemoryInitialization as initialization
    import SarahMemoryCognitiveServices as cognitive
    import SarahMemoryIntegration as integration
    
    import SarahMemoryDiagnostics as diagnostics
    try:
        from SarahMemoryARILE import start_arile_runtime, arile_emit
        arile_status = start_arile_runtime(reason="main.phase2_core_module_initialization")
        logger.info(f"[v9.0][ARILE] Runtime status: {arile_status}")
        arile_emit(source="SarahMemoryMain", kind="runtime_start", failure_type="arile_phase2_start", severity=0.25, confidence=0.95, summary="ARILE interlock started during Phase 2 core module initialization.")
    except Exception as arile_err:
        logger.warning(f"[v9.0][ARILE] Runtime start skipped: {arile_err}")
    # optional safe warmup (no network, no execution)
    try:
        cognitive.ensure_response_table()   # optional legacy table
        cognitive._ensure_tables()          # governor event table
    except Exception:
        pass
    logger.info("[v9.0][PHASE 2] Core modules loaded successfully")

    # Unified boot environment capture: this is the single authoritative scan
    # for CPU/GPU/RAM/storage/network/driver readiness. Downstream boot phases,
    # Globals.hardware_score(), API endpoints, and chat answers reuse this same
    # persisted body map instead of probing hardware multiple times.
    try:
        initialization.capture_and_print_boot_environment_summary(
            force_refresh=_sm_boot_flag("BOOT_FORCE_ENV_REFRESH", False),
            detail=True,
            phase_context="phase2_core_module_initialization",
        )
    except Exception as env_err:
        logger.warning(f"[v9.0][PHASE 2][ENV] Unified environment capture failed: {env_err}")

    # ==========================================================================
    # PHASE 3: CONTEXT BUFFER INITIALIZATION (if enabled)
    # ==========================================================================
    if config.ENABLE_CONTEXT_BUFFER:
        print("[v9.0][PHASE 3] Initializing conversation context buffer...")
        import SarahMemoryAiFunctions as context
        logger.info(f"[v9.0][PHASE 3] Context buffer enabled with size: {config.CONTEXT_BUFFER_SIZE}")

    # ==========================================================================
    # PHASE 4: STARTUP INFORMATION & SYSTEM CHECKS
    # ==========================================================================
    print("[v9.0][PHASE 4] Running system diagnostics...")
    initialization.startup_info()  # Logs AI boot intro with v9.0 enhancements
    
    success = initialization.run_initial_checks()
    if not success:
        raise Exception("[v9.0] System initialization failed.")
    
    logger.info("[v9.0][PHASE 4] System diagnostics completed successfully")

    # COGNITIVE_LIVING_LOOP_BOOT_AUTOSTART
    # Start the bounded backend cognitive heartbeat after diagnostics have proven
    # the core runtime is stable. The loop is read-mostly and cannot self-authorize
    # physical/device action; emergency dispatch remains behind OperatorCore/MSDC.
    try:
        if bool(getattr(config, "SARAHMEMORY_LIVING_LOOP_AUTOSTART", True)):
            living_status = cognitive.autostart_cognitive_living_loop(reason="boot_phase4_autostart")
            logger.info(
                "[v9.0][PHASE 4] Cognitive Living Loop status: started=%s thread_alive=%s interval=%s",
                bool(((living_status or {}).get("state") or {}).get("started")),
                bool(((living_status or {}).get("state") or {}).get("thread_alive")),
                (((living_status or {}).get("state") or {}).get("interval_seconds")),
            )
        else:
            logger.info("[v9.0][PHASE 4] Cognitive Living Loop autostart disabled by config.")
    except Exception as living_err:
        logger.warning(f"[v9.0][PHASE 4] Cognitive Living Loop autostart skipped: {living_err}")

    # ==========================================================================
    # PHASE 5: SYNCHRONIZATION SEQUENCE
    # ==========================================================================
    print("[v9.0][PHASE 5] Running synchronization sequence...")
    initialization.run_sync_sequence()  # Optional sync with network hubs
    logger.info("[v9.0][PHASE 5] Synchronization completed")

    # ==========================================================================
    # PHASE 6: LOCAL API SERVER STARTUP
    # ==========================================================================
    print("[v9.0][PHASE 6] Starting local API server...")
    start_local_api_server()
    api_ready = wait_for_api_server(timeout=10)
    
    if api_ready:
        logger.info("[v9.0][PHASE 6] API server ready for requests")
    else:
        logger.warning("[v9.0][PHASE 6] API server may not be available - continuing anyway")

    # ==========================================================================
    # PHASE 7: MEDIA SUBSYSTEM INITIALIZATION (v9.0 NEW)
    # ==========================================================================
    print("[v9.0][PHASE 7] Probing media creation capabilities...")
    try:
        media_names = [
            ("SarahMemoryMusicGenerator", "MusicGenerator"),
            ("SarahMemoryLyricsToSong", "LyricsToSong"),
            ("SarahMemoryVideoEditorCore", "VideoEditor"),
            ("SarahMemoryCanvasStudio", "CanvasStudio"),
        ]
        available_media = []
        load_at_boot = not bool(getattr(config, "BOOT_MEDIA_CAPABILITY_PROBE_ONLY", True))
        for module_name, label in media_names:
            try:
                found = importlib.util.find_spec(module_name) is not None
            except Exception:
                found = False
            if found:
                available_media.append(label)
                if load_at_boot:
                    try:
                        __import__(module_name)
                    except Exception as exc:
                        logger.debug("[v9.0][PHASE 7] %s deferred after import failure: %s", label, exc)
        if available_media:
            print(f"[v9.0][PHASE 7] Media capabilities available on demand: {', '.join(available_media)}")
            logger.info("[v9.0][PHASE 7] Capability probe found %s media subsystems; boot imports=%s", len(available_media), load_at_boot)
        else:
            logger.info("[v9.0][PHASE 7] No optional media subsystem modules found")
    except Exception as e:
        logger.warning(f"[v9.0][PHASE 7] Media capability probe warning: {e}")

    # ==========================================================================
    # PHASE 8: LAUNCH INTEGRATION MENU
    # ==========================================================================
    print("\n[v9.0][PHASE 8] All systems ready. Launching SarahMemory AiOS v9.0...")
    logger.info("[v9.0][PHASE 8] Starting SarahMemory AI Bot Integration Menu")
    
    # Small delay for visual effect
    time.sleep(0.5)
    
    # Launch the main integration menu
    integration.integration_menu()

    # If the integration layer ever returns instead of exiting, still clean up.
    main_process_cleanup(reason="integration_menu_returned")

except KeyboardInterrupt:
    logger.info("[v9.0] User interrupted startup sequence (Ctrl+C)")
    print("\n[v9.0] Shutdown initiated by user.")
    main_process_cleanup(reason="keyboard_interrupt")
    sys.exit(0)

except Exception as e:
    try:
        main_process_cleanup(reason="critical_error")
    except Exception:
        pass
    logger.error(f"[v9.0] Critical error in main execution: {e}")
    print(f"\n[v9.0] An unexpected error occurred:")
    print(f"Error: {e}")
    print("\nPlease check the logs for more details:")
    print(f"Log file: {log_filename}")
    sys.exit(1)

# =============================================================================
# DATABASE SCHEMA VALIDATION - v9.0
# =============================================================================
def _ensure_response_table(db_path=None):
    """
    Ensure the response table exists in the database.
    v9.0 Enhanced with better error handling and logging.
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
            datasets_dir = getattr(config, "DATASETS_DIR", os.path.join(base, "data", "memory", "datasets"))
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

# Initialize response table
try:
    _ensure_response_table()
except Exception:
    pass

# ====================================================================
# END OF SarahMemoryMain.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryMain',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Unknown',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": [],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 40,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryMain.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryMain',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryMain', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

