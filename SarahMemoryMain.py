"""--==The SarahMemory Project==--
File: SarahMemoryMain.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-01-06
Time: 10:11:54
Author: © 2025,2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
SarahMemory v8.0 - The First True AI Operating System (AiOS)
World-Class Bootup Sequence with Full Media Integration
===============================================================================
"""

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
import logging
import datetime
import sys
import subprocess
import time
import json
import threading
import atexit
import requests
import platform
import SarahMemoryGlobals as config
import warnings
warnings.filterwarnings("error", category=SyntaxWarning)

# =============================================================================
# [v8.0] MAIN PROCESS HEARTBEAT / PID MARKER
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
    with open(_pid_file, "w", encoding="utf-8") as _f:
        _f.write(str(_pid))

    _state_file = os.path.join(_data_dir, "server_state.json")
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
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
log_filename = os.path.join(config.LOGS_DIR, "System.log")

# Centralized logging: write INFO+ to System.log, only show ERROR+ on console
root = logging.getLogger()
# Clear existing handlers to prevent duplicate logs
for h in list(root.handlers):
    root.removeHandler(h)
root.setLevel(logging.DEBUG)
os.makedirs(config.LOGS_DIR, exist_ok=True)

# File handler with enhanced formatting
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - v8.0 - %(levelname)s - %(name)s - %(message)s")
)

# Console handler - only errors
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

root.addHandler(file_handler)
root.addHandler(console_handler)
logger = logging.getLogger("SarahMemoryMain")

# -----------------------------------------------------------------------------
# Purpose:
# - Keep DATA_DIR/server_state.json "fresh" for the WebUI and /api/health consumers.
# - Mirror the MAIN_* fields written at boot (PID + last_seen) on a fixed cadence.
# - Never block boot; never crash the main loop if filesystem permissions are odd.
#
# Configuration (optional):
# - MAIN_HEARTBEAT_SECONDS (SarahMemoryGlobals.py or .env) default: 5 seconds
# =============================================================================
_MAIN_HEARTBEAT_SECONDS = 5
try:
    _MAIN_HEARTBEAT_SECONDS = int(getattr(config, "MAIN_HEARTBEAT_SECONDS", 5) or 5)
except Exception:
    _MAIN_HEARTBEAT_SECONDS = 5
if _MAIN_HEARTBEAT_SECONDS < 1:
    _MAIN_HEARTBEAT_SECONDS = 1

def _sm_write_main_state(_running: bool) -> None:
    """Best-effort: update data/server_state.json with MAIN_* + main_* compatibility keys."""
    try:
        data_dir = getattr(config, "DATA_DIR", None) or os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data")
        os.makedirs(data_dir, exist_ok=True)

        pid = int(os.getpid())
        now = float(time.time())

        state_file = os.path.join(data_dir, "server_state.json")
        state = {}
        try:
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    try:
                        state = json.load(f)
                    except Exception:
                        state = {}
        except Exception:
            state = {}

        if not isinstance(state, dict):
            state = {}

        # MAIN_* (legacy caps) + main_* (api state) keys
        state["MAIN_RUNNING"] = bool(_running)
        state["MAIN_PID"] = pid
        state["MAIN_LAST_SEEN_TS"] = now

        state["main_running"] = bool(_running)
        state["main_pid"] = pid
        state["main_last_seen_ts"] = now

        # If api isn't writing, at least keep a minimal truthy payload
        state.setdefault("ok", True)
        state.setdefault("running", True)
        state.setdefault("status", "ok")
        state.setdefault("notes", [])

        state["ts"] = now
        state.setdefault("source", "main_heartbeat")

        tmp = state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, state_file)

        # Keep pid file aligned (used by /api/health main_running checks)
        pid_file = os.path.join(data_dir, "sarahmemory.pid")
        try:
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(str(pid))
        except Exception:
            pass
    except Exception:
        # Never block boot or runtime on heartbeat failures.
        pass

def _sm_main_heartbeat_loop() -> None:
    """Daemon heartbeat loop."""
    # Initial mark (helps immediately after boot)
    _sm_write_main_state(True)
    while True:
        try:
            time.sleep(_MAIN_HEARTBEAT_SECONDS)
            _sm_write_main_state(True)
        except Exception:
            # Hard safety: never exit the loop.
            try:
                time.sleep(_MAIN_HEARTBEAT_SECONDS)
            except Exception:
                pass

# Start heartbeat thread ASAP (non-blocking)
try:
    _hb = threading.Thread(target=_sm_main_heartbeat_loop, name="SM_MainHeartbeat", daemon=True)
    _hb.start()
    logger.info(f"[v8.0] Main heartbeat started ({_MAIN_HEARTBEAT_SECONDS}s).")
except Exception as _e:
    logger.debug(f"[v8.0] Main heartbeat failed to start: {type(_e).__name__}: {_e}")

# On clean exit, mark MAIN_RUNNING false (best-effort)
try:
    atexit.register(lambda: _sm_write_main_state(False))
except Exception:
    pass
# =============================================================================
# OPTIONAL AUTONOMOUS SERVICES - v8.0 (Synapses / SelfAware / Evolution)
# -----------------------------------------------------------------------------
# Objective: keep SarahMemoryMain as the boot orchestrator while allowing
# controlled lab-mode autonomy when NEOSKYMATRIX + DEVELOPERSMODE are armed.
# - Never blocks boot (best-effort / non-fatal)
# - Keeps audit/logging centralized under ./data/logs
# =============================================================================
try:
    # 1) Synapses bootstrap (model directory contract)
    import SarahMemorySynapes as _SYN  # type: ignore
    if hasattr(_SYN, "ensure_sarahmemory_model_dirs"):
        _SYN.ensure_sarahmemory_model_dirs()  # type: ignore
        logger.info("[v8.0] Synapses bootstrap complete (model dirs ensured).")
except Exception as _e:
    logger.debug(f"[v8.0] Synapses bootstrap skipped/failed: {type(_e).__name__}: {_e}")

try:
    # 2) Gate autonomous services
    _neosky = bool(getattr(config, "NEOSKYMATRIX", False)) or str(os.getenv("NEOSKYMATRIX", "")).strip().lower() in ("1","true","yes","on","enabled")
    _dev = bool(getattr(config, "DEVELOPERSMODE", False)) or str(os.getenv("DEVELOPERSMODE", "")).strip().lower() in ("1","true","yes","on","enabled")

    if _neosky and _dev:
        import threading

        # SelfAware autonomous loop (runs forever, daemon thread)
        try:
            import SarahMemorySelfAware as _SMA  # type: ignore
            if hasattr(_SMA, "run_autonomous_loop"):
                _t = threading.Thread(target=_SMA.run_autonomous_loop, name="SM_SelfAware", daemon=True)
                _t.start()
                logger.warning("[v8.0] SelfAware ARMED (NEOSKYMATRIX+DEVELOPERSMODE) — autonomous loop started.")
        except Exception as _e:
            logger.exception(f"[v8.0] SelfAware start failed: {type(_e).__name__}: {_e}")

        # Evolution cycle (non-blocking one-shot; internal weekly gate controls execution)
        try:
            import SarahMemoryEvolution as _EVO  # type: ignore
            if hasattr(_EVO, "evolve_once"):
                _t2 = threading.Thread(
                    target=lambda: _EVO.evolve_once(autonomous=True, weekly_gate=True),  # type: ignore
                    name="SM_Evolution",
                    daemon=True
                )
                _t2.start()
                logger.warning("[v8.0] Evolution cycle scheduled (one-shot with weekly gate).")
        except Exception as _e:
            logger.exception(f"[v8.0] Evolution start failed: {type(_e).__name__}: {_e}")
except Exception as _e:
    logger.debug(f"[v8.0] Autonomous services gate evaluation skipped: {type(_e).__name__}: {_e}")


# =============================================================================
# API SERVER MANAGEMENT - v8.0 Enhanced
# =============================================================================
def start_local_api_server():
    """
    Launch the local API server with enhanced v8.0 features.
    Supports both Windows and Linux/headless environments.

    v8.0.0 Patch:
      - SarahMemory-local_api_server.py is deprecated/removed.
      - Always launch the unified Flask server at /api/server/app.py.
      - Uses BASE_DIR/API_DIR from SarahMemoryGlobals to build an absolute path.
      - Sets cwd to BASE_DIR so static/UI paths resolve consistently.
    """
    try:
        # Resolve absolute app.py path under ../api/server/app.py
        base_dir = getattr(config, "BASE_DIR", os.getcwd())
        api_dir = getattr(config, "API_DIR", os.path.join(base_dir, "api"))
        api_server_script = os.path.join(api_dir, "server", "app.py")

        if not os.path.exists(api_server_script):
            # Last-resort fallbacks (keep boot non-blocking)
            alt1 = os.path.join(base_dir, "api", "server", "app.py")
            alt2 = os.path.join("api", "server", "app.py")
            for cand in (alt1, alt2):
                if os.path.exists(cand):
                    api_server_script = cand
                    break

        if not os.path.exists(api_server_script):
            logger.warning("[BOOT] API server script not found (expected ../api/server/app.py). Skipping API server startup.")
            return

        # Launch API server as background process
        # v8.0 Hotfix: keep a handle/PID so we can reliably stop it on shutdown
        creationflags = 0
        if platform.system() == "Windows":
            # New process group so taskkill /T will terminate the entire tree
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            [sys.executable, api_server_script],
            cwd=base_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            start_new_session=(platform.system() != "Windows")
        )
        # Store PID for shutdown_sequence to terminate cleanly
        os.environ["SARAHMEMORY_LOCAL_API_PID"] = str(getattr(proc, "pid", ""))
        logger.info("[BOOT][v8.0] Local API server process launched successfully (pid=%s).", getattr(proc, "pid", "?"))
    except Exception as e:
        logger.error(f"[BOOT ERROR][v8.0] Failed to launch local API server: {e}")



def wait_for_api_server(timeout=10):
    """
    Enhanced v8.0: Check if the local API server is online before launching integration.
    Includes retry logic and better error handling.
    
    Args:
        timeout: Maximum seconds to wait for server response
    
    Returns:
        bool: True if server is ready, False otherwise
    """
    logger.info("[v8.0] Waiting for local API server to respond...")
    url = f"http://{config.DEFAULT_HOST}:{config.DEFAULT_PORT}/api/status"
    
    for attempt in range(timeout):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"[READY][v8.0] Local API server is online (attempt {attempt + 1}/{timeout}).")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Visual progress indicator
        if attempt < timeout - 1:
            print(f"[v8.0] API Server startup... {attempt + 1}/{timeout}", end='\r')
            time.sleep(1)
    
    logger.warning(f"[TIMEOUT][v8.0] Local API server did not respond within {timeout} seconds.")
    return False
# =============================================================================
# OPTIONAL: API server keepalive / status probes
# =============================================================================
def check_api_server_health(url="http://127.0.0.1:5000/api/health", timeout=1.25):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        return False, {"status_code": r.status_code}
    except Exception as e:
        return False, {"error": str(e)}

# =============================================================================
# v8.0 BOOTUP BANNER - World-Class Visual Identity
# =============================================================================
def display_v8_banner():
    """
    Display the SarahMemory v8.0 bootup banner with visual flair.
    Cross-platform compatible.
    """
    banner = """
───────────────────────────────────────────────────────────────────────────────
                             S A R A H M E M O R Y   A i O S
                    THE FIRST FULL AI-DRIVEN OPERATING SYSTEM
                                   Version 8.0.0
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

    © 2025 Brian Lee Baros | SOFTDEV0 LLC
    ═══════════════════════════════════════════════════════════════════════════
"""
    print(banner)
    logger.info("[v8.0] SarahMemory AiOS v8.0.0 Bootup Initiated")

# =============================================================================
# v8.0 MAIN EXECUTION BLOCK
# =============================================================================
try:
    # Display v8.0 banner
    display_v8_banner()
    
    logger.info("[v8.0] Starting SarahMemory AI Bot Main Launcher...")
    logger.info(f"[v8.0] Platform: {platform.system()} {platform.release()}")
    logger.info(f"[v8.0] Python: {platform.python_version()}")
    logger.info(f"[v8.0] Run Mode: {config.RUN_MODE}")
    logger.info(f"[v8.0] Device Mode: {config.DEVICE_MODE}")

    # ==========================================================================
    # PHASE 1: EARLY UPDATER HOOK (v8.0 Enhanced)
    # ==========================================================================
    # Before importing heavy modules, run the updater to apply any minimal fixes.
    # If no internet connectivity or errors occur, the updater will skip silently
    # without blocking startup. This ensures the latest code updates are applied
    # when available.
    print("[v8.0][PHASE 1] Checking for system updates...")
    try:
        from SarahMemoryUpdater import run_updater
        run_updater(invoked_by_main=True)
        logger.info("[v8.0][PHASE 1] Update check completed successfully")
    except Exception as e:
        # Never block boot if anything goes wrong here
        print(f"[v8.0][Updater] Skipped due to error: {e}")
        logger.warning(f"[v8.0][Updater] Skipped due to error: {e}")

    # ==========================================================================
    # PHASE 2: CORE MODULE INITIALIZATION
    # ==========================================================================
    print("[v8.0][PHASE 2] Initializing core modules...")
    import SarahMemoryInitialization as initialization
    import SarahMemoryCognitiveServices as cognitive
    import SarahMemoryIntegration as integration
    
    # optional safe warmup (no network, no execution)
    try:
        cognitive.ensure_response_table()   # optional legacy table
        cognitive._ensure_tables()          # governor event table
    except Exception:
        pass
    logger.info("[v8.0][PHASE 2] Core modules loaded successfully")

    # ==========================================================================
    # PHASE 3: CONTEXT BUFFER INITIALIZATION (if enabled)
    # ==========================================================================
    if config.ENABLE_CONTEXT_BUFFER:
        print("[v8.0][PHASE 3] Initializing conversation context buffer...")
        import SarahMemoryAiFunctions as context
        logger.info(f"[v8.0][PHASE 3] Context buffer enabled with size: {config.CONTEXT_BUFFER_SIZE}")

    # ==========================================================================
    # PHASE 4: STARTUP INFORMATION & SYSTEM CHECKS
    # ==========================================================================
    print("[v8.0][PHASE 4] Running system diagnostics...")
    initialization.startup_info()  # Logs AI boot intro with v8.0 enhancements
    
    success = initialization.run_initial_checks()
    if not success:
        raise Exception("[v8.0] System initialization failed.")
    
    logger.info("[v8.0][PHASE 4] System diagnostics completed successfully")

    # ==========================================================================
    # PHASE 5: SYNCHRONIZATION SEQUENCE
    # ==========================================================================
    print("[v8.0][PHASE 5] Running synchronization sequence...")
    initialization.run_sync_sequence()  # Optional sync with network hubs
    logger.info("[v8.0][PHASE 5] Synchronization completed")

    # ==========================================================================
    # PHASE 6: LOCAL API SERVER STARTUP
    # ==========================================================================
    print("[v8.0][PHASE 6] Starting local API server...")
    start_local_api_server()
    api_ready = wait_for_api_server(timeout=10)
    
    if api_ready:
        logger.info("[v8.0][PHASE 6] API server ready for requests")
    else:
        logger.warning("[v8.0][PHASE 6] API server may not be available - continuing anyway")

    # ==========================================================================
    # PHASE 7: MEDIA SUBSYSTEM INITIALIZATION (v8.0 NEW)
    # ==========================================================================
    print("[v8.0][PHASE 7] Initializing media creation subsystems...")
    try:
        # Initialize media generators if available
        media_modules = []
        
        try:
            import SarahMemoryMusicGenerator
            media_modules.append("MusicGenerator")
            logger.info("[v8.0][PHASE 7] Music Generator initialized")
        except Exception as e:
            logger.debug(f"[v8.0][PHASE 7] Music Generator not available: {e}")
        
        try:
            import SarahMemoryLyricsToSong
            media_modules.append("LyricsToSong")
            logger.info("[v8.0][PHASE 7] Lyrics-to-Song Engine initialized")
        except Exception as e:
            logger.debug(f"[v8.0][PHASE 7] Lyrics-to-Song not available: {e}")
        
        try:
            import SarahMemoryVideoEditorCore
            media_modules.append("VideoEditor")
            logger.info("[v8.0][PHASE 7] Video Editor Core initialized")
        except Exception as e:
            logger.debug(f"[v8.0][PHASE 7] Video Editor not available: {e}")
        
        try:
            import SarahMemoryCanvasStudio
            media_modules.append("CanvasStudio")
            logger.info("[v8.0][PHASE 7] Canvas Studio initialized")
        except Exception as e:
            logger.debug(f"[v8.0][PHASE 7] Canvas Studio not available: {e}")
        
        if media_modules:
            print(f"[v8.0][PHASE 7] Media modules loaded: {', '.join(media_modules)}")
            logger.info(f"[v8.0][PHASE 7] {len(media_modules)} media subsystems ready")
        else:
            logger.info("[v8.0][PHASE 7] No media subsystems available (optional)")
    
    except Exception as e:
        logger.warning(f"[v8.0][PHASE 7] Media subsystem initialization warning: {e}")

    # ==========================================================================
    # PHASE 8: LAUNCH INTEGRATION MENU
    # ==========================================================================
    print("\n[v8.0][PHASE 8] All systems ready. Launching SarahMemory AiOS v8.0...")
    logger.info("[v8.0][PHASE 8] Starting SarahMemory AI Bot Integration Menu")
    
    # Small delay for visual effect
    time.sleep(0.5)
    
    # Launch the main integration menu
    integration.integration_menu()

except KeyboardInterrupt:
    logger.info("[v8.0] User interrupted startup sequence (Ctrl+C)")
    print("\n[v8.0] Shutdown initiated by user.")
    sys.exit(0)

except Exception as e:
    logger.error(f"[v8.0] Critical error in main execution: {e}")
    print(f"\n[v8.0] An unexpected error occurred:")
    print(f"Error: {e}")
    print("\nPlease check the logs for more details:")
    print(f"Log file: {log_filename}")
    sys.exit(1)

# =============================================================================
# DATABASE SCHEMA VALIDATION - v8.0
# =============================================================================
def _ensure_response_table(db_path=None):
    """
    Ensure the response table exists in the database.
    v8.0 Enhanced with better error handling and logging.
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
        
        logging.debug("[v8.0][DB] Ensured table `response` in %s", db_path)
    
    except Exception as e:
        try:
            import logging
            logging.warning("[v8.0][DB] Ensure `response` failed: %s", e)
        except Exception:
            pass

# Initialize response table
try:
    _ensure_response_table()
except Exception:
    pass

# =============================================================================
# END OF SarahMemoryMain.py v8.0.0
# =============================================================================
