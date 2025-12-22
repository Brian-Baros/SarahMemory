"""--==The SarahMemory Project==--
File: SarahMemoryMain.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
Time: 16:30:00
Author: © 2025 Brian Lee Baros. All Rights Reserved.
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
import warnings
import requests
import platform
import SarahMemoryGlobals as config

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
    import SarahMemoryIntegration as integration
    
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
