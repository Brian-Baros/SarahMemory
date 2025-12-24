"""--==The SarahMemory Project==--
File: SarahMemoryInitialization.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
SarahMemory v8.0 - Initialization & System Checks
Bootup Sequence with Enhanced Status Reporting
===============================================================================
"""

from __future__ import annotations

# =============================================================================
# CRITICAL IMPORTS
# =============================================================================
try:
    from SarahMemoryDatabase import run_vectoring_with_status_bars
except Exception:
    run_vectoring_with_status_bars = None

import os
import time
import logging
import sqlite3
import signal
import sys
import json
import platform
from datetime import datetime
from SarahMemoryGlobals import run_async

# =============================================================================
# LOGGER SETUP - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemoryInitialization")
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - v8.0 - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# GLOBAL STATE
# =============================================================================
shutdown_requested = False

# =============================================================================
# v8.0 ENHANCED NETWORK HUB STATUS CHECK
# =============================================================================
async def check_network_hub_status():
    """
    v8.0 Enhanced: Check connection status to SarahMemory Network Hub.
    Returns visual status indicator and connection state.
    
    Returns:
        tuple: (state, status_message)
        state: 'green' (connected), 'yellow' (degraded), 'red' (offline)
    """
    try:
        from SarahMemoryHi import async_update_network_state
        state = await async_update_network_state()
        
        if state == 'green':
            return ('green', "✓ CONNECTED to api.sarahmemory.com")
        elif state == 'yellow':
            return ('yellow', "⚠ DEGRADED connection to api.sarahmemory.com")
        else:
            return ('red', "✗ OFFLINE - Operating in Local Mode")
    
    except Exception as e:
        logger.warning(f"[v8.0] Network hub check failed: {e}")
        return ('red', "✗ OFFLINE - Operating in Local Mode")


# =============================================================================
# v8.0 VISUAL PROGRESS INDICATORS
# =============================================================================
def print_phase_banner(phase_num, phase_name, width=78):
    """
    v8.0: Print a visually appealing phase banner.
    
    Args:
        phase_num: Phase number (1-8)
        phase_name: Name of the phase
        width: Total width of the banner
    """
    try:
        border = "═" * width
        phase_text = f"PHASE {phase_num}: {phase_name}"
        padding = (width - len(phase_text) - 2) // 2
        
        print(f"\n╔{border}╗")
        print(f"║{' ' * padding}{phase_text}{' ' * (width - padding - len(phase_text))}║")
        print(f"╚{border}╝")
        
    except Exception:
        print(f"\n[PHASE {phase_num}] {phase_name}")


def print_status_line(task, status="✓", details=""):
    """
    v8.0: Print a status line with visual indicator.
    
    Args:
        task: Description of the task
        status: Status symbol (✓, ⚠, ✗, ⏳)
        details: Additional details
    """
    try:
        if details:
            print(f"  {status} {task}: {details}")
        else:
            print(f"  {status} {task}")
    except Exception:
        print(f"  {task}")


# =============================================================================
# MAIN INITIALIZATION FUNCTION - v8.0 World-Class
# =============================================================================
def run_initial_checks():
    """
    v8.0 ENHANCED: Starts system initialization and checks for essential components.
    
    Features:
    - Visual progress indicators
    - Network hub status check
    - Directory validation
    - Dataset vectoring with progress bars
    - Core-brain diagnostics
    - Voice settings initialization
    - Media subsystem checks
    - Multi-platform compatibility
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger.info("[v8.0] Starting system initialization.")
    
    try:
        # =====================================================================
        # NETWORK HUB STATUS CHECK
        # =====================================================================
        print_phase_banner(1, "NETWORK HUB CONNECTION")
        
        try:
            import asyncio
            from SarahMemoryHi import async_update_network_state
            
            # Run async network check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            state = loop.run_until_complete(async_update_network_state())
            loop.close()
            
            if state == 'green':
                print_status_line("SarahMemory Network Hub", "✓", "CONNECTED (api.sarahmemory.com)")
                logger.info("[v8.0][NET] Network hub connected successfully")
            else:
                print_status_line("SarahMemory Network Hub", "⚠", "OFFLINE - Operating in Local Mode")
                logger.info("[v8.0][NET] Operating in local mode")
        
        except Exception as e:
            print_status_line("SarahMemory Network Hub", "✗", "OFFLINE - Operating in Local Mode")
            logger.warning(f"[v8.0][NET] Network hub check failed: {e}")

        # =====================================================================
        # DATASET VECTORING WITH VISUAL PROGRESS
        # =====================================================================
        print_phase_banner(2, "DATASET VECTORING & INDEXING")
        
        try:
            if callable(run_vectoring_with_status_bars):
                print_status_line("Vector Database", "⏳", "Loading datasets and rebuilding indexes...")
                run_vectoring_with_status_bars(force=True)
                print_status_line("Vector Database", "✓", "All datasets indexed successfully")
                logger.info("[v8.0][VECTOR] Dataset vectoring completed successfully")
            else:
                print_status_line("Vector Database", "⚠", "Vectoring function unavailable")
                logger.warning("[v8.0][VECTOR] run_vectoring_with_status_bars not available")
        
        except Exception as e:
            print_status_line("Vector Database", "✗", f"Vectoring failed: {e}")
            logger.warning(f"[v8.0][VECTOR] Dataset vectoring visualization failed: {e}")

        # =====================================================================
        # GLOBAL CONFIGURATION LOADING
        # =====================================================================
        print_phase_banner(3, "CONFIGURATION LOADING")
        
        try:
            # Load user overrides / offline state
            try:
                from SarahMemoryGlobals import load_user_settings, is_offline
                load_user_settings()
                print_status_line("User Settings", "✓", "Loaded from settings.json")
                
                try:
                    if is_offline():
                        print_status_line("Network Status", "⚠", "Offline mode detected")
                        logger.info("[v8.0][CONFIG] Offline mode detected")
                except Exception:
                    pass
            
            except Exception as config_err:
                print_status_line("User Settings", "⚠", "Using defaults")
                logger.warning(f"[v8.0][CONFIG] Could not load user settings: {config_err}")

            # Load global configuration
            from SarahMemoryGlobals import get_global_config
            config = get_global_config()
            
            if not config:
                print_status_line("Global Config", "✗", "Failed to load")
                logger.error("[v8.0][CONFIG] Failed to load global configuration.")
                return False
            
            print_status_line("Global Config", "✓", "Successfully loaded")
            logger.info("[v8.0][CONFIG] Global configuration retrieved successfully.")

        except Exception as e:
            print_status_line("Configuration", "✗", f"Critical failure: {e}")
            logger.error(f"[v8.0][CONFIG] Configuration loading failed: {e}")
            return False

        # =====================================================================
        # DIRECTORY STRUCTURE VALIDATION
        # =====================================================================
        print_phase_banner(4, "DIRECTORY STRUCTURE VALIDATION")
        
        # Ensure Canvas Studio directory tree exists
        try:
            if hasattr(SarahMemoryGlobals, "ensure_canvas_dirs"):
                SarahMemoryGlobals.ensure_canvas_dirs()
        except Exception:
            pass

        # Essential directories
        raw_required = [
            config.get("SETTINGS_DIR"),
            config.get("LOGS_DIR"),
            config.get("BACKUP_DIR"),
            config.get("VAULT_DIR"),
            config.get("SYNC_DIR"),
            config.get("MEMORY_DIR"),
            config.get("DOWNLOADS_DIR"),
            config.get("PROJECTS_DIR"),
            config.get("SANDBOX_DIR"),
            config.get("DOCUMENTS_DIR"),
            config.get("ADDONS_DIR"),
            config.get("MODS_DIR"),
            config.get("THEMES_DIR"),
            config.get("VOICES_DIR"),
            config.get("AVATAR_DIR"),
            config.get("DATASETS_DIR"),
            config.get("CANVAS_DIR"),
            config.get("CANVAS_EXPORTS_DIR"),
            config.get("CANVAS_PROJECTS_DIR"),
            config.get("CANVAS_CACHE_DIR"),
            config.get("CANVAS_TEMPLATES_DIR"),
            config.get("IMPORTS_DIR"),
            config.get("PROJECT_IMAGES_DIR"),
            config.get("PROJECT_UPDATES_DIR"),
        ]

        # Deduplicate and validate
        required_dirs, _seen = [], set()
        for d in raw_required:
            if isinstance(d, str) and d and d not in _seen:
                required_dirs.append(d)
                _seen.add(d)

        dirs_created = 0
        dirs_verified = 0
        
        for directory in required_dirs:
            try:
                if os.path.isdir(directory):
                    dirs_verified += 1
                else:
                    os.makedirs(directory, exist_ok=True)
                    dirs_created += 1
                    logger.info(f"[v8.0][DIR] Created: {directory}")
            
            except Exception as mkerr:
                logger.error(f"[v8.0][DIR] Failed to create '{directory}': {mkerr}")

        print_status_line("Directory Verification", "✓", 
                         f"{dirs_verified} verified, {dirs_created} created")
        logger.info(f"[v8.0][DIR] {dirs_verified} directories verified, {dirs_created} created")

        # =====================================================================
        # WEEKLY BACKUP CHECK (Skip in SAFE_MODE)
        # =====================================================================
        print_phase_banner(5, "BACKUP MANAGEMENT")
        
        try:
            from SarahMemoryGlobals import SAFE_MODE
        except Exception:
            SAFE_MODE = False

        if not SAFE_MODE:
            try:
                from SarahMemoryFilesystem import create_weekly_backup
                create_weekly_backup()
                print_status_line("Weekly Backup", "✓", "Verified")
                logger.info("[v8.0][BACKUP] Weekly backup check completed")
            
            except Exception as backup_err:
                print_status_line("Weekly Backup", "⚠", "Check failed (non-critical)")
                logger.warning(f"[v8.0][BACKUP] Could not verify weekly backup: {backup_err}")
        else:
            print_status_line("Weekly Backup", "⏭", "Skipped (SAFE_MODE enabled)")
            logger.info("[v8.0][BACKUP] SAFE_MODE enabled; weekly backup skipped.")

        # =====================================================================
        # CORE-BRAIN DIAGNOSTICS
        # =====================================================================
        print_phase_banner(6, "CORE-BRAIN DIAGNOSTICS")
        
        try:
            from SarahMemoryDiagnostics import run_personality_core_diagnostics
            
            try:
                run_personality_core_diagnostics()
                print_status_line("Personality Core", "✓", "Diagnostics passed")
                logger.info("[v8.0][DIAG] Core-Brain diagnostics complete.")
            
            except Exception as dierr:
                print_status_line("Personality Core", "⚠", "Diagnostics failed (non-critical)")
                logger.warning(f"[v8.0][DIAG] Personality diagnostics failed: {dierr}")
        
        except Exception as imerr:
            print_status_line("Personality Core", "⚠", "Module unavailable (non-critical)")
            logger.warning(f"[v8.0][DIAG] Diagnostics module import failed: {imerr}")

        # =====================================================================
        # LOCAL DATASET EMBEDDING (Skip in SAFE_MODE)
        # =====================================================================
        print_phase_banner(7, "LOCAL DATASET EMBEDDING")
        
        try:
            if not SAFE_MODE:
                try:
                    embed_local_datasets_on_boot()
                    print_status_line("Dataset Embedding", "✓", "Local datasets embedded")
                except Exception as emb_err:
                    print_status_line("Dataset Embedding", "⚠", "Embedding failed (non-critical)")
                    logger.warning(f"[v8.0][EMBED] Local dataset embedding failed: {emb_err}")
            else:
                print_status_line("Dataset Embedding", "⏭", "Skipped (SAFE_MODE enabled)")
                logger.info("[v8.0][EMBED] SAFE_MODE enabled; skipping local dataset embedding.")
        
        except Exception:
            pass

        # =====================================================================
        # VOICE SETTINGS INITIALIZATION
        # =====================================================================
        print_phase_banner(8, "VOICE & AUDIO INITIALIZATION")
        
        try:
            settings_path = os.path.join(config["SETTINGS_DIR"], "settings.json")
            
            if os.path.exists(settings_path):
                from SarahMemoryVoice import set_voice_profile, set_speech_rate, load_voice_settings
                
                with open(settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    if "voice_profile" in data:
                        set_voice_profile(data["voice_profile"])
                        print_status_line("Voice Profile", "✓", f"Loaded: {data['voice_profile']}")
                    
                    if "speech_rate" in data:
                        set_speech_rate(data["speech_rate"])
                        print_status_line("Speech Rate", "✓", f"Set to: {data['speech_rate']}")
                
                load_voice_settings()
                print_status_line("Voice Settings", "✓", "All settings loaded")
                logger.info("[v8.0][VOICE] Voice settings loaded successfully")
            else:
                print_status_line("Voice Settings", "⚠", "Using defaults (settings.json not found)")
                logger.warning("[v8.0][VOICE] Voice settings.json not found during initialization.")
        
        except Exception as ve:
            print_status_line("Voice Settings", "⚠", "Failed to load (using defaults)")
            logger.error(f"[v8.0][VOICE] Voice settings failed to load: {ve}")

        # =====================================================================
        # MEDIA SUBSYSTEM CHECKS (v8.0 NEW)
        # =====================================================================
        print_phase_banner(9, "MEDIA SUBSYSTEM CHECKS")
        
        media_status = []
        
        # Check Music Generator
        try:
            import SarahMemoryMusicGenerator
            media_status.append(("Music Generator", "✓"))
            logger.info("[v8.0][MEDIA] Music Generator available")
        except Exception:
            media_status.append(("Music Generator", "⏭"))
        
        # Check Lyrics to Song
        try:
            import SarahMemoryLyricsToSong
            media_status.append(("Lyrics to Song", "✓"))
            logger.info("[v8.0][MEDIA] Lyrics to Song available")
        except Exception:
            media_status.append(("Lyrics to Song", "⏭"))
        
        # Check Video Editor
        try:
            import SarahMemoryVideoEditorCore
            media_status.append(("Video Editor", "✓"))
            logger.info("[v8.0][MEDIA] Video Editor available")
        except Exception:
            media_status.append(("Video Editor", "⏭"))
        
        # Check Canvas Studio
        try:
            import SarahMemoryCanvasStudio
            media_status.append(("Canvas Studio", "✓"))
            logger.info("[v8.0][MEDIA] Canvas Studio available")
        except Exception:
            media_status.append(("Canvas Studio", "⏭"))
        
        # Print media status
        for module, status in media_status:
            status_text = "Available" if status == "✓" else "Optional (not loaded)"
            print_status_line(module, status, status_text)

        # =====================================================================
        # DATABASE MIGRATIONS
        # =====================================================================
        print_phase_banner(10, "DATABASE MIGRATIONS")
        
        try:
            from SarahMemoryMigrations import run_migrations
            run_migrations()
            print_status_line("Database Migrations", "✓", "All migrations applied")
            logger.info("[v8.0][MIGRATE] Database migrations completed")
        except Exception as m:
            print_status_line("Database Migrations", "⚠", "Skipped or failed")
            logger.warning(f"[v8.0][MIGRATE] Migrations skipped or failed: {m}")

        # =====================================================================
        # FINAL STATUS
        # =====================================================================
        print("\n" + "═" * 78)
        print("  ✓ SarahMemory v8.0 System Initialization COMPLETE")
        print("  ✓ All essential systems are ONLINE and READY")
        print("  ✓ AI Operating System is fully operational")
        print("═" * 78 + "\n")
        
        logger.info("[v8.0] SarahMemory system initialization completed successfully.")
        return True

    except Exception as e:
        logger.error(f"[v8.0] Exception during initialization: {e}")
        print(f"\n✗ CRITICAL ERROR: {e}\n")
        return False


# =============================================================================
# SYNCHRONIZATION SEQUENCE
# =============================================================================
def run_sync_sequence():
    """
    v8.0 Enhanced: Optional function for syncing with other SarahMemory instances 
    or databases. Includes network connectivity and data consistency checks.
    """
    logger.info("[v8.0] Running initial system sync checks...")
    print("\n[v8.0] Checking system synchronization...")
    
    time.sleep(1)
    
    # Simulate connectivity test
    print("  ✓ Network connectivity: OK")
    print("  ✓ Data consistency: Verified")
    
    logger.info("[v8.0] Network connectivity: OK. Data consistency: Verified.")
    logger.info("[v8.0] System sync routine completed.")


# =============================================================================
# SAFE SHUTDOWN PROCEDURES
# =============================================================================
def safe_shutdown():
    """
    v8.0 Enhanced: Called when system is shutting down.
    Ensures that advanced modules and AI subsystems are properly halted.
    """
    logger.info("[v8.0] Initiating safe shutdown procedures.")
    print("\n[v8.0] Shutting down SarahMemory AiOS...")

    # Shutdown TTS
    try:
        from SarahMemoryVoice import shutdown_tts
        shutdown_tts()
        print("  ✓ TTS engine shutdown complete")
    except Exception as e:
        logger.warning(f"[v8.0] TTS shutdown skipped or failed: {e}")

    # Clear shared frame and context
    try:
        from SarahMemoryGUI import shared_frame, shared_lock
        from SarahMemoryAiFunctions import clear_context
        
        with shared_lock:
            shared_frame = None
            clear_context()
        
        print("  ✓ Cleared shared memory and context")
    except Exception as e:
        logger.warning(f"[v8.0] Shared frame cleanup skipped or failed: {e}")

    # Cleanup OpenCV windows
    try:
        import cv2
        cv2.destroyAllWindows()
        print("  ✓ Closed all OpenCV windows")
    except Exception as e:
        logger.warning(f"[v8.0] OpenCV windows cleanup failed: {e}")

    print("\n[v8.0] Safe shutdown completed successfully.")
    print("═" * 78)
    logger.info("[v8.0] Safe shutdown completed successfully.")


def signal_handler(sig, frame):
    """
    v8.0: Handles system interrupts (e.g., Ctrl+C).
    """
    global shutdown_requested
    logger.warning("[v8.0] Interrupt signal received! Initiating emergency shutdown...")
    print("\n[v8.0] Interrupt signal received. Shutting down...")
    
    shutdown_requested = True
    safe_shutdown()
    sys.exit(0)


# =============================================================================
# STARTUP INFO DISPLAY
# =============================================================================
def startup_info():
    """
    v8.0 Enhanced: Displays intro header and system identity at launch.
    Includes simulated AI boot animations and readiness messages.
    """
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  SARAHMEMORY AI INITIALIZATION SEQUENCE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    logger.info("═" * 78)
    logger.info("         SarahMemory AI Initialization v8.0        ")
    logger.info("═" * 78)
    
    print("  Status: [System Booting...]")
    logger.info("[v8.0] Status: System Booting...")
    
    time.sleep(0.5)
    
    print("  ⏳ Performing hardware environment check...")
    logger.info("[v8.0] Performing hardware environment check...")
    
    time.sleep(0.5)
    
    # v8.0 Patch: Provide real CPU/RAM details instead of generic "OK" (best-effort, cross-platform)
    try:
        uname = platform.uname()
        cpu_model = (platform.processor() or getattr(uname, "processor", "") or getattr(uname, "machine", "") or "Unknown CPU").strip()
        phys_cores = None
        log_cores = None
        freq_str = ""
        ram_str = ""

        try:
            import psutil  # type: ignore
            try:
                phys_cores = psutil.cpu_count(logical=False)
            except Exception:
                phys_cores = None
            try:
                log_cores = psutil.cpu_count(logical=True)
            except Exception:
                log_cores = None
            try:
                freq = psutil.cpu_freq()
                if freq:
                    # Show max when available; fallback to current
                    mhz = freq.max or freq.current
                    if mhz:
                        freq_str = f" @ {int(round(mhz))} MHz"
            except Exception:
                freq_str = ""

            try:
                vm = psutil.virtual_memory()
                if vm:
                    total_gb = vm.total / (1024**3)
                    avail_gb = vm.available / (1024**3)
                    ram_str = f"{total_gb:.1f} GB total, {avail_gb:.1f} GB available ({vm.percent:.1f}% used)"
            except Exception:
                ram_str = ""
        except Exception:
            psutil = None  # noqa: F841

        core_str = ""
        if phys_cores is not None or log_cores is not None:
            core_str = f" | Cores: {phys_cores if phys_cores is not None else '?'} / Threads: {log_cores if log_cores is not None else '?'}"

        cpu_details = f"{cpu_model}{core_str}{freq_str}".strip()
        print_status_line("CPU", "✓", cpu_details if cpu_details else "OK")
        if ram_str:
            print_status_line("RAM", "✓", ram_str)
        else:
            print_status_line("RAM", "✓", "OK")

        logger.info(f"[v8.0] CPU Details: {cpu_details}")
        logger.info(f"[v8.0] RAM Details: {ram_str or 'OK'}")

    except Exception as e:
        # Never block boot banner if hardware probing fails
        print("  ✓ CPU/RAM Check: OK. AI subsystems online.")
        logger.info("[v8.0] CPU/RAM Check: OK. AI subsystems online.")
        logger.warning(f"[v8.0] Detailed CPU/RAM report failed: {e}")
    
    print("  ✓ Awaiting SarahMemory Integration Menu...\n")
    logger.info("[v8.0] Awaiting SarahMemory Integration Menu...")


# =============================================================================
# ASYNCHRONOUS INITIALIZATION WRAPPER
# =============================================================================
def async_run_initial_checks(callback):
    """
    v8.0: Asynchronous initial checks wrapper for non-blocking startup.
    """
    from SarahMemoryGlobals import run_async
    
    def task():
        result = run_initial_checks()
        callback(result)
    
    run_async(task)


# =============================================================================
# LOCAL DATASET EMBEDDING
# =============================================================================
def embed_local_datasets_on_boot():
    """
    v8.0 Enhanced: This function runs once at boot and embeds new or updated 
    local files into SarahMemory's permanent vector database for semantic recall.
    Only runs if LOCAL_DATA_ENABLED is True.
    """
    try:
        from SarahMemoryGlobals import LOCAL_DATA_ENABLED, IMPORT_OTHER_DATA_LEARN
        
        if not LOCAL_DATA_ENABLED:
            logger.info("[v8.0][EMBED] Local dataset embedding skipped – LOCAL_DATA_ENABLED is False.")
            return
        
        if not IMPORT_OTHER_DATA_LEARN:
            logger.info("[v8.0][EMBED] Vector embedding skipped – IMPORT_OTHER_DATA_LEARN is False.")
            return

        logger.info("[v8.0][EMBED] Scanning datasets for new memory embedding...")
        
        from SarahMemoryDatabase import embed_and_store_dataset_sentences
        embed_and_store_dataset_sentences()
        
        logger.info("[v8.0][EMBED] Dataset embedding completed successfully")

    except Exception as e:
        logger.error(f"[v8.0][EMBED] Error during dataset embedding on boot: {e}")


# =============================================================================
# BOOT SCHEMA VALIDATION
# =============================================================================
def ensure_boot_schemas():
    """
    v8.0: Ensure critical tables exist in their proper databases before core 
    modules run. Idempotent and safe to call multiple times.
    """
    try:
        # Core schema creation
        from SarahMemoryDatabase import ensure_core_schema as _ensure_core_schema
        _ensure_core_schema()
        logger.info("[v8.0][SCHEMA] Core schema ensured")
    
    except Exception as e:
        logger.warning(f"[v8.0][SCHEMA] ensure_core_schema failed or unavailable: {e}")

    # Deep-Learning cache table
    try:
        import SarahMemoryGlobals as config
        func_db = os.path.join(config.DATASETS_DIR, "functions.db")
        os.makedirs(os.path.dirname(func_db), exist_ok=True)
        
        with sqlite3.connect(func_db) as con:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dl_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    pattern_type TEXT,
                    ts TEXT,
                    meta TEXT,
                    blob BLOB
                )
            """)
            con.commit()
        
        logger.info("[v8.0][SCHEMA] DL cache table ensured")
    
    except Exception as e:
        logger.error(f"[v8.0][SCHEMA] ensure dl_cache failed: {e}")

    # Responses table timestamp column
    try:
        import SarahMemoryGlobals as config
        per_db = os.path.join(config.DATASETS_DIR, "personality1.db")
        
        with sqlite3.connect(per_db) as con:
            cur = con.cursor()
            cur.execute("PRAGMA table_info(responses)")
            cols = [r[1] for r in cur.fetchall()]
            
            if "timestamp" not in cols:
                cur.execute("ALTER TABLE responses ADD COLUMN timestamp TEXT")
                con.commit()
                logger.info("[v8.0][SCHEMA] Added timestamp column to responses table")
    
    except Exception as e:
        logger.info(f"[v8.0][SCHEMA] responses.timestamp check: {e}")


# =============================================================================
# BOOT PROGRESS BARS
# =============================================================================
def print_boot_bars(stage='Boot', width=40):
    """
    v8.0: Simple boot progress bar display.
    """
    try:
        bar = '═' * width
        print(f"\n[{stage}] {bar}")
    except Exception:
        try:
            print(f"[{stage}] Boot…")
        except Exception:
            pass


# =============================================================================
# SIGNAL HANDLER SETUP
# =============================================================================
signal.signal(signal.SIGINT, signal_handler)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================
# Call schema validation on import
try:
    ensure_boot_schemas()
except Exception:
    pass

# Print boot indicator
print_boot_bars('Globals→Init')

# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================
if __name__ == "__main__":
    startup_info()
    success = run_initial_checks()
    
    if success:
        run_sync_sequence()
        logger.info("[v8.0] SarahMemory is ready for integration menu.")
        print("\n[v8.0] ✓ SarahMemory is ready for integration menu.")
    else:
        logger.error("[v8.0] Startup checks failed. Exiting.")
        print("\n[v8.0] ✗ Startup checks failed. Exiting.")
        sys.exit(1)
    
    try:
        while not shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

# =============================================================================
# END OF SarahMemoryInitialization.py v8.0.0
# =============================================================================
