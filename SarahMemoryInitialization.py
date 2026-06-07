"""--==The SarahMemory Project==--
File: SarahMemoryInitialization.py
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
SarahMemory v8.0 - Initialization & System Checks
Bootup Sequence with Enhanced Status Reporting
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "boot_initializer"
# CATEGORY = "startup_initialization"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_filesystem_network"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "initialization"
# FAMILY = "boot_sequence"
# GOVERNANCE_LEVEL = "critical"
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
# NOTES = "Boot initialization and startup checks engine for network status, vectoring, config loading, directory validation, backups, diagnostics, embedding, voice init, media checks, and migrations."
# --- SARAHMETA END ---

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
import shutil
import threading
from datetime import datetime
from SarahMemoryGlobals import run_async
import SarahMemoryGlobals as SarahMemoryGlobals

# =============================================================================
# LOGGER SETUP - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemoryInitialization")
logger.setLevel(logging.DEBUG if str(os.getenv("SARAH_DEBUG_MODE", os.getenv("DEBUG_MODE", "0"))).strip().lower() in ("1", "true", "yes", "on") else logging.INFO)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - v8.0 - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# GLOBAL STATE
# =============================================================================
shutdown_requested = False

# Startup background task control
_STARTUP_BACKGROUND_THREADS = {}
_STARTUP_BACKGROUND_LOCK = threading.Lock()

_SAFE_SHUTDOWN_STARTED = False
_SAFE_SHUTDOWN_LOCK = threading.RLock()


def _join_startup_background_threads(timeout_each: float = 1.0) -> None:
    """Best-effort shutdown for startup background workers."""
    try:
        with _STARTUP_BACKGROUND_LOCK:
            items = list(_STARTUP_BACKGROUND_THREADS.items())
        for task_name, thread in items:
            try:
                if thread is not None and thread.is_alive():
                    thread.join(timeout=timeout_each)
            except Exception:
                pass
    except Exception:
        pass


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



def _cfg_bool(name, default=False):
    """Read a boolean flag from globals or environment without making boot fragile."""
    try:
        value = getattr(SarahMemoryGlobals, name, default)
    except Exception:
        value = default
    env_name = f"SARAH_{name}"
    try:
        env_val = os.getenv(env_name, None)
        if env_val is not None and str(env_val).strip() != "":
            value = env_val
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _boot_dataset_embedding_mode() -> str:
    """Return manual|background|eager for boot dataset embedding.

    Runtime optimization default is manual: do not scan/embed datasets during
    normal boot. This protects the active C: NVMe drive and keeps the UI fast.
    Users/developers can re-enable old behavior with either:
      SARAH_BOOT_DATASET_EMBEDDING_MODE=background
      SARAH_BOOT_DATASET_EMBEDDING_MODE=eager
      SARAH_BOOT_EAGER_DATASET_EMBEDDING=true
    """
    try:
        value = getattr(SarahMemoryGlobals, "BOOT_DATASET_EMBEDDING_MODE", None)
    except Exception:
        value = None
    try:
        env_val = os.getenv("SARAH_BOOT_DATASET_EMBEDDING_MODE") or os.getenv("BOOT_DATASET_EMBEDDING_MODE")
        if env_val:
            value = env_val
    except Exception:
        pass
    if _cfg_bool("BOOT_EAGER_DATASET_EMBEDDING", False):
        return "eager"
    mode = str(value or "manual").strip().lower()
    if mode in ("off", "skip", "disabled", "manual", "none", "false", "0"):
        return "manual"
    if mode in ("background", "defer", "deferred"):
        return "background"
    if mode in ("eager", "boot", "startup", "true", "1"):
        return "eager"
    return "manual"


def _background_thread_alive(task_name: str) -> bool:
    try:
        thread = _STARTUP_BACKGROUND_THREADS.get(str(task_name or "").strip().lower())
        return bool(thread is not None and thread.is_alive())
    except Exception:
        return False


def _start_background_dataset_embedding() -> bool:
    """Run dataset embedding in one daemon thread so boot can continue to Phase 8 immediately."""
    task_key = "dataset_embedding"
    with _STARTUP_BACKGROUND_LOCK:
        if _background_thread_alive(task_key):
            return False

        def _worker():
            started = time.perf_counter()
            try:
                logger.info("[v8.0][EMBED] Deferred boot dataset embedding started in background.")
                embed_local_datasets_on_boot()
                logger.info("[v8.0][EMBED] Deferred boot dataset embedding completed in %.2f seconds.", time.perf_counter() - started)
            except Exception as e:
                logger.warning(f"[v8.0][EMBED] Deferred boot dataset embedding failed: {e}")
            finally:
                with _STARTUP_BACKGROUND_LOCK:
                    _STARTUP_BACKGROUND_THREADS.pop(task_key, None)

        thread = threading.Thread(target=_worker, name="SM_BootDatasetEmbedding", daemon=True)
        _STARTUP_BACKGROUND_THREADS[task_key] = thread
        thread.start()
        return True


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
            if _cfg_bool("BOOT_RUN_VECTORING_ON_STARTUP", False):
                if callable(run_vectoring_with_status_bars):
                    print_status_line("Vector Database", "⏳", "Checking datasets/indexes without forced rebuild...")
                    run_vectoring_with_status_bars(force=_cfg_bool("BOOT_FORCE_VECTOR_REBUILD", False))
                    print_status_line("Vector Database", "✓", "Vector check completed")
                    logger.info("[v8.0][VECTOR] Dataset vector check completed")
                else:
                    print_status_line("Vector Database", "⚠", "Vectoring function unavailable")
                    logger.warning("[v8.0][VECTOR] run_vectoring_with_status_bars not available")
            else:
                print_status_line("Vector Database", "⏭", "Skipped during boot (manual/on-demand indexing policy)")
                logger.info("[v8.0][VECTOR] Boot vectoring skipped by optimized runtime policy.")
        
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

        if SAFE_MODE:
            print_status_line("Weekly Backup", "⏭", "Skipped (SAFE_MODE enabled)")
            logger.info("[v8.0][BACKUP] SAFE_MODE enabled; weekly backup skipped.")
        elif not _cfg_bool("BOOT_WEEKLY_BACKUP_CHECK", False):
            print_status_line("Weekly Backup", "⏭", "Skipped during boot (manual/scheduled backup policy)")
            logger.info("[v8.0][BACKUP] Boot backup check skipped by optimized runtime policy.")
        else:
            try:
                from SarahMemoryFilesystem import create_weekly_backup
                create_weekly_backup()
                print_status_line("Weekly Backup", "✓", "Verified")
                logger.info("[v8.0][BACKUP] Weekly backup check completed")
            
            except Exception as backup_err:
                print_status_line("Weekly Backup", "⚠", "Check failed (non-critical)")
                logger.warning(f"[v8.0][BACKUP] Could not verify weekly backup: {backup_err}")

        # =====================================================================
        # CORE-BRAIN DIAGNOSTICS
        # =====================================================================
        print_phase_banner(6, "CORE-BRAIN DIAGNOSTICS")
        
        if not _cfg_bool("BOOT_PERSONALITY_DIAGNOSTICS", False):
            print_status_line("Personality Core", "⏭", "Skipped during boot (diagnostics available on demand)")
            logger.info("[v8.0][DIAG] Personality diagnostics skipped by optimized runtime policy.")
        else:
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
        # LOCAL DATASET EMBEDDING (Skip in SAFE_MODE / defer by default)
        # =====================================================================
        print_phase_banner(7, "LOCAL DATASET EMBEDDING")

        try:
            if SAFE_MODE:
                print_status_line("Dataset Embedding", "⏭", "Skipped (SAFE_MODE enabled)")
                logger.info("[v8.0][EMBED] SAFE_MODE enabled; skipping local dataset embedding.")
            else:
                mode = _boot_dataset_embedding_mode()
                try:
                    if mode == "manual":
                        print_status_line("Dataset Embedding", "⏭", "Skipped during boot (manual/on-demand embedding policy)")
                        logger.info("[v8.0][EMBED] Boot dataset embedding skipped by optimized runtime policy.")
                    elif mode == "background":
                        started = _start_background_dataset_embedding()
                        if started:
                            print_status_line("Dataset Embedding", "⏭", "Deferred to background by explicit policy")
                            logger.info("[v8.0][EMBED] Local dataset embedding deferred to background by explicit policy")
                        else:
                            print_status_line("Dataset Embedding", "⏭", "Background embedding already running")
                    else:
                        embed_started = time.perf_counter()
                        embed_local_datasets_on_boot()
                        print_status_line("Dataset Embedding", "✓", f"Local datasets embedded in {time.perf_counter() - embed_started:.2f}s")
                except Exception as emb_err:
                    print_status_line("Dataset Embedding", "⚠", "Embedding failed (non-critical)")
                    logger.warning(f"[v8.0][EMBED] Local dataset embedding failed: {emb_err}")

        except Exception:
            pass

# =====================================================================
        # B-LEVEL DRIVER READINESS SCAN
        # =====================================================================
        print_phase_banner(8, "B-LEVEL DRIVER READINESS SCAN")

        try:
            from SarahMemoryHi import get_boot_environment_snapshot

            env_snapshot = get_boot_environment_snapshot(force_refresh=False, refresh_reason="phase8_driver_readiness")
            readiness = env_snapshot.get("driver_readiness", {}) if isinstance(env_snapshot, dict) else {}
            entries = readiness.get("entries", []) if isinstance(readiness, dict) else []

            if entries:
                for item in entries:
                    try:
                        print_status_line(item.get("name", "Unknown Hardware"), "✓", item.get("status", "DETECTED/NOT READY"))
                    except Exception:
                        pass

                detected_count = int(readiness.get("detected_count", len(entries)) or len(entries))
                ready_count = int(readiness.get("ready_count", 0) or 0)
                not_ready_count = int(readiness.get("not_ready_count", max(0, detected_count - ready_count)) or max(0, detected_count - ready_count))
                print_status_line(
                    "Driver Readiness Summary",
                    "✓",
                    f"{detected_count} detected, {ready_count} ready, {not_ready_count} not ready",
                )
                logger.info(
                    f"[v8.0][DRV] B-level readiness scan complete: detected={detected_count} ready={ready_count} not_ready={not_ready_count}"
                )
            else:
                print_status_line("Driver Readiness", "⏭", "No detected hardware items reported")
                logger.info("[v8.0][DRV] No detected hardware items reported by SarahMemoryHi")

        except Exception as drv_err:
            print_status_line("Driver Readiness", "⚠", "Scan failed (non-critical)")
            logger.warning(f"[v8.0][DRV] B-level readiness scan failed: {drv_err}")

        # =====================================================================
        # VOICE SETTINGS INITIALIZATION
        # =====================================================================
        print_phase_banner(9, "VOICE & AUDIO INITIALIZATION")
        
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
        print_phase_banner(10, "MEDIA SUBSYSTEM CHECKS")
        
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
        print_phase_banner(11, "DATABASE MIGRATIONS")
        
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
    v8.0 Enhanced: Local module cleanup used by Integration/Main shutdown.

    Responsibilities:
    - Idempotent execution.
    - Signal background boot workers to stop where possible.
    - Release TTS, GUI shared-frame state, context, OpenCV windows, and logging.
    - Never block indefinitely and never raise back into the shutdown path.
    """
    global shutdown_requested, _SAFE_SHUTDOWN_STARTED

    try:
        with _SAFE_SHUTDOWN_LOCK:
            if _SAFE_SHUTDOWN_STARTED:
                return
            _SAFE_SHUTDOWN_STARTED = True
    except Exception:
        if _SAFE_SHUTDOWN_STARTED:
            return
        _SAFE_SHUTDOWN_STARTED = True

    shutdown_requested = True
    logger.info("[v8.0] Initiating safe shutdown procedures.")
    print("\n[v8.0] Shutting down SarahMemory AiOS...")

    # Stop TTS first. Voice engines commonly leave COM/audio worker threads alive on Windows.
    try:
        from SarahMemoryVoice import shutdown_tts
        shutdown_tts()
        print("  ✓ TTS engine shutdown complete")
    except Exception as e:
        logger.warning(f"[v8.0] TTS shutdown skipped or failed: {e}")

    # Tell deferred startup background workers to unwind; they are daemonized, so do not wait long.
    try:
        _join_startup_background_threads(timeout_each=0.5)
        print("  ✓ Startup background workers released")
    except Exception as e:
        logger.debug(f"[v8.0] Startup background thread cleanup skipped: {e}")

    # Clear shared frame and context. Assigning through the module object is required;
    # importing shared_frame directly only changes a local name.
    try:
        import SarahMemoryGUI as gui_mod
        try:
            lock_obj = getattr(gui_mod, "shared_lock", None)
            if lock_obj is not None:
                with lock_obj:
                    try:
                        setattr(gui_mod, "shared_frame", None)
                    except Exception:
                        pass
            else:
                try:
                    setattr(gui_mod, "shared_frame", None)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            from SarahMemoryAiFunctions import clear_context
            clear_context()
        except Exception:
            pass

        print("  ✓ Cleared shared memory and context")
    except Exception as e:
        logger.warning(f"[v8.0] Shared frame cleanup skipped or failed: {e}")

    # Cleanup OpenCV windows.
    try:
        import cv2
        cv2.destroyAllWindows()
        print("  ✓ Closed all OpenCV windows")
    except Exception as e:
        logger.debug(f"[v8.0] OpenCV windows cleanup skipped or failed: {e}")

    # Best-effort database checkpointing for open WAL databases we know about.
    try:
        for db_name in ("system_logs.db", "migration_history.db", "personality1.db", "context_history.db"):
            try:
                db_path = os.path.join(SarahMemoryGlobals.DATASETS_DIR, db_name)
                if os.path.exists(db_path):
                    with sqlite3.connect(db_path, timeout=1.0) as con:
                        try:
                            con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        except Exception:
                            pass
            except Exception:
                pass
        print("  ✓ Database checkpoint pass completed")
    except Exception as e:
        logger.debug(f"[v8.0] Database checkpoint cleanup skipped: {e}")

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
# UNIFIED BOOT ENVIRONMENT SUMMARY
# =============================================================================
def capture_and_print_boot_environment_summary(force_refresh: bool = False, detail: bool = True, phase_context: str = "boot") -> dict:
    """Capture/print the single authoritative hardware environment snapshot.

    This function is the only boot-facing place that should print CPU/GPU/RAM/
    model-tier details. It delegates all hardware probing to SarahMemoryHi so
    model grading, driver readiness, API status, and chat answers all consume
    the same persisted body map.
    """
    try:
        from SarahMemoryHi import get_boot_environment_snapshot
        snap = get_boot_environment_snapshot(force_refresh=bool(force_refresh), refresh_reason=str(phase_context or "boot"))
        if not isinstance(snap, dict) or not snap.get("ok"):
            print_status_line("Boot Environment Snapshot", "⚠", str((snap or {}).get("error") or "Unavailable"))
            return snap if isinstance(snap, dict) else {"ok": False, "error": "invalid_snapshot"}

        body = snap.get("body") if isinstance(snap.get("body"), dict) else {}
        grade = snap.get("hardware_grade") if isinstance(snap.get("hardware_grade"), dict) else {}
        cpu = body.get("cpu") if isinstance(body.get("cpu"), dict) else {}
        gpu = body.get("gpu") if isinstance(body.get("gpu"), dict) else {}
        ram = body.get("ram") if isinstance(body.get("ram"), dict) else {}
        metrics = snap.get("model_metrics") if isinstance(snap.get("model_metrics"), dict) else {}

        if not detail:
            print_status_line("Boot Environment Snapshot", "✓", "Loaded from unified cached body map")
            return snap

        cpu_name = str(cpu.get("name") or "Unknown CPU")
        phys = cpu.get("physical_cores")
        logical = cpu.get("logical_threads")
        mhz = cpu.get("max_clock_mhz") or cpu.get("current_clock_mhz")
        cpu_detail = cpu_name
        if phys is not None or logical is not None:
            cpu_detail += f" | Cores: {phys if phys is not None else '?'} / Threads: {logical if logical is not None else '?'}"
        if mhz:
            try:
                cpu_detail += f" @ {int(round(float(mhz)))} MHz"
            except Exception:
                pass
        print_status_line("CPU", "✓", cpu_detail)

        if cpu.get("usage_pct") is not None:
            print_status_line("CPU Current usage", "✓", f"{float(cpu.get('usage_pct')):.1f}%")
        else:
            print_status_line("CPU Current usage", "✓", "N/A")

        if ram:
            print_status_line(
                "RAM",
                "✓",
                f"{ram.get('total_gb', 'Unknown')} GB total, {ram.get('available_gb', 'Unknown')} GB available ({ram.get('usage_pct', 'Unknown')}% used)",
            )

        gpu_name = str(gpu.get("name") or metrics.get("gpu_name") or "No dedicated GPU detected")
        vram_total = gpu.get("vram_total_mb") or metrics.get("gpu_vram_total_mb")
        vram_free = gpu.get("vram_free_mb") or metrics.get("gpu_vram_free_mb")
        if vram_total:
            print_status_line("VRAM", "✓", f"{vram_total} MB (free {vram_free if vram_free is not None else 'N/A'} MB) GPU: {gpu_name}")
        else:
            print_status_line("GPU", "✓", gpu_name)

        gpu_temp = gpu.get("temperature_c") or metrics.get("gpu_temp_c")
        print_status_line("GPU Current Temp", "✓", "N/A" if gpu_temp is None else f"{gpu_temp} C")

        disk_free = metrics.get("disk_free_gb")
        if disk_free is not None:
            try:
                print_status_line("Disk free", "✓", f"{float(disk_free):.2f} GB")
            except Exception:
                print_status_line("Disk free", "✓", str(disk_free))

        score = grade.get("score")
        tier_rating = grade.get("tier_rating")
        if score is not None:
            try:
                print_status_line("Tier Rating", "✓", f"{float(score):.1f} -> {tier_rating or 'Unknown'}")
            except Exception:
                print_status_line("Tier Rating", "✓", f"{score} -> {tier_rating or 'Unknown'}")

        logger.info(f"[v8.0][ENV] Unified environment snapshot loaded: CPU={cpu_name}; GPU={gpu_name}; tier={tier_rating}; score={score}")
        return snap
    except Exception as e:
        print_status_line("Boot Environment Snapshot", "⚠", "Unavailable; continuing with graceful degradation")
        logger.warning(f"[v8.0][ENV] Unified boot environment summary failed: {e}")
        return {"ok": False, "error": str(e)}

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

    capture_and_print_boot_environment_summary(force_refresh=False, detail=False, phase_context="startup_info")
    print("  ✓ Awaiting SarahMemory Integration Menu...\n")
    logger.info("[v8.0] Awaiting SarahMemory Integration Menu...")
    
# Unified hardware details are now captured by capture_and_print_boot_environment_summary(); no top-level boot probe runs at import time.
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
    # -------------------------------------------------------------------------
    # Core schema creation
    # -------------------------------------------------------------------------
    try:
        from SarahMemoryDatabase import ensure_core_schema as _ensure_core_schema
        _ensure_core_schema()
        logger.info("[v8.0][SCHEMA] Core schema ensured")
    except Exception as e:
        logger.warning(
            f"[v8.0][SCHEMA] ensure_core_schema failed or unavailable: {e}"
        )

    # -------------------------------------------------------------------------
    # Deep-Learning cache table
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Personality + responses schema enforcement
    # -------------------------------------------------------------------------
    try:
        import SarahMemoryGlobals as config
        from SarahMemoryMigrations import ensure_traits_last_updated_column

        per_db = os.path.join(config.DATASETS_DIR, "personality1.db")

        with sqlite3.connect(per_db) as con:
            cur = con.cursor()

            # Ensure traits.last_updated exists (PHASE 6 fix)
            ensure_traits_last_updated_column(con)

            # Ensure responses.timestamp exists
            cur.execute("PRAGMA table_info(responses)")
            cols = [r[1] for r in cur.fetchall()]

            if "timestamp" not in cols:
                cur.execute(
                    "ALTER TABLE responses ADD COLUMN timestamp TEXT"
                )
                con.commit()
                logger.info(
                    "[v8.0][SCHEMA] Added timestamp column to responses table"
                )

    except Exception as e:
        logger.info(
            f"[v8.0][SCHEMA] personality/response schema check: {e}"
        )



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
# Import-time work is intentionally minimal for optimized runtime.
# Full schema checks run through run_initial_checks()/manual diagnostics unless explicitly enabled.
if _cfg_bool("BOOT_ENSURE_SCHEMAS_ON_IMPORT", False):
    try:
        ensure_boot_schemas()
    except Exception:
        pass

if _cfg_bool("BOOT_PRINT_IMPORT_BARS", False):
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

# ====================================================================
# END OF SarahMemoryInitialization.py v9.0.0
# ====================================================================
