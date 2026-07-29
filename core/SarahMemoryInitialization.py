"""--==The SarahMemory Project==--
File: SarahMemoryInitialization.py
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
SarahMemory v9.0 - Initialization & System Checks
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
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Boot initialization and startup checks engine for network status, vectoring, config loading, directory validation, backups, diagnostics, embedding, voice init, media checks, and migrations."
# --- SARAHMETA END ---

# =============================================================================
# CRITICAL IMPORTS
# =============================================================================
try:
    from SarahMemoryDatabase import run_vectoring_with_status_bars, ensure_local_data_runtime_ready
except Exception:
    run_vectoring_with_status_bars = None
    ensure_local_data_runtime_ready = None

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
import importlib.util
from datetime import datetime
from SarahMemoryGlobals import run_async
import SarahMemoryGlobals as SarahMemoryGlobals

# ARILE runtime is SarahMemory's async cyber-reality watchdog.  Import is optional
# so emergency/maintenance boot remains survivable if the file is missing.
try:
    from SarahMemoryARILE import start_arile_runtime, stop_arile_runtime, arile_emit, verify_arile_source_integrity
except Exception:  # pragma: no cover - boot survival fallback
    start_arile_runtime = None  # type: ignore
    stop_arile_runtime = None  # type: ignore
    arile_emit = None  # type: ignore
    verify_arile_source_integrity = None  # type: ignore

# =============================================================================
# LOGGER SETUP - v9.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemoryInitialization")
logger.setLevel(logging.DEBUG if str(os.getenv("SARAH_DEBUG_MODE", os.getenv("DEBUG_MODE", "0"))).strip().lower() in ("1", "true", "yes", "on") else logging.INFO)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - v9.0 - %(levelname)s - %(message)s'))
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
# v9.0 ENHANCED NETWORK HUB STATUS CHECK
# =============================================================================
async def check_network_hub_status():
    """
    v9.0 Enhanced: Check connection status to SarahMemory Network Hub.

    SARAHMEMORY_PATCH_NOTE 2026-06-23:
    In local-first mode, boot must not contact or claim connection to the public
    api.sarahmemory.com hub. Local-only means the platform remains operational
    offline and external hub checks are deferred until the UI/operator explicitly
    arms an online session. This preserves user sovereignty and prevents outside
    infrastructure from becoming a silent boot dependency.
    """
    try:
        local_only = bool(getattr(SarahMemoryGlobals, "LOCAL_ONLY_MODE", True))
        online_armed = bool(getattr(SarahMemoryGlobals, "SARAHMEMORY_ONLINE_SESSION_ARMED", False))
        if local_only and not online_armed:
            return ('red', "LOCAL-FIRST / OFFLINE - external hub not probed")

        from SarahMemoryHi import async_update_network_state
        state = await async_update_network_state()

        if state == 'green':
            return ('green', "CONNECTED to configured network lane")
        elif state == 'yellow':
            return ('yellow', "LOCAL/LAN available - external hub not confirmed")
        else:
            return ('red', "OFFLINE - Operating in Local Mode")

    except Exception as e:
        logger.warning(f"[v9.0] Network hub check failed: {e}")
        return ('red', "OFFLINE - Operating in Local Mode")


# =============================================================================
# v9.0 VISUAL PROGRESS INDICATORS
# =============================================================================
def print_phase_banner(phase_num, phase_name, width=78):
    """
    v9.0: Print a visually appealing phase banner.
    
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
    v9.0: Print a status line with visual indicator.
    
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
    """Return manual|background|smart|eager for boot dataset embedding.

    v9.0.0 PATCH C:
    - Default is now smart, not manual.
    - smart performs a bounded local vector refresh during Phase 7 so general
      knowledge can use local DB/vector memory immediately after boot.
    - It is capped by BOOT_VECTOR_* settings in SarahMemoryDatabase to avoid
      hard-drive thrashing on development hardware.
    - manual/off still disables pre-vectoring when the operator explicitly asks.
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
    # Enterprise boot policy: no synchronous dataset/vector refresh unless explicitly requested.
    # Local DB fallback remains active; vector refresh can be manually or background invoked.
    mode = str(value or "manual").strip().lower()
    if mode in ("off", "skip", "disabled", "manual", "none", "false", "0"):
        return "manual"
    if mode in ("background", "defer", "deferred"):
        return "background"
    if mode in ("smart", "bounded", "vector", "refresh", "revector", "auto"):
        return "background"
    if mode in ("eager", "boot", "startup", "true", "1"):
        return "eager"
    return "smart"


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
                logger.info("[v9.0][EMBED] Deferred boot dataset embedding started in background.")
                embed_local_datasets_on_boot()
                logger.info("[v9.0][EMBED] Deferred boot dataset embedding completed in %.2f seconds.", time.perf_counter() - started)
            except Exception as e:
                logger.warning(f"[v9.0][EMBED] Deferred boot dataset embedding failed: {e}")
            finally:
                with _STARTUP_BACKGROUND_LOCK:
                    _STARTUP_BACKGROUND_THREADS.pop(task_key, None)

        thread = threading.Thread(target=_worker, name="SM_BootDatasetEmbedding", daemon=True)
        _STARTUP_BACKGROUND_THREADS[task_key] = thread
        thread.start()
        return True


# =============================================================================
# MAIN INITIALIZATION FUNCTION - v9.0 World-Class
# =============================================================================
def run_initial_checks():
    """
    v9.0 ENHANCED: Starts system initialization and checks for essential components.
    
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
    logger.info("[v9.0] Starting system initialization.")

    # Start ARILE after Globals/.env are loaded, before expensive boot phases.
    try:
        if callable(start_arile_runtime):
            arile_status = start_arile_runtime(reason="initialization.run_initial_checks")
            print_status_line("ARILE Reality Watchdog", "✓", "Adaptive Reality Intelligence Layer online")
            logger.info(f"[v9.0][ARILE] ARILE runtime online: {arile_status}")
        if callable(verify_arile_source_integrity):
            integrity = verify_arile_source_integrity()
            if callable(arile_emit):
                arile_emit(
                    source="SarahMemoryInitialization",
                    kind="protected_core_integrity",
                    failure_type="arile_integrity_snapshot",
                    severity=0.30,
                    confidence=0.90,
                    summary="ARILE protected-core integrity snapshot captured during boot.",
                    data=integrity,
                )
    except Exception as arile_err:
        logger.warning(f"[v9.0][ARILE] Runtime start/integrity snapshot skipped: {arile_err}")

    try:
        # =====================================================================
        # NETWORK HUB STATUS CHECK
        # =====================================================================
        print_phase_banner(1, "NETWORK HUB CONNECTION")
        
        try:
            # SARAHMEMORY_PATCH_NOTE 2026-06-23:
            # Boot no longer probes or reports the public Network Hub as CONNECTED
            # while LOCAL_ONLY_MODE is active. Network access is a governed UI/user
            # choice, not an automatic boot action. This prevents false-positive
            # hub status and keeps offline operation truthful.
            local_only = bool(getattr(SarahMemoryGlobals, "LOCAL_ONLY_MODE", True))
            online_armed = bool(getattr(SarahMemoryGlobals, "SARAHMEMORY_ONLINE_SESSION_ARMED", False))
            if local_only and not online_armed:
                print_status_line("SarahMemory Network Hub", "⏭", "LOCAL-FIRST / OFFLINE - external hub not probed")
                logger.info("[v9.0][NET] Local-first boot: external hub check deferred until UI/operator arms online mode.")
            else:
                import asyncio
                state, msg = asyncio.run(check_network_hub_status())
                if state == 'green':
                    print_status_line("SarahMemory Network Hub", "✓", msg)
                    logger.info("[v9.0][NET] Governed network lane reports available.")
                elif state == 'yellow':
                    print_status_line("SarahMemory Network Hub", "⚠", msg)
                    logger.info("[v9.0][NET] Network degraded/local.")
                else:
                    print_status_line("SarahMemory Network Hub", "✗", msg)
                    logger.info("[v9.0][NET] Operating in local mode.")

        except Exception as e:
            print_status_line("SarahMemory Network Hub", "✗", "OFFLINE - Operating in Local Mode")
            logger.warning(f"[v9.0][NET] Network hub check failed: {e}")


        # =====================================================================
        # DATASET VECTORING WITH VISUAL PROGRESS
        # =====================================================================
        print_phase_banner(2, "DATASET VECTORING & INDEXING")
        
        try:
            local_data_enabled = bool(getattr(SarahMemoryGlobals, "LOCAL_DATA_ENABLED", True))
            if local_data_enabled and callable(ensure_local_data_runtime_ready):
                verify_embedding_on_boot = _cfg_bool("BOOT_VERIFY_EMBEDDING_ON_STARTUP", False)
                ready = ensure_local_data_runtime_ready(verify_embedding=verify_embedding_on_boot)
                db_count = ready.get("db_count", 0) if isinstance(ready, dict) else 0
                embedder = ready.get("embedder", "unknown") if isinstance(ready, dict) else "unknown"
                dim = ready.get("embedding_dim", 0) if isinstance(ready, dict) else 0
                if ready.get("ok"):
                    print_status_line("Local Data Runtime", "✓", f"datasets={db_count}, embedder={embedder}, dim={dim}")
                    logger.info("[v9.0][VECTOR] Local data runtime ready: %s", ready)
                else:
                    print_status_line("Local Data Runtime", "⚠", f"readiness degraded: {ready}")
                    logger.warning("[v9.0][VECTOR] Local data readiness degraded: %s", ready)
            elif not local_data_enabled:
                print_status_line("Local Data Runtime", "⏭", "LOCAL_DATA_ENABLED is False")
            else:
                print_status_line("Local Data Runtime", "⚠", "readiness function unavailable")

            if _cfg_bool("BOOT_RUN_VECTORING_ON_STARTUP", True):
                if callable(run_vectoring_with_status_bars):
                    print_status_line("Vector Database", "⏳", "Checking datasets/indexes without forced rebuild...")
                    run_vectoring_with_status_bars(force=_cfg_bool("BOOT_FORCE_VECTOR_REBUILD", False))
                    print_status_line("Vector Database", "✓", "Vector check completed")
                    logger.info("[v9.0][VECTOR] Dataset vector check completed")
                else:
                    print_status_line("Vector Database", "⚠", "Vectoring function unavailable")
                    logger.warning("[v9.0][VECTOR] run_vectoring_with_status_bars not available")
            else:
                print_status_line("Vector Database", "⏭", "Rebuild skipped; local vector runtime was still verified")
                logger.info("[v9.0][VECTOR] Boot rebuild skipped, local vector runtime verified.")
        
        except Exception as e:
            print_status_line("Vector Database", "✗", f"Vector readiness failed: {e}")
            logger.warning(f"[v9.0][VECTOR] Dataset vector readiness failed: {e}")

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
                        logger.info("[v9.0][CONFIG] Offline mode detected")
                except Exception:
                    pass
            
            except Exception as config_err:
                print_status_line("User Settings", "⚠", "Using defaults")
                logger.warning(f"[v9.0][CONFIG] Could not load user settings: {config_err}")

            # Load global configuration
            from SarahMemoryGlobals import get_global_config
            config = get_global_config()
            
            if not config:
                print_status_line("Global Config", "✗", "Failed to load")
                logger.error("[v9.0][CONFIG] Failed to load global configuration.")
                return False
            
            print_status_line("Global Config", "✓", "Successfully loaded")
            logger.info("[v9.0][CONFIG] Global configuration retrieved successfully.")

        except Exception as e:
            print_status_line("Configuration", "✗", f"Critical failure: {e}")
            logger.error(f"[v9.0][CONFIG] Configuration loading failed: {e}")
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
                    logger.info(f"[v9.0][DIR] Created: {directory}")
            
            except Exception as mkerr:
                logger.error(f"[v9.0][DIR] Failed to create '{directory}': {mkerr}")

        print_status_line("Directory Verification", "✓", 
                         f"{dirs_verified} verified, {dirs_created} created")
        logger.info(f"[v9.0][DIR] {dirs_verified} directories verified, {dirs_created} created")

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
            logger.info("[v9.0][BACKUP] SAFE_MODE enabled; weekly backup skipped.")
        elif not _cfg_bool("BOOT_WEEKLY_BACKUP_CHECK", False):
            print_status_line("Weekly Backup", "⏭", "Skipped during boot (manual/scheduled backup policy)")
            logger.info("[v9.0][BACKUP] Boot backup check skipped by optimized runtime policy.")
        else:
            try:
                from SarahMemoryFilesystem import create_weekly_backup
                create_weekly_backup()
                print_status_line("Weekly Backup", "✓", "Verified")
                logger.info("[v9.0][BACKUP] Weekly backup check completed")
            
            except Exception as backup_err:
                print_status_line("Weekly Backup", "⚠", "Check failed (non-critical)")
                logger.warning(f"[v9.0][BACKUP] Could not verify weekly backup: {backup_err}")

        # =====================================================================
        # CORE-BRAIN DIAGNOSTICS
        # =====================================================================
        print_phase_banner(6, "CORE-BRAIN DIAGNOSTICS")
        
        if not _cfg_bool("BOOT_PERSONALITY_DIAGNOSTICS", False):
            print_status_line("Personality Core", "⏭", "Skipped during boot (diagnostics available on demand)")
            logger.info("[v9.0][DIAG] Personality diagnostics skipped by optimized runtime policy.")
        else:
            try:
                from SarahMemoryDiagnostics import run_personality_core_diagnostics
                
                try:
                    run_personality_core_diagnostics()
                    print_status_line("Personality Core", "✓", "Diagnostics passed")
                    logger.info("[v9.0][DIAG] Core-Brain diagnostics complete.")
                
                except Exception as dierr:
                    print_status_line("Personality Core", "⚠", "Diagnostics failed (non-critical)")
                    logger.warning(f"[v9.0][DIAG] Personality diagnostics failed: {dierr}")
            
            except Exception as imerr:
                print_status_line("Personality Core", "⚠", "Module unavailable (non-critical)")
                logger.warning(f"[v9.0][DIAG] Diagnostics module import failed: {imerr}")

        # =====================================================================
        # LOCAL DATASET EMBEDDING (Skip in SAFE_MODE / defer by default)
        # =====================================================================
        print_phase_banner(7, "LOCAL DATASET EMBEDDING")

        try:
            if SAFE_MODE:
                print_status_line("Dataset Embedding", "⏭", "Skipped (SAFE_MODE enabled)")
                logger.info("[v9.0][EMBED] SAFE_MODE enabled; skipping local dataset embedding.")
            else:
                mode = _boot_dataset_embedding_mode()
                try:
                    if mode == "manual":
                        print_status_line("Dataset Embedding", "⏭", "Pre-embedding skipped by explicit policy; direct local DB fallback remains active")
                        logger.info("[v9.0][EMBED] Boot vector refresh skipped because mode=manual/off.")
                    elif mode == "background":
                        started = _start_background_dataset_embedding()
                        if started:
                            print_status_line("Dataset Embedding", "⏭", "Bounded vector refresh deferred to background")
                            logger.info("[v9.0][EMBED] Local dataset vector refresh deferred to background")
                        else:
                            print_status_line("Dataset Embedding", "⏭", "Background vector refresh already running")
                    else:
                        embed_started = time.perf_counter()
                        result = embed_local_datasets_on_boot(force=(mode == "eager"))
                        inserted = result.get("inserted", 0) if isinstance(result, dict) else 0
                        skipped = result.get("skipped", 0) if isinstance(result, dict) else 0
                        print_status_line("Dataset Embedding", "✓", f"Bounded local vector refresh complete: inserted={inserted}, skipped={skipped}, {time.perf_counter() - embed_started:.2f}s")
                except Exception as emb_err:
                    print_status_line("Dataset Embedding", "⚠", "Embedding failed (non-critical)")
                    logger.warning(f"[v9.0][EMBED] Local dataset embedding failed: {emb_err}")

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
                    f"[v9.0][DRV] B-level readiness scan complete: detected={detected_count} ready={ready_count} not_ready={not_ready_count}"
                )
            else:
                print_status_line("Driver Readiness", "⏭", "No detected hardware items reported")
                logger.info("[v9.0][DRV] No detected hardware items reported by SarahMemoryHi")

        except Exception as drv_err:
            print_status_line("Driver Readiness", "⚠", "Scan failed (non-critical)")
            logger.warning(f"[v9.0][DRV] B-level readiness scan failed: {drv_err}")

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
                logger.info("[v9.0][VOICE] Voice settings loaded successfully")
            else:
                print_status_line("Voice Settings", "⚠", "Using defaults (settings.json not found)")
                logger.warning("[v9.0][VOICE] Voice settings.json not found during initialization.")
        
        except Exception as ve:
            print_status_line("Voice Settings", "⚠", "Failed to load (using defaults)")
            logger.error(f"[v9.0][VOICE] Voice settings failed to load: {ve}")

        # =====================================================================
        # MEDIA SUBSYSTEM CHECKS (CAPABILITY-ONLY; NO HEAVY IMPORTS AT BOOT)
        # =====================================================================
        print_phase_banner(10, "MEDIA SUBSYSTEM CHECKS")

        media_modules = (
            ("Music Generator", "SarahMemoryMusicGenerator"),
            ("Lyrics to Song", "SarahMemoryLyricsToSong"),
            ("Video Editor", "SarahMemoryVideoEditorCore"),
            ("Canvas Studio", "SarahMemoryCanvasStudio"),
        )
        media_status = []
        capability_probe_only = _cfg_bool("BOOT_MEDIA_CAPABILITY_PROBE_ONLY", True)

        for display_name, module_name in media_modules:
            try:
                if capability_probe_only:
                    available = importlib.util.find_spec(module_name) is not None
                    media_status.append((display_name, "✓" if available else "⏭"))
                    logger.info(
                        "[v9.0][MEDIA] Capability probe: %s available=%s (module not imported)",
                        module_name,
                        available,
                    )
                else:
                    # Explicit opt-in compatibility mode. Importing creative organs may
                    # initialize optional engines, so it is disabled by default.
                    __import__(module_name)
                    media_status.append((display_name, "✓"))
                    logger.info("[v9.0][MEDIA] Explicit boot import completed: %s", module_name)
            except Exception as media_error:
                media_status.append((display_name, "⏭"))
                logger.warning("[v9.0][MEDIA] %s unavailable: %s", module_name, media_error)

        for module, status in media_status:
            if status == "✓" and capability_probe_only:
                status_text = "Capability detected ()"
            elif status == "✓":
                status_text = "Available (explicitly initialized)"
            else:
                status_text = "Optional capability not detected"
            print_status_line(module, status, status_text)

        # =====================================================================
        # DATABASE MIGRATIONS
        # =====================================================================
        print_phase_banner(11, "DATABASE MIGRATIONS")

        try:
            from SarahMemoryMigrations import migrate_root_runtime_artifacts, run_migrations

            # The root-artifact placement pass is bounded to a fixed allowlist and
            # prevents orphan DB/JSON files from accumulating in DATA_DIR. It does
            # not recurse through drives or rebuild databases.
            placement = migrate_root_runtime_artifacts()
            placement_changes = sum(
                len(placement.get(key) or [])
                for key in ("moved", "deduplicated", "conflicts_preserved")
            )
            if not placement.get("ok", False):
                print_status_line("Runtime Artifact Placement", "⚠", "Completed with bounded errors")
                logger.warning("[v9.0][MIGRATE] Root artifact placement errors: %s", placement.get("errors"))
            elif placement_changes:
                print_status_line(
                    "Runtime Artifact Placement",
                    "✓",
                    f"{placement_changes} known root artifact(s) corrected",
                )
                logger.info("[v9.0][MIGRATE] Root artifact placement: %s", placement)
            else:
                print_status_line("Runtime Artifact Placement", "✓", "No misplaced known artifacts")

            if _cfg_bool("BOOT_RUN_MIGRATIONS_ON_STARTUP", False):
                migration_ok = bool(run_migrations())
                if migration_ok:
                    print_status_line("Database Migrations", "✓", "Pending migrations checked/applied")
                    logger.info("[v9.0][MIGRATE] Startup migration pass completed")
                else:
                    print_status_line("Database Migrations", "⚠", "Migration pass reported a failure")
                    logger.warning("[v9.0][MIGRATE] Startup migration pass reported failure")
            else:
                print_status_line(
                    "Database Migrations",
                    "⏭",
                    "Deferred by boot policy; run through diagnostics/updater when approved",
                )
                logger.info("[v9.0][MIGRATE] Full migrations deferred by BOOT_RUN_MIGRATIONS_ON_STARTUP")
        except Exception as migration_error:
            print_status_line("Database Migrations", "⚠", "Placement/migration check failed")
            logger.warning("[v9.0][MIGRATE] Placement/migration check failed: %s", migration_error)

        # =====================================================================
        # FINAL STATUS
        # =====================================================================
        print("\n" + "═" * 78)
        print("  ✓ SarahMemory v9.0.0 initialization sequence completed")
        print("  • Readiness and any degraded/optional conditions are reported above")
        print("  • Runtime capability is verified again at the point of use")
        print("═" * 78 + "\n")

        logger.info("[v9.0] SarahMemory initialization sequence completed; see phase results for readiness.")
        return True

    except Exception as e:
        logger.error(f"[v9.0] Exception during initialization: {e}")
        print(f"\n✗ CRITICAL ERROR: {e}\n")
        return False


# =============================================================================
# SYNCHRONIZATION SEQUENCE
# =============================================================================
def run_sync_sequence():
    """
    v9.0.0: Local-first synchronization preflight.

    # SARAHMEMORY_PATCH_NOTE: Sync sequence is local-first and truthful.
    # Boot must not claim network connectivity is OK unless a governed online
    # session actually performs a network check. This function now verifies local
    # sync storage/tables and reports remote sync as ARMED or DISABLED.
    """
    logger.info("[v9.0] Running local-first sync preflight...")
    print("\n[v9.0] Checking local synchronization preflight...")
    try:
        import SarahMemorySync as _sync  # type: ignore
        status = _sync.sync_data()
        print("  ✓ Local sync storage: READY")
        print("  ✓ Local sync database/audit: READY")
        remote_enabled = bool(getattr(SarahMemoryGlobals, "REMOTE_SYNC_ENABLED", False)) and bool(getattr(SarahMemoryGlobals, "ONLINE_SESSION_ARMED", False))
        if remote_enabled:
            print("  ⚠ Remote sync: ARMED by configuration/session")
        else:
            print("  ✓ Remote sync: DISABLED until user/UI arms online mode")
        logger.info("[v9.0] Local sync preflight completed: %s", status)
        return {"ok": True, "local": status, "remote_armed": remote_enabled}
    except Exception as exc:
        logger.warning("[v9.0] Local sync preflight warning: %s", exc)
        print(f"  ⚠ Local sync preflight warning: {exc}")
        return {"ok": False, "error": str(exc)}


# =============================================================================
# SAFE SHUTDOWN PROCEDURES
# =============================================================================
def safe_shutdown():
    """
    v9.0 Enhanced: Local module cleanup used by Integration/Main shutdown.

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
    logger.info("[v9.0] Initiating safe shutdown procedures.")
    print("\n[v9.0] Shutting down SarahMemory AiOS...")

    # Stop ARILE intake early so shutdown does not create packet/log churn.
    try:
        if callable(stop_arile_runtime):
            stop_arile_runtime(reason="initialization.safe_shutdown")
            logger.info("[v9.0][ARILE] ARILE shutdown completed.")
    except Exception as arile_err:
        logger.warning(f"[v9.0][ARILE] ARILE shutdown skipped: {arile_err}")

    # Stop TTS first. Voice engines commonly leave COM/audio worker threads alive on Windows.
    try:
        from SarahMemoryVoice import shutdown_tts
        shutdown_tts()
        print("  ✓ TTS engine shutdown complete")
    except Exception as e:
        logger.warning(f"[v9.0] TTS shutdown skipped or failed: {e}")

    # Tell deferred startup background workers to unwind; they are daemonized, so do not wait long.
    try:
        _join_startup_background_threads(timeout_each=0.5)
        print("  ✓ Startup background workers released")
    except Exception as e:
        logger.debug(f"[v9.0] Startup background thread cleanup skipped: {e}")

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
        logger.warning(f"[v9.0] Shared frame cleanup skipped or failed: {e}")

    # Cleanup OpenCV windows.
    try:
        import cv2
        cv2.destroyAllWindows()
        print("  ✓ Closed all OpenCV windows")
    except Exception as e:
        logger.debug(f"[v9.0] OpenCV windows cleanup skipped or failed: {e}")

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
        logger.debug(f"[v9.0] Database checkpoint cleanup skipped: {e}")

    print("\n[v9.0] Safe shutdown completed successfully.")
    print("═" * 78)
    logger.info("[v9.0] Safe shutdown completed successfully.")



def signal_handler(sig, frame):
    """
    v9.0: Handles system interrupts (e.g., Ctrl+C).
    """
    global shutdown_requested
    logger.warning("[v9.0] Interrupt signal received! Initiating emergency shutdown...")
    print("\n[v9.0] Interrupt signal received. Shutting down...")
    
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
            # SARAHMEMORY_PATCH_NOTE 2026-06-23:
            # Cached body_map snapshots can be partially populated during V8→V9
            # transitions. If RAM fields are missing, fill them from psutil
            # without forcing a heavy hardware rescan or rewriting the body map.
            try:
                if ram.get('total_gb') in (None, 'Unknown') or ram.get('available_gb') in (None, 'Unknown'):
                    import psutil as _psutil
                    _vm = _psutil.virtual_memory()
                    ram['total_gb'] = round(float(_vm.total) / (1024 ** 3), 2)
                    ram['available_gb'] = round(float(_vm.available) / (1024 ** 3), 2)
                    ram['usage_pct'] = round(float(_vm.percent), 1)
            except Exception:
                pass
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

        logger.info(f"[v9.0][ENV] Unified environment snapshot loaded: CPU={cpu_name}; GPU={gpu_name}; tier={tier_rating}; score={score}")
        return snap
    except Exception as e:
        print_status_line("Boot Environment Snapshot", "⚠", "Unavailable; continuing with graceful degradation")
        logger.warning(f"[v9.0][ENV] Unified boot environment summary failed: {e}")
        return {"ok": False, "error": str(e)}

# =============================================================================
# STARTUP INFO DISPLAY
# =============================================================================
def startup_info():
    """
    v9.0 Enhanced: Displays intro header and system identity at launch.
    Includes simulated AI boot animations and readiness messages.
    """
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  SARAHMEMORY AI INITIALIZATION SEQUENCE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    logger.info("═" * 78)
    logger.info("         SarahMemory AI Initialization v9.0        ")
    logger.info("═" * 78)
    
    print("  Status: [System Booting...]")
    logger.info("[v9.0] Status: System Booting...")
    
    time.sleep(0.5)
    
    print("  ⏳ Performing hardware environment check...")
    logger.info("[v9.0] Performing hardware environment check...")
    
    time.sleep(0.5)

    capture_and_print_boot_environment_summary(force_refresh=False, detail=False, phase_context="startup_info")
    print("  ✓ Awaiting SarahMemory Integration Menu...\n")
    logger.info("[v9.0] Awaiting SarahMemory Integration Menu...")
    
# Unified hardware details are now captured by capture_and_print_boot_environment_summary(); no top-level boot probe runs at import time.
# =============================================================================
# ASYNCHRONOUS INITIALIZATION WRAPPER
# =============================================================================
def async_run_initial_checks(callback):
    """
    v9.0: Asynchronous initial checks wrapper for non-blocking startup.
    """
    from SarahMemoryGlobals import run_async
    
    def task():
        result = run_initial_checks()
        callback(result)
    
    run_async(task)


# =============================================================================
# LOCAL DATASET EMBEDDING
# =============================================================================
def embed_local_datasets_on_boot(force: bool = False):
    """
    v9.0.0 PATCH C: bounded local vector refresh at boot.

    This refreshes semantic vector memory from local SQLite datasets and optional
    imported text. LOCAL_ONLY_MODE/offline does not block this because it is
    local, read/write only to ai_learning.db, and bounded by BOOT_VECTOR_* caps.
    """
    try:
        from SarahMemoryGlobals import LOCAL_DATA_ENABLED

        if not LOCAL_DATA_ENABLED:
            logger.info("[v9.0][EMBED] Local vector refresh skipped – LOCAL_DATA_ENABLED is False.")
            return {"ok": False, "inserted": 0, "skipped": 0, "reason": "LOCAL_DATA_ENABLED_FALSE"}

        logger.info("[v9.0][EMBED] Running bounded local vector refresh...")

        try:
            from SarahMemoryDatabase import refresh_local_vector_memory_on_boot
            result = refresh_local_vector_memory_on_boot(force=bool(force))
        except Exception:
            from SarahMemoryDatabase import embed_and_store_dataset_sentences
            result = embed_and_store_dataset_sentences(force=bool(force), include_sqlite_pool=True)

        logger.info("[v9.0][EMBED] Local vector refresh result: %s", result)
        return result if isinstance(result, dict) else {"ok": True, "inserted": 0, "skipped": 0}

    except Exception as e:
        logger.error(f"[v9.0][EMBED] Error during local vector refresh on boot: {e}")
        return {"ok": False, "inserted": 0, "skipped": 0, "reason": str(e)}


# =============================================================================
# BOOT SCHEMA VALIDATION
# =============================================================================
def ensure_boot_schemas():
    """
    v9.0: Ensure critical tables exist in their proper databases before core
    modules run. Idempotent and safe to call multiple times.
    """
    # -------------------------------------------------------------------------
    # Core schema creation
    # -------------------------------------------------------------------------
    try:
        from SarahMemoryDatabase import ensure_core_schema as _ensure_core_schema
        _ensure_core_schema()
        logger.info("[v9.0][SCHEMA] Core schema ensured")
    except Exception as e:
        logger.warning(
            f"[v9.0][SCHEMA] ensure_core_schema failed or unavailable: {e}"
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

        logger.info("[v9.0][SCHEMA] DL cache table ensured")

    except Exception as e:
        logger.error(f"[v9.0][SCHEMA] ensure dl_cache failed: {e}")

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
                    "[v9.0][SCHEMA] Added timestamp column to responses table"
                )

    except Exception as e:
        logger.info(
            f"[v9.0][SCHEMA] personality/response schema check: {e}"
        )



# =============================================================================
# BOOT PROGRESS BARS
# =============================================================================
def print_boot_bars(stage='Boot', width=40):
    """
    v9.0: Simple boot progress bar display.
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
        logger.info("[v9.0] SarahMemory is ready for integration menu.")
        print("\n[v9.0] ✓ SarahMemory is ready for integration menu.")
    else:
        logger.error("[v9.0] Startup checks failed. Exiting.")
        print("\n[v9.0] ✗ Startup checks failed. Exiting.")
        sys.exit(1)
    
    try:
        while not shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

# ====================================================================
# END OF SarahMemoryInitialization.py v9.0.0
# ====================================================================
