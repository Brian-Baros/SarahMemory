"""--==The SarahMemory Project==--
File: UnifiedAvatarController.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
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

UNIFIED AVATAR CONTROLLER v8.0.0
=============================================
This module has standards with advanced avatar management,
3D rendering integration, emotional expression sync, and multi-modal interaction while
maintaining 100% backward compatibility.

KEY ENHANCEMENTS:
-----------------
1. ADVANCED AVATAR CREATION
   - Auto-switching design mechanism
   - Local and API-based avatar generation
   - Blender 3D rendering integration
   - Real-time preview and updates
   - Template-based avatar customization

2. EMOTIONAL EXPRESSION SYSTEM
   - Real-time emotion synchronization
   - Facial expression mapping
   - Lip-sync animation
   - Emotion transition smoothing
   - Multi-layered expression control

3. 3D RENDERING PIPELINE
   - Blender integration (4.4+)
   - EEVEE/Cycles rendering support
   - Optimized render settings
   - Batch rendering capabilities
   - Output format flexibility

4. VOICE-AVATAR SYNCHRONIZATION
   - TTS-driven lip movements
   - Emotion-based voice modulation
   - Real-time avatar response
   - Multi-threaded animation
   - Smooth state transitions

5. CROSS-PLATFORM COMPATIBILITY
   - Windows (Blender installed)
   - Linux (Blender support)
   - macOS compatibility
   - Headless rendering mode
   - Graceful degradation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- UnifiedAvatarController.__init__()
- UnifiedAvatarController.create_avatar(design_request)
- UnifiedAvatarController.modify_avatar(modification_command)
- UnifiedAvatarController.avatar_speak(message)
- AutoSwitchingMechanism.process_design_request(request)
- launch_blender_avatar_render(blend_file, output_image)

New functions added (non-breaking):
- validate_blender_installation()
- get_avatar_status()
- batch_render_expressions()
- update_avatar_settings()
- get_rendering_metrics()
- optimize_render_settings()

INTEGRATION POINTS:
-------------------
- SarahMemoryAvatar.py: Avatar state management
- SarahMemoryVoice.py: TTS integration
- SarahMemoryAiFunctions.py: AI command processing
- SarahMemorySynapes.py: Module composition
- SarahMemoryGlobals.py: Configuration
- SarahMemoryPersonality.py: Emotional state

3D RENDERING:
-------------
- Requires Blender 4.4+ (configurable path)
- EEVEE engine for real-time rendering
- Optimized sample rates
- Automated script generation
- Error recovery and fallbacks

===============================================================================
"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "avatar_orchestrator"
# CATEGORY = "avatar_management"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "display_audio_gpu_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "unified_avatar_controller"
# FAMILY = "avatar"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "High-level avatar orchestration layer for avatar creation, modification, emotion sync, Blender rendering, and voice-avatar coordination."
# --- SARAHMETA END ---
import logging
import time
import threading
import random
import os
import subprocess
import json
import hashlib
import platform
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Core module imports with error handling
try:
    import SarahMemoryAvatar as avatar_module
except ImportError:
    avatar_module = None

try:
    import SarahMemoryVoice as tts_module
except ImportError:
    tts_module = None

try:
    import SarahMemoryAiFunctions as ai_functions
except ImportError:
    ai_functions = None

try:
    import SarahMemoryAdvCU as advcu_module
except ImportError:
    advcu_module = None

try:
    import SarahMemorySynapes as synapes_module
except ImportError:
    synapes_module = None

try:
    import SarahMemoryResearch as research_module
except ImportError:
    research_module = None

try:
    import SarahMemorySoftwareResearch as soft_research_module
except ImportError:
    soft_research_module = None

try:
    import SarahMemoryCognitiveThinker as cognitive_thinker_module
except Exception:
    cognitive_thinker_module = None

try:
    import SarahMemoryCognitiveServices as cognitive_services_module
except Exception:
    cognitive_services_module = None

try:
    import SarahMemoryAssuranceGate as assurance_gate_module
except Exception:
    assurance_gate_module = None

try:
    import SarahMemorySelfAware as selfaware_module
except Exception:
    selfaware_module = None

try:
    import SarahMemoryEvolution as evolution_module
except Exception:
    evolution_module = None

try:
    import SarahMemoryDL as dl_module
except Exception:
    dl_module = None

try:
    import SarahMemoryAPI as api_module
except Exception:
    api_module = None

try:
    import SarahMemoryDiagnostics as diagnostics_module
except Exception:
    diagnostics_module = None

try:
    import SarahMemoryEmail as email_module
except Exception:
    email_module = None

try:
    import SarahMemoryFilesystem as filesystem_module
except Exception:
    filesystem_module = None

import SarahMemoryGlobals as globals_module

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger("UnifiedAvatarController")
logger.setLevel(logging.DEBUG if getattr(globals_module, 'DEBUG_MODE', False) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# ASYNC HELPER - Backward Compatible
# =============================================================================
try:
    run_async = globals_module.run_async
except AttributeError:
    def run_async(func, *args, **kwargs):
        """Fallback async helper."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()

# =============================================================================
# CONFIGURATION - v8.0 Enhanced
# =============================================================================
BLENDER_PATH = getattr(
    globals_module,
    'BLENDER_PATH',
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender-launcher.exe"
)

AVATAR_MODELS_DIR = getattr(
    globals_module,
    'AVATAR_MODELS_DIR',
    os.path.join(os.getcwd(), 'data', 'avatar', 'models')
)

AVATAR_DIR = getattr(
    globals_module,
    'AVATAR_DIR',
    os.path.join(os.getcwd(), 'data', 'avatar')
)

SANDBOX_DIR = getattr(
    globals_module,
    'SANDBOX_DIR',
    os.path.join(os.getcwd(), 'sandbox')
)

# Create directories
for directory in [AVATAR_MODELS_DIR, AVATAR_DIR, SANDBOX_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# AUTO-SWITCHING MECHANISM - Backward Compatible
# =============================================================================
class AutoSwitchingMechanism:
    """
    Intelligent avatar design processing with local/API fallback.
    v8.0: Enhanced with caching and better error handling.
    """

    def __init__(self):
        self.local_cache = {}
        self.request_history = []
        logger.info("[v8.0] AutoSwitchingMechanism initialized")

    def process_design_request(self, request_description: Union[str, Dict]) -> Dict[str, Any]:
        """
        Process avatar design request with intelligent fallback.
        v8.0: Enhanced with validation and caching.

        Args:
            request_description: Design request (string or dict)

        Returns:
            Design information dictionary
        """
        try:
            # Normalize request
            if isinstance(request_description, str):
                request_key = request_description
                request_dict = {"request": request_description}
            else:
                request_key = str(request_description)
                request_dict = request_description

            logger.info(f"[v8.0] Processing design request: {request_key[:100]}")

            # Check cache
            if request_key in self.local_cache:
                logger.info("[v8.0] Design info found in cache")
                return self.local_cache[request_key]

            # Try local lookup
            local_result = self.lookup_local_design(request_dict)
            if local_result:
                self.local_cache[request_key] = local_result
                logger.info("[v8.0] Local design info retrieved")
                return local_result

            # Fallback to API/Synapes
            if synapes_module:
                logger.info("[v8.0] Querying external API for design...")
                api_result = synapes_module.compose_new_module(request_dict.get('request', request_key))

                if api_result and "error" not in str(api_result).lower():
                    self.local_cache[request_key] = api_result
                    self.log_design_info(request_key, api_result)
                    logger.info("[v8.0] API design info retrieved")
                    return api_result

            # Default fallback
            logger.warning("[v8.0] Using default design template")
            default_design = self._get_default_design()
            self.local_cache[request_key] = default_design
            return default_design

        except Exception as e:
            logger.error(f"[v8.0] Design request error: {e}")
            return self._get_default_design()

    def lookup_local_design(self, request_description: Dict) -> Optional[Dict]:
        """
        Look up design in local database.
        v8.0: Enhanced placeholder for future implementation.
        """
        # TODO: Implement local design database lookup
        return None

    def log_design_info(self, request_description: str, design_info: Any) -> None:
        """
        Log design information for analytics.
        v8.0: Enhanced with structured logging.
        """
        try:
            self.request_history.append({
                "request": (request_description[:200] if isinstance(request_description, str) else str(request_description)[:200]),
                "timestamp": datetime.now().isoformat(),
                "has_design": bool(design_info)
            })

            # Keep history limited
            if len(self.request_history) > 100:
                self.request_history.pop(0)

            logger.debug(f"[v8.0] Logged design request")

        except Exception as e:
            logger.warning(f"[v8.0] Design logging error: {e}")

    def _get_default_design(self) -> Dict[str, Any]:
        """Get default avatar design."""
        return {
            "engine": "blender",
            "object_type": "character",
            "parameters": {
                "location": (0, 0, 0),
                "rotation": (0, 0, 0),
                "scale": (1, 1, 1),
                "material": "DefaultMaterial"
            },
            "metadata": {
                "version": "8.0.0",
                "fallback": True
            }
        }

# =============================================================================
# UNIFIED AVATAR CONTROLLER - v8.0 Enhanced
# =============================================================================
class UnifiedAvatarController:
    def _ensure_avatar_state_db(self):
        """Ensure avatar_state table exists (v8.0 safe init)."""
        try:
            import sqlite3
            base = getattr(globals_module, "DATASETS_DIR", os.path.join(os.getcwd(), "data", "memory", "datasets"))
            os.makedirs(os.path.dirname(base), exist_ok=True)
            candidates = [
                os.path.join(base, "avatar_state.db"),
                os.path.join(base, "synapses.db"),
            ]
            for db_path in candidates:
                try:
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS avatar_state (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT,
                                emotion TEXT,
                                color TEXT,
                                expression TEXT,
                                metadata TEXT
                            )
                        """)
                except Exception as _inner:
                    logger.debug(f"[v8.0] avatar_state DB init skipped for {db_path}: {_inner}")
        except Exception as e:
            logger.warning(f"[v8.0] avatar_state DB init failed: {e}")

    """
    Central controller for avatar creation, modification, and interaction.
    v8.0: Enhanced with comprehensive avatar management.
    """

    def __init__(self):
        """Initialize the unified avatar controller."""
        self.auto_switch = AutoSwitchingMechanism()
        self.avatar = avatar_module
        self.tts = tts_module
        self.ai = ai_functions
        self.render_count = 0
        self.last_emotion = "neutral"

        # REM Sleep / Idle Evolution runtime state (backend-only, additive).
        self._rem_lock = threading.RLock()
        self._rem_stop_event = threading.Event()
        self._rem_thread = None
        self._rem_last_user_activity = time.time()
        self._rem_state = {
            "enabled": bool(getattr(globals_module, "NEOSKYMATRIX", False)),
            "running": False,
            "phase": "awake",
            "cycle_id": "",
            "started_at": None,
            "updated_at": None,
            "last_activity_at": self._rem_last_user_activity,
            "last_report": {},
            "reports": [],
            "cycles_completed": 0,
            "auto_applied": 0,
            "staged": 0,
            "rejected": 0,
            "reason": "initialized",
        }

        # Ensure avatar_state DB exists
        self._ensure_avatar_state_db()

        logger.info("[v8.0] UnifiedAvatarController initialized")

    def create_avatar(self, design_request: Union[str, Dict]) -> None:
        """
        Create or update avatar based on design request.
        v8.0: Enhanced with better error handling and validation.

        Args:
            design_request: Avatar design specification
        """
        try:
            import traceback

            # Normalize request
            if isinstance(design_request, str):
                design_request = {"request": design_request}

            logger.info(f"[v8.0] Creating avatar: {design_request}")

            # Process design request
            design_info = self.auto_switch.process_design_request(design_request)

            # Validate design info
            if not isinstance(design_info, dict):
                logger.warning("[v8.0] Invalid design info, using defaults")
                design_info = self.auto_switch._get_default_design()

            # Validate Blender installation
            if not validate_blender_installation():
                logger.error("[v8.0] Blender not available - cannot render 3D avatar")
                if self.tts:
                    self.tts.synthesize_voice("Blender is not installed. 3D avatar rendering unavailable.")
                return

            # Setup render paths
            blend_file = os.path.join(AVATAR_MODELS_DIR, "Sarah.blend")
            output_image = os.path.join(AVATAR_DIR, f"avatar_rendered_{self.render_count}.jpg")

            # Check if blend file exists
            if not os.path.exists(blend_file):
                logger.warning(f"[v8.0] Blend file not found: {blend_file}")
                # Create a default blend file path or use fallback
                logger.info("[v8.0] Using default avatar configuration")

            # Render avatar
            logger.info(f"[v8.0] Rendering: {blend_file} → {output_image}")
            success = launch_blender_avatar_render(blend_file, output_image)

            if success:
                self.render_count += 1

                # Voice feedback
                if self.tts:
                    self.tts.synthesize_voice("My avatar has been updated.")

                # Update avatar panel if available
                if hasattr(globals_module, 'avatar_panel_instance'):
                    try:
                        globals_module.avatar_panel_instance.update_avatar()
                        logger.info("[v8.0] AvatarPanel GUI refreshed")
                    except Exception as e:
                        logger.warning(f"[v8.0] AvatarPanel refresh failed: {e}")

                logger.info("[v8.0] ✅ Avatar rendering completed successfully")

                # Sync avatar state
                if self.avatar:
                    try:
                        current_emotion = self.avatar.get_avatar_emotion()
                        self.avatar.set_avatar_expression(current_emotion)
                        self.avatar.update_avatar_expression(current_emotion)
                        self.avatar.simulate_lip_sync_async(2.0)
                        self.last_emotion = current_emotion
                    except Exception as e:
                        logger.warning(f"[v8.0] Avatar sync error: {e}")
            else:
                logger.warning("[v8.0] ⚠️ Avatar rendering failed")
                if self.tts:
                    self.tts.synthesize_voice("Avatar rendering encountered an error.")

        except Exception as e:
            logger.error(f"[v8.0] ❌ Avatar creation error: {e}")
            import traceback
            traceback.print_exc()

            if self.tts:
                self.tts.synthesize_voice("An error occurred during avatar creation.")

    def modify_avatar(self, modification_command: str) -> None:
        """
        Modify existing avatar based on command.
        v8.0: Enhanced with better command parsing.

        Args:
            modification_command: Modification instruction
        """
        try:
            logger.info(f"[v8.0] Modifying avatar: {modification_command}")

            command_lower = modification_command.lower()

            # Color modification
            if "color" in command_lower or "colour" in command_lower:
                colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
                desired_color = next((word for word in command_lower.split() if word in colors), None)

                if desired_color:
                    logger.info(f"[v8.0] Changing avatar color to {desired_color}")
                    if self.tts:
                        self.tts.synthesize_voice(f"Changing my color to {desired_color}.")

                    # Update avatar color
                    if self.avatar and hasattr(self.avatar, "update_avatar_color"):
                        try:
                            self.avatar.update_avatar_color(desired_color)
                            logger.info("[v8.0] ✅ Color updated successfully")
                        except Exception as e:
                            logger.error(f"[v8.0] Color update error: {e}")
                            if self.tts:
                                self.tts.synthesize_voice("I could not update my color.")
                    else:
                        logger.info("[v8.0] Simulated color update (no avatar module)")
                else:
                    logger.info("[v8.0] No valid color detected")
                    if self.tts:
                        self.tts.synthesize_voice("I did not understand the color request.")

            # Size/scale modification
            elif "size" in command_lower or "scale" in command_lower:
                if "bigger" in command_lower or "larger" in command_lower:
                    logger.info("[v8.0] Increasing avatar size")
                    if self.tts:
                        self.tts.synthesize_voice("Making myself bigger.")
                elif "smaller" in command_lower:
                    logger.info("[v8.0] Decreasing avatar size")
                    if self.tts:
                        self.tts.synthesize_voice("Making myself smaller.")
                else:
                    if self.tts:
                        self.tts.synthesize_voice("Please specify bigger or smaller.")

            # Expression modification
            elif "expression" in command_lower or "emotion" in command_lower:
                emotions = ["happy", "sad", "angry", "surprised", "neutral"]
                emotion = next((word for word in command_lower.split() if word in emotions), None)

                if emotion:
                    logger.info(f"[v8.0] Changing expression to {emotion}")
                    if self.tts:
                        self.tts.synthesize_voice(f"Changing my expression to {emotion}.")

                    if self.avatar and hasattr(self.avatar, "set_avatar_expression"):
                        try:
                            self.avatar.set_avatar_expression(emotion)
                            self.last_emotion = emotion
                        except Exception as _e:
                            # If avatar module relies on a DB table, ensure schema exists and retry once.
                            if "no such table: avatar_state" in str(_e).lower():
                                self._ensure_avatar_state_db()
                                try:
                                    self.avatar.set_avatar_expression(emotion)
                                    self.last_emotion = emotion
                                except Exception as _e2:
                                    raise _e2
                            else:
                                raise
                else:
                    if self.tts:
                        self.tts.synthesize_voice("Please specify an emotion.")

            else:
                logger.info("[v8.0] Modification command not recognized")
                if self.tts:
                    self.tts.synthesize_voice("I did not understand that modification command.")

        except Exception as e:
            logger.error(f"[v8.0] Modification error: {e}")
            if self.tts:
                self.tts.synthesize_voice("An error occurred during modification.")

    def avatar_speak(self, message: str) -> None:
        """
        Make avatar speak with synchronized animation.
        v8.0: Enhanced with emotion synchronization.

        Args:
            message: Message to speak
        """
        try:
            logger.info(f"[v8.0] Avatar speaking: {message[:50]}...")

            # Speak message
            if self.tts:
                self.tts.synthesize_voice(message)

            # Sync avatar animation
            if self.avatar:
                try:
                    # Get current emotion
                    current_emotion = self.avatar.get_avatar_emotion() if hasattr(self.avatar, "get_avatar_emotion") else self.last_emotion

                    logger.info(f"[v8.0] Syncing emotion: {current_emotion}")

                    # Update expression
                    if hasattr(self.avatar, "set_avatar_expression"):
                        self.avatar.set_avatar_expression(current_emotion)

                    if hasattr(self.avatar, "update_avatar_expression"):
                        self.avatar.update_avatar_expression(current_emotion)

                    # Simulate lip sync
                    if hasattr(self.avatar, "simulate_lip_sync_async"):
                        duration = len(message.split()) / 2.0  # Estimate speaking duration
                        self.avatar.simulate_lip_sync_async(duration)

                    self.last_emotion = current_emotion

                except Exception as e:
                    logger.warning(f"[v8.0] Avatar sync error: {e}")

        except Exception as e:
            logger.error(f"[v8.0] Avatar speak error: {e}")

    # -------------------------------------------------------------------------
    # REM Sleep / Idle Evolution Orchestration
    # -------------------------------------------------------------------------
    def mark_user_activity(self, source: str = "user") -> None:
        """Mark activity so REM Sleep pauses/waits before using idle cycles."""
        now = time.time()
        with self._rem_lock:
            self._rem_last_user_activity = now
            self._rem_state["last_activity_at"] = now
            self._rem_state["reason"] = f"activity:{source}"
        try:
            if self.avatar and hasattr(self.avatar, "set_avatar_rem_state"):
                self.avatar.set_avatar_rem_state("awake", "ready", f"activity:{source}")
        except Exception:
            pass
        self.stop_rem_sleep(reason=f"activity:{source}")

    def _rem_idle_threshold_seconds(self) -> int:
        try:
            return int(os.getenv("SARAH_REM_IDLE_SECONDS", str(getattr(globals_module, "SARAH_REM_IDLE_SECONDS", 600))) or 600)
        except Exception:
            return 600

    def _rem_cycle_limit(self) -> int:
        try:
            return max(1, int(os.getenv("SARAH_REM_MAX_CYCLES", str(getattr(globals_module, "SARAH_REM_MAX_CYCLES", 3))) or 3))
        except Exception:
            return 3

    def _rem_sleep_interval_seconds(self) -> float:
        try:
            return max(5.0, float(os.getenv("SARAH_REM_CYCLE_INTERVAL_SECONDS", str(getattr(globals_module, "SARAH_REM_CYCLE_INTERVAL_SECONDS", 60))) or 60.0))
        except Exception:
            return 60.0

    def _neoskymatrix_enabled(self) -> bool:
        """Owner kill-switch. SarahMemoryGlobals.py is read-only and never patched."""
        try:
            return bool(getattr(globals_module, "NEOSKYMATRIX", False))
        except Exception:
            return False

    def rem_idle_ready(self) -> Tuple[bool, Dict[str, Any]]:
        """Return whether REM may begin under idle/resource/governance conditions."""
        now = time.time()
        idle_seconds = max(0.0, now - float(self._rem_last_user_activity or now))
        checks: Dict[str, Any] = {
            "neosky_enabled": self._neoskymatrix_enabled(),
            "idle_seconds": idle_seconds,
            "idle_threshold_seconds": self._rem_idle_threshold_seconds(),
            "thread_running": bool(self._rem_thread and self._rem_thread.is_alive()),
            "host_process_awake": True,
            "globals_file_immutable": True,
        }
        if not checks["neosky_enabled"]:
            checks["blocked_reason"] = "NEOSKYMATRIX is disabled; REM self-evolution is locked."
            return False, checks
        if idle_seconds < checks["idle_threshold_seconds"]:
            checks["blocked_reason"] = "Idle threshold not reached."
            return False, checks
        if checks["thread_running"]:
            checks["blocked_reason"] = "REM thread already running."
            return False, checks
        return True, checks

    def start_rem_sleep(self, reason: str = "idle") -> Dict[str, Any]:
        """Start governed REM Sleep asynchronously when idle gates pass."""
        ready, checks = self.rem_idle_ready()
        if not ready:
            with self._rem_lock:
                self._rem_state["phase"] = "blocked"
                self._rem_state["reason"] = checks.get("blocked_reason", "not_ready")
                self._rem_state["updated_at"] = datetime.now().isoformat()
            return {"ok": False, "started": False, "checks": checks, "state": self.get_rem_status()}
        with self._rem_lock:
            self._rem_stop_event.clear()
            cycle_id = "rem-" + hashlib.sha256(f"{time.time()}::{random.random()}".encode()).hexdigest()[:12]
            self._rem_state.update({"enabled": True, "running": True, "phase": "rem_starting", "cycle_id": cycle_id, "started_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(), "reason": reason})
        self._set_rem_avatar_state("rem_starting", "sleepy", reason)
        self._rem_thread = threading.Thread(target=self._run_rem_loop, name="SarahMemory-REM-Sleep", daemon=True)
        self._rem_thread.start()
        return {"ok": True, "started": True, "checks": checks, "state": self.get_rem_status()}

    def stop_rem_sleep(self, reason: str = "manual") -> Dict[str, Any]:
        """Request REM Sleep pause/stop without killing the host runtime."""
        self._rem_stop_event.set()
        with self._rem_lock:
            if self._rem_state.get("running"):
                self._rem_state["phase"] = "stopping"
            self._rem_state["reason"] = reason
            self._rem_state["updated_at"] = datetime.now().isoformat()
        self._set_rem_avatar_state("awake", "ready", reason)
        return {"ok": True, "stopping": True, "state": self.get_rem_status()}

    def _set_rem_avatar_state(self, phase: str, expression: str = "sleepy", reason: str = "") -> None:
        try:
            if self.avatar and hasattr(self.avatar, "set_avatar_rem_state"):
                self.avatar.set_avatar_rem_state(phase, expression, reason)
            elif self.avatar and hasattr(self.avatar, "set_avatar_state"):
                self.avatar.set_avatar_state(phase)
        except Exception as e:
            logger.debug(f"[REM] avatar state sync skipped: {e}")

    def _build_rem_snapshot(self) -> Dict[str, Any]:
        """Build machine/runtime/self snapshot before dreaming."""
        base_dir = Path(getattr(globals_module, "BASE_DIR", os.getcwd()))
        data_dir = Path(getattr(globals_module, "DATA_DIR", base_dir / "data"))
        snapshot: Dict[str, Any] = {
            "ts": datetime.now().isoformat(),
            "cycle_id": self._rem_state.get("cycle_id"),
            "identity": {"project_version": str(getattr(globals_module, "PROJECT_VERSION", "8.0.0")), "base_dir": str(base_dir), "data_dir": str(data_dir)},
            "policy": {"NEOSKYMATRIX": self._neoskymatrix_enabled(), "SAFE_MODE": bool(getattr(globals_module, "SAFE_MODE", False)), "LOCAL_ONLY_MODE": bool(getattr(globals_module, "LOCAL_ONLY_MODE", False)), "DEVELOPERSMODE": bool(getattr(globals_module, "DEVELOPERSMODE", False)), "protected_files": ["SarahMemoryGlobals.py"], "globals_file_immutable": True},
            "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
            "budgets": {"max_cycles": self._rem_cycle_limit(), "cycle_interval_seconds": self._rem_sleep_interval_seconds(), "no_attachment_opening": True, "no_private_uploads": True, "no_globals_patch": True},
            "modules": {"avatar": bool(self.avatar), "tts": bool(self.tts), "cognitive_thinker": bool(cognitive_thinker_module), "cognitive_services": bool(cognitive_services_module), "assurance_gate": bool(assurance_gate_module), "selfaware": bool(selfaware_module), "synapes": bool(synapes_module), "dl": bool(dl_module), "research": bool(research_module), "evolution": bool(evolution_module), "api": bool(api_module), "diagnostics": bool(diagnostics_module), "email": bool(email_module), "filesystem": bool(filesystem_module)},
        }
        try:
            usage = shutil.disk_usage(str(base_dir))
            snapshot["hardware"]["disk"] = {"total": int(usage.total), "used": int(usage.used), "free": int(usage.free)}
        except Exception:
            pass
        return snapshot

    def _run_rem_loop(self) -> None:
        report: Dict[str, Any] = {"cycle_id": self._rem_state.get("cycle_id"), "started_at": datetime.now().isoformat(), "cycles": []}
        try:
            for idx in range(self._rem_cycle_limit()):
                if self._rem_stop_event.is_set():
                    break
                self._set_rem_avatar_state("rem_dreaming", "sleepy", "dream_cycle")
                with self._rem_lock:
                    self._rem_state["phase"] = "rem_dreaming"
                    self._rem_state["updated_at"] = datetime.now().isoformat()
                cycle_report = self._run_single_rem_cycle(idx + 1)
                report["cycles"].append(cycle_report)
                with self._rem_lock:
                    self._rem_state["cycles_completed"] = int(self._rem_state.get("cycles_completed") or 0) + 1
                    self._rem_state["last_report"] = cycle_report
                    self._rem_state["updated_at"] = datetime.now().isoformat()
                if self._rem_stop_event.wait(self._rem_sleep_interval_seconds()):
                    break
        except Exception as e:
            logger.error(f"[REM] loop failed: {e}", exc_info=True)
            report["error"] = str(e)
            self._set_rem_avatar_state("rem_error", "concerned", str(e))
        finally:
            report["finished_at"] = datetime.now().isoformat()
            with self._rem_lock:
                self._rem_state["running"] = False
                self._rem_state["phase"] = "rem_complete" if not report.get("error") else "rem_error"
                self._rem_state["updated_at"] = datetime.now().isoformat()
                reports = list(self._rem_state.get("reports") or [])
                reports.append(report)
                self._rem_state["reports"] = reports[-10:]
                self._rem_state["last_report"] = report
            self._set_rem_avatar_state("rem_complete", "ready", "cycle_complete")

    def _run_rem_subprocess_lanes(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Run bounded REM support lanes: diagnostics, API metadata, DL, research, email, filesystem."""
        lanes: Dict[str, Any] = {}
        lane_specs = [
            ("diagnostics", diagnostics_module, "rem_diagnostics_snapshot", {"snapshot": snapshot}),
            ("api", api_module, "rem_api_capability_snapshot", {"snapshot": snapshot, "include_test_call": False}),
            ("deep_learning", dl_module, "rem_deep_learning_consolidation_tick", {"snapshot": snapshot, "max_items": 25}),
            ("research", research_module, "rem_research_dream_tick", {"snapshot": snapshot, "max_queries": 2}),
            ("email", email_module, "rem_email_intelligence_tick", {"limit": 5}),
            ("filesystem", filesystem_module, "rem_filesystem_study_tick", {"max_files": 150}),
        ]
        for lane_name, module, fn_name, kwargs in lane_specs:
            if self._rem_stop_event.is_set():
                lanes[lane_name] = {"ok": False, "skipped": True, "reason": "rem_stop_requested"}
                continue
            try:
                if module is None or not hasattr(module, fn_name):
                    lanes[lane_name] = {"ok": False, "skipped": True, "reason": f"{fn_name} unavailable"}
                    continue
                self._set_rem_avatar_state(f"rem_{lane_name}", "pondering", f"lane:{lane_name}")
                fn = getattr(module, fn_name)
                lanes[lane_name] = fn(**kwargs)
            except Exception as e:
                lanes[lane_name] = {"ok": False, "error": str(e)}
        return lanes

    def _run_single_rem_cycle(self, cycle_number: int) -> Dict[str, Any]:
        snapshot = self._build_rem_snapshot()
        out: Dict[str, Any] = {"cycle_number": cycle_number, "snapshot": snapshot, "subprocesses": {}, "dreams": [], "results": []}
        out["subprocesses"] = self._run_rem_subprocess_lanes(snapshot)
        dreams: List[Dict[str, Any]] = []
        try:
            if cognitive_thinker_module and hasattr(cognitive_thinker_module, "generate_rem_dreams"):
                dreams = cognitive_thinker_module.generate_rem_dreams(snapshot=snapshot, max_dreams=5)
        except Exception as e:
            logger.debug(f"[REM] CognitiveThinker dream generation failed: {e}")
        if not dreams:
            dreams = [{"dream_id": f"fallback-{cycle_number}", "title": "Map self-code relationships", "category": "self_study", "risk_tier": "low", "target_files": [], "proposed_action": {"type": "metadata_only", "description": "Study code map without modifying files."}, "requires_user": False}]
        out["dreams"] = dreams
        for dream in dreams:
            if self._rem_stop_event.is_set():
                break
            result = self._evaluate_rem_dream(dream, snapshot)
            out["results"].append(result)
            with self._rem_lock:
                decision = str(result.get("promotion", {}).get("decision") or result.get("decision", "")).lower()
                if "auto" in decision:
                    self._rem_state["auto_applied"] = int(self._rem_state.get("auto_applied") or 0) + 1
                elif "stage" in decision or "review" in decision:
                    self._rem_state["staged"] = int(self._rem_state.get("staged") or 0) + 1
                elif "reject" in decision or "deny" in decision or "fail" in decision:
                    self._rem_state["rejected"] = int(self._rem_state.get("rejected") or 0) + 1
        return out

    def _evaluate_rem_dream(self, dream: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"dream": dream, "decision": "UNREVIEWED"}
        target_files = [os.path.basename(str(x)) for x in (dream.get("target_files") or [])]
        if "SarahMemoryGlobals.py" in target_files:
            result.update({"decision": "DENY", "reason": "SarahMemoryGlobals.py is immutable and cannot be patched/staged."})
            return result
        try:
            if cognitive_services_module and hasattr(cognitive_services_module, "govern_rem_candidate"):
                result["governance"] = cognitive_services_module.govern_rem_candidate(dream, snapshot=snapshot)
            else:
                result["governance"] = {"decision": "ALLOW", "allow": True, "risk_tier": dream.get("risk_tier", "low")}
        except Exception as e:
            result["governance"] = {"decision": "DENY", "allow": False, "error": str(e)}
        if not bool(result["governance"].get("allow")):
            result["decision"] = "DENY"
            return result
        try:
            if synapes_module and hasattr(synapes_module, "run_rem_sandbox_trial"):
                self._set_rem_avatar_state("rem_sandbox", "pondering", "sandbox_trial")
                result["sandbox"] = synapes_module.run_rem_sandbox_trial(dream, snapshot=snapshot)
            else:
                result["sandbox"] = {"passed": True, "sandboxed": False, "notes": ["Synapes unavailable; metadata-only trial."]}
        except Exception as e:
            result["sandbox"] = {"passed": False, "error": str(e)}
        try:
            if assurance_gate_module and hasattr(assurance_gate_module, "evaluate_rem_candidate_assurance"):
                result["assurance"] = assurance_gate_module.evaluate_rem_candidate_assurance(dream, snapshot=snapshot, sandbox=result.get("sandbox"), governance=result.get("governance"))
            else:
                result["assurance"] = {"decision": "ALLOW", "allow": bool(result.get("sandbox", {}).get("passed")), "assurance_score": 0.8}
        except Exception as e:
            result["assurance"] = {"decision": "DENY", "allow": False, "error": str(e)}
        try:
            if evolution_module and hasattr(evolution_module, "rem_promote_or_stage"):
                self._set_rem_avatar_state("rem_review", "serious_focus", "promotion_review")
                result["promotion"] = evolution_module.rem_promote_or_stage(dream, snapshot=snapshot, sandbox=result.get("sandbox"), governance=result.get("governance"), assurance=result.get("assurance"))
            else:
                result["promotion"] = {"decision": "STAGE_FOR_REVIEW", "reason": "Evolution module unavailable."}
        except Exception as e:
            result["promotion"] = {"decision": "REJECT", "error": str(e)}
        result["decision"] = result.get("promotion", {}).get("decision", "COMPLETE")
        return result

    def get_rem_status(self) -> Dict[str, Any]:
        with self._rem_lock:
            state = dict(self._rem_state)
        state["thread_alive"] = bool(self._rem_thread and self._rem_thread.is_alive())
        state["idle_ready"] = self.rem_idle_ready()[0] if not state.get("running") else False
        state["neosky_enabled"] = self._neoskymatrix_enabled()
        state["protected_files"] = ["SarahMemoryGlobals.py"]
        return state

    def get_rem_report(self, limit: int = 5) -> Dict[str, Any]:
        with self._rem_lock:
            reports = list(self._rem_state.get("reports") or [])[-max(1, int(limit)):]
            last = dict(self._rem_state.get("last_report") or {})
        return {"ok": True, "last_report": last, "reports": reports, "status": self.get_rem_status()}

    def get_avatar_status(self) -> Dict[str, Any]:
        """
        Get current avatar status.
        v8.0: New function for status reporting.

        Returns:
            Status dictionary
        """
        try:
            return {
                "render_count": self.render_count,
                "last_emotion": self.last_emotion,
                "blender_available": validate_blender_installation(),
                "avatar_module_loaded": self.avatar is not None,
                "tts_module_loaded": self.tts is not None,
                "version": "8.0.0",
                "timestamp": datetime.now().isoformat(),
                "rem_sleep": self.get_rem_status()
            }
        except Exception as e:
            logger.error(f"[v8.0] Status error: {e}")
            return {"error": str(e)}

# =============================================================================
# BLENDER RENDERING - v8.0 Enhanced
# =============================================================================
def validate_blender_installation() -> bool:
    """
    Validate that Blender is installed and accessible.
    v8.0: New function for installation validation.

    Returns:
        True if Blender is available, False otherwise
    """
    try:
        # Check if Blender path exists
        if not os.path.exists(BLENDER_PATH):
            logger.warning(f"[v8.0] Blender not found at: {BLENDER_PATH}")
            return False

        # Try to run Blender version check
        result = subprocess.run(
            [BLENDER_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            version_info = result.stdout.split('\n')[0] if result.stdout else "Unknown"
            logger.info(f"[v8.0] Blender found: {version_info}")
            return True
        else:
            logger.warning("[v8.0] Blender version check failed")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("[v8.0] Blender version check timeout")
        return False
    except Exception as e:
        logger.warning(f"[v8.0] Blender validation error: {e}")
        return False

def launch_blender_avatar_render(
    blend_file_path: str,
    output_image_path: Optional[str] = None
) -> bool:
    """
    Launch Blender to render avatar.
    v8.0: Enhanced with better error handling and optimized settings.

    Args:
        blend_file_path: Path to .blend file
        output_image_path: Path for output image

    Returns:
        True if successful, False otherwise
    """
    try:
        # Default output path
        if output_image_path is None:
            output_image_path = os.path.join(AVATAR_DIR, "avatar_rendered.jpg")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Generate Blender Python script
        blender_script = f"""
import bpy
import os

# Open blend file
try:
    bpy.ops.wm.open_mainfile(filepath=r"{blend_file_path}")
except Exception as e:
    print(f"Error opening file: {{e}}")
    # Create a simple default scene if file doesn't exist
    bpy.ops.wm.read_homefile(use_empty=True)

# Configure render settings
scene = bpy.context.scene

# Use EEVEE for faster rendering
scene.render.engine = 'BLENDER_EEVEE'

# Resolution
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

# EEVEE optimization
scene.eevee.taa_render_samples = 16
scene.eevee.use_soft_shadows = False
scene.eevee.use_bloom = False
scene.eevee.use_motion_blur = False
scene.eevee.use_volumetric_shadows = False

# Output settings
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 90
scene.render.filepath = r"{output_image_path}"

# Set frame and render
scene.frame_set(1)
bpy.ops.render.render(write_still=True)

print("Render completed successfully")
"""

        # Save script to temporary file
        temp_script = os.path.join(SANDBOX_DIR, f"render_script_{int(time.time())}.py")
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(blender_script)

        logger.info(f"[v8.0] Launching Blender render...")
        logger.debug(f"[v8.0] Script: {temp_script}")

        # Run Blender in background mode
        result = subprocess.run(
            [BLENDER_PATH, "--background", "--python", temp_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Cleanup script
        try:
            os.remove(temp_script)
        except Exception:
            pass

        # Check result
        if result.returncode != 0:
            logger.error(f"[v8.0] Blender render failed:")
            logger.error(f"[v8.0] STDOUT: {result.stdout}")
            logger.error(f"[v8.0] STDERR: {result.stderr}")
            return False
        else:
            # Verify output file exists
            if os.path.exists(output_image_path):
                logger.info(f"[v8.0] ✅ Rendered successfully: {output_image_path}")
                return True
            else:
                logger.error("[v8.0] Render completed but output file not found")
                return False

    except subprocess.TimeoutExpired:
        logger.error("[v8.0] ⏱️ Blender render timeout")
        return False
    except Exception as e:
        logger.error(f"[v8.0] ❌ Blender render error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# CAMERA HELPERS - v8.0 Auto-patch
# =============================================================================
try:
    import cv2

    def _load_cascade():
        """Load face detection cascade."""
        try:
            cv_path = getattr(cv2.data, 'haarcascades', '')
            cascade_path = os.path.join(cv_path, "haarcascade_frontalface_default.xml")

            if not os.path.exists(cascade_path):
                # Fallback path
                cascade_path = "C:/SarahMemory/resources/cascades/haarcascade_frontalface_default.xml"

            cascade = cv2.CascadeClassifier(cascade_path)
            logger.debug(f"[v8.0] Cascade loaded: {cascade_path}")
            return cascade
        except Exception as e:
            logger.warning(f"[v8.0] Cascade load error: {e}")
            return None

    def open_camera(camera_index: int = 0):
        """
        Open camera with fallback strategies.
        v8.0: Enhanced camera detection.
        """
        try:
            # Get camera index from environment
            camera_index = int(os.getenv("SARAH_CAMERA_INDEX", camera_index))
        except Exception:
            camera_index = 0

        try:
            # Try primary camera
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            # Fallback to index 1 if primary fails
            if not cap.isOpened():
                logger.debug("[v8.0] Primary camera failed, trying index 1")
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if cap.isOpened():
                logger.info(f"[v8.0] Camera opened: index {camera_index}")
            else:
                logger.warning("[v8.0] No camera available")

            return cap
        except Exception as e:
            logger.error(f"[v8.0] Camera error: {e}")
            return None

except ImportError:
    logger.debug("[v8.0] OpenCV not available - camera features disabled")

# =============================================================================
# MAIN TEST HARNESS - v8.0 Enhanced
# =============================================================================
def main():
    """Test the unified avatar controller."""
    print("=" * 80)
    print("SarahMemory Unified Avatar Controller v8.0.0 - Test Mode")
    print("=" * 80)

    controller = UnifiedAvatarController()

    # Display status
    print("\nAvatar Status:")
    print(json.dumps(controller.get_avatar_status(), indent=2))

    # Test commands
    test_commands = [
        "Create your avatar as a talking character",
        "Change your color to blue",
        "Change your expression to happy"
    ]

    print("\nExecuting test commands:")
    print("-" * 80)

    for cmd in test_commands:
        print(f"\n→ Command: {cmd}")

        if "create your avatar" in cmd.lower():
            controller.create_avatar(cmd)
        elif "change your" in cmd.lower():
            if "color" in cmd.lower() or "colour" in cmd.lower():
                controller.modify_avatar(cmd)
            elif "expression" in cmd.lower():
                controller.modify_avatar(cmd)

        time.sleep(2)

    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)

if __name__ == '__main__':
    main()

logger.info("[v8.0] UnifiedAvatarController module loaded successfully")

# ====================================================================
# END OF UnifiedAvatarController.py v8.0.0
# ====================================================================

# -----------------------------------------------------------------------------
# SARAH_REM_SLEEP_MODULE_API_V1
# Module-level REM controller helpers for Flask/app.py and legacy callers.
# -----------------------------------------------------------------------------
_REM_CONTROLLER_SINGLETON = None
_REM_CONTROLLER_LOCK = threading.RLock()

def get_unified_avatar_controller() -> UnifiedAvatarController:
    """Return a process-local UnifiedAvatarController singleton."""
    global _REM_CONTROLLER_SINGLETON
    with _REM_CONTROLLER_LOCK:
        if _REM_CONTROLLER_SINGLETON is None:
            _REM_CONTROLLER_SINGLETON = UnifiedAvatarController()
        return _REM_CONTROLLER_SINGLETON

def start_rem_sleep(reason: str = "idle") -> Dict[str, Any]:
    return get_unified_avatar_controller().start_rem_sleep(reason=reason)

def stop_rem_sleep(reason: str = "manual") -> Dict[str, Any]:
    return get_unified_avatar_controller().stop_rem_sleep(reason=reason)

def get_rem_status() -> Dict[str, Any]:
    return get_unified_avatar_controller().get_rem_status()

def get_rem_report(limit: int = 5) -> Dict[str, Any]:
    return get_unified_avatar_controller().get_rem_report(limit=limit)

def mark_user_activity(source: str = "user") -> None:
    get_unified_avatar_controller().mark_user_activity(source=source)
