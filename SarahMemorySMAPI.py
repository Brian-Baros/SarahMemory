"""--==The SarahMemory Project==--
File: SarahMemorySMAPI.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-12
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
SarahMemorySMAPI - System Management & Introspection API (v8.0)
===============================================================================

This module exposes a *safe*, *auditable* Python API for internal components
(AiFunctions, Synapes, Diagnostics, Optimization, Voice, Avatar, Web API, etc.)
to inspect and adjust SarahMemory's runtime configuration and health.

GOLDEN RULES:
- No direct mutation of globals without passing through SAFE_MODE checks.
- All important changes should be logged via AiFunctions / system logs.
- This module must NEVER break boot – failures degrade gracefully to logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
import importlib
from typing import Any, Dict, List, Optional, Callable

# Assume logging is configured globally as app_logger in app.py
logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_global_config_value(key: str, default: Any = None) -> Any:
    """Helper to safely get a value from SarahMemoryGlobals."""
    try:
        import SarahMemoryGlobals as G  # type: ignore
        return getattr(G, key, default)
    except Exception as e:
        logger.debug(f"[SMAPI] Could not retrieve '%s' from SarahMemoryGlobals: %s", key, e)
        return default


def _get_module_function(module_name: str, func_name: str) -> Optional[Callable[..., Any]]:
    """Helper to safely import a module and get a callable function."""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name, None)
        if callable(func):
            return func
        logger.warning("[SMAPI] Function '%s' not callable in module '%s'.", func_name, module_name)
        return None
    except ImportError:
        logger.debug("[SMAPI] Module '%s' not found.", module_name)
        return None
    except Exception as e:
        logger.error("[SMAPI] Error accessing function '%s' in module '%s': %s",
                     func_name, module_name, e)
        return None


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for path if it does not exist."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        logger.debug("[SMAPI] Failed to ensure directory for %s: %s", path, e)


# =============================================================================
# CORE SMAPI CLASS
# =============================================================================

class SarahMemorySMAPI:
    """
    System Management API.

    Lightweight, dependency-safe interface for:
    - System status snapshots
    - Settings read/write with SAFE_MODE enforcement
    - Logging of configuration / state changes
    """

    def __init__(self) -> None:
        # Cache paths, preferably from SarahMemoryGlobals if available
        base_data = _get_global_config_value("DATA_DIR", os.path.join(os.getcwd(), "data"))
        self.DATA_DIR = base_data
        self.SETTINGS_FILE = os.path.join(self.DATA_DIR, "settings", "settings.json")

        # v8.0: Default location for AiFunctions / system-log DBs
        datasets_dir = _get_global_config_value(
            "DATASETS_DIR",
            os.path.join(self.DATA_DIR, "memory", "datasets")
        )
        self.DATASETS_DIR = datasets_dir
        self.AI_FUNCTIONS_DB_PATH = os.path.join(datasets_dir, "functions.db")

        logger.debug("[SMAPI] Initialized with DATA_DIR=%s DATASETS_DIR=%s",
                     self.DATA_DIR, self.DATASETS_DIR)

    # -------------------------------------------------------------------------
    # SYSTEM STATUS
    # -------------------------------------------------------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """
        Provides a high-level snapshot of the system's operational status.

        This call should NEVER raise; on error it returns a minimal status
        with an 'error' field instead.
        """
        status: Dict[str, Any] = {
            "project_version": _get_global_config_value("PROJECT_VERSION", "Unknown"),
            "node_name": _get_global_config_value("NODE_NAME", "SarahMemoryNode"),
            "run_mode": _get_global_config_value("RUN_MODE", "local"),
            "safe_mode": _get_global_config_value("SAFE_MODE", False),
            "local_only_mode": _get_global_config_value("LOCAL_ONLY_MODE", False),
            "current_time": time.time(),
            "core_modules_status": {},
        }

        try:
            status["core_modules_status"] = {
                "AiFunctions": bool(_get_module_function("SarahMemoryAiFunctions", "route_intent_response")),
                "Database": bool(_get_module_function("SarahMemoryDatabase", "init_database")),
                "Ledger": bool(_get_module_function("SarahMemoryLedger", "top_nodes")),
                "Voice": bool(_get_module_function("SarahMemoryVoice", "list_voices")),
                "Avatar": bool(_get_module_function("UnifiedAvatarController", "get_panel_api")),
                "Synapes": bool(_get_module_function("SarahMemorySynapes", "get_status")),
                "Diagnostics": bool(_get_module_function("SarahMemoryDiagnostics", "run_all_checks")),
                "Optimization": bool(_get_module_function("SarahMemoryOptimization", "get_runtime_status")),
            }

            # Extend with diagnostics summary if available
            diagnostics_runner = _get_module_function("SarahMemoryDiagnostics", "run_all_checks")
            if diagnostics_runner:
                try:
                    diag_results = diagnostics_runner()
                    if isinstance(diag_results, dict):
                        status["diagnostics"] = diag_results
                    else:
                        status["diagnostics"] = {"raw": diag_results}
                except Exception as e:
                    logger.error("[SMAPI] Error running Diagnostics via SMAPI: %s", e)
                    status.setdefault("errors", []).append(f"diagnostics: {e}")

        except Exception as e:
            logger.error("[SMAPI] get_system_status failed: %s", e, exc_info=True)
            status.setdefault("errors", []).append(str(e))

        return status

    # -------------------------------------------------------------------------
    # USER SETTINGS
    # -------------------------------------------------------------------------

    def _load_settings(self) -> Dict[str, Any]:
        """Internal helper to load settings.json (never raises)."""
        try:
            if not os.path.exists(self.SETTINGS_FILE):
                return {}
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            logger.error("[SMAPI] Error loading settings file '%s': %s",
                         self.SETTINGS_FILE, e)
            return {}

    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """Internal helper to save settings.json (never raises to caller)."""
        try:
            _ensure_parent_dir(self.SETTINGS_FILE)
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("[SMAPI] Error saving settings file '%s': %s",
                         self.SETTINGS_FILE, e)
            return False

    def get_user_setting(self, key: str) -> Any:
        """Retrieves a user setting by key; returns None if missing or on error."""
        try:
            settings = self._load_settings()
            return settings.get(key)
        except Exception as e:
            logger.error("[SMAPI] Error getting user setting '%s': %s", key, e)
            return None

    def set_user_setting(self, key: str, value: Any) -> bool:
        """
        Sets a user setting and logs the change to AiFunctions/system logs.

        SAFE_MODE:
            - If SAFE_MODE is True, setting changes are rejected.
        """
        if _get_global_config_value("SAFE_MODE", False):
            logger.warning("[SMAPI] Attempted to set setting '%s' in SAFE_MODE. Operation denied.", key)
            return False

        old_value = self.get_user_setting(key)
        try:
            settings = self._load_settings()
            settings[key] = value

            if not self._save_settings(settings):
                return False

            # Log the change (best-effort)
            self._log_ai_change(
                subsystem="configuration",
                change_type="set_user_setting",
                target=key,
                old_value=old_value,
                new_value=value,
            )
            return True
        except Exception as e:
            logger.error("[SMAPI] Error setting user setting '%s' to '%s': %s", key, value, e)
            return False

    # -------------------------------------------------------------------------
    # LOGGING HOOKS (AiFunctions / system logs)
    # -------------------------------------------------------------------------

    def _log_ai_change(
        self,
        subsystem: str,
        change_type: str,
        target: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """
        Internal function to log changes into AiFunctions/system logs.

        Priority:
            1) Use SarahMemoryAiFunctions.log_ai_functions_event if available.
            2) Fallback to SarahMemoryDatabase.log_ai_functions_event if present.
            3) Fallback to simple logger.info.
        """
        try:
            payload = {
                "subsystem": subsystem,
                "change_type": change_type,
                "target": target,
                "old_value": old_value,
                "new_value": new_value,
                "ts": time.time(),
            }
            details = json.dumps(payload, ensure_ascii=False)

            # 1) Preferred: AiFunctions logging
            ai_log = _get_module_function("SarahMemoryAiFunctions", "log_ai_functions_event")
            if ai_log:
                ai_log("smapi_change", details)
                return

            # 2) Fallback: Database logging helper (legacy)
            db_log = _get_module_function("SarahMemoryDatabase", "log_ai_functions_event")
            if db_log:
                db_log("smapi_change", details)
                return

            # 3) Final fallback: standard logger
            logger.info("[SMAPI-CHANGE] %s", details)

        except Exception as e:
            # Never let logging failures bubble up
            logger.error("[SMAPI] Failed to log AI change: %s", e, exc_info=True)


# Instantiate the API for use by other modules
sm_api = SarahMemorySMAPI()

# ---------------------------------------------------------------------
# Example (DOCUMENTATION ONLY) of integration with SarahMemoryAiFunctions
# ---------------------------------------------------------------------
#
# from SarahMemorySMAPI import sm_api
#
# def route_intent_response(user_input: str, ...) -> Any:
#     # ... intent classification ...
#     if intent == "query_system_status":
#         return sm_api.get_system_status()
#     elif intent == "change_setting":
#         key = extract_key(user_input)
#         value = extract_value(user_input)
#         if sm_api.set_user_setting(key, value):
#             return f"Setting '{key}' updated to '{value}'."
#         else:
#             return f"Failed to update setting '{key}'."
#     # ... other logic ...
#