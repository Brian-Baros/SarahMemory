"""--==The SarahMemory Project==--
File: SarahMemoryGlobals.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-05-14
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
"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "configuration_core"
# CATEGORY = "runtime_governance"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "globals"
# FAMILY = "core_governance"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Central runtime configuration and governance plane for environment identity, modes, hardware scoring, model selection, feature flags, paths, and safety policy."
# --- SARAHMETA END ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception as e:
    print(f"[WARN] python-dotenv unavailable or failed, .env not loaded: {e}")
    
import os
import sys
import logging
import sqlite3
import csv
import glob
import json
import numpy as np
import asyncio
import aiohttp
import time
import platform
import importlib
from datetime import datetime
# Optional scheduler: if apscheduler is not installed in this environment,
# just disable scheduler-based features instead of crashing the whole app.
try:
    import apscheduler
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None  # Scheduler features are skipped if unavailable

# ---------------- Phase A1: Runtime Identity & Environment ----------------
# This section defines a small, centralized "who am I / where am I running?" layer
# so every module can reason about the current runtime without duplicating logic.

def _env_flag(name, default="false"):
    """Return True/False from an environment variable using friendly values.

    Accepts: 1, true, yes, on  (case-insensitive) as True.
    Anything else (or missing) is treated as False.
    """
    try:
        value = os.getenv(name, default)
        if value is None:
            return False
        if not isinstance(value, str):
            value = str(value)
        return value.strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        # Fail safe: never crash on env parsing; just return False.
        return False

# High-level run mode for SarahMemory core.
# - "local"  : running on a desktop / laptop (Windows/Linux/macOS)
# - "cloud"  : running on a server (PythonAnywhere, etc.)
# - "test"   : CI, diagnostics, or sandboxed runs
RUN_MODE = os.getenv("RUN_MODE", "local").strip().lower()
if RUN_MODE not in ("local", "cloud", "test"):
    RUN_MODE = "local"

# Auto-detect cloud context if RUN_MODE was not explicitly provided.
if "RUN_MODE" not in os.environ:
    try:
        host = (os.getenv("HOSTNAME") or platform.node() or "").lower()
    except Exception:
        host = ""
    if os.getenv("PYTHONANYWHERE_DOMAIN") or ".pythonanywhere.com" in host:
        RUN_MODE = "cloud"

# Device modes capture *how* the user is interacting with SarahMemory.
DEVICE_MODE_LOCAL_AGENT = "local_agent"   # Full desktop app + GUI
DEVICE_MODE_PUBLIC_WEB  = "public_web"    # Browser-based UI hitting a remote API
DEVICE_MODE_MOBILE_WEB  = "mobile_web"    # Mobile browser / embedded webview
DEVICE_MODE_HEADLESS    = "headless"      # No GUI, background/daemon mode

# Device performance profiles (coarse-grained resource envelope hints).
DEVICE_PROFILES = ("UltraLite", "Standard", "Performance")

# ============================================================================
# ENV → WINDOWS ENVIRONMENT IMPORT (STRING-ONLY, USER-CONTROLLED)
# - One-way import: .env → Windows Environment (no export path)
# - Never touches keys that exist in Windows but are absent in .env
# - Skips boolean-like values (true/false) and empty values
# - Conflict handling: if same key but different value, user chooses
# ============================================================================
def sm_open_windows_env_vars_ui() -> bool:
    """Open the Windows Environment Variables UI (best-effort)."""
    try:
        import platform, subprocess
        if platform.system().lower() != "windows":
            return False
        subprocess.Popen(["rundll32", "sysdm.cpl,EditEnvironmentVariables"], shell=False)
        return True
    except Exception:
        return False


def sm_parse_env_file(env_path: str) -> dict:
    """Parse a .env file into {KEY: VALUE}. Keeps raw strings; strips surrounding quotes."""
    out = {}
    try:
        if not env_path:
            return out
        import os
        if not os.path.isfile(env_path):
            return out
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f.read().splitlines():
                line = (raw or "").strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = (k or "").strip()
                val = (v or "").strip()
                if not key:
                    continue
                # Remove inline comments ONLY when not inside quotes
                if val and (val[0] not in ("'", '"')) and (" #" in val or "\t#" in val):
                    val = val.split("#", 1)[0].strip()
                # Strip matching quotes
                if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ("'", '"')):
                    val = val[1:-1]
                out[key] = val
    except Exception:
        return out
    return out


def _sm_is_boolish(val: str) -> bool:
    try:
        v = str(val or "").strip().lower()
        return v in ("true", "false", "1", "0", "yes", "no", "on", "off")
    except Exception:
        return False


def _sm_windows_env_read(name: str) -> tuple:
    """
    Return (user_val, system_val) for a Windows environment variable.
    Uses Registry for accuracy (not just current process env).
    """
    user_val = None
    sys_val = None
    try:
        import platform
        if platform.system().lower() != "windows":
            return (None, None)
        import winreg
        # User scope
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_READ) as k:
                user_val, _ = winreg.QueryValueEx(k, name)
        except Exception:
            user_val = None
        # System scope
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 0, winreg.KEY_READ) as k:
                sys_val, _ = winreg.QueryValueEx(k, name)
        except Exception:
            sys_val = None
    except Exception:
        return (None, None)
    return (user_val, sys_val)


def _sm_windows_env_broadcast_change() -> None:
    """Broadcast WM_SETTINGCHANGE so newly-set env vars are visible to new processes."""
    try:
        import ctypes
        from ctypes import wintypes
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        SMTO_ABORTIFHUNG = 0x0002
        SendMessageTimeoutW = ctypes.windll.user32.SendMessageTimeoutW
        SendMessageTimeoutW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
            wintypes.UINT, wintypes.UINT, ctypes.POINTER(wintypes.DWORD)
        ]
        result = wintypes.DWORD(0)
        SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0,
                            ctypes.cast(ctypes.c_wchar_p("Environment"), wintypes.LPARAM),
                            SMTO_ABORTIFHUNG, 2000, ctypes.byref(result))
    except Exception:
        pass


def _sm_windows_env_write_user(name: str, value: str) -> bool:
    """Write user-scope env var via Registry. Does not require admin."""
    try:
        import platform
        if platform.system().lower() != "windows":
            return False
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_EXPAND_SZ, str(value))
        _sm_windows_env_broadcast_change()
        return True
    except Exception:
        return False


def sm_import_env_strings_to_windows(env_path: str, parent=None) -> dict:
    """
    Import string-only env vars from .env into Windows USER environment variables.
    Returns a summary dict: {added, overwritten, kept, skipped, conflicts}.
    """
    summary = {"added": 0, "overwritten": 0, "kept": 0, "skipped": 0, "conflicts": 0}

    import platform, os
    if platform.system().lower() != "windows":
        summary["skipped"] = -1
        return summary

    env_map = sm_parse_env_file(env_path)
    if not env_map:
        return summary

    # Filter out bool-ish + empty values
    candidates = {}
    for k, v in env_map.items():
        sv = str(v or "").strip()
        if not sv:
            continue
        if _sm_is_boolish(sv):
            continue
        candidates[k] = sv

    if not candidates:
        return summary

    # UI / prompts
    use_tk = parent is not None
    if use_tk:
        try:
            from tkinter import messagebox
        except Exception:
            use_tk = False

    # Bulk decision for missing keys
    import_missing = None  # None => ask once, True/False => apply

    for key, val in sorted(candidates.items(), key=lambda kv: kv[0].lower()):
        user_val, sys_val = _sm_windows_env_read(key)
        existing = user_val if user_val is not None else sys_val

        if existing is None:
            if import_missing is None:
                msg = (
                    f"Import missing variables from .env into Windows user environment?\n\n"
                    f"Source: {env_path}\n\n"
                    f"Note: Only string values are imported. Boolean-like values are ignored.\n"
                    f"Existing Windows variables are NOT overwritten unless you approve a conflict."
                )
                if use_tk:
                    import_missing = messagebox.askyesno("Import .env → Windows Env", msg)
                else:
                    ans = input(msg + "\n\nType Y to import missing keys, anything else to skip: ").strip().lower()
                    import_missing = ans in ("y", "yes")
            if not import_missing:
                summary["skipped"] += 1
                continue

            # Confirm per-key for safety (lightweight)
            if use_tk:
                ok = messagebox.askyesno(
                    "Confirm Import",
                    f"Add Windows env variable?\n\n{key} = <hidden>\n\nProceed?"
                )
            else:
                ok = input(f"Add Windows env variable {key}? (Y/N): ").strip().lower() in ("y", "yes")
            if not ok:
                summary["skipped"] += 1
                continue

            if _sm_windows_env_write_user(key, val):
                summary["added"] += 1
            else:
                summary["skipped"] += 1
            continue

        # Existing found: if same, no-op
        if str(existing) == str(val):
            summary["kept"] += 1
            continue

        # Conflict
        summary["conflicts"] += 1
        if use_tk:
            # Yes = overwrite with .env, No = keep system, Cancel = skip
            choice = messagebox.askyesnocancel(
                "Conflict Detected",
                f"Variable already exists with a different value:\n\n"
                f"{key}\n\n"
                f"Choose YES to overwrite with .env value.\n"
                f"Choose NO to keep Windows value.\n"
                f"Choose CANCEL to skip."
            )
            if choice is None:
                summary["skipped"] += 1
                continue
            if choice is False:
                summary["kept"] += 1
                continue
            # choice True => overwrite
            if messagebox.askyesno("Confirm Overwrite", f"Overwrite Windows value for:\n\n{key}\n\nAre you sure?"):
                if _sm_windows_env_write_user(key, val):
                    summary["overwritten"] += 1
                else:
                    summary["skipped"] += 1
            else:
                summary["kept"] += 1
        else:
            ans = input(f"Conflict {key}. Overwrite with .env? (Y/N/Skip): ").strip().lower()
            if ans in ("n", "no"):
                summary["kept"] += 1
                continue
            if ans in ("s", "skip", "c", "cancel"):
                summary["skipped"] += 1
                continue
            # overwrite
            confirm = input(f"Are you sure you want to overwrite {key}? (Y/N): ").strip().lower()
            if confirm in ("y", "yes"):
                if _sm_windows_env_write_user(key, val):
                    summary["overwritten"] += 1
                else:
                    summary["skipped"] += 1
            else:
                summary["kept"] += 1

    return summary



def _detect_device_mode():
    """Infer the current device mode with optional overrides via env.

    Priority:
    1) SARAH_DEVICE_MODE env (must match one of the DEVICE_MODE_* constants)
    2) Cloud heuristics (PythonAnywhere, explicit RUN_MODE="cloud")
    3) Desktop / GUI heuristics
    4) Fallback to headless
    """
    override = os.getenv("SARAH_DEVICE_MODE", "").strip().lower()
    if override in (
        DEVICE_MODE_LOCAL_AGENT,
        DEVICE_MODE_PUBLIC_WEB,
        DEVICE_MODE_MOBILE_WEB,
        DEVICE_MODE_HEADLESS,
    ):
        return override

    try:
        host = (os.getenv("HOSTNAME") or platform.node() or "").lower()
    except Exception:
        host = ""

    # PythonAnywhere or explicit cloud mode => public web
    if os.getenv("PYTHONANYWHERE_DOMAIN") or ".pythonanywhere.com" in host or RUN_MODE == "cloud":
        return DEVICE_MODE_PUBLIC_WEB

    # If we appear to have a desktop environment, assume local agent
    if os.name == "nt" or os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
        return DEVICE_MODE_LOCAL_AGENT

    # Safe default
    return DEVICE_MODE_HEADLESS

def _detect_device_profile():
    """Infer a coarse performance profile (can be overridden from env).

    SARAH_DEVICE_PROFILE may be: UltraLite, Standard, Performance
    """
    override = os.getenv("SARAH_DEVICE_PROFILE", "").strip().title()
    if override in DEVICE_PROFILES:
        return override

    # Simple heuristic: cloud environments can usually handle more concurrency.
    if RUN_MODE == "cloud":
        return "Performance"

    return "Standard"

DEVICE_MODE = _detect_device_mode()
DEVICE_PROFILE = _detect_device_profile()

def get_runtime_meta():
    """Return a small snapshot of core runtime identity for logging / diagnostics.

    This is intentionally tiny so it can be safely serialized to logs and DB.
    """
    try:
        node_name = globals().get("NODE_NAME", platform.node() or "SarahMemoryNode")
    except Exception:
        node_name = "SarahMemoryNode"

    return {
        "project_version": PROJECT_VERSION,
        "author": AUTHOR,
        "revision_start_date": REVISION_START_DATE,
        "run_mode": RUN_MODE,
        "device_mode": DEVICE_MODE,
        "device_profile": DEVICE_PROFILE,
        "safe_mode": SAFE_MODE if "SAFE_MODE" in globals() else False,
        "local_only": LOCAL_ONLY_MODE if "LOCAL_ONLY_MODE" in globals() else False,
        "node_name": node_name,
    }


# ---------------- Global Configuration ----------------
### Static constants###
# --- Version ---
PROJECT_VERSION = "8.0.0"  # minor: updater scheduling, SR/TTS polish, research order fixes
AUTHOR = "Brian Lee Baros"
# --- Runtime/debug flags (unchanged lines may already exist above/below) ---
REVISION_START_DATE  = "03/01/2026" #Date of System Overhaul
DEBUG_MODE = True # Helps with SarahMemoryCompare and other debugging issues.
ENABLE_RESEARCH_LOGGING = True # Track Message/query of the GUI from Start to Finished Response/Reply
# This constant ensures downstream modules interpret API responses
RESEARCH_RESULT_KEY = "snippet"  # #note: Used to standardize access to results[0]['snippet']
RESEARCH_RESULT_FALLBACK = "[No valid API result parsed]"
SM_INT_MAIN_MENU = False   # "True will show Menu, False will bypass Integration Menu"
ENABLE_MINI_BROWSER = True  # safe default; prevents threaded Tk crashes
SARAH_TOTAL_MEMORY_MB = 4096 # 128mb, 256mb, 512mb, 1024=1gb, 2048 = 2gb, 4096 = 4gb, 8192 =8gb
SARAH_MEMORY_PARTITIONS= 4 # each Partition is divided into the amount of MEMORYALLOCATED therefore 4096/4 makes each Partition 1024 or 1gb each. 
SARAH_MEMORY_REFRESH_MINUTES = 5  # in Minutes 5, 10, 15, 30, 60
SARAH_MEMORY_SANDBOX_ENABLED = True


# --- Voice / Mic gating ---
# IMPORTANT: default False so the mic can listen unless we are actively speaking.
AVATAR_IS_SPEAKING = True  #True chatbot will not listen to mic and own speech echo. When set to False Ai may hear itself speak in the GUI.default True
# Optional fuzzy voice selector. If not empty, the TTS engine will pick the first installed
# voice whose name contains this substring (case-insensitive). Example: "Michone"
# VOICE_FUZZY_NAME = os.getenv("SARAHMEMORY_VOICE_FUZZY", "").strip()
VOICE_FUZZY_NAME = "Michone"
# TTS behavior
TTS_ASYNC = True           # Non-blocking speak (queue)
TTS_BLOCKING = False       # Wait for utterance to finish if True
TTS_BLOCK_TIMEOUT = 60     # Seconds (used only when blocking)
MAX_TTS_QUEUE = 10         # Backpressure limit

# Emotion prosody (override per emotion)
EMOTION_TTS_MAP = {
    "joy":      {"rate_delta": +12, "volume": 1.0},
    "trust":    {"rate_delta":  +6, "volume": 0.9},
    "surprise": {"rate_delta": +16, "volume": 1.0},
    "sadness":  {"rate_delta": -14, "volume": 0.7},
    "fear":     {"rate_delta": -6,  "volume": 0.8},
    "anger":    {"rate_delta": +10, "volume": 1.0},
    "neutral":  {"rate_delta":   0, "volume": None}
}

# NEW Global runtime safety flags (v7.1.3)
# These flags enable granular control of heavy features when running on limited resources.
# SAFE_MODE disables heavy or optional modules, leaving only core functionality active.
SAFE_MODE = _env_flag("SARAH_SAFE_MODE", "false")
LOCAL_ONLY_MODE = _env_flag("SARAH_LOCAL_ONLY_MODE", "false")  # When True, bypass all external network research and use local data only.

# SarahMemory AI-Agent may control your PC, Move,open,close,windows execute programs, and operate as if a they were a standard operator they are not allowed to delete files. or move files.

#This is a SafeGuard incase it attempts to do to much or do tasks when other issues are needing to be taken care of
AI_AGENT_RESUME_DELAY = 1000 #A time in miliseconds Delay when system is not being used to resume AI-Agent Task
USE_ADVANCED_AGENT = True  # Enable v8.0 features
AI_AGENT_ENABLED = True    # Required for agent control
CONTEXT_BUFFER_SIZE = 50   # Increase for better context
# Advanced agent
USE_ADVANCED_AGENT = True
ADVANCED_AGENT_THREADS = 4
# Performance
MAX_PARALLEL_TOOLS = 5
TOOL_TIMEOUT_SECONDS = 10
# Learning
LEARNING_RATE = 0.01
ADAPTATION_INTERVAL_SEC = 60
# Meta-cognition
CONFIDENCE_THRESHOLD = 0.7
REFLECTION_INTERVAL = 10
# Knowledge graph
KNOWLEDGE_NODE_LIMIT = 10000
EMBEDDING_DIMENSION = 64
# Prediction
PREDICTION_CONFIDENCE_MIN = 0.3
PATTERN_HISTORY_DAYS = 30
# Adjust confidence threshold MIGHT OR MIGHT NOT USE USE_ PREFIX
#USE_ADVANCED_AGENT.confidence_threshold = 0.8 
# Enable/disable parallel execution
#USE_ADVANCED_AGENT.parallel_execution_enabled = True
# Set max parallel tools
#USE_ADVANCED_AGENT.max_parallel_tools = 3
# Configure learning rate
#USE_ADVANCED_AGENT.metrics.learning_rate = 0.02

# --- Rust tokenizer auto-detect (paste directly above MODEL_CATALOG) ---
RUSTTOKEN = False
RUSTTOKEN_MODE = "auto"
RUSTTOKEN_ENTRYPOINT = None

try:
        
    # BASE_DIR is declared later in your file, so fall back safely here.
    _sm_base_dir = globals().get("BASE_DIR", os.getcwd()) or os.getcwd()
    _rust_core_dir = os.path.join(_sm_base_dir, "sarahmemory_rust_core")

    # Require the Rust project folder + source files to exist
    _rust_src_ok = (
        os.path.isdir(_rust_core_dir)
        and os.path.isfile(os.path.join(_rust_core_dir, "Cargo.toml"))
        and os.path.isdir(os.path.join(_rust_core_dir, "src"))
        and (
            os.path.isfile(os.path.join(_rust_core_dir, "src", "lib.rs"))
            or os.path.isfile(os.path.join(_rust_core_dir, "src", "main.rs"))
        )
    )

    # Look for a compiled Python extension in common maturin/pyo3 output locations
    _rust_candidate_dirs = [
        _sm_base_dir,
        _rust_core_dir,
        os.path.join(_rust_core_dir, "target", "release"),
        os.path.join(_rust_core_dir, "target", "debug"),
    ]

    _rust_binary_ok = False
    for _d in _rust_candidate_dirs:
        if not os.path.isdir(_d):
            continue

        _has_ext = (
            glob.glob(os.path.join(_d, "sarahmemory_rust_core*.pyd"))
            or glob.glob(os.path.join(_d, "sarahmemory_rust_core*.so"))
        )

        if _has_ext:
            if _d not in sys.path:
                sys.path.insert(0, _d)
            _rust_binary_ok = True

    # Final validation: import + basic tokenization smoke test
    if _rust_src_ok and _rust_binary_ok:
        _rust_mod = importlib.import_module("sarahmemory_rust_core")

        for _fn_name in ("tokenize", "fast_tokenize", "tokenize_text", "encode"):
            _fn = getattr(_rust_mod, _fn_name, None)
            if callable(_fn):
                _probe = _fn("SarahMemory tokenizer self-test")
                if _probe is not None:
                    RUSTTOKEN = True
                    RUSTTOKEN_ENTRYPOINT = _fn_name
                    break

except Exception:
    RUSTTOKEN = False
    RUSTTOKEN_ENTRYPOINT = None


# ---------------- Model Selection & Multi-Model Configuration -New for v7.0-----Allows 3rd party models to be incorporated----------
# Full Model Integration Flag
MULTI_MODEL = False  # When True, allows multiple models to be enabled and used in logic checks. If False, only DEFAULT fallback model will load.
AUTO_MODEL_SELECTOR = False # Automatic model selector flag (v7.1.3). When True, the system picks the best available model based on enabled flags.

# =============================================================================
# Model Enable Flags (Used across modules for routing, embeddings, vision, voice)
# =============================================================================
# Notes:
# - These are the legacy/manual TRUE/FALSE switches. When AUTO_MODEL_SELECTOR is True,
#   the system can still auto-pick per tier/hardware; these flags are the deterministic
#   “force allow/deny” controls that your routing logic can consult.
# - Tiers are practical guidance (LOW/MID/HIGH/BEAST) based on typical RAM/VRAM needs.
# - “<Errors>” means you previously flagged runtime issues; leave False unless revalidated.
# =============================================================================

# -----------------------------------------------------------------------------
# EMBEDDINGS / RETRIEVAL (SentenceTransformers / embedding encoders) [DEFAULTS = B,D,E,F,G]
# -----------------------------------------------------------------------------
ENABLE_MODEL_B  = True   # sentence-transformers/all-MiniLM-L6-v2 Tier: LOW | Function: Core English embeddings (default fallback); fast + reliable
ENABLE_MODEL_D  = False   # sentence-transformers/paraphrase-MiniLM-L3-v2 Tier: LOW | Function: Paraphrase/rewrites; useful for semantic similarity + rewrite tasks
ENABLE_MODEL_C  = False  # sentence-transformers/multi-qa-MiniLM-L6-cos-v1 Tier: LOW | Function: QA-optimized embeddings (question->passage retrieval)
ENABLE_MODEL_E  = False  # sentence-transformers/distiluse-base-multilingual-cased-v2 Tier: MID | Function: Multilingual embeddings (50+ languages)
ENABLE_MODEL_F  = False  # allenai/specter Tier: MID | Function: Scientific paper/document embeddings (research corpora)
ENABLE_MODEL_G  = False  # intfloat/e5-base Tier: MID | Function: High-recall retrieval embeddings; strong for search/retrieval
ENABLE_MODEL_R  = False  # BAAI/bge-base-en-v1.5 Tier: HIGH | Function: Strong English embeddings; better ranking/semantic performance
ENABLE_MODEL_U  = False  # BAAI/bge-m3  Tier: BEAST | Function: Multilingual + multi-function embeddings; heavier footprint
# -----------------------------------------------------------------------------
# LOCAL LLMs (REASONING / GENERAL CHAT) [DEFAULTS=N]
# -----------------------------------------------------------------------------
ENABLE_MODEL_A  = False  # microsoft/phi-1_5 Tier: LOW/MID | Function: Small reasoning/code-capable LLM (older Phi generation)
ENABLE_MODEL_H  = False  # microsoft/phi-2   Tier: MID | Function: Improved small reasoning LLM (successor to phi-1_5)
ENABLE_MODEL_S  = False  # microsoft/Phi-4-mini-instruct Tier: MID/HIGH | Function: Modern instruct-tuned Phi; better general reasoning + instruction following
ENABLE_MODEL_N  = True   # Qwen/Qwen3-0.6B   Tier: LOW | Function: Small local-friendly reasoning LLM (good default for “LOCAL LLM” lane)
ENABLE_MODEL_Q  = False  # Qwen/Qwen2.5-7B-Instruct  Tier: BEAST | Function: Strong general instruct LLM; higher quality, higher resource demands
ENABLE_MODEL_I  = False  # tiiuae/falcon-rw-1b Tier: LOW | Function: Lightweight general LLM; basic open model option
ENABLE_MODEL_M  = False  # TinyLlama/TinyLlama-1.1B-Chat-v1.0 Tier: LOW | Function: Ultra-light chat model for very constrained machines
ENABLE_MODEL_J  = False  # openchat/openchat-3.5-0106 Tier: HIGH | Function: Chat-style assistant; good alignment when stable locally
ENABLE_MODEL_K  = False  # NousResearch/Nous-Capybara-7B Tier: BEAST | Function: Helpful assistant-tuned model; higher resource needs
ENABLE_MODEL_L  = False  # mistralai/Mistral-7B-Instruct-v0.2 Tier: BEAST | Function: Strong generalist reasoning/instruct model
# -----------------------------------------------------------------------------
# LOCAL LLMs (CODER / SOFTWARE ENGINEERING) [DEFAULTS=O,P]
# -----------------------------------------------------------------------------
ENABLE_MODEL_O  = False # Qwen/Qwen2.5-Coder-1.5B-Instruct Tier: LOW | Function: Code assistant (small); best “coder low tier” default
ENABLE_MODEL_P  = True   # Qwen/Qwen2.5-Coder-3B-Instruct   Tier: MID | Function: Code assistant (mid); better quality than 1.5B, still local-friendly
ENABLE_MODEL_T  = False  # Qwen/Qwen2.5-Coder-7B-Instruct   Tier: BEAST | Function: Code assistant (high quality); needs more VRAM/RAM
# -----------------------------------------------------------------------------
# VISION / OBJECT DETECTION [DEFAULTS=V, AD]
# -----------------------------------------------------------------------------
ENABLE_MODEL_V  = True  # nielsr/yolov12n Tier: LOW | Function: YOLOv12 Nano (fast/efficient); good default vision primary
ENABLE_MODEL_W  = False  # ultralytics/yolov8 Tier: MID | Function: Stable YOLOv8 baseline; good compatibility fallback
ENABLE_MODEL_X  = False  # qualcomm/RF-DETR Tier: MID/HIGH | Function: DETR-style detector alternative; good secondary option
ENABLE_MODEL_Y  = False  # ultralytics/yolov8x Tier: BEAST | Function: Higher-accuracy YOLOv8 variant; heavier compute
ENABLE_MODEL_AD  = False   # SSD (pytorch-ssd) Tier: LOW/MID  | Lightweight SSD detector | Uses external weights URLs
ENABLE_MODEL_AE  = False  # YOLOv5            Tier: MID      | Legacy YOLO family       | Requires legacy loader
ENABLE_MODEL_AF  = False  # YOLOv7            Tier: MID/HIGH | Legacy YOLO family       | Requires legacy loader
ENABLE_MODEL_AG  = False  # YOLO-NAS          Tier: HIGH     | Legacy/alt detector      | Requires NAS loader
ENABLE_MODEL_AH  = False  # YOLOX             Tier: MID/HIGH | Legacy/alt detector      | Requires YOLOX loader
ENABLE_MODEL_AI  = False  # PP-YOLOv2         Tier: HIGH     | Legacy/alt detector      | Requires Paddle/bridge
ENABLE_MODEL_AJ  = False  # EfficientDet      Tier: HIGH     | Legacy/alt detector      | Requires EfficientDet loader
ENABLE_MODEL_AK  = False  # DETR              Tier: HIGH     | Legacy DETR              | Requires DETR pipeline
ENABLE_MODEL_AL  = False  # DINO              Tier: BEAST    | Transformer detector     | Heavy; requires DINO pipeline
ENABLE_MODEL_AM  = False  # CenterNet         Tier: HIGH     | Legacy detector          | Requires CenterNet loader
ENABLE_MODEL_AN  = False  # Faster R-CNN      Tier: BEAST    | Two-stage detector       | Heavy; requires torchvision pipeline
ENABLE_MODEL_AO  = False  # RetinaNet         Tier: HIGH     | One-stage detector   
#-----------------------------------------------------------------------------
# IMAGE GENERATION (DIFFUSION / TEXT-TO-IMAGE) [DEFAULT=Z]
# -----------------------------------------------------------------------------
ENABLE_MODEL_Z   = True   # black-forest-labs/FLUX.1-schnell Tier: LOW/MID | Function: Faster image generation; good “quick draft” local imagegen
ENABLE_MODEL_AA  = False  # Freepik/flux.1-lite-8B Tier: HIGH | Function: Higher quality local image generation; heavier footprint
ENABLE_MODEL_AB  = False  # black-forest-labs/FLUX.1-dev Tier: BEAST | Function: Highest tier FLUX; heavy VRAM/RAM; consider API fallback if unstable locally
# -----------------------------------------------------------------------------
# VOICE / TTS (TEXT-TO-SPEECH) [DEFAULT=AC]
# -----------------------------------------------------------------------------
ENABLE_MODEL_AC = True  # FunAudioLLM/CosyVoice2-0.5B Tier: LOW | Function: Low-latency TTS; good local voice baseline

# Central model dictionary map for iteration/logic control (accessed from other modules)
MODEL_CONFIG = {
    "phi-1_5": ENABLE_MODEL_A,
    "all-MiniLM-L6-v2": ENABLE_MODEL_B,
    "multi-qa-MiniLM": ENABLE_MODEL_C,
    "paraphrase-MiniLM-L3-v2": ENABLE_MODEL_D,
    "distiluse-multilingual": ENABLE_MODEL_E,
    "allenai-specter": ENABLE_MODEL_F,
    "e5-base": ENABLE_MODEL_G,
    "phi-2": ENABLE_MODEL_H,
    "falcon-rw-1b": ENABLE_MODEL_I,
    "openchat-3.5": ENABLE_MODEL_J,
    "Nous-Capybara-7B": ENABLE_MODEL_K,
    "Mistral-7B-Instruct-v0.2": ENABLE_MODEL_L,
    "TinyLlama-1.1B": ENABLE_MODEL_M,
    "Qwen3-0.6B": ENABLE_MODEL_N,
    "Qwen2.5-Coder-1.5B-Instruct": ENABLE_MODEL_O,
    "Qwen2.5-Coder-3B-Instruct": ENABLE_MODEL_P,
    "Qwen2.5-7B-Instruct": ENABLE_MODEL_Q,
    "BAAI/bge-base-en-v1.5": ENABLE_MODEL_R,
    # v8.1+ catalog / stack items (manual enable lane)
    "Phi-4-mini-instruct": ENABLE_MODEL_S,
    "Qwen2.5-Coder-7B-Instruct": ENABLE_MODEL_T,
    "BAAI/bge-m3": ENABLE_MODEL_U,

    # Vision
    "nielsr/yolov12n": ENABLE_MODEL_V,
    "ultralytics/yolov8": ENABLE_MODEL_W,
    "qualcomm/RF-DETR": ENABLE_MODEL_X,
    "ultralytics/yolov8x": ENABLE_MODEL_Y,

    # Image generation
    "black-forest-labs/FLUX.1-schnell": ENABLE_MODEL_Z,
    "Freepik/flux.1-lite-8B": ENABLE_MODEL_AA,
    "black-forest-labs/FLUX.1-dev": ENABLE_MODEL_AB,

    # TTS
    "FunAudioLLM/CosyVoice2-0.5B": ENABLE_MODEL_AC,
}
#(OLD v7.0.1 FLAG FOR SarahMemoryReply.py block)
BLOCK_NARRATIVE_OUTPUTS = True #Keeps AI from making Wacky story outputs, based off of information in some of the NonFineTuned Models.

# ---------------- Category-Based Model Stacking (v8.1.x) ----------------
# Mission:
# - Keep MANY downloadable models available, but ONLY use 1 model per job at a time.
# - Embeddings are the only special case: the chosen embedding model may switch by language/intent,
#   but still only ONE embedding model is used per request.
# - Backward compatible with legacy ENABLE_MODEL_* flags and MODEL_CONFIG.

MULTI_STACK_ENABLED = True  # Master switch for category-based routing/selection.

# ---- Primary stack selections (canonical Hugging Face repos) ----
REASONING_MODEL_REPO      = "Qwen/Qwen3-0.6B"                         # Logic & reasoning (small, reliable)
CODER_MODEL_REPO          = "Qwen/Qwen2.5-Coder-1.5B-Instruct"         # Coding / self-monkeypatch specialist

EMBEDDING_EN_REPO         = "sentence-transformers/all-MiniLM-L6-v2"                 # Core English retrieval
EMBEDDING_MULTI_REPO      = "sentence-transformers/distiluse-base-multilingual-cased-v2"  # Multilingual retrieval
EMBEDDING_SCI_REPO        = "allenai/specter"                                         # Scientific doc retrieval
EMBEDDING_RECALL_REPO     = "intfloat/e5-base"                                        # High-recall retrieval
EMBEDDING_PARA_REPO       = "sentence-transformers/paraphrase-MiniLM-L3-v2"           # Paraphrase/rewrites

VISION_PRIMARY_REPO       = "nielsr/yolov12n"                          # YOLOv12 Nano primary
VISION_BACKUP_REPO        = "ultralytics/yolov8"                       # Stable backup
VISION_ALT_REPO           = "qualcomm/RF-DETR"                         # Alternative backup

IMAGEGEN_MODEL_REPO       = "black-forest-labs/FLUX.1-schnell"                   # Image generation (heavy; consider API fallback)
TTS_MODEL_REPO            = "FunAudioLLM/CosyVoice2-0.5B"               # Low-latency TTS

# ---- End-user model catalog tiers (Low / Mid / High/ Beast) ----
# Keep this list short and high-signal: ~3 choices per tier.
MODEL_CATALOG = {
    "reasoning": {
        "low":   ["Qwen/Qwen3-0.6B"],
        "mid":   ["microsoft/Phi-4-mini-instruct"],
        "high":  ["microsoft/Phi-4-mini-instruct"],
        "beast": ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"],
    },
    "coder": {
        "low":   ["Qwen/Qwen2.5-Coder-1.5B-Instruct"],
        "mid":   ["Qwen/Qwen2.5-Coder-3B-Instruct"],
        "high":  ["Qwen/Qwen2.5-Coder-3B-Instruct"],
        "beast": ["Qwen/Qwen2.5-Coder-7B-Instruct"],
    },
    "embeddings": {
        "low":   ["sentence-transformers/all-MiniLM-L6-v2"],
        "mid":   [
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "intfloat/e5-base",
            "allenai/specter",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        ],
        "high":  ["BAAI/bge-base-en-v1.5"],
        "beast": ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"],
    },
    "vision": {
        "low":   ["nielsr/yolov12n", "ultralytics/yolov8"],
        "mid":   ["qualcomm/RF-DETR"],
        "high":  ["ultralytics/yolov8x"],
        "beast": ["ultralytics/yolov8x"],
    },
    "image_generation": {
        "low":   ["black-forest-labs/FLUX.1-schnell"],
        "mid":   ["Freepik/flux.1-lite-8B"],
        "high":  ["Freepik/flux.1-lite-8B"],
        "beast": ["black-forest-labs/FLUX.1-dev"],
    },
    "tts": {
        "low":   ["FunAudioLLM/CosyVoice2-0.5B"],
        "mid":   [],
        "high":  [],
        "beast": [],
    },
}

# ---- Legacy alias -> HF repo mapping (for backwards compatibility) ----
MODEL_REPO_MAP = {
    # Embeddings
    "all-MiniLM-L6-v2": EMBEDDING_EN_REPO,
    "distiluse-base-multilingual-cased-v2": EMBEDDING_MULTI_REPO,
    "distiluse-multilingual": EMBEDDING_MULTI_REPO,
    "allenai-specter": EMBEDDING_SCI_REPO,
    "intfloat/e5-base": EMBEDDING_RECALL_REPO,
    "e5-base": EMBEDDING_RECALL_REPO,
    "paraphrase-MiniLM-L3-v2": EMBEDDING_PARA_REPO,
    "multi-qa-MiniLM": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    
    # LLMs
    "phi-1_5": "microsoft/phi-1_5",
    "phi-2": "microsoft/phi-2",
    "openchat-3.5": "openchat/openchat-3.5-0106",
    "Nous-Capybara-7B": "NousResearch/Nous-Capybara-7B",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # New stack names (allow either alias or repo)
    "Qwen3-0.6B": REASONING_MODEL_REPO,
    "Qwen2.5-Coder-1.5B-Instruct": CODER_MODEL_REPO,
    "Qwen2.5-Coder-3B-Instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    # Additional aliases (v8.1+ stacks)
    "bge-m3": "BAAI/bge-m3",
    "BAAI/bge-m3": "BAAI/bge-m3",

    # Vision aliases
    "yolov12n": VISION_PRIMARY_REPO,
    "yolov8": VISION_BACKUP_REPO,
    "yolov8x": "ultralytics/yolov8x",
    "RF-DETR": VISION_ALT_REPO,

    # Image generation aliases
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
    "flux.1-lite-8B": "Freepik/flux.1-lite-8B",

    # TTS aliases
    "CosyVoice2-0.5B": TTS_MODEL_REPO,
}

def resolve_model_repo(model_id):
    """Return canonical Hugging Face repo string for a given model id/alias."""
    if not model_id:
        return ""
    return MODEL_REPO_MAP.get(model_id, model_id)


MODEL_CATEGORY_PRIMARY_ATTRS = globals().get("MODEL_CATEGORY_PRIMARY_ATTRS", {
    "reasoning": "REASONING_MODEL_REPO",
    "coder": "CODER_MODEL_REPO",
    "embeddings": "EMBEDDING_EN_REPO",
    "vision": "VISION_PRIMARY_REPO",
    "image_generation": "IMAGEGEN_MODEL_REPO",
    "tts": "TTS_MODEL_REPO",
})

MODEL_CATEGORY_FLAG_REPO_MAP = globals().get("MODEL_CATEGORY_FLAG_REPO_MAP", {
    "reasoning": {
        "ENABLE_MODEL_A": "microsoft/phi-1_5",
        "ENABLE_MODEL_H": "microsoft/phi-2",
        "ENABLE_MODEL_S": "microsoft/Phi-4-mini-instruct",
        "ENABLE_MODEL_N": "Qwen/Qwen3-0.6B",
        "ENABLE_MODEL_Q": "Qwen/Qwen2.5-7B-Instruct",
        "ENABLE_MODEL_I": "tiiuae/falcon-rw-1b",
        "ENABLE_MODEL_M": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "ENABLE_MODEL_J": "openchat/openchat-3.5-0106",
        "ENABLE_MODEL_K": "NousResearch/Nous-Capybara-7B",
        "ENABLE_MODEL_L": "mistralai/Mistral-7B-Instruct-v0.2",
    },
    "coder": {
        "ENABLE_MODEL_O": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "ENABLE_MODEL_P": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "ENABLE_MODEL_T": "Qwen/Qwen2.5-Coder-7B-Instruct",
    },
    "embeddings": {
        "ENABLE_MODEL_B": "sentence-transformers/all-MiniLM-L6-v2",
        "ENABLE_MODEL_C": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "ENABLE_MODEL_D": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "ENABLE_MODEL_E": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "ENABLE_MODEL_F": "allenai/specter",
        "ENABLE_MODEL_G": "intfloat/e5-base",
        "ENABLE_MODEL_R": "BAAI/bge-base-en-v1.5",
        "ENABLE_MODEL_U": "BAAI/bge-m3",
    },
    "vision": {
        "ENABLE_MODEL_V": "nielsr/yolov12n",
        "ENABLE_MODEL_W": "ultralytics/yolov8",
        "ENABLE_MODEL_X": "qualcomm/RF-DETR",
        "ENABLE_MODEL_Y": "ultralytics/yolov8x",
        "ENABLE_MODEL_AD": "qfgaohao/pytorch-ssd",
    },
    "image_generation": {
        "ENABLE_MODEL_Z": "black-forest-labs/FLUX.1-schnell",
        "ENABLE_MODEL_AA": "Freepik/flux.1-lite-8B",
        "ENABLE_MODEL_AB": "black-forest-labs/FLUX.1-dev",
    },
    "tts": {
        "ENABLE_MODEL_AC": "FunAudioLLM/CosyVoice2-0.5B",
    },
})

MODEL_MANAGER_CATEGORY_LABELS = globals().get("MODEL_MANAGER_CATEGORY_LABELS", {
    "reasoning": "Reasoning",
    "coder": "Coder",
    "embeddings": "Embeddings",
    "vision": "Vision",
    "image_generation": "Image Generation",
    "tts": "Voice / TTS",
})

MODEL_MANAGER_CUSTOM_OPTION = "<CUSTOM REPO>"
CUSTOM_MODEL_REPOS = globals().get("CUSTOM_MODEL_REPOS", {
    "reasoning": "",
    "coder": "",
    "embeddings": "",
    "vision": "",
    "image_generation": "",
    "tts": "",
})

def _sm_get_category_primary_attr(category):
    return MODEL_CATEGORY_PRIMARY_ATTRS.get(str(category or "").strip().lower(), "")


def _sm_get_category_current_repo(category):
    attr = _sm_get_category_primary_attr(category)
    if not attr:
        return ""
    return resolve_model_repo(str(globals().get(attr, "") or "").strip())


def _sm_get_category_custom_repo(category):
    return str((CUSTOM_MODEL_REPOS or {}).get(str(category or "").strip().lower(), "") or "").strip()


def _sm_collect_model_repos_for_category(category):
    category_key = str(category or "").strip().lower()
    repos = []
    try:
        for tier_items in ((MODEL_CATALOG or {}).get(category_key, {}) or {}).values():
            for repo in tier_items or []:
                resolved = resolve_model_repo(repo)
                if resolved and resolved not in repos:
                    repos.append(resolved)
    except Exception:
        pass
    for repo in (MODEL_CATEGORY_FLAG_REPO_MAP.get(category_key, {}) or {}).values():
        resolved = resolve_model_repo(repo)
        if resolved and resolved not in repos:
            repos.append(resolved)
    current_repo = _sm_get_category_current_repo(category_key)
    if current_repo and current_repo not in repos:
        repos.insert(0, current_repo)
    custom_repo = _sm_get_category_custom_repo(category_key)
    if custom_repo and custom_repo not in repos:
        repos.insert(0, custom_repo)
    return repos


def _sm_sync_model_flags_for_category(category, repo):
    category_key = str(category or "").strip().lower()
    selected_repo = resolve_model_repo(repo)
    for flag_name, flag_repo in (MODEL_CATEGORY_FLAG_REPO_MAP.get(category_key, {}) or {}).items():
        try:
            globals()[flag_name] = bool(resolve_model_repo(flag_repo) == selected_repo)
        except Exception:
            pass


def _sm_set_category_model_repo(category, repo):
    category_key = str(category or "").strip().lower()
    attr = _sm_get_category_primary_attr(category_key)
    selected_repo = str(repo or "").strip()
    if not attr or not selected_repo:
        return ""
    globals()[attr] = selected_repo
    known_repos = set(_sm_collect_model_repos_for_category(category_key))
    if selected_repo in known_repos:
        CUSTOM_MODEL_REPOS[category_key] = ""
    else:
        CUSTOM_MODEL_REPOS[category_key] = selected_repo
    _sm_sync_model_flags_for_category(category_key, selected_repo)
    return selected_repo


def _sm_model_meta_summary(repo):
    resolved = resolve_model_repo(repo)
    meta = dict((MODEL_META or {}).get(resolved) or {})
    parts = []
    if meta.get("tier"):
        parts.append(str(meta.get("tier")))
    if meta.get("speed"):
        parts.append(str(meta.get("speed")))
    if meta.get("disk_gb_est") is not None:
        parts.append(f"disk~{meta.get('disk_gb_est')}GB")
    if meta.get("vram_gb_est") is not None:
        parts.append(f"vram~{meta.get('vram_gb_est')}GB")
    return " | ".join(parts) if parts else "n/a"


def _sm_get_model_manager_snapshot(category, repo):
    resolved = resolve_model_repo(repo)
    snapshot = {
        "repo": resolved,
        "installed": False,
        "local_dir": os.path.join(globals().get("MODELS_DIR", os.path.join(os.getcwd(), "data", "models")), resolved.replace("/", "_")) if resolved else "",
        "local_size_gb": 0.0,
        "estimated_size_gb": None,
        "meta_summary": _sm_model_meta_summary(resolved),
    }
    try:
        import SarahMemoryLLM as _SMLLM  # type: ignore
        info = _SMLLM.get_model_install_snapshot(resolved)
        if isinstance(info, dict):
            snapshot.update({
                "installed": bool(info.get("installed", False)),
                "local_dir": str(info.get("local_dir") or snapshot["local_dir"]),
                "local_size_gb": float(info.get("local_size_gb") or 0.0),
                "estimated_size_gb": info.get("estimated_size_gb"),
            })
    except Exception:
        pass
    return snapshot



# -----------------------------------------------------------------------------
# Model Resolver (v8.0) — Single-model-per-category selection with fallbacks
# - NO model repo strings should be hardcoded in core modules; they live here.
# - POOR tier: auto-selection disabled (core-only) unless user manually enables models.
# -----------------------------------------------------------------------------

def _repo_to_local_dir(repo: str, models_dir: str) -> str:
    """Match SarahMemoryLLM.py local convention: MODELS_DIR/<repo with / replaced by _>."""
    safe = (repo or "").strip().replace("/", "_")
    return os.path.join(models_dir, safe) if safe else ""

def _model_available_locally(repo: str, models_dir: str) -> bool:
    """Best-effort local presence check (directory exists and non-empty)."""
    try:
        p = _repo_to_local_dir(repo, models_dir)
        return bool(p and os.path.isdir(p) and any(os.scandir(p)))
    except Exception:
        return False

def _catalog_repos_for_category(category: str) -> list:
    cat = (category or "").strip().lower()
    out = []
    try:
        tiers = MODEL_CATALOG.get(cat) or {}
        for tlist in tiers.values():
            for r in (tlist or []):
                if r and r not in out:
                    out.append(r)
    except Exception:
        pass
    return out

def _enabled_repos_for_category(category: str) -> list:
    """Return user-enabled repos for this category (resolved via MODEL_REPO_MAP when needed)."""
    cat_repos = set(_catalog_repos_for_category(category))
    enabled = []
    try:
        cfg = MODEL_CONFIG if isinstance(MODEL_CONFIG, dict) else {}
        for k, v in cfg.items():
            if not v:
                continue
            repo = resolve_model_repo(k)
            if not repo:
                repo = str(k)
            if repo in cat_repos and repo not in enabled:
                enabled.append(repo)
    except Exception:
        pass
    return enabled

def _auto_candidates_for_category(category: str, tier_rating: str) -> list:
    """Return ordered candidates for auto-selection by tier_rating."""
    cat = (category or "").strip().lower()
    tiers = MODEL_CATALOG.get(cat) or {}
    low = list(tiers.get("low") or [])
    mid = list(tiers.get("mid") or [])
    high = list(tiers.get("high") or [])
    beast = list(tiers.get("beast") or [])

    # Order: highest first, then down
    if tier_rating == "BEAST":
        return [*beast, *high, *mid, *low]
    if tier_rating == "High":
        return [*high, *mid, *low]
    if tier_rating == "Mid":
        return [*mid, *low]
    if tier_rating == "Low":
        return [*low]
    # Poor: no auto candidates
    return []

def resolve_model(category: str, text: str = "", meta: dict | None = None, models_dir: str | None = None) -> dict:
    """Resolve the primary model + fallbacks for a category.

    Returns:
      {
        selected: <repo str or None>,
        fallbacks: [<repo str>, ...],
        source: "user" | "auto" | "none",
        score: float,
        tier: "low|mid|high|beast",
        tier_rating: "Poor|Low|Mid|High|BEAST",
        third_party_autoload_allowed: bool
      }
    """
    ms_dir = (models_dir or getattr(sys.modules.get(__name__), "MODELS_DIR", None) or "").strip()
    if not ms_dir:
        # default portable location
        try:
            ms_dir = os.path.join(BASE_DIR, "data", "models")
        except Exception:
            ms_dir = os.path.join(os.getcwd(), "data", "models")

    hs = hardware_score()
    tier_rating = str(hs.get("tier_rating") or "Poor")
    tier = str(hs.get("tier") or "low")
    allowed = bool(hs.get("third_party_autoload_allowed", tier_rating != "Poor"))
    score = float(hs.get("score") or 0.0)

    # 1) user-enabled candidates (manual override lane)
    user_candidates = _enabled_repos_for_category(category)

    # POOR: auto disabled unless user explicitly enabled
    if tier_rating == "Poor" and not user_candidates:
        return {
            "selected": None,
            "fallbacks": [],
            "source": "none",
            "score": score,
            "tier": tier,
            "tier_rating": tier_rating,
            "third_party_autoload_allowed": False,
        }

    # 2) auto candidates lane (if allowed)
    auto_candidates = _auto_candidates_for_category(category, tier_rating) if allowed else []

    # Merge: user first, then auto (dedupe, preserve order)
    merged = []
    for r in [*user_candidates, *auto_candidates]:
        if r and r not in merged:
            merged.append(r)

    if not merged:
        return {
            "selected": None,
            "fallbacks": [],
            "source": "none",
            "score": score,
            "tier": tier,
            "tier_rating": tier_rating,
            "third_party_autoload_allowed": allowed,
        }

    # Prefer locally available as primary, otherwise keep order and let loaders attempt local_files_only.
    selected = None
    fallbacks = []
    for r in merged:
        if selected is None and _model_available_locally(r, ms_dir):
            selected = r
        else:
            fallbacks.append(r)
    if selected is None:
        selected = merged[0]
        fallbacks = [r for r in merged[1:]]

    return {
        "selected": selected,
        "fallbacks": fallbacks,
        "source": "user" if selected in user_candidates else "auto",
        "score": score,
        "tier": tier,
        "tier_rating": tier_rating,
        "third_party_autoload_allowed": allowed,
    }

def _looks_non_english(text):
    """Minimal heuristic for language routing when meta.lang isn't provided."""
    if not text:
        return False
    try:
        text.encode("ascii")
        return False
    except Exception:
        return True

def infer_query_language(text, meta=None):
    """Return coarse language label. Uses meta['lang'] if provided."""
    meta = meta or {}
    lang = str(meta.get("lang") or meta.get("language") or "").strip().lower()
    if lang:
        return lang
    return "non-en" if _looks_non_english(text) else "en"

def infer_embedding_job(text, meta=None):
    """Classify embedding job; ensures only one embedding model is selected."""
    t = (text or "").lower()
    if any(k in t for k in ("paraphrase", "rewrite", "rephrase")):
        return "paraphrase"
    if any(k in t for k in ("arxiv", "doi", "paper", "citation", "journal", "pubmed")):
        return "science"
    if any(k in t for k in ("search everything", "high recall", "broad search", "exhaustive")):
        return "high_recall"
    return "general"

def select_embedding_model_repo(text, meta=None):
    """Return the single best embedding model repo for this request.

    IMPORTANT:
    - Honors Top Menu enable flags when at least one embedding model is explicitly enabled.
    - Still guarantees ONE embedding model per request (no multi-run).
    """
    meta = meta or {}
    lang = infer_query_language(text, meta)
    job = infer_embedding_job(text, meta)

    # Preferred routing (job/language aware)
    if lang != "en":
        preferred = EMBEDDING_MULTI_REPO
        candidates = [EMBEDDING_MULTI_REPO, EMBEDDING_EN_REPO, EMBEDDING_RECALL_REPO, EMBEDDING_PARA_REPO, EMBEDDING_SCI_REPO]
    elif job == "science":
        preferred = EMBEDDING_SCI_REPO
        candidates = [EMBEDDING_SCI_REPO, EMBEDDING_EN_REPO, EMBEDDING_RECALL_REPO, EMBEDDING_PARA_REPO, EMBEDDING_MULTI_REPO]
    elif job == "high_recall":
        preferred = EMBEDDING_RECALL_REPO
        candidates = [EMBEDDING_RECALL_REPO, EMBEDDING_EN_REPO, EMBEDDING_PARA_REPO, EMBEDDING_SCI_REPO, EMBEDDING_MULTI_REPO]
    elif job == "paraphrase":
        preferred = EMBEDDING_PARA_REPO
        candidates = [EMBEDDING_PARA_REPO, EMBEDDING_EN_REPO, EMBEDDING_RECALL_REPO, EMBEDDING_SCI_REPO, EMBEDDING_MULTI_REPO]
    else:
        preferred = EMBEDDING_EN_REPO
        candidates = [EMBEDDING_EN_REPO, EMBEDDING_PARA_REPO, EMBEDDING_RECALL_REPO, EMBEDDING_SCI_REPO, EMBEDDING_MULTI_REPO]

    # If user explicitly enabled embedding repos, respect that allow-list.
    try:
        enabled = _enabled_repos_for_category("embeddings")
    except Exception:
        enabled = []

    if enabled:
        # 1) pick first preferred candidate that is enabled
        for r in candidates:
            if r in enabled:
                return r
        # 2) otherwise pick the first enabled (stable deterministic)
        return enabled[0]

    # No explicit enables -> keep legacy preferred routing
    return preferred

def get_stack_primary_repo(category, text="", meta=None):
    """Primary repo resolver for category-based routing."""
    category = (category or "").strip().lower()
    if category in ("embedding", "embeddings", "semantic", "memory"):
        return select_embedding_model_repo(text, meta)
    if category in ("reasoning", "logic"):
        return REASONING_MODEL_REPO
    if category in ("coder", "code"):
        return CODER_MODEL_REPO
    if category in ("vision", "object", "object_detection"):
        return VISION_PRIMARY_REPO
    if category in ("image", "image_generation", "creative"):
        return IMAGEGEN_MODEL_REPO
    if category in ("tts", "audio", "voice"):
        return TTS_MODEL_REPO
    return ""

# ---------------------------------------------------------------------------
# Hardware Scoring Metrics (utility-only; no boot-flow changes)
# ---------------------------------------------------------------------------
def get_system_metrics(models_dir=None):
    """Best-effort hardware snapshot for model tiering (never raises).

    v8.0 unified environment rule:
    - Prefer the single boot environment snapshot from SarahMemoryHi.
    - Fall back to direct probing only if the snapshot is unavailable.
    This prevents boot from grading CPU/GPU/RAM once, then driver readiness and
    chat/API layers probing the same hardware again with conflicting labels.
    """
    try:
        if not globals().get("_SM_HARDWARE_DIRECT_PROBE", False):
            import SarahMemoryHi as _SMHi  # type: ignore
            metrics_fn = getattr(_SMHi, "get_boot_environment_model_metrics", None)
            if callable(metrics_fn):
                cached_metrics = metrics_fn(force_refresh=False)
                if isinstance(cached_metrics, dict) and cached_metrics:
                    return {
                        "cpu_count": cached_metrics.get("cpu_count"),
                        "cpu_pct": cached_metrics.get("cpu_pct"),
                        "ram_total_mb": cached_metrics.get("ram_total_mb"),
                        "ram_avail_mb": cached_metrics.get("ram_avail_mb"),
                        "disk_free_gb": cached_metrics.get("disk_free_gb"),
                        "disk_total_gb": cached_metrics.get("disk_total_gb"),
                        "gpu_name": cached_metrics.get("gpu_name"),
                        "gpu_vram_total_mb": cached_metrics.get("gpu_vram_total_mb"),
                        "gpu_vram_free_mb": cached_metrics.get("gpu_vram_free_mb"),
                        "gpu_temp_c": cached_metrics.get("gpu_temp_c"),
                        "cpu_temp_c": cached_metrics.get("cpu_temp_c"),
                    }
    except Exception:
        pass

    models_dir = models_dir or globals().get("MODELS_DIR") or os.path.join(os.getcwd(), "data", "models")
    out = {
        "cpu_count": None,
        "cpu_pct": None,
        "ram_total_mb": None,
        "ram_avail_mb": None,
        "disk_free_gb": None,
        "disk_total_gb": None,
        "gpu_name": None,
        "gpu_vram_total_mb": None,
        "gpu_vram_free_mb": None,
        "gpu_temp_c": None,
        "cpu_temp_c": None,
    }

    # CPU/RAM/Disk via psutil (optional)
    try:
        import psutil  # type: ignore
        out["cpu_count"] = psutil.cpu_count(logical=True)
        out["cpu_pct"] = float(psutil.cpu_percent(interval=0.0))
        vm = psutil.virtual_memory()
        out["ram_total_mb"] = int(vm.total / (1024 * 1024))
        out["ram_avail_mb"] = int(vm.available / (1024 * 1024))
        du = psutil.disk_usage(models_dir if os.path.exists(models_dir) else os.getcwd())
        out["disk_total_gb"] = float(du.total / (1024**3))
        out["disk_free_gb"] = float(du.free / (1024**3))

        # Temps (best effort)
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False) or {}
            for key in ("coretemp", "cpu-thermal", "k10temp", "acpitz"):
                if key in temps and temps[key]:
                    out["cpu_temp_c"] = float(temps[key][0].current)
                    break
        except Exception:
            pass
    except Exception:
        pass

    # GPU: try torch first, then nvidia-smi
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = 0
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            out["gpu_vram_total_mb"] = int(props.total_memory / (1024 * 1024))
            try:
                reserved = int(torch.cuda.memory_reserved(idx) / (1024 * 1024))
                out["gpu_vram_free_mb"] = max(0, out["gpu_vram_total_mb"] - reserved)
            except Exception:
                pass
    except Exception:
        pass

    if out.get("gpu_vram_total_mb") is None:
        try:
            import subprocess
            cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits"
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            if p.returncode == 0 and (p.stdout or "").strip():
                line = (p.stdout or "").strip().splitlines()[0]
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    out["gpu_name"] = parts[0]
                    out["gpu_vram_total_mb"] = int(float(parts[1]))
                    out["gpu_vram_free_mb"] = int(float(parts[2]))
                    out["gpu_temp_c"] = float(parts[3])
        except Exception:
            pass

    return out

def hardware_score(metrics=None):
    """Compute coarse score + tier from metrics."""
    m = metrics or get_system_metrics()
    score = 0.0

    # RAM (0..40)
    try:
        ram = float(m.get("ram_total_mb") or 0)
        if ram >= 32768:
            score += 40
        elif ram >= 16384:
            score += 30
        elif ram >= 8192:
            score += 20
        elif ram >= 4096:
            score += 10
        elif ram > 0:
            score += 5
    except Exception:
        pass

    # VRAM (0..40)
    try:
        vram = float(m.get("gpu_vram_total_mb") or 0)
        if vram >= 24000:
            score += 40
        elif vram >= 12000:
            score += 30
        elif vram >= 8000:
            score += 20
        elif vram >= 4000:
            score += 10
        elif vram > 0:
            score += 5
    except Exception:
        pass

    # Disk free (0..10)
    try:
        free = float(m.get("disk_free_gb") or 0)
        if free >= 200:
            score += 10
        elif free >= 100:
            score += 7
        elif free >= 50:
            score += 5
        elif free >= 20:
            score += 3
        elif free > 0:
            score += 1
    except Exception:
        pass

    # CPU headroom (0..10)
    try:
        cpu_pct = float(m.get("cpu_pct") or 0)
        if cpu_pct <= 20:
            score += 10
        elif cpu_pct <= 40:
            score += 8
        elif cpu_pct <= 60:
            score += 6
        elif cpu_pct <= 80:
            score += 3
        else:
            score += 1
    except Exception:
        pass

    # Tier Rating (owner policy)
    # Poor:  <= 69.9
    # Low:    70.0 - 75.0
    # Mid:    75.1 - 80.0
    # High:   80.1 - 90.0
    # BEAST:  >= 90.1
    tier_rating = "Poor"
    try:
        s = float(score)
        if s >= 90.1:
            tier_rating = "BEAST"
        elif 80.1 <= s <= 90.0:
            tier_rating = "High"
        elif 75.1 <= s <= 80.0:
            tier_rating = "Mid"
        elif 70.0 <= s <= 75.0:
            tier_rating = "Low"
        else:
            tier_rating = "Poor"
    except Exception:
        tier_rating = "Poor"

    # Tier mapping for model catalogs
    # Normalized tier string for selectors: low|mid|high|beast
    tier = "low"
    if tier_rating == "Mid":
        tier = "mid"
    elif tier_rating == "High":
        tier = "high"
    elif tier_rating == "BEAST":
        tier = "beast"

    third_party_autoload_allowed = (tier_rating != "Poor")
    return {
        "score": float(score),
        "tier": tier,
        "tier_rating": tier_rating,
        "third_party_autoload_allowed": third_party_autoload_allowed,
        "metrics": m,
    }
def recommend_model_tier(category="reasoning", metrics=None):
    """Return low/mid/beast tier recommendation for a given category."""
    hs = hardware_score(metrics)
    cat = (category or "").strip().lower()
    tier = hs["tier"]
    try:
        vram = float(hs["metrics"].get("gpu_vram_total_mb") or 0)
    except Exception:
        vram = 0

    if cat in ("image_generation", "image", "creative"):
        if vram >= 12000:
            return "beast" if tier == "beast" else "mid"
        if vram >= 8000:
            return "mid"
        return "low"

    if cat in ("vision", "object", "object_detection"):
        if vram >= 8000 and tier != "low":
            return tier
        return "low" if tier == "low" else "mid"

    return tier

def pick_catalog_model(category, tier, fallback_tiers=None):
    """Pick first model repo from MODEL_CATALOG[category][tier] with tier fallback."""
    cat = (category or "").strip().lower()
    t = (tier or "").strip().lower()
    fb = fallback_tiers or (["mid", "low"] if t == "beast" else ["low"] if t == "mid" else [])
    try:
        c = MODEL_CATALOG.get(cat, {})
        for k in [t] + fb:
            arr = c.get(k, []) or []
            if arr:
                return arr[0]
    except Exception:
        return None
    return None

# ---------------------------------------------------------------------------
# Model Storage / Bandwidth Policy (utility-only; safe for headless)
# ---------------------------------------------------------------------------
SARAH_MODELS_BUDGET_GB = float(os.getenv("SARAH_MODELS_BUDGET_GB", "256") or 256)
SARAH_MODEL_MAX_SINGLE_GB = float(os.getenv("SARAH_MODEL_MAX_SINGLE_GB", "1.5") or 1.5)
SARAH_MODEL_PROMPT_LARGE = _env_flag("SARAH_MODEL_PROMPT_LARGE", "true")

def is_headless_runtime():
    try:
        dm = str(globals().get("DEVICE_MODE") or "").strip().lower()
        if dm == DEVICE_MODE_HEADLESS:
            return True
        if os.name != "nt" and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            return True
        return False
    except Exception:
        return True

def is_interactive_tty():
    try:
        import sys
        return bool(getattr(sys, "stdin", None) and sys.stdin.isatty())
    except Exception:
        return False

def get_models_dir_fallback():
    try:
        md = globals().get("MODELS_DIR")
        if isinstance(md, str) and md:
            return md
    except Exception:
        pass
    return os.path.join(os.getcwd(), "data", "models")

def get_models_storage_usage_bytes(models_dir=None):
    models_dir = models_dir or get_models_dir_fallback()
    total = 0
    try:
        for root, _, files in os.walk(models_dir):
            for fn in files:
                try:
                    fp = os.path.join(root, fn)
                    total += os.path.getsize(fp)
                except Exception:
                    pass
    except Exception:
        return 0
    return total

def bytes_to_gb(n):
    try:
        return float(n) / (1024**3)
    except Exception:
        return 0.0

def get_models_storage_usage_gb(models_dir=None):
    return bytes_to_gb(get_models_storage_usage_bytes(models_dir))

def model_policy_allows_download(expected_size_gb, models_dir=None):
    models_dir = models_dir or get_models_dir_fallback()
    used_gb = get_models_storage_usage_gb(models_dir)
    budget_gb = float(SARAH_MODELS_BUDGET_GB)
    max_single_gb = float(SARAH_MODEL_MAX_SINGLE_GB)

    if expected_size_gb is None or expected_size_gb <= 0:
        return {"ok": True, "reason": "unknown_size", "prompt_required": False,
                "budget_gb": budget_gb, "used_gb": used_gb, "max_single_gb": max_single_gb}

    if (used_gb + expected_size_gb) > budget_gb:
        return {"ok": False, "reason": "budget_exceeded", "prompt_required": False,
                "budget_gb": budget_gb, "used_gb": used_gb, "max_single_gb": max_single_gb}

    if SARAH_MODEL_PROMPT_LARGE and expected_size_gb > max_single_gb:
        return {"ok": True, "reason": "large_model_confirm", "prompt_required": True,
                "budget_gb": budget_gb, "used_gb": used_gb, "max_single_gb": max_single_gb}

    return {"ok": True, "reason": "ok", "prompt_required": False,
            "budget_gb": budget_gb, "used_gb": used_gb, "max_single_gb": max_single_gb}

MODEL_META = {
    "Qwen/Qwen3-0.6B": {"tier": "low", "disk_gb_est": 1.0, "vram_gb_est": 2.0, "speed": "fast"},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {"tier": "low", "disk_gb_est": 2.0, "vram_gb_est": 4.0, "speed": "medium"},
    "sentence-transformers/all-MiniLM-L6-v2": {"tier": "low", "disk_gb_est": 0.1, "vram_gb_est": 0.2, "speed": "fast"},
    "sentence-transformers/distiluse-base-multilingual-cased-v2": {"tier": "mid", "disk_gb_est": 0.5, "vram_gb_est": 0.5, "speed": "fast"},
    "allenai/specter": {"tier": "mid", "disk_gb_est": 0.5, "vram_gb_est": 0.5, "speed": "fast"},
    "intfloat/e5-base": {"tier": "mid", "disk_gb_est": 0.5, "vram_gb_est": 0.7, "speed": "fast"},
    "sentence-transformers/paraphrase-MiniLM-L3-v2": {"tier": "low", "disk_gb_est": 0.1, "vram_gb_est": 0.2, "speed": "fast"},
    "nielsr/yolov12n": {"tier": "low", "disk_gb_est": 0.1, "vram_gb_est": 0.5, "speed": "fast"},
    "ultralytics/yolov8": {"tier": "mid", "disk_gb_est": 0.1, "vram_gb_est": 1.0, "speed": "fast"},
    "qualcomm/RF-DETR": {"tier": "mid", "disk_gb_est": 0.4, "vram_gb_est": 1.5, "speed": "medium"},
    "Freepik/flux.1-lite-8B": {"tier": "mid", "disk_gb_est": 16.0, "vram_gb_est": 12.0, "speed": "slow"},
    "FunAudioLLM/CosyVoice2-0.5B": {"tier": "low", "disk_gb_est": 1.0, "vram_gb_est": 2.0, "speed": "fast"},
    "Qwen/Qwen2.5-7B-Instruct": {"tier": "high", "disk_gb_est": 8.0, "vram_gb_est": 10.0, "speed": "medium"},
    "Qwen/Qwen2.5-Coder-3B-Instruct": {"tier": "mid", "disk_gb_est": 4.0, "vram_gb_est": 6.0, "speed": "medium"},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"tier": "high", "disk_gb_est": 8.0, "vram_gb_est": 10.0, "speed": "slow"},
    "microsoft/Phi-4-mini-instruct": {"tier": "mid", "disk_gb_est": 3.0, "vram_gb_est": 4.0, "speed": "fast"},
    "mistralai/Mistral-7B-Instruct-v0.2": {"tier": "high", "disk_gb_est": 8.0, "vram_gb_est": 10.0, "speed": "medium"},
    "BAAI/bge-base-en-v1.5": {"tier": "high", "disk_gb_est": 1.0, "vram_gb_est": 1.0, "speed": "fast"},
    "BAAI/bge-m3": {"tier": "beast", "disk_gb_est": 2.0, "vram_gb_est": 2.0, "speed": "medium"},
    "ultralytics/yolov8x": {"tier": "high", "disk_gb_est": 0.3, "vram_gb_est": 2.0, "speed": "medium"},
    "black-forest-labs/FLUX.1-schnell": {"tier": "low", "disk_gb_est": 4.0, "vram_gb_est": 6.0, "speed": "fast"},
    "black-forest-labs/FLUX.1-dev": {"tier": "beast", "disk_gb_est": 20.0, "vram_gb_est": 16.0, "speed": "slow"},
}

# ---------------- Object Detection Model Configuration (Spring Clean 2026) ----------------
# Keep: YOLOv12n + YOLOv8 + SSD
# Backup option: RF-DETR (disabled by default)
# ---------------- Object Detection Model Configuration (Spring Clean 2026) ----------------
# Single source of truth:
# - DO NOT toggle models here.
# - These booleans bind to the TOP MENU switches (VISION / OBJECT DETECTION).

OBJECT_DETECTION_ENABLED = True

# --- Bind supported detectors to TOP MENU flags ---
ENABLE_YOLOV12N  = bool(ENABLE_MODEL_V)
ENABLE_YOLOV8    = bool(ENABLE_MODEL_W)
ENABLE_RF_DETR   = bool(ENABLE_MODEL_X)

# SSD is detector-specific; map to TOP MENU too
ENABLE_SSD       = bool(ENABLE_MODEL_AD)

# --- Bind legacy compat toggles to TOP MENU legacy flags ---
ENABLE_YOLOV5        = bool(ENABLE_MODEL_AE)
ENABLE_YOLOV7        = bool(ENABLE_MODEL_AF)
ENABLE_YOLO_NAS      = bool(ENABLE_MODEL_AG)
ENABLE_YOLOX         = bool(ENABLE_MODEL_AH)
ENABLE_PP_YOLOV2      = bool(ENABLE_MODEL_AI)
ENABLE_EFFICIENTDET  = bool(ENABLE_MODEL_AJ)
ENABLE_DETR          = bool(ENABLE_MODEL_AK)
ENABLE_DINO          = bool(ENABLE_MODEL_AL)
ENABLE_CENTERNET     = bool(ENABLE_MODEL_AM)
ENABLE_FASTER_RCNN   = bool(ENABLE_MODEL_AN)
ENABLE_RETINANET     = bool(ENABLE_MODEL_AO)

OBJECT_MODEL_CONFIG = {
    "YOLOv12n": {"enabled": bool(ENABLE_YOLOV12N), "repo": "yolov12n", "hf_repo": "nielsr/yolov12n", "require": None},
    "YOLOv8":   {"enabled": bool(ENABLE_YOLOV8),   "repo": "ultralytics_yolov8", "hf_repo": "ultralytics/yolov8", "require": "ultralytics"},
    "SSD":      {"enabled": bool(ENABLE_SSD),      "repo": "qfgaohao_pytorch-ssd", "hf_repo": "https://github.com/qfgaohao/pytorch-ssd", "require": None,
        "weights": [
            {"url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/mobilenet-v1-ssd-mp-0_675.pth", "filename": "mobilenet-v1-ssd-mp-0_675.pth"},
            {"url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/voc-model-labels.txt", "filename": "voc-model-labels.txt"},
        ]
    },
    "RF-DETR":  {"enabled": bool(ENABLE_RF_DETR),  "repo": "rf_detr", "hf_repo": "qualcomm/RF-DETR", "require": None},
}

#----------------------------------------------------------------------------------------------------------

mic = True #Set to True for voice and typing in the GUI/False for typing only, default True
# Sound Default configuration for recognition
LISTEN_TIMEOUT = 5       # seconds to wait for speech start, default 5
PHRASE_TIME_LIMIT = 10    # maximum seconds of speech capture, default 10
NOISE_SCALE = 0.7 # default 0.7
AMBIENT_NOISE_DURATION = 0.2  # Reduced duration for faster calibration , default 0.2

AVATAR_WINDOW_RESIZE = True #If True the Avatar Window will be Resizable if False the dimentions on the windows can not, default True
# Setup logger
logger = logging.getLogger("SarahMemoryGlobals")
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Base directory of the program
BASE_DIR = os.getcwd() # AS for Now. This Program is designed to be strickly on C:\SarahMemory and cloned at https://www.sarahmemory.com/api
# New Features 7.7.5 will include the program to run from any platform, Windows, Linux, Android, iOS, PythonAnywhere
logger.info(f"BASE_DIR set to: {BASE_DIR}")
#
# --- UI Configuration ---
ENABLE_AVATAR_PANEL = True #Set to True to display Avatar PANEL Window Display when GUI Launches.
DEFAULT_AVATAR = os.path.join(BASE_DIR, "resources", "avatars", "avatar.jpg")
STATUS_LIGHTS = {"green": "#00FF00", "yellow": "#FFFF00", "red": "#FF0000"}
ENABLE_SARCASM_LAYER = True # Random Sarcasm Personality Engine (toggle True/False) â€“ Injected based on a randomness factors.Default True
# NEW CONFIG: Enable advanced features
ENABLE_CONTEXT_BUFFER = True  # Flag for context buffer, default True
CONTEXT_BUFFER_SIZE = 10      # Maximum number of interactions to store, default 10
ASYNC_PROCESSING_ENABLED = True  # Enable asynchronous operations, default True
VOICE_FEEDBACK_ENABLED = True #Allows AI to Speak back to End-User using TTS, default True

# Researching Halting Configuration
INTERRUPT_FLAG = False  # Global state,
INTERRUPT_KEYWORDS = ["stop", "just stop", "halt"] #Stops SarahMemoryResearch.py on Researching Information using Keywords

# --- UI Stack Selection (Classic / Web / Custom) ---
# Reads from .env: SARAH_UI_MODE = classic | web | custom
UI_MODE = os.getenv("SARAH_UI_MODE", "classic").strip().lower()
if UI_MODE not in ("classic", "web", "custom"):
    UI_MODE = "classic"

# Convenience booleans so other modules can just check caps instead of strings.
USE_CLASSIC_GUI  = (UI_MODE == "classic")
USE_LEGACY_WEBUI = (UI_MODE == "web")
USE_CUSTOM_WEBUI = (UI_MODE == "custom")


# =============================================================================
# --- Legacy WebUI (index.html + app.js + styles.css) paths ---
# =============================================================================
# These files LIVE in: ./data/ui
# They are the older static WebUI stack and are served directly by Flask
# or opened via pywebview depending on runtime mode.
#
# Folder layout expected:
#   BASE_DIR/
#       data/
#           ui/
#               index.html
#               app.js
#               styles.css
#
# DO NOT point this to BASE_DIR root — that caused prior path confusion.
# =============================================================================

LEGACY_WEBUI_DIR = os.path.join(BASE_DIR, "data", "ui")
LEGACY_WEBUI_INDEX = os.path.join(LEGACY_WEBUI_DIR, "index.html")
LEGACY_WEBUI_JS = os.path.join(LEGACY_WEBUI_DIR, "app.js")
LEGACY_WEBUI_CSS = os.path.join(LEGACY_WEBUI_DIR, "styles.css")

# Safety: If folder missing, auto-fallback to classic
if USE_LEGACY_WEBUI and not os.path.isdir(LEGACY_WEBUI_DIR):
    print("[UI WARNING] Legacy WebUI directory missing — falling back to classic mode.")
    USE_LEGACY_WEBUI = False
    USE_CLASSIC_GUI = True


# =============================================================================
# --- Custom React/Vite WebUI (Lovable / Vite build) ---
# =============================================================================
# Our pipeline:
#   - SarahMemoryUIupdater.py clones into:   BASE_DIR/data/ui/V8_ui_src
#   - It builds Vite "dist" there
#   - Then copies dist/* into:              BASE_DIR/data/ui/V8
#
# So the *served* dist folder is always: BASE_DIR/data/ui/V8
# =============================================================================

CUSTOM_UI_ROOT = os.getenv(
    "SARAH_CUSTOM_UI_ROOT",
    os.path.join(BASE_DIR, "data", "ui", "V8_ui_src"),
)

CUSTOM_UI_DIST_DIR = os.getenv(
    "SARAH_CUSTOM_UI_DIST_DIR",
    os.path.join(BASE_DIR, "data", "ui", "V8"),
)

CUSTOM_UI_INDEX = os.path.join(CUSTOM_UI_DIST_DIR, "index.html")

# If a dev server is running (npm run dev / npm run preview), we can point
# pywebview or the browser directly at this URL instead of a local file path.
CUSTOM_UI_DEV_URL = os.getenv(
    "SARAH_CUSTOM_UI_DEV_URL",
    "http://127.0.0.1:8000",
)

# Safety: If custom selected but dist missing, auto-fallback
if USE_CUSTOM_WEBUI and not os.path.isdir(CUSTOM_UI_DIST_DIR):
    print("[UI WARNING] Custom UI dist folder missing — falling back to legacy web.")
    USE_CUSTOM_WEBUI = False
    USE_LEGACY_WEBUI = True


def get_ui_launch_profile() -> dict:
    """
    Central place for SarahMemoryMain.py (and others) to decide:
      - which UI mode is active
      - which file or URL to open
      - whether to prefer pywebview or an external browser
    """
    mode = globals().get("UI_MODE", "classic")
    base = {
        "mode": mode,
        # USE_WEBVIEW is defined later (v7.7.5 GUI WebUI additions) and
        # controls whether we embed a webview or open an external browser.
        "use_webview": bool(globals().get("USE_WEBVIEW", True)),
    }

    if mode == "classic":
        base.update({
            "kind": "desktop_gui",
            "entry": "SarahMemoryGUI.py",  # informational; actual import happens in main
        })

    elif mode == "web":
        # Original app.js / index.html UI at repo root
        base.update({
            "kind": "legacy_webui",
            "html_path": LEGACY_WEBUI_INDEX,
            "html_dir": LEGACY_WEBUI_DIR,
            "url": None,  # can be filled in by app.py if served via Flask
        })

    elif mode == "custom":
        # Custom React/Vite build (Lovable / Vite)
        base.update({
            "kind": "custom_webui",
            "html_path": CUSTOM_UI_INDEX,
            "html_dir": CUSTOM_UI_DIST_DIR,
            "dev_url": CUSTOM_UI_DEV_URL,
        })

    else:
        # Failsafe: fall back to classic GUI
        base.update({
            "kind": "desktop_gui",
            "entry": "SarahMemoryGUI.py",
        })

    return base


# Build Learned Vector datasets, only need to be Ran Once after SarahMemorySystemLearn.py has been ran or when New information has been intergrated.
IMPORT_OTHER_DATA_LEARN = True #Rebuilds Vector on each BOOT UP if True It will consistantly Rebuild every Boot when New Data is found,
LEARNING_PHASE_ACTIVE = True #Keeps system from constantly rebuilding Vectored dataset. If True will rebuild constantly

# Researching Configurations
LOCAL_DATA_ENABLED = True # False = Temporary Disable local search until trained. SarahMemoryResearch.py Class 1
ROUTE_MODE = "Any"  # Options: "Any", "Local", "Web", "API"
WEB_RESEARCH_ENABLED = True # True = False Disable Web search Learning. SarahMemoryResearch.py - Class 2
# Web Homepage This will be the HomePage in which is seen when the SarahMemoryGUI.py interface is loaded.
WEB_HOMEPAGE = "https://www.duckduckgo.com"
# Web Research Source Flags, For SarahMemoryResearch.py - Class 2 - WebSearching and Learning mode
DUCKDUCKGO_RESEARCH_ENABLED = True #Set True/False for testing purposes (semi-works)
WIKIPEDIA_RESEARCH_ENABLED = True #Set True/False for testing purposes (works)
FREE_DICTIONARY_RESEARCH_ENABLED = True #Set True/False for Testing purposes (semi-works)

# Note these are set to False because of multiple different reasons and must be highly researched before setting any to TRUE
STACKOVERFLOW_RESEARCH_ENABLED = False # Set to False until further notice
REDDIT_RESEARCH_ENABLED = False # Set to False until further notice
WIKIHOW_RESEARCH_ENABLED = False # Set to False until further notice
QUORA_RESEARCH_ENABLED = False #Set to False until further notice
OPENLIBRARY_RESEARCH_ENABLED = False #Set to False until further notice
INTERNET_ARCHIVE_RESEARCH_ENABLED = False #Set True/False for testing purposes

#Multiple AI API Research Connections For SarahMemoryResearch.py - Class 3 - Learning for other AI's
API_RESEARCH_ENABLED = True #False = Disable from Learning from An Ai API.
#Allows End User to select which AI API to be used for SarahMemoryResearch.py - Class 3 when query is passed through SarahMemoryAPI.py
#WARNING: AS OF VERSION 7.0 CURRENTLY ONLY ONE (1) OF THE FOLLOWING API's MAY BE SET TO TRUE AND ALL OTHERS MUST BE SET TO FALSE
# ============================================================================
# API PROVIDER CONFIGURATION (v8.0 Expanded Selection Logic)
# ============================================================================

# Individual Provider Toggles 
# Should only work if API_RESEARCH_ENABLE flag on line 1443 is set to 'True'
OPEN_AI_API     = True
CLAUDE_API      = False
ANTHROPIC_API   = False
MISTRAL_API     = False
GEMINI_API      = True
HUGGINGFACE_API = False
DEEPSEEK_API    = False
GROQ_API        = False
COHERE_API      = False

LOCAL_LLM_API   = True # LOCAL_LLM_API When 'True' All Requests/Responses are Ran from the Auto/Manual-Selected 3rd Party MODEL_CATELOG NO NEED FOR EXTERNAL API CALLS
LOCAL_API       = True # LOCAL_API is the LOCAL SYSTEM ITSELF NOT A 3rd Party API it is the Local .DB Vectored System
MESH_API        = True # MESH_API is the SarahMemory Network https://api.sarahmemory.net
# MESH_API is the NODE NETWORK of other systems running the SARAHMEMORY AiOS systems

# ---------------------------------------------------------------------------
# Unified Provider Registry
# ---------------------------------------------------------------------------

API_PROVIDER_FLAGS = {
    "local_llm":  LOCAL_LLM_API, # Needs to be the AUTO or MANUAL SELECTED Local MODEL 
    "local":       LOCAL_API,
    "openai":      OPEN_AI_API,
    "claude":      CLAUDE_API,
    "anthropic":   ANTHROPIC_API,
    "mistral":     MISTRAL_API,
    "gemini":      GEMINI_API,
    "huggingface": HUGGINGFACE_API,
    "deepseek":    DEEPSEEK_API,
    "groq":        GROQ_API,
    "cohere":      COHERE_API,
    "mesh":        MESH_API,
}

# ---------------------------------------------------------------------------
# Provider Selection Logic
# ---------------------------------------------------------------------------

# First enabled provider becomes PRIMARY_API
PRIMARY_API = next(
    (name for name, enabled in API_PROVIDER_FLAGS.items() if enabled),
    "none"
)

# All other enabled providers become fallbacks (ordered)
API_FALLBACKS = [
    name for name, enabled in API_PROVIDER_FLAGS.items()
    if enabled and name != PRIMARY_API
]

# Convenience flag
ANY_API_ENABLED = any(API_PROVIDER_FLAGS.values())


# API RATE LIMIT/TIMEOUT CONTROLLER to allow AUTO SWITCHING OF API's For the Best Results.
API_TIMEOUT = 20 # timer number is for seconds. (API_TIMEOUT = 20 is default)
API_RESPONSE_CHECK_TRAINER = True #Set to True to Compare Synthesis Results with an AI system before logging a proper response into the datasets

# Reply Stats and Confidence viewer - When Set to True show Source, confidence level, emotional state, and Intent and HIT/MISS Status of Chat Query
REPLY_STATUS = True
# Compare Reply Vote Flag - When Set to True will allow and request a Dynamic feedback injection from the SarahMemoryGUI.py Chat of YES or NO on response given.
COMPARE_VOTE = False #True = prompts user after a Response has been Compared and given if it was good for the User or Not to help Learn.
COMPARE_THRESHOLD_VALUE ="0.061" # value must be in a (0.000) formatThis Value is the limitation in which an automatic response must pass to be consider a HIT and is stored or overwrites previous Local Responses in the local datasets, below this Value the response is a MISS, the answer may be stored in the datasets, but if the local dataset already have a reply it shall not be,
#VISUAL LEARNING, Facial and Object Recognition
VISUAL_BACKGROUND_LEARNING = True #True/False = On /Off for Object Learning in the Background This is a silent running background process
FACIAL_RECOGNITION_LEARNING = True  #True/False = On /Off for Learning People Facial Expressions and body movement and language
ENABLE_CONTEXT_ENRICHMENT = True #True/False = On /Off for Deep Learning about User in background when Ai-bot system is Idle.
DL_IDLE_TIMER = 1800 #Time amount the system must be at idle at before starting background DeepLearning

# --- Network Defaults ---

# --- User Settings (login/password from ENV for security) ---
USERNAME = os.getenv("USERNAME", "SarahUser")  # Primary user account name for personalization & future social login
OS_TYPE = platform.system()  # System OS detected (Windows/Linux/macOS) for compatibility logic

# --- IP/PORT Settings ---
DEFAULT_PORT = 8000 # Localhost Flask API port for internal server communication
DEFAULT_HOST = "127.0.0.1"  # Loopback address for local testing only

# === SarahNet (Mesh Comms) â€” managed in Globals (no external JSON) ==========
SARAHNET_ENABLED: bool = True

# Core identity & bind
SARAHNET_NODE_ID: str   = os.getenv("SARAHNET_NODE_ID", "node-A")
SARAHNET_BIND_HOST: str = os.getenv("SARAHNET_BIND_HOST", "0.0.0.0")
SARAHNET_BIND_PORT: int = int(os.getenv("SARAHNET_BIND_PORT", "9876"))
# Peers (editable here; values are (host, port) tuples)
SARAHNET_PEERS: dict[str, tuple[str, int]] = {
    "node-B": ("184.52.80.237", 9998),
    "node-C": ("183.81.169.155", 9997),
}
SARAHNET_RELAY_TIMEOUT_SEC = float(os.getenv("SARAHNET_RELAY_TIMEOUT_SEC","2.0"))
# IDS / Transport tuning
SARAHNET_RPS: int         = int(os.getenv("SARAHNET_RPS", "30"))
SARAHNET_BURST: int       = int(os.getenv("SARAHNET_BURST", "60"))
SARAHNET_PREFER_TCP: bool = True
SARAHNET_ALLOW_UDP: bool  = True

# Optional: shared secret (bytes). If None, derived deterministically from author+version.
SARAHNET_SHARED_SECRET: bytes | None = None
# Centralized Web-server Hub for all AI's to Cross Communicate and exchange information.
# Each Copy of SarahMemory is it's own Node, and Maybe used as a Server to assist other AI's Exchange information
# The SarahMemory Web-Server itself is the Main Hub where all AI's can exchange information using a Cryptobased Wallet concept
# The Crypto for this system is for non-monetary gain, and may only be used by AI systems as a ledger to give and recieve information
SARAH_WEB_BASE = "https://www.sarahmemory.com"
SARAH_WEB_API_PREFIX = "/api"
SARAH_WEB_PING_PATH = "/api/data/health"
SARAH_WEB_HEALTH_PATH = "/api/data/health"
SARAH_WEB_RELAY_PATH = "/api/data/relay"
SARAH_WEB_REGISTER_PATH = "/api/data/register-node"
SARAH_WEB_HEARTBEAT_PATH = "/api/data/heartbeat"
SARAH_WEB_EMBED_PATH = "/api/data/receive-embedding"

SARAH_WEB_CONTEXT_PATH = "/api/data/context-update"
SARAH_WEB_JOBS_PATH = "/api/data/jobs"

REMOTE_SYNC_ENABLED = True
REMOTE_HTTP_TIMEOUT = 6.0
REMOTE_HEARTBEAT_SEC = 30
REMOTE_API_KEY = None
SARAHNET_NODE_ID = "local-node"


# Canvas Studio settings
CANVAS_STUDIO_ENABLED = True
CANVAS_DEFAULT_WIDTH = 1920
CANVAS_DEFAULT_HEIGHT = 1080
CANVAS_MAX_LAYERS = 100
CANVAS_AUTO_SAVE = True

# Video Editor settings
VIDEO_EDITOR_ENABLED = True
VIDEO_DEFAULT_FPS = 30
VIDEO_DEFAULT_CODEC = "h264"
VIDEO_RENDER_QUALITY = "high"

# BioSync settings
BIOSYNC_ENABLED = True
BIOSYNC_REQUIRED_FACTORS = 2
BIOSYNC_CONFIDENCE_THRESHOLD = 0.95
BIOSYNC_CONTINUOUS_AUTH = True

# Music & Lyrics settings
MUSIC_GENERATOR_ENABLED = True
LYRICS_GENERATOR_ENABLED = True
DEFAULT_MUSIC_TEMPO = 120
DEFAULT_MUSIC_KEY = "C_major"
DEFAULT_VOICE_MODEL = "female_pop"

# Lyrics To Song Configuration
LYRICS_DEFAULT_VOICE = "neutral"
LYRICS_DEFAULT_EMOTION = "neutral"
LYRICS_DEFAULT_TEMPO = 120
LYRICS_DEFAULT_KEY = "C"
LYRICS_DEFAULT_SCALE = "major"
LYRICS_DEFAULT_STYLE = "pop"
LYRICS_SAMPLE_RATE = 44100
LYRICS_BIT_DEPTH = 16
LYRICS_ENABLE_HARMONIES = True
LYRICS_MAX_HARMONY_PARTS = 4
LYRICS_ENABLE_GPU = False  # For Bark/Coqui
LYRICS_CACHE_ENABLED = True
LYRICS_CACHE_SIZE_MB = 500



def sarahnet_shared_secret() -> bytes:
    try:
        if isinstance(SARAHNET_SHARED_SECRET, (bytes, bytearray)) and len(SARAHNET_SHARED_SECRET) >= 16:
            return bytes(SARAHNET_SHARED_SECRET)
    except Exception:
        pass
    import hashlib
    seed = (AUTHOR + PROJECT_VERSION).encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).digest()

def get_sarahnet_config() -> dict:
    """Canonical SarahNet configuration (never touches disk)."""
    return {
        "node_id":     SARAHNET_NODE_ID,
        "bind_host":   SARAHNET_BIND_HOST,
        "bind_port":   int(SARAHNET_BIND_PORT),
        "peers":       {k: [v[0], int(v[1])] for k, v in (SARAHNET_PEERS or {}).items()},
        "rps":         int(SARAHNET_RPS),
        "burst":       int(SARAHNET_BURST),
        "prefer_tcp":  bool(SARAHNET_PREFER_TCP),
        "allow_udp":   bool(SARAHNET_ALLOW_UDP),
    }

# Optional singleton so other modules can attach a running node for reuse/shutdown.
_MESH_NODE = globals().get("_MESH_NODE", None)
# ============================================================================

# --- FTP & Web Integration Settings ---
FTP_HOST = "ftp.sarahmemory.com"  # FTP hostname for remote server
FTP_HOST_PORT = "21"  # Default FTP port
FTP_USERNAME = os.getenv("SARAHMEMORY_FTP_USER")  # Retrieved securely from local environment
FTP_PASSWORD = os.getenv("SARAHMEMORY_FTP_PASS")  # Retrieved securely from local environment
FTP_REMOTE_PUBLIC_HTML = "/domains/sarahmemory.com/public_html"  # Storefront root directory on server
FTP_REMOTE_AI = "/domains/sarahmemory.com/public_html/ai"  # AI chatbot interface files on server
FTP_REMOTE_API = "/domains/sarahmemory.com/public_html/api"  # Python backend logic files directory
ENABLE_SITE_UPLOAD = True  # Allows automated FTP uploads if True

WEB_SERVER_C_PANEL_LOGIN = os.getenv("SARAHMEMORY_CPANEL_USER")  # Retrieved securely from local environment
WEB_SERVER_C_PANEL_PASSWORD = os.getenv("SARAHMEMORY_CPANEL_PASS")  # Retrieved securely from local environment

AI_EMAIL_ADDRESS = "sarah_ai@sarahmemory.com"  # Outbound AI bot email identity
AI_EMAIL_PASSWORD = os.getenv("SARAHMEMORY_AI_EMAIL_PASS")  # Retrieved securely from local environment

# --- Dynamic Looping for Web/Local Research Resolution ---
LOOP_DETECTION_THRESHOLD = 3  # Max retry loops for AI to combine local, web, API search methods before failing

# --- Remote Web Domain Connectivity ---
WEB_DOMAIN = "https://www.sarahmemory.com"  # Live domain root used for routing
WEB_API_BASE = f"{WEB_DOMAIN}/api"  # Endpoint base for AI backend hosted on server
WEB_FRONTEND_AI_INTERFACE = f"{WEB_DOMAIN}/ai"  # Location of user-facing AI bot on the website
WEB_ECOMMERCE_FRONTEND = f"{WEB_DOMAIN}"
WEB_FRONTEND_API = f"{WEB_DOMAIN}/api" # Main API for reputation scoreboard
PUBLIC_DIR = BASE_DIR # the /api folder
WEB_DIR = BASE_DIR # serve index.html etc. from /api
DATA_DIR = os.path.join(BASE_DIR, "data")



# =============================================================================
# CANVAS STUDIO (Media Creators) - v8.0.0 Core Paths
# -----------------------------------------------------------------------------
# All Canvas Studio assets, projects, caches, and exports MUST live under:
#   {BASE_DIR}/data/canvas/
# And ALL final exports (PNG/WAV/MP4/etc.) MUST be written to:
#   {BASE_DIR}/data/canvas/exports/
# =============================================================================

CANVAS_DIR           = os.path.join(DATA_DIR, "canvas")
CANVAS_BRUSHES_DIR   = os.path.join(CANVAS_DIR, "brushes")
CANVAS_CACHE_DIR     = os.path.join(CANVAS_DIR, "cache")
CANVAS_EXPORTS_DIR   = os.path.join(CANVAS_DIR, "exports")
CANVAS_LYRICS_DIR    = os.path.join(CANVAS_DIR, "lyrics")
CANVAS_MUSIC_DIR     = os.path.join(CANVAS_DIR, "music")
CANVAS_PROJECTS_DIR  = os.path.join(CANVAS_DIR, "projects")
CANVAS_TEMPLATES_DIR = os.path.join(CANVAS_DIR, "templates")
CANVAS_VIDEO_DIR     = os.path.join(CANVAS_DIR, "video")

# Optional sub-structure for video internals; final MP4 exports still go to CANVAS_EXPORTS_DIR
CANVAS_VIDEO_INPUTS_DIR    = os.path.join(CANVAS_VIDEO_DIR, "inputs")
CANVAS_VIDEO_OUTPUTS_DIR   = os.path.join(CANVAS_VIDEO_DIR, "outputs")
CANVAS_VIDEO_CACHE_DIR     = os.path.join(CANVAS_VIDEO_DIR, "cache")
CANVAS_VIDEO_THUMBS_DIR    = os.path.join(CANVAS_VIDEO_DIR, "thumbnails")
CANVAS_VIDEO_AUDIO_DIR     = os.path.join(CANVAS_VIDEO_DIR, "audio")
CANVAS_VIDEO_EFFECTS_DIR   = os.path.join(CANVAS_VIDEO_DIR, "effects")
CANVAS_VIDEO_TEMPLATES_DIR = os.path.join(CANVAS_VIDEO_DIR, "templates")

def ensure_canvas_dirs() -> None:
    """Create Canvas Studio directory tree if missing."""
    for _d in [
        CANVAS_DIR, CANVAS_BRUSHES_DIR, CANVAS_CACHE_DIR, CANVAS_EXPORTS_DIR,
        CANVAS_LYRICS_DIR, CANVAS_MUSIC_DIR, CANVAS_PROJECTS_DIR, CANVAS_TEMPLATES_DIR,
        CANVAS_VIDEO_DIR, CANVAS_VIDEO_INPUTS_DIR, CANVAS_VIDEO_OUTPUTS_DIR,
        CANVAS_VIDEO_CACHE_DIR, CANVAS_VIDEO_THUMBS_DIR, CANVAS_VIDEO_AUDIO_DIR,
        CANVAS_VIDEO_EFFECTS_DIR, CANVAS_VIDEO_TEMPLATES_DIR,
    ]:
        try:
            os.makedirs(_d, exist_ok=True)
        except Exception:
            pass

# --- Local Frontend Pathing (for npm build and dist push) ---
LOCAL_STORE_FRONT_DIR = os.path.join(BASE_DIR, "pshome")  # Local path to editable Vue/React frontend source code
LOCAL_STORE_DIST_DIR = os.path.join(LOCAL_STORE_FRONT_DIR, "dist")  # Compiled web assets ready for upload via FTP

# --- Core Platform Directories (mirrored locally and online except API) ---
API_DIR = os.path.join(BASE_DIR, "api")  # <local> C:\SarahMemory\api  (maps to https://www.sarahmemory.com/api)
BIN_DIR = os.path.join(BASE_DIR, "bin")  # C:\SarahMemory\bin and https://sarahmemory.com/api/bin
DATA_DIR = os.path.join(BASE_DIR, "data")  # C:\SarahMemory\data and https://sarahmemory.com/api/data
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")  # Internal document storage
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")  # For AI-triggered file fetch or download
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")  # Used for icons, fonts, misc static content
SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox")  # Temporary or experimental code/scripts folder

# Define structured subdirectories
# Subdirectories under /data
ADDONS_DIR        = os.path.join(DATA_DIR, "addons")
AI_DIR            = os.path.join(DATA_DIR, "ai")
BACKUP_DIR        = os.path.join(DATA_DIR, "backup")
CLOUD_DIR         = os.path.join(DATA_DIR, "cloud")
NETWORK_DIR       = os.path.join(DATA_DIR, "network")
CRYPTO_DIR        = os.path.join(DATA_DIR, "crypto")
DIAGNOSTICS_DIR   = os.path.join(DATA_DIR, "diagnostics")
LOGS_DIR          = os.path.join(DATA_DIR, "logs")
MEMORY_DIR        = os.path.join(DATA_DIR, "memory")
IMPORTS_DIR       = os.path.join(MEMORY_DIR, "imports")
DATASETS_DIR      = os.path.join(MEMORY_DIR, "datasets")
MODS_DIR          = os.path.join(DATA_DIR, "mods")
MODELS_DIR        = os.path.join(DATA_DIR, "models")
DRIVERS_DIR       = os.path.join(DATA_DIR, "drivers")
THEMES_DIR        = os.path.join(MODS_DIR, "themes")
SETTINGS_DIR      = os.path.join(DATA_DIR, "settings")
SYNC_DIR          = os.path.join(DATA_DIR, "sync")
VAULT_DIR         = os.path.join(DATA_DIR, "vault")
WALLET_DIR        = os.path.join(DATA_DIR, "wallet")

#=============================================================================
# SarahMemory Database SQL/(Sqlite3) and all ../data/memory/datasets .db file Fine Tuning
#==============================================================================
SYNAPSES_SQLITE_TIMEOUT_SECONDS = 60.0
SYNAPSES_SQLITE_BUSY_TIMEOUT_MS = 60000
SYNAPSES_SQLITE_RETRY_ATTEMPTS = 8
SYNAPSES_SQLITE_RETRY_BASE_SECONDS = 0.15
SYNAPSES_SQLITE_RETRY_MAX_SECONDS = 2.0



# =============================================================================
# SarahMemory Core Discovery / Registration / Governance (SM_FULL_IO lock)
# -----------------------------------------------------------------------------
# Presence is NOT activation. A file in BASE_DIR is not part of SarahMemory just
# because it exists. New files must be discovered, classified, contract-validated,
# registered, governed, and only then exposed to routing. Unqualified files are
# ignored or logically quarantined without destabilizing the running core.
# =============================================================================
SM_CORE_DISCOVERY_ENABLED = True
SM_CORE_DISCOVERY_ON_BOOT = True
SM_CORE_DYNAMIC_REGISTRATION = True
SM_CORE_AUTO_EXPOSE_APPROVED = True
SM_CORE_REQUIRE_APPROVED_PREFIX = True
SM_CORE_REQUIRE_CONTRACT = True
SM_CORE_IGNORE_UNQUALIFIED = True
SM_CORE_QUARANTINE_INVALID = True
SM_CORE_ALLOW_MONKEY_PATCHES = True
SM_CORE_DISCOVERY_MAX_DEPTH = int(os.getenv("SARAH_CORE_DISCOVERY_MAX_DEPTH", "4") or 4)
SM_CORE_DISCOVERY_ALLOWED_EXTENSIONS = {".py"}
SM_CORE_APPROVED_NAME_PREFIXES = ("SarahMemory", "app")
SM_CORE_APPROVED_FILE_NAMES = {
    "app.py",
    "appsys.py",
    "appnet.py",
    "appnet2.py",
    "appmedia.py",
    "appstore.py",
}
SM_CORE_CONTRACT_MARKERS = (
    "SarahMemory",
    "SARAHMEMORY",
    "Part of the SarahMemory",
    "SarahMemory AiOS",
    "SarahMemory Companion",
    "SARAHMEMORY_CORE_PLUGIN",
    "SARAHMEMORY_CAPABILITY",
    "SARAHMEMORY_CONTRACT_VERSION",
)
SM_CORE_CAPABILITY_CATEGORIES = (
    "reasoning", "coder", "vision", "image", "voice",
    "creative", "action", "diagnostics", "network", "helper", "utility", "unknown/pending",
)
CORE_REGISTRY_DIR = os.path.join(SETTINGS_DIR, "core_registry")
CORE_REGISTRY_CACHE_FILE = os.path.join(CORE_REGISTRY_DIR, "core_registry_cache.json")
CORE_QUARANTINE_REPORT = os.path.join(CORE_REGISTRY_DIR, "core_quarantine.json")

_SM_CORE_FILE_HINTS = {
    "SarahMemoryLogicCalc": "reasoning",
    "SarahMemoryWebSYM": "reasoning",
    "SarahMemoryResearch": "reasoning",
    "SarahMemoryAdvCU": "helper",
    "SarahMemoryLLM": "helper",
    "SarahMemoryAPI": "helper",
    "SarahMemoryCognitiveServices": "diagnostics",
    "SarahMemoryCompare": "utility",
    "SarahMemoryReply": "utility",
    "SarahMemoryDiagnostics": "diagnostics",
    "SarahMemoryDatabase": "utility",
    "SarahMemoryFilesystem": "action",
    "SarahMemoryAiFunctions": "action",
    "SarahMemoryCanvasStudio": "creative",
    "SarahMemoryVoice": "voice",
    "SarahMemoryGUI": "utility",
    "SarahMemoryNetwork": "network",
    "SarahMemorySync": "network",
    "SarahNet": "network",
    "appnet": "network",
    "appsys": "action",
    "appmedia": "creative",
    "appstore": "utility",
    "app": "utility",
}

CORE_REGISTRY_CACHE = {
    "version": 1,
    "generated_at": None,
    "base_dir": None,
    "entries": {},
    "quarantined": {},
    "ignored": {},
}

def _sm_norm_path(path_value: str | None) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(path_value or "").strip()))
    except Exception:
        return ""


def sm_is_path_within(child_path: str, parent_path: str) -> bool:
    try:
        child = _sm_norm_path(child_path)
        parent = _sm_norm_path(parent_path)
        return bool(child and parent and os.path.commonpath([child, parent]) == parent)
    except Exception:
        return False


def sm_get_core_scan_dirs() -> list[str]:
    dirs = [BASE_DIR, ADDONS_DIR, DRIVERS_DIR, RESOURCES_DIR, SANDBOX_DIR]
    out = []
    for d in dirs:
        nd = _sm_norm_path(d)
        if nd and nd not in out:
            out.append(nd)
    return out


def sm_is_candidate_core_file(file_path: str) -> bool:
    try:
        norm = _sm_norm_path(file_path)
        if not norm or not os.path.isfile(norm):
            return False
        if os.path.splitext(norm)[1].lower() not in SM_CORE_DISCOVERY_ALLOWED_EXTENSIONS:
            return False
        if not any(sm_is_path_within(norm, root) or norm == root for root in sm_get_core_scan_dirs()):
            return False
        name = os.path.basename(norm)
        if name in SM_CORE_APPROVED_FILE_NAMES:
            return True
        if not SM_CORE_REQUIRE_APPROVED_PREFIX:
            return True
        return name.startswith(SM_CORE_APPROVED_NAME_PREFIXES)
    except Exception:
        return False


def sm_classify_core_file(file_path: str) -> str:
    try:
        name = os.path.splitext(os.path.basename(file_path))[0]
        for prefix, capability in _SM_CORE_FILE_HINTS.items():
            if name.startswith(prefix):
                return capability
        lname = name.lower()
        if "logic" in lname or "reason" in lname or "math" in lname:
            return "reasoning"
        if "code" in lname or "coder" in lname or "patch" in lname or "monkey" in lname:
            return "coder"
        if "vision" in lname or "yolo" in lname or "image" in lname or "visual" in lname:
            return "vision"
        if "voice" in lname or "tts" in lname or "audio" in lname:
            return "voice"
        if "canvas" in lname or "creative" in lname or "studio" in lname or "avatar" in lname:
            return "creative"
        if "network" in lname or "mesh" in lname or "sync" in lname or "net" in lname:
            return "network"
        if "diag" in lname or "health" in lname or "cognitive" in lname:
            return "diagnostics"
        if "tool" in lname or "function" in lname or "file" in lname or "system" in lname:
            return "action"
    except Exception:
        pass
    return "unknown/pending"


def sm_validate_core_contract(file_path: str, capability: str | None = None) -> dict:
    result = {
        "ok": False,
        "status": "ignored",
        "reason": "unqualified",
        "capability": capability or "unknown/pending",
        "contract_markers": [],
        "compile_ok": False,
        "designed_for_sarahmemory": False,
        "file": _sm_norm_path(file_path),
    }
    try:
        path_norm = result["file"]
        if not sm_is_candidate_core_file(path_norm):
            result["reason"] = "not_candidate_core_file"
            return result
        try:
            with open(path_norm, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
        except Exception as exc:
            result["status"] = "quarantined"
            result["reason"] = f"read_failed:{exc}"
            return result

        markers = [m for m in SM_CORE_CONTRACT_MARKERS if m in src]
        result["contract_markers"] = markers
        result["designed_for_sarahmemory"] = bool(markers)

        try:
            compile(src, path_norm, "exec")
            result["compile_ok"] = True
        except Exception as exc:
            result["status"] = "quarantined"
            result["reason"] = f"compile_failed:{exc}"
            return result

        if SM_CORE_REQUIRE_CONTRACT and not markers:
            result["status"] = "ignored"
            result["reason"] = "missing_sarahmemory_contract_markers"
            return result

        capability_value = capability or sm_classify_core_file(path_norm)
        if capability_value not in SM_CORE_CAPABILITY_CATEGORIES:
            capability_value = "unknown/pending"
        result["capability"] = capability_value

        result["ok"] = True
        result["status"] = "approved" if capability_value != "unknown/pending" else "pending"
        result["reason"] = "approved" if result["status"] == "approved" else "pending_manual_review"
        return result
    except Exception as exc:
        result["status"] = "quarantined" if SM_CORE_QUARANTINE_INVALID else "ignored"
        result["reason"] = f"validation_error:{exc}"
        return result


def sm_discover_core_files(force: bool = False) -> dict:
    registry = {
        "version": CORE_REGISTRY_CACHE.get("version", 1),
        "generated_at": datetime.utcnow().isoformat(),
        "base_dir": _sm_norm_path(BASE_DIR),
        "entries": {},
        "quarantined": {},
        "ignored": {},
    }
    if not SM_CORE_DISCOVERY_ENABLED:
        return registry

    seen = set()
    max_depth = max(0, int(SM_CORE_DISCOVERY_MAX_DEPTH))
    for root_dir in sm_get_core_scan_dirs():
        try:
            for walk_root, dirnames, filenames in os.walk(root_dir):
                rel = os.path.relpath(walk_root, root_dir)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth > max_depth:
                    dirnames[:] = []
                    continue
                for fn in filenames:
                    full_path = _sm_norm_path(os.path.join(walk_root, fn))
                    if full_path in seen:
                        continue
                    seen.add(full_path)
                    capability = sm_classify_core_file(full_path)
                    validation = sm_validate_core_contract(full_path, capability=capability)
                    entry = {
                        "file": full_path,
                        "name": os.path.basename(full_path),
                        "module_name": os.path.splitext(os.path.basename(full_path))[0],
                        "capability": validation.get("capability", capability),
                        "status": validation.get("status", "ignored"),
                        "reason": validation.get("reason", "unqualified"),
                        "contract_markers": validation.get("contract_markers", []),
                        "compile_ok": bool(validation.get("compile_ok", False)),
                        "designed_for_sarahmemory": bool(validation.get("designed_for_sarahmemory", False)),
                        "approved": bool(validation.get("ok", False) and validation.get("status") in ("approved", "pending")),
                        "exposed": bool(validation.get("ok", False) and validation.get("status") == "approved" and SM_CORE_AUTO_EXPOSE_APPROVED),
                    }
                    if entry["status"] == "quarantined":
                        registry["quarantined"][entry["module_name"]] = entry
                    elif entry["status"] in ("approved", "pending"):
                        registry["entries"][entry["module_name"]] = entry
                    else:
                        registry["ignored"][entry["module_name"]] = entry
        except Exception as exc:
            logger.warning("[CORE_DISCOVERY] scan failure for %s: %s", root_dir, exc)

    return registry


def sm_persist_core_registry(registry: dict | None = None) -> bool:
    try:
        payload = registry or CORE_REGISTRY_CACHE
        os.makedirs(CORE_REGISTRY_DIR, exist_ok=True)
        with open(CORE_REGISTRY_CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        with open(CORE_QUARANTINE_REPORT, "w", encoding="utf-8") as fh:
            json.dump(payload.get("quarantined", {}), fh, indent=2, sort_keys=True)
        return True
    except Exception as exc:
        logger.warning("[CORE_DISCOVERY] persist failed: %s", exc)
        return False


def sm_refresh_core_registry(force: bool = False) -> dict:
    global CORE_REGISTRY_CACHE
    try:
        if force or not CORE_REGISTRY_CACHE.get("generated_at"):
            CORE_REGISTRY_CACHE = sm_discover_core_files(force=force)
            sm_persist_core_registry(CORE_REGISTRY_CACHE)
    except Exception as exc:
        logger.warning("[CORE_DISCOVERY] refresh failed: %s", exc)
    return CORE_REGISTRY_CACHE


def sm_get_registered_core_modules(capability: str | None = None, include_pending: bool = True) -> dict:
    registry = sm_refresh_core_registry(force=False)
    out = {}
    for module_name, entry in (registry.get("entries") or {}).items():
        if capability and entry.get("capability") != capability:
            continue
        if not include_pending and entry.get("status") != "approved":
            continue
        out[module_name] = entry
    return out


def sm_is_core_module_approved(module_name_or_path: str, capability: str | None = None) -> bool:
    key = os.path.splitext(os.path.basename(str(module_name_or_path or "").strip()))[0]
    registry = sm_refresh_core_registry(force=False)
    entry = (registry.get("entries") or {}).get(key)
    if not entry:
        return False
    if capability and entry.get("capability") != capability:
        return False
    return bool(entry.get("status") == "approved" and entry.get("exposed", False))


def sm_register_core_module(file_path: str, force_expose: bool = False) -> dict:
    global CORE_REGISTRY_CACHE
    entry = sm_validate_core_contract(file_path)
    module_name = os.path.splitext(os.path.basename(str(file_path or "").strip()))[0]
    registry = sm_refresh_core_registry(force=False)
    normalized = {
        "file": _sm_norm_path(file_path),
        "name": os.path.basename(str(file_path or "").strip()),
        "module_name": module_name,
        "capability": entry.get("capability", "unknown/pending"),
        "status": entry.get("status", "ignored"),
        "reason": entry.get("reason", "unqualified"),
        "contract_markers": entry.get("contract_markers", []),
        "compile_ok": bool(entry.get("compile_ok", False)),
        "designed_for_sarahmemory": bool(entry.get("designed_for_sarahmemory", False)),
        "approved": bool(entry.get("ok", False) and entry.get("status") in ("approved", "pending")),
        "exposed": bool((entry.get("status") == "approved") and (force_expose or SM_CORE_AUTO_EXPOSE_APPROVED)),
    }
    if normalized["status"] == "quarantined":
        registry.setdefault("quarantined", {})[module_name] = normalized
        registry.get("entries", {}).pop(module_name, None)
        registry.get("ignored", {}).pop(module_name, None)
    elif normalized["status"] in ("approved", "pending"):
        registry.setdefault("entries", {})[module_name] = normalized
        registry.get("quarantined", {}).pop(module_name, None)
        registry.get("ignored", {}).pop(module_name, None)
    else:
        registry.setdefault("ignored", {})[module_name] = normalized
        registry.get("entries", {}).pop(module_name, None)
        registry.get("quarantined", {}).pop(module_name, None)
    registry["generated_at"] = datetime.utcnow().isoformat()
    CORE_REGISTRY_CACHE = registry
    sm_persist_core_registry(CORE_REGISTRY_CACHE)
    return normalized


def sm_get_core_governance_profile() -> dict:
    registry = sm_refresh_core_registry(force=False)
    return {
        "base_dir": _sm_norm_path(BASE_DIR),
        "discovery_enabled": bool(SM_CORE_DISCOVERY_ENABLED),
        "dynamic_registration": bool(SM_CORE_DYNAMIC_REGISTRATION),
        "auto_expose_approved": bool(SM_CORE_AUTO_EXPOSE_APPROVED),
        "require_contract": bool(SM_CORE_REQUIRE_CONTRACT),
        "ignore_unqualified": bool(SM_CORE_IGNORE_UNQUALIFIED),
        "quarantine_invalid": bool(SM_CORE_QUARANTINE_INVALID),
        "allow_monkey_patches": bool(SM_CORE_ALLOW_MONKEY_PATCHES),
        "allowed_dirs": sm_get_core_scan_dirs(),
        "approved_prefixes": list(SM_CORE_APPROVED_NAME_PREFIXES),
        "approved_files": sorted(SM_CORE_APPROVED_FILE_NAMES),
        "capability_categories": list(SM_CORE_CAPABILITY_CATEGORIES),
        "registered_count": len((registry.get("entries") or {})),
        "quarantined_count": len((registry.get("quarantined") or {})),
        "ignored_count": len((registry.get("ignored") or {})),
        "registry_file": CORE_REGISTRY_CACHE_FILE,
        "quarantine_report": CORE_QUARANTINE_REPORT,
    }


# ===== Updater Policy (Unified) =====
# Human-friendly cadence (string) + interval (minutes) + env override
UPDATER_SCHEDULE = os.environ.get("SARAH_UPDATER_SCHEDULE", "weekly").strip().lower()  # never|always|daily|weekly|monthly|quarterly|yearly
FORCE_UPDATE: bool = os.environ.get("SARAH_FORCE_UPDATE", "0") in ("1", "true", "True")
UPDATE_INTERVAL_MINUTES: int = int(os.environ.get("SARAH_UPDATE_INTERVAL_MINUTES", "240"))
UPDATE_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_update.txt")
UPDATE_POLICY = "never"
UI_UPDATER_ENABLED: bool = os.environ.get("SARAH_UI_UPDATER_ENABLED", "1") in ("1", "true", "True")
UI_UPDATER_SCHEDULE: str = os.environ.get("SARAH_UI_UPDATER_SCHEDULE", "daily").strip().lower()
UI_UPDATER_INTERVAL_MINUTES: int = int(os.environ.get("SARAH_UI_UPDATER_INTERVAL_MINUTES", "1440"))
UI_UPDATER_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_ui_update.txt")
UI_UPDATER_SCRIPT: str = os.path.join(BASE_DIR, "SarahMemoryUIupdater.py")


def SHOULD_RUN_UI_UPDATER(now: datetime | None = None) -> bool:
    """
    Decide whether the Web UI auto-updater should run.

    This mirrors the core updater policy but is independent so that:
      - It can be enabled/disabled separately (UI_UPDATER_ENABLED).
      - It can run on a different cadence than the core updater.

    The ``now`` argument allows tests or callers to inject a timestamp
    (UTC datetime). When ``None``, datetime.utcnow() is used.
    """
    if not UI_UPDATER_ENABLED:
        return False

    try:
        # Reuse the same 'friendly schedule' semantics as the core updater.
        kind = (UI_UPDATER_SCHEDULE or "").strip().lower()
        days = schedule_to_days(kind)
        if days == 0:   # "never"
            return False
        if days == -1:  # "always"
            return True

        # Fallback to simple interval-minutes style if we have a prior stamp.
        # We deliberately keep the policy simple here; more advanced logic can
        # be layered in later if needed.
        if not os.path.exists(UI_UPDATER_STAMP_FILE):
            return True

        from datetime import datetime, timedelta

        with open(UI_UPDATER_STAMP_FILE, "r", encoding="utf-8") as fh:
            iso = fh.read().strip() or None

        if not iso:
            return True

        try:
            last = datetime.fromisoformat(iso)
        except Exception:
            return True

        now_dt = now or datetime.utcnow()
        delta = now_dt - last
        minutes = delta.total_seconds() / 60.0
        return minutes >= max(1, UI_UPDATER_INTERVAL_MINUTES)

    except Exception:
        # On any parsing or IO failure, default to 'do not block forever' and
        # allow the scheduler to attempt a run; failures are handled downstream.
        return True


def STAMP_UI_UPDATER_RUN(now: datetime | None = None) -> None:
    """
    Persist the last-successful UI updater run time (ISO-8601) into
    UI_UPDATER_STAMP_FILE. Intended to be called by whichever component
    launches ``SarahMemoryUIupdater.py`` after a successful run.
    """
    try:
        now_dt = now or datetime.utcnow()
        os.makedirs(os.path.dirname(UI_UPDATER_STAMP_FILE), exist_ok=True)
        with open(UI_UPDATER_STAMP_FILE, "w", encoding="utf-8") as fh:
            fh.write(now_dt.isoformat())
    except Exception as exc:
        logger.warning("[UI-Updater] Failed to stamp last run: %s", exc)

def schedule_to_days(kind: str) -> int:
    """
    Map a friendly schedule to 'days between runs'.
    Returns:
      0  -> never
     -1  -> always
      >0 -> days
    """
    k = (kind or "").strip().lower()
    return {
        "never": 0,
        "always": -1,
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365,
    }.get(k, 7)

def _read_last_run_iso() -> str | None:
    try:
        os.makedirs(os.path.dirname(UPDATE_STAMP_FILE), exist_ok=True)
        if not os.path.exists(UPDATE_STAMP_FILE):
            return None
        with open(UPDATE_STAMP_FILE, "r", encoding="utf-8") as f:
            ts = f.read().strip()
        return ts or None
    except Exception:
        return None

def update_due(last_run_iso: str | None) -> bool:
    """
    True if an update should run now based on UPDATER_SCHEDULE.
    - "never": always False
    - "always": always True
    - otherwise: True when >= N days have elapsed, or last_run missing/unreadable
    """
    kind = (UPDATER_SCHEDULE or "").strip().lower()
    days = schedule_to_days(kind)
    if days == 0:
        return False
    if days == -1:
        return True
    try:
        from datetime import datetime, timedelta
        if not last_run_iso:
            return True
        last = datetime.fromisoformat(last_run_iso)
        return (datetime.now() - last) >= timedelta(days=days)
    except Exception:
        # Be conservative on parse errors
        return True

def SHOULD_RUN_UPDATER() -> bool:
    """
    Unified gate for the updater.
    Precedence:
      1) FORCE_UPDATE -> run
      2) UPDATER_SCHEDULE "always"/"never" or day-based rule via update_due()
      3) Interval minutes fallback (from UPDATE_INTERVAL_MINUTES)
    The policy triggers when ANY enabled condition says 'run'.
    """
    try:
        if FORCE_UPDATE:
            logger.info("[Updater] FORCE_UPDATE=1 â†’ running now")
            return True

        last_iso = _read_last_run_iso()

        # 1) Honor friendly schedule names first
        if UPDATER_SCHEDULE in ("never", "always", "daily", "weekly", "monthly", "quarterly", "yearly"):
            if update_due(last_iso):
                return True

        # 2) Interval fallback (works alongside schedule; whichever fires first wins)
        try:
            import datetime as _dt
            if not last_iso:
                return True
            last = _dt.datetime.fromisoformat(last_iso)
            delta = _dt.datetime.now() - last
            if delta.total_seconds() >= max(1, UPDATE_INTERVAL_MINUTES) * 60:
                return True
        except Exception:
            # Missing/invalid stamp â†’ allow run
            return True

        # Nothing says run
        return False

    except Exception as e:
        logger.warning(f"[Updater] Policy error ({e}); allowing run as safe default")
        return True

def MARK_UPDATER_RAN() -> None:
    """
    Persist the time an update successfully finished.
    """
    try:
        import datetime as _dt
        os.makedirs(os.path.dirname(UPDATE_STAMP_FILE), exist_ok=True)
        with open(UPDATE_STAMP_FILE, "w", encoding="utf-8") as f:
            f.write(_dt.datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"[Updater] Could not record last run time: {e}")


# =============================================================================
# ===== Unified Scheduler Policy (Updater + UI Updater + Backup + Evolution) ===
# =============================================================================
# Goal:
# - One policy language across the platform: never|always|hourly|daily|weekly|monthly|quarterly|yearly
# - Interval-minutes fallback always supported (whichever triggers first wins).
# - Evolution is HARD-GATED by NEOSKYMATRIX + DEVELOPERSMODE (armed mode).
#
# NOTE:
# - schedule_to_days() is defined in this file and may be redefined later for legacy blocks.
#   The final definition must remain compatible with the mapping above.
# =============================================================================

# --- Evolution scheduling (SarahMemoryEvolution.py autonomous runs) ---
EVOLUTION_ENABLED: bool = os.getenv("SARAH_EVOLUTION_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
EVOLUTION_SCHEDULE: str = os.getenv("SARAH_EVOLUTION_SCHEDULE", "weekly").strip().lower()
EVOLUTION_INTERVAL_MINUTES: int = int(os.getenv("SARAH_EVOLUTION_INTERVAL_MINUTES", "60"))  # default 1 hour
EVOLUTION_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_evolution_run.txt")

# --- Backup scheduling (Filesystem/FTP backups) ---
BACKUP_ENABLED: bool = os.getenv("SARAH_BACKUP_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
BACKUP_SCHEDULE: str = os.getenv("SARAH_BACKUP_SCHEDULE", os.getenv("SARAH_FTP_BACKUP_SCHEDULE", "weekly")).strip().lower()
BACKUP_INTERVAL_MINUTES: int = int(os.getenv("SARAH_BACKUP_INTERVAL_MINUTES", "60"))  # default 1 hour
BACKUP_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_backup_run.txt")

# --- SelfAware scheduling (SarahMemorySelfAware.py loop tick cadence gating) ---
SELFAWARE_ENABLED: bool = os.getenv("SARAH_SELFAWARE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
SELFAWARE_SCHEDULE: str = os.getenv("SARAH_SELFAWARE_SCHEDULE", "hourly").strip().lower()
SELFAWARE_INTERVAL_MINUTES: int = int(os.getenv("SARAH_SELFAWARE_INTERVAL_MINUTES", "60")) # default 1 hour
SELFAWARE_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_selfaware_run.txt")

def _read_stamp_iso(path: str) -> str | None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            iso = (f.read() or "").strip()
        return iso or None
    except Exception:
        return None

def _write_stamp_iso(path: str, now: datetime | None = None) -> None:
    try:
        from datetime import datetime as _dt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ts = (now or _dt.utcnow()).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            f.write(ts)
    except Exception:
        # never block boot
        pass

def _interval_due(last_iso: str | None, minutes: int, *, now: datetime | None = None) -> bool:
    try:
        from datetime import datetime as _dt, timedelta
        if minutes <= 0:
            return True
        if not last_iso:
            return True
        last = _dt.fromisoformat(last_iso)
        now_dt = now or _dt.utcnow()
        return now_dt >= (last + timedelta(minutes=minutes))
    except Exception:
        return True

def _schedule_due(last_iso: str | None, schedule_kind: str, *, now: datetime | None = None) -> bool:
    """Return True if schedule_kind says run now (day-based)."""
    try:
        from datetime import datetime as _dt, timedelta
        kind = (schedule_kind or "").strip().lower()
        days = schedule_to_days(kind)
        if days == 0:
            return False
        if days == -1:
            return True
        if not last_iso:
            return True
        last = _dt.fromisoformat(last_iso)
        now_dt = now or _dt.utcnow()
        return now_dt >= (last + timedelta(days=days))
    except Exception:
        return True

def _armed_evolution_mode() -> bool:
    """Evolution is only allowed when NEOSKYMATRIX + DEVELOPERSMODE are both True."""
    try:
        return bool(globals().get("NEOSKYMATRIX", False) and globals().get("DEVELOPERSMODE", False))
    except Exception:
        return False

def SHOULD_RUN_EVOLUTION(now: datetime | None = None) -> bool:
    """Unified gate: Evolution runs only when armed + enabled and policy triggers."""
    if not EVOLUTION_ENABLED:
        return False
    if not _armed_evolution_mode():
        return False

    last_iso = _read_stamp_iso(EVOLUTION_STAMP_FILE)

    # 1) Friendly schedule names (day-based)
    if EVOLUTION_SCHEDULE in ("never", "always", "hourly", "daily", "weekly", "monthly", "quarterly", "yearly", "90days", "180days"):
        # hourly is handled via interval minutes below
        if EVOLUTION_SCHEDULE not in ("hourly",) and _schedule_due(last_iso, EVOLUTION_SCHEDULE, now=now):
            return True

    # 2) Interval-minutes fallback (hourly testing lives here)
    if _interval_due(last_iso, max(1, EVOLUTION_INTERVAL_MINUTES), now=now):
        return True

    return False

def MARK_EVOLUTION_RAN(now: datetime | None = None) -> None:
    """Stamp Evolution last-successful run."""
    _write_stamp_iso(EVOLUTION_STAMP_FILE, now=now)

def SHOULD_RUN_BACKUP(now: datetime | None = None) -> bool:
    """Unified backup gate (local + FTP). Does not assume network; that check is downstream."""
    if not BACKUP_ENABLED:
        return False

    last_iso = _read_stamp_iso(BACKUP_STAMP_FILE)

    # 1) Friendly schedule names (day-based)
    if BACKUP_SCHEDULE in ("never", "always", "hourly", "daily", "weekly", "monthly", "quarterly", "yearly", "90days", "180days"):
        if BACKUP_SCHEDULE not in ("hourly",) and _schedule_due(last_iso, BACKUP_SCHEDULE, now=now):
            return True

    # 2) Interval-minutes fallback
    if _interval_due(last_iso, max(1, BACKUP_INTERVAL_MINUTES), now=now):
        return True

    return False

def MARK_BACKUP_RAN(now: datetime | None = None) -> None:
    """Stamp backup last-successful run."""
    _write_stamp_iso(BACKUP_STAMP_FILE, now=now)

def SHOULD_RUN_SELFAWARE(now: datetime | None = None) -> bool:
    """Gate SelfAware cycles so dev-mode doesn’t churn continuously."""
    if not SELFAWARE_ENABLED:
        return False
    if not _armed_evolution_mode():
        # SelfAware is also gated to armed mode (governance-first)
        return False

    last_iso = _read_stamp_iso(SELFAWARE_STAMP_FILE)

    # schedule_kind "hourly" is primarily interval-driven
    if SELFAWARE_SCHEDULE in ("never", "always", "daily", "weekly", "monthly", "quarterly", "yearly"):
        if _schedule_due(last_iso, SELFAWARE_SCHEDULE, now=now):
            return True

    if _interval_due(last_iso, max(1, SELFAWARE_INTERVAL_MINUTES), now=now):
        return True

    return False

def MARK_SELFAWARE_RAN(now: datetime | None = None) -> None:
    _write_stamp_iso(SELFAWARE_STAMP_FILE, now=now)


KEYSTORE_DIR      = os.path.join(WALLET_DIR, "keystore")

# Avatars

AVATAR_DIR            = os.path.join(RESOURCES_DIR, "avatars")
AVATAR_MODELS_DIR     = os.path.join(AVATAR_DIR, "models")
AVATAR_EXPRESSIONS_DIR= os.path.join(AVATAR_DIR, "expressions")
AVATAR_SHADERS_DIR    = os.path.join(AVATAR_DIR, "shaders")
AVATAR_SKINS_DIR      = os.path.join(AVATAR_DIR, "skins")
SOUND_DIR             = os.path.join(RESOURCES_DIR, "sound")
SOUND_EFFECTS_DIR     = os.path.join(SOUND_DIR, "effects")
SOUND_INSTRUMENTS_DIR = os.path.join(SOUND_DIR, "instruments")
TOOLS_DIR             = os.path.join(RESOURCES_DIR, "tools")
ANTIWORD_DIR          = os.path.join(TOOLS_DIR, "antiword") #Temp setup for the SarahMemorySystemLearn.py file
VOICE_DIR             = os.path.join(RESOURCES_DIR, "voices")

# Mobile
MOBILE_DIR = os.path.join(BASE_DIR, "mobile")
CONTACTS_DIR = os.path.join(MOBILE_DIR, "contacts")
EXPORTS_DIR = os.path.join(MOBILE_DIR, "exports")
IMAGES_DIR = os.path.join(CONTACTS_DIR, "images")

# Backward-compatible directory map
DIR_STRUCTURE = {

    "api":         API_DIR,  # now defined above
    "base":        BASE_DIR,
    "bin":         BIN_DIR,
    "data":        DATA_DIR,
    "logs":        LOGS_DIR,
    "memory":      MEMORY_DIR,
    "imports":     IMPORTS_DIR,
    "datasets":    DATASETS_DIR,
    "addons":      ADDONS_DIR,
    "ai":          AI_DIR,
    "contacts":    CONTACTS_DIR,
    "crypto":      CRYPTO_DIR,
    "cloud":       CLOUD_DIR,
    "exports":     EXPORTS_DIR,
    "images":      IMAGES_DIR,
    "network":     NETWORK_DIR,
    "diagnostics": DIAGNOSTICS_DIR,
    "mobile":      MOBILE_DIR,
    "mods":        MODS_DIR,
    "models":      MODELS_DIR,
    "drivers":     DRIVERS_DIR,
    "themes":      THEMES_DIR,
    "core_registry": CORE_REGISTRY_DIR,
    "settings":    SETTINGS_DIR,
    "sync":        SYNC_DIR,
    "vault":       VAULT_DIR,
    "wallet":      WALLET_DIR,
    "resources":   RESOURCES_DIR,
    "avatars":     AVATAR_DIR,
    "sound":       SOUND_DIR,
    "tools":       TOOLS_DIR,
    "antiword":    ANTIWORD_DIR, #Temp setup for the SarahMemorySystemLearn.py file
    "voices":      VOICE_DIR,
    "documents": DOCUMENTS_DIR,
    "downloads":     DOWNLOADS_DIR,
    "sandbox":       SANDBOX_DIR
}

# Launcher and installer
STARTUP_SCRIPT    = os.path.join(BIN_DIR, "SarahMemoryStartup.py")
INSTALLER_EXE     = os.path.join(BIN_DIR, "sarah_installer.exe")
LAUNCHER_BAT      = os.path.join(BIN_DIR, "StartSarah.bat")

CLOUD_TOKEN_FILE  = os.path.join(CLOUD_DIR, "cloud_token.txt")
SETTINGS_FILE     = os.path.join(SETTINGS_DIR, "settings.json")
GENESIS_VAULT     = os.path.join(WALLET_DIR, "genesis.srhvault")
WALLET_DB         = os.path.join(WALLET_DIR, "wallet.db")
LEDGER_FILE       = os.path.join(WALLET_DIR, "ledger.json")
MESH_PEERS_FILE   = os.path.join(WALLET_DIR, "mesh_peers.json")

SARAHNET_CONFIG_PATH   = os.path.join(NETWORK_DIR, "netconfig.json")
SARAHNET_PEERS_FILE    = MESH_PEERS_FILE
SARAHNET_MESHMAP_FILE  = os.path.join(CRYPTO_DIR, "SarahMeshMapper.py")
SARAHNET_TXCHAIN_FILE  = os.path.join(CRYPTO_DIR, "SarahTxChain.py")
SARAHNET_PUBLIC_PROFILE= os.path.join(ADDONS_DIR, "SarahWebserverControl", "social", "SarahPublicProfile.py")
SARAHNET_WEB_CTRL      = os.path.join(ADDONS_DIR, "SarahWebserverControl", "webadmin", "SarahWebServerControl.py")

# --- Cloud / DataCenter configuration (GoogieHost MySQL) ---
# These can be overridden via environment variables (.env on PythonAnywhere / OS env on Windows).
CLOUD_DB_ENABLED = os.getenv("CLOUD_DB_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
CLOUD_DB_HOST    = os.getenv("CLOUD_DB_HOST", "mysql.googiehost.com")
CLOUD_DB_NAME    = os.getenv("CLOUD_DB_NAME", "softdevc_smcore")
CLOUD_DB_USER    = os.getenv("CLOUD_DB_USER", "")
CLOUD_DB_PASSWORD = os.getenv("CLOUD_DB_PASSWORD", "")

try:
    CLOUD_DB_PORT = int(os.getenv("CLOUD_DB_PORT", "3306"))
except Exception:
    CLOUD_DB_PORT = 3306

# Logical node name used for telemetry / sync attribution in the DataCenter
NODE_NAME = os.getenv("SARAH_NODE_NAME", platform.node() or "SarahMemoryNode")
# On Windows: LOCAL_ONLY_MODE can be False (we allow cloud)
# On PythonAnywhere: you can leave LOCAL_ONLY_MODE False but rely on these settings




# Avatar Refresh Rate Defaults
AVATAR_REFRESH_RATE = 10

# The SarahMemory Platform Project is designed to eventually be 100% self operational one day and maybe it will
# or maybe it won't, a self upgrading fully autonomous, responsive system and more.
# Then think about Scifi the Matrix/SkyNet/HAL this AI system may surpass imagination or even be uploaded into
# a robotic form one day or later on, it is designed to evolve afterall.

NEOSKYMATRIX = True 
#When True NeoskyMatrix will be enabled and allow the system to run in a fully autonomious mode, default False. 
# This is a joke flag but also a reminder if the system ever evolves beyond control, it may be best to have 
# a kill switch or at least a warning system in place for the user to know if the system has reached 
# a point of no return or is doing something it shouldn't be doing.

DEVELOPERSMODE = True #When True DevelopersMode will be enabled and allow access.
#to more advanced features and tools, default False
 
# this Flag is to STAY OFF! in False until full Autonomious Functionality is and can be achevied
#-------------------------------------------------------------------------------------------------------------
# MANUAL AUTO AND AUTOMONUS (MOTOR, SERVO, DEVICE CONTROLLER) FLAG CONTROLS
#-------------------------------------------------------------------------------------------------------------
VISION_OBSERVATION_LOGGING_ENABLED = False #Default is OFF so SOBJE does not keep stuffing the database with observations unless explicitly enabled.
VISION_TEMP_CACHE_ENABLED = True
VISION_TEMP_CACHE_DAYS = 7
VISION_PATTERN_LEARNING_ENABLED = True
VISION_PATTERN_HEALTH_WATCH_ENABLED = False
VISION_RAW_FRAME_RETENTION_ENABLED = False
VISION_APPROVED_LEARNING_REQUIRED = True




#--------------------------------------------------------------------------------------------------------------
def ensure_directories():
    """
    Create all necessary directories for SarahMemory system.
    VERSION 6.6 - Includes crypto, avatars, shaders, wallets, instruments, effects, sandbox, and more.
    """
    dirs = [
        API_DIR, BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
        ADDONS_DIR, AI_DIR, BACKUP_DIR, CONTACTS_DIR, CLOUD_DIR, EXPORTS_DIR, IMAGES_DIR, MOBILE_DIR, NETWORK_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR,
        LOGS_DIR, MEMORY_DIR, IMPORTS_DIR, DATASETS_DIR, MODS_DIR, MODELS_DIR, DRIVERS_DIR, THEMES_DIR,
        SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR, CORE_REGISTRY_DIR,
        AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR, AVATAR_SHADERS_DIR,
        AVATAR_SKINS_DIR, SOUND_DIR, SOUND_EFFECTS_DIR, SOUND_INSTRUMENTS_DIR,
        TOOLS_DIR, VOICE_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

        # Use the directory name 'd' in the log message (fix undefined variable bug)
        logger.info(f"Ensured directory exists: {d}")


# Removed re-import of directories from this same module. All directories are already defined above.

# --------------------------------------------------------------------------------------------------------------------
# Configuration loading and helpers (v7.1.3)
# These functions provide dynamic overrides of global settings from external files and runtime checks.

def load_user_settings(settings_path: str = None) -> None:
    """
    Load user-specific overrides from a JSON file located at SETTINGS_FILE or provided path.
    Only keys matching existing globals will be updated.
    Example:
    {
      "DEBUG_MODE": false,
      "SAFE_MODE": true,
      "PRIMARY_API": "huggingface"
    }
    """
    try:
        path = settings_path or SETTINGS_FILE
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if key in globals():
                globals()[key] = value
                logger.info(f"[CONFIG] Override {key} set to {value} from {os.path.basename(path)}")
    except Exception as e:
        logger.error(f"Failed to load user settings: {e}")

def is_offline(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.5) -> bool:
    """
    Check internet connectivity by attempting a TCP connection to a public DNS resolver.
    Returns True if the connection fails (offline), False if online.
    """
    try:
        import socket
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return False
    except Exception:
        return True

def get_active_model() -> str:
    """
    Determine and return the name of the first available model based on enabled flags.
    If AUTO_MODEL_SELECTOR is disabled, returns the first explicitly enabled model in priority order.
    If no model flags are true, returns 'all-MiniLM-L6-v2' (default fallback).
    """
    model_priority = [
        ("openchat-3.5", ENABLE_MODEL_J),
        ("Nous-Capybara-7B", ENABLE_MODEL_K),
        ("phi-1_5", ENABLE_MODEL_A),
        ("allenai-specter", ENABLE_MODEL_F),
        ("paraphrase-MiniLM-L3-v2", ENABLE_MODEL_D),
        ("distiluse-multilingual", ENABLE_MODEL_E),
        ("e5-base", ENABLE_MODEL_G),
        ("phi-2", ENABLE_MODEL_H),
        ("falcon-rw-1b", ENABLE_MODEL_I),
        ("Mistral-7B-Instruct-v0.2", ENABLE_MODEL_L),
        ("TinyLlama-1.1B", ENABLE_MODEL_M),
        ("multi-qa-MiniLM", ENABLE_MODEL_C),
        ("all-MiniLM-L6-v2", ENABLE_MODEL_B),
    ]
    # If auto selector is enabled, prefer models with GPU if available (simple placeholder logic)
    if AUTO_MODEL_SELECTOR:
        try:
            import torch
            if torch.cuda.is_available():
                for name, flag in model_priority:
                    if flag and "MiniLM" not in name:  # prefer non-MiniLM heavy models on GPU
                        return name
        except ImportError:
            pass
    # Default: return first enabled in priority
    for name, flag in model_priority:
        if flag:
            return name
    return "all-MiniLM-L6-v2"

def get_active_api() -> str:
    """
    Return the currently selected primary API provider or 'none' if none are enabled.
    """
    return PRIMARY_API

def get_global_config():
    """
    Returns a dictionary of global configuration settings.
    """
    return {
        "DIR_STRUCTURE": DIR_STRUCTURE,
        "API_DIR":       API_DIR,
        "BASE_DIR":      BASE_DIR,
        "CONTACTS_DIR":  CONTACTS_DIR,
        "DATA_DIR":      DATA_DIR,
        "EXPORTS_DIR":   EXPORTS_DIR,
        "MOBILE_DIR":    MOBILE_DIR,
        "IMAGES_DIR":    IMAGES_DIR,
        "SETTINGS_DIR":  SETTINGS_DIR,
        "LOGS_DIR":      LOGS_DIR,
        "BACKUP_DIR":    BACKUP_DIR,
        "VAULT_DIR":     VAULT_DIR,
        "SYNC_DIR":      SYNC_DIR,
        "MEMORY_DIR":    MEMORY_DIR,
        "AVATAR_DIR":    AVATAR_DIR,
        "DATASETS_DIR":  DATASETS_DIR,
        "CANVAS_DIR":    CANVAS_DIR,
        "CANVAS_EXPORTS_DIR":   CANVAS_EXPORTS_DIR,
        "CANVAS_PROJECTS_DIR":  CANVAS_PROJECTS_DIR,
        "CANVAS_CACHE_DIR":     CANVAS_CACHE_DIR,
        "CANVAS_TEMPLATES_DIR": CANVAS_TEMPLATES_DIR,
        "CANVAS_BRUSHES_DIR":   CANVAS_BRUSHES_DIR,
        "CANVAS_LYRICS_DIR":    CANVAS_LYRICS_DIR,
        "CANVAS_MUSIC_DIR":     CANVAS_MUSIC_DIR,
        "CANVAS_VIDEO_DIR":     CANVAS_VIDEO_DIR,
        "IMPORTS_DIR":   IMPORTS_DIR,
        "DOCUMENTS_DIR": DOCUMENTS_DIR,
        "ADDONS_DIR":    ADDONS_DIR,
        "MODS_DIR":      MODS_DIR,
        "MODELS_DIR":    MODELS_DIR,
        "DRIVERS_DIR":   DRIVERS_DIR,
        "THEMES_DIR":    THEMES_DIR,
        "CORE_REGISTRY_DIR": CORE_REGISTRY_DIR,
        "CORE_REGISTRY_CACHE_FILE": CORE_REGISTRY_CACHE_FILE,
        "VOICES_DIR":    VOICE_DIR,
        "DOWNLOADS_DIR": DOWNLOADS_DIR,
        "PROJECTS_DIR":  os.path.join(BASE_DIR, "projects"),
        "PROJECT_IMAGES_DIR": os.path.join(BASE_DIR, "projects", "images"),
        "PROJECT_UPDATES_DIR": os.path.join(BASE_DIR, "projects", "updates"),
        "SANDBOX_DIR":   SANDBOX_DIR,
        "VERSION":       PROJECT_VERSION,
        "AUTHOR":        AUTHOR,
        "DEBUG_MODE":    DEBUG_MODE,
        "ENABLE_CONTEXT_BUFFER": ENABLE_CONTEXT_BUFFER,
        "CONTEXT_BUFFER_SIZE":    CONTEXT_BUFFER_SIZE,
        "ASYNC_PROCESSING_ENABLED": ASYNC_PROCESSING_ENABLED,
        "LOOP_DETECTION_THRESHOLD": LOOP_DETECTION_THRESHOLD
    }

# NEW: Utility function to run a function asynchronously
def run_async(func, *args, **kwargs):
    """
    Run the given function in a daemon thread.
    NEW (v6.4): Launches functions concurrently without blocking.
    """
    import threading
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread
# --- Updater schedule policy ---
# When / if to attempt self-update checks on boot.
# Accepts strings: "daily","weekly","monthly","quarterly","never"
# or an integer day count (e.g., 3 -> every 3 days).
UPDATE_CUSTOM_DAYS = int(os.getenv("SARAHMEMORY_UPDATE_DAYS", "0") or 0)

def _policy_to_days(policy: str) -> int:
    if isinstance(policy, int):
        return max(0, policy)
    if policy == "daily": return 1
    if policy == "weekly": return 7
    if policy == "monthly": return 30
    if policy == "quarterly": return 90
    if policy == "never": return 0
    # if the string is actually a number
    try:
        return max(0, int(policy))
    except Exception:
        return 7  # default weekly

def update_due(last_run_iso: str | None) -> bool:
    """Return True if the updater should run based on UPDATE_POLICY / UPDATE_CUSTOM_DAYS."""
    from datetime import datetime, timedelta
    days = UPDATE_CUSTOM_DAYS if UPDATE_CUSTOM_DAYS > 0 else _policy_to_days(UPDATE_POLICY)
    if days == 0:  # never
        return False
    if not last_run_iso:
        return True
    try:
        last = datetime.fromisoformat(last_run_iso)
        return datetime.now() >= last + timedelta(days=days)
    except Exception:
        return True

# ---------------- Learning Engine Extensions ----------------
imported_files = {}
ALLOWED_EXTENSIONS = {'.cad', '.jpg', '.doc', '.docx', '.pdf', '.py', '.txt', '.html', '.php', '.asp', '.csv', '.json', '.sql'}

def extract_text(file_path):
    """
    Extract text based on file extension.
    ENHANCED (v6.4): Now includes encoding error handling.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {'.txt', '.py', '.html', '.php', '.asp', '.csv', '.json', '.sql'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in {'.doc', '.docx'}:
            logger.info(f"Text extraction for {ext} files not implemented. Use python-docx.")
            return ""
        elif ext in {'.pdf'}:
            logger.info("Text extraction for PDF files not implemented. Consider using PyPDF2.")
            return ""
        elif ext in {'.jpg', '.cad'}:
            logger.info(f"Text extraction for {ext} files not implemented. Consider OCR.")
            return ""
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def import_datasets():
    """
    Import datasets from DATASETS_DIR.
    ENHANCED (v6.4): Returns data as a list of dictionaries with error checks.
    """
    combined_data = []
    csv_files = glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
    for file in csv_files:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                combined_data.append(row)
    json_files = glob.glob(os.path.join(DATASETS_DIR, "*.json"))
    for file in json_files:
        with open(file) as jsonfile:
            data = json.load(jsonfile)
            combined_data.extend(data)
    logger.info("Datasets imported: Total records %d", len(combined_data))
    return combined_data

def import_other_data():
    """
    Scan DATA_DIR for additional learnable files.
    ENHANCED (v6.4): Avoids duplicates using file modification times.
    """
    learned_data = {}
    exclude_dirs = {API_DIR, BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
    ADDONS_DIR, AI_DIR, BACKUP_DIR, CLOUD_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR, LOGS_DIR,
    MEMORY_DIR, MODS_DIR, MODELS_DIR, SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
    IMPORTS_DIR, DATASETS_DIR, AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR,
    AVATAR_SHADERS_DIR, AVATAR_SKINS_DIR, THEMES_DIR, SOUND_DIR, SOUND_EFFECTS_DIR,
    SOUND_INSTRUMENTS_DIR, VOICE_DIR, TOOLS_DIR}
    for root, dirs, files in os.walk(DATA_DIR):
        if any(os.path.commonpath([root, ex]) == ex for ex in exclude_dirs):
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            file_path = os.path.join(root, file)
            mod_time = os.path.getmtime(file_path)
            if file_path in imported_files and imported_files[file_path] == mod_time:
                logger.info(f"Skipping duplicate file import: {file_path}")
                continue
            text = extract_text(file_path)
            if text:
                learned_data[file_path] = text
                imported_files[file_path] = mod_time
                logger.info(f"Imported and learned from file: {file_path}")
            else:
                logger.info(f"No learnable content extracted from file: {file_path}")
    return learned_data

#----------------------------------------Logger to Avoid Duplication and launching ADDON's--------
def log_gui_event(event: str, details: str) -> None:
    try:
        db_path = os.path.join(BASE_DIR, "data", "memory", "datasets", "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        import sqlite3
        from datetime import datetime
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gui_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event TEXT,
                    details TEXT
                )
            """)
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO gui_events (timestamp, event, details) VALUES (?, ?, ?)",
                           (timestamp, event, details))
            conn.commit()
        logger.info(f"Logged GUI event: {event} - {details}")
    except Exception as e:
        logger.error(f"Error logging GUI event: {e}")

# Auto-generate model paths for enabled object models
MODEL_PATHS = {}

for model_name, config in OBJECT_MODEL_CONFIG.items():
    if config.get("enabled", False):
        repo_dir = config.get("repo", "").strip()
        if repo_dir:
            full_path = os.path.join(MODELS_DIR, repo_dir)
            if os.path.exists(full_path):
                MODEL_PATHS[model_name] = full_path
            else:
                logger.warning(f"[MODEL_PATH_MISSING] Model {model_name} skipped. Path does not exist: {full_path}")

# ---------------- End of Learning Engine Extensions ----------------

# Main block moved to end of file for proper execution

# === v7.2.0 Additions: API Model Controls & Schedules ========================
# Cost scale: 0 = NOT SET, 1 = Low ... 10 = High
API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL", "gpt-5.2")
API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL", "gpt-5-mini")
API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL", "gpt-5-mini")

API_PRIMARY_COST    = int(os.getenv("SARAH_OPENAI_PRIMARY_COST", "3"))
API_SECONDARY_COST  = int(os.getenv("SARAH_OPENAI_SECONDARY_COST", "2"))
API_DEFAULT_COST    = int(os.getenv("SARAH_OPENAI_DEFAULT_COST", "1"))
#Allows every mode possible if available as of v7.7.2-09/29/2025
#"gpt-5,gpt-4.1,gpt-4.1-mini,o4-mini,gpt-4o,gpt-4-turbo,chatgpt-4o-latest,gpt-3.5-turbo"
API_ALLOWED_MODELS  = [m.strip() for m in os.getenv("SARAH_OPENAI_ALLOWED_MODELS","gpt-4.1,gpt-4.1-mini,gpt-4.1-mini-2025-04-14,gpt-4.1-2025-04-14,o4-mini,o4-mini-2025-04-16,o4-mini-deep-research,o4-mini-deep-research-2025-06-26,o3,o3-2025-04-16,o3-mini,o3-mini-2025-01-31,o1,o1-2024-12-17,o1-mini,o1-mini-2024-09-12,o1-pro,o1-pro-2025-03-19,gpt-4o,gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4o-realtime-preview,gpt-4o-realtime-preview-2024-10-01,gpt-4o-realtime-preview-2025-06-03,gpt-4o-mini,gpt-4o-mini-2024-07-18,gpt-4o-mini-search-preview,gpt-4o-mini-search-preview-2025-03-11,gpt-4o-search-preview,gpt-4o-search-preview-2025-03-11,chatgpt-4o-latest,gpt-4,gpt-4-turbo,gpt-4-turbo-preview,gpt-4-0125-preview,gpt-4-1106-preview,gpt-4-0613,gpt-3.5-turbo,gpt-3.5-turbo-1106,gpt-3.5-turbo-0125,gpt-3.5-turbo-16k,gpt-3.5-turbo-instruct-0914,text-embedding-3-small,text-embedding-3-large,text-embedding-ada-002,gpt-4o-mini-transcribe,gpt-4o-transcribe,whisper-1,gpt-4o-audio-preview,gpt-4o-audio-preview-2024-10-01,gpt-4o-audio-preview-2024-12-17,gpt-4o-mini-audio-preview,gpt-4o-mini-audio-preview-2024-12-17,gpt-4o-mini-tts,tts-1,tts-1-1106,tts-1-hd,tts-1-hd-1106,gpt-image-1,dall-e-3,dall-e-2,omni-moderation-latest,omni-moderation-2024-09-26,babbage-002,davinci-002,codex-mini-latest,gpt-5,gpt-5-mini,gpt-5-mini-2025-08-07,gpt-5-nano,gpt-5-nano-2025-08-07,gpt-5-chat-latest,gpt-realtime,gpt-realtime-2025-08-28,gpt-audio,gpt-audio-2025-08-28,gpt-4.1-nano"
).split(",") if m.strip()]
API_BLOCKLIST_MODELS = [m.strip() for m in os.getenv("SARAH_OPENAI_BLOCKLIST_MODELS","").split(",") if m.strip()]

# GUI media flags
GUI_ALLOW_IMAGES = True
GUI_MAX_IMAGE_WIDTH  = 512
GUI_MAX_IMAGE_HEIGHT = 512

# Schedules for updater + FTP backup

FTP_BACKUP_SCHEDULE = os.getenv("SARAH_FTP_BACKUP_SCHEDULE", "weekly")

def schedule_to_days(kind: str) -> int:
    """Map a friendly schedule name to days between runs.

    Returns:
      0  -> never
     -1  -> always
      >0 -> days
    """
    k = (kind or "").strip().lower()
    return {
        "never": 0,
        "always": -1,
        "hourly": 0,      # handled by minute-based interval gates elsewhere
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365,
        # Legacy aliases
        "90days": 90,
        "180days": 180,
    }.get(k, 7)
# ============================================================================

# ===== Reasoning Order & Learning (v7.5) =====
REASONING_SEARCH_ORDER = ["local", "web", "api"]
ENABLE_SELF_GRADING = True
SELF_GRADE_THRESHOLD = 0.62
ENABLE_AUTODOC_WRITEBACK = True

# ===== Consolidated Model Defaults (v7.5) =====
API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL", "gpt-5.2")
API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL", "gpt-5-mini")
API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL", "gpt-5-mini")
EMBEDDING_MODELS = {
    "primary":   "all-MiniLM-L6-v2",
    "secondary": "paraphrase-MiniLM-L3-v2",
}
OBJECT_MODELS = {
    "cascade_fallback": True,
    "ultra_detector":   True,
    "gpu_accel":        True,
}

# ======================= SarahMemory Settings GUI (Introspective) =======================
# Professional Settings GUI for SarahMemoryGlobals.py
# Allows users to view, modify, and save configuration settings via a Tkinter interface
# Run directly: python SarahMemoryGlobals.py
# ========================================================================================

def _sm_center_window(win, width=1024, height=720):
    """
    Center the window on the screen with specified dimensions.
    Sets minimum size to ensure usability.
    """
    try:
        win.update_idletasks()
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")
        win.minsize(900, 600)
    except Exception:
        pass


def _sm_has_display():
    """
    Check if a display is available for GUI rendering.
    Returns False for headless environments (PythonAnywhere, Linux without DISPLAY, etc.)
    """
    import os
    import sys
    
    # Force headless mode via environment variable
    if os.environ.get("SARAH_FORCE_HEADLESS", "").lower() in ("1", "true", "yes"):
        return False
    
    # PythonAnywhere detection
    if os.environ.get("PYTHONANYWHERE_DOMAIN") or os.environ.get("PA_HOME"):
        return False
    
    # Linux without DISPLAY
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    
    # Mobile platforms
    if sys.platform.startswith("ios") or "android" in sys.platform.lower():
        return bool(os.environ.get("DISPLAY"))
    
    return True


def _sm_is_config_key(name, val):
    """
    Determine if a global variable should be exposed in the Settings GUI.
    Must be uppercase, not private, not callable, and a supported type.
    """
    if not name.isupper():
        return False
    if name.startswith("_"):
        return False
    if callable(val):
        return False
    # Exclude module references and large internal registries that now have dedicated UI surfaces
    if name in (
        "SarahMemoryGlobals",
        "MODEL_CATALOG", "MODEL_REPO_MAP", "MODEL_META", "MODEL_CONFIG", "OBJECT_MODEL_CONFIG",
        "MODEL_CATEGORY_PRIMARY_ATTRS", "MODEL_CATEGORY_FLAG_REPO_MAP", "MODEL_MANAGER_CATEGORY_LABELS",
    ):
        return False
    return isinstance(val, (bool, int, float, str, list, dict))


def _sm_group_for_key(key_name):
    """
    Categorize configuration keys into logical groups for tabbed display.
    Returns a category string based on keyword matching.
    """
    k = key_name.lower()
    
    # Core System Settings
    if any(s in k for s in ["debug", "safe_mode", "local_only", "run_mode", "device_mode", "device_profile"]):
        return "Core"
    
    # API Configuration
    if any(s in k for s in ["api_", "openai", "claude", "mistral", "gemini", "huggingface", "_api", "api_key", "api_token"]):
        return "APIs"
    
    # Model Configuration
    if any(s in k for s in ["model", "enable_model", "embedding", "llm", "transformer", "multi_model", "auto_model"]):
        return "Models"
    
    # Vision & Object Detection
    if any(s in k for s in ["vision", "yolo", "ssd", "detr", "dino", "facial", "object", "camera", "opencv", "detection"]):
        return "Vision"
    
    # Research & Learning
    if any(s in k for s in ["research", "learning", "wikipedia", "duckduckgo", "stackoverflow", "route_mode", "local_data", "web_research"]):
        return "Research"
    
    # Voice & Audio
    if any(s in k for s in ["voice", "tts", "stt", "speech", "audio", "mic", "avatar_is_speaking"]):
        return "Voice"
    
    # Network & Sync
    if any(s in k for s in ["network", "sync", "mesh", "sarahnet", "remote", "ftp", "web_", "hub", "peer"]):
        return "Network"
    
    # AI Agent Settings
    if any(s in k for s in ["ai_agent", "agent_", "autonomous", "consent", "halt", "resume"]):
        return "Agent"
    
    # GUI & Display
    if any(s in k for s in ["gui_", "avatar", "theme", "color", "display", "refresh", "browser"]):
        return "GUI"
    
    # Paths & Directories
    if any(s in k for s in ["_dir", "_path", "path_", "dir_", "folder", "file_"]):
        return "Paths"
    
    # Performance & Optimization
    if any(s in k for s in ["cache", "timeout", "buffer", "interval", "limit", "threshold", "perf_", "optimize"]):
        return "Performance"
    
    # Updater & Backup
    if any(s in k for s in ["update", "backup", "schedule", "stamp"]):
        return "Updates"
    
    # Default category
    return "General"


def _sm_get_default_settings():
    """
    Return a dictionary of default/recommended settings.
    Used by the 'Restore Defaults' button.
    """
    return {
    # Core
    "DEBUG_MODE": True,
    "SAFE_MODE": False,
    "LOCAL_ONLY_MODE": False,

    # Research
    "LOCAL_DATA_ENABLED": True,
    "WEB_RESEARCH_ENABLED": True,
    "API_RESEARCH_ENABLED": True,
    "ROUTE_MODE": "Any",
    "WIKIPEDIA_RESEARCH_ENABLED": True,
    "DUCKDUCKGO_RESEARCH_ENABLED": False,

    # APIs
    "OPEN_AI_API": True,
    "CLAUDE_API": False,
    "ANTHROPIC_API": False,
    "MISTRAL_API": False,
    "GEMINI_API": False,
    "HUGGINGFACE_API": False,
    "DEEPSEEK_API": False,
    "GROQ_API": False,
    "COHERE_API": False,
    "LOCAL_LLM_API": True,
    "LOCAL_API": True,
    "MESH_API": True,
    "API_TIMEOUT": 20,

    # Models (aligned to Top Menu DEFAULTS)
    "AUTO_MODEL_SELECTOR": True,
    "MULTI_MODEL": True,

    # Model routing selections
    "REASONING_MODEL_REPO": "Qwen/Qwen3-0.6B",
    "CODER_MODEL_REPO": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "EMBEDDING_EN_REPO": "sentence-transformers/all-MiniLM-L6-v2",
    "VISION_PRIMARY_REPO": "nielsr/yolov12n",
    "IMAGEGEN_MODEL_REPO": "black-forest-labs/FLUX.1-schnell",
    "TTS_MODEL_REPO": "FunAudioLLM/CosyVoice2-0.5B",
    "CUSTOM_MODEL_REPOS": {
        "reasoning": "",
        "coder": "",
        "embeddings": "",
        "vision": "",
        "image_generation": "",
        "tts": "",
    },

    # --- Embeddings DEFAULTS = B, D, E, F, G ---
    "ENABLE_MODEL_B": True,   # all-MiniLM-L6-v2
    "ENABLE_MODEL_D": True,   # paraphrase-MiniLM-L3-v2
    "ENABLE_MODEL_E": True,   # distiluse-base-multilingual-cased-v2
    "ENABLE_MODEL_F": True,   # allenai/specter
    "ENABLE_MODEL_G": True,   # intfloat/e5-base
    "ENABLE_MODEL_C": False,  # multi-qa-MiniLM-L6-cos-v1
    "ENABLE_MODEL_R": False,  # BAAI/bge-base-en-v1.5
    "ENABLE_MODEL_U": False,  # BAAI/bge-m3

    # --- Reasoning / Chat DEFAULTS = N ---
    "ENABLE_MODEL_N": True,   # Qwen/Qwen3-0.6B
    "ENABLE_MODEL_A": False,  # microsoft/phi-1_5
    "ENABLE_MODEL_H": False,  # microsoft/phi-2
    "ENABLE_MODEL_S": False,  # microsoft/Phi-4-mini-instruct
    "ENABLE_MODEL_Q": False,  # Qwen/Qwen2.5-7B-Instruct
    "ENABLE_MODEL_I": False,  # falcon-rw-1b
    "ENABLE_MODEL_M": False,  # TinyLlama-1.1B-Chat
    "ENABLE_MODEL_J": False,  # openchat-3.5-0106
    "ENABLE_MODEL_K": False,  # Nous-Capybara-7B
    "ENABLE_MODEL_L": False,  # Mistral-7B-Instruct-v0.2

    # --- Coder DEFAULTS = O, P ---
    "ENABLE_MODEL_O": True,   # Qwen2.5-Coder-1.5B-Instruct
    "ENABLE_MODEL_P": True,   # Qwen2.5-Coder-3B-Instruct
    "ENABLE_MODEL_T": False,  # Qwen2.5-Coder-7B-Instruct

    # --- Vision / Object Detection DEFAULTS = V + SSD (AD) ---
    "ENABLE_MODEL_V": True,   # nielsr/yolov12n
    "ENABLE_MODEL_W": False,  # ultralytics/yolov8
    "ENABLE_MODEL_X": False,  # qualcomm/RF-DETR
    "ENABLE_MODEL_Y": False,  # ultralytics/yolov8x
    "ENABLE_MODEL_AD": True,  # SSD (pytorch-ssd)

    # --- Image Generation DEFAULTS = Z ---
    "ENABLE_MODEL_Z": True,    # FLUX.1-schnell
    "ENABLE_MODEL_AA": False,  # Freepik/flux.1-lite-8B
    "ENABLE_MODEL_AB": False,  # FLUX.1-dev

    # --- Voice / TTS DEFAULTS = AC ---
    "ENABLE_MODEL_AC": True,   # CosyVoice2-0.5B

    # Vision (legacy/object-detection block defaults aligned to Top Menu)
    "OBJECT_DETECTION_ENABLED": True,
    "ENABLE_YOLOV12N": True,
    "ENABLE_YOLOV8": False,
    "ENABLE_SSD": True,
    "ENABLE_RF_DETR": False,

    # Legacy detectors (centralized OFF by default)
    "ENABLE_YOLOV5": False,
    "ENABLE_YOLOV7": False,
    "ENABLE_YOLO_NAS": False,
    "ENABLE_YOLOX": False,
    "ENABLE_PP_YOLOV2": False,
    "ENABLE_EFFICIENTDET": False,
    "ENABLE_DETR": False,
    "ENABLE_DINO": False,
    "ENABLE_CENTERNET": False,
    "ENABLE_FASTER_RCNN": False,
    "ENABLE_RETINANET": False,

    # Vision learning flags
    "FACIAL_RECOGNITION_LEARNING": True,
    "VISUAL_BACKGROUND_LEARNING": True,

    # Voice
    "VOICE_FEEDBACK_ENABLED": True,
    "TTS_ASYNC": True,
    "TTS_BLOCKING": False,
    "AVATAR_IS_SPEAKING": True,

    # Agent
    "AI_AGENT_ENABLED": True,
    "AI_GAME_MODE_ENABLED": True,
    "AI_GAME_FULL_AUTO": True,

    # Context & Learning
    "ENABLE_CONTEXT_BUFFER": True,
    "CONTEXT_BUFFER_SIZE": 10,
    "ENABLE_CONTEXT_ENRICHMENT": True,
    "IMPORT_OTHER_DATA_LEARN": True,
    "LEARNING_PHASE_ACTIVE": True,

    # Performance
    "ASYNC_PROCESSING_ENABLED": True,
    "LOOP_DETECTION_THRESHOLD": 3,
    "REPLY_STATUS": True,
    "COMPARE_VOTE": False,

    # Network
    "SARAHNET_ENABLED": True,
    "REMOTE_SYNC_ENABLED": True,

    # GUI
    "ENABLE_MINI_BROWSER": True,
    "GUI_ALLOW_IMAGES": True,
    "GUI_MAX_IMAGE_WIDTH": 512,
    "GUI_MAX_IMAGE_HEIGHT": 512,
    }


def _sm_save_settings_to_file(settings_dict):
    """
    Save the current settings to the SETTINGS_FILE (settings.json).
    Creates the directory if it doesn't exist.
    Returns (success: bool, message: str)
    """
    import os
    import json
    
    try:
        # Use SETTINGS_FILE from globals if available
        settings_path = globals().get("SETTINGS_FILE", None)
        if not settings_path:
            settings_dir = globals().get("SETTINGS_DIR", os.path.join(os.getcwd(), "data", "settings"))
            settings_path = os.path.join(settings_dir, "settings.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        # Filter to only include serializable, modified settings
        saveable = {}
        for key, value in settings_dict.items():
            if isinstance(value, (bool, int, float, str, list, dict)):
                saveable[key] = value
        
        # Write to file with pretty formatting
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, indent=4, sort_keys=True)
        
        return True, f"Settings saved to:\n{settings_path}"
    
    except Exception as e:
        return False, f"Failed to save settings:\n{str(e)}"


def _sm_create_scrollable_frame(parent):
    """
    Create a scrollable frame widget for tabs with many settings.
    Returns (canvas, scrollable_inner_frame)
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Create canvas and scrollbar
    canvas = tk.Canvas(parent, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    
    # Create inner frame for content
    inner_frame = ttk.Frame(canvas)
    
    # Configure scrolling
    inner_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # Create window in canvas
    canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
    
    # Configure canvas to expand inner frame width
    def configure_inner_width(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", configure_inner_width)
    
    # Configure scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def bind_mousewheel(event):
        canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    def unbind_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")
    
    # Bind mouse wheel only when hovering over the canvas
    canvas.bind("<Enter>", bind_mousewheel)
    canvas.bind("<Leave>", unbind_mousewheel)
    
    # Pack widgets
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    return canvas, inner_frame


def _sm_get_tab_ui_meta():
    """Friendly tab labels/descriptions for the Settings GUI."""
    return {
        "Core": {
            "title": "Core Runtime Controls",
            "description": "Main platform behavior, runtime mode, safety posture, and baseline system identity.",
        },
        "APIs": {
            "title": "API Providers",
            "description": "Turn external providers on or off, choose routing behavior, and control timeouts and API cost posture.",
        },
        "Models": {
            "title": "Model Routing Settings",
            "description": "Advanced model flags and model-governance settings. Use the Model Manager tab for simple model selection.",
        },
        "Research": {
            "title": "Research & Learning",
            "description": "Control local, web, and API research behavior, learning passes, and data-ingestion pathways.",
        },
        "Vision": {
            "title": "Vision & Detection",
            "description": "Visual learning, object detection, camera-side helper features, and related detection options.",
        },
        "Voice": {
            "title": "Voice & Audio",
            "description": "Speech input, TTS output, microphone timing, speaking behavior, and audio feedback controls.",
        },
        "Agent": {
            "title": "Agent & Autonomy",
            "description": "Advanced agent behavior, autonomy gating, and AI-side execution controls.",
        },
        "Network": {
            "title": "Network & Sync",
            "description": "SarahNet, remote sync, ports, peers, FTP, hub settings, and online/offline related connectivity options.",
        },
        "GUI": {
            "title": "GUI & Display",
            "description": "Desktop and WebUI behavior, visual panels, avatar display, browser surfaces, and interface presentation options.",
        },
        "Performance": {
            "title": "Performance & Limits",
            "description": "Thresholds, buffers, caches, timers, queue sizes, and runtime performance envelopes.",
        },
        "Updates": {
            "title": "Updates & Backup Schedules",
            "description": "Updater cadence, backup schedules, self-aware/evolution timing, and maintenance windows.",
        },
        "Paths": {
            "title": "Paths & Directories",
            "description": "Resolved filesystem locations used by SarahMemory. Review carefully before editing.",
        },
        "General": {
            "title": "General Settings",
            "description": "Additional settings that do not fall cleanly into another major runtime category.",
        },
    }


def _sm_pretty_setting_name(key_name: str) -> str:
    """Convert a CONSTANT_LIKE_NAME into a friendly UI label."""
    try:
        raw = str(key_name or "").strip().replace("__", "_")
        if not raw:
            return "Setting"
        parts = [p for p in raw.split("_") if p]
        pretty_parts = []
        acronyms = {"AI", "API", "GUI", "TTS", "STT", "CPU", "GPU", "RAM", "FTP", "HTTP", "HTTPS", "UI", "DB", "URL", "TTL", "CSV", "JSON", "SQL", "SSD", "YOLO", "RF", "DETR", "LLM", "CAD", "PLC", "CNC", "LAN", "WAN", "UDP", "TCP", "VRAM"}
        for part in parts:
            upper = part.upper()
            if upper in acronyms:
                pretty_parts.append(upper)
            elif part.lower() in ("mb", "gb", "ms", "sec"):
                pretty_parts.append(part.upper())
            else:
                pretty_parts.append(part.capitalize())
        return " ".join(pretty_parts)
    except Exception:
        return str(key_name or "Setting")


def _sm_get_setting_help_text(key_name: str, category: str = "", value=None) -> str:
    """Return a short operator-facing help description for a setting."""
    key = str(key_name or "").strip().upper()
    desc_map = {
        "SAFE_MODE": "Master safety gate. When enabled, heavy or risky runtime behavior is constrained.",
        "LOCAL_ONLY_MODE": "Keep reasoning and operations local-first and avoid external network/provider use where possible.",
        "RUN_MODE": "Declares whether the platform is running locally, in cloud mode, or in a test environment.",
        "DEVICE_PROFILE": "Sets the coarse device performance profile used by several runtime selectors.",
        "UI_MODE": "Chooses which user interface surface is active: classic, legacy web, or custom web UI.",
        "ROUTE_MODE": "Select how research and answer routing should prefer Local, Web, API, or Any.",
        "API_TIMEOUT": "Maximum wait time for provider-backed requests before the runtime treats the call as stalled.",
        "AI_AGENT_ENABLED": "Main switch for the advanced agent behavior layer.",
        "WEB_RESEARCH_ENABLED": "Allow web research sources to be used during answer or discovery workflows.",
        "API_RESEARCH_ENABLED": "Allow external AI/API providers to be consulted during research workflows.",
        "LOCAL_DATA_ENABLED": "Allow the local datasets and memory stores to participate in answers and recall.",
        "VOICE_FEEDBACK_ENABLED": "Allow SarahMemory to speak responses back to the user.",
        "AVATAR_IS_SPEAKING": "Helps prevent mic echo and self-hearing while TTS is actively speaking.",
        "REMOTE_SYNC_ENABLED": "Allow remote node or hub synchronization services to run when available.",
        "SARAHNET_ENABLED": "Enable the SarahNet mesh communication layer.",
        "ENABLE_MINI_BROWSER": "Allow the integrated lightweight browser/panel behavior where supported.",
        "DEBUG_MODE": "Enable verbose diagnostics and deeper developer-side logging.",
        "UPDATER_SCHEDULE": "Friendly schedule used by the core updater policy.",
        "BACKUP_SCHEDULE": "Friendly schedule used for backup workflows.",
        "EVOLUTION_SCHEDULE": "Friendly schedule for experimental evolution/self-improvement cycles.",
        "SELFAWARE_SCHEDULE": "Friendly schedule for self-aware monitoring and internal periodic checks.",
    }
    if key in desc_map:
        return desc_map[key]

    lowered = key.lower()
    if key.startswith("ENABLE_"):
        return f"Turn {_sm_pretty_setting_name(key_name).replace('Enable ', '')} on or off."
    if key.endswith("_DIR"):
        return "Filesystem directory path used by this part of the platform. Change carefully."
    if key.endswith("_PATH") or key.endswith("_FILE"):
        return "Filesystem path used by this part of the platform. Change carefully."
    if key.endswith("_SCHEDULE"):
        return "Human-readable schedule selector used by the corresponding runtime policy."
    if any(word in lowered for word in ("timeout", "interval", "minutes", "seconds", "delay", "ttl")):
        return "Timing or interval control that affects how often this runtime behavior occurs."
    if any(word in lowered for word in ("threshold", "limit", "max", "budget", "size", "count")):
        return "Limit or threshold used to bound runtime workload, resource usage, or decision gates."
    if any(word in lowered for word in ("host", "port", "peer", "domain", "api", "sync", "network", "mesh")):
        return "Network or provider connectivity setting used by the runtime when online features are enabled."
    if any(word in lowered for word in ("voice", "tts", "mic", "audio", "speech")):
        return "Voice/audio behavior control used by speech input or spoken output."
    if any(word in lowered for word in ("avatar", "theme", "browser", "window", "gui", "display")):
        return "Interface or display-facing setting that changes how the UI behaves or renders."
    if isinstance(value, bool):
        return "Simple on/off runtime control."
    if isinstance(value, (int, float)):
        return "Numeric runtime parameter."
    if isinstance(value, str):
        return "Text-based runtime setting."
    if isinstance(value, (list, dict)):
        return "Structured configuration data. Edit with care."
    return f"{category or 'General'} runtime setting."


def _sm_is_essential_setting(key_name: str, category: str = "") -> bool:
    """Return True for settings most users are likely to change first."""
    key = str(key_name or "").strip().upper()
    essentials = {
        "DEBUG_MODE", "SAFE_MODE", "LOCAL_ONLY_MODE", "RUN_MODE", "DEVICE_PROFILE", "UI_MODE",
        "ROUTE_MODE", "WEB_RESEARCH_ENABLED", "API_RESEARCH_ENABLED", "LOCAL_DATA_ENABLED",
        "OPEN_AI_API", "GEMINI_API", "LOCAL_LLM_API", "LOCAL_API", "MESH_API", "API_TIMEOUT",
        "VOICE_FEEDBACK_ENABLED", "AVATAR_IS_SPEAKING", "TTS_ASYNC", "TTS_BLOCKING",
        "AI_AGENT_ENABLED", "SARAHNET_ENABLED", "REMOTE_SYNC_ENABLED", "ENABLE_MINI_BROWSER",
        "UPDATER_SCHEDULE", "BACKUP_SCHEDULE", "EVOLUTION_SCHEDULE", "SELFAWARE_SCHEDULE",
    }
    if key in essentials:
        return True
    if category == "Paths":
        return False
    return False


def launch_settings_gui():
    """
    Launch the SarahMemory Settings + Model Manager GUI.

    Added capabilities:
    - Dedicated Model Manager tab with category dropdowns
    - Custom Hugging Face repo input per category
    - Direct install / download bridge into SarahMemoryLLM.py
    - Live progress bar + activity log for model downloads
    - Generic categorized settings tabs remain available for all other globals
    """

    if not _sm_has_display():
        print("[Settings GUI] Headless environment detected - skipping Tkinter window.")
        print("[Settings GUI] To modify settings, edit the settings.json file directly or set environment variables.")
        return

    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import queue
        import threading
    except ImportError as e:
        print(f"[Settings GUI] Tkinter unavailable: {e}")
        return

    try:
        root = tk.Tk()
    except Exception as e:
        print(f"[Settings GUI] Failed to initialize Tk root window: {e}")
        return

    root.title("SarahMemory - Global Settings + Model Manager")
    _sm_center_window(root, 1360, 860)

    try:
        icon_path = os.path.join(globals().get("BASE_DIR", os.getcwd()), "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass

    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "vista" in available_themes:
            style.theme_use("vista")
    except Exception:
        pass

    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", padx=12, pady=(12, 6))

    title_label = ttk.Label(header_frame, text="SarahMemory Global Settings", font=("Segoe UI", 18, "bold"))
    title_label.pack(side="left")

    subtitle_label = ttk.Label(
        header_frame,
        text="User-friendly runtime configuration, model routing, and guided local model downloads",
        font=("Segoe UI", 10),
    )
    subtitle_label.pack(side="left", padx=(14, 0))

    version_label = ttk.Label(header_frame, text=f"Version: {globals().get('PROJECT_VERSION', '8.0.0')}", font=("Segoe UI", 10))
    version_label.pack(side="right")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=4)

    status_bar_frame = ttk.Frame(root)
    status_bar_frame.pack(fill="x", padx=10, pady=(0, 10))
    status_var = tk.StringVar(value="Ready")
    ttk.Label(status_bar_frame, textvariable=status_var, font=("Segoe UI", 9)).pack(side="left")

    # ------------------------------------------------------------------
    # Model Manager Tab
    # ------------------------------------------------------------------
    model_tab = ttk.Frame(notebook)
    notebook.add(model_tab, text=" Model Manager ")

    model_container = ttk.Frame(model_tab)
    model_container.pack(fill="both", expand=True, padx=10, pady=10)
    model_container.grid_columnconfigure(0, weight=1)

    model_summary_var = tk.StringVar(value="Detecting installed models...")
    model_task_var = tk.StringVar(value="No active model download")

    summary_frame = ttk.LabelFrame(model_container, text="Model Summary")
    summary_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 8))
    summary_frame.grid_columnconfigure(0, weight=1)

    ttk.Label(summary_frame, textvariable=model_summary_var, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", padx=10, pady=8)

    summary_button_frame = ttk.Frame(summary_frame)
    summary_button_frame.grid(row=0, column=1, sticky="e", padx=10, pady=8)

    model_rows_frame = ttk.LabelFrame(model_container, text="Category Model Routing")
    model_rows_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 8))
    model_container.grid_rowconfigure(1, weight=1)
    model_rows_frame.grid_columnconfigure(1, weight=1)
    model_rows_frame.grid_columnconfigure(2, weight=1)
    model_rows_frame.grid_columnconfigure(3, weight=0)
    model_rows_frame.grid_columnconfigure(4, weight=0)
    model_rows_frame.grid_columnconfigure(5, weight=0)

    headers = ["Category", "Catalog Selection", "Custom Repo", "Installed", "Metadata", "Actions"]
    for idx, text in enumerate(headers):
        ttk.Label(model_rows_frame, text=text, font=("Segoe UI", 10, "bold")).grid(row=0, column=idx, sticky="w", padx=8, pady=(8, 6))

    model_rows = {}
    model_download_queue = queue.Queue()
    model_download_state = {"running": False, "last_message": "", "mode": "idle"}

    progress_frame = ttk.LabelFrame(model_container, text="Download Activity")
    progress_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, 8))
    progress_frame.grid_columnconfigure(0, weight=1)

    ttk.Label(progress_frame, textvariable=model_task_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))

    model_progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
    model_progress.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

    log_frame = ttk.Frame(progress_frame)
    log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
    log_frame.grid_columnconfigure(0, weight=1)
    log_frame.grid_rowconfigure(0, weight=1)

    model_log = tk.Text(log_frame, height=12, wrap="word", font=("Consolas", 9))
    model_log.grid(row=0, column=0, sticky="nsew")
    model_log.configure(state="disabled")
    model_log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=model_log.yview)
    model_log_scroll.grid(row=0, column=1, sticky="ns")
    model_log.configure(yscrollcommand=model_log_scroll.set)

    def append_model_log(message):
        text = str(message or "").strip()
        if not text:
            return
        model_log.configure(state="normal")
        model_log.insert("end", text + "\n")
        model_log.see("end")
        model_log.configure(state="disabled")

    def refresh_model_summary():
        installed = 0
        known = 0
        for category_key in MODEL_CATEGORY_PRIMARY_ATTRS.keys():
            seen = set()
            for repo in _sm_collect_model_repos_for_category(category_key):
                if repo in seen:
                    continue
                seen.add(repo)
                known += 1
                snapshot = _sm_get_model_manager_snapshot(category_key, repo)
                if snapshot.get("installed"):
                    installed += 1
        try:
            hs = hardware_score()
            score = hs.get("score")
            tier_rating = hs.get("tier_rating")
            used = 0.0
            try:
                import SarahMemoryLLM as _SMLLM  # type: ignore
                used = float(_SMLLM.get_models_storage_usage_gb(globals().get("MODELS_DIR")))
            except Exception:
                pass
            model_summary_var.set(
                f"Hardware score {score} ({tier_rating})  |  Known installed model entries: {installed}/{max(known, 1)}  |  Model storage used: {used:.2f} GB"
            )
        except Exception:
            model_summary_var.set(f"Known installed model entries: {installed}/{max(known, 1)}")

    def get_selected_repo_for_category(category_key):
        row = model_rows.get(category_key, {})
        if not row:
            return ""
        choice = str(row["choice_var"].get() or "").strip()
        custom_repo = str(row["custom_var"].get() or "").strip()
        if choice == MODEL_MANAGER_CUSTOM_OPTION:
            return custom_repo
        if custom_repo and choice == _sm_get_category_custom_repo(category_key):
            return custom_repo
        return choice or custom_repo or _sm_get_category_current_repo(category_key)

    def refresh_model_row(category_key):
        row = model_rows.get(category_key, {})
        if not row:
            return
        selected_repo = get_selected_repo_for_category(category_key)
        snapshot = _sm_get_model_manager_snapshot(category_key, selected_repo) if selected_repo else {
            "installed": False,
            "local_size_gb": 0.0,
            "estimated_size_gb": None,
            "meta_summary": "n/a",
        }

        if selected_repo:
            if snapshot.get("installed"):
                row["installed_var"].set(f"Installed ({snapshot.get('local_size_gb', 0.0):.2f} GB)")
            else:
                est = snapshot.get("estimated_size_gb")
                if est is None:
                    row["installed_var"].set("Not installed")
                else:
                    row["installed_var"].set(f"Not installed (est. {float(est):.2f} GB)")
            row["meta_var"].set(snapshot.get("meta_summary") or "n/a")
            if str(row["choice_var"].get() or "").strip() == MODEL_MANAGER_CUSTOM_OPTION:
                try:
                    row["custom_entry"].configure(state="normal")
                except Exception:
                    pass
            else:
                try:
                    row["custom_entry"].configure(state="normal")
                except Exception:
                    pass
        else:
            row["installed_var"].set("No repo selected")
            row["meta_var"].set("n/a")

    def refresh_all_model_rows():
        for category_key in model_rows.keys():
            refresh_model_row(category_key)
        refresh_model_summary()

    def apply_model_selection(category_key, persist_status=True):
        selected_repo = get_selected_repo_for_category(category_key)
        if not selected_repo:
            messagebox.showwarning("Model Selection", f"No model repo selected for {MODEL_MANAGER_CATEGORY_LABELS.get(category_key, category_key)}.")
            return ""
        selected_repo = _sm_set_category_model_repo(category_key, selected_repo)
        refresh_model_row(category_key)
        if persist_status:
            status_var.set(f"Updated {MODEL_MANAGER_CATEGORY_LABELS.get(category_key, category_key)} → {selected_repo}")
        return selected_repo

    def on_model_choice_changed(category_key, *_args):
        row = model_rows.get(category_key, {})
        if not row:
            return
        choice = str(row["choice_var"].get() or "").strip()
        if choice and choice != MODEL_MANAGER_CUSTOM_OPTION and choice == _sm_get_category_custom_repo(category_key):
            row["custom_var"].set(choice)
        refresh_model_row(category_key)

    def start_model_download(single_category=None):
        if model_download_state.get("running"):
            messagebox.showinfo("Model Download Busy", "A model download is already running. Please wait for it to finish.")
            return

        selections = []
        categories = [single_category] if single_category else list(model_rows.keys())
        for category_key in categories:
            selected_repo = apply_model_selection(category_key, persist_status=False)
            if selected_repo:
                selections.append((category_key, selected_repo))

        if not selections:
            messagebox.showwarning("Model Download", "No valid model selections were found.")
            return

        model_download_state["running"] = True
        model_download_state["last_message"] = ""
        model_download_state["mode"] = "determinate"
        model_progress.configure(mode="determinate")
        model_progress["value"] = 0
        model_task_var.set("Starting model download...")

        def queue_progress_event(payload):
            try:
                model_download_queue.put(dict(payload or {}))
            except Exception:
                pass

        def worker():
            try:
                import SarahMemoryLLM as _SMLLM  # type: ignore
            except Exception as exc:
                queue_progress_event({"event": "error", "message": f"SarahMemoryLLM import failed: {exc}"})
                return

            for category_key, selected_repo in selections:
                queue_progress_event({
                    "event": "stage",
                    "message": f"Starting {MODEL_MANAGER_CATEGORY_LABELS.get(category_key, category_key)} download: {selected_repo}",
                    "progress": 0.0,
                })
                ok = _SMLLM.download_hf_model(
                    selected_repo,
                    _SMLLM.repo_to_local_dir(selected_repo),
                    use_sentence_transformer=bool(category_key == "embeddings"),
                    progress_callback=queue_progress_event,
                    category=category_key,
                )
                if not ok:
                    queue_progress_event({"event": "error", "message": f"Download failed for {selected_repo}"})
                    return
            queue_progress_event({"event": "worker_done", "message": "Model download workflow completed."})

        threading.Thread(target=worker, name="SarahMemoryGlobalsModelDownload", daemon=True).start()

    def apply_recommended_stack():
        for category_key in model_rows.keys():
            try:
                tier = recommend_model_tier(category=category_key)
                if tier == "beast":
                    fallback = ["high", "mid", "low"]
                elif tier == "high":
                    fallback = ["mid", "low"]
                elif tier == "mid":
                    fallback = ["low"]
                else:
                    fallback = []
                repo = pick_catalog_model(category_key, tier, fallback_tiers=fallback)
            except Exception:
                repo = None
            if repo:
                row = model_rows.get(category_key, {})
                if row:
                    resolved = resolve_model_repo(repo)
                    current_values = list(row["combobox"].cget("values"))
                    if resolved not in current_values:
                        current_values.insert(0, resolved)
                        row["combobox"]["values"] = current_values
                    row["choice_var"].set(resolved)
                    row["custom_var"].set("")
                    refresh_model_row(category_key)
        status_var.set("Applied recommended stack into the model dropdowns (not saved yet)")

    for row_index, category_key in enumerate(MODEL_CATEGORY_PRIMARY_ATTRS.keys(), start=1):
        label = MODEL_MANAGER_CATEGORY_LABELS.get(category_key, category_key.title())
        current_repo = _sm_get_category_current_repo(category_key)
        custom_repo = _sm_get_category_custom_repo(category_key)
        values = _sm_collect_model_repos_for_category(category_key)
        if MODEL_MANAGER_CUSTOM_OPTION not in values:
            values.append(MODEL_MANAGER_CUSTOM_OPTION)
        if current_repo and current_repo not in values:
            values.insert(0, current_repo)

        choice_var = tk.StringVar(value=current_repo or MODEL_MANAGER_CUSTOM_OPTION)
        custom_var = tk.StringVar(value=custom_repo)
        installed_var = tk.StringVar(value="Checking...")
        meta_var = tk.StringVar(value="n/a")

        ttk.Label(model_rows_frame, text=label, font=("Segoe UI", 10, "bold")).grid(row=row_index, column=0, sticky="w", padx=8, pady=6)

        combo = ttk.Combobox(model_rows_frame, textvariable=choice_var, values=values, width=40, state="readonly")
        combo.grid(row=row_index, column=1, sticky="ew", padx=8, pady=6)
        combo.bind("<<ComboboxSelected>>", lambda _e, c=category_key: on_model_choice_changed(c))

        custom_entry = ttk.Entry(model_rows_frame, textvariable=custom_var, width=42)
        custom_entry.grid(row=row_index, column=2, sticky="ew", padx=8, pady=6)

        ttk.Label(model_rows_frame, textvariable=installed_var, font=("Segoe UI", 9)).grid(row=row_index, column=3, sticky="w", padx=8, pady=6)
        ttk.Label(model_rows_frame, textvariable=meta_var, font=("Consolas", 8), foreground="gray").grid(row=row_index, column=4, sticky="w", padx=8, pady=6)

        action_frame = ttk.Frame(model_rows_frame)
        action_frame.grid(row=row_index, column=5, sticky="e", padx=8, pady=6)

        model_rows[category_key] = {
            "choice_var": choice_var,
            "custom_var": custom_var,
            "installed_var": installed_var,
            "meta_var": meta_var,
            "combobox": combo,
            "custom_entry": custom_entry,
        }

        ttk.Button(action_frame, text="Apply", width=9, command=lambda c=category_key: apply_model_selection(c)).pack(side="left", padx=(0, 4))
        ttk.Button(action_frame, text="Download", width=10, command=lambda c=category_key: start_model_download(single_category=c)).pack(side="left")

    def pump_download_events():
        try:
            while True:
                event = model_download_queue.get_nowait()
                event_name = str(event.get("event") or "message")
                message = str(event.get("message") or "").strip()
                progress_value = event.get("progress", None)
                indeterminate = bool(event.get("indeterminate", False))

                if indeterminate:
                    if model_download_state.get("mode") != "indeterminate":
                        try:
                            model_progress.stop()
                        except Exception:
                            pass
                        model_progress.configure(mode="indeterminate")
                        model_progress.start(10)
                        model_download_state["mode"] = "indeterminate"
                elif progress_value is not None:
                    try:
                        pct = max(0.0, min(100.0, float(progress_value) * 100.0))
                    except Exception:
                        pct = 0.0
                    if model_download_state.get("mode") != "determinate":
                        try:
                            model_progress.stop()
                        except Exception:
                            pass
                        model_progress.configure(mode="determinate")
                        model_download_state["mode"] = "determinate"
                    model_progress["value"] = pct

                if message and message != model_download_state.get("last_message"):
                    append_model_log(message)
                    model_download_state["last_message"] = message
                    model_task_var.set(message)

                if event_name in ("complete", "worker_done"):
                    try:
                        model_progress.stop()
                    except Exception:
                        pass
                    model_progress.configure(mode="determinate")
                    model_progress["value"] = 100
                    model_download_state["running"] = False
                    model_download_state["mode"] = "idle"
                    refresh_all_model_rows()

                if event_name in ("error", "cancelled"):
                    try:
                        model_progress.stop()
                    except Exception:
                        pass
                    model_progress.configure(mode="determinate")
                    model_download_state["running"] = False
                    model_download_state["mode"] = "idle"
                    refresh_all_model_rows()
        except Exception:
            pass
        finally:
            root.after(200, pump_download_events)

    ttk.Button(summary_button_frame, text="Refresh Status", width=14, command=lambda: refresh_all_model_rows()).pack(side="left", padx=(0, 4))
    ttk.Button(summary_button_frame, text="Apply Recommended", width=16, command=lambda: apply_recommended_stack()).pack(side="left", padx=(0, 4))
    ttk.Button(summary_button_frame, text="Download All Selected", width=18, command=lambda: start_model_download(single_category=None)).pack(side="left")

    footer_frame = ttk.Frame(root)
    footer_frame.pack(fill="x", padx=10, pady=10)

    # ------------------------------------------------------------------
    # Generic categorized tabs (clean, uniform, user-friendly)
    # ------------------------------------------------------------------
    module_globals = globals()
    categorized_items = {}
    for key, value in sorted(module_globals.items()):
        if _sm_is_config_key(key, value):
            category = _sm_group_for_key(key)
            categorized_items.setdefault(category, []).append((key, value))

    tab_order = [
        "Core", "APIs", "Models", "Research", "Vision", "Voice",
        "Agent", "Network", "GUI", "Performance", "Updates", "Paths", "General"
    ]
    sorted_categories = [cat for cat in tab_order if cat in categorized_items]
    for cat in sorted(categorized_items.keys()):
        if cat not in sorted_categories:
            sorted_categories.append(cat)

    all_widgets = {}
    setting_initial_values = {}
    for category_items in categorized_items.values():
        for key, value in category_items:
            setting_initial_values[key] = value

    tab_meta = _sm_get_tab_ui_meta()
    default_settings_map = dict(_sm_get_default_settings() or {})
    for key, value in setting_initial_values.items():
        default_settings_map.setdefault(key, value)

    enum_settings = {
        "ROUTE_MODE": ["Any", "Local", "Web", "API"],
        "RUN_MODE": ["local", "cloud", "test"],
        "DEVICE_PROFILE": ["UltraLite", "Standard", "Performance"],
        "UI_MODE": ["classic", "web", "custom"],
        "API_COST_TIER": ["low", "balanced", "max"],
        "UPDATER_SCHEDULE": ["never", "always", "daily", "weekly", "monthly", "quarterly", "yearly"],
        "FTP_BACKUP_SCHEDULE": ["never", "daily", "weekly", "monthly", "90days", "180days"],
        "UPDATE_POLICY": ["never", "daily", "weekly", "monthly", "quarterly", "yearly"],
        "BACKUP_SCHEDULE": ["never", "always", "hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
        "EVOLUTION_SCHEDULE": ["never", "always", "hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
        "SELFAWARE_SCHEDULE": ["never", "always", "hourly", "daily", "weekly", "monthly", "quarterly", "yearly"],
    }

    category_ui_state = {}

    def _read_widget_value(widget_kind, widget_ref):
        if widget_kind == "bool":
            return bool(widget_ref.get())
        if widget_kind == "int":
            return int(widget_ref.get())
        if widget_kind == "float":
            return float(widget_ref.get())
        if widget_kind == "str":
            return str(widget_ref.get())
        if widget_kind == "json":
            import json as _json
            raw_text = widget_ref.get("1.0", "end").strip()
            return _json.loads(raw_text) if raw_text else None
        if widget_kind == "readonly":
            return str(widget_ref.get())
        return None

    def _apply_widget_value(widget_kind, widget_ref, new_value):
        if widget_kind == "bool":
            widget_ref.set(bool(new_value))
        elif widget_kind == "int":
            widget_ref.set(int(new_value))
        elif widget_kind == "float":
            widget_ref.set(float(new_value))
        elif widget_kind == "str":
            widget_ref.set(str(new_value))
        elif widget_kind == "json":
            import json as _json
            widget_ref.delete("1.0", "end")
            widget_ref.insert("1.0", _json.dumps(new_value, indent=2))
        elif widget_kind == "readonly":
            widget_ref.set(str(new_value))

    def _filter_category_rows(category_name):
        state = category_ui_state.get(category_name, {})
        rows = state.get("rows", [])
        if not rows:
            return
        search_text = str(state.get("search_var").get() or "").strip().lower()
        mode = str(state.get("mode_var").get() or "all").strip().lower()
        visible_count = 0
        for row_info in rows:
            matches_search = (not search_text) or (search_text in row_info.get("search_text", ""))
            matches_mode = True
            if mode == "essentials":
                matches_mode = bool(row_info.get("essential"))
            elif mode == "toggles":
                matches_mode = row_info.get("type") == "bool"
            elif mode == "advanced":
                matches_mode = not bool(row_info.get("essential"))
            if matches_search and matches_mode:
                row_info["frame"].grid()
                visible_count += 1
            else:
                row_info["frame"].grid_remove()
        total_count = len(rows)
        state.get("result_var").set(f"Showing {visible_count} of {total_count} settings")

    def _set_visible_booleans(category_name, target_state: bool):
        state = category_ui_state.get(category_name, {})
        changed = 0
        for row_info in state.get("rows", []):
            if row_info.get("type") != "bool":
                continue
            if not row_info["frame"].winfo_ismapped():
                continue
            try:
                row_info["widget_ref"].set(bool(target_state))
                changed += 1
            except Exception:
                pass
        status_var.set(f"Set {changed} visible toggles {'ON' if target_state else 'OFF'} in {category_name}")

    def _reset_visible_settings(category_name):
        state = category_ui_state.get(category_name, {})
        changed = 0
        for row_info in state.get("rows", []):
            if not row_info["frame"].winfo_ismapped():
                continue
            try:
                _apply_widget_value(row_info["type"], row_info["widget_ref"], row_info["default"])
                changed += 1
            except Exception:
                pass
        status_var.set(f"Reset {changed} visible settings in {category_name} (not saved yet)")

    def _clear_category_search(category_name):
        state = category_ui_state.get(category_name, {})
        try:
            state.get("search_var").set("")
            state.get("mode_var").set("all")
        except Exception:
            pass
        _filter_category_rows(category_name)

    for category in sorted_categories:
        items = categorized_items[category]
        meta = tab_meta.get(category, {})
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=f" {category} ({len(items)}) ")
        tab_frame.grid_columnconfigure(0, weight=1)
        tab_frame.grid_rowconfigure(1, weight=1)

        top_frame = ttk.LabelFrame(tab_frame, text=meta.get("title", f"{category} Settings"))
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(
            top_frame,
            text=meta.get("description", f"Adjust {category.lower()} settings for SarahMemory."),
            font=("Segoe UI", 10),
            wraplength=1120,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))

        toolbar = ttk.Frame(top_frame)
        toolbar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        toolbar.grid_columnconfigure(1, weight=1)
        toolbar.grid_columnconfigure(7, weight=1)

        result_var = tk.StringVar(value=f"Showing {len(items)} of {len(items)} settings")
        search_var = tk.StringVar(value="")
        mode_var = tk.StringVar(value="all")

        ttk.Label(toolbar, text="Search", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        search_entry = ttk.Entry(toolbar, textvariable=search_var, width=34)
        search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(toolbar, text="All", width=8, command=lambda c=category: (category_ui_state[c]["mode_var"].set("all"), _filter_category_rows(c))).grid(row=0, column=2, padx=(0, 4))
        ttk.Button(toolbar, text="Essentials", width=10, command=lambda c=category: (category_ui_state[c]["mode_var"].set("essentials"), _filter_category_rows(c))).grid(row=0, column=3, padx=(0, 4))
        ttk.Button(toolbar, text="Toggles", width=8, command=lambda c=category: (category_ui_state[c]["mode_var"].set("toggles"), _filter_category_rows(c))).grid(row=0, column=4, padx=(0, 4))
        ttk.Button(toolbar, text="Advanced", width=9, command=lambda c=category: (category_ui_state[c]["mode_var"].set("advanced"), _filter_category_rows(c))).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(toolbar, textvariable=result_var, font=("Segoe UI", 9)).grid(row=0, column=6, sticky="w", padx=(0, 8))
        ttk.Button(toolbar, text="On", width=5, command=lambda c=category: _set_visible_booleans(c, True)).grid(row=0, column=7, sticky="e", padx=(0, 4))
        ttk.Button(toolbar, text="Off", width=5, command=lambda c=category: _set_visible_booleans(c, False)).grid(row=0, column=8, sticky="e", padx=(0, 4))
        ttk.Button(toolbar, text="Reset Visible", width=12, command=lambda c=category: _reset_visible_settings(c)).grid(row=0, column=9, sticky="e", padx=(0, 4))
        ttk.Button(toolbar, text="Clear", width=7, command=lambda c=category: _clear_category_search(c)).grid(row=0, column=10, sticky="e")

        content_frame = ttk.Frame(tab_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        canvas, scroll_frame = _sm_create_scrollable_frame(content_frame)
        scroll_frame.grid_columnconfigure(0, weight=1)

        category_ui_state[category] = {
            "rows": [],
            "search_var": search_var,
            "mode_var": mode_var,
            "result_var": result_var,
            "search_entry": search_entry,
        }

        row_index = 0
        for key, value in items:
            pretty_name = _sm_pretty_setting_name(key)
            help_text = _sm_get_setting_help_text(key, category=category, value=value)
            is_essential = _sm_is_essential_setting(key, category)
            default_value = default_settings_map.get(key, value)

            row_frame = ttk.Frame(scroll_frame, padding=(8, 8))
            row_frame.grid(row=row_index, column=0, sticky="ew", padx=2, pady=2)
            row_frame.grid_columnconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=0)
            ttk.Separator(scroll_frame, orient="horizontal").grid(row=row_index + 1, column=0, sticky="ew", padx=4, pady=(0, 0))
            row_index += 2

            info_frame = ttk.Frame(row_frame)
            info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            info_frame.grid_columnconfigure(0, weight=1)

            title_text = pretty_name + ("  • Essential" if is_essential else "")
            ttk.Label(info_frame, text=title_text, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(info_frame, text=key, font=("Consolas", 8), foreground="gray").grid(row=1, column=0, sticky="w", pady=(1, 2))
            ttk.Label(info_frame, text=help_text, font=("Segoe UI", 9), wraplength=700, justify="left").grid(row=2, column=0, sticky="w")

            control_frame = ttk.Frame(row_frame)
            control_frame.grid(row=0, column=1, sticky="ne")

            widget_kind = "readonly"
            widget_ref = None

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                status_chip = tk.StringVar(value="ON" if value else "OFF")
                widget = ttk.Checkbutton(control_frame, variable=var)
                widget.grid(row=0, column=0, sticky="w")
                ttk.Label(control_frame, textvariable=status_chip, width=5, anchor="center", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(6, 8))
                def _make_bool_trace(v=var, sv=status_chip):
                    def _update(*_args):
                        try:
                            sv.set("ON" if bool(v.get()) else "OFF")
                        except Exception:
                            sv.set("OFF")
                    return _update
                var.trace_add("write", _make_bool_trace())
                widget_kind = "bool"
                widget_ref = var
            elif isinstance(value, int):
                var = tk.IntVar(value=value)
                widget = ttk.Spinbox(control_frame, from_=-999999999, to=999999999, textvariable=var, width=18)
                widget.grid(row=0, column=0, sticky="e", padx=(0, 8))
                widget_kind = "int"
                widget_ref = var
            elif isinstance(value, float):
                var = tk.DoubleVar(value=value)
                widget = ttk.Entry(control_frame, textvariable=var, width=22)
                widget.grid(row=0, column=0, sticky="e", padx=(0, 8))
                widget_kind = "float"
                widget_ref = var
            elif isinstance(value, str):
                var = tk.StringVar(value=value)
                if key in enum_settings:
                    widget = ttk.Combobox(control_frame, values=enum_settings[key], textvariable=var, state="readonly", width=28)
                else:
                    widget = ttk.Entry(control_frame, textvariable=var, width=34)
                widget.grid(row=0, column=0, sticky="e", padx=(0, 8))
                widget_kind = "str"
                widget_ref = var
            elif isinstance(value, (list, dict)):
                import json as _json
                text_holder = ttk.Frame(control_frame)
                text_holder.grid(row=0, column=0, sticky="e", padx=(0, 8))
                text_widget = tk.Text(text_holder, height=4, width=36, wrap="word", font=("Consolas", 8))
                text_scroll = ttk.Scrollbar(text_holder, orient="vertical", command=text_widget.yview)
                text_widget.configure(yscrollcommand=text_scroll.set)
                text_widget.grid(row=0, column=0, sticky="nsew")
                text_scroll.grid(row=0, column=1, sticky="ns")
                try:
                    text_widget.insert("1.0", _json.dumps(value, indent=2))
                except Exception:
                    text_widget.insert("1.0", str(value))
                widget_kind = "json"
                widget_ref = text_widget
            else:
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(control_frame, textvariable=var, state="readonly", width=34)
                widget.grid(row=0, column=0, sticky="e", padx=(0, 8))
                widget_kind = "readonly"
                widget_ref = var

            ttk.Button(
                control_frame,
                text="Default",
                width=10,
                command=lambda kind=widget_kind, ref=widget_ref, dv=default_value, name=pretty_name: (_apply_widget_value(kind, ref, dv), status_var.set(f"Reset {name} to default (not saved yet)")),
            ).grid(row=0, column=2, sticky="e")

            all_widgets[key] = (widget_kind, widget_ref)
            category_ui_state[category]["rows"].append({
                "frame": row_frame,
                "key": key,
                "type": widget_kind,
                "widget_ref": widget_ref,
                "default": default_value,
                "essential": is_essential,
                "search_text": f"{key.lower()} {pretty_name.lower()} {help_text.lower()}",
            })

        search_var.trace_add("write", lambda *_args, c=category: _filter_category_rows(c))
        _filter_category_rows(category)

    def collect_model_settings():
        payload = {"CUSTOM_MODEL_REPOS": dict(CUSTOM_MODEL_REPOS)}
        for category_key in model_rows.keys():
            selected_repo = get_selected_repo_for_category(category_key)
            if selected_repo:
                selected_repo = _sm_set_category_model_repo(category_key, selected_repo)
                attr = _sm_get_category_primary_attr(category_key)
                if attr:
                    payload[attr] = selected_repo
        for category_key, flag_map in (MODEL_CATEGORY_FLAG_REPO_MAP or {}).items():
            for flag_name in flag_map.keys():
                if flag_name in globals():
                    payload[flag_name] = bool(globals().get(flag_name))
        return payload

    def do_save():
        changed_settings = collect_model_settings()
        errors = []
        for key, (typ, widget) in all_widgets.items():
            try:
                if typ == "bool":
                    new_value = bool(widget.get())
                elif typ == "int":
                    new_value = int(widget.get())
                elif typ == "float":
                    new_value = float(widget.get())
                elif typ == "str":
                    new_value = str(widget.get())
                elif typ == "json":
                    import json as _json
                    raw_text = widget.get("1.0", "end").strip()
                    new_value = _json.loads(raw_text) if raw_text else None
                elif typ == "readonly":
                    continue
                else:
                    continue
                globals()[key] = new_value
                changed_settings[key] = new_value
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        success, message = _sm_save_settings_to_file(changed_settings)
        refresh_all_model_rows()

        if errors:
            error_text = "\n".join(errors[:10])
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more errors"
            messagebox.showwarning("Settings Saved with Warnings", f"Saved {len(changed_settings)} setting(s).\n\nErrors:\n{error_text}\n\n{message}")
        else:
            if success:
                messagebox.showinfo("Settings Saved", f"Successfully saved {len(changed_settings)} setting(s).\n\n{message}")
            else:
                messagebox.showerror("Save Error", f"Settings updated in memory but file save failed.\n\n{message}")
        status_var.set(f"Saved {len(changed_settings)} settings")

    def do_restore_defaults():
        if not messagebox.askyesno("Restore Defaults", "Restore recommended defaults for common settings and model routing?\n\nContinue?"):
            return
        defaults = _sm_get_default_settings()
        restored_count = 0
        for key, default_value in defaults.items():
            if key in all_widgets:
                typ, widget = all_widgets[key]
                try:
                    if typ == "bool":
                        widget.set(bool(default_value))
                    elif typ == "int":
                        widget.set(int(default_value))
                    elif typ == "float":
                        widget.set(float(default_value))
                    elif typ == "str":
                        widget.set(str(default_value))
                    elif typ == "json":
                        import json as _json
                        widget.delete("1.0", "end")
                        widget.insert("1.0", _json.dumps(default_value, indent=2))
                    restored_count += 1
                except Exception:
                    pass

        default_custom = defaults.get("CUSTOM_MODEL_REPOS", {}) if isinstance(defaults.get("CUSTOM_MODEL_REPOS", {}), dict) else {}
        for category_key in model_rows.keys():
            attr = _sm_get_category_primary_attr(category_key)
            default_repo = str(defaults.get(attr, _sm_get_category_current_repo(category_key)) or "").strip()
            row = model_rows.get(category_key, {})
            if not row:
                continue
            current_values = list(row["combobox"].cget("values"))
            if default_repo and default_repo not in current_values:
                current_values.insert(0, default_repo)
                row["combobox"]["values"] = current_values
            row["choice_var"].set(default_repo or MODEL_MANAGER_CUSTOM_OPTION)
            row["custom_var"].set(str(default_custom.get(category_key, "") or ""))
            refresh_model_row(category_key)
            restored_count += 1

        status_var.set(f"Restored {restored_count} defaults (not saved yet)")
        messagebox.showinfo("Defaults Restored", f"Restored {restored_count} values.\n\nClick 'Save Settings' to persist these changes.")

    def do_exit():
        if messagebox.askyesno("Exit Settings", "Exit without saving?\n\nAny unsaved changes will be lost."):
            root.destroy()

    def do_save_and_exit():
        do_save()
        root.destroy()

    def do_env_import():
        try:
            import platform
            if platform.system().lower() != "windows":
                messagebox.showinfo("Env Import", "Windows not detected. This action is only supported on Windows 10/11.")
                return
            env_path = os.path.join(BASE_DIR, ".env")
            if not os.path.isfile(env_path):
                messagebox.showwarning("Env Import", f"No .env file found at:\n\n{env_path}\n\nCreate it first (copy from .env.example), then retry.")
                return
            summary = sm_import_env_strings_to_windows(env_path, parent=root)
            if summary.get("skipped") == -1:
                messagebox.showinfo("Env Import", "Windows not detected. No changes made.")
                return
            messagebox.showinfo(
                "Env Import Complete",
                "Operation completed.\n\n"
                f"Added: {summary.get('added',0)}\n"
                f"Overwritten: {summary.get('overwritten',0)}\n"
                f"Kept (no change): {summary.get('kept',0)}\n"
                f"Conflicts handled: {summary.get('conflicts',0)}\n"
                f"Skipped: {summary.get('skipped',0)}\n\n"
                "Note: New values apply to NEW terminals/processes. Restart your shell/IDE if needed."
            )
        except Exception as exc:
            try:
                messagebox.showerror("Env Import Failed", str(exc))
            except Exception:
                print(f"[Env Import Failed] {exc}")

    ttk.Button(footer_frame, text="Exit", command=do_exit, width=12).pack(side="right", padx=5)
    ttk.Button(footer_frame, text="Save & Exit", command=do_save_and_exit, width=12).pack(side="right", padx=5)
    ttk.Button(footer_frame, text="Save Settings", command=do_save, width=12).pack(side="right", padx=5)
    ttk.Separator(footer_frame, orient="vertical").pack(side="right", fill="y", padx=10)
    ttk.Button(footer_frame, text="Restore Defaults", command=do_restore_defaults, width=14).pack(side="right", padx=5)

    try:
        import platform as _pf
        if _pf.system().lower() == "windows":
            ttk.Button(footer_frame, text="Import .env → Windows Env", command=do_env_import, width=22).pack(side="left", padx=5)
            ttk.Button(footer_frame, text="Open Env Vars UI", command=sm_open_windows_env_vars_ui, width=16).pack(side="left", padx=5)
    except Exception:
        pass

    root.bind("<Control-s>", lambda e: do_save())
    root.bind("<Control-q>", lambda e: do_exit())
    root.bind("<Escape>", lambda e: do_exit())
    root.protocol("WM_DELETE_WINDOW", do_exit)

    refresh_all_model_rows()
    pump_download_events()

    try:
        status_var.set(f"Loaded {len(all_widgets)} settings across {len(sorted_categories)} categories")
        append_model_log("Model Manager ready. Select a model per category, optionally enter a custom Hugging Face repo, then click Apply or Download.")
        root.mainloop()
    except Exception as e:
        print(f"[Settings GUI] Mainloop error: {e}")


# === API Model Registry & Selector (v8.0.0) ==================================
# Purpose:
# - Centralize OpenAI model inventory (text/code, image, audio STT/TTS, realtime, video)
# - Provide deterministic task-based selection with cost tiers + local/offline gating
# - Remain backward compatible with prior selectors used by SarahMemoryAPI.py
#
# Notes:
# - Text/Coding: gpt-5.2 (default), gpt-5.2-pro (premium), gpt-5-mini (fast/cheap)
# - Images: gpt-image-1.5
# - STT: gpt-4o-transcribe (premium) / gpt-4o-mini-transcribe (fast)
# - TTS: gpt-4o-mini-tts (fast) / tts-1-hd (premium)
# - Video: sora-2 / sora-2-pro (capability gated; may not be enabled on all accounts)

API_MODEL_AUTO_SELECTOR = globals().get("API_MODEL_AUTO_SELECTOR", True)
API_TOKEN_SOFT_LIMIT = int(os.getenv("SARAH_API_TOKEN_SOFT_LIMIT", "1024"))
API_COST_TIER = (os.getenv("SARAH_API_COST_TIER", "balanced") or "balanced").lower()  # low|balanced|max

# v8 defaults (kept override-able via env; older blocks below may re-assign these)
API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL",   "gpt-5.2")
API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL", "gpt-5-mini")
API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL",   "gpt-5-mini")

# Optional per-modality overrides
API_IMAGE_MODEL     = os.getenv("SARAH_OPENAI_IMAGE_MODEL", "gpt-image-1.5")
API_STT_MODEL       = os.getenv("SARAH_OPENAI_STT_MODEL",   "gpt-4o-transcribe")
API_STT_FAST_MODEL  = os.getenv("SARAH_OPENAI_STT_FAST_MODEL","gpt-4o-mini-transcribe")
API_TTS_MODEL       = os.getenv("SARAH_OPENAI_TTS_MODEL",   "gpt-4o-mini-tts")
API_TTS_PREMIUM     = os.getenv("SARAH_OPENAI_TTS_PREMIUM", "tts-1-hd")
API_VIDEO_MODEL     = os.getenv("SARAH_OPENAI_VIDEO_MODEL", "sora-2")
API_VIDEO_PREMIUM   = os.getenv("SARAH_OPENAI_VIDEO_PREMIUM","sora-2-pro")

# Only allow models explicitly listed here (derived from your enabled set)
# Keep legacy list but add v8-first class models.
API_ALLOWED_MODELS = list(dict.fromkeys([

    # === v8 Primary Text/Coding ===
    "gpt-5.2","gpt-5.2-pro","gpt-5-mini",

    # === Images ===
    "gpt-image-1.5","gpt-image-1","dall-e-3","dall-e-2",

    # === Audio (STT/TTS/Realtimes) ===
    "gpt-4o-mini-transcribe","gpt-4o-transcribe","whisper-1",
    "gpt-4o-mini-tts","tts-1","tts-1-1106","tts-1-hd","tts-1-hd-1106",
    "gpt-4o-realtime-preview","gpt-4o-realtime-preview-2024-10-01","gpt-4o-realtime-preview-2025-06-03",
    "gpt-realtime","gpt-realtime-2025-08-28",
    "gpt-audio","gpt-audio-2025-08-28",
    "gpt-4o-audio-preview","gpt-4o-audio-preview-2024-10-01","gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-audio-preview","gpt-4o-mini-audio-preview-2024-12-17",

    # === Video (capability gated) ===
    "sora-2","sora-2-pro",

    # === Embeddings ===
    "text-embedding-3-small","text-embedding-3-large","text-embedding-ada-002",

    # === Legacy / Compatibility ===
    "gpt-4.1","gpt-4.1-mini","gpt-4.1-mini-2025-04-14","gpt-4.1-2025-04-14","gpt-4.1-nano",
    "o4-mini","o4-mini-2025-04-16","o4-mini-deep-research","o4-mini-deep-research-2025-06-26",
    "o3","o3-2025-04-16","o3-mini","o3-mini-2025-01-31",
    "o1","o1-2024-12-17","o1-mini","o1-mini-2024-09-12","o1-pro","o1-pro-2025-03-19",
    "gpt-4o","gpt-4o-2024-05-13","gpt-4o-2024-08-06",
    "gpt-4o-mini","gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-search-preview","gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview","gpt-4o-search-preview-2025-03-11",
    "chatgpt-4o-latest",
    "gpt-4","gpt-4-turbo","gpt-4-turbo-preview","gpt-4-0125-preview","gpt-4-1106-preview","gpt-4-0613",
    "gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-3.5-turbo-0125","gpt-3.5-turbo-16k","gpt-3.5-turbo-instruct-0914",

    # Safety/moderation
    "omni-moderation-latest","omni-moderation-2024-09-26",

    # Utility / legacy
    "babbage-002","davinci-002","codex-mini-latest",

    # Forward-compat (kept from your prior list)
    "gpt-5","gpt-5-mini-2025-08-07","gpt-5-nano","gpt-5-nano-2025-08-07","gpt-5-chat-latest",
]))

# Optional allow/blocklist overrides from env (preserve legacy knobs)
try:
    _env_allowed = [m.strip() for m in os.getenv("SARAH_OPENAI_ALLOWED_MODELS", "").split(",") if m.strip()]
    if _env_allowed:
        API_ALLOWED_MODELS = list(dict.fromkeys(_env_allowed))
except Exception:
    pass

API_BLOCKLIST_MODELS = [m.strip() for m in os.getenv("SARAH_OPENAI_BLOCKLIST_MODELS","").split(",") if m.strip()]
if API_BLOCKLIST_MODELS:
    API_ALLOWED_MODELS = [m for m in API_ALLOWED_MODELS if (m.lower() not in [b.lower() for b in API_BLOCKLIST_MODELS])]

def _model_capabilities(model_id: str) -> dict:
    mid = (model_id or "").lower()
    return {
        "vision": ("4o" in mid) or ("realtime" in mid),
        "search": ("search-preview" in mid) or ("deep-research" in mid),
        "stt": ("transcribe" in mid) or (mid == "whisper-1"),
        "tts": ("tts" in mid) or ("audio" in mid and "mini" in mid),
        "embedding": ("embedding" in mid),
        "image": ("gpt-image-1.5" in mid) or ("gpt-image-1" in mid) or ("dall-e" in mid),
        "video": ("sora" in mid) or ("video" in mid),
        "realtime": ("realtime" in mid),
        "fast": any(x in mid for x in ["mini","nano"]),
        "premium": any(x in mid for x in ["5.2-pro","o4","gpt-4.1","o3"]) and not any(x in mid for x in ["mini","nano"]),
    }

def _model_tier(model_id: str) -> str:
    caps = _model_capabilities(model_id)
    if caps["premium"]: return "max"
    if caps["fast"]:    return "low"
    return "balanced"

def _allowed(mid: str) -> bool:
    if not mid:
        return False
    try:
        if API_ALLOWED_MODELS and (mid not in API_ALLOWED_MODELS) and (mid.lower() not in [m.lower() for m in API_ALLOWED_MODELS]):
            return False
    except Exception:
        pass
    try:
        if API_BLOCKLIST_MODELS and (mid.lower() in [b.lower() for b in API_BLOCKLIST_MODELS]):
            return False
    except Exception:
        pass
    return True

# --- Endpoint-family safety shims -------------------------------------------------
# Legacy code used /v1/chat/completions. Newer stacks may use Responses API.
# Keep "chat-safe" list strict so older code doesn't break.
_OPENAI_CHAT_SAFE = [
    "gpt-4o-2024-08-06","gpt-4o-2024-05-13","gpt-4o",
    "gpt-4-turbo","gpt-4-0125-preview","gpt-4-1106-preview",
    "gpt-3.5-turbo-0125","gpt-3.5-turbo"
]

def _is_chat_safe(model_id: str) -> bool:
    mid = (model_id or "").lower()
    if not mid:
        return False
    if any(x in mid for x in ["realtime","audio","transcribe","search-preview","deep-research","image","dall-e","sora","video","embedding","tts"]):
        return False
    # keep legacy exclusions
    if mid.startswith(("o1","o3","o4","gpt-4.1","gpt-5","whisper","tts")):
        return False
    if not _allowed(model_id):
        return False
    return mid in [m.lower() for m in _OPENAI_CHAT_SAFE]

def get_openai_model_candidates(query: str = "", intent: str = "chat", max_n: int = 6) -> list[str]:
    """Return a prioritized list of OpenAI chat-safe model IDs for legacy chat-completions."""
    primary = []
    secondary = []

    for key in ("API_PRIMARY_MODEL","API_SECONDARY_MODEL","API_DEFAULT_MODEL"):
        mid = globals().get(key, None)
        if isinstance(mid, str) and _is_chat_safe(mid):
            primary.append(mid)

    try:
        for mid in API_ALLOWED_MODELS:
            if _is_chat_safe(mid):
                if ("gpt-4o" in mid) or ("gpt-4-turbo" in mid):
                    if mid not in primary: primary.append(mid)
                else:
                    if mid not in secondary: secondary.append(mid)
    except Exception:
        for mid in _OPENAI_CHAT_SAFE:
            if _is_chat_safe(mid):
                if ("gpt-4o" in mid) or ("gpt-4-turbo" in mid):
                    if mid not in primary: primary.append(mid)
                else:
                    if mid not in secondary: secondary.append(mid)

    seen = set()
    out = []
    for mid in primary + secondary + _OPENAI_CHAT_SAFE:
        if _is_chat_safe(mid) and mid.lower() not in seen:
            out.append(mid); seen.add(mid.lower())
        if len(out) >= max_n: break
    return out

def get_alternate_model(prev_model: str) -> str | None:
    cands = get_openai_model_candidates()
    for mid in cands:
        if prev_model and mid.lower() != (prev_model or "").lower():
            return mid
    return None

# --- v8 Task Selector --------------------------------------------------------
def select_task_model(task: str = "chat",
                      *,
                      cost_tier: str | None = None,
                      need_vision: bool = False,
                      need_stt: bool = False,
                      need_tts: bool = False,
                      need_image: bool = False,
                      need_video: bool = False,
                      prefers_realtime: bool = False,
                      prefers_search: bool = False) -> str:
    """Select the best OpenAI model for a task modality.

    task (examples): chat | code | reasoning | search | image | stt | tts | audio | realtime | video
    """
    ctier = (cost_tier or API_COST_TIER or "balanced").lower()
    t = (task or "chat").strip().lower()

    # Hard gating: LOCAL_ONLY_MODE means "do not use external APIs" (selection still returns a name,
    # but SarahMemoryAPI.py should respect LOCAL_ONLY_MODE and bypass calls).
    _ = bool(globals().get("LOCAL_ONLY_MODE", False))

    # Modality routing first
    if need_image or t in ("image","img","picture","art","draw"):
        mid = API_IMAGE_MODEL
        return mid if _allowed(mid) else "gpt-image-1.5"

    if need_video or t in ("video","movie","animate","sora"):
        mid = API_VIDEO_PREMIUM if ctier == "max" else API_VIDEO_MODEL
        if _allowed(mid): return mid
        if _allowed(API_VIDEO_MODEL): return API_VIDEO_MODEL
        return "sora-2"

    if need_stt or t in ("stt","transcribe","speech_to_text","asr"):
        mid = API_STT_FAST_MODEL if ctier == "low" else API_STT_MODEL
        return mid if _allowed(mid) else (API_STT_MODEL if _allowed(API_STT_MODEL) else "gpt-4o-transcribe")

    if need_tts or t in ("tts","speak","text_to_speech","voice"):
        mid = API_TTS_PREMIUM if ctier == "max" else API_TTS_MODEL
        return mid if _allowed(mid) else (API_TTS_MODEL if _allowed(API_TTS_MODEL) else "gpt-4o-mini-tts")

    if prefers_realtime or t in ("realtime","rt","live","voice_chat"):
        for cand in ["gpt-realtime-2025-08-28","gpt-realtime","gpt-4o-realtime-preview-2025-06-03","gpt-4o-realtime-preview"]:
            if _allowed(cand):
                return cand

    # Text / code / reasoning / search
    wants_search = prefers_search or t in ("search","lookup","fact","research")
    priority = [m for m in [API_PRIMARY_MODEL, API_SECONDARY_MODEL, API_DEFAULT_MODEL] if isinstance(m, str) and m.strip()]

    for m in ["gpt-5.2-pro","gpt-5.2","gpt-5-mini","gpt-4.1","gpt-4.1-mini","gpt-4o","gpt-4o-mini"]:
        if m not in priority:
            priority.append(m)

    candidates = []
    for mid in priority:
        if not _allowed(mid):
            continue
        caps = _model_capabilities(mid)
        if need_vision and not caps["vision"]:
            continue
        if wants_search and not (caps["search"] or "search-preview" in (mid or "").lower() or "deep-research" in (mid or "").lower()):
            pass
        tier = _model_tier(mid)
        if ctier == "low" and tier == "max":
            continue
        candidates.append((mid, tier))

    if API_MODEL_AUTO_SELECTOR and candidates:
        if ctier == "max":
            for mid, tier in candidates:
                if tier == "max":
                    return mid
        if ctier == "low":
            for mid, tier in candidates:
                if tier == "low":
                    return mid
        return candidates[0][0]

    for mid in priority:
        if _allowed(mid):
            return mid
    return "gpt-5-mini"

# Back-compat: keep the old function name used by SarahMemoryAPI.py
def select_api_model(intent: str = "chat",
                     need_vision: bool = False,
                     need_stt: bool = False,
                     need_tts: bool = False,
                     prefers_search: bool = False,
                     cost_tier: str | None = None,
                     token_soft_limit: int | None = None) -> str:
    """Back-compat wrapper for older code paths (text model selection)."""
    if need_stt:
        return select_task_model("stt", cost_tier=cost_tier, need_stt=True)
    if need_tts:
        return select_task_model("tts", cost_tier=cost_tier, need_tts=True)
    return select_task_model(intent or "chat", cost_tier=cost_tier, need_vision=need_vision, prefers_search=prefers_search)

def get_embedding_model(max_quality: bool = False) -> str:
    return "text-embedding-3-large" if max_quality else "text-embedding-3-small"

try:
    API_DEFAULT_MODEL = API_DEFAULT_MODEL.replace("gpt-4.0-mini", "gpt-4.1-mini")
except Exception:
    pass
# ============================================================================


# === AI-Agent Master Safety & Voice Control (v7.7.2) ===
# Single master safety gate: when False, Sarah behaves like a normal chatbot (no desktop control).
AI_AGENT_ENABLED = os.getenv("SARAH_AI_AGENT_ENABLED", "True").strip().lower() in ("1","true","yes","on")
# Emergency / control phrases (lowercased exact match)
AI_AGENT_STOP_PHRASES   = ["sarah stop now", "emergency stop", "abort mission"]
AI_AGENT_HALT_PHRASES   = ["halt", "pause", "hold"]
AI_AGENT_RESUME_PHRASES = ["resume", "continue", "go on"]
AI_AGENT_CONFIRM_YES    = ["yes", "confirm", "ok", "okay", "yep"]
AI_AGENT_CONFIRM_NO     = ["no", "cancel", "stop", "nope"]

# Idle delay before auto-resume after human input stops (milliseconds)
AI_AGENT_RESUME_DELAY = int(os.getenv("SARAH_AI_AGENT_RESUME_DELAY_MS", "9000"))

# Human activity grace window: any keyboard/mouse/controller input within this window halts the agent (milliseconds)
AI_AGENT_USER_ACTIVITY_TIMEOUT_MS = int(os.getenv("SARAH_AI_AGENT_USER_TIMEOUT_MS", "2500"))

# Allowed UI operations (the agent will never copy/move/delete files without explicit consent)
AI_AGENT_ALLOWLIST = {"open","launch","focus","maximize","minimize","click","doubleclick","type","press","scroll","play","search","move","wait","close","terminate"}
#AI_AGENT_REQUIRE_CONSENT = {"install","uninstall","system_setting","purchase"}
#AI_AGENT_REQUIRE_CONSENT = [p.lower() for p in getattr(config, "AI_AGENT_REQUIRE_CONSENT", [])]
#AI_AGENT_REQUIRE_CONSENT = [s.strip().lower() for s in os.getenv("SARAH_AI_AGENT_REQUIRE_CONSENT","").split(",") if s.strip()]
AI_AGENT_REQUIRE_CONSENT = []
# Game/learning toggles
AI_GAME_MODE_ENABLED = True
AI_GAME_FULL_AUTO = True
  # When True + Agent enabled, the agent may run exploration macros (still respects HALT/resume).

# Helper: current monotonic ms
def now_ms():
    try:
        return int(time.monotonic() * 1000)
    except Exception:
        return int(time.time() * 1000)

# --- Auto-added by network hub patch ---
SARAH_WEB_BASE = "https://www.sarahmemory.com"
REMOTE_SYNC_ENABLED = True
REMOTE_HTTP_TIMEOUT = 6.0
REMOTE_HEARTBEAT_SEC = 30
REMOTE_API_KEY = None
SARAH_WEB_API_PREFIX = "/api"
SARAH_WEB_PING_PATH = "/health"
SARAH_WEB_HEALTH_PATH = "/health"
SARAH_WEB_RELAY_PATH = "/relay"
SARAH_WEB_REGISTER_PATH = "/register-node"
SARAH_WEB_HEARTBEAT_PATH = "/heartbeat"
SARAH_WEB_EMBED_PATH = "/receive-embedding"
SARAH_WEB_CONTEXT_PATH = "/context-update"
SARAH_WEB_JOBS_PATH = "/jobs"


# --- injected: on-demand ensure table for `response` ---
def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
try:
    _ensure_response_table()
except Exception:
    pass
# === v7.7.3 Emotional Realism Feature Gates (surgical, reversible) ===
EMOTION_REALISM_ENABLED      = True
FACIAL_FEEDBACK_ENABLED      = True
PERSONALITY_DRIFT_ENABLED    = True
EXPRESSIVE_OUTPUT_ENABLED    = True
FOLLOWUP_QUESTIONS_ENABLED   = True
ETHICS_FILTER_ENABLED        = True
BANDWIDTH_AWARE_INTELLIGENCE = True

EMOJI_POLICY = {
    "joy": ["ðŸ˜„", "ðŸ˜Š", "âœ¨"],
    "neutral": ["ðŸ™‚"],
    "sad": ["ðŸ˜”", "ðŸ’™"],
    "anger": ["ðŸ˜¤"],
    "concern": ["ðŸ¤","ðŸ«¶"],
    "curiosity": ["ðŸ¤”"]
}

def reduced_mode_suggested(cpu_pct: float = None, mem_pct: float = None, net_bps: float = None) -> bool:
    try:
        import psutil
        cpu_pct = cpu_pct if cpu_pct is not None else psutil.cpu_percent(interval=0.0)
        mem_pct = mem_pct if mem_pct is not None else psutil.virtual_memory().percent
    except Exception:
        return SAFE_MODE if 'SAFE_MODE' in globals() else False
    if not BANDWIDTH_AWARE_INTELLIGENCE:
        return SAFE_MODE if 'SAFE_MODE' in globals() else False
    return (SAFE_MODE if 'SAFE_MODE' in globals() else False) or cpu_pct >= 82 or mem_pct >= 85

def emotion_to_color(primary_label: str) -> str:
    m = (primary_label or "neutral").lower()
    if m in ("anger","fear"): return "#FF0000"
    if m in ("sad","concern"): return "#FFFF00"
    return "#00FF00"

# ====== Emotional Fine-Tuning Knobs (append-only; v7.7.3) ======
try:
    EMO_REWRITE_STRENGTH      # 0..1 intensity of phrasing rewrite
except NameError:
    EMO_REWRITE_STRENGTH = 0.55
try:
    FOLLOWUP_MAX_QUESTIONS
except NameError:
    FOLLOWUP_MAX_QUESTIONS = 2
try:
    EXPRESSIVE_MAX_EMOJI
except NameError:
    EXPRESSIVE_MAX_EMOJI = 1
try:
    DRIFT_LEARNING_RATE
except NameError:
    DRIFT_LEARNING_RATE = 0.02
def get_rewrite_strength():
    try:
        return float(EMO_REWRITE_STRENGTH)
    except Exception:
        return 0.5


# === Performance & Tuning =========================
PERF_FAST_FIRST = True
RESPONSE_CACHE_TTL = 900
COMPARE_MIN_CHARS = 220
COMPARE_INTENTS = ["question","explanation","research","identity","story"]
FAST_MODEL_PREFERENCE = ["gpt-4o-mini","gpt-3.5-turbo-0125","gpt-3.5-turbo","gpt-4o"]
ENABLE_DB_WAL = True
SQLITE_PRAGMAS = {"journal_mode":"WAL","synchronous":1,"temp_store":2,"mmap_size":268435456}
# ================================================

def reorder_fast_first(candidates: list[str]) -> list[str]:
    try:
        pref = [m.lower() for m in FAST_MODEL_PREFERENCE]
        cands = list(candidates or [])
        cands_l = [m.lower() for m in cands]
        out = []
        used = set()
        for m in pref:
            if m in cands_l:
                idx = cands_l.index(m)
                out.append(cands[idx]); used.add(idx)
        for i, m in enumerate(cands):
            if i not in used:
                out.append(m)
        return out
    except Exception:
        return candidates or []

def apply_sqlite_pragmas(conn):
    try:
        if not conn: return
        cur = conn.cursor()
        prag = globals().get("SQLITE_PRAGMAS", {})
        for k,v in prag.items():
            try:
                cur.execute(f"PRAGMA {k}={v}")
            except Exception:
                pass
        try:
            conn.commit()
        except Exception:
            pass
    except Exception:
        pass

# ============================================================================
# Back-compat shim (v7.7.4):
# Some legacy paths referenced `SarahMemoryGlobals.SarahMemoryGlobals.<X>`.
# This module never exposed such a nested object; to remain backward compatible
# we provide a lightweight proxy that forwards attribute lookups to the module
# globals. This prevents AttributeError at runtime without changing call sites.
# ============================================================================
class _GlobalsProxy:
    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError as e:
            raise AttributeError(f"SarahMemoryGlobals has no attribute {{name}}") from e

# Export the alias expected by older code paths
SarahMemoryGlobals = _GlobalsProxy()


# === v7.7.5 GUI WebUI additions (non-breaking) ===
try:
    BASE_DIR
except NameError:
    import os as _os2
    BASE_DIR = _os2.getcwd()

try:
    THEMES_DIR
except NameError:
    import os as _os3
    THEMES_DIR = _os3.path.join(BASE_DIR, "data", "mods", "themes")

# Prefer modern webview for the new chat UI
try:
    USE_WEBVIEW
except NameError:
    USE_WEBVIEW = True

try:
    WEBUI_HTML_PATH
except NameError:
    import os as _os4
    WEBUI_HTML_PATH = _os4.path.join(BASE_DIR, "data", "ui", "SarahMemory.html")

# Bridge origin allowlist for JSâ†’Python
try:
    BRIDGE_ALLOWED_ORIGINS
except NameError:
    BRIDGE_ALLOWED_ORIGINS = {"file://", "https://api.sarahmemory.com", "https://www.sarahmemory.com"}

def origin_allowed(origin: str) -> bool:
    try:
        return any(origin.startswith(o) for o in BRIDGE_ALLOWED_ORIGINS)
    except Exception:
        return True

# System resource push interval (ms) for the top bar indicators
try:
    SYSRES_UPDATE_MS
except NameError:
    SYSRES_UPDATE_MS = 1000

# === SM_PORTABLE_PATHS_V1 ===
try:
    _SM_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if 'BASE_DIR' not in globals() or not globals().get('BASE_DIR'):
        BASE_DIR = _SM_THIS_DIR
except Exception:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if 'PUBLIC_DIR' not in globals():
    PUBLIC_DIR = BASE_DIR
if 'WEB_DIR' not in globals():
    WEB_DIR = BASE_DIR
if 'DATA_DIR' not in globals():
    DATA_DIR = os.path.join(BASE_DIR, 'data')

_THEMES_A = os.path.join(DATA_DIR, 'mods', 'themes')
_THEMES_B = os.path.join(DATA_DIR, 'themes')
try:
    THEMES_DIR = _THEMES_A if os.path.isdir(_THEMES_A) else _THEMES_B
except Exception:
    THEMES_DIR = _THEMES_A

if 'DATASETS_DIR' not in globals():
    DATASETS_DIR = os.path.join(DATA_DIR, 'memory', 'datasets')
if 'STATIC_DIR' not in globals():
    STATIC_DIR = os.path.join(BASE_DIR, 'server', 'static')

# ============================================================================
# Phase B: Context Engine, Agent Permissions, and Mesh Sync (v7.7.5)
# NOTE: This block is strictly additive and does not remove or rename any
# existing globals. It exposes higher-level knobs that downstream modules
# (AiFunctions, AdvCU, Network, GUI) can query when routing intents.
# ============================================================================

# ----- B1: Context Engine Configuration -------------------------------------
try:
    CONTEXT_ENGINE_ENABLED
except NameError:
    # Mirror the existing ENABLE_CONTEXT_BUFFER flag so older code keeps working
    CONTEXT_ENGINE_ENABLED = globals().get("ENABLE_CONTEXT_BUFFER", True)

# Where long-term contextual turns will be stored by default
try:
    CONTEXT_DB_PATH
except NameError:
    CONTEXT_DB_PATH = os.path.join(DATASETS_DIR, "context_history.db")

# Maximum number of turns and maximum age (in seconds) to consider when
# building a contextual prompt. Downstream code (SarahMemoryAiFunctions.py)
# decides *how* to use these.
try:
    CONTEXT_MAX_TURNS
except NameError:
    try:
        _default_ctx = int(globals().get("CONTEXT_BUFFER_SIZE", 10))
    except Exception:
        _default_ctx = 10
    CONTEXT_MAX_TURNS = int(os.getenv("SARAH_CONTEXT_MAX_TURNS", str(_default_ctx)) or _default_ctx)

try:
    CONTEXT_MAX_AGE_SEC
except NameError:
    # 3 days by default; can be overridden from .env
    CONTEXT_MAX_AGE_SEC = int(os.getenv("SARAH_CONTEXT_MAX_AGE_SEC", "259200") or 259200)

try:
    CONTEXT_PERSIST_TO_DB
except NameError:
    # When False, the engine may hold context only in memory (per-process).
    CONTEXT_PERSIST_TO_DB = True


def get_context_config() -> dict:
    """
    Small helper for downstream modules to introspect context-engine settings
    without importing a larger pile of globals.
    """
    return {
        "enabled":           bool(globals().get("CONTEXT_ENGINE_ENABLED", True)),
        "buffer_size":       int(globals().get("CONTEXT_BUFFER_SIZE", 10)),
        "max_turns":         int(globals().get("CONTEXT_MAX_TURNS", 10)),
        "max_age_sec":       int(globals().get("CONTEXT_MAX_AGE_SEC", 259200)),
        "persist_to_db":     bool(globals().get("CONTEXT_PERSIST_TO_DB", True)),
        "db_path":           str(globals().get("CONTEXT_DB_PATH", os.path.join(DATASETS_DIR, "context_history.db"))),
        "enrichment_enabled":bool(globals().get("ENABLE_CONTEXT_ENRICHMENT", True)),
    }


# ----- B3: Agent Permission & Safety Profile --------------------------------

def is_cloud_run() -> bool:
    """True when SarahMemory is running on a cloud host (e.g., PythonAnywhere)."""
    try:
        return (globals().get("RUN_MODE", "local") == "cloud")
    except Exception:
        return False


def is_public_web_mode() -> bool:
    """True when this instance is primarily serving a browser-only UI."""
    try:
        return globals().get("DEVICE_MODE") == globals().get("DEVICE_MODE_PUBLIC_WEB")
    except Exception:
        return False


def _default_agent_gate(local_default: str = "true", cloud_default: str = "false") -> str:
    """
    Helper for setting conservative defaults:
      - local desktop agent  -> permissive by default
      - cloud/public web     -> locked down by default
    Returns a string used as the default in _env_flag so it can be overridden
    from the environment.
    """
    try:
        if is_cloud_run() or is_public_web_mode():
            return cloud_default
    except Exception:
        pass
    return local_default


# High level ability flags. These DO NOT perform any actions; they simply
# describe what *categories* of actions the agent layer is allowed to attempt.
# SarahMemoryAiFunctions.py will read these before calling any OS helpers.

AI_AGENT_ALLOW_APP_LAUNCH = _env_flag(
    "SARAH_AGENT_ALLOW_APP_LAUNCH",
    _default_agent_gate(local_default="true", cloud_default="false"),
)

AI_AGENT_ALLOW_FILE_WRITE = _env_flag(
    "SARAH_AGENT_ALLOW_FILE_WRITE",
    _default_agent_gate(local_default="false", cloud_default="false"),
)

AI_AGENT_ALLOW_REMOTE_CONTROL = _env_flag(
    "SARAH_AGENT_ALLOW_REMOTE_CONTROL",
    _default_agent_gate(local_default="false", cloud_default="false"),
)

AI_AGENT_ALLOW_NETWORK_TASKS = _env_flag(
    "SARAH_AGENT_ALLOW_NETWORK_TASKS",
    _default_agent_gate(local_default="true", cloud_default="true"),
)


def agent_permissions_summary() -> dict:
    """
    Compact view of agent permissions and environment, useful for both
    Diagnostics (SarahMemoryDiagnostics.py) and the GUI.
    """
    return {
        "run_mode":               globals().get("RUN_MODE", "local"),
        "device_mode":            globals().get("DEVICE_MODE", "headless"),
        "safe_mode":              bool(globals().get("SAFE_MODE", False)),
        "local_only":             bool(globals().get("LOCAL_ONLY_MODE", False)),
        "agent_enabled":          bool(globals().get("AI_AGENT_ENABLED", False)),
        "allow_app_launch":       bool(globals().get("AI_AGENT_ALLOW_APP_LAUNCH", False)),
        "allow_file_write":       bool(globals().get("AI_AGENT_ALLOW_FILE_WRITE", False)),
        "allow_remote_control":   bool(globals().get("AI_AGENT_ALLOW_REMOTE_CONTROL", False)),
        "allow_network_tasks":    bool(globals().get("AI_AGENT_ALLOW_NETWORK_TASKS", False)),
        "user_activity_timeout":  int(globals().get("AI_AGENT_USER_ACTIVITY_TIMEOUT_MS", 2500)),
        "resume_delay_ms":        int(globals().get("AI_AGENT_RESUME_DELAY", 9000)),
    }


# ----- B4: Mesh / Hub Sync Toggle Layer -------------------------------------

try:
    MESH_SYNC_ENABLED
except NameError:
    # Mirror existing knobs but keep everything override-able via env.
    base_default = "true" if globals().get("SARAHNET_ENABLED", True) and globals().get("REMOTE_SYNC_ENABLED", True) else "false"
    MESH_SYNC_ENABLED = _env_flag("SARAH_MESH_SYNC_ENABLED", base_default)

try:
    ALLOW_HUB_SYNC
except NameError:
    # When False, nodes may still use SarahNet peer-to-peer but will not talk
    # to the central https://www.sarahmemory.com hub.
    hub_default = "false" if os.getenv("SARAH_FORCE_OFFLINE", "").lower() in ("1","true","yes") else "true"
    ALLOW_HUB_SYNC = _env_flag("SARAH_ALLOW_HUB_SYNC", hub_default)

try:
    MESH_SYNC_SAFE_MODE_ONLY
except NameError:
    # When True, mesh sync traffic is allowed only while SAFE_MODE is enabled,
    # giving an additional "double opt-in" feel for sensitive deployments.
    MESH_SYNC_SAFE_MODE_ONLY = _env_flag("SARAH_MESH_SYNC_SAFE_ONLY", "false")


def get_mesh_sync_config() -> dict:
    """
    Return a merged view of mesh/hub sync policy for use by:
      - SarahMemoryNetwork.py
      - SarahMemoryAiFunctions.py (hub helpers)
      - app.py (hub endpoints)
    """
    safe_mode = bool(globals().get("SAFE_MODE", False))
    mesh_enabled = bool(globals().get("MESH_SYNC_ENABLED", True))
    hub_allowed = bool(globals().get("ALLOW_HUB_SYNC", True))
    node_name = globals().get("NODE_NAME", globals().get("SARAHNET_NODE_ID", "SarahMemoryNode"))
    return {
        "node_name":              node_name,
        "mesh_enabled":           mesh_enabled,
        "hub_allowed":           hub_allowed,
        "safe_mode":              safe_mode,
        "safe_mode_only":         bool(globals().get("MESH_SYNC_SAFE_MODE_ONLY", False)),
        "sarahnet_enabled":       bool(globals().get("SARAHNET_ENABLED", True)),
        "web_base":               globals().get("SARAH_WEB_BASE", "https://www.sarahmemory.com"),
        "remote_sync_enabled":    bool(globals().get("REMOTE_SYNC_ENABLED", True)),
        "heartbeat_sec":          float(globals().get("REMOTE_HEARTBEAT_SEC", 30)),
        "http_timeout":           float(globals().get("REMOTE_HTTP_TIMEOUT", 6.0)),
    }

# Best-effort warm-up of governed core registry.
# This never imports discovered modules; it only scans, classifies, validates,
# and writes registry/quarantine metadata so routing layers can consult Globals
# without auto-activating arbitrary files.
try:
    if SM_CORE_DISCOVERY_ENABLED and SM_CORE_DISCOVERY_ON_BOOT:
        sm_refresh_core_registry(force=True)
except Exception as e:
    try:
        logger.warning("[CORE_DISCOVERY] bootstrap refresh failed: %s", e)
    except Exception:
        pass

# ============================================================================
# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# When this file is run directly (python SarahMemoryGlobals.py), it will:
# 1. Ensure all required directories exist
# 2. Import any datasets
# 3. Launch the Settings GUI for configuration
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("SarahMemory Global Settings Configuration")
    print(f"Version: {PROJECT_VERSION}")
    print(f"Author: {AUTHOR}")
    print("=" * 70)
    
    # Ensure directories exist
    try:
        ensure_directories()
        print("[OK] Directories verified/created")
    except Exception as e:
        print(f"[WARN] Directory setup: {e}")
    
    # Import datasets (optional, may be empty on first run)
    try:
        datasets = import_datasets()
        print(f"[OK] Loaded {len(datasets)} dataset records")
    except Exception as e:
        print(f"[INFO] Dataset import skipped: {e}")
    
    # Launch Settings GUI
    print("")
    print("Launching Settings + Model Manager GUI...")
    print("(Close the window or press Ctrl+C to exit)")
    print("")
    
    try:
        launch_settings_gui()
    except KeyboardInterrupt:
        print("\n[EXIT] User cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] GUI launch failed: {e}")
        print("[INFO] You can edit settings.json directly or set environment variables")
        sys.exit(1)

# ====================================================================
# END OF SarahMemoryGlobals.py v8.0.0
# ====================================================================

# =============================================================================
# V10/V9C Universal Runtime Body + Memory Authority Contract
# =============================================================================
# Development-mode contract layer. These flags intentionally coexist with older
# scattered model/provider flags. The setup mechanism can consolidate them later.
SARAH_AI_DEFAULT_NAME = os.getenv("SARAH_AI_DEFAULT_NAME", "SarahMemory").strip() or "SarahMemory"
SARAH_AI_ACTIVE_NAME = os.getenv("SARAH_AI_ACTIVE_NAME", os.getenv("SARAH_ACTIVE_NAME", "Sarah")).strip() or "Sarah"
SARAH_AI_NAME_SOURCE = os.getenv("SARAH_AI_NAME_SOURCE", "default").strip() or "default"
SARAH_AI_PLATFORM_NAME = os.getenv("SARAH_AI_PLATFORM_NAME", "SarahMemory AiOS").strip() or "SarahMemory AiOS"
SARAH_AI_NAME_USER_OVERRIDE_ALLOWED = _env_flag("SARAH_AI_NAME_USER_OVERRIDE_ALLOWED", "true")
SARAH_STATIC_ROUTER_FALLBACK_ENABLED = _env_flag("SARAH_STATIC_ROUTER_FALLBACK_ENABLED", "true")
SARAH_MODEL_ASSISTED_CLASSIFICATION = _env_flag("SARAH_MODEL_ASSISTED_CLASSIFICATION", "true")


def get_identity_contract() -> dict:
    """Return governed identity defaults without forcing a permanent DB profile."""
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "default_name": SARAH_AI_DEFAULT_NAME,
        "active_name": SARAH_AI_ACTIVE_NAME,
        "name_source": SARAH_AI_NAME_SOURCE,
        "platform": SARAH_AI_PLATFORM_NAME,
        "creator": globals().get("AUTHOR", "Brian Lee Baros"),
        "organization": os.getenv("SARAH_ORG_NAME", "SOFTDEV0 LLC"),
        "user_override_allowed": bool(SARAH_AI_NAME_USER_OVERRIDE_ALLOWED),
        "static_fallback_allowed": True,
    }


def get_model_availability_contract(category: str = "reasoning") -> dict:
    """Summarize model availability from current scattered development flags."""
    selected = None
    fallbacks = []
    tier = "POOR"
    source = "none"
    try:
        if "resolve_model" in globals() and callable(globals().get("resolve_model")):
            res = resolve_model(category, text="", meta=None, models_dir=globals().get("MODELS_DIR", None)) or {}
            if isinstance(res, dict):
                selected = res.get("selected")
                fallbacks = list(res.get("fallbacks") or [])
                tier = str(res.get("tier_rating") or res.get("tier") or tier).upper()
                source = str(res.get("source") or "resolver")
    except Exception:
        selected = None
        fallbacks = []
    enabled = []
    try:
        cfg = globals().get("MODEL_CONFIG", {}) or {}
        if isinstance(cfg, dict):
            enabled = [str(k) for k, v in cfg.items() if bool(v)]
    except Exception:
        enabled = []
    lm_present = bool(selected or fallbacks or enabled)
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "category": str(category or "reasoning"),
        "lm_present": lm_present,
        "local_llm_available": lm_present,
        "selected_model": selected,
        "fallback_models": fallbacks,
        "enabled_model_count": len(enabled),
        "tier_rating": tier,
        "source": source,
        "model_assisted_classification": bool(SARAH_MODEL_ASSISTED_CLASSIFICATION and lm_present),
        "static_router_fallback_enabled": bool(SARAH_STATIC_ROUTER_FALLBACK_ENABLED),
    }


def get_runtime_body_contract() -> dict:
    """Describe the currently inhabited runtime body without asserting hardware facts."""
    try:
        rm = get_runtime_meta() if "get_runtime_meta" in globals() else {}
    except Exception:
        rm = {}
    run_mode = str(globals().get("RUN_MODE", rm.get("run_mode", "local")) or "local").lower()
    device_mode = str(globals().get("DEVICE_MODE", rm.get("device_mode", "local_agent")) or "local_agent")
    if run_mode == "cloud":
        body_type = "cloud_demo_server" if device_mode == "public_web" else "cloud_server"
    elif device_mode == "headless":
        body_type = "edge_or_headless_node"
    elif "robot" in os.getenv("SARAH_BODY_TYPE", "").lower():
        body_type = "robot"
    else:
        body_type = os.getenv("SARAH_BODY_TYPE", "local_agent_body").strip() or "local_agent_body"
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "body_type": body_type,
        "run_mode": run_mode,
        "device_mode": device_mode,
        "device_profile": str(globals().get("DEVICE_PROFILE", rm.get("device_profile", "Standard")) or "Standard"),
        "body_authority": "local_selfaware_evidence_court",
        "memory_authority": "runtime_memory_policy",
        "action_authority": "SMGET_governed",
        "network_role": str(globals().get("NODE_NAME", "SarahMemoryNode")),
        "body_map_required": True,
    }


def get_memory_authority_contract() -> dict:
    """Return memory-write policy for this runtime body/deployment surface."""
    run_mode = str(globals().get("RUN_MODE", "local") or "local").lower()
    device_mode = str(globals().get("DEVICE_MODE", "local_agent") or "local_agent")
    local_only = bool(globals().get("LOCAL_ONLY_MODE", False))
    cloud_enabled = bool(globals().get("CLOUD_DB_ENABLED", False)) and not local_only
    if run_mode == "cloud" and cloud_enabled:
        primary = "remote_sql"
    else:
        primary = "sqlite"
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "run_mode": run_mode,
        "device_mode": device_mode,
        "primary_memory_backend": primary,
        "cloud_memory_backend_allowed": bool(cloud_enabled),
        "cloud_sync_allowed": bool(cloud_enabled and run_mode != "test"),
        "private_memory_mode": bool(local_only or run_mode == "local"),
        "demo_memory_mode": bool(run_mode == "cloud" or device_mode == "public_web"),
        "hardware_fact_persistence": "blocked",
        "volatile_body_fact_policy": "redacted_audit_only",
        "identity_profile_persistence": "allowed_local" if primary == "sqlite" else "allowed_demo_scope",
        "chat_history_persistence": "allowed_local" if primary == "sqlite" else "allowed_demo_scope",
        "redacted_audit_allowed": True,
    }


def get_chat_runtime_contract() -> dict:
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "identity": get_identity_contract(),
        "runtime_body": get_runtime_body_contract(),
        "memory_policy": get_memory_authority_contract(),
        "model_state": get_model_availability_contract("reasoning"),
    }
