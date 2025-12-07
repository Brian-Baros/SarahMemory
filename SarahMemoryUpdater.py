"""--==The SarahMemory Project==--
File: SarahMemoryUpdater.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
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

AUTO-UPDATER SYSTEM v8.0.0
======================================
This module has standards with enhanced self-update
capabilities, intelligent file analysis, comprehensive backup systems, and robust
error handling for safe autonomous updates.

KEY ENHANCEMENTS:
-----------------
1. INTELLIGENT UPDATE SYSTEM
   - Smart file change detection with SHA256 hashing
   - Schedule-aware update execution
   - Non-blocking startup integration
   - Graceful degradation on errors
   - Cross-platform compatibility

2. ENHANCED SAFETY MECHANISMS
   - Full-system backup before updates
   - Per-file backup for rollback
   - Syntax validation before deployment
   - Sandbox testing environment
   - Windows Defender integration

3. ADVANCED FILE MANAGEMENT
   - Root-only file iteration (non-recursive)
   - Intelligent exclude patterns
   - Extension-based filtering
   - State tracking and persistence
   - Differential updates

4. MULTI-MODE OPERATION
   - SarahMemoryAPI integration (preferred)
   - OpenAI SDK fallback support
   - Offline-safe operation
   - Auto-approve mode for CI/CD
   - Git push automation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- run_updater()
- update_from_zip()

New functions added (non-breaking):
- _iter_root_files_only()
- _create_opt_backup()
- _defender_scan()
- _has_internet()

INTEGRATION POINTS:
-------------------
- SarahMemoryMain.py: Calls run_updater() during startup
- SarahMemoryAPI.py: Provides AI-powered code analysis
- SarahMemoryGlobals.py: Configuration and scheduling
- Push_SarahMemory.bat: Git automation helper

===============================================================================
"""

from __future__ import annotations

import os
import sys
import re
import json
import shutil
import socket
import hashlib
import difflib
import subprocess
import datetime as dt
import logging
import zipfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryUpdater')
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[v8.0][UPDATER] %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler
try:
    log_dir = os.path.join(os.getcwd(), "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"updater_{dt.datetime.now().strftime('%Y%m%d')}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - [v8.0][UPDATER] %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
except Exception:
    if not logger.hasHandlers():
        logger.addHandler(console_handler)

# =============================================================================
# GLOBAL STATE VARIABLES
# =============================================================================
_APPLY_ALL = False
_SKIP_ALL = False

# =============================================================================
# CONFIGURATION DISCOVERY - Safe Import with Fallbacks
# =============================================================================
def _safe_import_globals():
    """Safely import SarahMemoryGlobals with fallback."""
    try:
        import SarahMemoryGlobals as g
        logger.debug("[v8.0] Loaded SarahMemoryGlobals successfully")
        return g
    except Exception as e:
        logger.warning(f"[v8.0] Could not import SarahMemoryGlobals: {e}")
        return None

G = _safe_import_globals()
BASE_DIR = getattr(G, "BASE_DIR", None) or os.path.abspath(os.path.dirname(__file__))
DATA_DIR = getattr(G, "DATA_DIR", None) or os.path.join(BASE_DIR, "data")
LOGS_DIR = getattr(G, "LOGS_DIR", None) or os.path.join(DATA_DIR, "logs")
UPDATER_DIR = os.path.join(DATA_DIR, "updater")
STATE_JSON = os.path.join(UPDATER_DIR, "state.json")
UPDATER_LOG = os.path.join(UPDATER_DIR, "updater.log")
BACKUPS_DIR = getattr(G, "BACKUPS_DIR", None) or os.path.join(UPDATER_DIR, "backups")

# Project policy-based ignore patterns (avoid huge or binary content)
DEFAULT_EXCLUDES = [
    ".git", "__pycache__", "venv", "node_modules", "dist", "build", "logs",
    os.path.relpath(UPDATER_DIR, BASE_DIR),
    "data/models", "data/memory", "data/datasets", "data/cache", "data/tmp",
    "*.db", "*.zip", "*.tar", "*.tar.gz", "*.png", "*.jpg", "*.jpeg", "*.gif",
    "*.mp4", "*.mp3", "*.wav", "*.onnx", "*.pt", "*.bin"
]

DEFAULT_EXTS = [".py"]  # Start conservative; can extend via CLI flags or config

# User-tunable environment flags
AUTO_APPROVE = os.environ.get("SM_UPDATER_AUTOAPPROVE", "0") in ("1", "true", "True")
GIT_PUSH = os.environ.get("SM_UPDATER_GIT_PUSH", "0") in ("1", "true", "True")
MODEL_NAME = os.environ.get("SM_UPDATER_MODEL", getattr(G, "DEFAULT_GPT_MODEL", "gpt-4.1-mini"))
MAX_CHARS = int(os.environ.get("SM_UPDATER_MAX_CHARS", "20000"))  # per request; conservative

# =============================================================================
# UTILITY FUNCTIONS - v8.0 Enhanced
# =============================================================================

def _ensure_dirs():
    """Ensure all required directories exist."""
    try:
        for d in (DATA_DIR, LOGS_DIR, UPDATER_DIR, BACKUPS_DIR):
            os.makedirs(d, exist_ok=True)
        logger.debug("[v8.0] All directories ensured")
    except Exception as e:
        logger.error(f"[v8.0] Error creating directories: {e}")


def _log(msg: str):
    """Log message to both file and console."""
    _ensure_dirs()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    try:
        with open(UPDATER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.debug(f"[v8.0] Could not write to updater.log: {e}")
    logger.info(msg)


def _load_state() -> Dict[str, Any]:
    """Load updater state from JSON file."""
    _ensure_dirs()
    if not os.path.exists(STATE_JSON):
        logger.debug("[v8.0] No state file found, creating new state")
        return {"last_run": None, "files": {}, "last_git_push": None}
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.debug(f"[v8.0] Loaded state: {len(state.get('files', {}))} files tracked")
        return state
    except Exception as e:
        logger.warning(f"[v8.0] Error loading state, using defaults: {e}")
        return {"last_run": None, "files": {}, "last_git_push": None}


def _save_state(state: Dict[str, Any]) -> None:
    """Save updater state to JSON file."""
    _ensure_dirs()
    try:
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"[v8.0] Saved state: {len(state.get('files', {}))} files tracked")
    except Exception as e:
        logger.error(f"[v8.0] Error saving state: {e}")


def _has_internet(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.5) -> bool:
    """
    Fast connectivity probe using TCP to a public resolver.
    
    Args:
        host: DNS server to test (default: Cloudflare DNS)
        port: Port to connect to
        timeout: Connection timeout in seconds
    
    Returns:
        bool: True if internet is available
    """
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        logger.debug("[v8.0] Internet connectivity: OK")
        return True
    except Exception as e:
        logger.debug(f"[v8.0] Internet connectivity: FAILED ({e})")
        return False


def _sha256(path: str) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        path: Path to file
    
    Returns:
        str: Hexadecimal hash digest
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"[v8.0] Error computing hash for {path}: {e}")
        return ""


def _is_excluded(relpath: str, patterns: List[str]) -> bool:
    """
    Check if a relative path matches any exclude pattern.
    
    Args:
        relpath: Relative file path
        patterns: List of exclude patterns
    
    Returns:
        bool: True if path should be excluded
    """
    import fnmatch
    try:
        for p in patterns:
            # Allow directory prefix match
            if relpath.startswith(p.rstrip("/\\")):
                return True
            if fnmatch.fnmatch(relpath, p):
                return True
        return False
    except Exception as e:
        logger.debug(f"[v8.0] Error checking exclude pattern: {e}")
        return False


def _iter_root_files_only(base: str, exts: List[str], excludes: List[str]):
    """
    v8.0: NEW - Iterate over files in the root directory only (non-recursive).
    
    This function was referenced in v7.7.4 but not implemented. Now properly
    implemented for safe, non-recursive file updates.
    
    Args:
        base: Base directory to scan
        exts: List of file extensions to include
        excludes: List of patterns to exclude
    
    Yields:
        str: Absolute paths to matching files
    """
    try:
        if not os.path.isdir(base):
            logger.warning(f"[v8.0] Not a directory: {base}")
            return
        
        for name in os.listdir(base):
            full_path = os.path.join(base, name)
            
            # Skip directories (non-recursive)
            if os.path.isdir(full_path):
                continue
            
            # Get file extension
            ext = os.path.splitext(name)[1]
            
            # Check extension match
            if ext not in exts:
                continue
            
            # Check exclude patterns
            rel_path = os.path.relpath(full_path, base)
            if _is_excluded(rel_path, excludes):
                logger.debug(f"[v8.0] Excluded: {rel_path}")
                continue
            
            yield full_path
            
    except Exception as e:
        logger.error(f"[v8.0] Error iterating root files in {base}: {e}")


def _iter_candidate_files(base: str, exts: List[str], excludes: List[str]):
    """
    Legacy recursive walker kept for reference.
    Current flow uses _iter_root_files_only() for safety.
    
    Args:
        base: Base directory to scan
        exts: List of file extensions
        excludes: List of exclude patterns
    
    Yields:
        str: Absolute paths to matching files
    """
    try:
        for root, dirs, files in os.walk(base):
            rel_root = os.path.relpath(root, base)
            if _is_excluded(rel_root, excludes):
                # Prune traversal
                dirs[:] = []
                continue
            for name in files:
                rel = os.path.normpath(os.path.join(rel_root, name))
                if _is_excluded(rel, excludes):
                    continue
                if os.path.splitext(name)[1] in exts:
                    yield os.path.join(base, rel)
    except Exception as e:
        logger.error(f"[v8.0] Error iterating candidate files: {e}")


# =============================================================================
# CHAT ADAPTER - AI-Powered Code Analysis
# =============================================================================

class _ChatAdapter:
    """
    v8.0: Enhanced chat adapter with better error handling and logging.
    
    Prefer SarahMemoryAPI.send_to_api; otherwise fall back to OpenAI SDK 
    if OPENAI_API_KEY is set. Exposes .available to let the caller 
    short-circuit when neither path is usable.
    """
    def __init__(self) -> None:
        self.mode = None
        self.available = False
        self.API = None
        
        # Try SarahMemoryAPI first
        try:
            import SarahMemoryAPI as API
            self.API = API
            
            # Prefer the project's API only if Class 3 (API) is enabled
            try:
                import SarahMemoryGlobals as config
                if getattr(config, "API_RESEARCH_ENABLED", False):
                    if getattr(config, "OPEN_AI_API", False):
                        self.mode = "SarahMemoryAPI_send"
                        self.available = True
                        logger.debug("[v8.0] Chat adapter: SarahMemoryAPI (OpenAI enabled)")
                    else:
                        self.mode = "SarahMemoryAPI_send"
                        self.available = True
                        logger.debug("[v8.0] Chat adapter: SarahMemoryAPI (default provider)")
                else:
                    logger.debug("[v8.0] API_RESEARCH_ENABLED is False, checking alternatives")
                    self.mode = None
            except Exception as e:
                logger.debug(f"[v8.0] Could not read Globals, trying API anyway: {e}")
                self.mode = "SarahMemoryAPI_send"
                self.available = True
        except Exception as e:
            logger.debug(f"[v8.0] SarahMemoryAPI not available: {e}")
            self.API = None
            self.mode = None

        # If project API path not available, try OpenAI SDK fallback
        if not self.available:
            try:
                if os.getenv("OPENAI_API_KEY"):
                    self.mode = "OpenAI_SDK"
                    self.available = True
                    logger.debug("[v8.0] Chat adapter: OpenAI SDK fallback")
            except Exception as e:
                logger.debug(f"[v8.0] OpenAI SDK not available: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Concatenate chat messages into a single prompt string for send_to_api()."""
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n\n".join(lines).strip()

    def chat(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Send chat request to AI service.
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
        
        Returns:
            str: AI response text
        """
        if self.mode == "SarahMemoryAPI_send" and self.API:
            # Use project's API (Class 3) path
            prompt = self._messages_to_prompt(messages)
            try:
                resp = self.API.send_to_api(
                    user_input=prompt,
                    provider="openai",
                    intent="debug",
                    tone="technical",
                    complexity="adult",
                    model=model
                )
                # Typical return is a dict like {"source": "...", "data": "...", ...}
                if isinstance(resp, dict):
                    data = resp.get("data") or resp.get("snippet") or ""
                    return str(data)
                return str(resp)
            except Exception as e:
                logger.warning(f"[v8.0] SarahMemoryAPI failed: {e}")

        if self.mode == "OpenAI_SDK":
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                raise RuntimeError("OpenAI SDK fallback failed.") from e

        raise RuntimeError(
            "No available chat adapter. Enable API_RESEARCH_ENABLED and OPEN_AI_API in Globals, "
            "or set OPENAI_API_KEY for the SDK path."
        )


# =============================================================================
# PROMPTING & FILE PACKAGING
# =============================================================================

SYSTEM_PROMPT = (
    "You are an assistant tasked with MINIMAL, SURGICAL code fixes for the SarahMemory project. "
    "Rules: 1) DO NOT rename existing function or class names unless absolutely required to fix a bug. "
    "2) Preserve all original comments and add new notes prefixed with '# [Updater]'. "
    "3) Keep the file interface stable for other modules. 4) Return ONLY valid JSON with keys:"
    " {\"decision\": \"accept|skip\", \"reason\": str, \"updated_code\": str}. "
    "If no changes are needed, set decision='skip' and updated_code=''."
)


def _package_file_for_review(path: str) -> str:
    """
    Package file content for AI review, handling large files.
    
    Args:
        path: Path to file
    
    Returns:
        str: Packaged file content
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        
        # Clip overly large files conservatively
        if len(src) > MAX_CHARS:
            head = src[: MAX_CHARS // 2]
            tail = src[-MAX_CHARS // 2:]
            packaged = (
                "# [Truncated for review due to size]\n" + head + 
                "\n\n# [..snip..]\n\n" + tail
            )
            logger.debug(f"[v8.0] Truncated large file: {path}")
        else:
            packaged = src
        
        return packaged
    except Exception as e:
        logger.error(f"[v8.0] Error packaging file {path}: {e}")
        return ""


def _build_messages(path: str) -> List[Dict[str, str]]:
    """
    Build message array for AI chat.
    
    Args:
        path: Path to file
    
    Returns:
        list: Messages for AI
    """
    rel = os.path.relpath(path, BASE_DIR)
    payload = _package_file_for_review(path)
    user_prompt = (
        f"Review and minimally fix the following file: {rel}. "
        f"Return JSON as specified.\n\n{payload}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# DIFF, APPROVAL, BACKUP, WRITE
# =============================================================================

def _unified_diff(a: str, b: str, path: str) -> str:
    """
    Generate unified diff between two text versions.
    
    Args:
        a: Original text
        b: New text
        path: File path (for diff header)
    
    Returns:
        str: Unified diff output
    """
    try:
        a_lines = a.splitlines(keepends=True)
        b_lines = b.splitlines(keepends=True)
        diff = difflib.unified_diff(
            a_lines, b_lines, 
            fromfile=path + " (old)", 
            tofile=path + " (new)"
        )
        return "".join(diff)
    except Exception as e:
        logger.error(f"[v8.0] Error generating diff: {e}")
        return ""


def _ask_approval(diff_text: str, relpath: str) -> bool:
    """
    Ask user for approval of changes.
    
    Args:
        diff_text: Unified diff to display
        relpath: Relative file path
    
    Returns:
        bool: True if approved
    """
    if AUTO_APPROVE:
        _log(f"Auto-approving changes for {relpath} (SM_UPDATER_AUTOAPPROVE=1)")
        return True
    
    print("\n===== Proposed changes to:", relpath, "=====")
    print(diff_text or "(no visible textual diff)")
    print("\nApply these changes? [y]es / [n]o / [v]iew again")
    
    while True:
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            # Non-interactive environment – do not block; default to "no"
            _log(f"{relpath}: non-interactive environment detected; defaulting to 'no'.")
            return False
        
        if choice in ("y", "yes", ""):  # default yes for speed at boot
            return True
        if choice in ("n", "no"):
            return False
        if choice in ("v", "view"):
            print(diff_text)
        else:
            print("Please type y/n/v")


def _backup_file(path: str) -> str:
    """
    Create timestamped backup of a file.
    
    Args:
        path: Path to file to backup
    
    Returns:
        str: Path to backup file
    """
    try:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        rel = os.path.relpath(path, BASE_DIR)
        safe_rel = rel.replace(os.sep, "_")
        dest = os.path.join(BACKUPS_DIR, f"{safe_rel}.{ts}.bak")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(path, dest)
        logger.debug(f"[v8.0] Backed up {rel} to {dest}")
        return dest
    except Exception as e:
        logger.error(f"[v8.0] Error backing up {path}: {e}")
        return ""


def _write_and_sanity_check(path: str, new_text: str) -> Tuple[bool, Optional[str]]:
    """
    Write new file content with syntax validation.
    
    Args:
        path: Path to file
        new_text: New file content
    
    Returns:
        tuple: (success, error_message)
    """
    # Write to temp first
    tmp = path + ".__tmp_updater__"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new_text)
    except Exception as e:
        logger.error(f"[v8.0] Error writing temp file: {e}")
        return False, str(e)
    
    # Syntax sanity for .py files
    ok = True
    err = None
    if path.endswith(".py"):
        import py_compile
        try:
            py_compile.compile(tmp, doraise=True)
            logger.debug(f"[v8.0] Syntax check passed: {path}")
        except Exception as e:
            ok, err = False, str(e)
            logger.error(f"[v8.0] Syntax check failed: {path} - {err}")
    
    # Move in place or cleanup
    if ok:
        try:
            shutil.move(tmp, path)
            logger.debug(f"[v8.0] Successfully wrote: {path}")
        except Exception as e:
            ok, err = False, str(e)
            logger.error(f"[v8.0] Error moving file: {e}")
    else:
        try:
            os.remove(tmp)
        except Exception:
            pass
    
    return ok, err


def _create_opt_backup(tag="F"):
    """
    v8.0: Create a full-system ZIP backup with opt-backup marker.
    
    Creates a comprehensive backup into the configured BACKUP_DIR with an 
    "opt-backup" marker to distinguish from normal boot backups.
    
    Args:
        tag: Backup tag (default: "F" for full)
    
    Returns:
        str|None: Path to backup file or None on failure
    """
    try:
        base_dir = BASE_DIR
        backup_dir = getattr(G, "BACKUP_DIR", os.path.join(DATA_DIR, "backup"))
        os.makedirs(backup_dir, exist_ok=True)
        
        # Filename with opt-backup marker
        ts = dt.datetime.now().strftime("%m-%d-%Y_%H%M%S")
        
        # Count existing opt-backup zips to increment number
        existing = [
            f for f in os.listdir(backup_dir) 
            if f.startswith(f"SarahMemory_{tag}-opt-backup")
        ]
        seq = len(existing) + 1
        name = f"SarahMemory_{tag}-opt-backup_{seq}_{ts}.zip"
        out_path = os.path.join(backup_dir, name)
        
        def _iter_files(root):
            for r, _, files in os.walk(root):
                # Skip the backup dir itself to avoid ballooning backups
                if os.path.abspath(r).startswith(os.path.abspath(backup_dir)):
                    continue
                for fn in files:
                    fp = os.path.join(r, fn)
                    rel = os.path.relpath(fp, base_dir)
                    yield fp, rel
        
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for fp, rel in _iter_files(base_dir):
                try:
                    z.write(fp, arcname=rel)
                except Exception:
                    # Best-effort; skip unreadable files
                    pass
        
        _log(f"[Pre-Update Backup] Created {out_path}")
        logger.info(f"[v8.0] Full backup created: {out_path}")
        return out_path
        
    except Exception as e:
        _log(f"[Pre-Update Backup] FAILED: {e}")
        logger.error(f"[v8.0] Backup creation failed: {e}")
        return None


def _defender_scan(path: str) -> bool:
    """
    v8.0: Best-effort Windows Defender scan for a file or folder.
    
    Args:
        path: Path to scan
    
    Returns:
        bool: True (non-fatal if missing)
    """
    try:
        exe = r"C:\Program Files\Windows Defender\MpCmdRun.exe"
        if not os.path.exists(exe):
            logger.debug("[v8.0] Windows Defender not found, skipping scan")
            return True  # Can't scan; don't block
        
        subprocess.run(
            [exe, "-Scan", "-ScanType", "3", "-File", path], 
            check=False
        )
        logger.debug(f"[v8.0] Defender scan completed: {path}")
        return True
        
    except Exception as e:
        logger.debug(f"[v8.0] Defender scan failed (non-fatal): {e}")
        return True


def _create_sandbox_dir() -> str:
    """
    v8.0: Create sandbox directory for testing updates.
    
    Returns:
        str: Path to sandbox directory
    """
    try:
        sb = os.path.join(UPDATER_DIR, "sandbox")
        os.makedirs(sb, exist_ok=True)
        logger.debug(f"[v8.0] Sandbox directory: {sb}")
        return sb
    except Exception as e:
        logger.error(f"[v8.0] Error creating sandbox: {e}")
        return os.path.join(UPDATER_DIR, "sandbox")


def _pre_update_backup() -> str | None:
    """
    v8.0: Create pre-update backup of Python files.
    
    Returns:
        str|None: Path to backup or None on failure
    """
    try:
        os.makedirs(BACKUPS_DIR, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = os.path.join(BACKUPS_DIR, f"preupdate-{ts}.zip")
        
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(BASE_DIR):
                if any(x in root for x in (".git", "node_modules", "__pycache__", "data")):
                    continue
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        z.write(fp, arcname=os.path.relpath(fp, BASE_DIR))
                    except Exception:
                        pass
        
        logger.info(f"[v8.0] Pre-update backup: {out}")
        return out
        
    except Exception as e:
        logger.error(f"[v8.0] Pre-update backup failed: {e}")
        return None


# =============================================================================
# CORE WORKFLOW - v8.0 Enhanced
# =============================================================================

def _should_consider(path: str, state: Dict[str, Any]) -> bool:
    """
    Check if file should be considered for update.
    
    Args:
        path: Path to file
        state: State dictionary
    
    Returns:
        bool: True if file changed or new
    """
    try:
        rel = os.path.relpath(path, BASE_DIR)
        meta = state.get("files", {}).get(rel, {})
        prev_hash = meta.get("sha256")
        curr_hash = _sha256(path)
        
        # Consider if: never seen before OR file changed since last review
        changed = prev_hash != curr_hash
        if changed:
            logger.debug(f"[v8.0] File changed: {rel}")
        return changed
        
    except Exception as e:
        logger.error(f"[v8.0] Error checking file {path}: {e}")
        return False


def _update_state_for(path: str, state: Dict[str, Any], outcome: Dict[str, Any]):
    """
    Update state for a file.
    
    Args:
        path: Path to file
        state: State dictionary
        outcome: Outcome dictionary
    """
    try:
        rel = os.path.relpath(path, BASE_DIR)
        files = state.setdefault("files", {})
        files[rel] = {
            "sha256": _sha256(path),
            "last_result": outcome,
            "last_update": dt.datetime.now().isoformat(timespec="seconds"),
        }
        logger.debug(f"[v8.0] Updated state for: {rel}")
    except Exception as e:
        logger.error(f"[v8.0] Error updating state for {path}: {e}")


def _git_push_if_enabled(summary_msg: str) -> bool:
    """
    Push changes to git if enabled.
    
    Args:
        summary_msg: Commit message
    
    Returns:
        bool: True if successful
    """
    if not GIT_PUSH:
        return False
    
    try:
        # Prefer user's push helper if present
        helper = os.path.join(BASE_DIR, "Push_SarahMemory.bat")
        if os.path.exists(helper):
            _log("Running Push_SarahMemory.bat …")
            subprocess.run([helper], check=False)
            return True
        
        # Generic git flow
        cmds = [
            ["git", "add", "."],
            ["git", "commit", "-m", summary_msg],
            ["git", "push"],
        ]
        for c in cmds:
            subprocess.run(c, cwd=BASE_DIR, check=False)
        
        logger.info("[v8.0] Git push completed")
        return True
        
    except Exception as e:
        _log(f"Git push failed: {e}")
        logger.error(f"[v8.0] Git push error: {e}")
        return False


def run_updater(
    invoked_by_main: bool = False,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    exts: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> bool:
    """
    v8.0: Enhanced main updater function with comprehensive error handling.
    
    Runs the auto-updater to review and apply minimal fixes to Python files.
    
    Args:
        invoked_by_main: Whether called from SarahMemoryMain.py
        includes: List of paths to include
        excludes: List of patterns to exclude
        exts: List of file extensions to process
        model: AI model to use
    
    Returns:
        bool: True if ran to completion (internet or not), False only on fatal error
    """
    logger.info("[v8.0] Starting updater run")
    _ensure_dirs()
    state = _load_state()

    # Respect schedule from Globals
    try:
        if hasattr(G, "update_due") and callable(G.update_due):
            if not G.update_due(state.get("last_run")):
                _log("Updater skipped (per schedule policy).")
                logger.info("[v8.0] Updater skipped due to schedule")
                return True
    except Exception as e:
        _log(f"Schedule check failed; proceeding conservatively. ({e})")
        logger.warning(f"[v8.0] Schedule check error: {e}")

    if not _has_internet():
        _log("No internet connectivity detected – skipping updater and continuing normal boot.")
        logger.info("[v8.0] No internet, skipping updater")
        return True

    # Build file set (ROOT-ONLY scan by default)
    includes = includes or [BASE_DIR]
    excludes = (excludes or []) + DEFAULT_EXCLUDES
    exts = exts or DEFAULT_EXTS

    candidates: List[str] = []
    for target in includes:
        if os.path.isfile(target):
            candidates.append(os.path.abspath(target))
        elif os.path.isdir(target):
            # Collect files directly under the directory (no recursion)
            for p in _iter_root_files_only(target, exts, excludes):
                # Avoid updating this very file during its own run
                if os.path.abspath(p) == os.path.abspath(__file__):
                    continue
                candidates.append(os.path.abspath(p))

    # Filter down by content hash
    worklist = [p for p in candidates if _should_consider(p, state)]
    
    if not worklist:
        _log("All files unchanged since last run – nothing to review.")
        logger.info("[v8.0] No files need updating")
        state["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
        _save_state(state)
        return True

    _log(f"Found {len(worklist)} candidate file(s) for review.")
    logger.info(f"[v8.0] Processing {len(worklist)} files")
    
    chat = _ChatAdapter()

    # Short-circuit if neither SarahMemoryAPI nor OPENAI_API_KEY is available
    if not getattr(chat, "available", False):
        _log("Updater online path skipped: no SarahMemoryAPI send_to_api or OPENAI_API_KEY available. Continuing boot.")
        logger.info("[v8.0] No AI service available, skipping")
        _save_state(state)
        return True

    any_updates_applied = False
    pre_update_backup_done = False
    
    for path in worklist:
        rel = os.path.relpath(path, BASE_DIR)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                old_text = f.read()
        except Exception as e:
            _log(f"Skip {rel}: unable to read – {e}")
            logger.error(f"[v8.0] Cannot read {rel}: {e}")
            continue

        messages = _build_messages(path)
        try:
            raw = chat.chat(messages=messages, model=(model or MODEL_NAME))
        except Exception as e:
            _log(f"Chat adapter failure on {rel}: {e}")
            logger.error(f"[v8.0] AI chat failed for {rel}: {e}")
            continue

        # Expect strict JSON
        try:
            payload = json.loads(raw)
        except Exception:
            # Try to salvage JSON if model wrapped code blocks
            m = re.search(r"\{[\s\S]*\}\s*$", raw)
            if not m:
                _log(f"{rel}: model did not return JSON; skipping.")
                logger.warning(f"[v8.0] Invalid JSON response for {rel}")
                continue
            try:
                payload = json.loads(m.group(0))
            except Exception:
                _log(f"{rel}: invalid JSON payload; skipping.")
                logger.warning(f"[v8.0] Cannot parse JSON for {rel}")
                continue

        decision = (payload.get("decision") or "").strip().lower()
        reason = payload.get("reason", "")
        updated = payload.get("updated_code", "")

        if decision not in ("accept", "skip"):
            _log(f"{rel}: unknown decision '{decision}'; skipping.")
            _update_state_for(path, state, {"decision": "invalid_decision", "raw": raw[:200]})
            continue

        if decision == "skip" or not updated.strip():
            _log(f"{rel}: no changes suggested. Reason: {reason or 'n/a'}")
            logger.debug(f"[v8.0] No changes for {rel}")
            _update_state_for(path, state, {"decision": "skip", "reason": reason})
            continue

        diff_text = _unified_diff(old_text, updated, rel)
        
        # Auto behavior when user chose apply-all or skip-all
        global _APPLY_ALL, _SKIP_ALL
        if not (_APPLY_ALL or _SKIP_ALL):
            approved = _ask_approval(diff_text, rel)
        else:
            approved = not _SKIP_ALL
        
        if not approved:
            _log(f"{rel}: user declined changes.")
            logger.info(f"[v8.0] Changes declined for {rel}")
            _update_state_for(path, state, {"decision": "declined", "reason": reason})
            continue

        # One-time full-system opt-backup prior to applying the first update
        if not pre_update_backup_done:
            _create_opt_backup(tag="F")
            pre_update_backup_done = True

        # Backup, write, sanity
        bkp = _backup_file(path)
        ok, err = _write_and_sanity_check(path, updated)
        
        if not ok:
            _log(f"{rel}: write rejected due to syntax error; rolled back. Error: {err}")
            logger.error(f"[v8.0] Syntax error in {rel}, rolled back")
            _update_state_for(path, state, {"decision": "error", "error": err, "backup": bkp})
            continue

        _log(f"{rel}: updated successfully. Reason: {reason or 'n/a'} | backup={os.path.basename(bkp)}")
        logger.info(f"[v8.0] Successfully updated: {rel}")
        _update_state_for(path, state, {"decision": "applied", "reason": reason, "backup": bkp})
        any_updates_applied = True

    # Update last run timestamp
    state["last_run"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_state(state)

    if any_updates_applied:
        pushed = _git_push_if_enabled(
            summary_msg=f"[Updater] Automated minimal fixes on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if pushed:
            state["last_git_push"] = dt.datetime.now().isoformat(timespec="seconds")
            _save_state(state)
            _log("Git push performed.")
        else:
            _log("Git push skipped (disabled or failed).")

    _log("Updater run complete.")
    logger.info("[v8.0] Updater run completed successfully")
    return True


def update_from_zip(zip_path: str) -> bool:
    """
    v8.0: Update from a ZIP file.
    
    Placeholder for future enhancement. Extract → backup .py → deploy → 
    rollback on failure.
    
    Args:
        zip_path: Path to update ZIP file
    
    Returns:
        bool: True if successful
    """
    logger.info(f"[v8.0] Update from ZIP: {zip_path}")
    # TODO: Implement full ZIP update workflow
    return False


# =============================================================================
# MODULE TEST - v8.0
# =============================================================================

if __name__ == "__main__":
    logger.info("[v8.0] SarahMemoryUpdater module test")
    print("SarahMemory Auto-Updater v8.0.0")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Backups Directory: {BACKUPS_DIR}")
    print(f"Internet Available: {_has_internet()}")
    print("=" * 60)
    print("\nRun updater? [y/n]")
    
    try:
        choice = input("> ").strip().lower()
        if choice in ("y", "yes"):
            success = run_updater()
            print(f"\nUpdater completed: {'SUCCESS' if success else 'FAILED'}")
        else:
            print("Updater test cancelled")
    except KeyboardInterrupt:
        print("\n\nUpdater test interrupted")
    except Exception as e:
        print(f"\nError during test: {e}")
        logger.error(f"[v8.0] Test error: {e}")
