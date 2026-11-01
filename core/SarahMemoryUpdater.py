"""--==The SarahMemory Project==--
File: SarahMemoryUpdater.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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

AUTO-UPDATER SYSTEM v9.0.0
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

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "update_engine"
# CATEGORY = "self_update_and_patch_feeds"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "update_management"
# HARDWARE_DOMAIN = "filesystem_network_security"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "updater"
# FAMILY = "self_evolution"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Governed auto-updater with backup, hashing, syntax validation, patch-feed sync, license-gated corporate feeds, rollback, and safe multi-mode update handling."
# --- SARAHMETA END ---

import os
import sys
import re
import json
import shutil
import socket
import hashlib
import base64
import difflib
import subprocess
import datetime as dt
import logging
from logging.handlers import RotatingFileHandler
import zipfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings

# ARILE protected cleanup/update guard. Protected immune-system files are never
# cleanup/update targets from automated maintenance lanes.
try:
    from SarahMemoryARILE import arile_is_protected_core_file, arile_emit
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    def arile_is_protected_core_file(path_value):
        try:
            return Path(path_value).name.lower() in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}
        except Exception:
            return False

def _arile_block_if_protected_target(path_value: str, operation: str = "maintenance") -> None:
    if arile_is_protected_core_file(path_value):
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, kind="protected_core_variance", failure_type="maintenance_target_blocked", severity=0.92, confidence=0.90, risk="critical", summary=f"Blocked {operation} against protected core file.", requires_governance=True, retention="security_audit", data={"target": str(path_value)})
        except Exception:
            pass
        raise PermissionError(f"Protected core maintenance target blocked: {path_value}")

warnings.filterwarnings("error", category=SyntaxWarning)

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryUpdater')
logger.setLevel(logging.DEBUG if os.environ.get('SM_UPDATER_DEBUG','0') in ('1','true','True','yes','on') else logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[v8.0][UPDATER] %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler
try:
    log_dir = os.path.join(os.getcwd(), "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f"updater_{dt.datetime.now().strftime('%Y%m%d')}.log"),
        maxBytes=int(os.environ.get("SM_UPDATER_LOG_MAX_BYTES", "1048576") or 1048576),
        backupCount=int(os.environ.get("SM_UPDATER_LOG_BACKUPS", "3") or 3),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG if os.environ.get('SM_UPDATER_DEBUG','0') in ('1','true','True','yes','on') else logging.INFO)
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
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", ".pytest_cache",
    "node_modules", "dist", "build", ".next", ".vite", "coverage", "logs", "log",
    "cache", "tmp", "temp", "runtime", "backup", "backups", "archive", "archives",
    os.path.relpath(UPDATER_DIR, BASE_DIR),
    "data/models", "data/memory", "data/datasets", "data/cache", "data/tmp", "data/runtime",
    "*.db", "*.db-wal", "*.db-shm", "*.sqlite", "*.zip", "*.tar", "*.tar.gz", "*.png", "*.jpg", "*.jpeg", "*.gif",
    "*.mp4", "*.mp3", "*.wav", "*.onnx", "*.pt", "*.pth", "*.safetensors", "*.gguf", "*.bin", "*.log"
]

_UPDATER_SKIP_DIR_PARTS = {str(x).lower() for x in DEFAULT_EXCLUDES if not str(x).startswith("*") and "." not in str(x).replace(".venv", "")}
_UPDATER_BACKUP_MAX_FILE_MB = float(os.environ.get("SM_UPDATER_BACKUP_MAX_FILE_MB", "16") or 16)
_UPDATER_OPT_BACKUP_ENABLED = os.environ.get("SM_UPDATER_OPT_BACKUP_ENABLED", "0") in ("1", "true", "True", "yes", "on")


def _updater_should_skip_path(path: str) -> bool:
    try:
        rel = os.path.relpath(os.path.abspath(path), BASE_DIR).replace("\\", "/").lower()
        parts = set(rel.split("/"))
        if parts & {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "dist", "build", ".next", ".vite", "logs", "cache", "tmp", "temp", "runtime", "backup", "backups", "archive", "archives", "models"}:
            return True
        if any(rel.startswith(str(x).replace("\\", "/").lower().rstrip("/") + "/") for x in ("data/models", "data/memory", "data/datasets", "data/cache", "data/tmp", "data/runtime")):
            return True
        return False
    except Exception:
        return True

DEFAULT_EXTS = [".py"]  # Start conservative; can extend via CLI flags or config

# User-tunable environment flags
AUTO_APPROVE = os.environ.get("SM_UPDATER_AUTOAPPROVE", "0") in ("1", "true", "True")
GIT_PUSH = os.environ.get("SM_UPDATER_GIT_PUSH", "0") in ("1", "true", "True")
MODEL_NAME = os.environ.get("SM_UPDATER_MODEL", getattr(G, "DEFAULT_GPT_MODEL", "gpt-4.1-mini"))
MAX_CHARS = int(os.environ.get("SM_UPDATER_MAX_CHARS", "20000"))  # per request; conservative

# =============================================================================
# MODS / PATCH FEEDS + CORPORATE LICENSE GATING (v8.0)
#
# Goal:
# - Community users: pull patch manifests/files from GitHub (public)
# - Corporate users (valid SarahMemory.lic): pull patch manifests/files from GoogieHost
#   (served under /public_html/api/data/mods)
#
# Notes:
# - This does NOT lock the base platform. It only controls access to CORPORATE update feeds.
# - License verification is designed to be future-proof:
#     1) Prefer Ed25519 signature verification (if crypto libs present + public key configured)
#     2) Optional HMAC verification (if a shared secret is configured)
#     3) If verification cannot be performed, license is treated as INVALID (safe default)
# =============================================================================

LICENSE_DIR = os.path.join(DATA_DIR, "license")
LICENSE_FILE = os.path.join(LICENSE_DIR, "SarahMemory.lic")

MODS_ROOT = os.environ.get("SM_MODS_ROOT", os.path.join(DATA_DIR, "mods"))
MODS_ACTIVE_VERSION = os.environ.get("SM_MODS_ACTIVE_VERSION", "v800")
MODS_ENABLED = os.environ.get("SM_MODS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

# Community patch feed (GitHub raw)
SM_GITHUB_PATCH_REPO = os.environ.get("SM_GITHUB_PATCH_REPO", "")
SM_GITHUB_PATCH_BRANCH = os.environ.get("SM_GITHUB_PATCH_BRANCH", "main")
SM_GITHUB_PATCH_MANIFEST = os.environ.get("SM_GITHUB_PATCH_MANIFEST", "patches/manifest.json")
SM_GITHUB_READ_TOKEN = os.environ.get("SM_GITHUB_READ_TOKEN", "").strip()

# Corporate patch feed (GoogieHost / public_html/api/data/mods)
# Default base URL derives from PUBLIC_API_BASE, falling back to api.sarahmemory.com if unset.
SM_CORP_PATCH_BASE_URL = os.environ.get("SM_CORP_PATCH_BASE_URL", "").strip()
SM_CORP_PATCH_MANIFEST = os.environ.get("SM_CORP_PATCH_MANIFEST", "manifest.json").strip()

# License verification config
# Provide ONE of:
# - SM_LICENSE_PUBLIC_KEY_B64: Ed25519 public key (base64) for signature verification
# - SM_LICENSE_HMAC_SECRET: shared secret for HMAC verification (not recommended for broad distribution)
SM_LICENSE_PUBLIC_KEY_B64 = os.environ.get("SM_LICENSE_PUBLIC_KEY_B64", "").strip()
SM_LICENSE_HMAC_SECRET = os.environ.get("SM_LICENSE_HMAC_SECRET", "").strip()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def _parse_iso8601(s: str) -> Optional[dt.datetime]:
    try:
        # Accept "Z" suffix
        s2 = s.strip().replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s2)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _safe_json_load(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[v8.0] License/manifest JSON read failed: {e}")
        return None


def _license_payload_bytes(lic: dict) -> bytes:
    """
    Canonicalize license payload for signing:
    - copy license dict
    - remove signature fields
    - JSON-dump with stable sorting and separators
    """
    payload = dict(lic or {})
    payload.pop("signature", None)
    payload.pop("sig", None)
    payload.pop("signature_b64", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_license_signature_ed25519(payload: bytes, signature_b64: str) -> bool:
    """
    Verify Ed25519 signature. Requires either:
    - 'cryptography' library, or
    - 'nacl' (PyNaCl)
    Safe default: return False on any error.
    """
    if not SM_LICENSE_PUBLIC_KEY_B64:
        return False
    try:
        pub = base64.b64decode(SM_LICENSE_PUBLIC_KEY_B64)
        sig = base64.b64decode(signature_b64)
    except Exception:
        return False

    # Try cryptography first
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore
        pk = Ed25519PublicKey.from_public_bytes(pub)
        pk.verify(sig, payload)
        return True
    except Exception:
        pass

    # Try PyNaCl
    try:
        import nacl.signing  # type: ignore
        vk = nacl.signing.VerifyKey(pub)
        vk.verify(payload, sig)
        return True
    except Exception:
        return False


def _verify_license_signature_hmac(payload: bytes, signature_b64: str) -> bool:
    """
    Optional HMAC verification: signature_b64 is expected to be base64(HMAC_SHA256(payload)).
    Safe default: False if secret not configured.
    """
    if not SM_LICENSE_HMAC_SECRET:
        return False
    try:
        import hmac
        sig = base64.b64decode(signature_b64)
        mac = hmac.new(SM_LICENSE_HMAC_SECRET.encode("utf-8"), payload, digestmod=hashlib.sha256).digest()
        return hmac.compare_digest(sig, mac)
    except Exception:
        return False


def _load_and_verify_license() -> Optional[dict]:
    """
    Returns parsed license dict if valid, else None.
    Valid means:
    - file exists
    - tier indicates corporate/enterprise
    - not expired
    - signature validates (ed25519 preferred, hmac optional)
    """
    try:
        if not os.path.isdir(LICENSE_DIR) or not os.path.isfile(LICENSE_FILE):
            return None

        lic = _safe_json_load(LICENSE_FILE)
        if not lic:
            return None

        tier = str(lic.get("tier") or lic.get("license_tier") or "").strip().lower()
        if tier not in ("corporate", "enterprise"):
            logger.info("[v8.0] License present but tier not corporate/enterprise; treating as community.")
            return None

        exp = lic.get("expires_at") or lic.get("expires") or lic.get("expiry")
        if not exp:
            logger.warning("[v8.0] License missing expires_at; treating as invalid.")
            return None

        exp_dt = _parse_iso8601(str(exp))
        if not exp_dt:
            logger.warning("[v8.0] License expiry parse failed; treating as invalid.")
            return None

        if _utc_now() >= exp_dt:
            logger.warning("[v8.0] License expired; corporate feed disabled.")
            return None

        sig = lic.get("signature") or lic.get("sig") or lic.get("signature_b64")
        if not sig:
            logger.warning("[v8.0] License missing signature; treating as invalid.")
            return None

        payload = _license_payload_bytes(lic)

        ok = False
        # Prefer ed25519 verification; fallback to HMAC if configured
        if _verify_license_signature_ed25519(payload, str(sig)):
            ok = True
        elif _verify_license_signature_hmac(payload, str(sig)):
            ok = True

        if not ok:
            logger.warning("[v8.0] License signature invalid or verification unavailable; corporate feed disabled.")
            return None

        logger.info("[v8.0] Corporate license verified successfully.")
        return lic
    except Exception as e:
        logger.warning(f"[v8.0] License verification error: {e}")
        return None


def _http_get(url: str, headers: Optional[dict] = None, timeout_sec: int = 15) -> Optional[bytes]:
    """
    Minimal HTTP GET with urllib (no external deps). Returns bytes or None.
    """
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.read()
    except Exception as e:
        logger.debug(f"[v8.0] HTTP GET failed for {url}: {e}")
        return None


def _write_file_atomic(path: str, data: bytes) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.warning(f"[v8.0] Write failed for {path}: {e}")
        return False


def _mods_manifest_url_corporate() -> str:
    base = SM_CORP_PATCH_BASE_URL
    if not base:
        base = os.environ.get("PUBLIC_API_BASE", "").strip() or "https://api.sarahmemory.com"
    # Corporate manifests hosted under /data/mods/<manifest>
    base = base.rstrip("/")
    return f"{base}/data/mods/{SM_CORP_PATCH_MANIFEST.lstrip('/')}"


def _mods_manifest_url_github() -> Optional[str]:
    if not SM_GITHUB_PATCH_REPO:
        return None
    repo = SM_GITHUB_PATCH_REPO.strip()
    branch = (SM_GITHUB_PATCH_BRANCH or "main").strip()
    manifest = (SM_GITHUB_PATCH_MANIFEST or "patches/manifest.json").lstrip("/")
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{manifest}"


def _mods_file_url_corporate(rel_path: str) -> str:
    base = SM_CORP_PATCH_BASE_URL
    if not base:
        base = os.environ.get("PUBLIC_API_BASE", "").strip() or "https://api.sarahmemory.com"
    base = base.rstrip("/")
    rel = rel_path.lstrip("/")
    return f"{base}/data/mods/{rel}"


def _mods_file_url_github(rel_path: str) -> Optional[str]:
    if not SM_GITHUB_PATCH_REPO:
        return None
    repo = SM_GITHUB_PATCH_REPO.strip()
    branch = (SM_GITHUB_PATCH_BRANCH or "main").strip()
    rel = rel_path.lstrip("/")
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{rel}"


def _parse_manifest_bytes(b: bytes) -> Optional[dict]:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception as e:
        logger.warning(f"[v8.0] Manifest JSON parse failed: {e}")
        return None


def _manifest_iter_files(manifest: dict) -> List[dict]:
    """
    Corporate/community manifest format (future-proof):
    {
      "channel": "community|corporate|enterprise",
      "generated_at": "2025-12-21T00:00:00Z",
      "min_app_version": "9.0.0",
      "files": [
        {"path": "v800/patches/SM_v800_patch_20251221_214953_rankings_fix.py",
         "sha256": "<hex>",
         "size": 1234,
         "optional": false}
      ]
    }
    """
    files = manifest.get("files") or []
    out = []
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict) and item.get("path"):
                out.append(item)
    return out


def _sync_mods_from_manifest(manifest: dict, source: str) -> Tuple[int, int, List[str]]:
    """Patch feeds are inspect-only until signed manifest and promotion verification."""
    files = _manifest_iter_files(manifest) if isinstance(manifest, dict) else []
    return 0, len(files), ['feed_staging_required: no authenticated promotion receipt']


def sync_patch_feeds() -> Tuple[bool, List[str]]:
    """
    Sync mods from the appropriate feed:
    - If valid corporate license is present: pull from corporate (GoogieHost) feed
    - Else: pull from GitHub feed (community)
    Returns (ok, notes)
    """
    notes: List[str] = []
    if not MODS_ENABLED:
        return True, ["mods disabled via SM_MODS_ENABLED"]

    os.makedirs(MODS_ROOT, exist_ok=True)

    lic = _load_and_verify_license()
    if lic:
        url = _mods_manifest_url_corporate()
        b = _http_get(url)
        if not b:
            return False, [f"corporate manifest download failed: {url}"]
        manifest = _parse_manifest_bytes(b)
        if not manifest:
            return False, ["corporate manifest invalid JSON"]
        d, s, n = _sync_mods_from_manifest(manifest, source="corporate")
        notes.append(f"corporate sync: downloaded={d}, skipped={s}")
        notes.extend(n[:20])
        return True, notes

    # Community feed fallback (GitHub)
    url = _mods_manifest_url_github()
    if not url:
        return False, ["github patch repo not configured (SM_GITHUB_PATCH_REPO missing)"]
    headers = {}
    if SM_GITHUB_READ_TOKEN:
        headers["Authorization"] = f"token {SM_GITHUB_READ_TOKEN}"
    b = _http_get(url, headers=headers)
    if not b:
        return False, [f"github manifest download failed: {url}"]
    manifest = _parse_manifest_bytes(b)
    if not manifest:
        return False, ["github manifest invalid JSON"]
    d, s, n = _sync_mods_from_manifest(manifest, source="github")
    notes.append(f"github sync: downloaded={d}, skipped={s}")
    notes.extend(n[:20])
    return True, notes


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
    """Source mutation belongs to governed promotion, not the review/updater organ."""
    try:
        compile(str(new_text), str(path), 'exec')
    except Exception as exc:
        return False, 'candidate_syntax_invalid:'+type(exc).__name__
    return False, 'governed_promotion_receipt_required'


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
        
        if not _UPDATER_OPT_BACKUP_ENABLED:
            _log("[Pre-Update Backup] Full opt-backup skipped by optimized runtime policy.")
            return None

        def _iter_files(root):
            for r, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if not _updater_should_skip_path(os.path.join(r, d))]
                if os.path.abspath(r).startswith(os.path.abspath(backup_dir)) or _updater_should_skip_path(r):
                    continue
                for fn in files:
                    fp = os.path.join(r, fn)
                    if _updater_should_skip_path(fp):
                        continue
                    try:
                        if os.path.getsize(fp) > int(_UPDATER_BACKUP_MAX_FILE_MB * 1024 * 1024):
                            continue
                    except Exception:
                        continue
                    rel = os.path.relpath(fp, base_dir)
                    yield fp, rel
        
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
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
        
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for root, dirs, files in os.walk(BASE_DIR):
                dirs[:] = [d for d in dirs if not _updater_should_skip_path(os.path.join(root, d))]
                if _updater_should_skip_path(root):
                    continue
                for fn in files:
                    fp = os.path.join(root, fn)
                    if _updater_should_skip_path(fp):
                        continue
                    try:
                        if os.path.getsize(fp) > int(_UPDATER_BACKUP_MAX_FILE_MB * 1024 * 1024):
                            continue
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
    """Never push production code from an AI-generated review result."""
    return False


def run_updater(invoked_by_main: bool = False, includes: Optional[List[str]] = None, excludes: Optional[List[str]] = None, exts: Optional[List[str]] = None, model: Optional[str] = None) -> bool:
    """Legacy auto-update is unavailable without a verified governed review/promotion flow.

    This function does not read source to cloud, change CORE, push Git, or claim a
    successful update. The existing patch-feed preview helpers remain separate.
    """
    logger.warning('Updater held in staging mode: verified execution receipt required')
    return False


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
    print("SarahMemory Auto-Updater v9.0.0")
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

# ====================================================================
# END OF SarahMemoryUpdater.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryUpdater',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Infrastructure',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['infrastructure'],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 55,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryUpdater.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return __import__("copy").deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Unknown",
        "availability": None,
        "integrity": None,
        "performance": None,
        "reliability": None,
        "confidence": 0.0,
        "latency_ms": None,
        "stability": None,
        "compatibility": None,
        "notes": ["SML registration confirmed; runtime measurements not performed"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryUpdater',
        "sml_adapter": True,
        "metadata": __import__("copy").deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryUpdater', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

