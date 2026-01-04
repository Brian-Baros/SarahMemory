# --==The SarahMemory Project==--
# File: /home/Softdev0/SarahMemory/data/mods/v800/sm_v800_globals_sarahnet_patch.py
# Patch: v8.0.0 Globals Communication Network + Pathing Fix
"""
SarahNet Communication Mode Patch
======================================

Author: Brian Lee Baros
Project: SarahMemory v8.0.0 AiOS

Purpose:
- Override SarahMemoryGlobals networking flags at runtime
- WITHOUT modifying SarahMemoryGlobals.py
- Use ../data/settings/sarahnet.comms.json as authority

NEW (Multi-fix):
- Enforce canonical directory layout:
  BASE_DIR/
    api/        (server code only)
    data/       (ALL wallets, ledgers, logs, settings, mods, memory, etc.)
- Prevent orphaned writes into api/server/data or api/data by forcing Globals dirs.

Communication Selection Array:
OFF | LAN_ONLY | CLOUD_ONLY | CLOUD_LAN
"""

import json
import time
import socket
import sys
from pathlib import Path

import SarahMemoryGlobals as G


# ---------------------------------------------------------------------
# Paths (canonical root enforcement)
# ---------------------------------------------------------------------

def _canonical_base_dir() -> Path:
    """
    Determine BASE_DIR reliably.

    Strongest signal:
      This patch file lives at: <BASE_DIR>/data/mods/v800/<this_file>
      so BASE_DIR is 3 parents up from v800 folder:
        v800 -> mods -> data -> BASE_DIR

    Fallback:
      Use SarahMemoryGlobals.BASE_DIR or CWD.
    """
    try:
        here = Path(__file__).resolve()
        # .../data/mods/v800/this_file.py  -> parents[3] == BASE_DIR
        if len(here.parents) >= 4 and here.parents[2].name == "data":
            return here.parents[3]
        # If layout differs, try walking upward to find a sibling "data" + "api"
        for p in here.parents:
            if (p / "data").is_dir():
                return p
    except Exception:
        pass

    try:
        bd = getattr(G, "BASE_DIR", None)
        if bd:
            return Path(bd).resolve()
    except Exception:
        pass

    return Path.cwd().resolve()


BASE_DIR = _canonical_base_dir()
DATA_DIR = BASE_DIR / "data"

SETTINGS_DIR = DATA_DIR / "settings"
COMMS_FILE = SETTINGS_DIR / "sarahnet.comms.json"
IDENTITY_FILE = SETTINGS_DIR / "sarahnet.identity.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now():
    return int(time.time())


def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cloud_reachable(base: str, health_path: str, timeout_ms: int) -> bool:
    """
    Lightweight connectivity test.
    Avoids requests dependency; socket-level DNS + TCP test only.
    """
    try:
        host = base.replace("https://", "").replace("http://", "").split("/")[0]
        port = 443 if base.startswith("https") else 80
        sock = socket.create_connection((host, port), timeout=timeout_ms / 1000.0)
        sock.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# NEW: Canonical Globals Directory Enforcement
# ---------------------------------------------------------------------

def _set_if_present(name: str, value):
    """Only overwrite existing Globals attributes (no new public config surface)."""
    try:
        if hasattr(G, name):
            setattr(G, name, value)
    except Exception:
        pass


def _apply_canonical_dirs():
    """
    Force all core paths to live under BASE_DIR/data (never under api/*).

    This is intentionally conservative:
    - only overwrites attributes that already exist in SarahMemoryGlobals
    - updates DIR_STRUCTURE map if present
    """
    # Ensure minimal dirs exist (best-effort)
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Root
    _set_if_present("BASE_DIR", str(BASE_DIR))
    _set_if_present("ROOT_DIR", str(BASE_DIR))  # some modules use ROOT_DIR

    # Core dirs
    _set_if_present("DATA_DIR", str(DATA_DIR))
    _set_if_present("SETTINGS_DIR", str(SETTINGS_DIR))

    # Common SarahMemoryGlobals directory names (only if present)
    candidates = {
        "LOGS_DIR": DATA_DIR / "logs",
        "MODS_DIR": DATA_DIR / "mods",
        "THEMES_DIR": (DATA_DIR / "mods" / "themes"),
        "ADDONS_DIR": DATA_DIR / "addons",
        "MEMORY_DIR": DATA_DIR / "memory",
        "DATASETS_DIR": (DATA_DIR / "memory" / "datasets"),
        "IMPORTS_DIR": (DATA_DIR / "memory" / "imports"),
        "SANDBOX_DIR": BASE_DIR / "sandbox",
        "VAULT_DIR": DATA_DIR / "vault",
        "WALLET_DIR": DATA_DIR / "wallet",
        "NETWORK_DIR": DATA_DIR / "network",
        "CRYPTO_DIR": DATA_DIR / "crypto",
        "DIAGNOSTICS_DIR": DATA_DIR / "diagnostics",
        "CLOUD_DIR": DATA_DIR / "cloud",
        "MODELS_DIR": DATA_DIR / "models",
        "AI_DIR": DATA_DIR / "ai",
        "BACKUP_DIR": DATA_DIR / "backup",
        "SYNC_DIR": DATA_DIR / "sync",
    }

    for k, p in candidates.items():
        _set_if_present(k, str(p))

    # Repair DIR_STRUCTURE if it exists (keeps UI/diagnostics consistent)
    try:
        ds = getattr(G, "DIR_STRUCTURE", None)
        if isinstance(ds, dict):
            ds["base"] = str(BASE_DIR)
            ds["data"] = str(DATA_DIR)
            if "logs" in ds and hasattr(G, "LOGS_DIR"):
                ds["logs"] = str(Path(getattr(G, "LOGS_DIR")))
            if "settings" in ds and hasattr(G, "SETTINGS_DIR"):
                ds["settings"] = str(Path(getattr(G, "SETTINGS_DIR")))
            if "mods" in ds and hasattr(G, "MODS_DIR"):
                ds["mods"] = str(Path(getattr(G, "MODS_DIR")))
            if "wallet" in ds and hasattr(G, "WALLET_DIR"):
                ds["wallet"] = str(Path(getattr(G, "WALLET_DIR")))
            setattr(G, "DIR_STRUCTURE", ds)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Core Resolver
# ---------------------------------------------------------------------

def resolve_sarahnet_comms():
    """
    Resolve effective communication mode and override Globals.
    """

    # SAFE MODE ALWAYS WINS
    if getattr(G, "SAFE_MODE", False):
        _apply_off("SAFE_MODE")
        return

    comms = _safe_load_json(COMMS_FILE)

    mode = comms.get("mode", "OFF").upper()

    cloud_cfg = comms.get("cloud", {})
    lan_cfg = comms.get("lan", {})
    fb_cfg = comms.get("fallback", {})

    cloud_enabled = cloud_cfg.get("enabled", True)
    lan_enabled = lan_cfg.get("enabled", True)

    allow_cloud = mode in ("CLOUD_ONLY", "CLOUD_LAN") and cloud_enabled
    allow_lan = mode in ("LAN_ONLY", "CLOUD_LAN") and lan_enabled

    # -------------------------------------------------------------
    # CLOUD FIRST (per your requirement)
    # -------------------------------------------------------------
    if allow_cloud:
        base = cloud_cfg.get("rendezvous_base", getattr(G, "SARAH_WEB_BASE", ""))
        health = cloud_cfg.get("health_path", "/api/health")
        timeout = cloud_cfg.get("timeout_ms", 2500)

        if base and _cloud_reachable(base, health, timeout):
            _apply_cloud(base)
            _write_resolution(comms, "CLOUD", "cloud_reachable")
            return

    # -------------------------------------------------------------
    # FALLBACK TO LAN
    # -------------------------------------------------------------
    if allow_lan and fb_cfg.get("on_no_internet", True):
        _apply_lan()
        _write_resolution(comms, "LAN", "cloud_unreachable")
        return

    # -------------------------------------------------------------
    # FINAL FALLBACK
    # -------------------------------------------------------------
    _apply_off("no_valid_path")
    _write_resolution(comms, "OFF", "fallback_off")


# ---------------------------------------------------------------------
# Apply Modes (ONLY override Globals – no new defs)
# ---------------------------------------------------------------------

def _apply_off(reason: str):
    G.SARAHNET_ENABLED = False
    G.REMOTE_SYNC_ENABLED = False
    G.SARAHNET_CLOUD_ENABLED = False
    G.SARAHNET_LAN_ENABLED = False
    G.SARAHNET_EFFECTIVE_MODE = "OFF"
    G.SARAHNET_REASON = reason


def _apply_lan():
    G.SARAHNET_ENABLED = True
    G.REMOTE_SYNC_ENABLED = False
    G.SARAHNET_CLOUD_ENABLED = False
    G.SARAHNET_LAN_ENABLED = True
    G.SARAHNET_EFFECTIVE_MODE = "LAN"
    G.SARAHNET_REASON = "lan_only"


def _apply_cloud(base: str):
    G.SARAHNET_ENABLED = True
    G.REMOTE_SYNC_ENABLED = True
    G.SARAHNET_CLOUD_ENABLED = True
    G.SARAHNET_LAN_ENABLED = False
    G.SARAHNET_EFFECTIVE_MODE = "CLOUD"
    G.SARAHNET_REASON = "cloud_primary"

    # Reassert hub base in case Globals overwrote it
    if base:
        G.SARAH_WEB_BASE = base


# ---------------------------------------------------------------------
# Identity Reassertion (prevents Globals stomp)
# ---------------------------------------------------------------------

def _apply_identity():
    ident = _safe_load_json(IDENTITY_FILE)
    node_id = ident.get("node_id")
    if node_id:
        G.SARAHNET_NODE_ID = node_id


# ---------------------------------------------------------------------
# Persist Resolution (for UI / diagnostics)
# ---------------------------------------------------------------------

def _write_resolution(comms: dict, effective: str, reason: str):
    try:
        comms.setdefault("last_resolved", {})
        comms["last_resolved"].update({
            "effective_mode": effective,
            "reason": reason,
            "ts": _now()
        })
        COMMS_FILE.write_text(json.dumps(comms, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------
# ENTRY POINT (called by mods loader)
# ---------------------------------------------------------------------

def apply_patch():
    # NEW: enforce canonical directories FIRST (prevents api/server/data drift)
    _apply_canonical_dirs()

    # Existing behavior
    _apply_identity()
    resolve_sarahnet_comms()


# Auto-execute when imported by mod loader
apply_patch()
