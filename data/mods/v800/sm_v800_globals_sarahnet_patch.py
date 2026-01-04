# --==The SarahMemory Project==--
# File: /home/Softdev0/SarahMemory/data/mods/v800/sm_v800_globals_sarahnet_patch.py
# Patch: v8.0.0 Globals Communication Network Fix
"""
SarahNet Communication Mode Patch
======================================

Author: Brian Lee Baros
Project: SarahMemory v8.0.0 AiOS

Purpose:
- Override SarahMemoryGlobals networking flags at runtime
- WITHOUT modifying SarahMemoryGlobals.py
- Use ../data/settings/sarahnet.comms.json as authority

Communication Selection Array:
OFF | LAN_ONLY | CLOUD_ONLY | CLOUD_LAN
"""

import json
import time
import socket
from pathlib import Path

import SarahMemoryGlobals as G


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(getattr(G, "BASE_DIR", Path.cwd()))
SETTINGS_DIR = BASE_DIR / "data" / "settings"
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
    _apply_identity()
    resolve_sarahnet_comms()


# Auto-execute when imported by mod loader
apply_patch()
