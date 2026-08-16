"""--==The SarahMemory Project==--
File: SarahWalletVisualizer.py
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
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "D"
# ROLE = "utility_tool"
# CATEGORY = "wallet_visualization"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "standalone_tool"
# API_DOMAIN = "ledger"
# HARDWARE_DOMAIN = "display_filesystem_network"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "wallet_visualizer"
# FAMILY = "utilities"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Wallet balance visualization utility using Ledger API or legacy wallet data to plot transaction-derived balance history."
# --- SARAHMETA END ---

# Visualizes wallet balance over time.
# - Primary source: SarahMemory Ledger API  (env: LEDGER_API_BASE, SARAH_NODE_ID)
# - Fallback: legacy JSON wallet at data/crypto/wallet.srh
import os
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

LEGACY_WALLET_PATH = os.path.join("data", "crypto", "wallet.srh")

LEDGER_API_BASE = os.environ.get("LEDGER_API_BASE", "").rstrip("/")
SARAH_NODE_ID    = os.environ.get("SARAH_NODE_ID", "local-node")

def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.utcnow()

def _fetch_ledger_wallet() -> dict | None:
    if not LEDGER_API_BASE:
        return None
    try:
        url = f"{LEDGER_API_BASE}/api/wallet/{SARAH_NODE_ID}"
        req = Request(url, headers={"User-Agent":"SarahWalletVisualizer/1.0"})
        with urlopen(req, timeout=8) as r:
            if r.status != 200:
                return None
            data = r.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except (URLError, HTTPError, TimeoutError, OSError, ValueError):
        return None

def _load_legacy_wallet() -> dict | None:
    if os.path.exists(LEGACY_WALLET_PATH):
        try:
            with open(LEGACY_WALLET_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _iter_transactions(wallet: dict):
    """
    Yield (timestamp: datetime, delta: float) from various schemas:
      - {'type': 'receive'|'send', 'amount': 1.23, 'timestamp': '...iso...'}
      - {'delta': +1.23/-1.23, 'ts'|'timestamp': '...'}
      - {'amount': +/-1.23, 'ts'|'timestamp': '...'}
    """
    txs = wallet.get("transactions") or wallet.get("tx") or []
    for tx in txs:
        # timestamp
        ts = tx.get("timestamp") or tx.get("ts") or ""
        dt = _parse_iso(ts) if ts else datetime.utcnow()

        # delta
        if "delta" in tx:
            d = float(tx.get("delta", 0.0))
        elif "type" in tx and "amount" in tx:
            amt = float(tx.get("amount", 0.0))
            t = str(tx.get("type", "")).lower()
            if t == "receive" or t == "in" or t == "credit":
                d = amt
            elif t == "send" or t == "out" or t == "debit":
                d = -amt
            else:
                d = amt
        elif "amount" in tx:
            d = float(tx.get("amount", 0.0))
        else:
            d = 0.0
        yield dt, d

def load_wallet_any() -> dict | None:
    # Prefer Ledger API
    w = _fetch_ledger_wallet()
    if w and isinstance(w, dict) and ("transactions" in w or "tx" in w):
        return w
    # Fallback to legacy file
    return _load_legacy_wallet()

def series_from_wallet(wallet: dict):
    # sort by timestamp, compute running balance
    rows = list(_iter_transactions(wallet))
    rows.sort(key=lambda x: x[0])
    times = []
    balances = []
    bal = float(wallet.get("balance") or 0.0)
    # If no explicit starting balance, infer from first-run convention (0 then add deltas)
    if math.isclose(bal, 0.0, abs_tol=1e-9):
        bal = 0.0
    for dt, delta in rows:
        bal += float(delta)
        times.append(dt)
        balances.append(bal)
    return times, balances

def plot_balance(times, balances, title="Wallet Balance Over Time"):
    plt.figure(figsize=(10, 5))
    plt.plot(times, balances, marker='o', linestyle='-')  # no explicit colors
    plt.title(title)
    plt.xlabel("Date/Time")
    plt.ylabel("SRH Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wallet = load_wallet_any()
    if not wallet:
        print("[WalletVisualizer] No wallet data from Ledger API or legacy file.")
    else:
        times, balances = series_from_wallet(wallet)
        if not times:
            print("[WalletVisualizer] Wallet has no transactions to plot.")
        else:
            node = wallet.get("node_id") or os.environ.get("SARAH_NODE_ID", "node")
            title = f"SarahMemory Wallet — {node}"
            plot_balance(times, balances, title=title)

# ====================================================================
# END OF SarahWalletVisualizer.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahWalletVisualizer',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Presentation',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['presentation'],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 45,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahWalletVisualizer.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahWalletVisualizer',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahWalletVisualizer', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

