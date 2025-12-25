"""--==The SarahMemory Project==--
File: SarahWalletVisualizer.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
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
"""
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
# END OF SarahMemoryVisualizer.py v8.0.0
# ====================================================================
