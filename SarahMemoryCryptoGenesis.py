"""--==The SarahMemory Project==--
File: SarahMemoryCryptoGenesis.py
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

"""
# Genesis Block Creation, Wallet System, SRH Token Economy Core, THIS FILE IS NOT TO BE DISTRUBITED TO OTHER DEVICES
# Designed for AI cross communication as a form of exchanging information, from one ai system to another give and take
# experimental concept. Tokens generated may not be used for Monetary Gains and are strickly a form of creating a chain-block and
# reputation for AI systems. ---AUTHOR AND DEVELOPMENT USAGE ONLY---

import hashlib
import json
import os
import time
from datetime import datetime

# --- Optional secure persistence into the SarahMemoryVault ---
try:
    import os
    import SarahMemoryVault as Vault
    _VAULT_PASS = os.environ.get("SARAH_VAULT_PASSWORD", "").strip()
    def _persist_genesis_to_vault(seed: str, address: str) -> None:
        # If passphrase is set, write to passphrase-secure config.
        if _VAULT_PASS:
            Vault.initialize_secure_config(_VAULT_PASS, {"seed": seed, "address": address})
        else:
            # Fall back to random-key vault storage (back-compat)
            Vault.add_item("genesis_seed", seed)
            Vault.add_item("genesis_address", address)
except Exception as _e:
    def _persist_genesis_to_vault(seed: str, address: str) -> None:
        # Vault not available; skip silently.
        pass

WALLET_PATH = os.path.join("wallet", "genesis_wallet.json")
LEDGER_PATH = os.path.join("wallet", "ledger.json")

GENESIS_SUPPLY = 100_000_000  # Total SRH tokens created at genesis
TOKEN_NAME = "SarahCoin"
TOKEN_SYMBOL = "SRH"
DECIMALS = 8  # Can divide into 0.00000001 units

# --- Wallet Generator ---
def create_genesis_wallet():
    if not os.path.exists("wallet"):
        os.makedirs("wallet")

    if os.path.exists(WALLET_PATH):
        return load_wallet()

    seed_phrase = hashlib.sha256(os.urandom(256)).hexdigest()
    public_key = hashlib.sha256(seed_phrase.encode()).hexdigest()

    wallet = {
        "address": public_key[:32],
        "balance": GENESIS_SUPPLY,
        "seed": seed_phrase,
        "timestamp": time.time()
    }

    with open(WALLET_PATH, 'w') as f:
        json.dump(wallet, f, indent=4)

    try:
        _persist_genesis_to_vault(seed_phrase, wallet['address'])
    except Exception:
        pass
    return wallet


def load_wallet():
    with open(WALLET_PATH, 'r') as f:
        return json.load(f)


# --- Transaction Record Keeper ---
def init_ledger():
    wallet_dir = os.path.dirname(LEDGER_PATH)
    if wallet_dir and not os.path.exists(wallet_dir):
        os.makedirs(wallet_dir, exist_ok=True)

    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, 'w') as f:
            json.dump({"transactions": []}, f, indent=4)


def record_transaction(from_addr, to_addr, amount):
    with open(LEDGER_PATH, 'r') as f:
        ledger = json.load(f)

    tx = {
        "from": from_addr,
        "to": to_addr,
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat()
    }

    ledger["transactions"].append(tx)

    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=4)


# --- SRH Token System Manager ---
def get_balance(address):
    wallet = load_wallet()
    if wallet["address"] == address:
        try:
            _persist_genesis_to_vault(wallet.get("seed", ""), wallet['address'])
        except Exception:
            pass
        return wallet["balance"]
    return 0


def transfer_tokens(to_address, amount):
    wallet = load_wallet()
    balance = wallet["balance"]

    if amount > balance:
        return False

    wallet["balance"] -= amount
    record_transaction(wallet["address"], to_address, amount)

    with open(WALLET_PATH, 'w') as f:
        json.dump(wallet, f, indent=4)
    return True


if __name__ == '__main__':
    init_ledger()
    wallet = create_genesis_wallet()
    print("[SRH CRYPTO] Genesis Wallet Initialized:")
    print(f" Address: {wallet['address']}")
    print(f" Balance: {wallet['balance']} {TOKEN_SYMBOL}")