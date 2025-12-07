"""--==The SarahMemory Project==--
File: SarahMemoryVault.py
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

import os
import json
import logging
import base64
import secrets
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

import SarahMemoryGlobals as config

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger('SarahMemoryVault')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Back-compat random-key vault (unchanged behavior)
VAULT_FILE = os.path.join(config.VAULT_DIR, 'vault.dat')
VAULT_KEY_FILE = os.path.join(config.VAULT_DIR, 'vault.key')

# New: passphrase-protected "secure config" (migrated from SarahVaultCore.py)
VAULT_CONFIG_FILE = os.path.join(config.VAULT_DIR, 'vault_config.json')
VAULT_SALT_FILE = os.path.join(config.VAULT_DIR, 'vault.salt')

# Ensure VAULT_DIR exists
os.makedirs(config.VAULT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Back-compat Random-Key Vault (existing API preserved)
# -----------------------------------------------------------------------------
def generate_vault_key() -> bytes:
    """Generate and persist a random Fernet key (back-compat behavior)."""
    try:
        key = Fernet.generate_key()
        with open(VAULT_KEY_FILE, 'wb') as f:
            f.write(key)
        logger.info("Vault key generated and saved.")
        return key
    except Exception as e:
        logger.error(f"Error generating vault key: {e}")
        return None

def load_vault_key() -> bytes | None:
    """Load the persisted random Fernet key; generate if missing."""
    try:
        if not os.path.exists(VAULT_KEY_FILE):
            logger.warning("Vault key file not found. Generating a new key.")
            return generate_vault_key()
        with open(VAULT_KEY_FILE, 'rb') as f:
            key = f.read()
        logger.info("Vault key loaded successfully.")
        return key
    except Exception as e:
        logger.error(f"Error loading vault key: {e}")
        return None

def _fernet_from_random_key() -> Fernet | None:
    key = load_vault_key()
    if not key:
        logger.error("Failed to obtain Fernet object due to key issues.")
        return None
    return Fernet(key)

def load_vault() -> dict:
    """Decrypt and load the random-key vault JSON map."""
    try:
        if not os.path.exists(VAULT_FILE):
            logger.info("Vault file not found. Returning empty vault.")
            return {}
        fernet = _fernet_from_random_key()
        if not fernet:
            return {}
        with open(VAULT_FILE, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        vault_data = json.loads(decrypted_data.decode())
        logger.info("Vault data loaded and decrypted successfully.")
        return vault_data
    except Exception as e:
        logger.error(f"Error loading vault: {e}")
        return {}

def save_vault(vault_data: dict) -> bool:
    """Encrypt and save the random-key vault JSON map."""
    try:
        fernet = _fernet_from_random_key()
        if not fernet:
            return False
        data_str = json.dumps(vault_data)
        encrypted_data = fernet.encrypt(data_str.encode())
        with open(VAULT_FILE, 'wb') as f:
            f.write(encrypted_data)
        logger.info("Vault data encrypted and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving vault: {e}")
        return False

def add_item(key: str, value) -> bool:
    """Add/update a key in the random-key vault map."""
    try:
        vault = load_vault()
        vault[key] = value
        return save_vault(vault)
    except Exception as e:
        logger.error(f"Error adding item '{key}' to vault: {e}")
        return False

def get_item(key: str):
    """Get a key from the random-key vault map."""
    try:
        vault = load_vault()
        return vault.get(key, None)
    except Exception as e:
        logger.error(f"Error retrieving item '{key}' from vault: {e}")
        return None

def remove_item(key: str) -> bool:
    """Remove a key from the random-key vault map."""
    try:
        vault = load_vault()
        if key in vault:
            del vault[key]
            return save_vault(vault)
        else:
            logger.warning(f"Item '{key}' not found in vault.")
            return False
    except Exception as e:
        logger.error(f"Error removing item '{key}' from vault: {e}")
        return False

# -----------------------------------------------------------------------------
# Passphrase-Protected Secure Config (migrated & hardened)
# -----------------------------------------------------------------------------
KDF_ITERATIONS = 390_000  # modern default; adjust as needed

def _derive_key_from_password(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from password+salt using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))

def _load_or_create_salt() -> bytes:
    """Load the vault salt or create a new one (16 bytes) for passphrase mode."""
    if os.path.exists(VAULT_SALT_FILE):
        with open(VAULT_SALT_FILE, 'rb') as f:
            return f.read()
    salt = secrets.token_bytes(16)
    with open(VAULT_SALT_FILE, 'wb') as f:
        f.write(salt)
    return salt

def initialize_secure_config(password: str, seed_data: dict) -> bool:
    """
    Initialize the passphrase-based secure configuration.
    Writes:
      - VAULT_SALT_FILE (random salt)
      - VAULT_CONFIG_FILE (Fernet token of JSON seed_data)
    NOTE: We NEVER persist the derived key.
    """
    try:
        salt = _load_or_create_salt()
        key = _derive_key_from_password(password, salt)
        fernet = Fernet(key)
        token = fernet.encrypt(json.dumps(seed_data).encode('utf-8')).decode('utf-8')
        payload = {
            "created": datetime.utcnow().isoformat(),
            "kdf": "PBKDF2HMAC-SHA256",
            "kdf_iterations": KDF_ITERATIONS,
            "data": token,
        }
        with open(VAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logger.info("[Vault] Secure config initialized.")
        return True
    except Exception as e:
        logger.error(f"[Vault] initialize_secure_config error: {e}")
        return False

def unlock_vault_config(password: str) -> dict | None:
    """
    Decrypt and return the passphrase-based secure configuration.
    Returns None on failure (incl. wrong password).
    """
    try:
        if not os.path.exists(VAULT_CONFIG_FILE):
            logger.warning("[Vault] No secure config file found.")
            return None
        if not os.path.exists(VAULT_SALT_FILE):
            logger.error("[Vault] Salt file missing; cannot derive key.")
            return None
        with open(VAULT_SALT_FILE, 'rb') as f:
            salt = f.read()
        key = _derive_key_from_password(password, salt)
        with open(VAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        fernet = Fernet(key)
        try:
            plaintext = fernet.decrypt(payload['data'].encode('utf-8')).decode('utf-8')
        except InvalidToken:
            logger.error("[Vault] Invalid password (decryption failed).")
            return None
        return json.loads(plaintext)
    except Exception as e:
        logger.error(f"[Vault] unlock_vault_config error: {e}")
        return None

def has_secure_config() -> bool:
    """True if both salt + config files exist."""
    return os.path.exists(VAULT_SALT_FILE) and os.path.exists(VAULT_CONFIG_FILE)

# -----------------------------------------------------------------------------
# Demo (manual test)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Back-compat demo
    add_item("api_secret", "my_super_secret_api_key")
    assert get_item("api_secret") == "my_super_secret_api_key"
    remove_item("api_secret")

    # Passphrase demo (do not ship with real passwords)
    pwd = "sarah_secret"
    sample = {"SRH_balance": 1000.0, "owner": "Brian", "node": "Genesis"}
    initialize_secure_config(pwd, sample)
    print("[Vault Demo] unlocked =>", unlock_vault_config(pwd))
