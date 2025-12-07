"""--==The SarahMemory Project==--
File: SarahMemoryEncryption.py
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


from __future__ import annotations
import logging
import os
from cryptography.fernet import Fernet
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config

# Setup logging for the encryption module
logger = logging.getLogger('SarahMemoryEncryption')
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Define the path for storing the encryption key
KEY_FILE = os.path.join(os.getcwd(), 'encryption.key')

def log_encryption_event(event, details):
    """
    Logs an encryption-related event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encryption_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO encryption_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged encryption event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging encryption event: {e}")

def generate_key():
    """
    Generate a new Fernet encryption key and save it to a file.
    ENHANCED (v6.4): Now caches the key for subsequent operations.
    """
    try:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        logger.info("Encryption key generated and saved.")
        log_encryption_event("Generate Key", "Encryption key generated and saved successfully.")
        return key
    except Exception as e:
        logger.error(f"Error generating encryption key: {e}")
        log_encryption_event("Generate Key Error", f"Error generating encryption key: {e}")
        return None

def load_key():
    """
    Load the Fernet encryption key from the key file.
    ENHANCED (v6.4): If not found, automatically generates and caches a new key.
    """
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as key_file:
                key = key_file.read()
            logger.info("Encryption key loaded from file.")
            log_encryption_event("Load Key", "Encryption key loaded from file successfully.")
            return key
        else:
            logger.warning("Encryption key file not found. Generating a new key.")
            log_encryption_event("Load Key Warning", "Encryption key file not found. Generating a new key.")
            return generate_key()
    except Exception as e:
        logger.error(f"Error loading encryption key: {e}")
        log_encryption_event("Load Key Error", f"Error loading encryption key: {e}")
        return None

def encrypt_data(data):
    """
    Encrypt the provided data using Fernet encryption.
    ENHANCED (v6.4): Improved error handling and returns a UTF-8 decoded string.
    """
    try:
        key = load_key()
        if key is None:
            logger.error("Encryption key could not be loaded; encryption aborted.")
            log_encryption_event("Encrypt Data Error", "Encryption key could not be loaded; encryption aborted.")
            return None
        fernet = Fernet(key)
        encrypted = fernet.encrypt(data.encode())
        logger.info("Data encrypted successfully.")
        log_encryption_event("Encrypt Data", "Data encrypted successfully.")
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        log_encryption_event("Encrypt Data Error", f"Error encrypting data: {e}")
        return None

def decrypt_data(token):
    """
    Decrypt the provided token using Fernet encryption.
    ENHANCED (v6.4): Improved error recovery and detailed logging.
    """
    try:
        key = load_key()
        if key is None:
            logger.error("Encryption key could not be loaded; decryption aborted.")
            log_encryption_event("Decrypt Data Error", "Encryption key could not be loaded; decryption aborted.")
            return None
        fernet = Fernet(key)
        decrypted = fernet.decrypt(token.encode())
        logger.info("Data decrypted successfully.")
        log_encryption_event("Decrypt Data", "Data decrypted successfully.")
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        log_encryption_event("Decrypt Data Error", f"Error decrypting data: {e}")
        return None


# === SarahNetCrypto (embedded; no separate file needed) =====================
# Lightweight E2E crypto for constrained nodes; prefers ChaCha20-Poly1305,
# falls back to AES-CTR + HMAC-SHA256. No plaintext mode.
import hmac, hashlib, secrets, struct

try:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    _SM_CRYPTO_BACKEND = "cryptography"
except Exception:
    try:
        from Crypto.Cipher import ChaCha20_Poly1305 as _PyCha
        from Crypto.Cipher import AES as _PyAES
        _SM_CRYPTO_BACKEND = "pycryptodome"
    except Exception:
        _SM_CRYPTO_BACKEND = None

class SarahNetCrypto:
    HKDF_SALT = b"SarahMemory.Net.hkdf.v1"
    KEY_LEN   = 32
    NONCE_LEN = 12
    HMAC_LEN  = 32
    AEAD_TAG  = 16

    @staticmethod
    def hkdf_256(shared_secret: bytes, info: bytes = b"") -> bytes:
        prk = hmac.new(SarahNetCrypto.HKDF_SALT, shared_secret, hashlib.sha256).digest()
        return hmac.new(prk, info + b"\\x01", hashlib.sha256).digest()

    @staticmethod
    def _aes_ctr_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
        if _SM_CRYPTO_BACKEND == "cryptography":
            cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
            enc = cipher.encryptor()
            return enc.update(plaintext) + enc.finalize()
        elif _SM_CRYPTO_BACKEND == "pycryptodome":
            cipher = _PyAES.new(key, _PyAES.MODE_CTR, nonce=nonce)
            return cipher.encrypt(plaintext)
        raise RuntimeError("No crypto backend available.")

    @staticmethod
    def _aes_ctr_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
        if _SM_CRYPTO_BACKEND == "cryptography":
            cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
            dec = cipher.decryptor()
            return dec.update(ciphertext) + dec.finalize()
        elif _SM_CRYPTO_BACKEND == "pycryptodome":
            cipher = _PyAES.new(key, _PyAES.MODE_CTR, nonce=nonce)
            return cipher.decrypt(ciphertext)
        raise RuntimeError("No crypto backend available.")

    @staticmethod
    def seal(key: bytes, plaintext: bytes, aad: bytes = b"", prefer_chacha: bool = True) -> bytes:
        """
        Encrypt + authenticate.
        blob = b'SN' | flags(1) | nonce(12) | ct | tag_or_hmac(16/32)
        flags bit0: 1=chacha20poly1305, 0=aes-ctr+hmac
        """
        if _SM_CRYPTO_BACKEND is None:
            raise RuntimeError("No crypto backend available on this node.")
        flags = 0
        nonce = secrets.token_bytes(SarahNetCrypto.NONCE_LEN)

        # AEAD fast path
        if prefer_chacha and _SM_CRYPTO_BACKEND:
            try:
                if _SM_CRYPTO_BACKEND == "cryptography":
                    aead = ChaCha20Poly1305(key)
                    ct = aead.encrypt(nonce, plaintext, aad)
                else:
                    aead = _PyCha.new(key=key, nonce=nonce)
                    aead.update(aad)
                    ct, tag = aead.encrypt_and_digest(plaintext)
                    ct = ct + tag
                flags |= 1
                return b"SN" + struct.pack("B", flags) + nonce + ct
            except Exception:
                pass  # fall through

        # AES-CTR + HMAC
        ct = SarahNetCrypto._aes_ctr_encrypt(key, nonce, plaintext)
        mac = hmac.new(key, aad + nonce + ct, hashlib.sha256).digest()
        return b"SN" + struct.pack("B", flags) + nonce + ct + mac

    @staticmethod
    def open(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
        assert blob[:2] == b"SN", "Bad magic"
        flags = blob[2]
        nonce = blob[3:3+SarahNetCrypto.NONCE_LEN]
        body  = blob[3+SarahNetCrypto.NONCE_LEN:]
        if flags & 1:
            if _SM_CRYPTO_BACKEND == "cryptography":
                aead = ChaCha20Poly1305(key)
                return aead.decrypt(nonce, body, aad)
            ct, tag = body[:-SarahNetCrypto.AEAD_TAG], body[-SarahNetCrypto.AEAD_TAG:]
            aead = _PyCha.new(key=key, nonce=nonce)
            aead.update(aad)
            return aead.decrypt_and_verify(ct, tag)
        # AES-CTR + HMAC
        ct, mac = body[:-SarahNetCrypto.HMAC_LEN], body[-SarahNetCrypto.HMAC_LEN:]
        exp = hmac.new(key, aad + nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, exp):
            raise ValueError("HMAC failed")
        return SarahNetCrypto._aes_ctr_decrypt(key, nonce, ct)

# Convenience aliases for callers
hkdf_256 = SarahNetCrypto.hkdf_256
seal     = SarahNetCrypto.seal
open     = SarahNetCrypto.open
# ============================================================================


if __name__ == '__main__':
    logger.info("Starting SarahMemoryEncryption module test.")
    sample_text = "Sensitive information that needs encryption."
    encrypted_text = encrypt_data(sample_text)
    if encrypted_text:
        logger.info(f"Encrypted Text: {encrypted_text}")
        decrypted_text = decrypt_data(encrypted_text)
        logger.info(f"Decrypted Text: {decrypted_text}")
    else:
        logger.error("Encryption test failed.")
    logger.info("SarahMemoryEncryption module testing complete.")