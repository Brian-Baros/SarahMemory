"""--==The SarahMemory Project==--
File: SarahMemoryEncryption.py
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


# ============================================================================
# SarahMemory v8.x — Corporate License Crypto + Tk GUI Utilities
# NOTE: Added as an extension layer to avoid repeated core edits.
# - Generates Ed25519 keypair (private/public)
# - Signs and verifies SarahMemory.lic (offline)
# - Optional Fernet vault key generator
# - Optional Tk GUI (invoked with --gui or SM_ENCRYPTION_GUI=true)
# ============================================================================

import json as _json
import base64 as _base64
from datetime import datetime as _dt, timedelta as _td
from typing import Optional as _Optional, Tuple as _Tuple, Dict as _Dict, Any as _Any

class SarahMemoryLicenseCrypto:
    """
    License signing & verification for SarahMemory corporate update channel.

    Security model:
      - PRIVATE KEY (Ed25519) is used ONLY to sign license files (admin-side).
      - PUBLIC KEY is stored in .env (SM_LICENSE_PUBLIC_KEY_B64) for verification (client-side).
      - License verification is offline and tamper-evident.

    License file path (default):
      ./data/license/SarahMemory.lic
    """

    @staticmethod
    def _ed25519_available() -> bool:
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def generate_ed25519_keypair_b64() -> _Tuple[str, str]:
        """
        Returns (private_key_b64, public_key_b64).
        """
        if not SarahMemoryLicenseCrypto._ed25519_available():
            raise RuntimeError("cryptography ed25519 unavailable (install/enable cryptography)")

        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()

        priv_bytes = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return (_base64.b64encode(priv_bytes).decode("utf-8"), _base64.b64encode(pub_bytes).decode("utf-8"))

    @staticmethod
    def _canonical_payload(lic: _Dict[str, _Any]) -> bytes:
        """
        Canonical JSON bytes for signing/verifying (sorted keys, compact separators).
        """
        tmp = dict(lic)
        tmp.pop("signature", None)
        return _json.dumps(tmp, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @staticmethod
    def sign_license_dict(lic: _Dict[str, _Any], private_key_b64: str) -> _Dict[str, _Any]:
        """
        Returns a new dict with a valid 'signature' field (base64).
        """
        if not SarahMemoryLicenseCrypto._ed25519_available():
            raise RuntimeError("cryptography ed25519 unavailable (install/enable cryptography)")

        from cryptography.hazmat.primitives.asymmetric import ed25519

        payload = SarahMemoryLicenseCrypto._canonical_payload(lic)
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(_base64.b64decode(private_key_b64))
        sig = priv.sign(payload)

        out = dict(lic)
        out["signature"] = _base64.b64encode(sig).decode("utf-8")
        return out

    @staticmethod
    def verify_license_dict(lic: _Dict[str, _Any], public_key_b64: str, now_utc: _Optional[_dt] = None) -> _Tuple[bool, str]:
        """
        Returns (ok, reason).
        Also validates expiration when expires_at is present.

        expires_at rules:
          - missing or null => lifetime
          - ISO8601 string (Z allowed) => must be >= now
        """
        if not SarahMemoryLicenseCrypto._ed25519_available():
            return (False, "cryptography ed25519 unavailable")

        from cryptography.hazmat.primitives.asymmetric import ed25519

        if "signature" not in lic or not lic.get("signature"):
            return (False, "missing signature")

        payload = SarahMemoryLicenseCrypto._canonical_payload(lic)
        try:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(_base64.b64decode(public_key_b64))
            pub.verify(_base64.b64decode(str(lic.get("signature"))), payload)
        except Exception:
            return (False, "signature invalid")

        # Expiration check
        now = now_utc or _dt.utcnow()
        exp = lic.get("expires_at", None)
        if exp in (None, "", "null"):
            return (True, "ok (lifetime)")

        try:
            exp_s = str(exp).replace("Z", "+00:00")
            exp_dt = _dt.fromisoformat(exp_s)
            # normalize naive => assume UTC
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=None)
                # compare naive with naive
                if exp_dt < now:
                    return (False, "license expired")
            else:
                # compare aware UTC
                if exp_dt.astimezone(tz=None).replace(tzinfo=None) < now:
                    return (False, "license expired")
        except Exception:
            # If date format is bad, fail closed
            return (False, "expires_at invalid format")

        return (True, "ok")

    @staticmethod
    def save_license_file(path: str, lic: _Dict[str, _Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(lic, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_license_file(path: str) -> _Dict[str, _Any]:
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)


def _sm_project_root() -> str:
    """
    Best-effort project root detection:
    - Prefer a directory that contains ./data/
    - Start from current working directory; fall back to this file's directory
    """
    # 1) CWD and parents
    cwd = Path(os.getcwd()).resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / "data").is_dir():
            return str(p)

    # 2) File directory and parents
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / "data").is_dir():
            return str(p)

    return str(cwd)


def _sm_license_path() -> str:
    root = _sm_project_root()
    return str(Path(root) / "data" / "license" / "SarahMemory.lic")


def _sm_vault_key_path() -> str:
    root = _sm_project_root()
    return str(Path(root) / "data" / "vault" / "encryption_key.key")


def _duration_to_expires(days: _Optional[int] = None) -> _Optional[str]:
    if days is None:
        return None
    exp = _dt.utcnow() + _td(days=int(days))
    return exp.replace(microsecond=0).isoformat() + "Z"


def _default_license_payload(org: str, tier: str, duration_days: _Optional[int]) -> _Dict[str, _Any]:
    now = _dt.utcnow().replace(microsecond=0).isoformat() + "Z"
    expires = _duration_to_expires(duration_days)
    return {
        "license_id": f"SM-{tier.upper()}-{_dt.utcnow().strftime('%Y%m%d')}-0001",
        "organization": org or "Unknown Organization",
        "organization_id": (org or "ORG").upper().replace(" ", "_")[:64],
        "tier": tier or "corporate",
        "issued_to": "",
        "issued_by": "SoftDev0 LLC",
        "issued_at": now,
        "expires_at": expires,  # None => lifetime
        "allowed_channels": ["corporate"] if (tier or "corporate").lower() != "community" else ["community"],
        "features": ["corporate_updates"] if (tier or "corporate").lower() != "community" else ["community_updates"],
        "hardware_binding": {"required": False, "machine_fingerprint": None},
        "notes": "",
    }


class SarahMemoryEncryptionGUI:
    """
    Tk GUI to:
      - Generate Ed25519 keypair (private/public)
      - Generate + SIGN SarahMemory.lic into ./data/license/SarahMemory.lic (overwrite)
      - Verify existing SarahMemory.lic against the generated/public key
      - Optionally create Fernet vault key at ./data/vault/encryption_key.key (overwrite)
    """

    DURATION_CHOICES = [
        ("7-days", 7),
        ("30-days", 30),
        ("60-days", 60),
        ("90-days", 90),
        ("6-months", 183),
        ("1yr", 365),
        ("3yr", 365 * 3),
        ("5yr", 365 * 5),
        ("LIFETIME", None),
    ]

    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
        except Exception as e:
            raise RuntimeError(f"tkinter unavailable: {e}")

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox

        self.root = tk.Tk()
        self.root.title("SarahMemory Encryption Tools (v8.0.0)")
        self.root.geometry("920x700")

        # state
        self.private_key_b64: _Optional[str] = None
        self.public_key_b64: _Optional[str] = None

        # Vars
        self.org_var = tk.StringVar(value="Your Organization")
        self.tier_var = tk.StringVar(value="corporate")
        self.duration_var = tk.StringVar(value="90-days")
        self.make_vault_var = tk.BooleanVar(value=False)

        self._build()

    def _build(self) -> None:
        ttk = self.ttk

        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="both", expand=True)

        title = ttk.Label(top, text="SarahMemory Encryption + License Toolkit", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        # License config frame
        lf = ttk.Labelframe(top, text="Corporate License Generator", padding=10)
        lf.pack(fill="x", pady=(0, 10))

        row1 = ttk.Frame(lf)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Organization:", width=16).pack(side="left")
        ttk.Entry(row1, textvariable=self.org_var, width=52).pack(side="left")

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Tier:", width=16).pack(side="left")
        ttk.Combobox(row2, textvariable=self.tier_var, values=("corporate", "community"), state="readonly", width=18).pack(side="left")

        ttk.Label(row2, text="Duration:", padding=(18, 0, 0, 0)).pack(side="left")
        ttk.Combobox(
            row2,
            textvariable=self.duration_var,
            values=[label for (label, _) in self.DURATION_CHOICES],
            state="readonly",
            width=18,
        ).pack(side="left")

        ttk.Checkbutton(lf, text="Also create/overwrite Fernet vault key (data/vault/encryption_key.key)", variable=self.make_vault_var).pack(anchor="w", pady=(6, 0))

        row3 = ttk.Frame(lf)
        row3.pack(fill="x", pady=8)
        ttk.Button(row3, text="Generate Keys + Create Signed SarahMemory.lic (overwrite)", command=self._do_generate_license).pack(side="left")
        ttk.Button(row3, text="Verify Existing SarahMemory.lic", command=self._do_verify_existing_license).pack(side="left", padx=(8, 0))
        ttk.Button(row3, text="Generate Fernet Vault Key Only (overwrite)", command=self._do_generate_vault_key).pack(side="left", padx=(8, 0))

        # Outputs
        out = ttk.Labelframe(top, text="Outputs", padding=10)
        out.pack(fill="both", expand=True)

        ttk.Label(out, text="PUBLIC KEY (.env): SM_LICENSE_PUBLIC_KEY_B64=...").pack(anchor="w")
        self.pub_txt = self.tk.Text(out, height=3, wrap="word")
        self.pub_txt.pack(fill="x", pady=(2, 10))

        ttk.Label(out, text="PRIVATE KEY (KEEP SECRET — DO NOT COMMIT):").pack(anchor="w")
        self.priv_txt = self.tk.Text(out, height=3, wrap="word")
        self.priv_txt.pack(fill="x", pady=(2, 10))

        row_copy = ttk.Frame(out)
        row_copy.pack(fill="x", pady=(0, 10))
        ttk.Button(row_copy, text="Copy PUBLIC .env line to clipboard", command=self._copy_public_env_line).pack(side="left")
        ttk.Button(row_copy, text="Copy PRIVATE key to clipboard", command=self._copy_private_key).pack(side="left", padx=(8, 0))

        ttk.Label(out, text="Log:").pack(anchor="w")
        self.log_txt = self.tk.Text(out, height=14, wrap="word")
        self.log_txt.pack(fill="both", expand=True)

        # Footer buttons
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.pack(fill="x")
        ttk.Button(footer, text="Exit", command=self._exit).pack(side="right")

        self._log(f"Project root: {_sm_project_root()}")
        self._log(f"License path: {_sm_license_path()}")
        self._log(f"Vault key path: {_sm_vault_key_path()}")

    def _log(self, msg: str) -> None:
        try:
            self.log_txt.insert("end", msg + "\n")
            self.log_txt.see("end")
        except Exception:
            pass

    def _selected_duration_days(self) -> _Optional[int]:
        label = (self.duration_var.get() or "").strip()
        for (lab, days) in self.DURATION_CHOICES:
            if lab == label:
                return days
        # default to 90
        return 90

    def _do_generate_license(self) -> None:
        org = (self.org_var.get() or "").strip()
        tier = (self.tier_var.get() or "corporate").strip().lower()
        duration_days = self._selected_duration_days()

        try:
            priv_b64, pub_b64 = SarahMemoryLicenseCrypto.generate_ed25519_keypair_b64()
            payload = _default_license_payload(org=org, tier=tier, duration_days=duration_days)
            signed = SarahMemoryLicenseCrypto.sign_license_dict(payload, private_key_b64=priv_b64)

            lic_path = _sm_license_path()
            SarahMemoryLicenseCrypto.save_license_file(lic_path, signed)

            # Verify immediately
            ok, reason = SarahMemoryLicenseCrypto.verify_license_dict(signed, public_key_b64=pub_b64)

            self.private_key_b64 = priv_b64
            self.public_key_b64 = pub_b64

            self.pub_txt.delete("1.0", "end")
            self.pub_txt.insert("1.0", f"SM_LICENSE_PUBLIC_KEY_B64={pub_b64}")

            self.priv_txt.delete("1.0", "end")
            self.priv_txt.insert("1.0", priv_b64)

            self._log("Generated Ed25519 keypair.")
            self._log(f"Wrote signed license: {lic_path}")
            self._log(f"License verify: {ok} ({reason})")

            if bool(self.make_vault_var.get()):
                self._do_generate_vault_key()

            if not ok:
                self.messagebox.showwarning("Generated, but verify failed", f"License was written but did NOT verify:\n{reason}\n\nCheck public key usage.")
            else:
                self.messagebox.showinfo("Success", f"License generated and saved to:\n{lic_path}\n\nPublic key line is shown for .env.")
        except Exception as e:
            self._log(f"ERROR generating license: {e}")
            self.messagebox.showerror("Error", f"Failed to generate license:\n{e}")

    def _do_verify_existing_license(self) -> None:
        try:
            lic_path = _sm_license_path()
            if not os.path.exists(lic_path):
                self.messagebox.showwarning("Not found", f"No license file found at:\n{lic_path}")
                return

            lic = SarahMemoryLicenseCrypto.load_license_file(lic_path)

            pub = self.public_key_b64
            # If user hasn't generated keys in this session, allow them to paste public key into pub box
            if not pub:
                # Try read from pub text box (.env line)
                txt = self.pub_txt.get("1.0", "end").strip()
                if "SM_LICENSE_PUBLIC_KEY_B64=" in txt:
                    pub = txt.split("SM_LICENSE_PUBLIC_KEY_B64=", 1)[1].strip()
                else:
                    pub = txt.strip() or None

            if not pub:
                self.messagebox.showwarning("Missing public key", "No public key available.\nGenerate keys first or paste public key into PUBLIC box.")
                return

            ok, reason = SarahMemoryLicenseCrypto.verify_license_dict(lic, public_key_b64=pub)
            self._log(f"Verify existing license: {ok} ({reason})")
            if ok:
                self.messagebox.showinfo("Verified", f"License is VALID.\n{reason}")
            else:
                self.messagebox.showerror("Invalid", f"License is INVALID.\n{reason}")
        except Exception as e:
            self._log(f"ERROR verifying license: {e}")
            self.messagebox.showerror("Error", f"Failed to verify license:\n{e}")

    def _do_generate_vault_key(self) -> None:
        try:
            # Use existing generator if present; else fallback to cryptography.fernet
            key_path = _sm_vault_key_path()
            os.makedirs(os.path.dirname(key_path), exist_ok=True)

            key_bytes = None
            try:
                # If this module already has generate_key() for fernet, prefer it.
                if "generate_key" in globals() and callable(globals().get("generate_key")):
                    key_bytes = globals()["generate_key"]()
                    # generate_key() may return bytes or write to default path; normalize to bytes
                    if isinstance(key_bytes, str):
                        key_bytes = key_bytes.encode("utf-8")
            except Exception:
                key_bytes = None

            if not key_bytes:
                from cryptography.fernet import Fernet
                key_bytes = Fernet.generate_key()

            with open(key_path, "wb") as f:
                f.write(key_bytes)

            self._log(f"Wrote Fernet vault key: {key_path}")
            self.messagebox.showinfo("Vault key created", f"Vault encryption key saved to:\n{key_path}")
        except Exception as e:
            self._log(f"ERROR generating vault key: {e}")
            self.messagebox.showerror("Error", f"Failed to generate vault key:\n{e}")

    def _copy_public_env_line(self) -> None:
        try:
            line = self.pub_txt.get("1.0", "end").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(line)
            self._log("Copied public .env line to clipboard.")
        except Exception as e:
            self._log(f"Clipboard error: {e}")

    def _copy_private_key(self) -> None:
        try:
            line = self.priv_txt.get("1.0", "end").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(line)
            self._log("Copied private key to clipboard.")
        except Exception as e:
            self._log(f"Clipboard error: {e}")

    def _exit(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


def _sm_should_launch_gui() -> bool:
    """
    Launch rules:
      - python SarahMemoryEncryption.py --gui
      - OR SM_ENCRYPTION_GUI=true
    """
    try:
        argv = [a.strip().lower() for a in sys.argv[1:]]
        if "--gui" in argv:
            return True
    except Exception:
        pass
    try:
        return str(os.environ.get("SM_ENCRYPTION_GUI", "")).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


# Hook into module execution without breaking existing behavior
try:
    if __name__ == "__main__" and _sm_should_launch_gui():
        gui = SarahMemoryEncryptionGUI()
        gui.run()
except Exception as _e_gui:
    try:
        logger.error(f"[SarahMemoryEncryptionGUI] Failed to start GUI: {_e_gui}")
    except Exception:
        pass


# ====================================================================
# END OF SarahMemoryEncryption.py v8.0.0
# ====================================================================