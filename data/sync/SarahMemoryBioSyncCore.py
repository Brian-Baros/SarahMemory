# SarahBioSyncCore.py
# Handles biometric signature syncing and verification for SarahMemory authentication

import os
import json
import hashlib
from datetime import datetime

BIOSYNC_PATH = os.path.join("data", "security", "biosignature.json")

if not os.path.exists(os.path.dirname(BIOSYNC_PATH)):
    os.makedirs(os.path.dirname(BIOSYNC_PATH))


def generate_signature(input_data):
    hashed = hashlib.sha256(input_data.encode()).hexdigest()
    return hashed


def store_signature(name, biometrics):
    signature = generate_signature(biometrics)
    profile = {
        "name": name,
        "signature": signature,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(BIOSYNC_PATH, 'w') as f:
        json.dump(profile, f, indent=4)
    print("[BioSync] Signature stored.")


def verify_signature(input_data):
    if not os.path.exists(BIOSYNC_PATH):
        return False
    with open(BIOSYNC_PATH, 'r') as f:
        saved = json.load(f)
    return saved['signature'] == generate_signature(input_data)


if __name__ == '__main__':
    # Test usage
    store_signature("Brian", "voiceprint1234-retinaXYZ")
    valid = verify_signature("voiceprint1234-retinaXYZ")
    print(f"[BioSync] Access {'granted' if valid else 'denied'}.")
