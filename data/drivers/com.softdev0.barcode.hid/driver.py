# SarahMemory Driver Package: com.softdev0.barcode.hid
# Name: Barcode Scanner HID Driver
#
# SAFE skeleton driver implementation.
# No privileged OS driver installs. No auto-probing unless invoked via actions.
#
# Recommended interface:
#   driver_info(), driver_validate(config), driver_init(context, config),
#   driver_shutdown(context), driver_status(context), driver_action(action_id, context, payload)

import time
from datetime import datetime

DRIVER_ID = "com.softdev0.barcode.hid"
DRIVER_NAME = "Barcode Scanner HID Driver"
VERSION = "0.1.0"

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
}

def _now_id():
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper()
    )

def driver_info():
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": dict(_SESSION),
    }

def driver_validate(config: dict):
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    return {"ok": True, "errors": [], "warnings": []}

def driver_init(context=None, config=None):
    config = config or {}
    v = driver_validate(config)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["meta"] = {"context": context or {}, "config": config}
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}

def driver_shutdown(context=None):
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True

def driver_status(context=None):
    return {"ok": True, "session": dict(_SESSION)}

def driver_action(action_id: str, context=None, payload=None):
    return {
        "ok": False,
        "error": f"Action '{action_id}' not implemented by {DRIVER_ID}",
        "action_id": action_id
    }
