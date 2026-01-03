# router/capability_router/addon.py
# SarahMemory AiOS Default SoftPack - Capability Router

import os
import time
import json
from pathlib import Path

_ADDON_ID = "softpack.router.capability"
_SESSION = {"instance_id": None, "started_ts": None, "status": "idle", "error": None}

def _data_dir(context):
    try:
        if context and isinstance(context, dict) and context.get("data_dir"):
            return Path(context["data_dir"]).expanduser().resolve()
    except Exception:
        pass
    return Path(os.getcwd()).resolve() / "data"

def _read_cache(context):
    p = _data_dir(context) / "registry" / "software_index.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def addon_info():
    return {"id": _ADDON_ID, "name": "Capability Router", "session": dict(_SESSION)}

def addon_validate(config: dict):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    return {"ok": True, "errors": [], "warnings": []}

def addon_init(context=None, config=None):
    v = addon_validate(config or {})
    if not v.get("ok"):
        _SESSION.update({"status":"error","error":"validation_failed"})
        return {"ok": False, "error":"validation_failed", "details": v}
    _SESSION.update({"instance_id": f"SP-ROUTE-{int(time.time())}", "started_ts": time.time(), "status":"ready", "error": None})
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": addon_info()}

def addon_shutdown(context=None):
    _SESSION.update({"instance_id": None, "started_ts": None, "status":"stopped", "error": None})
    return True

def addon_action(action_id: str, context=None, payload=None):
    if action_id not in ("capability.route","route"):
        return {"ok": False, "error": f"Action '{action_id}' not implemented"}
    payload = payload or {}
    capability = payload.get("capability")
    if not capability or not isinstance(capability, str):
        return {"ok": False, "error": "missing capability"}

    # 1) DB mapping hook (optional)
    try:
        if context and isinstance(context, dict):
            cmap = context.get("capability_map")
            if isinstance(cmap, dict) and capability in cmap:
                return {"ok": True, "source": "context.capability_map", "route": cmap[capability]}
    except Exception:
        pass

    # 2) cache hints (for UI selection / logging)
    hints = _read_cache(context)

    # 3) fallback provider always works
    route = {"provider": "softpack.provider.os_default", "action": capability}
    return {"ok": True, "source": "fallback", "route": route, "hints": hints}
