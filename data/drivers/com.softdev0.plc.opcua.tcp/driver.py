# com.softdev0.plc.opcua.tcp/driver.py
# OPC UA client driver for SarahMemory AiOS
# Transport: tcp

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.plc.opcua.tcp"
_DEFAULT_ENDPOINT = "opc.tcp://127.0.0.1:4840"

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "endpoint": None,
    "client_backend": None,
    "subscriptions": {},
    "last_result": None,
    "history": [],
}

_STATE = {
    "config": {},
    "client": None,
    "backend": None,
    "subscription_counter": 0,
    "subscription_handles": {},
    "events": [],
}


def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _now():
    return time.time()


def _append_note(msg: str):
    _SESSION.setdefault("notes", []).append(msg)
    _SESSION["notes"] = _SESSION["notes"][-50:]


def _record(action: str, ok: bool, **extra):
    row = {"ts": _now(), "action": action, "ok": bool(ok)}
    row.update(extra)
    _SESSION["last_result"] = row
    hist = _SESSION.setdefault("history", [])
    hist.append(row)
    limit = int(_STATE.get("config", {}).get("history_limit", 200) or 200)
    _SESSION["history"] = hist[-limit:]
    return row


def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg or {})
    for key in ["password", "private_key_path", "certificate_path", "application_uri"]:
        if out.get(key):
            out[key] = "***"
    return out


def _coerce_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config or {})
    return {
        "enabled": _coerce_bool(cfg.get("enabled", True), True),
        "autoload": _coerce_bool(cfg.get("autoload", False), False),
        "endpoint": str(cfg.get("endpoint") or _DEFAULT_ENDPOINT),
        "security_mode": str(cfg.get("security_mode") or "None"),
        "security_policy": str(cfg.get("security_policy") or "None"),
        "username": str(cfg.get("username") or ""),
        "password": str(cfg.get("password") or ""),
        "certificate_path": str(cfg.get("certificate_path") or ""),
        "private_key_path": str(cfg.get("private_key_path") or ""),
        "application_uri": str(cfg.get("application_uri") or ""),
        "timeout": float(cfg.get("timeout", 4.0) or 4.0),
        "session_timeout": float(cfg.get("session_timeout", 60.0) or 60.0),
        "read_only": _coerce_bool(cfg.get("read_only", True), True),
        "require_write_confirmation": _coerce_bool(cfg.get("require_write_confirmation", True), True),
        "allow_method_calls": _coerce_bool(cfg.get("allow_method_calls", False), False),
        "require_method_confirmation": _coerce_bool(cfg.get("require_method_confirmation", True), True),
        "subscription_period_ms": _coerce_int(cfg.get("subscription_period_ms", 500), 500),
        "event_buffer_limit": _coerce_int(cfg.get("event_buffer_limit", 500), 500),
        "history_limit": _coerce_int(cfg.get("history_limit", 200), 200),
        "namespace_hints": list(cfg.get("namespace_hints") or []),
    }


def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "PLC OPC UA Driver",
        "version": "1.0.0",
        "transport": "tcp",
        "session": dict(_SESSION),
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    cfg = _normalize_config(config)
    if not cfg["endpoint"].startswith("opc.tcp://"):
        errors.append("endpoint must start with opc.tcp://")
    if cfg["security_mode"] not in {"None", "Sign", "SignAndEncrypt"}:
        errors.append("security_mode must be one of None, Sign, SignAndEncrypt")
    if cfg["security_policy"] not in {"None", "Basic128Rsa15", "Basic256", "Basic256Sha256", "Aes128_Sha256_RsaOaep", "Aes256_Sha256_RsaPss"}:
        warnings.append("security_policy is not in the known policy list; backend-specific handling may fail")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "normalized": cfg}


def _probe_backends():
    result = {"asyncua": False, "opcua": False}
    try:
        import asyncua  # noqa: F401
        result["asyncua"] = True
    except Exception:
        pass
    try:
        import opcua  # noqa: F401
        result["opcua"] = True
    except Exception:
        pass
    result["preferred"] = "asyncua" if result["asyncua"] else ("opcua" if result["opcua"] else None)
    return result


def _reset_runtime():
    _STATE["client"] = None
    _STATE["backend"] = None
    _STATE["subscription_counter"] = 0
    _STATE["subscription_handles"] = {}
    _STATE["events"] = []
    _SESSION["connected"] = False
    _SESSION["endpoint"] = None
    _SESSION["client_backend"] = None
    _SESSION["subscriptions"] = {}


def driver_init(context=None, config=None):
    cfg = _normalize_config(config or {})
    v = driver_validate(cfg)
    if not v.get("ok"):
        _SESSION.update({"status": "error", "error": "validation_failed"})
        return {"ok": False, "error": "validation_failed", "details": v}

    _reset_runtime()
    _STATE["config"] = cfg
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = _now()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = ["OPC UA client driver initialized."]
    backends = _probe_backends()
    if not backends["preferred"]:
        _append_note("No OPC UA backend installed; connect/read/write actions will report backend_unavailable.")
    _record("init", True, backends=backends)
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "backends": backends, "info": driver_info()}


def driver_shutdown(context=None):
    try:
        _disconnect()
    except Exception:
        pass
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["notes"] = ["Shutdown completed."]
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _mask_config(_STATE.get("config", {})),
        "event_count": len(_STATE.get("events", [])),
        "backends": _probe_backends(),
    }


def _get_node(node_id: str):
    client = _STATE.get("client")
    backend = _STATE.get("backend")
    if not client or not backend:
        raise RuntimeError("not_connected")
    if backend == "asyncua":
        return client.get_node(node_id)
    return client.get_node(node_id)


def _parse_data_value(value):
    try:
        if hasattr(value, "Value") and hasattr(value.Value, "Value"):
            return value.Value.Value
    except Exception:
        pass
    return value


def _connect(cfg: Dict[str, Any]):
    endpoint = cfg["endpoint"]
    backends = _probe_backends()
    if backends["asyncua"]:
        from asyncua.sync import Client
        client = Client(endpoint, timeout=int(cfg["timeout"] or 4))
        if cfg.get("application_uri"):
            try:
                client.application_uri = cfg["application_uri"]
            except Exception:
                pass
        if cfg.get("username"):
            client.set_user(cfg["username"])
        if cfg.get("password"):
            client.set_password(cfg["password"])
        # Certificate-based security wiring can vary by install and backend version.
        client.connect()
        _STATE["client"] = client
        _STATE["backend"] = "asyncua"
    elif backends["opcua"]:
        from opcua import Client
        client = Client(endpoint, timeout=cfg["timeout"])
        if cfg.get("username"):
            client.set_user(cfg["username"])
        if cfg.get("password"):
            client.set_password(cfg["password"])
        client.connect()
        _STATE["client"] = client
        _STATE["backend"] = "opcua"
    else:
        return {"ok": False, "error": "backend_unavailable", "details": backends}
    _SESSION["connected"] = True
    _SESSION["endpoint"] = endpoint
    _SESSION["client_backend"] = _STATE["backend"]
    _SESSION["status"] = "connected"
    return {"ok": True, "backend": _STATE["backend"], "endpoint": endpoint}


def _disconnect():
    client = _STATE.get("client")
    if client is not None:
        try:
            client.disconnect()
        except Exception:
            try:
                client.close_session()
            except Exception:
                pass
    _reset_runtime()
    _SESSION["status"] = "ready"
    return {"ok": True}


def _ensure_connected():
    if not _STATE.get("client"):
        raise RuntimeError("not_connected")


def _browse(node_id: str = "i=84"):
    _ensure_connected()
    node = _get_node(node_id)
    refs = node.get_children()
    items = []
    for child in refs:
        try:
            browse_name = child.get_browse_name()
            display_name = child.get_display_name().Text
            items.append({
                "node_id": child.nodeid.to_string() if hasattr(child.nodeid, "to_string") else str(child.nodeid),
                "browse_name": str(browse_name),
                "display_name": display_name,
            })
        except Exception as exc:
            items.append({"node_id": str(child), "error": str(exc)})
    return {"ok": True, "items": items, "count": len(items)}


def _read_node(node_id: str, attribute: str = "Value"):
    _ensure_connected()
    node = _get_node(node_id)
    if attribute == "Value":
        val = node.get_data_value() if hasattr(node, "get_data_value") else node.get_value()
        return {"ok": True, "node_id": node_id, "attribute": attribute, "value": _parse_data_value(val)}
    if attribute == "DisplayName":
        return {"ok": True, "node_id": node_id, "attribute": attribute, "value": node.get_display_name().Text}
    if attribute == "BrowseName":
        return {"ok": True, "node_id": node_id, "attribute": attribute, "value": str(node.get_browse_name())}
    if attribute == "NodeClass":
        return {"ok": True, "node_id": node_id, "attribute": attribute, "value": str(node.get_node_class())}
    return {"ok": False, "error": "unsupported_attribute", "attribute": attribute}


def _coerce_write_value(value, data_type: str | None = None):
    if data_type in (None, "auto", ""):
        return value
    dt = str(data_type).lower()
    if dt in {"bool", "boolean"}:
        return _coerce_bool(value)
    if dt in {"int", "int32", "int16", "int64", "uint16", "uint32", "uint64", "byte", "sbyte"}:
        return int(value)
    if dt in {"float", "double"}:
        return float(value)
    if dt in {"json", "dict", "list"} and isinstance(value, str):
        return json.loads(value)
    return str(value) if dt in {"string", "text"} else value


def _write_node(node_id: str, value: Any, data_type: str | None, confirm_write: bool):
    cfg = _STATE["config"]
    if cfg.get("read_only", True):
        return {"ok": False, "error": "read_only"}
    if cfg.get("require_write_confirmation", True) and not confirm_write:
        return {"ok": False, "error": "confirmation_required"}
    _ensure_connected()
    node = _get_node(node_id)
    typed = _coerce_write_value(value, data_type)
    node.set_value(typed)
    return {"ok": True, "node_id": node_id, "written": typed}


def _call_method(object_node_id: str, method_node_id: str, args: List[Any], confirm_method: bool):
    cfg = _STATE["config"]
    if not cfg.get("allow_method_calls", False):
        return {"ok": False, "error": "method_calls_disabled"}
    if cfg.get("require_method_confirmation", True) and not confirm_method:
        return {"ok": False, "error": "confirmation_required"}
    _ensure_connected()
    obj = _get_node(object_node_id)
    method = _get_node(method_node_id)
    result = obj.call_method(method, *(args or []))
    return {"ok": True, "result": result}


class _SubHandler:
    def __init__(self, sub_id: str):
        self.sub_id = sub_id

    def datachange_notification(self, node, val, data):
        _push_event({
            "kind": "datachange",
            "subscription_id": self.sub_id,
            "node": str(getattr(node, "nodeid", node)),
            "value": val,
            "ts": _now(),
        })

    def event_notification(self, event):
        _push_event({
            "kind": "event",
            "subscription_id": self.sub_id,
            "event": repr(event),
            "ts": _now(),
        })


def _push_event(evt: Dict[str, Any]):
    events = _STATE.setdefault("events", [])
    events.append(evt)
    limit = int(_STATE.get("config", {}).get("event_buffer_limit", 500) or 500)
    _STATE["events"] = events[-limit:]


def _create_subscription(node_ids: List[str], period_ms: Optional[int] = None):
    _ensure_connected()
    if not isinstance(node_ids, list) or not node_ids:
        return {"ok": False, "error": "node_ids_required"}
    _STATE["subscription_counter"] += 1
    sid = f"sub-{_STATE['subscription_counter']}"
    period = int(period_ms or _STATE["config"].get("subscription_period_ms", 500))
    handler = _SubHandler(sid)
    client = _STATE["client"]
    sub = client.create_subscription(period, handler)
    handles = []
    for nid in node_ids:
        node = _get_node(nid)
        try:
            h = sub.subscribe_data_change(node)
        except Exception:
            h = None
        handles.append({"node_id": nid, "handle": h})
    _STATE["subscription_handles"][sid] = {"subscription": sub, "handles": handles, "nodes": list(node_ids), "period_ms": period}
    _SESSION["subscriptions"][sid] = {"nodes": list(node_ids), "period_ms": period}
    return {"ok": True, "subscription_id": sid, "nodes": node_ids, "period_ms": period}


def _delete_subscription(subscription_id: str):
    rec = _STATE["subscription_handles"].get(subscription_id)
    if not rec:
        return {"ok": False, "error": "subscription_not_found"}
    try:
        rec["subscription"].delete()
    except Exception:
        pass
    _STATE["subscription_handles"].pop(subscription_id, None)
    _SESSION["subscriptions"].pop(subscription_id, None)
    return {"ok": True, "subscription_id": subscription_id}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    try:
        if action_id == "ping":
            result = {"ok": True, "pong": True, "ts": _now(), "backends": _probe_backends()}
        elif action_id == "probe_backends":
            result = {"ok": True, "backends": _probe_backends()}
        elif action_id == "connect":
            cfg = _STATE["config"].copy()
            cfg.update({k: v for k, v in payload.items() if k in cfg})
            _STATE["config"] = cfg
            result = _connect(cfg)
        elif action_id == "disconnect":
            result = _disconnect()
        elif action_id == "browse":
            result = _browse(str(payload.get("node_id") or "i=84"))
        elif action_id == "read_node":
            result = _read_node(str(payload.get("node_id") or ""), str(payload.get("attribute") or "Value"))
        elif action_id == "write_node":
            result = _write_node(str(payload.get("node_id") or ""), payload.get("value"), payload.get("data_type"), _coerce_bool(payload.get("confirm_write", False), False))
        elif action_id == "call_method":
            result = _call_method(str(payload.get("object_node_id") or ""), str(payload.get("method_node_id") or ""), list(payload.get("args") or []), _coerce_bool(payload.get("confirm_method", False), False))
        elif action_id == "create_subscription":
            result = _create_subscription(list(payload.get("node_ids") or []), payload.get("period_ms"))
        elif action_id == "delete_subscription":
            result = _delete_subscription(str(payload.get("subscription_id") or ""))
        elif action_id == "get_events":
            result = {"ok": True, "events": list(_STATE.get("events", [])), "count": len(_STATE.get("events", []))}
        elif action_id == "clear_events":
            _STATE["events"] = []
            result = {"ok": True, "cleared": True}
        elif action_id == "namespace_array":
            _ensure_connected()
            client = _STATE["client"]
            result = {"ok": True, "namespace_array": client.get_namespace_array()}
        elif action_id == "get_config":
            result = {"ok": True, "config": _mask_config(_STATE.get("config", {}))}
        elif action_id == "set_config":
            cfg = _STATE["config"].copy(); cfg.update(payload or {})
            _STATE["config"] = _normalize_config(cfg)
            result = {"ok": True, "config": _mask_config(_STATE["config"])}
        elif action_id == "safe_stop":
            result = _disconnect()
        else:
            result = {"ok": False, "error": f"Action '{action_id}' not implemented"}
    except RuntimeError as exc:
        result = {"ok": False, "error": str(exc)}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        result = {"ok": False, "error": type(exc).__name__, "details": str(exc)}
    _record(action_id, bool(result.get("ok")), result=result)
    return result
