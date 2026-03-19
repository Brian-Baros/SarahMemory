# SarahMemory Driver Package: com.softdev0.plc.modbus.tcp
# Name: PLC Modbus/TCP Driver
# Purpose: Governed Modbus/TCP read/write driver for SarahMemory AiOS.
# Notes:
# - Optional dependency: pymodbus
# - Writes are gated by read_only and explicit confirmation.
# - No auto-probing outside explicit actions.

from __future__ import annotations

import socket
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.plc.modbus.tcp"
DRIVER_NAME = "PLC Modbus/TCP Driver"
VERSION = "1.0.0"

try:
    from pymodbus.client import ModbusTcpClient as _PyModbusTcpClient  # type: ignore
except Exception:
    _PyModbusTcpClient = None

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "client_connected": False,
    "last_action": None,
    "last_result": None,
    "meta": {},
    "history": [],
    "client": None,
    "config": {},
}


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _record(action: str, ok: bool, detail: Any = None) -> None:
    _SESSION["last_action"] = action
    _SESSION["last_result"] = detail
    hist = _SESSION.setdefault("history", [])
    hist.append({"ts": _utc_now(), "action": action, "ok": bool(ok), "detail": detail})
    keep = int(_SESSION.get("config", {}).get("history_limit", 200) or 200)
    if len(hist) > keep:
        del hist[:-keep]


def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(cfg or {})
    for k in ("password", "auth_token", "api_key"):
        if masked.get(k):
            masked[k] = "***MASKED***"
    return masked


def _cfg(context: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {}
    if isinstance(_SESSION.get("config"), dict):
        base.update(_SESSION["config"])
    if isinstance(context, dict) and isinstance(context.get("config"), dict):
        base.update(context["config"])
    if isinstance(override, dict):
        base.update(override)
    return base


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _require_write_confirmation(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if _bool(cfg.get("read_only"), True):
        return {"ok": False, "error": "read_only_enabled"}
    if _bool(cfg.get("require_write_confirmation"), True) and not _bool(payload.get("confirm_write"), False):
        return {"ok": False, "error": "confirmation_required", "message": "Set confirm_write=true to perform write operations."}
    return None


def _client_call(client: Any, method_name: str, **kwargs: Any) -> Any:
    # Compatibility across pymodbus versions that use device_id, slave, or unit.
    method = getattr(client, method_name)
    last_exc = None
    for key in ("device_id", "slave", "unit"):
        call_kwargs = dict(kwargs)
        unit_value = call_kwargs.pop("unit_id", 1)
        call_kwargs[key] = unit_value
        try:
            return method(**call_kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return method(**kwargs)


def _is_error_response(resp: Any) -> bool:
    if resp is None:
        return True
    try:
        if hasattr(resp, "isError"):
            return bool(resp.isError())
    except Exception:
        pass
    return False


def _register_result(resp: Any) -> Dict[str, Any]:
    if resp is None:
        return {"ok": False, "error": "no_response"}
    if _is_error_response(resp):
        return {"ok": False, "error": str(resp)}
    data = {
        "ok": True,
        "bits": list(getattr(resp, "bits", []) or []),
        "registers": list(getattr(resp, "registers", []) or []),
    }
    for attr in ("address", "count", "function_code", "value"):
        if hasattr(resp, attr):
            data[attr] = getattr(resp, attr)
    return data


def _safe_socket_ping(host: str, port: int, timeout_sec: float) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_sec)
    started = time.time()
    try:
        s.connect((host, port))
        return {"ok": True, "reachable": True, "latency_ms": round((time.time() - started) * 1000, 2)}
    except Exception as exc:
        return {"ok": False, "reachable": False, "error": str(exc)}
    finally:
        try:
            s.close()
        except Exception:
            pass


def _make_client(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if _PyModbusTcpClient is None:
        return {"ok": False, "error": "pymodbus_not_installed"}
    host = str(cfg.get("host") or "").strip()
    if not host:
        return {"ok": False, "error": "missing_host"}
    port = _int(cfg.get("port"), 502)
    timeout_sec = _float(cfg.get("timeout_sec"), 5.0)
    try:
        client = _PyModbusTcpClient(host=host, port=port, timeout=timeout_sec)
        return {"ok": True, "client": client}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ensure_client(cfg: Dict[str, Any], connect: bool = True) -> Dict[str, Any]:
    client = _SESSION.get("client")
    if client is None:
        made = _make_client(cfg)
        if not made.get("ok"):
            return made
        client = made["client"]
        _SESSION["client"] = client
    if connect and not _SESSION.get("client_connected"):
        try:
            connected = bool(client.connect())
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        _SESSION["client_connected"] = connected
        if not connected:
            return {"ok": False, "error": "connect_failed"}
    return {"ok": True, "client": client}


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": {
            "instance_id": _SESSION.get("instance_id"),
            "started_ts": _SESSION.get("started_ts"),
            "status": _SESSION.get("status"),
            "error": _SESSION.get("error"),
            "client_connected": _SESSION.get("client_connected"),
            "last_action": _SESSION.get("last_action"),
        },
    }


def driver_validate(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    port = _int(config.get("port"), 502)
    unit_id = _int(config.get("unit_id"), 1)
    timeout_sec = _float(config.get("timeout_sec"), 5.0)
    if port < 1 or port > 65535:
        errors.append("port must be between 1 and 65535")
    if unit_id < 0 or unit_id > 255:
        errors.append("unit_id must be between 0 and 255")
    if timeout_sec <= 0:
        errors.append("timeout_sec must be > 0")
    if not str(config.get("host") or "").strip():
        warnings.append("host is empty; connect actions will fail until host is provided")
    if _bool(config.get("read_only"), True):
        warnings.append("read_only is enabled; write actions will be blocked")
    if _PyModbusTcpClient is None:
        warnings.append("pymodbus is not installed; only socket-level ping will work")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def driver_init(context: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    _SESSION["client_connected"] = False
    _SESSION["client"] = None
    _SESSION["config"] = dict(config)
    _SESSION["meta"] = {"context": context or {}, "config": _mask_config(config)}
    _record("init", True, {"validate": v})
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}


def driver_shutdown(context: Optional[Dict[str, Any]] = None) -> bool:
    client = _SESSION.get("client")
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["client_connected"] = False
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["client"] = None
    _SESSION["config"] = {}
    _record("shutdown", True, None)
    return True


def driver_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "session": {
            "instance_id": _SESSION.get("instance_id"),
            "started_ts": _SESSION.get("started_ts"),
            "status": _SESSION.get("status"),
            "error": _SESSION.get("error"),
            "client_connected": _SESSION.get("client_connected"),
            "config": _mask_config(_SESSION.get("config") or {}),
            "last_action": _SESSION.get("last_action"),
            "last_result": _SESSION.get("last_result"),
            "history_size": len(_SESSION.get("history", [])),
        },
    }


def driver_action(action_id: str, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _cfg(context, payload.get("config") if isinstance(payload.get("config"), dict) else None)
    action_id = str(action_id or "").strip()
    _SESSION["config"] = dict(cfg)

    try:
        if action_id == "ping":
            result = _safe_socket_ping(str(cfg.get("host") or ""), _int(cfg.get("port"), 502), _float(cfg.get("timeout_sec"), 5.0))
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "connect":
            ensured = _ensure_client(cfg, connect=True)
            if ensured.get("ok"):
                result = {"ok": True, "connected": True, "host": cfg.get("host"), "port": _int(cfg.get("port"), 502)}
                _SESSION["status"] = "connected"
            else:
                result = ensured
                _SESSION["status"] = "error"
                _SESSION["error"] = result.get("error")
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "disconnect":
            client = _SESSION.get("client")
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            _SESSION["client_connected"] = False
            _SESSION["status"] = "ready"
            result = {"ok": True, "connected": False}
            _record(action_id, True, result)
            return result

        if action_id == "get_config":
            result = {"ok": True, "config": _mask_config(cfg)}
            _record(action_id, True, result)
            return result

        if action_id == "set_config":
            if not isinstance(payload, dict):
                result = {"ok": False, "error": "payload_must_be_object"}
            else:
                merged = dict(_SESSION.get("config") or {})
                merged.update(payload.get("updates") or {})
                val = driver_validate(merged)
                if not val.get("ok"):
                    result = {"ok": False, "error": "validation_failed", "details": val}
                else:
                    _SESSION["config"] = merged
                    result = {"ok": True, "config": _mask_config(merged), "validate": val}
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "read_coils":
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "read_coils", address=_int(payload.get("address"), 0), count=_int(payload.get("count"), 1), unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "read_discrete_inputs":
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "read_discrete_inputs", address=_int(payload.get("address"), 0), count=_int(payload.get("count"), 1), unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "read_holding_registers":
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "read_holding_registers", address=_int(payload.get("address"), 0), count=_int(payload.get("count"), 1), unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "read_input_registers":
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "read_input_registers", address=_int(payload.get("address"), 0), count=_int(payload.get("count"), 1), unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "write_coil":
            blocked = _require_write_confirmation(cfg, payload)
            if blocked:
                _record(action_id, False, blocked)
                return blocked
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            value = _bool(payload.get("value"), False)
            resp = _client_call(ensured["client"], "write_coil", address=_int(payload.get("address"), 0), value=value, unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "write_register":
            blocked = _require_write_confirmation(cfg, payload)
            if blocked:
                _record(action_id, False, blocked)
                return blocked
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "write_register", address=_int(payload.get("address"), 0), value=_int(payload.get("value"), 0), unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "write_registers":
            blocked = _require_write_confirmation(cfg, payload)
            if blocked:
                _record(action_id, False, blocked)
                return blocked
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            values = payload.get("values")
            if not isinstance(values, list) or not values:
                result = {"ok": False, "error": "values_must_be_nonempty_list"}
                _record(action_id, False, result)
                return result
            resp = _client_call(ensured["client"], "write_registers", address=_int(payload.get("address"), 0), values=[_int(v, 0) for v in values], unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "write_coils":
            blocked = _require_write_confirmation(cfg, payload)
            if blocked:
                _record(action_id, False, blocked)
                return blocked
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            values = payload.get("values")
            if not isinstance(values, list) or not values:
                result = {"ok": False, "error": "values_must_be_nonempty_list"}
                _record(action_id, False, result)
                return result
            resp = _client_call(ensured["client"], "write_coils", address=_int(payload.get("address"), 0), values=[_bool(v, False) for v in values], unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "read_exception_status":
            ensured = _ensure_client(cfg, connect=True)
            if not ensured.get("ok"):
                _record(action_id, False, ensured)
                return ensured
            resp = _client_call(ensured["client"], "read_exception_status", unit_id=_int(payload.get("unit_id"), _int(cfg.get("unit_id"), 1)))
            result = _register_result(resp)
            _record(action_id, result.get("ok", False), result)
            return result

        if action_id == "safe_stop":
            client = _SESSION.get("client")
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            _SESSION["client_connected"] = False
            _SESSION["status"] = "ready"
            result = {"ok": True, "stopped": True}
            _record(action_id, True, result)
            return result

        result = {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
        _record(action_id, False, result)
        return result
    except Exception as exc:
        _SESSION["status"] = "error"
        _SESSION["error"] = str(exc)
        result = {"ok": False, "error": str(exc), "action_id": action_id}
        _record(action_id, False, result)
        return result
