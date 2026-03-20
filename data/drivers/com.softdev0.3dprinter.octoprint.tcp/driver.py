"""
--==The SarahMemory Project==--
 File: ./data/drivers/com.softdev0.[DRIVERNAME].[TYPE]/driver.py
 Purpose: SarahMemory AiOS governed boot-layer bridge driver.
 Part of the SarahMemory Companion AI-bot Platform
 Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
 www.linkedin.com/in/brian-baros-29962a176
 https://www.facebook.com/bbaros
 brian.baros@sarahmemory.com
 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
 https://www.sarahmemory.com
 https://api.sarahmemory.com
 https://ai.sarahmemory.com
 https://store.sarahmemory.com
 ==============================================================================================
"""
# com.softdev0.3dprinter.octoprint.tcp/driver.py
# SarahMemory AiOS governed driver for 3D Printer (OctoPrint)
# Transport: tcp/http

import copy
import json
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional
from urllib import error, parse, request

_DRIVER_ID = "com.softdev0.3dprinter.octoprint.tcp"
_DRIVER_NAME = "3D Printer (OctoPrint) Driver"
_DRIVER_VERSION = "1.1.0"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_CONFIG = {
    "enabled": True,
    "autoload": False,
    "base_url": "",
    "api_key": "",
    "verify_tls": True,
    "timeout_sec": _DEFAULT_TIMEOUT,
    "auto_connect_on_init": False,
    "connection": {
        "port": "",
        "baudrate": 0,
        "printerProfile": "",
        "save": False,
        "autoconnect": False,
    },
    "safe_stop": {
        "cancel_job_first": True,
        "cooldown_hotend": True,
        "cooldown_bed": True,
        "disable_steppers": False,
        "park_head": False,
        "park_x": 0,
        "park_y": 0,
        "park_z_lift": 5,
    },
}
_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "last_seen_ts": None,
    "last_http_status": None,
    "printer_state": None,
    "job_state": None,
    "connected": False,
    "config_summary": {
        "enabled": True,
        "autoload": False,
        "base_url": "",
        "timeout_sec": _DEFAULT_TIMEOUT,
        "has_api_key": False,
        "auto_connect_on_init": False,
    },
    "last_status": None,
}
_RUNTIME = {
    "config": copy.deepcopy(_DEFAULT_CONFIG),
}


def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    if not isinstance(incoming, dict):
        return merged
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if isinstance(cfg.get("api_key"), str) and cfg.get("api_key"):
        cfg["api_key"] = "***masked***"
    return cfg


def _session_note(message: str):
    notes = _SESSION.setdefault("notes", [])
    notes.append(message)
    if len(notes) > 20:
        del notes[:-20]


def _set_status(status: str, error_text: Optional[str] = None):
    _SESSION["status"] = status
    _SESSION["error"] = error_text
    _SESSION["last_seen_ts"] = time.time()


def _normalize_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    return text.rstrip("/")


def _normalize_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except Exception:
        return _DEFAULT_TIMEOUT
    if timeout <= 0:
        return _DEFAULT_TIMEOUT
    return min(timeout, 120.0)


def _build_runtime_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _deep_merge(_DEFAULT_CONFIG, config or {})
    merged["base_url"] = _normalize_base_url(merged.get("base_url"))
    merged["timeout_sec"] = _normalize_timeout(merged.get("timeout_sec", _DEFAULT_TIMEOUT))
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["autoload"] = bool(merged.get("autoload", False))
    merged["verify_tls"] = bool(merged.get("verify_tls", True))
    merged["auto_connect_on_init"] = bool(merged.get("auto_connect_on_init", False))
    merged["api_key"] = str(merged.get("api_key") or "").strip()

    connection = merged.get("connection") or {}
    merged["connection"] = {
        "port": str(connection.get("port") or "").strip(),
        "baudrate": int(connection.get("baudrate") or 0),
        "printerProfile": str(connection.get("printerProfile") or "").strip(),
        "save": bool(connection.get("save", False)),
        "autoconnect": bool(connection.get("autoconnect", False)),
    }

    safe_stop = merged.get("safe_stop") or {}
    merged["safe_stop"] = {
        "cancel_job_first": bool(safe_stop.get("cancel_job_first", True)),
        "cooldown_hotend": bool(safe_stop.get("cooldown_hotend", True)),
        "cooldown_bed": bool(safe_stop.get("cooldown_bed", True)),
        "disable_steppers": bool(safe_stop.get("disable_steppers", False)),
        "park_head": bool(safe_stop.get("park_head", False)),
        "park_x": int(safe_stop.get("park_x") or 0),
        "park_y": int(safe_stop.get("park_y") or 0),
        "park_z_lift": int(safe_stop.get("park_z_lift") or 5),
    }
    return merged


def _update_config_summary(config: Dict[str, Any]):
    _SESSION["config_summary"] = {
        "enabled": bool(config.get("enabled", True)),
        "autoload": bool(config.get("autoload", False)),
        "base_url": config.get("base_url", ""),
        "timeout_sec": config.get("timeout_sec", _DEFAULT_TIMEOUT),
        "has_api_key": bool(config.get("api_key")),
        "auto_connect_on_init": bool(config.get("auto_connect_on_init", False)),
    }


def _headers() -> Dict[str, str]:
    cfg = _RUNTIME["config"]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"SarahMemoryAiOS/{_DRIVER_VERSION} ({_DRIVER_ID})",
    }
    if cfg.get("api_key"):
        headers["X-Api-Key"] = cfg["api_key"]
    return headers


def _api_url(path: str, query: Optional[Dict[str, Any]] = None) -> str:
    cfg = _RUNTIME["config"]
    base_url = cfg.get("base_url") or ""
    path = "/" + str(path or "").lstrip("/")
    url = f"{base_url}{path}"
    if query:
        clean = {k: v for k, v in query.items() if v is not None and v != ""}
        if clean:
            url += "?" + parse.urlencode(clean, doseq=True)
    return url


def _http_request(method: str, path: str, payload: Optional[Dict[str, Any]] = None, query: Optional[Dict[str, Any]] = None):
    cfg = _RUNTIME["config"]
    url = _api_url(path, query=query)
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=_headers(), method=method.upper())
    timeout = float(cfg.get("timeout_sec", _DEFAULT_TIMEOUT))
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read() or b""
            text = raw.decode("utf-8", errors="replace") if raw else ""
            parsed = None
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = text
            _SESSION["last_http_status"] = getattr(resp, "status", 200)
            return {
                "ok": True,
                "status_code": getattr(resp, "status", 200),
                "headers": dict(resp.headers.items()),
                "data": parsed,
                "text": text,
                "url": url,
            }
    except error.HTTPError as exc:
        raw = exc.read() or b""
        text = raw.decode("utf-8", errors="replace") if raw else ""
        parsed = None
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = text
        _SESSION["last_http_status"] = getattr(exc, "code", None)
        return {
            "ok": False,
            "status_code": getattr(exc, "code", None),
            "error": f"http_{getattr(exc, 'code', 'error')}",
            "message": text or str(exc),
            "data": parsed,
            "url": url,
        }
    except error.URLError as exc:
        _SESSION["last_http_status"] = None
        return {
            "ok": False,
            "status_code": None,
            "error": "network_error",
            "message": str(exc.reason) if hasattr(exc, "reason") else str(exc),
            "url": url,
        }
    except Exception as exc:
        _SESSION["last_http_status"] = None
        return {
            "ok": False,
            "status_code": None,
            "error": "request_failed",
            "message": str(exc),
            "url": url,
        }


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(result.get("ok")),
        "status_code": result.get("status_code"),
        "error": result.get("error"),
        "message": result.get("message"),
        "url": result.get("url"),
    }


def _fetch_status_bundle() -> Dict[str, Any]:
    version = _http_request("GET", "/api/version")
    if not version.get("ok"):
        _set_status("error", version.get("error") or "version_check_failed")
        _SESSION["connected"] = False
        return {
            "ok": False,
            "error": version.get("error") or "version_check_failed",
            "version": _compact_result(version),
        }

    server = _http_request("GET", "/api/server")
    connection = _http_request("GET", "/api/connection")
    printer = _http_request("GET", "/api/printer", query={"exclude": "temperature,sd"})
    job = _http_request("GET", "/api/job")

    printer_data = printer.get("data") if printer.get("ok") and isinstance(printer.get("data"), dict) else {}
    job_data = job.get("data") if job.get("ok") and isinstance(job.get("data"), dict) else {}
    connection_data = connection.get("data") if connection.get("ok") and isinstance(connection.get("data"), dict) else {}

    printer_state = None
    if isinstance(printer_data.get("state"), dict):
        printer_state = printer_data["state"].get("text") or printer_data["state"].get("flags")
    if not printer_state:
        printer_state = job_data.get("state") or connection_data.get("current", {}).get("state")

    connected = False
    current = connection_data.get("current") if isinstance(connection_data.get("current"), dict) else {}
    current_state = str(current.get("state") or "")
    if current_state:
        connected = current_state.lower() not in {"closed", "offline", "error", "unknown"}
    elif isinstance(printer_data.get("state"), dict):
        flags = printer_data["state"].get("flags") or {}
        connected = bool(flags.get("operational") or flags.get("printing") or flags.get("paused"))

    bundle = {
        "ok": True,
        "ts": time.time(),
        "version": version.get("data"),
        "server": server.get("data") if server.get("ok") else _compact_result(server),
        "connection": connection_data if connection.get("ok") else _compact_result(connection),
        "printer": printer_data if printer.get("ok") else _compact_result(printer),
        "job": job_data if job.get("ok") else _compact_result(job),
        "printer_state": printer_state,
        "job_state": job_data.get("state"),
        "connected": connected,
    }

    _SESSION["last_status"] = bundle
    _SESSION["printer_state"] = bundle.get("printer_state")
    _SESSION["job_state"] = bundle.get("job_state")
    _SESSION["connected"] = bundle.get("connected", False)
    _set_status("ready" if connected else "degraded", None)
    return bundle


def _job_command(command: str, action: Optional[str] = None):
    payload = {"command": command}
    if action:
        payload["action"] = action
    return _http_request("POST", "/api/job", payload=payload)


def _connection_command(command: str, extra: Optional[Dict[str, Any]] = None):
    payload = {"command": command}
    if extra:
        payload.update(extra)
    return _http_request("POST", "/api/connection", payload=payload)


def _send_printer_commands(commands: Iterable[str]):
    clean = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
    if not clean:
        return {"ok": False, "error": "no_commands", "message": "No printer commands supplied"}
    payload = {"commands": clean}
    return _http_request("POST", "/api/printer/command", payload=payload)


def _safe_stop(payload: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    safe_cfg = copy.deepcopy(_RUNTIME["config"].get("safe_stop") or {})
    if isinstance(payload.get("safe_stop"), dict):
        safe_cfg = _deep_merge(safe_cfg, payload["safe_stop"])

    status = _fetch_status_bundle()
    if not status.get("ok"):
        return status

    steps = []
    job_state = str(status.get("job_state") or "").lower()
    should_cancel = bool(safe_cfg.get("cancel_job_first", True)) and any(
        token in job_state for token in ("printing", "paused", "pausing", "resuming", "cancelling")
    )
    if should_cancel:
        cancel_result = _job_command("cancel")
        steps.append({"step": "cancel_job", **_compact_result(cancel_result)})

    gcode = []
    if safe_cfg.get("park_head"):
        z_lift = int(safe_cfg.get("park_z_lift", 5))
        park_x = int(safe_cfg.get("park_x", 0))
        park_y = int(safe_cfg.get("park_y", 0))
        gcode.extend([
            "G91",
            f"G1 Z{z_lift} F3000",
            "G90",
            f"G1 X{park_x} Y{park_y} F6000",
        ])
    if safe_cfg.get("cooldown_hotend", True):
        gcode.append("M104 S0")
    if safe_cfg.get("cooldown_bed", True):
        gcode.append("M140 S0")
    if safe_cfg.get("disable_steppers", False):
        gcode.append("M84")

    command_result = None
    if gcode:
        command_result = _send_printer_commands(gcode)
        steps.append({"step": "printer_command", **_compact_result(command_result), "commands": gcode})

    refreshed = _fetch_status_bundle()
    return {
        "ok": True,
        "action": "safe_stop",
        "steps": steps,
        "status": refreshed,
    }


def _connect_printer(payload: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    cfg = _RUNTIME["config"]
    base = copy.deepcopy(cfg.get("connection") or {})
    if isinstance(payload.get("connection"), dict):
        base = _deep_merge(base, payload["connection"])
    extra = {
        "port": str(base.get("port") or "").strip(),
        "baudrate": int(base.get("baudrate") or 0),
        "printerProfile": str(base.get("printerProfile") or "").strip(),
        "save": bool(base.get("save", False)),
        "autoconnect": bool(base.get("autoconnect", False)),
    }
    clean = {k: v for k, v in extra.items() if v not in ("", 0, None)}
    clean["save"] = extra["save"]
    clean["autoconnect"] = extra["autoconnect"]
    result = _connection_command("connect", clean)
    if not result.get("ok"):
        _set_status("error", result.get("error") or "connect_failed")
        return {"ok": False, "error": result.get("error") or "connect_failed", "details": result}
    _session_note("Connection command accepted by OctoPrint")
    return {"ok": True, "action": "connect", "details": _compact_result(result), "status": _fetch_status_bundle()}


def _disconnect_printer():
    result = _connection_command("disconnect")
    if not result.get("ok"):
        _set_status("error", result.get("error") or "disconnect_failed")
        return {"ok": False, "error": result.get("error") or "disconnect_failed", "details": result}
    _session_note("Disconnect command accepted by OctoPrint")
    return {"ok": True, "action": "disconnect", "details": _compact_result(result), "status": _fetch_status_bundle()}


def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "tcp/http",
        "session": copy.deepcopy(_SESSION),
        "config": _mask_config(_RUNTIME["config"]),
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}

    if "enabled" in config and not isinstance(config.get("enabled"), bool):
        errors.append("enabled must be a boolean")
    if "autoload" in config and not isinstance(config.get("autoload"), bool):
        errors.append("autoload must be a boolean")
    if "base_url" in config:
        if not isinstance(config.get("base_url"), str) or not config.get("base_url", "").strip():
            warnings.append("base_url is empty; network calls will not be available until configured")
        elif not str(config.get("base_url")).strip().lower().startswith(("http://", "https://")):
            errors.append("base_url must start with http:// or https://")
    else:
        warnings.append("base_url not set; network calls will not be available until configured")

    if "api_key" in config and config.get("api_key") is not None and not isinstance(config.get("api_key"), str):
        errors.append("api_key must be a string")
    elif not str(config.get("api_key") or "").strip():
        warnings.append("api_key is empty; authenticated OctoPrint endpoints will fail until configured")

    if "timeout_sec" in config:
        try:
            timeout = float(config.get("timeout_sec"))
            if timeout <= 0:
                errors.append("timeout_sec must be greater than 0")
        except Exception:
            errors.append("timeout_sec must be numeric")

    if "connection" in config and not isinstance(config.get("connection"), dict):
        errors.append("connection must be a JSON object when provided")
    if "safe_stop" in config and not isinstance(config.get("safe_stop"), dict):
        errors.append("safe_stop must be a JSON object when provided")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def driver_init(context=None, config=None):
    config = config or {}
    runtime_cfg = _build_runtime_config(config)
    v = driver_validate(runtime_cfg)
    _RUNTIME["config"] = runtime_cfg
    _update_config_summary(runtime_cfg)

    if not v.get("ok"):
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
        _SESSION["printer_state"] = None
        _SESSION["job_state"] = None
        _SESSION["connected"] = False
        _set_status("error", "validation_failed")
        _session_note("Initialization blocked by configuration validation failure")
        return {"ok": False, "error": "validation_failed", "details": v, "info": driver_info()}

    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["printer_state"] = None
    _SESSION["job_state"] = None
    _SESSION["connected"] = False
    _set_status("ready", None)
    _SESSION["notes"] = []
    _session_note("Driver initialized")

    status = None
    if runtime_cfg.get("base_url") and runtime_cfg.get("api_key"):
        status = _fetch_status_bundle()
        if status.get("ok"):
            _session_note("OctoPrint endpoint reachable")
        else:
            _session_note("OctoPrint endpoint not reachable during init")
        if runtime_cfg.get("auto_connect_on_init"):
            connect_result = _connect_printer({"connection": runtime_cfg.get("connection") or {}})
            status = connect_result.get("status") if connect_result.get("ok") else status
    else:
        _set_status("degraded", None)
        _session_note("Driver initialized without full network credentials")

    return {
        "ok": True,
        "instance_id": _SESSION["instance_id"],
        "validate": v,
        "status": status,
        "info": driver_info(),
    }


def driver_shutdown(context=None):
    _set_status("stopped", None)
    _session_note("Shutdown completed")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["printer_state"] = None
    _SESSION["job_state"] = None
    _SESSION["connected"] = False
    return True


def driver_status(context=None):
    if _RUNTIME["config"].get("base_url") and _RUNTIME["config"].get("api_key") and _SESSION["status"] != "stopped":
        live = _fetch_status_bundle()
        return {"ok": live.get("ok", False), "session": copy.deepcopy(_SESSION), "live": live, "info": driver_info()}
    return {"ok": True, "session": copy.deepcopy(_SESSION), "live": _SESSION.get("last_status"), "info": driver_info()}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    action_id = str(action_id or "").strip().lower()

    if action_id == "ping":
        if _RUNTIME["config"].get("base_url") and _RUNTIME["config"].get("api_key"):
            version = _http_request("GET", "/api/version")
            if version.get("ok"):
                _set_status("ready", None)
                return {"ok": True, "pong": True, "ts": time.time(), "version": version.get("data")}
            _set_status("error", version.get("error") or "ping_failed")
            return {"ok": False, "error": version.get("error") or "ping_failed", "details": version}
        return {"ok": True, "pong": True, "ts": time.time(), "mode": "local_config_only"}

    if action_id == "status":
        return driver_status(context=context)

    if action_id == "connect":
        return _connect_printer(payload)

    if action_id == "disconnect":
        return _disconnect_printer()

    if action_id == "safe_stop":
        return _safe_stop(payload)

    if action_id == "cancel_job":
        result = _job_command("cancel")
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "start_job":
        result = _job_command("start")
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "pause_job":
        result = _job_command("pause", "pause")
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "resume_job":
        result = _job_command("pause", "resume")
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "restart_job":
        result = _job_command("restart")
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "send_gcode":
        commands = payload.get("commands")
        if isinstance(commands, str):
            commands = [commands]
        result = _send_printer_commands(commands or [])
        return {"ok": result.get("ok", False), "action": action_id, "details": result, "status": _fetch_status_bundle()}

    if action_id == "version":
        result = _http_request("GET", "/api/version")
        return {"ok": result.get("ok", False), "action": action_id, "details": result}

    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
