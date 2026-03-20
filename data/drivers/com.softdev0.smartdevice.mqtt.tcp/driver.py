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
# com.softdev0.smartdevice.mqtt.tcp/driver.py
# Smart Device MQTT Driver
# Protocol-driven MQTT client for heterogeneous smart devices.

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from copy import deepcopy
from datetime import datetime
from fnmatch import fnmatchcase
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.smartdevice.mqtt.tcp"
_DRIVER_NAME = "Smart Device MQTT Driver"
_DRIVER_VERSION = "2.0.0"

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:
    mqtt = None

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "host": "",
    "port": 1883,
    "client_id": "",
    "username": "",
    "password": "",
    "keepalive": 60,
    "use_tls": False,
    "tls_insecure": False,
    "base_topic": "home/sarahmemory",
    "default_qos": 0,
    "default_retain": False,
    "auto_subscribe": True,
    "auto_subscribe_topics": ["homeassistant/#", "+/status", "+/state"],
    "listen_topics": [],
    "state_topic_suffix": "state",
    "command_topic_suffix": "set",
    "availability_topic_suffix": "availability",
    "discovery_prefix": "homeassistant",
    "allow_publish": True,
    "allow_raw_publish": False,
    "require_publish_confirmation": True,
    "allow_retained_publish": True,
    "allow_device_discovery_publish": True,
    "event_buffer_limit": 500,
    "device_buffer_limit": 200,
    "publish_timeout": 5.0,
}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "broker": None,
    "subscriptions": [],
    "last_publish": None,
    "last_message": None,
    "backend": "paho-mqtt" if mqtt else None,
}

_STATE: Dict[str, Any] = {
    "config": deepcopy(_DEFAULTS),
    "client": None,
    "event_buffer": [],
    "device_registry": {},
    "topic_state": {},
    "command_queue": None,
    "loop_thread": None,
    "stop_event": None,
}


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    if not merged.get("client_id"):
        merged["client_id"] = f"sarahmemory-{int(time.time())}"
    if not isinstance(merged.get("auto_subscribe_topics"), list):
        merged["auto_subscribe_topics"] = list(_DEFAULTS["auto_subscribe_topics"])
    if not isinstance(merged.get("listen_topics"), list):
        merged["listen_topics"] = []
    return merged


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + ("*" * max(0, len(value) - 4)) + value[-2:]


def _normalize_payload(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_payload(value: bytes) -> Any:
    try:
        text = value.decode("utf-8", errors="replace")
    except Exception:
        return {"raw_hex": value.hex()}
    text_s = text.strip()
    if not text_s:
        return ""
    try:
        return json.loads(text_s)
    except Exception:
        return text


def _trim_list(items: List[Any], limit: int) -> None:
    overflow = len(items) - max(1, int(limit))
    if overflow > 0:
        del items[0:overflow]


def _record_note(note: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append(f"{_utc_now()} {note}")
    _trim_list(notes, 50)


def _safe_topic_match(pattern: str, topic: str) -> bool:
    if pattern == topic:
        return True
    glob_pat = pattern.replace("#", "*").replace("+", "*")
    return fnmatchcase(topic, glob_pat)


def _topic_for_device(config: Dict[str, Any], device_id: str, suffix: str) -> str:
    base = str(config.get("base_topic") or "home/sarahmemory").strip("/")
    return f"{base}/{device_id}/{suffix.strip('/')}"


def _broker_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def driver_info() -> Dict[str, Any]:
    cfg = deepcopy(_STATE["config"])
    if cfg.get("password"):
        cfg["password"] = _mask_secret(str(cfg["password"]))
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "tcp",
        "session": dict(_SESSION),
        "config": cfg,
    }


def driver_validate(config: dict) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}

    checks = {
        "enabled": bool,
        "autoload": bool,
        "host": str,
        "client_id": str,
        "username": str,
        "password": str,
        "use_tls": bool,
        "tls_insecure": bool,
        "base_topic": str,
        "default_retain": bool,
        "auto_subscribe": bool,
        "allow_publish": bool,
        "allow_raw_publish": bool,
        "require_publish_confirmation": bool,
        "allow_retained_publish": bool,
        "allow_device_discovery_publish": bool,
    }
    for key, typ in checks.items():
        if key in config and not isinstance(config.get(key), typ):
            errors.append(f"{key} must be a {typ.__name__}")

    for key in ("port", "keepalive", "default_qos", "event_buffer_limit", "device_buffer_limit"):
        if key in config and not _is_number(config.get(key)):
            errors.append(f"{key} must be numeric")

    if "port" in config and _is_number(config.get("port")):
        port = int(config["port"])
        if port < 1 or port > 65535:
            errors.append("port must be in 1..65535")

    if "default_qos" in config and _is_number(config.get("default_qos")):
        qos = int(config["default_qos"])
        if qos not in (0, 1, 2):
            errors.append("default_qos must be 0, 1, or 2")

    if config.get("listen_topics") is not None and not isinstance(config.get("listen_topics"), list):
        errors.append("listen_topics must be a list")
    if config.get("auto_subscribe_topics") is not None and not isinstance(config.get("auto_subscribe_topics"), list):
        errors.append("auto_subscribe_topics must be a list")

    if not config.get("host"):
        warnings.append("host is empty; network connect actions will fail until configured")
    if mqtt is None:
        warnings.append("paho-mqtt is not installed; live MQTT actions are unavailable")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _reset_runtime(keep_config: bool = True) -> None:
    cfg = _STATE["config"] if keep_config else deepcopy(_DEFAULTS)
    _STATE.clear()
    _STATE.update(
        {
            "config": cfg,
            "client": None,
            "event_buffer": [],
            "device_registry": {},
            "topic_state": {},
            "command_queue": None,
            "loop_thread": None,
            "stop_event": None,
        }
    )


def _mqtt_on_connect(client, userdata, flags, reason_code, properties=None):
    _SESSION["connected"] = True
    _SESSION["status"] = "connected"
    _SESSION["error"] = None
    _record_note(f"Connected to MQTT broker rc={reason_code}")
    cfg = _STATE["config"]
    topics = []
    if cfg.get("auto_subscribe"):
        topics.extend(cfg.get("auto_subscribe_topics") or [])
    topics.extend(cfg.get("listen_topics") or [])
    unique: List[str] = []
    for t in topics:
        if isinstance(t, str) and t and t not in unique:
            unique.append(t)
    for topic in unique:
        try:
            client.subscribe(topic, qos=int(cfg.get("default_qos", 0)))
            if topic not in _SESSION["subscriptions"]:
                _SESSION["subscriptions"].append(topic)
        except Exception as exc:
            _record_note(f"Subscribe failed for {topic}: {exc}")


def _mqtt_on_disconnect(client, userdata, reason_code, properties=None):
    _SESSION["connected"] = False
    _SESSION["status"] = "ready"
    _record_note(f"Disconnected from MQTT broker rc={reason_code}")


def _maybe_register_device_from_topic(topic: str, payload: Any) -> None:
    if not topic:
        return
    parts = [p for p in topic.split("/") if p]
    if len(parts) < 2:
        return
    device_id = parts[-2] if parts[-1] in {"state", "status", "availability", "set", "cmd", "command"} else parts[-1]
    registry = _STATE["device_registry"]
    device = registry.setdefault(
        device_id,
        {
            "device_id": device_id,
            "topics": set(),
            "last_seen": None,
            "last_payload": None,
            "state": None,
        },
    )
    device["topics"].add(topic)
    device["last_seen"] = _utc_now()
    device["last_payload"] = payload
    if topic.endswith("/state") or topic.endswith("/status") or topic.endswith("/availability"):
        device["state"] = payload
    items = list(registry.items())
    limit = int(_STATE["config"].get("device_buffer_limit", 200))
    if len(items) > limit:
        for drop_id, _ in items[:-limit]:
            registry.pop(drop_id, None)


def _mqtt_on_message(client, userdata, message):
    payload = _decode_payload(bytes(message.payload))
    event = {
        "ts": _utc_now(),
        "topic": message.topic,
        "qos": int(getattr(message, "qos", 0)),
        "retain": bool(getattr(message, "retain", False)),
        "payload": payload,
    }
    _STATE["event_buffer"].append(event)
    _trim_list(_STATE["event_buffer"], int(_STATE["config"].get("event_buffer_limit", 500)))
    _STATE["topic_state"][message.topic] = payload
    _maybe_register_device_from_topic(message.topic, payload)
    _SESSION["last_message"] = event


def _build_client(config: Dict[str, Any]):
    if mqtt is None:
        raise RuntimeError("paho-mqtt backend is not installed")
    client = mqtt.Client(client_id=str(config.get("client_id") or ""), protocol=getattr(mqtt, "MQTTv311", None))
    if config.get("username"):
        client.username_pw_set(str(config.get("username") or ""), str(config.get("password") or ""))
    if config.get("use_tls"):
        client.tls_set()
        if config.get("tls_insecure"):
            client.tls_insecure_set(True)
    client.on_connect = _mqtt_on_connect
    client.on_disconnect = _mqtt_on_disconnect
    client.on_message = _mqtt_on_message
    return client


def _connect_client() -> Dict[str, Any]:
    cfg = _STATE["config"]
    host = str(cfg.get("host") or "").strip()
    port = int(cfg.get("port", 1883))
    if not host:
        return {"ok": False, "error": "host_required"}
    if mqtt is None:
        return {"ok": False, "error": "backend_unavailable", "details": "paho-mqtt not installed"}
    client = _build_client(cfg)
    try:
        client.connect(host, port, keepalive=int(cfg.get("keepalive", 60)))
        client.loop_start()
        _STATE["client"] = client
        _SESSION["broker"] = f"{host}:{port}"
        _SESSION["status"] = "connecting"
        return {"ok": True, "host": host, "port": port}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        _SESSION["status"] = "error"
        return {"ok": False, "error": "connect_failed", "details": str(exc)}


def _disconnect_client() -> Dict[str, Any]:
    client = _STATE.get("client")
    if client is None:
        return {"ok": True, "connected": False}
    try:
        client.loop_stop()
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass
    _STATE["client"] = None
    _SESSION["connected"] = False
    _SESSION["status"] = "ready"
    return {"ok": True, "connected": False}


def _publish(topic: str, payload: Any, qos: Optional[int] = None, retain: Optional[bool] = None) -> Dict[str, Any]:
    client = _STATE.get("client")
    if client is None:
        return {"ok": False, "error": "not_connected"}
    cfg = _STATE["config"]
    q = int(cfg.get("default_qos", 0) if qos is None else qos)
    r = bool(cfg.get("default_retain", False) if retain is None else retain)
    try:
        info = client.publish(topic, _normalize_payload(payload), qos=q, retain=r)
        timeout = float(cfg.get("publish_timeout", 5.0))
        try:
            info.wait_for_publish(timeout=timeout)
        except TypeError:
            info.wait_for_publish()
        event = {
            "ts": _utc_now(),
            "topic": topic,
            "qos": q,
            "retain": r,
            "payload": payload,
            "direction": "outbound",
        }
        _SESSION["last_publish"] = event
        return {"ok": True, "published": event}
    except Exception as exc:
        return {"ok": False, "error": "publish_failed", "details": str(exc)}


def _sanitize_registry() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for device_id, record in _STATE["device_registry"].items():
        cleaned = dict(record)
        if isinstance(cleaned.get("topics"), set):
            cleaned["topics"] = sorted(cleaned["topics"])
        out[device_id] = cleaned
    return out


def driver_init(context=None, config=None):
    merged = _merge_config(config)
    validation = driver_validate(merged)
    if not validation.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": validation}

    _reset_runtime(keep_config=False)
    _STATE["config"] = merged
    _SESSION.update(
        {
            "instance_id": _new_instance_id(),
            "started_ts": time.time(),
            "status": "ready",
            "error": None,
            "notes": [],
            "connected": False,
            "broker": None,
            "subscriptions": [],
            "last_publish": None,
            "last_message": None,
            "backend": "paho-mqtt" if mqtt else None,
        }
    )
    _record_note("MQTT smart-device driver initialized")
    if merged.get("auto_subscribe"):
        _record_note("Auto-subscribe enabled")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": validation, "info": driver_info()}


def driver_shutdown(context=None):
    _disconnect_client()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _record_note("Shutdown completed")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "connected": bool(_SESSION.get("connected")),
        "devices": _sanitize_registry(),
        "subscriptions": list(_SESSION.get("subscriptions") or []),
        "last_message": deepcopy(_SESSION.get("last_message")),
        "last_publish": deepcopy(_SESSION.get("last_publish")),
    }


def _require_publish_allowed(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cfg = _STATE["config"]
    if not cfg.get("allow_publish", True):
        return {"ok": False, "error": "publish_disabled"}
    if cfg.get("require_publish_confirmation", True) and not payload.get("confirm_publish"):
        return {"ok": False, "error": "publish_confirmation_required"}
    retain = payload.get("retain")
    if bool(retain) and not cfg.get("allow_retained_publish", True):
        return {"ok": False, "error": "retained_publish_disabled"}
    return None


def _publish_discovery(device_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _STATE["config"]
    if not cfg.get("allow_device_discovery_publish", True):
        return {"ok": False, "error": "discovery_publish_disabled"}
    component = str(payload.get("component") or "sensor")
    object_id = str(payload.get("object_id") or device_id)
    discovery_prefix = str(cfg.get("discovery_prefix") or "homeassistant").strip("/")
    topic = f"{discovery_prefix}/{component}/{device_id}/{object_id}/config"
    msg = dict(payload.get("config") or {})
    msg.setdefault("name", payload.get("name") or object_id)
    msg.setdefault("uniq_id", f"{device_id}_{object_id}")
    msg.setdefault("state_topic", _topic_for_device(cfg, device_id, cfg.get("state_topic_suffix", "state")))
    if payload.get("commandable"):
        msg.setdefault("command_topic", _topic_for_device(cfg, device_id, cfg.get("command_topic_suffix", "set")))
    msg.setdefault("availability_topic", _topic_for_device(cfg, device_id, cfg.get("availability_topic_suffix", "availability")))
    reg = _STATE["device_registry"].setdefault(device_id, {"device_id": device_id, "topics": set()})
    reg["discovery_topic"] = topic
    reg["last_seen"] = _utc_now()
    return _publish(topic, msg, qos=payload.get("qos"), retain=payload.get("retain", True))


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "payload_must_be_object"}

    if action_id == "ping":
        host = str(payload.get("host") or _STATE["config"].get("host") or "").strip()
        port = int(payload.get("port") or _STATE["config"].get("port") or 1883)
        if not host:
            return {"ok": True, "pong": True, "ts": time.time(), "reachable": False, "details": "host not configured"}
        return {"ok": True, "pong": True, "ts": time.time(), "reachable": _broker_reachable(host, port), "broker": f"{host}:{port}"}

    if action_id == "probe_backends":
        return {"ok": True, "mqtt_backend": "paho-mqtt" if mqtt else None, "available": bool(mqtt)}

    if action_id == "get_config":
        cfg = deepcopy(_STATE["config"])
        if cfg.get("password"):
            cfg["password"] = _mask_secret(str(cfg["password"]))
        return {"ok": True, "config": cfg}

    if action_id == "set_config":
        merged = _merge_config({**_STATE["config"], **payload})
        validation = driver_validate(merged)
        if not validation.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": validation}
        _STATE["config"] = merged
        return {"ok": True, "config": deepcopy(_STATE["config"]), "validate": validation}

    if action_id == "connect":
        return _connect_client()

    if action_id == "disconnect":
        return _disconnect_client()

    if action_id == "subscribe":
        topic = str(payload.get("topic") or "").strip()
        qos = int(payload.get("qos", _STATE["config"].get("default_qos", 0)))
        client = _STATE.get("client")
        if client is None:
            return {"ok": False, "error": "not_connected"}
        if not topic:
            return {"ok": False, "error": "topic_required"}
        try:
            client.subscribe(topic, qos=qos)
            if topic not in _SESSION["subscriptions"]:
                _SESSION["subscriptions"].append(topic)
            return {"ok": True, "topic": topic, "qos": qos}
        except Exception as exc:
            return {"ok": False, "error": "subscribe_failed", "details": str(exc)}

    if action_id == "unsubscribe":
        topic = str(payload.get("topic") or "").strip()
        client = _STATE.get("client")
        if client is None:
            return {"ok": False, "error": "not_connected"}
        if not topic:
            return {"ok": False, "error": "topic_required"}
        try:
            client.unsubscribe(topic)
            if topic in _SESSION["subscriptions"]:
                _SESSION["subscriptions"].remove(topic)
            return {"ok": True, "topic": topic}
        except Exception as exc:
            return {"ok": False, "error": "unsubscribe_failed", "details": str(exc)}

    if action_id == "publish":
        gate = _require_publish_allowed(payload)
        if gate:
            return gate
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "topic_required"}
        return _publish(topic, payload.get("message"), qos=payload.get("qos"), retain=payload.get("retain"))

    if action_id == "publish_raw":
        cfg = _STATE["config"]
        if not cfg.get("allow_raw_publish", False):
            return {"ok": False, "error": "raw_publish_disabled"}
        gate = _require_publish_allowed(payload)
        if gate:
            return gate
        topic = str(payload.get("topic") or "").strip()
        message = payload.get("bytes")
        if not topic:
            return {"ok": False, "error": "topic_required"}
        if isinstance(message, str):
            try:
                message = bytes.fromhex(message)
            except Exception:
                message = message.encode("utf-8")
        return _publish(topic, message, qos=payload.get("qos"), retain=payload.get("retain"))

    if action_id == "send_command":
        gate = _require_publish_allowed(payload)
        if gate:
            return gate
        device_id = str(payload.get("device_id") or "").strip()
        command = payload.get("command")
        if not device_id:
            return {"ok": False, "error": "device_id_required"}
        topic = str(payload.get("topic") or _topic_for_device(_STATE["config"], device_id, _STATE["config"].get("command_topic_suffix", "set")))
        return _publish(topic, command, qos=payload.get("qos"), retain=payload.get("retain"))

    if action_id == "publish_state":
        gate = _require_publish_allowed(payload)
        if gate:
            return gate
        device_id = str(payload.get("device_id") or "").strip()
        state = payload.get("state")
        if not device_id:
            return {"ok": False, "error": "device_id_required"}
        topic = str(payload.get("topic") or _topic_for_device(_STATE["config"], device_id, _STATE["config"].get("state_topic_suffix", "state")))
        return _publish(topic, state, qos=payload.get("qos"), retain=payload.get("retain"))

    if action_id == "publish_discovery":
        gate = _require_publish_allowed({**payload, "confirm_publish": payload.get("confirm_publish")})
        if gate:
            return gate
        device_id = str(payload.get("device_id") or "").strip()
        if not device_id:
            return {"ok": False, "error": "device_id_required"}
        return _publish_discovery(device_id, payload)

    if action_id == "list_devices":
        return {"ok": True, "devices": _sanitize_registry(), "count": len(_STATE["device_registry"])}

    if action_id == "get_topic_state":
        topic = str(payload.get("topic") or "").strip()
        if topic:
            return {"ok": True, "topic": topic, "state": deepcopy(_STATE["topic_state"].get(topic))}
        return {"ok": True, "topics": deepcopy(_STATE["topic_state"])}

    if action_id == "get_events":
        limit = int(payload.get("limit") or len(_STATE["event_buffer"]) or 0)
        if limit <= 0:
            return {"ok": True, "events": []}
        return {"ok": True, "events": deepcopy(_STATE["event_buffer"][-limit:])}

    if action_id == "clear_events":
        _STATE["event_buffer"].clear()
        return {"ok": True, "cleared": True}

    if action_id == "inject_message":
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "topic_required"}
        fake_payload = payload.get("message")
        event = {
            "ts": _utc_now(),
            "topic": topic,
            "qos": int(payload.get("qos", 0)),
            "retain": bool(payload.get("retain", False)),
            "payload": fake_payload,
            "direction": "simulated_inbound",
        }
        _STATE["event_buffer"].append(event)
        _trim_list(_STATE["event_buffer"], int(_STATE["config"].get("event_buffer_limit", 500)))
        _STATE["topic_state"][topic] = fake_payload
        _maybe_register_device_from_topic(topic, fake_payload)
        _SESSION["last_message"] = event
        return {"ok": True, "event": event}

    if action_id == "describe_capabilities":
        return {
            "ok": True,
            "protocol": "MQTT",
            "coverage": {
                "what_this_driver_can_do": [
                    "Broker connect/disconnect",
                    "Topic subscribe/unsubscribe",
                    "State/event buffering",
                    "Generic JSON/string/bytes publish",
                    "Device command/state topic conventions",
                    "Home Assistant discovery publish",
                ],
                "what_this_driver_cannot_guarantee": [
                    "Universal vendor semantic compatibility across all smart-device topic schemas",
                    "Automatic interpretation of every proprietary topic layout without mappings",
                    "Non-MQTT protocols such as Zigbee, Z-Wave, Matter, KNX, BLE, or vendor clouds",
                ],
            },
        }

    if action_id == "safe_stop":
        return _disconnect_client()

    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
