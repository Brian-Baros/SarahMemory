# com.softdev0.ir.lirc.gpio/driver.py
# Governed IR Receiver (LIRC/GPIO) Driver
# Transport: gpio / lirc / unix-socket

from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.ir.lirc.gpio"
_DRIVER_NAME = "IR Receiver (LIRC/GPIO) Driver"
_DRIVER_VERSION = "1.0.0"

_LOCK = threading.RLock()
_LISTENER_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()
_EVENT_BUFFER = deque(maxlen=500)
_CHILD_PROCS: Dict[str, subprocess.Popen] = {}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "listener_running": False,
    "last_event_ts": None,
    "last_result": None,
    "selected_device": None,
    "tool_probe": {},
}

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "gpio_pin": 18,
    "driver": "default",
    "device": "/dev/lirc0",
    "socket_path": "/var/run/lirc/lircd",
    "socket_path_fallbacks": ["/run/lirc/lircd", "/var/run/lirc/lircd", "/dev/lircd"],
    "lirc_conf": "/etc/lirc/lircd.conf",
    "lirc_conf_dir": "/etc/lirc/lircd.conf.d",
    "lirc_options": "/etc/lirc/lirc_options.conf",
    "plugin_dir": "/usr/lib/lirc/plugins",
    "allow_root_keep": False,
    "listen_timeout_ms": 250,
    "mode2_capture_seconds": 5,
    "mode2_gap_us": 10000,
    "event_buffer_size": 500,
    "learn_timeout_seconds": 45,
    "irrecord_force_raw": True,
    "irrecord_disable_namespace": False,
    "allow_daemon_control": False,
    "allow_shell_commands": False,
    "use_irw": False,
}

_CONFIG: Dict[str, Any] = dict(_DEFAULTS)


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(cfg)
    return masked


def _note(message: str) -> None:
    with _LOCK:
        notes = _SESSION.setdefault("notes", [])
        notes.append(message)
        if len(notes) > 50:
            del notes[:-50]


def _set_last_result(result: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        _SESSION["last_result"] = result
    return result


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _probe_tools() -> Dict[str, Any]:
    tools = {
        "mode2": _which("mode2"),
        "irrecord": _which("irrecord"),
        "irw": _which("irw"),
        "ir-ctl": _which("ir-ctl"),
        "lircd": _which("lircd"),
        "systemctl": _which("systemctl"),
        "service": _which("service"),
        "pkill": _which("pkill"),
    }
    probe = {
        "ok": True,
        "tools": {k: {"found": bool(v), "path": v} for k, v in tools.items()},
    }
    with _LOCK:
        _SESSION["tool_probe"] = probe
    return probe


def driver_info() -> Dict[str, Any]:
    with _LOCK:
        session = dict(_SESSION)
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "gpio",
        "session": session,
        "config": _mask_config(dict(_CONFIG)),
    }


def driver_validate(config: dict) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}

    bool_keys = [
        "enabled",
        "autoload",
        "allow_root_keep",
        "irrecord_force_raw",
        "irrecord_disable_namespace",
        "allow_daemon_control",
        "allow_shell_commands",
        "use_irw",
    ]
    int_keys = [
        "gpio_pin",
        "listen_timeout_ms",
        "mode2_capture_seconds",
        "mode2_gap_us",
        "event_buffer_size",
        "learn_timeout_seconds",
    ]
    str_keys = [
        "driver",
        "device",
        "socket_path",
        "lirc_conf",
        "lirc_conf_dir",
        "lirc_options",
        "plugin_dir",
    ]

    for key in bool_keys:
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")
    for key in int_keys:
        if key in config and not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer")
    for key in str_keys:
        if key in config and not isinstance(config.get(key), str):
            errors.append(f"{key} must be a string")
    if "socket_path_fallbacks" in config and not isinstance(config.get("socket_path_fallbacks"), list):
        errors.append("socket_path_fallbacks must be a list")
    if isinstance(config.get("gpio_pin"), int):
        if config["gpio_pin"] < 0 or config["gpio_pin"] > 40:
            errors.append("gpio_pin must be between 0 and 40")
    if isinstance(config.get("event_buffer_size"), int) and config["event_buffer_size"] < 1:
        errors.append("event_buffer_size must be >= 1")
    if not config.get("device"):
        warnings.append("device not set; /dev/lirc0 will be used by default")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(_DEFAULTS)
    merged.update(_CONFIG)
    if config:
        merged.update(config)
    return merged


def _resize_buffer(maxlen: int) -> None:
    global _EVENT_BUFFER
    current = list(_EVENT_BUFFER)
    _EVENT_BUFFER = deque(current[-maxlen:], maxlen=maxlen)


def driver_init(context=None, config=None):
    merged = _merge_config(config or {})
    v = driver_validate(merged)
    if not v.get("ok"):
        with _LOCK:
            _SESSION["status"] = "error"
            _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    with _LOCK:
        _CONFIG.clear()
        _CONFIG.update(merged)
        _resize_buffer(int(_CONFIG.get("event_buffer_size", 500)))
        _SESSION["instance_id"] = _new_instance_id()
        _SESSION["started_ts"] = time.time()
        _SESSION["status"] = "ready"
        _SESSION["error"] = None
        _SESSION["notes"] = []
        _SESSION["selected_device"] = _CONFIG.get("device")

    probe = _probe_tools()
    _note("IR LIRC/GPIO driver initialized.")
    return _set_last_result({
        "ok": True,
        "instance_id": _SESSION["instance_id"],
        "validate": v,
        "tool_probe": probe,
        "info": driver_info(),
    })


def _terminate_proc(key: str) -> None:
    proc = _CHILD_PROCS.pop(key, None)
    if not proc:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def driver_shutdown(context=None):
    _stop_listener_internal()
    for key in list(_CHILD_PROCS.keys()):
        _terminate_proc(key)
    with _LOCK:
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["notes"] = ["Shutdown completed."]
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
        _SESSION["listener_running"] = False
    return True


def driver_status(context=None):
    probe = _probe_tools()
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _mask_config(dict(_CONFIG)),
        "tool_probe": probe,
        "buffer": {
            "count": len(_EVENT_BUFFER),
            "max": _EVENT_BUFFER.maxlen,
        },
    }


def _run_cmd(args: List[str], timeout: int = 15, allow_fail: bool = True, stdin_text: Optional[str] = None) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "cmd": args,
        }
        if not allow_fail and completed.returncode != 0:
            result["error"] = "command_failed"
        return result
    except FileNotFoundError:
        return {"ok": False, "error": "tool_not_found", "cmd": args}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": "timeout",
            "cmd": args,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def _discover_devices() -> Dict[str, Any]:
    devices: List[Dict[str, Any]] = []
    for path in sorted([p for p in ["/dev/lirc0", "/dev/lirc1", "/dev/lirc2", "/dev/lirc3"] if os.path.exists(p)]):
        devices.append({"path": path, "kind": "lirc_char_device"})

    mode2_path = _which("mode2")
    if mode2_path:
        res = _run_cmd([mode2_path, "--driver", _CONFIG.get("driver", "default"), "--list-devices"], timeout=10)
        if res.get("stdout"):
            devices.append({"source": "mode2", "listing": res["stdout"]})
    irctl_path = _which("ir-ctl")
    if irctl_path:
        res = _run_cmd([irctl_path, "--features"], timeout=10)
        devices.append({"source": "ir-ctl", "features": res})

    selected = _CONFIG.get("device")
    with _LOCK:
        _SESSION["selected_device"] = selected
    return {"ok": True, "devices": devices, "selected_device": selected}


def _resolve_socket_path() -> Optional[str]:
    candidates = [_CONFIG.get("socket_path")] + list(_CONFIG.get("socket_path_fallbacks", []))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _check_lircd() -> Dict[str, Any]:
    socket_path = _resolve_socket_path()
    result = {
        "ok": bool(socket_path),
        "socket_path": socket_path,
        "socket_exists": bool(socket_path),
        "lircd_found": bool(_which("lircd")),
    }
    if _which("systemctl"):
        svc = _run_cmd([_which("systemctl"), "is-active", "lircd"], timeout=8)
        result["systemctl"] = svc
    return result


def _read_lircd_socket_once(timeout_ms: int = 1000) -> Dict[str, Any]:
    socket_path = _resolve_socket_path()
    if not socket_path:
        return {"ok": False, "error": "lircd_socket_not_found"}
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(max(timeout_ms / 1000.0, 0.2))
        client.connect(socket_path)
        try:
            data = client.recv(4096)
        finally:
            client.close()
        text = data.decode("utf-8", errors="replace")
        events = _parse_lircd_lines(text.splitlines())
        return {"ok": True, "socket_path": socket_path, "raw": text, "events": events}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "socket_path": socket_path}


def _parse_lircd_lines(lines: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        event = {
            "raw": line,
            "ts": time.time(),
        }
        if len(parts) >= 3:
            event.update({
                "hex": parts[0],
                "repeat": parts[1],
                "button": parts[2],
                "remote": parts[3] if len(parts) >= 4 else None,
            })
        events.append(event)
    return events


def _listener_loop() -> None:
    socket_path = _resolve_socket_path()
    if not socket_path:
        with _LOCK:
            _SESSION["listener_running"] = False
            _SESSION["status"] = "ready"
            _SESSION["error"] = "lircd_socket_not_found"
        return

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.setblocking(False)
    try:
        client.connect(socket_path)
    except BlockingIOError:
        pass
    except Exception:
        try:
            client.close()
        finally:
            with _LOCK:
                _SESSION["listener_running"] = False
                _SESSION["status"] = "ready"
                _SESSION["error"] = "listener_connect_failed"
            return

    with _LOCK:
        _SESSION["listener_running"] = True
        _SESSION["status"] = "listening"
        _SESSION["error"] = None

    buffer = b""
    timeout = max(int(_CONFIG.get("listen_timeout_ms", 250)), 50) / 1000.0
    try:
        while not _STOP_EVENT.is_set():
            rlist, _, _ = select.select([client], [], [], timeout)
            if not rlist:
                continue
            data = client.recv(4096)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                decoded = line.decode("utf-8", errors="replace").strip()
                events = _parse_lircd_lines([decoded])
                for event in events:
                    _EVENT_BUFFER.append(event)
                    with _LOCK:
                        _SESSION["last_event_ts"] = event["ts"]
    except Exception as exc:
        with _LOCK:
            _SESSION["error"] = str(exc)
    finally:
        try:
            client.close()
        except Exception:
            pass
        with _LOCK:
            _SESSION["listener_running"] = False
            if _SESSION.get("status") != "stopped":
                _SESSION["status"] = "ready"


def _start_listener() -> Dict[str, Any]:
    global _LISTENER_THREAD
    with _LOCK:
        if _SESSION.get("listener_running"):
            return {"ok": True, "already_running": True}
    socket_path = _resolve_socket_path()
    if not socket_path:
        return {"ok": False, "error": "lircd_socket_not_found"}
    _STOP_EVENT.clear()
    _LISTENER_THREAD = threading.Thread(target=_listener_loop, name="LIRCListener", daemon=True)
    _LISTENER_THREAD.start()
    return {"ok": True, "socket_path": socket_path, "listener_started": True}


def _stop_listener_internal() -> None:
    global _LISTENER_THREAD
    _STOP_EVENT.set()
    if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
        _LISTENER_THREAD.join(timeout=2.0)
    _LISTENER_THREAD = None
    with _LOCK:
        _SESSION["listener_running"] = False
        if _SESSION.get("status") == "listening":
            _SESSION["status"] = "ready"


def _stop_listener() -> Dict[str, Any]:
    _stop_listener_internal()
    return {"ok": True, "listener_running": False}


def _mode2_capture(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode2_path = _which("mode2")
    if not mode2_path:
        return {"ok": False, "error": "mode2_not_found"}
    driver_name = payload.get("driver") or _CONFIG.get("driver") or "default"
    device = payload.get("device") or _CONFIG.get("device") or "/dev/lirc0"
    seconds = int(payload.get("seconds") or _CONFIG.get("mode2_capture_seconds", 5))
    gap = int(payload.get("gap_us") or _CONFIG.get("mode2_gap_us", 10000))
    args = [mode2_path, "--driver", driver_name, "--device", device, "--gap", str(gap)]
    if _CONFIG.get("allow_root_keep"):
        args.append("--keep-root")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _CHILD_PROCS["mode2"] = proc
    try:
        time.sleep(max(seconds, 1))
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=3)
    except Exception:
        try:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=2)
        except Exception:
            stdout, stderr = "", ""
    finally:
        _CHILD_PROCS.pop("mode2", None)
    lines = [line for line in stdout.splitlines() if line.strip()]
    return {
        "ok": proc.returncode in (0, 130),
        "cmd": args,
        "stdout": stdout,
        "stderr": stderr,
        "line_count": len(lines),
        "sample": lines[:100],
    }


def _irrecord_learn(payload: Dict[str, Any]) -> Dict[str, Any]:
    irrecord_path = _which("irrecord")
    if not irrecord_path:
        return {"ok": False, "error": "irrecord_not_found"}
    output_path = payload.get("output_path") or payload.get("file") or "./irrecord.lircd.conf"
    device = payload.get("device") or _CONFIG.get("device") or "/dev/lirc0"
    driver_name = payload.get("driver") or _CONFIG.get("driver") or "default"
    timeout_s = int(payload.get("timeout_seconds") or _CONFIG.get("learn_timeout_seconds", 45))

    args = [irrecord_path, "--driver", driver_name, "--device", device]
    if bool(payload.get("force_raw", _CONFIG.get("irrecord_force_raw", True))):
        args.append("--force")
    if bool(payload.get("disable_namespace", _CONFIG.get("irrecord_disable_namespace", False))):
        args.append("--disable-namespace")
    if _CONFIG.get("allow_root_keep"):
        args.append("--keep-root")
    args.append(output_path)

    res = _run_cmd(args, timeout=timeout_s, allow_fail=True)
    res["output_path"] = output_path
    if os.path.exists(output_path):
        try:
            res["output_exists"] = True
            res["output_size"] = os.path.getsize(output_path)
        except Exception:
            res["output_exists"] = True
    else:
        res["output_exists"] = False
    return res


def _irrecord_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    irrecord_path = _which("irrecord")
    if not irrecord_path:
        return {"ok": False, "error": "irrecord_not_found"}
    file_path = payload.get("file") or payload.get("input_path")
    if not file_path:
        return {"ok": False, "error": "file_required"}
    return _run_cmd([irrecord_path, "--analyse", file_path], timeout=30)


def _reload_lircd() -> Dict[str, Any]:
    lircd_path = _which("lircd")
    if not lircd_path:
        return {"ok": False, "error": "lircd_not_found"}
    if not _CONFIG.get("allow_daemon_control"):
        return {"ok": False, "error": "daemon_control_disabled"}
    systemctl = _which("systemctl")
    if systemctl:
        return _run_cmd([systemctl, "reload", "lircd"], timeout=15)
    return {"ok": False, "error": "no_reload_method_available"}


def _read_events(payload: Dict[str, Any]) -> Dict[str, Any]:
    limit = int(payload.get("limit", 100))
    items = list(_EVENT_BUFFER)[-max(limit, 1):]
    return {"ok": True, "events": items, "count": len(items)}


def _clear_events() -> Dict[str, Any]:
    _EVENT_BUFFER.clear()
    return {"ok": True, "cleared": True}


def _list_namespace() -> Dict[str, Any]:
    irrecord_path = _which("irrecord")
    if not irrecord_path:
        return {"ok": False, "error": "irrecord_not_found"}
    return _run_cmd([irrecord_path, "--list-namespace"], timeout=15)


def _send_shell_command(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _CONFIG.get("allow_shell_commands"):
        return {"ok": False, "error": "shell_commands_disabled"}
    cmd = payload.get("command")
    if not cmd or not isinstance(cmd, str):
        return {"ok": False, "error": "command_required"}
    return _run_cmd(shlex.split(cmd), timeout=int(payload.get("timeout", 15)))


def _safe_stop() -> Dict[str, Any]:
    _stop_listener_internal()
    _terminate_proc("mode2")
    return {"ok": True, "safe_stop": True}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    action_id = (action_id or "").strip()

    if action_id == "ping":
        return _set_last_result({"ok": True, "pong": True, "ts": time.time()})
    if action_id in {"probe_tools", "tool_probe"}:
        return _set_last_result(_probe_tools())
    if action_id in {"discover_devices", "list_devices", "list_receivers"}:
        return _set_last_result(_discover_devices())
    if action_id in {"check_lircd", "lircd_status"}:
        return _set_last_result(_check_lircd())
    if action_id == "read_socket_once":
        return _set_last_result(_read_lircd_socket_once(int(payload.get("timeout_ms", 1000))))
    if action_id in {"start_listen", "listen_start", "start_listener"}:
        return _set_last_result(_start_listener())
    if action_id in {"stop_listen", "listen_stop", "stop_listener"}:
        return _set_last_result(_stop_listener())
    if action_id == "get_events":
        return _set_last_result(_read_events(payload))
    if action_id == "clear_events":
        return _set_last_result(_clear_events())
    if action_id in {"mode2_capture", "capture_raw", "capture_signal"}:
        return _set_last_result(_mode2_capture(payload))
    if action_id in {"learn", "learn_remote", "irrecord_learn"}:
        return _set_last_result(_irrecord_learn(payload))
    if action_id in {"analyse_config", "irrecord_analyse", "irrecord_analyze"}:
        return _set_last_result(_irrecord_analyze(payload))
    if action_id == "reload_lircd":
        return _set_last_result(_reload_lircd())
    if action_id in {"list_namespace", "list_buttons_namespace"}:
        return _set_last_result(_list_namespace())
    if action_id == "set_config":
        merged = dict(_CONFIG)
        merged.update(payload or {})
        validation = driver_validate(merged)
        if not validation.get("ok"):
            return _set_last_result({"ok": False, "error": "validation_failed", "details": validation})
        with _LOCK:
            _CONFIG.clear()
            _CONFIG.update(merged)
            _resize_buffer(int(_CONFIG.get("event_buffer_size", 500)))
        return _set_last_result({"ok": True, "config": _mask_config(dict(_CONFIG)), "validate": validation})
    if action_id == "get_config":
        return _set_last_result({"ok": True, "config": _mask_config(dict(_CONFIG))})
    if action_id == "shell_command":
        return _set_last_result(_send_shell_command(payload))
    if action_id == "safe_stop":
        return _set_last_result(_safe_stop())

    return _set_last_result({"ok": False, "error": f"Action '{action_id}' not implemented"})
