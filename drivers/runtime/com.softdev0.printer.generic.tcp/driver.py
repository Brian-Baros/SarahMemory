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
# com.softdev0.printer.generic.tcp/driver.py
# Protocol-driven generic network printer driver
# Supports raw TCP (JetDirect/AppSocket), LPR via host tools, IPP/IPPS via host spoolers, and basic file/text submission.
# Degrades cleanly when optional host spooler tools are unavailable.

import os
import io
import json
import time
import socket
import shutil
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

_DRIVER_ID = "com.softdev0.printer.generic.tcp"

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "history": [],
    "last_result": None,
}

_DEFAULTS = {
    "enabled": True,
    "autoload": False,
    "host": "",
    "port": 9100,
    "mode": "raw_tcp",  # raw_tcp|lpr|ipp|ipps|cups_socket|shell_lp|shell_lpr
    "timeout_s": 10.0,
    "connect_timeout_s": 3.0,
    "queue": "lp",
    "uri": "",
    "job_name": "SarahMemory Job",
    "encoding": "utf-8",
    "line_ending": "lf",  # lf|crlf|cr
    "copies": 1,
    "duplex": "none",  # none|long_edge|short_edge
    "media": "",
    "color_mode": "auto",  # auto|monochrome|color
    "page_ranges": "",
    "orientation": "portrait",
    "history_limit": 100,
    "allow_raw_bytes": False,
    "require_print_confirmation": False,
    "shell_timeout_s": 30.0,
    "spool_dir": "",
}

def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper()
    )

def _note(msg):
    _SESSION["notes"].append(str(msg))
    _SESSION["notes"] = _SESSION["notes"][-50:]

def _record(result):
    _SESSION["last_result"] = result
    _SESSION["history"].append({
        "ts": time.time(),
        "result": result,
    })
    limit = int(_CONFIG.get("history_limit", 100) or 100)
    _SESSION["history"] = _SESSION["history"][-limit:]
    return result

def _find_tool(*names):
    for name in names:
        p = shutil.which(name)
        if p:
            return p
    return None

def _normalize_config(config):
    cfg = dict(_DEFAULTS)
    if isinstance(config, dict):
        cfg.update(config)
    cfg["port"] = int(cfg.get("port", 9100) or 9100)
    cfg["timeout_s"] = float(cfg.get("timeout_s", 10.0) or 10.0)
    cfg["connect_timeout_s"] = float(cfg.get("connect_timeout_s", 3.0) or 3.0)
    cfg["shell_timeout_s"] = float(cfg.get("shell_timeout_s", 30.0) or 30.0)
    cfg["copies"] = max(1, int(cfg.get("copies", 1) or 1))
    cfg["history_limit"] = max(1, int(cfg.get("history_limit", 100) or 100))
    cfg["host"] = str(cfg.get("host", "") or "").strip()
    cfg["queue"] = str(cfg.get("queue", "lp") or "lp").strip()
    cfg["uri"] = str(cfg.get("uri", "") or "").strip()
    cfg["mode"] = str(cfg.get("mode", "raw_tcp") or "raw_tcp").strip().lower()
    cfg["encoding"] = str(cfg.get("encoding", "utf-8") or "utf-8")
    cfg["line_ending"] = str(cfg.get("line_ending", "lf") or "lf").lower()
    cfg["job_name"] = str(cfg.get("job_name", "SarahMemory Job") or "SarahMemory Job")
    cfg["spool_dir"] = str(cfg.get("spool_dir", "") or "").strip()
    return cfg

_CONFIG = dict(_DEFAULTS)

def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "Generic Network Printer Driver",
        "version": "2.0.0",
        "transport": "tcp",
        "session": dict(_SESSION),
    }

def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    mode = str(config.get("mode", "raw_tcp") or "raw_tcp").lower()
    if mode not in {"raw_tcp", "lpr", "ipp", "ipps", "cups_socket", "shell_lp", "shell_lpr"}:
        errors.append("mode must be one of raw_tcp|lpr|ipp|ipps|cups_socket|shell_lp|shell_lpr")
    if "port" in config:
        try:
            port = int(config.get("port"))
            if not (1 <= port <= 65535):
                errors.append("port must be 1-65535")
        except Exception:
            errors.append("port must be an integer")
    for key in ("enabled", "autoload", "allow_raw_bytes", "require_print_confirmation"):
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")
    if not config.get("host") and mode in {"raw_tcp", "lpr", "cups_socket"} and not config.get("uri"):
        warnings.append("host is empty; network submission will fail until configured")
    if mode in {"ipp", "ipps"} and not (config.get("uri") or config.get("host")):
        warnings.append("IPP/IPPS usually requires uri or host")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

def driver_init(context=None, config=None):
    global _CONFIG
    cfg = _normalize_config(config or {})
    v = driver_validate(cfg)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _CONFIG = cfg
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["connected"] = False
    _note("Generic printer driver initialized.")
    return _record({"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()})

def driver_shutdown(context=None):
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["connected"] = False
    _note("Shutdown completed.")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True

def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _masked_config(),
        "backends": _probe_backends(),
    }

def _masked_config():
    return dict(_CONFIG)

def _probe_backends():
    return {
        "raw_tcp": True,
        "lp": bool(_find_tool("lp")),
        "lpr": bool(_find_tool("lpr")),
        "powershell": bool(_find_tool("powershell", "pwsh")),
    }

def _uri_from_cfg(cfg):
    if cfg.get("uri"):
        return cfg["uri"]
    host = cfg.get("host", "")
    port = int(cfg.get("port", 9100) or 9100)
    mode = cfg.get("mode", "raw_tcp")
    if mode in {"raw_tcp", "cups_socket"}:
        return f"socket://{host}:{port}"
    if mode == "lpr":
        q = cfg.get("queue") or "lp"
        return f"lpd://{host}/{q}"
    if mode in {"ipp", "ipps"}:
        scheme = mode
        q = cfg.get("queue") or "ipp/print"
        return f"{scheme}://{host}:{port}/{q}"
    return ""

def _ping_target(host=None, port=None, timeout=None):
    host = host or _CONFIG.get("host")
    port = int(port or _CONFIG.get("port", 9100))
    timeout = float(timeout or _CONFIG.get("connect_timeout_s", 3.0))
    if not host:
        return {"ok": False, "error": "host_not_configured"}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    ts = time.time()
    try:
        s.connect((host, port))
        elapsed = round((time.time() - ts) * 1000.0, 2)
        return {"ok": True, "host": host, "port": port, "latency_ms": elapsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "host": host, "port": port}
    finally:
        try:
            s.close()
        except Exception:
            pass

def _apply_line_ending(text):
    le = _CONFIG.get("line_ending", "lf")
    if le == "crlf":
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    if le == "cr":
        return text.replace("\r\n", "\n").replace("\n", "\r")
    return text.replace("\r\n", "\n")

def _ensure_confirm(payload):
    if _CONFIG.get("require_print_confirmation") and not bool((payload or {}).get("confirm_print")):
        return {"ok": False, "error": "print_confirmation_required"}
    return {"ok": True}

def _send_raw_tcp(data: bytes):
    host = _CONFIG.get("host")
    port = int(_CONFIG.get("port", 9100))
    if not host:
        return {"ok": False, "error": "host_not_configured"}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(float(_CONFIG.get("timeout_s", 10.0)))
    try:
        s.connect((host, port))
        s.sendall(data)
        return {"ok": True, "bytes_sent": len(data), "host": host, "port": port}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        try:
            s.close()
        except Exception:
            pass

def _run_cmd(cmd, timeout=None):
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout or _CONFIG.get("shell_timeout_s", 30.0)),
            check=False
        )
        return {
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}

def _write_temp_file(data: bytes, suffix=".bin"):
    spool_dir = _CONFIG.get("spool_dir") or ""
    if spool_dir:
        os.makedirs(spool_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="sm_print_", suffix=suffix, dir=spool_dir)
    else:
        fd, path = tempfile.mkstemp(prefix="sm_print_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def _submit_via_lp(data: bytes):
    lp = _find_tool("lp")
    if not lp:
        return {"ok": False, "error": "lp_not_found"}
    path = _write_temp_file(data, ".dat")
    try:
        uri = _uri_from_cfg(_CONFIG)
        cmd = [lp, "-t", _CONFIG.get("job_name", "SarahMemory Job")]
        if _CONFIG.get("copies", 1) > 1:
            cmd += ["-n", str(_CONFIG.get("copies", 1))]
        media = _CONFIG.get("media")
        if media:
            cmd += ["-o", f"media={media}"]
        duplex = _CONFIG.get("duplex", "none")
        if duplex == "long_edge":
            cmd += ["-o", "sides=two-sided-long-edge"]
        elif duplex == "short_edge":
            cmd += ["-o", "sides=two-sided-short-edge"]
        if _CONFIG.get("color_mode") == "monochrome":
            cmd += ["-o", "print-color-mode=monochrome"]
        elif _CONFIG.get("color_mode") == "color":
            cmd += ["-o", "print-color-mode=color"]
        if _CONFIG.get("page_ranges"):
            cmd += ["-P", str(_CONFIG["page_ranges"])]
        if uri:
            cmd += ["-d", uri]
        cmd += [path]
        res = _run_cmd(cmd)
        res["temp_path"] = path
        return res
    finally:
        pass

def _submit_via_lpr(data: bytes):
    lpr = _find_tool("lpr")
    if not lpr:
        return {"ok": False, "error": "lpr_not_found"}
    path = _write_temp_file(data, ".dat")
    uri = _uri_from_cfg(_CONFIG)
    cmd = [lpr]
    host = _CONFIG.get("host")
    q = _CONFIG.get("queue") or "lp"
    if host:
        cmd += ["-H", host]
    if q:
        cmd += ["-P", q]
    if uri and uri.startswith("lpd://"):
        # host/queue already represented; keep native flags
        pass
    cmd += [path]
    res = _run_cmd(cmd)
    res["temp_path"] = path
    return res

def _build_lpr_control(data_len: int, job_name: str):
    host = socket.gethostname() or "localhost"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "sarahmemory"
    user = re_sub_nonalpha(user)
    job = str(int(time.time()) % 1000).zfill(3)
    cf_name = f"cfA{job}{host[:31]}"
    df_name = f"dfA{job}{host[:31]}"
    lines = [
        f"H{host}",
        f"P{user}",
        f"J{job_name[:99]}",
        f"ld{df_name}",
        f"U{df_name}",
        f"N{job_name[:99]}",
    ]
    control = "\n".join(lines) + "\n"
    return control.encode("utf-8"), df_name.encode("ascii"), cf_name.encode("ascii")

def re_sub_nonalpha(user):
    out = ''.join(ch for ch in user if ch.isalnum() or ch in ('_', '-', '.'))
    if not out or out[0].isdigit():
        return "sarahmemory"
    return out

def _submit_via_lpr_direct(data: bytes):
    # Minimal RFC1179 implementation. Some daemons require reserved source ports; this path is best-effort only.
    host = _CONFIG.get("host")
    queue = _CONFIG.get("queue") or "lp"
    port = int(_CONFIG.get("port", 515) or 515)
    if not host:
        return {"ok": False, "error": "host_not_configured"}
    ctrl, df_name, cf_name = _build_lpr_control(len(data), _CONFIG.get("job_name", "SarahMemory Job"))
    try:
        s = socket.create_connection((host, port), timeout=float(_CONFIG.get("timeout_s", 10.0)))
        # receive printer job
        s.sendall(b"\x02" + queue.encode("ascii", errors="ignore") + b"\n")
        ack = s.recv(1)
        if ack != b"\x00":
            return {"ok": False, "error": "lpr_receive_job_rejected", "ack": ack.hex() if ack else ""}
        # control file
        s.sendall(b"\x02" + str(len(ctrl)).encode("ascii") + b" " + cf_name + b"\n")
        ack = s.recv(1)
        if ack != b"\x00":
            return {"ok": False, "error": "lpr_control_header_rejected", "ack": ack.hex() if ack else ""}
        s.sendall(ctrl + b"\x00")
        ack = s.recv(1)
        if ack != b"\x00":
            return {"ok": False, "error": "lpr_control_file_rejected", "ack": ack.hex() if ack else ""}
        # data file
        s.sendall(b"\x03" + str(len(data)).encode("ascii") + b" " + df_name + b"\n")
        ack = s.recv(1)
        if ack != b"\x00":
            return {"ok": False, "error": "lpr_data_header_rejected", "ack": ack.hex() if ack else ""}
        s.sendall(data + b"\x00")
        ack = s.recv(1)
        if ack != b"\x00":
            return {"ok": False, "error": "lpr_data_file_rejected", "ack": ack.hex() if ack else ""}
        return {"ok": True, "protocol": "lpr_direct", "bytes_sent": len(data)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "protocol": "lpr_direct"}
    finally:
        try:
            s.close()
        except Exception:
            pass

def _submit_bytes(data: bytes):
    mode = _CONFIG.get("mode", "raw_tcp")
    if mode == "raw_tcp":
        return _send_raw_tcp(data)
    if mode == "lpr":
        # prefer host tool because direct RFC1179 has daemon-specific constraints
        if _find_tool("lpr"):
            return _submit_via_lpr(data)
        return _submit_via_lpr_direct(data)
    if mode in {"ipp", "ipps", "cups_socket", "shell_lp"}:
        return _submit_via_lp(data)
    if mode == "shell_lpr":
        return _submit_via_lpr(data)
    return {"ok": False, "error": f"unsupported_mode:{mode}"}

def _text_to_bytes(text):
    text = _apply_line_ending(text)
    return text.encode(_CONFIG.get("encoding", "utf-8"), errors="replace")

def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    action_id = str(action_id or "").strip().lower()

    if action_id == "ping":
        return _record(_ping_target(payload.get("host"), payload.get("port"), payload.get("timeout_s")))

    if action_id == "probe_backends":
        return _record({"ok": True, "backends": _probe_backends(), "uri": _uri_from_cfg(_CONFIG)})

    if action_id == "get_config":
        return _record({"ok": True, "config": _masked_config()})

    if action_id == "set_config":
        new_cfg = dict(_CONFIG)
        if isinstance(payload, dict):
            new_cfg.update(payload)
        new_cfg = _normalize_config(new_cfg)
        v = driver_validate(new_cfg)
        if not v.get("ok"):
            return _record({"ok": False, "error": "validation_failed", "details": v})
        _CONFIG.update(new_cfg)
        return _record({"ok": True, "config": _masked_config(), "validate": v})

    if action_id == "connect":
        res = _ping_target()
        _SESSION["connected"] = bool(res.get("ok"))
        _SESSION["status"] = "ready" if res.get("ok") else "error"
        return _record(res)

    if action_id == "disconnect":
        _SESSION["connected"] = False
        return _record({"ok": True, "disconnected": True})

    if action_id == "print_text":
        gate = _ensure_confirm(payload)
        if not gate.get("ok"):
            return _record(gate)
        text = str(payload.get("text", "") or "")
        data = _text_to_bytes(text)
        return _record(_submit_bytes(data))

    if action_id == "print_raw":
        if not _CONFIG.get("allow_raw_bytes"):
            return _record({"ok": False, "error": "raw_bytes_not_allowed"})
        gate = _ensure_confirm(payload)
        if not gate.get("ok"):
            return _record(gate)
        raw = payload.get("bytes")
        if isinstance(raw, str):
            # hex or plain text
            try:
                cleaned = raw.replace(" ", "").replace("\n", "")
                data = bytes.fromhex(cleaned)
            except Exception:
                data = raw.encode(_CONFIG.get("encoding", "utf-8"), errors="replace")
        elif isinstance(raw, list):
            data = bytes(int(x) & 0xFF for x in raw)
        else:
            return _record({"ok": False, "error": "bytes_payload_required"})
        return _record(_submit_bytes(data))

    if action_id == "print_file":
        gate = _ensure_confirm(payload)
        if not gate.get("ok"):
            return _record(gate)
        path = str(payload.get("path", "") or "").strip()
        if not path or not os.path.isfile(path):
            return _record({"ok": False, "error": "file_not_found", "path": path})
        with open(path, "rb") as f:
            data = f.read()
        return _record(_submit_bytes(data))

    if action_id == "print_test":
        gate = _ensure_confirm(payload)
        if not gate.get("ok"):
            return _record(gate)
        lines = [
            "SarahMemory Generic Network Printer Test Page",
            f"Timestamp: {datetime.utcnow().isoformat()}Z",
            f"Mode: {_CONFIG.get('mode')}",
            f"URI: {_uri_from_cfg(_CONFIG)}",
            "",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789",
            "",
        ]
        return _record(_submit_bytes(_text_to_bytes("\n".join(lines))))

    if action_id == "describe_capabilities":
        return _record({
            "ok": True,
            "capabilities": {
                "generic_network_printing": True,
                "manufacturer_specific_finishing": False,
                "modes": ["raw_tcp", "lpr", "ipp", "ipps", "cups_socket", "shell_lp", "shell_lpr"],
                "best_for": {
                    "raw_tcp": "PCL/PS/text data to port 9100 compatible devices",
                    "lpr": "LPD/LPR queue submission",
                    "ipp_ipps": "Host spooler submission to IPP/IPPS URIs",
                    "shell_lp_lpr": "Host-managed generic printing where CUPS/LPD tools exist",
                }
            }
        })

    if action_id == "safe_stop":
        _SESSION["connected"] = False
        _SESSION["status"] = "ready"
        return _record({"ok": True, "stopped": True})

    return _record({"ok": False, "error": f"Action '{action_id}' not implemented"})
