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
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.arduino.usb"
_DRIVER_NAME = "Arduino USB Driver (Uno R3 / AVR)"
_DRIVER_VERSION = "1.0.0"

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "last_upload_ts": 0.0,
    "last_compile_ts": 0.0,
    "port": None,
    "fqbn": None,
    "sketch_dir": None,
    "sketch_root": None,
    "last_action": None,
    "last_result": None,
    "status": "idle",
    "error": None,
    "notes": [],
}


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _note(message: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append({"ts": _utc_now(), "message": str(message)})
    if len(notes) > 50:
        del notes[:-50]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mask_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return path_value
    try:
        p = Path(path_value)
        return str(p)
    except Exception:
        return path_value


def _normalize_config(config: Optional[dict]) -> Dict[str, Any]:
    config = dict(config or {})
    return {
        "enabled": bool(config.get("enabled", True)),
        "autoload": bool(config.get("autoload", False)),
        "auto_detect_port": bool(config.get("auto_detect_port", True)),
        "port": str(config.get("port") or "").strip(),
        "baud": _safe_int(config.get("baud", 115200), 115200),
        "fqbn": str(config.get("fqbn") or "arduino:avr:uno").strip(),
        "arduino_cli_path": str(config.get("arduino_cli_path") or "arduino-cli").strip(),
        "sketch_dir": str(config.get("sketch_dir") or "../data/sandbox/arduino").strip(),
        "require_confirm_upload": bool(config.get("require_confirm_upload", True)),
        "upload_rate_limit_sec": max(0, _safe_int(config.get("upload_rate_limit_sec", 10), 10)),
        "verify_upload": bool(config.get("verify_upload", False)),
        "discovery_timeout": str(config.get("discovery_timeout") or "5s").strip(),
        "compile_warnings": str(config.get("compile_warnings") or "default").strip(),
    }


def _resolve_cli(cli_path: str) -> Optional[str]:
    cli_path = (cli_path or "").strip()
    if cli_path and os.path.isabs(cli_path) and os.path.exists(cli_path):
        return cli_path
    resolved = shutil.which(cli_path or "arduino-cli")
    if resolved:
        return resolved
    return cli_path or None


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": cmd,
        }
    except FileNotFoundError as exc:
        return {"ok": False, "code": 127, "stdout": "", "stderr": str(exc), "cmd": cmd}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "code": 124,
            "stdout": (exc.stdout or "").strip() if exc.stdout else "",
            "stderr": (exc.stderr or "").strip() if exc.stderr else f"Command timed out after {timeout}s",
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "code": 1, "stdout": "", "stderr": str(exc), "cmd": cmd}


def _run_json(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    res = _run(cmd + ["--format", "json"], cwd=cwd, timeout=timeout)
    if not res.get("ok"):
        return res
    try:
        res["json"] = json.loads(res.get("stdout") or "{}")
    except Exception as exc:
        res["ok"] = False
        res["stderr"] = f"Failed to parse JSON output: {exc}"
    return res


def _try_import_list_ports():
    try:
        from serial.tools import list_ports  # type: ignore
        return list_ports
    except Exception:
        return None


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "supports": ["arduino:avr:uno"],
        "session": dict(_SESSION),
    }


def scan_ports() -> Dict[str, Any]:
    lp = _try_import_list_ports()
    if not lp:
        return {
            "ok": False,
            "ports": [],
            "error": "pyserial not installed (serial.tools.list_ports unavailable)",
        }
    ports: List[Dict[str, str]] = []
    for port in lp.comports():
        ports.append(
            {
                "device": getattr(port, "device", ""),
                "name": getattr(port, "name", ""),
                "description": getattr(port, "description", ""),
                "hwid": getattr(port, "hwid", ""),
                "manufacturer": getattr(port, "manufacturer", "") or "",
                "product": getattr(port, "product", "") or "",
                "serial_number": getattr(port, "serial_number", "") or "",
                "vid": str(getattr(port, "vid", "") or ""),
                "pid": str(getattr(port, "pid", "") or ""),
            }
        )
    return {"ok": True, "ports": ports, "count": len(ports)}


def _auto_pick_port() -> str:
    res = scan_ports()
    if not res.get("ok"):
        return ""
    preferred_tokens = (
        "arduino",
        "usb serial",
        "ch340",
        "wch",
        "cp210",
        "ftdi",
        "ttyacm",
        "ttyusb",
    )
    for item in res.get("ports", []):
        haystack = " ".join(
            [
                str(item.get("description") or "").lower(),
                str(item.get("hwid") or "").lower(),
                str(item.get("manufacturer") or "").lower(),
                str(item.get("product") or "").lower(),
                str(item.get("device") or "").lower(),
            ]
        )
        if any(token in haystack for token in preferred_tokens):
            return str(item.get("device") or "")
    ports = res.get("ports", [])
    return str(ports[0].get("device") or "") if ports else ""


def _cli_version(cli_path: str) -> Dict[str, Any]:
    cli = _resolve_cli(cli_path)
    if not cli:
        return {"ok": False, "error": "arduino-cli not found"}
    res = _run([cli, "version"], timeout=30)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("stderr") or "arduino-cli version check failed", "raw": res}
    return {"ok": True, "path": cli, "raw": res.get("stdout") or ""}


def list_boards(config: Optional[dict] = None) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    cli = _resolve_cli(cfg.get("arduino_cli_path", ""))
    if not cli:
        return {"ok": False, "boards": [], "error": "arduino-cli not found"}
    res = _run_json([cli, "board", "list"], timeout=45)
    if not res.get("ok"):
        return {"ok": False, "boards": [], "error": res.get("stderr") or "board list failed", "raw": res}
    payload = res.get("json")
    boards = payload.get("detected_ports", []) if isinstance(payload, dict) else payload
    return {"ok": True, "boards": boards or [], "raw": payload}


def ensure_sketch_dir(sketch_dir: str) -> str:
    path = Path(sketch_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def write_sketch(sketch_dir: str, sketch_name: str, ino_code: str) -> Dict[str, Any]:
    if not str(sketch_name or "").strip():
        return {"ok": False, "error": "sketch_name is required"}
    base = Path(ensure_sketch_dir(sketch_dir))
    safe_name = "".join(ch for ch in str(sketch_name).strip() if ch.isalnum() or ch in ("_", "-"))
    if not safe_name:
        return {"ok": False, "error": "sketch_name resolved to empty after sanitization"}
    sketch_root = base / safe_name
    sketch_root.mkdir(parents=True, exist_ok=True)
    ino_path = sketch_root / f"{safe_name}.ino"
    ino_path.write_text(ino_code or "", encoding="utf-8")
    _SESSION["sketch_root"] = str(sketch_root)
    _note(f"Sketch written: {ino_path}")
    return {"ok": True, "sketch_root": str(sketch_root), "ino_path": str(ino_path), "sketch_name": safe_name}


def driver_validate(config: Optional[dict]) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    errors: List[str] = []
    warnings: List[str] = []

    if not cfg["enabled"]:
        warnings.append("Driver is disabled in config.")

    if not cfg["fqbn"]:
        errors.append("Missing fqbn (example: arduino:avr:uno).")

    cli = _resolve_cli(cfg["arduino_cli_path"])
    if not cli:
        errors.append("arduino-cli not found in PATH and configured path is invalid.")
    else:
        version_res = _cli_version(cfg["arduino_cli_path"])
        if not version_res.get("ok"):
            errors.append(version_res.get("error") or "arduino-cli version check failed.")

    if not cfg["sketch_dir"]:
        errors.append("Missing sketch_dir.")
    else:
        try:
            ensure_sketch_dir(cfg["sketch_dir"])
        except Exception as exc:
            errors.append(f"sketch_dir is not writable: {exc}")

    if not cfg["auto_detect_port"] and not cfg["port"]:
        errors.append("auto_detect_port is false but port is empty.")
    if cfg["auto_detect_port"] and not cfg["port"]:
        warnings.append("Port is empty; runtime will attempt auto-detect.")

    if cfg["baud"] < 1200 or cfg["baud"] > 2000000:
        errors.append("baud must be between 1200 and 2000000.")

    if cfg["upload_rate_limit_sec"] < 0:
        errors.append("upload_rate_limit_sec must be >= 0.")

    board_scan = list_boards(cfg) if cli else {"ok": False, "boards": [], "error": "arduino-cli not found"}
    if board_scan.get("ok"):
        boards = board_scan.get("boards", [])
        if not boards:
            warnings.append("arduino-cli board list returned no detected ports.")
    else:
        warnings.append(board_scan.get("error") or "Could not enumerate boards via arduino-cli.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_config": cfg,
        "cli_found": bool(cli),
    }


def _resolve_port(config: Dict[str, Any]) -> str:
    port = str(config.get("port") or "").strip()
    if config.get("auto_detect_port") and not port:
        port = _auto_pick_port()
    return port


def compile_sketch(config: Optional[dict], sketch_root: str) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    cli = _resolve_cli(cfg["arduino_cli_path"])
    if not cli:
        return {"ok": False, "error": "arduino-cli not found"}
    sketch_root = str(Path(sketch_root).expanduser().resolve())
    cmd = [cli, "compile", "--fqbn", cfg["fqbn"], "--warnings", cfg["compile_warnings"], sketch_root]
    res = _run(cmd, cwd=sketch_root, timeout=300)
    if res.get("ok"):
        _SESSION["last_compile_ts"] = time.time()
        _SESSION["sketch_root"] = sketch_root
        _note(f"Compile succeeded for {sketch_root}")
    else:
        _note(f"Compile failed for {sketch_root}: {res.get('stderr') or res.get('stdout')}")
    return res


def upload_sketch(config: Optional[dict], sketch_root: str, confirm: bool = False) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    if cfg.get("require_confirm_upload") and not confirm:
        return {"ok": False, "error": "Upload requires confirm=true."}

    rl = int(cfg.get("upload_rate_limit_sec") or 0)
    last = float(_SESSION.get("last_upload_ts") or 0.0)
    delta = time.time() - last
    if rl and delta < rl:
        return {"ok": False, "error": f"Rate limit: wait {max(1, int(rl - delta))}s before uploading again."}

    cli = _resolve_cli(cfg["arduino_cli_path"])
    if not cli:
        return {"ok": False, "error": "arduino-cli not found"}

    port = _resolve_port(cfg)
    if not port:
        return {"ok": False, "error": "No serial port selected or auto-detected."}

    sketch_root = str(Path(sketch_root).expanduser().resolve())
    cmd = [
        cli,
        "upload",
        "-p",
        port,
        "--fqbn",
        cfg["fqbn"],
        "--discovery-timeout",
        cfg["discovery_timeout"],
    ]
    if cfg.get("verify_upload"):
        cmd.append("--verify")
    cmd.append(sketch_root)

    res = _run(cmd, cwd=sketch_root, timeout=300)
    if res.get("ok"):
        _SESSION["last_upload_ts"] = time.time()
        _SESSION["port"] = port
        _SESSION["sketch_root"] = sketch_root
        _note(f"Upload succeeded on {port} for {sketch_root}")
    else:
        _note(f"Upload failed on {port}: {res.get('stderr') or res.get('stdout')}")
    return res


def safe_stop() -> Dict[str, Any]:
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["last_action"] = "safe_stop"
    _SESSION["last_result"] = {"ok": True, "message": "No persistent serial stream to close."}
    _note("Safe stop requested.")
    return {"ok": True, "message": "Driver session remains ready; no persistent serial stream was active."}


def driver_init(context: Optional[dict] = None, config: Optional[dict] = None) -> Dict[str, Any]:
    validation = driver_validate(config)
    cfg = validation.get("normalized_config") or _normalize_config(config)
    if not validation.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        _SESSION["last_result"] = validation
        return {"ok": False, "error": "validation_failed", "details": validation}

    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["fqbn"] = cfg.get("fqbn") or None
    _SESSION["sketch_dir"] = cfg.get("sketch_dir") or None
    _SESSION["last_action"] = "init"
    _SESSION["last_result"] = {"ok": True}
    _note("Driver session initialized.")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": validation}


def driver_shutdown(context: Optional[dict] = None) -> bool:
    _note("Driver session shutdown.")
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["port"] = None
    _SESSION["fqbn"] = None
    _SESSION["sketch_dir"] = None
    _SESSION["sketch_root"] = None
    _SESSION["last_action"] = "shutdown"
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context: Optional[dict] = None, config: Optional[dict] = None) -> Dict[str, Any]:
    cfg = _normalize_config(config)
    port_scan = scan_ports()
    cli = _cli_version(cfg.get("arduino_cli_path", ""))
    boards = list_boards(cfg) if cli.get("ok") else {"ok": False, "boards": []}
    return {
        "ok": True,
        "session": dict(_SESSION),
        "cli": cli,
        "ports": port_scan,
        "boards": boards,
        "config": {
            "enabled": cfg["enabled"],
            "autoload": cfg["autoload"],
            "auto_detect_port": cfg["auto_detect_port"],
            "port": cfg["port"],
            "baud": cfg["baud"],
            "fqbn": cfg["fqbn"],
            "arduino_cli_path": _mask_path(cfg["arduino_cli_path"]),
            "sketch_dir": _mask_path(cfg["sketch_dir"]),
            "require_confirm_upload": cfg["require_confirm_upload"],
            "upload_rate_limit_sec": cfg["upload_rate_limit_sec"],
            "verify_upload": cfg["verify_upload"],
            "discovery_timeout": cfg["discovery_timeout"],
            "compile_warnings": cfg["compile_warnings"],
        },
    }


def driver_actions() -> List[Dict[str, str]]:
    return [
        {"id": "scan_ports", "label": "Scan Ports"},
        {"id": "list_boards", "label": "List Boards"},
        {"id": "write_sketch", "label": "Write Sketch"},
        {"id": "dry_run_compile", "label": "Dry Run Compile"},
        {"id": "upload", "label": "Upload Sketch"},
        {"id": "safe_stop", "label": "Safe Stop"},
        {"id": "status", "label": "Driver Status"},
    ]


def driver_action(action: str, payload: Optional[dict] = None, context: Optional[dict] = None, config: Optional[dict] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    cfg = _normalize_config(config)
    action_id = str(action or "").strip().lower()
    _SESSION["last_action"] = action_id

    if action_id == "ping":
        result = {"ok": True, "driver": _DRIVER_ID, "ts": _utc_now()}
    elif action_id == "scan_ports":
        result = scan_ports()
    elif action_id == "list_boards":
        result = list_boards(cfg)
    elif action_id == "write_sketch":
        result = write_sketch(
            payload.get("sketch_dir") or cfg.get("sketch_dir") or "",
            payload.get("sketch_name") or "SarahSketch",
            payload.get("ino_code") or "",
        )
    elif action_id == "dry_run_compile":
        sketch_root = payload.get("sketch_root") or _SESSION.get("sketch_root")
        if not sketch_root:
            result = {"ok": False, "error": "sketch_root is required. Call write_sketch first or pass sketch_root."}
        else:
            result = compile_sketch(cfg, str(sketch_root))
    elif action_id == "upload":
        sketch_root = payload.get("sketch_root") or _SESSION.get("sketch_root")
        if not sketch_root:
            result = {"ok": False, "error": "sketch_root is required. Call write_sketch first or pass sketch_root."}
        else:
            result = upload_sketch(cfg, str(sketch_root), confirm=bool(payload.get("confirm", False)))
    elif action_id == "safe_stop":
        result = safe_stop()
    elif action_id == "status":
        result = driver_status(context=context, config=cfg)
    elif action_id == "validate":
        result = driver_validate(cfg)
    else:
        result = {"ok": False, "error": f"Unsupported action: {action_id}"}

    _SESSION["last_result"] = result
    if result.get("ok"):
        if action_id not in ("safe_stop", "status"):
            _SESSION["status"] = "ready"
        _SESSION["error"] = None
    else:
        _SESSION["status"] = "error"
        _SESSION["error"] = result.get("error") or "action_failed"
    return result
