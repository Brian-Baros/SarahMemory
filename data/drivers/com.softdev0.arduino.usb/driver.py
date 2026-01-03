# ../data/drivers/com.softdev0.arduino.usb/driver.py
import os
import json
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

_DRIVER_ID = "com.softdev0.arduino.usb"

# Runtime session state (non-persistent)
_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "last_upload_ts": 0,
    "port": None,
    "fqbn": None,
    "sketch_dir": None,
    "status": "idle",
    "error": None,
}

def _now_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper()
    )

def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "Arduino USB Driver (Uno R3 / AVR)",
        "version": "1.0.0",
        "supports": ["arduino:avr:uno"],
        "session": dict(_SESSION),
    }

def _try_import_list_ports():
    try:
        from serial.tools import list_ports  # type: ignore
        return list_ports
    except Exception:
        return None

def scan_ports():
    """Safe discovery. Returns a list of port strings."""
    lp = _try_import_list_ports()
    if not lp:
        return {"ok": False, "ports": [], "error": "pyserial not installed (serial.tools.list_ports unavailable)"}
    ports = []
    for p in lp.comports():
        ports.append({
            "device": getattr(p, "device", ""),
            "description": getattr(p, "description", ""),
            "hwid": getattr(p, "hwid", "")
        })
    return {"ok": True, "ports": ports}

def _resolve_cli(cli_path: str):
    # allow absolute path or rely on PATH
    if cli_path and os.path.isabs(cli_path) and os.path.exists(cli_path):
        return cli_path
    return cli_path or "arduino-cli"

def _run(cmd, cwd=None, timeout=120):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False
    )
    return {
        "ok": p.returncode == 0,
        "code": p.returncode,
        "stdout": (p.stdout or "").strip(),
        "stderr": (p.stderr or "").strip()
    }

def ensure_sketch_dir(sketch_dir: str):
    d = Path(sketch_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def write_sketch(sketch_dir: str, sketch_name: str, ino_code: str):
    """Create/overwrite a sketch folder and .ino file."""
    base = Path(ensure_sketch_dir(sketch_dir))
    sketch_root = base / sketch_name
    sketch_root.mkdir(parents=True, exist_ok=True)
    ino_path = sketch_root / f"{sketch_name}.ino"
    ino_path.write_text(ino_code or "", encoding="utf-8")
    return {"ok": True, "sketch_root": str(sketch_root), "ino_path": str(ino_path)}

def driver_validate(config: dict):
    """Dry validation. No upload, no hardware commands."""
    errs = []
    warns = []

    fqbn = (config.get("fqbn") or "").strip()
    if not fqbn:
        errs.append("Missing fqbn (e.g. arduino:avr:uno).")

    cli = _resolve_cli((config.get("arduino_cli_path") or "").strip())
    # If cli isn't resolvable, compile/upload will fail. We only warn here.
    if not cli:
        errs.append("Missing arduino_cli_path.")

    sketch_dir = (config.get("sketch_dir") or "").strip()
    if not sketch_dir:
        errs.append("Missing sketch_dir.")

    # Port is optional if auto-detect enabled, but still warn if blank.
    auto = bool(config.get("auto_detect_port", True))
    port = (config.get("port") or "").strip()
    if not auto and not port:
        errs.append("auto_detect_port is false but port is empty.")
    if auto and not port:
        warns.append("Port is empty; will rely on auto-detect at runtime.")

    # Rate limit sane
    try:
        rl = int(config.get("upload_rate_limit_sec", 0))
        if rl < 0:
            errs.append("upload_rate_limit_sec must be >= 0.")
    except Exception:
        errs.append("upload_rate_limit_sec must be an integer.")

    return {"ok": len(errs) == 0, "errors": errs, "warnings": warns}

def _auto_pick_port():
    res = scan_ports()
    if not res.get("ok"):
        return ""
    # heuristic: pick first port that looks like Arduino/USB serial
    for p in res.get("ports", []):
        desc = (p.get("description") or "").lower()
        hwid = (p.get("hwid") or "").lower()
        dev  = (p.get("device") or "")
        if "arduino" in desc or "usb serial" in desc or "wch" in hwid or "cp210" in hwid:
            return dev
    # fallback: first available
    ports = res.get("ports", [])
    return ports[0]["device"] if ports else ""

def compile_sketch(config: dict, sketch_root: str):
    cli = _resolve_cli((config.get("arduino_cli_path") or "").strip())
    fqbn = (config.get("fqbn") or "").strip()
    return _run([cli, "compile", "--fqbn", fqbn, sketch_root], cwd=sketch_root, timeout=180)

def upload_sketch(config: dict, sketch_root: str, confirm: bool = False):
    if bool(config.get("require_confirm_upload", True)) and not confirm:
        return {"ok": False, "error": "Upload requires confirm=true."}

    # Rate limit
    rl = int(config.get("upload_rate_limit_sec", 0) or 0)
    last = float(_SESSION.get("last_upload_ts") or 0)
    if rl and (time.time() - last) < rl:
        return {"ok": False, "error": f"Rate limit: wait {int(rl - (time.time()-last))}s before uploading again."}

    cli = _resolve_cli((config.get("arduino_cli_path") or "").strip())
    fqbn = (config.get("fqbn") or "").strip()

    port = (config.get("port") or "").strip()
    if bool(config.get("auto_detect_port", True)) and not port:
        port = _auto_pick_port()

    if not port:
        return {"ok": False, "error": "No serial port selected/detected."}

    res = _run([cli, "upload", "-p", port, "--fqbn", fqbn, sketch_root], cwd=sketch_root, timeout=180)
    if res.get("ok"):
        _SESSION["last_upload_ts"] = time.time()
        _SESSION["port"] = port
    return res

def driver_init(context=None, config=None):
    """Start a runtime session for this driver. No upload here; just session init."""
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
    _SESSION["fqbn"] = (config.get("fqbn") or "").strip() or None
    _SESSION["sketch_dir"] = (config.get("sketch_dir") or "").strip() or None

    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}

def driver_shutdown(context=None):
    """Release resources (serial monitors etc)."""
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["port"] = None
    _SESSION["fqbn"] = None
    _SESSION["sketch_dir"] = None

    # Release instance id on shutdown/reboot
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True

def driver_status(context=None):
    return {"ok": True, "session": dict(_SESSION)}

def driver_actions():
    """Expose actions to DriverLauncher / API layer."""
    return [
        {"id": "scan_ports", "label": "Scan Ports"},
        {"id": "dry_run_compile", "label": "Dry Run Compile"},
        {"id": "upload", "label": "Upload Sketch"},
        {"id": "safe_stop", "label": "Safe Stop"},
    ]
