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
import json, platform, shutil, subprocess, time
from pathlib import Path

SESSION = {"connected": False, "notes": [], "history": [], "buffer": [], "config": {}}
DRIVER_ID = "com.softdev0.network"
DRIVER_KIND = "network"
DEFAULTS_PATH = Path(__file__).with_name("defaults.json")
CONFIG_PATH = Path(__file__).with_name("config.json")

def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _note(msg):
    SESSION["notes"].append(f"[{_now()}] {msg}")
    SESSION["notes"] = SESSION["notes"][-50:]

def _hist(action, ok, **extra):
    row = {"ts": _now(), "action": action, "ok": bool(ok)}
    row.update(extra)
    SESSION["history"].append(row)
    limit = int(SESSION["config"].get("history_limit", 200) or 200)
    SESSION["history"] = SESSION["history"][-limit:]
    return row

def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _probe_tools(candidates):
    return {c: shutil.which(c) for c in candidates}

def _platform_summary():
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version()}

def _linux_list_netifs():
    base = Path("/sys/class/net")
    if not base.exists():
        return []
    out = []
    for p in sorted(base.iterdir()):
        try:
            out.append({"name": p.name, "mac": (p / "address").read_text().strip() if (p / "address").exists() else "", "state": (p / "operstate").read_text().strip() if (p / "operstate").exists() else ""})
        except Exception:
            out.append({"name": p.name})
    return out

def _linux_lsblk():
    if not shutil.which("lsblk"):
        return []
    r = _run(["lsblk", "-J", "-o", "NAME,KNAME,TYPE,SIZE,MODEL,VENDOR,TRAN,MOUNTPOINT,FSTYPE"])
    if not r.get("ok"):
        return []
    try:
        return json.loads(r["stdout"]).get("blockdevices", [])
    except Exception:
        return []

def _detect_localnpu():
    tools = _probe_tools(["lspci", "lsusb"])
    backends = {"openvino": False, "onnxruntime": False, "tflite_runtime": False, "pycoral": False}
    for mod in list(backends):
        try:
            __import__(mod)
            backends[mod] = True
        except Exception:
            pass
    hints = []
    if tools.get("lspci"):
        r = _run(["lspci"])
        for line in r.get("stdout", "").splitlines():
            s = line.lower()
            if any(k in s for k in ["coral", "movidius", "neural", "ai accelerator", "vpu"]):
                hints.append(line.strip())
    if tools.get("lsusb"):
        r = _run(["lsusb"])
        for line in r.get("stdout", "").splitlines():
            s = line.lower()
            if any(k in s for k in ["coral", "movidius", "google inc.", "intel"]):
                hints.append(line.strip())
    return {"tools": tools, "python_backends": backends, "accelerator_hints": hints}

def _detect_utpsip():
    tools = _probe_tools(["pjsua", "baresip", "linphonec", "arecord", "aplay"])
    return {"tools": tools, "hints": ["Telephony/SIP bridge; exact semantics depend on installed stack and hardware adapter."]}

def _detect_network():
    tools = _probe_tools(["ip", "nmcli", "iw", "mmcli", "networksetup"])
    data = {"tools": tools, "interfaces": _linux_list_netifs()}
    if tools.get("nmcli"):
        data["nmcli"] = _run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"])
    return data

def _detect_storage():
    tools = _probe_tools(["lsblk", "blkid", "smartctl", "nvme"])
    return {"tools": tools, "devices": _linux_lsblk()}

def driver_init(config=None):
    defaults = _load_json(DEFAULTS_PATH)
    current = _load_json(CONFIG_PATH)
    merged = {**defaults, **current, **(config or {})}
    SESSION["config"] = merged
    _save_json(CONFIG_PATH, merged)
    SESSION["connected"] = True
    _note("initialized")
    _hist("init", True)
    return {"ok": True, "driver_id": DRIVER_ID, "config": merged}

def driver_status():
    return {"ok": True, "driver_id": DRIVER_ID, "driver_kind": DRIVER_KIND, "connected": SESSION["connected"], "platform": _platform_summary(), "notes": SESSION["notes"][-10:], "history_count": len(SESSION["history"]), "config": SESSION["config"]}

def _describe_capabilities():
    if DRIVER_KIND == "localnpu":
        return {"bridge": "local AI accelerator runtime bridge", "targets": ["OpenVINO", "ONNX Runtime", "TensorFlow Lite", "PyCoral"], "actions": ["discover_accelerators", "select_runtime", "assign_model_target", "queue_benchmark"]}
    if DRIVER_KIND == "utpsip":
        return {"bridge": "telephony/SIP hardware and stack bridge", "actions": ["connect_stack", "disconnect_stack", "start_call", "hangup", "send_dtmf"]}
    if DRIVER_KIND == "network":
        return {"bridge": "network adapter inventory and preferred-interface bridge", "actions": ["discover_adapters", "list_interfaces", "select_preferred_interface"]}
    if DRIVER_KIND == "storage":
        return {"bridge": "storage/media inventory and host-tool probe bridge", "actions": ["discover_storage", "list_volumes", "smart_probe", "nvme_probe"]}
    return {"bridge": DRIVER_KIND}

def driver_action(action, **kwargs):
    if action == "get_config":
        return {"ok": True, "config": SESSION["config"]}
    if action == "set_config":
        SESSION["config"].update(kwargs.get("config") or {})
        _save_json(CONFIG_PATH, SESSION["config"])
        _hist("set_config", True)
        return {"ok": True, "config": SESSION["config"]}
    if action == "describe_capabilities":
        return {"ok": True, "capabilities": _describe_capabilities()}
    if action == "safe_stop":
        SESSION["connected"] = False
        _hist("safe_stop", True)
        return {"ok": True}
    if DRIVER_KIND == "localnpu":
        if action in ("probe_backends", "discover_accelerators"):
            data = _detect_localnpu()
            _hist(action, True, count=len(data.get("accelerator_hints", [])))
            return {"ok": True, **data}
        if action == "select_runtime":
            runtime = kwargs.get("runtime") or SESSION["config"].get("preferred_runtime", "auto")
            SESSION["config"]["selected_runtime"] = runtime
            _hist(action, True, runtime=runtime)
            return {"ok": True, "selected_runtime": runtime}
        if action == "assign_model_target":
            model = kwargs.get("model_path") or SESSION["config"].get("model_path", "")
            delegate = kwargs.get("delegate") or SESSION["config"].get("delegate_preference", "auto")
            row = {"model_path": model, "delegate": delegate}
            SESSION["buffer"].append(row)
            _hist(action, True, delegate=delegate)
            return {"ok": True, "assignment": row}
        if action == "queue_benchmark":
            if not SESSION["config"].get("allow_benchmarks", False) and not kwargs.get("confirm"):
                return {"ok": False, "error": "benchmark_not_allowed"}
            sec = int(kwargs.get("seconds") or SESSION["config"].get("benchmark_seconds", 5) or 5)
            _hist(action, True, seconds=sec)
            return {"ok": True, "queued": True, "seconds": sec}
    if DRIVER_KIND == "utpsip":
        if action in ("probe_backends", "discover_hardware"):
            data = _detect_utpsip()
            _hist(action, True)
            return {"ok": True, **data}
        if action == "connect_stack":
            SESSION["connected"] = True
            endpoint = f"{SESSION['config'].get('transport', 'udp')}://{SESSION['config'].get('sip_server', '')}:{SESSION['config'].get('sip_port', 5060)}"
            _hist(action, True, endpoint=endpoint)
            return {"ok": True, "endpoint": endpoint}
        if action == "disconnect_stack":
            SESSION["connected"] = False
            _hist(action, True)
            return {"ok": True}
        if action == "start_call":
            if not SESSION["config"].get("allow_call_control", False) and not kwargs.get("confirm"):
                return {"ok": False, "error": "call_control_not_allowed"}
            target = kwargs.get("target", "")
            _hist(action, True, target=target)
            return {"ok": True, "target": target, "state": "dialing"}
        if action == "hangup":
            _hist(action, True)
            return {"ok": True, "state": "ended"}
        if action == "send_dtmf":
            tones = str(kwargs.get("tones", ""))
            _hist(action, True, tones=tones)
            return {"ok": True, "tones": tones}
    if DRIVER_KIND == "network":
        if action in ("probe_backends", "discover_adapters", "list_interfaces"):
            data = _detect_network()
            _hist(action, True, count=len(data.get("interfaces", [])))
            return {"ok": True, **data}
        if action == "select_preferred_interface":
            if not SESSION["config"].get("allow_interface_switch", False) and not kwargs.get("confirm"):
                return {"ok": False, "error": "interface_switch_not_allowed"}
            iface = kwargs.get("name") or SESSION["config"].get("preferred_interface", "")
            SESSION["config"]["preferred_interface"] = iface
            _save_json(CONFIG_PATH, SESSION["config"])
            _hist(action, True, interface=iface)
            return {"ok": True, "preferred_interface": iface}
    if DRIVER_KIND == "storage":
        if action in ("probe_backends", "discover_storage", "list_volumes"):
            data = _detect_storage()
            _hist(action, True, count=len(data.get("devices", [])))
            return {"ok": True, **data}
        if action == "smart_probe":
            if not SESSION["config"].get("allow_smart_probe", False) and not kwargs.get("confirm"):
                return {"ok": False, "error": "smart_probe_not_allowed"}
            dev = kwargs.get("device", "")
            if shutil.which("smartctl") and dev:
                r = _run(["smartctl", "-a", dev])
                _hist(action, r.get("ok", False), device=dev)
                return {"ok": r.get("ok", False), "device": dev, "result": r}
            return {"ok": False, "error": "smartctl_missing_or_device_unspecified"}
        if action == "nvme_probe":
            if not SESSION["config"].get("allow_nvme_probe", False) and not kwargs.get("confirm"):
                return {"ok": False, "error": "nvme_probe_not_allowed"}
            dev = kwargs.get("device", "")
            if shutil.which("nvme") and dev:
                r = _run(["nvme", "id-ctrl", dev])
                _hist(action, r.get("ok", False), device=dev)
                return {"ok": r.get("ok", False), "device": dev, "result": r}
            return {"ok": False, "error": "nvme_cli_missing_or_device_unspecified"}
    _hist(action, False, error="not_implemented")
    return {"ok": False, "error": "not_implemented", "action": action}

def driver_shutdown():
    SESSION["connected"] = False
    _hist("shutdown", True)
    return {"ok": True, "driver_id": DRIVER_ID}
