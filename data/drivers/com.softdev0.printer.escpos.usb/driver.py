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
# com.softdev0.printer.escpos.usb/driver.py
# ESC/POS Receipt Printer Driver
# Transport: usb
#
# Notes:
# - Uses python-escpos when available for high-level printing.
# - Uses pyusb when available for discovery.
# - Degrades cleanly when optional backends are absent.

import base64
import io
import os
import time
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime

_DRIVER_ID = "com.softdev0.printer.escpos.usb"

_DEFAULTS = {
    "enabled": True,
    "autoload": False,
    "vendor_id": "",
    "product_id": "",
    "interface": 0,
    "in_ep": 0,
    "out_ep": 0,
    "timeout_ms": 5000,
    "encoding": "cp437",
    "profile": "default",
    "align": "left",
    "line_spacing": 30,
    "feed_after_print": 2,
    "barcode_enabled": True,
    "qr_enabled": True,
    "image_enabled": True,
    "cut_enabled": True,
    "allow_cash_drawer": False,
    "allow_raw": False,
    "drawer_pin": 2,
    "history_limit": 200,
    "image_max_width": 512,
}

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "printer_backend": None,
    "device": None,
    "history": [],
    "last_result": None,
    "config": deepcopy(_DEFAULTS),
}

_PRINTER = None


def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper()
    )


def _now():
    return time.time()


def _note(msg):
    _SESSION.setdefault("notes", []).append(str(msg))
    _SESSION["notes"] = _SESSION["notes"][-50:]


def _record(action, ok, details=None):
    row = {
        "ts": _now(),
        "action": action,
        "ok": bool(ok),
        "details": details or {},
    }
    _SESSION.setdefault("history", []).append(row)
    limit = int(_SESSION.get("config", {}).get("history_limit", _DEFAULTS["history_limit"]))
    _SESSION["history"] = _SESSION["history"][-max(1, limit):]
    _SESSION["last_result"] = row
    return row


def _safe_int(v, default=0):
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("0x"):
            try:
                return int(s, 16)
            except Exception:
                return default
        try:
            return int(s)
        except Exception:
            return default
    return default


def _merged_config(config):
    merged = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    return merged


def _masked_config(cfg):
    return dict(cfg) if isinstance(cfg, dict) else {}


def _probe_backends():
    probed = {"usb": False, "escpos": False, "pil": False}
    try:
        import usb.core  # noqa
        probed["usb"] = True
    except Exception:
        pass
    try:
        from escpos.printer import Usb  # noqa
        probed["escpos"] = True
    except Exception:
        pass
    try:
        from PIL import Image  # noqa
        probed["pil"] = True
    except Exception:
        pass
    return {"ok": True, "backends": probed}


def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "ESC/POS Receipt Printer Driver",
        "version": "1.0.0",
        "transport": "usb",
        "session": driver_status().get("session", {}),
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}

    for key in ("enabled", "autoload", "barcode_enabled", "qr_enabled", "image_enabled",
                "cut_enabled", "allow_cash_drawer", "allow_raw"):
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")

    for key in ("interface", "in_ep", "out_ep", "timeout_ms", "feed_after_print",
                "drawer_pin", "history_limit", "image_max_width", "line_spacing"):
        if key in config:
            try:
                int(config.get(key))
            except Exception:
                errors.append(f"{key} must be an integer")

    if "encoding" in config and not isinstance(config.get("encoding"), str):
        errors.append("encoding must be a string")
    if not config.get("vendor_id"):
        warnings.append("vendor_id not set")
    if not config.get("product_id"):
        warnings.append("product_id not set")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _discover_devices():
    items = []
    try:
        import usb.core
        devs = list(usb.core.find(find_all=True) or [])
        for d in devs:
            items.append({
                "vendor_id": hex(int(getattr(d, "idVendor", 0))),
                "product_id": hex(int(getattr(d, "idProduct", 0))),
                "bus": getattr(d, "bus", None),
                "address": getattr(d, "address", None),
            })
        return {"ok": True, "devices": items}
    except Exception as e:
        return {"ok": False, "error": f"usb_discovery_failed: {e}", "devices": items}


def _open_printer(cfg):
    global _PRINTER
    try:
        from escpos.printer import Usb
    except Exception as e:
        return {"ok": False, "error": f"escpos_backend_unavailable: {e}"}

    vid = _safe_int(cfg.get("vendor_id"), None)
    pid = _safe_int(cfg.get("product_id"), None)
    if vid is None or pid is None:
        return {"ok": False, "error": "vendor_id and product_id are required"}

    kwargs = {
        "idVendor": vid,
        "idProduct": pid,
        "timeout": int(cfg.get("timeout_ms", 5000)),
    }
    if int(cfg.get("interface", 0)) > 0:
        kwargs["interface"] = int(cfg.get("interface", 0))
    if int(cfg.get("in_ep", 0)) > 0:
        kwargs["in_ep"] = int(cfg.get("in_ep", 0))
    if int(cfg.get("out_ep", 0)) > 0:
        kwargs["out_ep"] = int(cfg.get("out_ep", 0))

    try:
        _PRINTER = Usb(**kwargs)
        _SESSION["connected"] = True
        _SESSION["printer_backend"] = "python-escpos"
        _SESSION["device"] = {
            "vendor_id": hex(vid),
            "product_id": hex(pid),
            "interface": int(cfg.get("interface", 0)),
            "in_ep": int(cfg.get("in_ep", 0)),
            "out_ep": int(cfg.get("out_ep", 0)),
        }
        return {"ok": True, "device": deepcopy(_SESSION["device"])}
    except Exception as e:
        return {"ok": False, "error": f"open_failed: {e}"}


def _require_connected():
    if not _SESSION.get("connected") or _PRINTER is None:
        return {"ok": False, "error": "printer_not_connected"}
    return {"ok": True}


def _apply_common(cfg):
    if _PRINTER is None:
        return
    align = str(cfg.get("align", "left")).lower()
    if align in ("left", "center", "right"):
        try:
            _PRINTER.set(align=align)
        except Exception:
            pass


def _feed(cfg, lines=None):
    if lines is None:
        lines = int(cfg.get("feed_after_print", 2))
    try:
        _PRINTER.feed(int(lines))
        return {"ok": True, "fed": int(lines)}
    except Exception as e:
        return {"ok": False, "error": f"feed_failed: {e}"}


def _cut(cfg, mode="full"):
    if not bool(cfg.get("cut_enabled", True)):
        return {"ok": False, "error": "cut_disabled"}
    try:
        if mode == "partial":
            _PRINTER.cut(mode="PART")
        else:
            _PRINTER.cut()
        return {"ok": True, "mode": mode}
    except Exception as e:
        return {"ok": False, "error": f"cut_failed: {e}"}


def _print_text(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    text = str((payload or {}).get("text", ""))
    if not text:
        return {"ok": False, "error": "text_required"}
    try:
        _apply_common(cfg)
        _PRINTER.text(text)
        if not text.endswith("\n"):
            _PRINTER.text("\n")
        _feed(cfg)
        return {"ok": True, "chars": len(text)}
    except Exception as e:
        return {"ok": False, "error": f"text_print_failed: {e}"}


def _print_test(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    try:
        _apply_common(cfg)
        _PRINTER.text("SarahMemory ESC/POS Test\n")
        _PRINTER.text(f"Driver: {_DRIVER_ID}\n")
        _PRINTER.text(datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S\n"))
        _PRINTER.text("------------------------------\n")
        _PRINTER.text("ABCDEFGHIJKLMNOPQRSTUVWXYZ\n")
        _PRINTER.text("0123456789\n")
        _PRINTER.text("------------------------------\n")
        _feed(cfg)
        if bool((payload or {}).get("cut", True)):
            _cut(cfg, str((payload or {}).get("cut_mode", "full")))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"test_print_failed: {e}"}


def _print_barcode(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    if not bool(cfg.get("barcode_enabled", True)):
        return {"ok": False, "error": "barcode_disabled"}
    data = str((payload or {}).get("data", ""))
    code_type = str((payload or {}).get("code_type", "CODE128"))
    if not data:
        return {"ok": False, "error": "barcode_data_required"}
    try:
        _apply_common(cfg)
        _PRINTER.barcode(data, code_type)
        _feed(cfg)
        return {"ok": True, "code_type": code_type, "data_len": len(data)}
    except Exception as e:
        return {"ok": False, "error": f"barcode_failed: {e}"}


def _print_qr(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    if not bool(cfg.get("qr_enabled", True)):
        return {"ok": False, "error": "qr_disabled"}
    data = str((payload or {}).get("data", ""))
    if not data:
        return {"ok": False, "error": "qr_data_required"}
    try:
        _apply_common(cfg)
        size = int((payload or {}).get("size", 6))
        _PRINTER.qr(data, size=size)
        _feed(cfg)
        return {"ok": True, "data_len": len(data), "size": size}
    except Exception as e:
        return {"ok": False, "error": f"qr_failed: {e}"}


def _print_image(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    if not bool(cfg.get("image_enabled", True)):
        return {"ok": False, "error": "image_disabled"}

    try:
        from PIL import Image
    except Exception as e:
        return {"ok": False, "error": f"pil_unavailable: {e}"}

    src_path = (payload or {}).get("path")
    b64 = (payload or {}).get("image_base64")
    if not src_path and not b64:
        return {"ok": False, "error": "path or image_base64 required"}

    tmp_path = None
    try:
        if b64:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
        else:
            img = Image.open(src_path)
        max_w = int(cfg.get("image_max_width", 512))
        if img.width > max_w:
            ratio = max_w / float(img.width)
            img = img.resize((max_w, max(1, int(img.height * ratio))))
        img = img.convert("L")
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp_path)
        _PRINTER.image(tmp_path)
        _feed(cfg)
        return {"ok": True, "width": img.width, "height": img.height}
    except Exception as e:
        return {"ok": False, "error": f"image_print_failed: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _pulse_drawer(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    if not bool(cfg.get("allow_cash_drawer", False)):
        return {"ok": False, "error": "cash_drawer_disabled"}
    try:
        pin = int((payload or {}).get("pin", cfg.get("drawer_pin", 2)))
        _PRINTER.cashdraw(pin)
        return {"ok": True, "pin": pin}
    except Exception as e:
        return {"ok": False, "error": f"cash_drawer_failed: {e}"}


def _send_raw(cfg, payload):
    check = _require_connected()
    if not check["ok"]:
        return check
    if not bool(cfg.get("allow_raw", False)):
        return {"ok": False, "error": "raw_disabled"}
    raw = (payload or {}).get("raw")
    if raw is None:
        return {"ok": False, "error": "raw payload required"}
    try:
        if isinstance(raw, str):
            if raw.strip() and all(c in "0123456789abcdefABCDEF " for c in raw.strip()):
                data = bytes.fromhex(raw)
            else:
                data = raw.encode(cfg.get("encoding", "cp437"), errors="replace")
        elif isinstance(raw, list):
            data = bytes(int(x) & 0xFF for x in raw)
        else:
            data = bytes(raw)
        _PRINTER._raw(data)
        return {"ok": True, "bytes": len(data)}
    except Exception as e:
        return {"ok": False, "error": f"raw_send_failed: {e}"}


def driver_init(context=None, config=None):
    cfg = _merged_config(config)
    v = driver_validate(cfg)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = _now()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = []
    _SESSION["connected"] = False
    _SESSION["printer_backend"] = None
    _SESSION["device"] = None
    _SESSION["history"] = []
    _SESSION["last_result"] = None
    _SESSION["config"] = cfg
    _note("Driver initialized.")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    global _PRINTER
    try:
        if _PRINTER is not None:
            try:
                _PRINTER.close()
            except Exception:
                pass
    finally:
        _PRINTER = None
        _SESSION["connected"] = False
        _SESSION["printer_backend"] = None
        _SESSION["device"] = None
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
        _note("Shutdown completed.")
    return True


def driver_status(context=None):
    session = {
        "instance_id": _SESSION.get("instance_id"),
        "started_ts": _SESSION.get("started_ts"),
        "status": _SESSION.get("status"),
        "error": _SESSION.get("error"),
        "notes": list(_SESSION.get("notes", [])),
        "connected": bool(_SESSION.get("connected")),
        "printer_backend": _SESSION.get("printer_backend"),
        "device": deepcopy(_SESSION.get("device")),
        "last_result": deepcopy(_SESSION.get("last_result")),
        "history_size": len(_SESSION.get("history", [])),
        "config": _masked_config(_SESSION.get("config", {})),
    }
    return {"ok": True, "session": session}


def driver_action(action_id: str, context=None, payload=None):
    global _PRINTER
    payload = payload or {}
    cfg = _merged_config(_SESSION.get("config", {}))

    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": _now()}
    if action_id == "probe_backends":
        res = _probe_backends()
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "discover_devices":
        res = _discover_devices()
        _record(action_id, res.get("ok"), {"count": len(res.get("devices", [])), "error": res.get("error")})
        return res
    if action_id == "get_config":
        return {"ok": True, "config": _masked_config(cfg)}
    if action_id == "set_config":
        new_cfg = deepcopy(cfg)
        updates = payload.get("config", payload if isinstance(payload, dict) else {})
        if not isinstance(updates, dict):
            return {"ok": False, "error": "config object required"}
        new_cfg.update(updates)
        v = driver_validate(new_cfg)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        _SESSION["config"] = new_cfg
        _record(action_id, True, {"updated_keys": sorted(list(updates.keys()))})
        return {"ok": True, "config": _masked_config(new_cfg), "validate": v}
    if action_id == "connect":
        res = _open_printer(cfg)
        _SESSION["status"] = "ready" if res.get("ok") else "error"
        _SESSION["error"] = None if res.get("ok") else res.get("error")
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "disconnect":
        try:
            if _PRINTER is not None:
                _PRINTER.close()
        except Exception:
            pass
        _PRINTER = None
        _SESSION["connected"] = False
        _SESSION["device"] = None
        _SESSION["printer_backend"] = None
        _record(action_id, True, {})
        return {"ok": True}
    if action_id == "print_text":
        res = _print_text(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "print_test":
        res = _print_test(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "print_barcode":
        res = _print_barcode(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "print_qr":
        res = _print_qr(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "print_image":
        res = _print_image(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "feed":
        res = _feed(cfg, payload.get("lines"))
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "cut":
        res = _cut(cfg, str(payload.get("mode", "full")))
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "cash_drawer":
        res = _pulse_drawer(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "send_raw":
        res = _send_raw(cfg, payload)
        _record(action_id, res.get("ok"), res)
        return res
    if action_id == "safe_stop":
        out = {"ok": True, "steps": []}
        try:
            if _SESSION.get("connected"):
                try:
                    _feed(cfg, 1)
                    out["steps"].append("feed")
                except Exception:
                    pass
                if bool(payload.get("cut", False)) and bool(cfg.get("cut_enabled", True)):
                    try:
                        _cut(cfg, str(payload.get("cut_mode", "partial")))
                        out["steps"].append("cut")
                    except Exception:
                        pass
                try:
                    if _PRINTER is not None:
                        _PRINTER.close()
                    out["steps"].append("disconnect")
                except Exception:
                    pass
            _PRINTER = None
            _SESSION["connected"] = False
            _SESSION["device"] = None
            _SESSION["printer_backend"] = None
            _record(action_id, True, out)
            return out
        except Exception as e:
            out = {"ok": False, "error": f"safe_stop_failed: {e}"}
            _record(action_id, False, out)
            return out
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
