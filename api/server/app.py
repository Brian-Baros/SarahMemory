"""--==The SarahMemory Project==--
File: api/server/app.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

ULTIMATE merged Flask server for SarahMemory (v9.0.0)
==============================================================================================
- Serves Web UI
- Hub (HMAC) endpoints
- Node registration / embeddings / context / jobs
- Leaderboard + wallet (with Ledger module preference + local fallback)
- Settings/Themes/Voices + Contacts + Reminders + Cleanup Tools
- Calendar/Chat History fetchers for Web UI
- File ingest / remote transfer
- Camera/Mic/Voice toggles + simple telecom stubs
- Safe fallbacks against missing core modules
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_server_core"
# CATEGORY = "flask_api_and_webui_runtime"
# USER_FACING = False
# UI_EXPOSURE = "api_surface"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "core"
# HARDWARE_DOMAIN = "filesystem_network_camera_microphone_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "api_server"
# FAMILY = "api_runtime"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Primary Flask API/WebUI server surface for SarahMemory routes, subsystem mounting, safe fallbacks, and governed runtime exposure."
# --- SARAHMETA END ---

import os, sys, json, time, glob, sqlite3, hmac, hashlib, base64, difflib, random, importlib.util, urllib.request, urllib.error, subprocess, signal
from pathlib import Path
from decimal import Decimal
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file, g, session, abort
# --- Flask CORS (safe import for CLI testing & WSGI) ---
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except Exception as e:
    CORS = None  # type: ignore
    _CORS_AVAILABLE = False
    print("[WARN] flask_cors not available:", e)

from dotenv import load_dotenv
load_dotenv()
import re
import jwt
import bcrypt
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
from datetime import datetime, timedelta
import threading
import importlib.util
import logging # Explicitly import logging

# ---------------------------------------------------------------------------
# Path resolution (prefer SarahMemoryGlobals; fallback to local server layout)
# ---------------------------------------------------------------------------
# Configure basic logging for the app.py directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)
logger = app_logger  # consistent alias



# ------------------OLD V8 Root-----------------------
#ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#if ROOT not in sys.path:
#    sys.path.insert(0, ROOT)
#-------------------------------------------------------
# ---------------------------------------------------------------------------
# NEW V8 Root/Path resolution (prefer SarahMemoryGlobals; fallback to local server layout)
# ---------------------------------------------------------------------------

def _find_project_root(start_dir: str, max_up: int = 6) -> str:
    """
    Walk upward from start_dir to locate SarahMemoryGlobals.py (project root marker).
    This fixes cases where app.py runs from /api/server and only adds /api to sys.path.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        marker = os.path.join(cur, "SarahMemoryGlobals.py")
        if os.path.exists(marker):
            return cur
        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)

# Start from app.py directory
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))

# Candidate roots:
# 1) parent (existing behavior)
# 2) grandparent (common: api/server -> api -> project)
# 3) auto-discovered marker walk
ROOT_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
ROOT_GRANDPARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
ROOT_DISCOVERED = _find_project_root(_THIS_DIR)

# Insert best root first
for p in (ROOT_DISCOVERED, ROOT_GRANDPARENT, ROOT_PARENT):
    if p and p not in sys.path:
        sys.path.insert(0, p)



# Attempt to load SarahMemoryGlobals for consistent pathing and versions
try:
    import SarahMemoryGlobals as config
    BASE_DIR = getattr(config, "BASE_DIR", os.getcwd())
    PUBLIC_DIR = getattr(config, "PUBLIC_DIR", os.path.join(BASE_DIR, "public_html"))
    WEB_DIR = getattr(config, "WEB_DIR", os.path.join(PUBLIC_DIR, "web"))
    DATA_DIR = getattr(config, "DATA_DIR", os.path.join(BASE_DIR, "data"))
    PROJECT_VERSION = getattr(config, "PROJECT_VERSION", "9.0.0") # Ensure v9.0.0 as per spec
except Exception as e:
    app_logger.warning(f"SarahMemoryGlobals (config) import failed or missing attributes. Falling back to local defaults: {e}")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /api/server
    PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # /api
    WEB_DIR = PUBLIC_DIR  # serve index.html from /api
    DATA_DIR = os.path.join(BASE_DIR, "data")  # /api/server/data
    PROJECT_VERSION = "9.0.0" # Ensure v9.0.0 as per spec


# Identity / branding (server-side source of truth)
BRAND_NAME = "Sarah"
PLATFORM_NAME = "SarahMemory AiOS"
CREATOR_NAME = "Brian Lee Baros"
ORG_NAME = "SOFTDEV0 LLC"

def _identity_payload():
    return {
        "name": BRAND_NAME,
        "platform": PLATFORM_NAME,
        "version": PROJECT_VERSION,
        "creator": CREATOR_NAME,
        "organization": ORG_NAME,
        "build": "webui-server",
    }

def _is_identity_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    # Do not let generic version wording steal hardware/runtime version questions
    # from SelfAware or system fact lanes.
    version_blockers = (
        "bios", "uefi", "firmware", "motherboard", "mainboard", "baseboard",
        "cpu", "gpu", "driver", "windows", "linux", "python", "node",
        "npm", "cuda", "torch", "pytorch", "chipset", "device", "adapter",
    )
    if "version" in t and any(b in t for b in version_blockers):
        return False

    keys = [
        "what is your name", "who are you", "your name",
        "what version are you", "what version are you running", "your version",
        "server version", "program version", "app version", "sarahmemory version",
        "version number",
        "who made you", "who created you", "creator",
        "who designed you", "designer", "engineer",
        "who engineered you", "who built you",
        "brian lee baros", "softdev0",
    ]

    return any(k in t for k in keys)

# ---------------------------------------------------------------------------
# SelfAware factual system-question bridge
# ---------------------------------------------------------------------------
_SM_V9G_QUERY_PACKET_VERSION = "V10_V9G_CANONICAL_QUERY_PACKET"
_SM_V9G_CORRECTIONS = {
    "temperture": "temperature",
    "tempertrue": "temperature",
    "tempature": "temperature",
    "thermo": "thermal",
    "wether": "weather",
    "wi fi": "wi-fi",
    "wifi": "wi-fi",
    "hardrive": "hard drive",
    "harddrive": "hard drive",
    "moter": "motor",
}


def _sm_v9g_normalize_text(text: str) -> tuple[str, dict]:
    raw = str(text or "")
    lower = raw.strip().lower()
    corrections: dict[str, str] = {}
    for bad, good in _SM_V9G_CORRECTIONS.items():
        if bad in lower:
            lower = lower.replace(bad, good)
            corrections[bad] = good
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower, corrections


def _sm_v9g_contains_any(text: str, words: tuple[str, ...] | list[str]) -> bool:
    return any(w in text for w in words)


def _sm_v9g_component_from_text(norm: str) -> str:
    if _sm_v9g_contains_any(norm, ("cpu", "processor")):
        return "cpu"
    if _sm_v9g_contains_any(norm, ("gpu", "graphics", "video card", "nvidia", "radeon")):
        return "gpu"
    if _sm_v9g_contains_any(norm, ("motherboard", "mainboard", "baseboard", "system board", "board", "chipset", "vrm")):
        return "motherboard"
    if _sm_v9g_contains_any(norm, ("drive", "disk", "disc", "storage", "ssd", "hdd", "nvme")):
        return "drive"
    if "battery" in norm:
        return "battery"
    if _sm_v9g_contains_any(norm, ("motor", "servo", "actuator", "controller")):
        return "motor_controller"
    if _sm_v9g_contains_any(norm, ("ambient", "room", "environment")):
        return "ambient"
    return ""


def _sm_build_canonical_query_packet(text: str, payload: dict | None = None, context_packet: dict | None = None) -> dict:
    raw = str(text or "")
    norm, corrections = _sm_v9g_normalize_text(raw)
    payload = payload or {}
    target = ""
    m = re.search(r"\b([a-zA-Z]):\\?\b", raw)
    if m:
        target = m.group(1).upper() + ":"

    component = _sm_v9g_component_from_text(norm)
    requested_metric = "identity"
    fact_kind = "general_system_fact"
    answer_shape = "summary"
    evidence_visibility = "normal"

    # Metric-first classification.  Metric words outrank component identity words.
    thermal_terms = ("temperature", "temp", "thermal", "heat", "hot", "degrees c", "degrees f", "celsius", "fahrenheit")
    if _sm_v9g_contains_any(norm, thermal_terms):
        requested_metric = "temperature"
        fact_kind = "temperature"
        target = component or target or "body_thermal"
        answer_shape = "direct_answer"
    elif _sm_v9g_contains_any(norm, ("fan", "rpm")):
        requested_metric = "fan_speed"
        fact_kind = "fan_speed"
        answer_shape = "direct_answer"
    elif _sm_v9g_contains_any(norm, ("bios", "uefi", "firmware")) and _sm_v9g_contains_any(norm, ("version", "revision", "release")):
        requested_metric = "bios_version"
        fact_kind = "bios_version"
        target = component or "motherboard"
        answer_shape = "direct_answer"
    elif _sm_v9g_contains_any(norm, ("body map", "body-map", "runtime body", "aios body")):
        requested_metric = "body_map"
        fact_kind = "body_map"
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("network adapter", "network card", "ethernet", "wi-fi", "wifi", "lan", "bluetooth network")):
        requested_metric = "connectivity" if ("ethernet" in norm or "wi-fi" in norm or "wifi" in norm) and re.search(r"\bare\s+you\s+connected|\bconnected\b", norm) else "network_adapters"
        fact_kind = "network"
        answer_shape = "direct_answer" if requested_metric == "connectivity" else "summary"
    elif _sm_v9g_contains_any(norm, ("gpu", "graphics", "video card")):
        requested_metric = "identity"
        fact_kind = "gpu"
        target = target or component
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("cpu", "processor")):
        requested_metric = "identity"
        fact_kind = "cpu"
        target = target or component
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("motherboard", "mainboard", "baseboard", "system board")):
        requested_metric = "identity"
        fact_kind = "motherboard"
        target = target or component
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("ram", "memory")):
        requested_metric = "memory_status"
        fact_kind = "memory"
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("disk", "disc", "drive", "storage", "space", "free gb", "used gb")):
        requested_metric = "storage_status"
        fact_kind = "disk_space"
        answer_shape = "summary"
    elif _sm_v9g_contains_any(norm, ("usb", "drive label", "volume label", "label on")):
        requested_metric = "label"
        fact_kind = "usb_label"
        answer_shape = "summary"

    self_scope = bool(
        _sm_v9g_contains_any(norm, (
            "my ", "your ", "you using", "am i using", "are you using", "system", "machine", "computer", "pc",
            "runtime", "body map", "body-map", "hardware", "motherboard", "cpu", "processor", "gpu", "graphics",
            "ram", "memory", "fan", "rpm", "temperature", "temp", "thermal", "network adapter", "ethernet", "wi-fi",
            "python version", "node name", "hostname", "bios", "uefi", "firmware",
        ))
    )
    fact_scope = fact_kind != "general_system_fact" or self_scope
    weather_phrases = ("outside", "weather", "forecast", "rain", "humidity", "wind chill", "heat index")
    if _sm_v9g_contains_any(norm, weather_phrases) and not _sm_v9g_contains_any(norm, ("cpu", "gpu", "fan", "drive", "disk", "usb", "system", "motherboard")):
        fact_scope = False

    return {
        "packet_type": "CanonicalQueryPacket",
        "version": _SM_V9G_QUERY_PACKET_VERSION,
        "raw_text": raw,
        "normalized_text": norm,
        "corrections": corrections,
        "domain": "selfaware_body" if fact_scope else "chat",
        "intent": "body_fact_query" if fact_scope else "general_chat",
        "requested_component": component or target,
        "requested_metric": requested_metric,
        "fact_kind": fact_kind,
        "target": target,
        "answer_shape": answer_shape,
        "evidence_visibility": evidence_visibility,
        "volatile_runtime_fact": bool(fact_scope),
        "do_not_write_sql": bool(fact_scope),
        "do_not_persist": bool(fact_scope),
        "do_not_learn": bool(fact_scope),
        "read_only": True,
        "action_taken": False,
    }


def _sm_is_selfaware_fact_question(text: str) -> bool:
    pkt = _sm_build_canonical_query_packet(text)
    return pkt.get("domain") == "selfaware_body"


def _sm_selfaware_fact_kind_and_target(text: str) -> tuple[str, str]:
    pkt = _sm_build_canonical_query_packet(text)
    return str(pkt.get("fact_kind") or "general_system_fact"), str(pkt.get("target") or "")


def _sm_compact_json_value(value, *, max_chars: int = 1600) -> str:
    try:
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (int, float, bool)) or value is None:
            text = str(value)
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


def _sm_v9g_component_label(value: str) -> str:
    v = str(value or "").strip().lower().replace("_", " ")
    labels = {
        "cpu": "CPU",
        "gpu": "GPU",
        "motherboard": "motherboard",
        "body thermal": "body thermal",
        "drive": "drive",
        "battery": "battery",
        "motor controller": "motor-controller",
        "ambient": "ambient",
    }
    return labels.get(v, v or "component")


def _sm_v9g_clean_denial(kind: str, claim: str, ticket: dict) -> str:
    kind = str(kind or "system_fact").lower()
    low = str(claim or "").lower()
    if kind == "fan_speed":
        return "I cannot verify fan RPM from the currently exposed sensors. No mapped fan-speed sensor is available in this runtime."
    if kind == "temperature":
        comp = str((ticket.get("target") or "") or "component").replace("_", " ")
        if "cpu" in low or comp == "cpu":
            return "I cannot verify CPU temperature from the currently exposed direct or mapped motherboard CPU-related sensors. I will not substitute GPU or generic thermal readings as CPU temperature."
        return f"I cannot verify a mapped {_sm_v9g_component_label(comp)} temperature sensor from the currently exposed evidence."
    if kind == "bios_version":
        return "I can identify the motherboard only if evidence is available, but I do not currently have a verified BIOS/UEFI version witness."
    if kind in {"network", "network_card", "wifi_card", "ethernet_card", "bluetooth_card", "lan"}:
        return "I cannot verify the requested network hardware state from the current evidence packet."
    return f"I cannot verify that {kind.replace('_', ' ')} fact from the current evidence packet. I will not guess."


def _sm_v9g_network_direct_answer(text: str, value: object) -> str | None:
    low = str(text or "").lower()
    if not ("connected" in low and ("ethernet" in low or "wi-fi" in low or "wifi" in low)):
        return None
    if not isinstance(value, dict):
        return None
    active = value.get("active_adapters") if isinstance(value.get("active_adapters"), list) else []
    inactive = value.get("inactive_adapters") if isinstance(value.get("inactive_adapters"), list) else []
    active_names = [str(a.get("name") or "") for a in active if isinstance(a, dict)]
    inactive_names = [str(a.get("name") or "") for a in inactive if isinstance(a, dict)]
    ethernet_active = any("ethernet" in n.lower() or "lan" in n.lower() for n in active_names)
    wifi_active = any("wi" in n.lower() or "wireless" in n.lower() for n in active_names)
    wifi_present = wifi_active or any("wi" in n.lower() or "wireless" in n.lower() for n in inactive_names)
    if ethernet_active and wifi_active:
        return "I currently have both Ethernet and Wi-Fi active. Sensitive IP and MAC details are redacted."
    if ethernet_active:
        return "I am currently connected through Ethernet. Wi-Fi is present but inactive." if wifi_present else "I am currently connected through Ethernet. I do not see an active Wi-Fi connection."
    if wifi_active:
        return "I am currently connected through Wi-Fi. I do not see an active Ethernet connection."
    return "I do not currently see an active Ethernet or Wi-Fi connection in the verified adapter summary."


def _sm_format_selfaware_fact_reply(ticket: dict) -> str:
    claim = str(ticket.get("claim") or "requested system fact").strip()
    decision = str(ticket.get("decision") or "UNKNOWN").upper()
    kind = str(ticket.get("requested_fact") or "system_fact").strip().lower()
    value = ticket.get("majority_value")
    pv = ticket.get("presentation_value")

    presentation_text = str(ticket.get("presentation_text") or "").strip()
    if presentation_text:
        blocked = ("verified selfaware fact", "selfaware could not verify", "verdict:", "quorum", "denied_no_evidence", "deniednoevidence", "cpu =", "gpu =", "motherboard =")
        if not any(b in presentation_text.lower() for b in blocked):
            return presentation_text

    if decision == "APPROVED_FACT":
        if kind == "temperature":
            tv = pv if isinstance(pv, dict) else value
            if isinstance(tv, dict):
                selected = tv.get("selected_reading") if isinstance(tv.get("selected_reading"), dict) else {}
                component = str(tv.get("requested_component") or selected.get("component") or ticket.get("target") or "thermal").replace("_", " ")
                temp = selected.get("temperature_c")
                source_type = str(selected.get("source_type") or "thermal_sensor").replace("_", " ").lower()
                if temp not in (None, ""):
                    if component.lower() == "cpu" and "motherboard" in source_type:
                        return f"I do not currently have a direct CPU temperature reading from a CPU thermal probe. This CPU is verified on my motherboard, and the motherboard exposes a CPU-related thermal sensor. Based on that verified board sensor, my CPU temperature is currently {temp}°C."
                    return f"My currently verified {_sm_v9g_component_label(component)} temperature is {temp}°C."
            return _sm_v9g_clean_denial(kind, claim, ticket)

        if kind == "cpu":
            if isinstance(value, dict):
                name = str(value.get("name") or value.get("Name") or "Unknown CPU").strip()
                cores = value.get("physical_cores") or value.get("NumberOfCores")
                threads = value.get("logical_threads") or value.get("NumberOfLogicalProcessors")
                clock = value.get("max_clock_mhz") or value.get("MaxClockSpeed") or value.get("current_clock_mhz")
                details = []
                if cores not in (None, ""): details.append(f"{cores} physical cores")
                if threads not in (None, ""): details.append(f"{threads} logical threads")
                if clock not in (None, ""): details.append(f"clock about {clock} MHz")
                return f"I currently have {name}" + (f" ({', '.join(details)})." if details else ".")
            return f"I currently have {_sm_compact_json_value(value)}."

        if kind == "gpu":
            if isinstance(value, dict):
                name = str(value.get("name") or value.get("Name") or "Unknown GPU").strip()
                temp = value.get("temperature_c")
                util = value.get("utilization_pct")
                vram = value.get("vram_total_mb") or value.get("memory")
                details = []
                if temp not in (None, ""): details.append(f"{temp}°C")
                if util not in (None, ""): details.append(f"{util}% utilization")
                if vram not in (None, ""): details.append(f"VRAM {vram} MB" if str(vram).isdigit() else f"VRAM {vram}")
                return f"My currently verified graphics hardware is {name}" + (f" ({', '.join(details)})." if details else ".")
            return f"My currently verified graphics hardware is {_sm_compact_json_value(value)}."

        if kind in {"network", "network_card", "wifi_card", "ethernet_card", "bluetooth_card", "lan"}:
            direct = _sm_v9g_network_direct_answer(claim, pv if isinstance(pv, dict) else value)
            if direct:
                return direct
            return str(presentation_text or f"My currently verified network adapter summary is: {_sm_compact_json_value(pv if pv not in (None, '') else value)}")

        if kind == "motherboard":
            return f"My currently verified motherboard is {_sm_compact_json_value(value)}."
        if kind == "memory":
            return f"My currently verified memory status is: {_sm_compact_json_value(pv if pv not in (None, '') else value)}."
        if kind in {"disk_space", "storage_topology", "storage_devices"}:
            return f"My currently verified storage status is: {_sm_compact_json_value(pv if pv not in (None, '') else value)}."
        if kind == "bios_version":
            return f"My currently verified BIOS/UEFI version is {_sm_compact_json_value(value)}."
        return f"My currently verified {kind.replace('_', ' ')} is: {_sm_compact_json_value(pv if pv not in (None, '') else value)}."

    # Partial/denied cases are still useful evidence states, but normal chat must not expose courtroom terms.
    if decision in {"ESCALATE_HIGH_REVIEW", "DENIED_WEAK_EVIDENCE", "DENIED_NO_EVIDENCE", "DENIEDNOEVIDENCE"}:
        return _sm_v9g_clean_denial(kind, claim, ticket)

    return _sm_v9g_clean_denial(kind, claim, ticket)


def _sm_import_appself_runtime():
    """Load the exact api/server/appself.py beside this app.py.

    This deliberately avoids sys.modules and normal import resolution because older
    appself modules can remain loaded during local restart/build cycles. The HTTP
    /api/self/fact-check endpoint already proves appself.py can produce the correct
    quorum; Chat must use that same physical file, not a stale module object.
    """
    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        server_dir = os.getcwd()

    appself_path = os.path.join(server_dir, "appself.py")
    if not os.path.exists(appself_path):
        raise RuntimeError(f"appself.py not found beside app.py: {appself_path}")

    module_name = f"_sarahmemory_runtime_appself_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, appself_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create import spec for appself.py")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not callable(getattr(mod, "run_selfaware_fact_check", None)) and not callable(getattr(mod, "_run_fact_ticket", None)):
        raise RuntimeError("runtime appself fact-ticket runner unavailable after direct load")
    return mod

def _sm_try_selfaware_fact_route(text: str, *, source: str = "api_chat") -> dict | None:
    """Route local hardware/runtime fact questions into appself's fact-ticket engine.

    This keeps ChatPanel factual system questions out of the sidekick/Neuron
    fallback path and forces them through the same SelfAware 3-source evidence
    court used by /api/self/fact-check.
    """
    # Identity is not body telemetry. Keep identity on the identity lane so
    # SelfAware cannot hijack stable name/version/creator responses just because
    # the wording contains "your" or "you".
    if _is_identity_question(text):
        return None

    canonical_packet = _sm_build_canonical_query_packet(text)
    if canonical_packet.get("domain") != "selfaware_body":
        return None

    kind = str(canonical_packet.get("fact_kind") or "general_system_fact")
    target = str(canonical_packet.get("target") or "")
    court_claim = str(canonical_packet.get("normalized_text") or text)
    try:
        _appself = _sm_import_appself_runtime()

        run_public = getattr(_appself, "run_selfaware_fact_check", None)
        run_private = getattr(_appself, "_run_fact_ticket", None)

        if callable(run_public):
            ticket = run_public(
                claim=text,
                kind=kind,
                target=target,
                source=source,
                meta={"source": source, "route": "api_chat_selfaware_fact", "bridge": "runtime_appself_public", "do_not_write_sql": True, "do_not_persist": True, "do_not_learn": True},
            )
        elif callable(run_private):
            ticket = run_private(
                claim=text,
                kind=kind,
                target=target,
                source=source,
                ticket_kind="SELF_FACT_TICKET",
                meta={"source": source, "route": "api_chat_selfaware_fact", "bridge": "runtime_appself_private", "do_not_write_sql": True, "do_not_persist": True, "do_not_learn": True},
            )
        else:
            raise RuntimeError("appself fact-ticket runner unavailable")

        if not isinstance(ticket, dict):
            raise RuntimeError("appself returned non-dict ticket")

        # Defensive: if a simple CPU/GPU/storage question weak-fails in chat while
        # /api/self/fact-check succeeds, record the module path for diagnosis.
        try:
            ticket.setdefault("meta", {})
            if isinstance(ticket.get("meta"), dict):
                ticket["meta"]["appself_module_file"] = str(getattr(_appself, "__file__", ""))
        except Exception:
            pass

        reply = _sm_format_selfaware_fact_reply(ticket)
        compare_result = {"accepted": True, "decision": "COMPARE_NOT_RUN"}
        try:
            import SarahMemoryCompare as _SMCompare  # type: ignore
            fn = getattr(_SMCompare, "compare_selfaware_answer_contract", None)
            if callable(fn):
                compare_result = fn(text, reply, canonical_packet=canonical_packet, meta={"source": "api_chat_selfaware_fact"})
                if isinstance(compare_result, dict) and not bool(compare_result.get("accepted", True)):
                    # Re-anchor response to the original metric/component instead of leaking a mismatched answer.
                    reply = _sm_v9g_clean_denial(kind, text, {"target": target, "decision": ticket.get("decision")})
        except Exception as _cmp_exc:
            compare_result = {"accepted": True, "decision": "COMPARE_UNAVAILABLE", "error": str(_cmp_exc)}
        bundle = _sm_make_outward_bundle(
            _sm_present_text(reply, intent="system_status", meta={"source": "selfaware_fact_ticket"}),
            meta={
                "source": "selfaware_fact_ticket",
                "engine": "appself.fact_ticket_runner",
                "intent": "system_status",
                "fact_kind": kind,
                "target": target,
                "canonical_query_packet": canonical_packet,
                "answer_shape": canonical_packet.get("answer_shape"),
                "requested_metric": canonical_packet.get("requested_metric"),
                "ticket_id": ticket.get("ticket_id"),
                "decision": ticket.get("decision"),
                "quorum": ticket.get("quorum"),
                "confidence": ticket.get("confidence"),
                "approved_fact": bool(ticket.get("approved_fact")),
                "appself_module_file": str(getattr(_appself, "__file__", "")),
                "version": PROJECT_VERSION,
                "compare_result": compare_result,
            },
            raw_answer=reply,
        )
        bundle["ok"] = True
        bundle.setdefault("actions", [])
        bundle["actions"].append({
            "type": "selfaware_fact_ticket",
            "ticket_id": ticket.get("ticket_id"),
            "decision": ticket.get("decision"),
            "quorum": ticket.get("quorum"),
            "requested_fact": ticket.get("requested_fact"),
        })
        return bundle
    except Exception as exc:
        app_logger.warning("SelfAware fact route failed: %s", exc, exc_info=True)
        bundle = _sm_make_outward_bundle(
            "SelfAware fact route is available, but this fact check failed internally. I did not guess the answer.",
            meta={
                "source": "selfaware_fact_ticket_error",
                "engine": "appself.fact_ticket_runner",
                "intent": "system_status",
                "fact_kind": kind,
                "target": target,
                "error": str(exc),
                "version": PROJECT_VERSION,
            },
            errors=[str(exc)],
        )
        bundle["ok"] = False
        return bundle

# Prefer server/static as templates if the SPA build exists
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SERVER_DIR, "static")
TEMPLATE_DIR = SERVER_DIR if os.path.exists(os.path.join(STATIC_DIR, "index.html")) else WEB_DIR

# Web UI dist root (Lovable/Vite build output)
# Expected: <PROJECT_ROOT>/data/ui/v8/
UI_DIST_DIR = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "data", "ui", "v8"))
WALLETS_DIR = os.path.join(DATA_DIR, "wallets")
META_DB = os.path.join(DATA_DIR, "meta.db") # merged meta DB
LOGS_DIR = os.path.join(DATA_DIR, "logs") # Default to DATA_DIR/logs

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(WALLETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Global runtime state (kept intentionally small and fast)
# ---------------------------------------------------------------------------
APP_VERSION = PROJECT_VERSION  # API/UI convenience alias

# Persistent state file (safe JSON, kept in DATA_DIR)
STATE_DB = os.path.join(DATA_DIR, "server_state.json")  # JSON, not sqlite
WALLET_DB = os.path.join(DATA_DIR, "wallets.db")        # sqlite (created on demand)

# Simple feature toggles (web UI can control these)
MIC_ON = True
TTS_ON = True

MIC_ENABLED = MIC_ON
TTS_ENABLED = TTS_ON
VOICE_OUTPUT_ON = TTS_ON
VOICE_OUTPUT_ENABLED = TTS_ON
# Small in-memory cache for hot endpoints (rankings/wallet/etc.)
_CACHE = {}

# Session-scoped live vision frame cache for Custom / Web UI handoff.
# Stores a lightweight latest-frame snapshot per UI session so /api/chat can
# attach the newest frame into the governed Context Packet without changing
# non-vision routes.
_VISION_FRAME_LOCK = threading.Lock()
_VISION_FRAME_CACHE: dict[str, dict] = {}
_VISION_FRAME_MAX_AGE_S = int(os.getenv("SM_VISION_FRAME_MAX_AGE_S", "45") or 45)
_VISION_FRAME_MAX_CHARS = int(os.getenv("SM_VISION_FRAME_MAX_CHARS", "1800000") or 1800000)
def _get_or_create_session_id(payload: dict | None = None) -> str:
    """Return a stable session identifier for UI->API coordination."""
    payload = payload or {}
    for key in ("session_id", "sid"):
        val = str(payload.get(key) or "").strip()
        if val:
            try:
                session["sm_session_id"] = val
            except Exception:
                pass
            return val
    header_sid = str(request.headers.get("X-Session-Id") or request.headers.get("X-Session-ID") or "").strip()
    if header_sid:
        try:
            session["sm_session_id"] = header_sid
        except Exception:
            pass
        return header_sid
    try:
        sid = str(session.get("sm_session_id") or "").strip()
    except Exception:
        sid = ""
    if sid:
        return sid
    sid = secrets.token_urlsafe(18)
    try:
        session["sm_session_id"] = sid
    except Exception:
        pass
    return sid
def _normalize_vision_frame_payload(payload: dict | None = None) -> dict | None:
    """Accept several frontend frame shapes and normalize to one dict."""
    payload = payload or {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    candidates = [
        payload.get("frame"),
        payload.get("image"),
        payload.get("image_data"),
        payload.get("imageData"),
        payload.get("image_base64"),
        payload.get("imageBase64"),
        payload.get("data_url"),
        payload.get("dataUrl"),
        payload.get("vision_frame"),
        payload.get("latest_frame"),
        meta.get("frame"),
        meta.get("image"),
        meta.get("image_data"),
        meta.get("imageData"),
        meta.get("image_base64"),
        meta.get("imageBase64"),
        meta.get("data_url"),
        meta.get("dataUrl"),
        meta.get("vision_frame"),
        meta.get("latest_frame"),
    ]
    frame_value = None
    for cand in candidates:
        if isinstance(cand, dict):
            inner = cand.get("image") or cand.get("imageBase64") or cand.get("image_base64") or cand.get("dataUrl") or cand.get("data_url") or cand.get("frame")
            if inner:
                frame_value = inner
                break
        elif isinstance(cand, str) and cand.strip():
            frame_value = cand.strip()
            break
    if not frame_value:
        return None
    if not isinstance(frame_value, str):
        try:
            frame_value = str(frame_value)
        except Exception:
            return None
    frame_value = frame_value.strip()
    if not frame_value:
        return None
    if len(frame_value) > _VISION_FRAME_MAX_CHARS:
        app_logger.warning("Vision frame rejected: payload too large (%s chars)", len(frame_value))
        return None
    return {
        "frame": frame_value,
        "ts": float(payload.get("ts") or meta.get("ts") or time.time()),
        "source": str(payload.get("source") or meta.get("source") or "ui").strip() or "ui",
        "width": payload.get("width") or meta.get("width"),
        "height": payload.get("height") or meta.get("height"),
        "mime": str(payload.get("mime") or meta.get("mime") or "image/jpeg").strip() or "image/jpeg",
    }
def _prune_vision_frame_cache(now_ts: float | None = None) -> None:
    now_ts = float(now_ts or time.time())
    stale_before = now_ts - float(_VISION_FRAME_MAX_AGE_S)
    with _VISION_FRAME_LOCK:
        for sid, rec in list(_VISION_FRAME_CACHE.items()):
            try:
                if float(rec.get("ts") or 0.0) < stale_before:
                    _VISION_FRAME_CACHE.pop(sid, None)
            except Exception:
                _VISION_FRAME_CACHE.pop(sid, None)
def _store_latest_vision_frame(session_id: str, frame_payload: dict) -> dict:
    rec = dict(frame_payload or {})
    rec["session_id"] = session_id
    rec["stored_ts"] = time.time()
    _prune_vision_frame_cache(rec["stored_ts"])
    with _VISION_FRAME_LOCK:
        _VISION_FRAME_CACHE[session_id] = rec
    return rec
def _get_latest_vision_frame(session_id: str, *, max_age_s: int | None = None) -> dict | None:
    if not session_id:
        return None
    max_age = int(max_age_s or _VISION_FRAME_MAX_AGE_S)
    stale_before = time.time() - max(1, max_age)
    try:
        with _VISION_FRAME_LOCK:
            rec = _VISION_FRAME_CACHE.get(session_id)
            if not rec:
                return None
            ts = float(rec.get("ts") or rec.get("stored_ts") or 0.0)
            if ts < stale_before:
                _VISION_FRAME_CACHE.pop(session_id, None)
                return None
            return dict(rec)
    except Exception:
        return None

def _sm_text_looks_like_visual_request(text: str, payload: dict | None = None, context_packet: dict | None = None) -> bool:
    """Return True when chat text needs the newest backend vision frame."""
    payload = payload if isinstance(payload, dict) else {}
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    meta = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}

    if bool(payload.get("force_latest_vision") or payload.get("use_latest_vision") or payload.get("vision_request")):
        return True
    if str(payload.get("intent") or meta.get("intent") or "").strip().lower() in {"vision", "visual", "camera", "scene"}:
        return True

    t = str(text or payload.get("text") or payload.get("message") or payload.get("q") or "").strip().lower()
    if not t:
        return False

    visual_phrases = (
        "what do you see", "what can you see", "describe what you see", "show me what you see",
        "can you see me", "do you see me", "look at me", "look at this", "look at that",
        "what color", "what colour", "color of", "colour of",
        "what is in my hand", "what's in my hand", "what am i holding", "what object is in my hand",
        "in my hand", "in my hands", "holding up", "holding",
        "do i have", "am i wearing", "what am i wearing", "what is on my",
        "shirt", "hat", "cap", "headset", "glasses", "face", "hand", "hands",
        "behind me", "in front of me", "left of me", "right of me", "next to me",
        "scene", "webcam", "camera", "frame", "object", "detect", "recognize", "recognise",
        "read this", "read the text", "text on", "say on", "ocr",
    )
    return any(p in t for p in visual_phrases)


def _sm_parse_appvision_ts(value: object) -> float:
    """Best-effort timestamp parser for appvision ISO/epoch timestamps."""
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except Exception:
        pass
    try:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


class _SMAppVisionGlobalsProxy:
    """Attribute proxy around mounted appvision.py blueprint globals."""
    def __init__(self, globals_dict: dict):
        self._globals_dict = globals_dict

    def __getattr__(self, name: str):
        if name in self._globals_dict:
            return self._globals_dict[name]
        raise AttributeError(name)


def _sm_get_appvision_proxy_from_flask_routes():
    """Find the mounted appvision blueprint globals from Flask's URL map."""
    try:
        from flask import current_app, has_app_context  # type: ignore
        if not has_app_context():
            return None
        view_functions = getattr(current_app, "view_functions", {}) or {}
        for endpoint, fn in list(view_functions.items()):
            endpoint_s = str(endpoint or "").lower()
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                continue
            if (
                "appvision" in endpoint_s
                or (
                    isinstance(g.get("_FRAME_CACHE"), dict)
                    and str(g.get("SMHUD_SCHEMA_VERSION") or "") == "SMHUD_PACKET_V1"
                )
            ):
                if isinstance(g.get("_FRAME_CACHE"), dict) or callable(g.get("get_latest_cached_frame_for_chat")):
                    return _SMAppVisionGlobalsProxy(g)
    except Exception:
        return None
    return None


def _sm_get_appvision_module_for_chat():
    """Return the live appvision module/proxy mounted in this Flask process.

    This avoids direct-loading appvision.py, which would create a second module
    instance with an empty frame cache. Chat must read the same cache used by
    /api/vision/frame/latest and the VR HUD renderer.
    """
    candidates = []
    try:
        if globals().get("_appvision") is not None:
            candidates.append(globals().get("_appvision"))
    except Exception:
        pass

    for name in ("appvision", "api.server.appvision", "server.appvision"):
        try:
            mod = sys.modules.get(name)
            if mod is not None:
                candidates.append(mod)
        except Exception:
            pass

    proxy = _sm_get_appvision_proxy_from_flask_routes()
    if proxy is not None:
        candidates.append(proxy)

    seen = set()
    for mod in candidates:
        if mod is None:
            continue
        ident = id(mod)
        if ident in seen:
            continue
        seen.add(ident)
        if hasattr(mod, "_FRAME_CACHE") or hasattr(mod, "get_latest_cached_frame_for_chat"):
            return mod
    return None


def _sm_get_appvision_frame_latest_http_fallback(*, max_age_s: int | None = None) -> dict | None:
    """Last-resort local read of /api/vision/frame/latest.

    Used only if the in-process module/proxy cannot be resolved. It remains
    read-only and does not open camera hardware.
    """
    try:
        base_url = ""
        try:
            base_url = str(request.host_url or "").rstrip("/")
        except Exception:
            base_url = ""
        if not base_url:
            base_url = str(os.getenv("SARAHMEMORY_LOCAL_API_BASE") or "http://127.0.0.1:8000").rstrip("/")
        url = base_url + "/api/vision/frame/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=0.75) as resp:
            raw = resp.read(2_200_000)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(data, dict) or not bool(data.get("has_frame")):
            return None
        frame_value = data.get("data_url") or data.get("image_b64")
        if not frame_value:
            return None
        if data.get("image_b64") and not str(frame_value).startswith("data:image"):
            frame_value = "data:image/jpeg;base64," + str(frame_value)
        ts_epoch = _sm_parse_appvision_ts(data.get("image_cached_ts") or data.get("ts"))
        max_age = int(max_age_s or _VISION_FRAME_MAX_AGE_S)
        if ts_epoch and (time.time() - ts_epoch) > max(1, max_age):
            return None
        return {
            "frame": frame_value,
            "ts": ts_epoch or time.time(),
            "source": str(data.get("source") or "appvision.frame_latest_http"),
            "width": data.get("width"),
            "height": data.get("height"),
            "mime": str(data.get("mime") or "image/jpeg"),
            "frame_id": data.get("frame_id"),
            "backend_cache": "appvision_http",
        }
    except Exception:
        return None


def _get_latest_appvision_frame_for_chat(*, max_age_s: int | None = None) -> dict | None:
    """Bridge Chat to appvision.py's governed live frame cache."""
    mod = _sm_get_appvision_module_for_chat()
    if mod is None:
        return _sm_get_appvision_frame_latest_http_fallback(max_age_s=max_age_s)

    try:
        helper = getattr(mod, "get_latest_cached_frame_for_chat", None)
        if callable(helper):
            rec = helper(max_age_s=max_age_s)
            if isinstance(rec, dict) and (rec.get("frame") or rec.get("data_url") or rec.get("image_b64")):
                frame_value = rec.get("frame") or rec.get("data_url") or rec.get("image_b64")
                if rec.get("image_b64") and not str(frame_value).startswith("data:image"):
                    frame_value = "data:image/jpeg;base64," + str(frame_value)
                return {
                    "frame": frame_value,
                    "ts": rec.get("ts") or rec.get("image_cached_ts") or time.time(),
                    "source": rec.get("source") or "appvision.frame_latest",
                    "width": rec.get("width"),
                    "height": rec.get("height"),
                    "mime": rec.get("mime") or "image/jpeg",
                    "frame_id": rec.get("frame_id"),
                    "backend_cache": "appvision",
                    "hud_packet_id": rec.get("hud_packet_id"),
                }
    except Exception:
        pass

    try:
        lock = getattr(mod, "_FRAME_LOCK", None)
        cache = getattr(mod, "_FRAME_CACHE", None)
        if not isinstance(cache, dict):
            return _sm_get_appvision_frame_latest_http_fallback(max_age_s=max_age_s)
        if lock is not None and hasattr(lock, "__enter__"):
            with lock:
                rec = dict(cache)
        else:
            rec = dict(cache)
    except Exception:
        return _sm_get_appvision_frame_latest_http_fallback(max_age_s=max_age_s)

    if not bool(rec.get("has_frame")):
        return _sm_get_appvision_frame_latest_http_fallback(max_age_s=max_age_s)

    frame_value = rec.get("data_url") or rec.get("image_b64")
    if not frame_value:
        return _sm_get_appvision_frame_latest_http_fallback(max_age_s=max_age_s)
    if rec.get("image_b64") and not str(frame_value).startswith("data:image"):
        frame_value = "data:image/jpeg;base64," + str(frame_value)

    ts_value = rec.get("image_cached_ts") or rec.get("ts")
    ts_epoch = _sm_parse_appvision_ts(ts_value)
    max_age = int(max_age_s or _VISION_FRAME_MAX_AGE_S)
    if ts_epoch and (time.time() - ts_epoch) > max(1, max_age):
        return None

    return {
        "frame": frame_value,
        "ts": ts_epoch or time.time(),
        "source": str(rec.get("source") or "appvision.frame_latest"),
        "width": rec.get("width"),
        "height": rec.get("height"),
        "mime": str(rec.get("mime") or "image/jpeg"),
        "frame_id": rec.get("frame_id"),
        "backend_cache": "appvision",
        "hud_packet_id": (rec.get("hud_packet") or {}).get("packet_id") if isinstance(rec.get("hud_packet"), dict) else None,
    }



def _sm_text_looks_like_desktop_visual_request(text: str, payload: dict | None = None, context_packet: dict | None = None) -> bool:
    """Return True when chat text specifically asks about the desktop/screen feed."""
    payload = payload if isinstance(payload, dict) else {}
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    meta = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}

    if bool(payload.get("force_latest_desktop") or payload.get("use_latest_desktop") or payload.get("desktop_request")):
        return True
    if str(payload.get("intent") or meta.get("intent") or "").strip().lower() in {"desktop", "screen", "desktop_mirror", "screen_mirror"}:
        return True

    t = str(text or payload.get("text") or payload.get("message") or payload.get("q") or "").strip().lower()
    if not t:
        return False

    desktop_phrases = (
        "my desktop", "the desktop", "desktop mirror", "desktop feed",
        "my screen", "the screen", "screen capture", "screen mirror", "monitor feed",
        "what is on my screen", "what's on my screen", "what do you see on my screen",
        "what is on my desktop", "what's on my desktop", "look at my desktop", "look at my screen",
        "read my screen", "read the screen", "read this screen", "read this window",
        "active window", "current window", "open window", "what window", "what app is open",
    )
    return any(p in t for p in desktop_phrases)


def _get_latest_desktop_frame_for_chat(*, max_age_s: int | None = None, auto_capture: bool = True) -> dict | None:
    """Bridge Chat to SarahMemoryDesktop's latest screen frame cache.

    This stays read-only. It does not perform desktop actions and does not enable
    OperatorCore execution. If desktop capture is unavailable, it returns None so
    existing camera/appvision behavior can continue.
    """
    try:
        import SarahMemoryDesktop as _SMDesktop  # type: ignore
        rt = _SMDesktop.get_desktop_runtime()
        rec = rt.latest(include_image=True, auto_capture=auto_capture)
        if not isinstance(rec, dict) or not bool(rec.get("has_frame")):
            return None
        ts = float(rec.get("ts") or time.time())
        max_age = int(max_age_s or _VISION_FRAME_MAX_AGE_S)
        if ts and (time.time() - ts) > max(1, max_age):
            return None
        frame_value = rec.get("data_url") or rec.get("frame")
        if not frame_value and rec.get("image_b64"):
            frame_value = "data:" + str(rec.get("mime") or "image/jpeg") + ";base64," + str(rec.get("image_b64"))
        if not frame_value:
            return None
        return {
            "frame": frame_value,
            "ts": ts,
            "source": "desktop_mirror.latest",
            "width": rec.get("width"),
            "height": rec.get("height"),
            "mime": rec.get("mime") or "image/jpeg",
            "frame_id": rec.get("frame_id"),
            "backend_cache": "desktop_mirror",
            "desktop_observe_only": True,
        }
    except Exception as exc:
        try:
            app_logger.debug("Desktop frame bridge unavailable: %s", exc)
        except Exception:
            pass
        return None


def _attach_cached_or_inline_vision_frame(payload: dict, context_packet: dict, user_text: str = "") -> tuple[dict, dict | None]:
    """Attach the freshest available frame into the Context Packet meta block.

    Priority:
    1) Inline frame/image in the chat payload.
    2) app.py's older session-scoped /api/vision/frame cache.
    3) appvision.py's governed /api/vision/frame/submit cache, only for visual prompts.
    """
    payload = payload if isinstance(payload, dict) else {}
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    meta_block = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}
    frame_rec = _normalize_vision_frame_payload(payload)
    session_id = str(context_packet.get("session_id") or _get_or_create_session_id(payload)).strip()

    if frame_rec is not None and session_id:
        frame_rec = _store_latest_vision_frame(session_id, frame_rec)
    elif session_id:
        frame_rec = _get_latest_vision_frame(session_id)

    bridge = "inline_or_session"
    desktop_visual_request = _sm_text_looks_like_desktop_visual_request(user_text, payload=payload, context_packet=context_packet)
    visual_request = _sm_text_looks_like_visual_request(user_text, payload=payload, context_packet=context_packet)

    if not frame_rec and desktop_visual_request:
        frame_rec = _get_latest_desktop_frame_for_chat(max_age_s=_VISION_FRAME_MAX_AGE_S, auto_capture=True)
        bridge = str((frame_rec or {}).get("backend_cache") or "desktop_mirror")
        if frame_rec is not None and session_id:
            try:
                frame_rec = _store_latest_vision_frame(session_id, frame_rec)
            except Exception:
                pass

    if not frame_rec and visual_request:
        frame_rec = _get_latest_appvision_frame_for_chat(max_age_s=_VISION_FRAME_MAX_AGE_S)
        bridge = str((frame_rec or {}).get("backend_cache") or "appvision")
        if frame_rec is not None and session_id:
            try:
                frame_rec = _store_latest_vision_frame(session_id, frame_rec)
            except Exception:
                pass

    if not frame_rec:
        meta_block["vision_frame_bridge"] = {
            "attached": False,
            "reason": "no_cached_or_inline_frame",
            "visual_request": visual_request,
            "desktop_visual_request": desktop_visual_request,
        }
        context_packet["meta"] = meta_block
        return context_packet, None

    frame_value = frame_rec.get("frame")
    meta_block["frame"] = frame_value
    meta_block["latest_frame"] = frame_value
    meta_block["vision_frame"] = {
        "ts": frame_rec.get("ts"),
        "source": frame_rec.get("source"),
        "width": frame_rec.get("width"),
        "height": frame_rec.get("height"),
        "mime": frame_rec.get("mime"),
        "frame_id": frame_rec.get("frame_id"),
        "backend_cache": frame_rec.get("backend_cache") or bridge,
        "hud_packet_id": frame_rec.get("hud_packet_id"),
    }
    meta_block["vision_frame_bridge"] = {
        "attached": True,
        "bridge": frame_rec.get("backend_cache") or bridge,
        "source": frame_rec.get("source"),
        "frame_id": frame_rec.get("frame_id"),
    }
    images = meta_block.get("images") if isinstance(meta_block.get("images"), list) else []
    if not images:
        images = [frame_value]
    elif frame_value not in images:
        images = [frame_value] + list(images)
    meta_block["images"] = [img for img in images[:3] if img]
    context_packet["meta"] = meta_block
    context_packet["session_id"] = session_id
    return context_packet, frame_rec

def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    value, expires_at = item
    if expires_at and time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value, ttl_s: float = 0.0):
    expires_at = (time.time() + ttl_s) if ttl_s and ttl_s > 0 else None
    _CACHE[key] = (value, expires_at)

def _cache_invalidate(prefix: str = ""):
    if not prefix:
        _CACHE.clear()
        return
    for k in list(_CACHE.keys()):
        if k.startswith(prefix):
            _CACHE.pop(k, None)

# Runtime anti-thrash: health/state writes are rate-limited because the UI polls /api/health.
_LAST_HEALTH_STATE_WRITE_TS = 0.0
_LAST_HEALTH_STATE_FINGERPRINT = ""
_HEALTH_STATE_WRITE_INTERVAL_SECONDS = float(os.environ.get("SARAH_HEALTH_STATE_WRITE_INTERVAL_SECONDS", "60"))

def _fingerprint_json(payload) -> str:
    try:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception:
        return ""

def load_state() -> dict:
    """Load persisted server state. Never raises."""
    try:
        if os.path.exists(STATE_DB):
            with open(STATE_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _write_json_if_changed(path: str, payload, *, ensure_ascii: bool = False) -> bool:
    """Atomic JSON write that skips disk I/O when content is unchanged."""
    try:
        text = json.dumps(payload or {}, indent=2, sort_keys=True, ensure_ascii=ensure_ascii)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    if f.read() == text:
                        return False
        except Exception:
            pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def save_state(state_or_key, value=None) -> None:
    """Persist server state safely.
    - If called with a dict, overwrites state.
    - If called with (key, value), updates that key.
    Never raises.
    Runtime optimization: skip write when JSON content is unchanged.
    """
    try:
        if value is None and isinstance(state_or_key, dict):
            state = state_or_key or {}
        else:
            key = str(state_or_key)
            state = load_state()
            state[key] = value
        _write_json_if_changed(STATE_DB, state, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# UI ACTION QUEUE + RESEARCH BROWSER STATE BRIDGE
# ---------------------------------------------------------------------------
_UI_ACTION_QUEUE_LOCK = threading.RLock()
_UI_ACTION_QUEUE: list[dict] = []
_UI_ACTION_SEQ = 0
_UI_ACTION_MAX = int(os.getenv("SM_UI_ACTION_QUEUE_MAX", "300") or 300)

def _browser_state_path() -> str:
    try:
        return os.path.join(DATA_DIR, "browser_state.json")
    except Exception:
        return os.path.join(os.getcwd(), "data", "browser_state.json")

def _read_browser_state() -> dict:
    try:
        p = _browser_state_path()
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _write_browser_state(state: dict) -> None:
    try:
        _write_json_if_changed(_browser_state_path(), state or {}, ensure_ascii=False)
    except Exception:
        pass

def _queue_ui_actions(actions, *, source: str = "backend", target: str = "webui") -> list[dict]:
    global _UI_ACTION_SEQ
    if not isinstance(actions, list):
        actions = [actions]
    out = []
    with _UI_ACTION_QUEUE_LOCK:
        for action in actions:
            if not isinstance(action, dict) or not action.get("type"):
                continue
            _UI_ACTION_SEQ += 1
            item = {
                "id": f"uia_{int(time.time()*1000)}_{_UI_ACTION_SEQ}",
                "ts": time.time(),
                "source": str(source or "backend"),
                "target": str(target or "webui"),
                "type": str(action.get("type")),
                "payload": action.get("payload") if isinstance(action.get("payload"), dict) else {},
            }
            _UI_ACTION_QUEUE.append(item)
            out.append(item)
        if len(_UI_ACTION_QUEUE) > _UI_ACTION_MAX:
            del _UI_ACTION_QUEUE[:-_UI_ACTION_MAX]
    return out

def _normalize_panel_url(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    if v.startswith(("http://", "https://")):
        return v
    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$", v, re.I):
        return "https://" + v
    return v

def _extract_url_candidate(text: str) -> str:
    t = text or ""
    m = re.search(r"https?://[^\s)]+", t, re.I)
    if m:
        return m.group(0).rstrip(".,;\"\')")
    m = re.search(r"\b([a-z0-9][a-z0-9.-]+\.[a-z]{2,}(?:/[^\s)]*)?)", t, re.I)
    if m:
        return _normalize_panel_url(m.group(1).rstrip(".,;\"\')"))
    return ""

def _extract_research_query(text: str) -> str:
    t = (text or "").strip()
    cleaned = re.sub(r"\b(open|launch|go to|load|use|search|research|browse|look up|find|in|with|using|the|research panel|research browser|browser panel)\b", " ", t, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :,-")
    return cleaned or t

def _panel_actions_for_text(text: str) -> list[dict]:
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return []

    actions: list[dict] = []

    wants_history = any(k in low for k in ("chat history", "history panel", "conversation history", "open history", "show history"))
    if wants_history:
        actions.extend([
            {"type": "navigate", "payload": {"screen": "history", "app": "chat"}},
            {"type": "desktop.set_app", "payload": {"app": "chat"}},
            {"type": "history_refresh", "payload": {"reason": "chat_command"}},
        ])
        return actions

    wants_research_panel = any(k in low for k in ("research browser", "research panel", "browser panel", "open website", "load website", "go to website", "browse to", "open url"))
    wants_search = any(k in low for k in ("search for", "research this", "research ", "look up", "find information", "web search"))
    wants_read_current = any(k in low for k in ("read current page", "read this page", "summarize this page", "summarize current page", "what website", "current website", "what page"))

    if wants_read_current:
        actions.extend([
            {"type": "navigate", "payload": {"screen": "research", "app": "research"}},
            {"type": "desktop.set_app", "payload": {"app": "research"}},
            {"type": "research_read_current", "payload": {"reason": "chat_command"}},
        ])
        return actions

    if wants_research_panel or ("research" in low and ("panel" in low or "browser" in low)):
        actions.append({"type": "navigate", "payload": {"screen": "research", "app": "research"}})
        actions.append({"type": "desktop.set_app", "payload": {"app": "research"}})
        url = _extract_url_candidate(t)
        if url:
            actions.append({"type": "research_open", "payload": {"url": url, "reason": "chat_command"}})
        elif wants_search:
            actions.append({"type": "research_search", "payload": {"query": _extract_research_query(t), "reason": "chat_command"}})
        return actions

    return []

def _attach_panel_actions_to_bundle(bundle: dict, text: str | None = None) -> dict:
    try:
        if not isinstance(bundle, dict):
            return bundle
        req_text = text
        if req_text is None:
            try:
                payload = request.get_json(silent=True) or {}
                req_text = str(payload.get("text") or "")
            except Exception:
                req_text = ""
        actions = _panel_actions_for_text(req_text or "")
        if not actions:
            return bundle
        existing = bundle.get("actions") if isinstance(bundle.get("actions"), list) else []
        # Avoid duplicate action types/payloads.
        serial_seen = set()
        merged = []
        for a in list(existing) + actions:
            try:
                key = json.dumps(a, sort_keys=True, default=str)
            except Exception:
                key = str(a)
            if key in serial_seen:
                continue
            serial_seen.add(key)
            merged.append(a)
        bundle["actions"] = merged
        # Chat responses already carry actions directly to the UI.
        # Keep the backend queue reserved for REM/background callers that POST /api/ui/actions.
    except Exception:
        pass
    return bundle

def _browser_state_answer_for_text(text: str) -> dict | None:
    low = (text or "").lower()
    if not any(k in low for k in ("read current page", "read this page", "summarize this page", "summarize current page", "what website", "current website", "what page")):
        return None
    state = _read_browser_state()
    if not state.get("url"):
        return _sm_make_outward_bundle(
            "The Research Browser has not reported an active page yet. Open a page in the Research panel first, then ask me to read it.",
            meta={"source": "browser_state", "engine": "research_browser_state", "intent": "research_browser", "version": PROJECT_VERSION},
            actions=[{"type": "navigate", "payload": {"screen": "research", "app": "research"}}, {"type": "desktop.set_app", "payload": {"app": "research"}}],
        )
    title = str(state.get("title") or state.get("url") or "Research Browser page")
    url = str(state.get("url") or "")
    page_text = str(state.get("text") or "").strip()
    if any(k in low for k in ("what website", "current website", "what page")):
        reply = f"The Research Browser is currently on: {title}\n{url}"
    else:
        excerpt = page_text[:1800].strip() if page_text else "No readable text was captured from the page yet."
        reply = f"Research Browser page: {title}\nURL: {url}\n\nReadable page excerpt:\n{excerpt}"
        if len(page_text) > len(excerpt):
            reply += "\n\n[Page text is longer; ask for a deeper summary or specific extraction.]"
    return _sm_make_outward_bundle(
        reply,
        meta={"source": "browser_state", "engine": "research_browser_state", "intent": "research_browser", "version": PROJECT_VERSION, "browser_url": url},
        actions=[{"type": "navigate", "payload": {"screen": "research", "app": "research"}}, {"type": "desktop.set_app", "payload": {"app": "research"}}],
    )

# Load persisted toggles at boot
_boot_state = load_state()
if isinstance(_boot_state, dict):
    MIC_ON = bool(_boot_state.get("MIC_ON", MIC_ON))
    TTS_ON = bool(_boot_state.get("TTS_ON", TTS_ON))
MIC_ENABLED = MIC_ON
TTS_ENABLED = TTS_ON
VOICE_OUTPUT_ON = TTS_ON
VOICE_OUTPUT_ENABLED = TTS_ON

# Optional core modules
ledger_mod = None
try:
    import SarahMemoryLedger as ledger_mod
except ImportError: # Use specific ImportError for module not found
    app_logger.info("SarahMemoryLedger module not found. Ledger functionality will be basic.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryLedger: {e}")


net_mod = None
try:
    import SarahMemoryNetwork as net_mod
except ImportError:
    app_logger.info("SarahMemoryNetwork module not found. Hub functionality will be basic.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryNetwork: {e}")

# Flask app (templates under WEB_DIR so /api/index.html is found)
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/api/static",
    template_folder=TEMPLATE_DIR
)

# Ensure Flask has a secret key for session cookies (used by /api/ui/bootstrap)
SECRET_KEY_FILE = os.path.join(DATA_DIR, ".secret_key")

def get_or_create_secret_key() -> str:
    try:
        _ensure_dir(DATA_DIR)
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
                k = (f.read() or "").strip()
                if k:
                    return k
        k = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
        with open(SECRET_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(k)
        try:
            os.chmod(SECRET_KEY_FILE, 0o600)
        except Exception:
            pass
        return k
    except Exception:
        # Fallback: ephemeral (sessions won't persist across restarts)
        return os.environ.get("SECRET_KEY") or secrets.token_hex(32)

try:
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = get_or_create_secret_key()
except Exception:
    # Not fatal; sessions will simply not persist.
    pass

# Apply CORS *after* app is created
# Tighten CORS based on env config
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("CORS_ORIGINS", "") or "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # Dev + known frontends fallback
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5055",
        "http://127.0.0.1:5055",
        "https://ai.sarahmemory.com",
        "https://api.sarahmemory.com",
    ]

if _CORS_AVAILABLE:
    try:
        CORS(
            app,
            resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
            supports_credentials=True,
        )
    except Exception as e:
        app_logger.error(f"CORS config failed: {e}")
else:
    app_logger.warning("Flask-CORS not installed; CORS disabled (same-origin still works).")

@app.route("/api/ui/actions", methods=["POST"])
def api_ui_actions_enqueue():
    try:
        data = request.get_json(silent=True) or {}
        actions = data.get("actions") if isinstance(data.get("actions"), list) else data.get("action")
        queued = _queue_ui_actions(actions, source=str(data.get("source") or "api"), target=str(data.get("target") or "webui"))
        return jsonify({"ok": True, "queued": len(queued), "items": queued}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/ui/actions/poll", methods=["GET"])
def api_ui_actions_poll():
    try:
        limit = max(1, min(100, int(request.args.get("limit") or 25)))
    except Exception:
        limit = 25
    surface = str(request.args.get("surface") or "webui")
    with _UI_ACTION_QUEUE_LOCK:
        picked = []
        keep = []
        for item in _UI_ACTION_QUEUE:
            target = str(item.get("target") or "webui")
            if len(picked) < limit and target in ("webui", surface, "all", "*"):
                picked.append(item)
            else:
                keep.append(item)
        _UI_ACTION_QUEUE[:] = keep
    actions = [{"type": i.get("type"), "payload": i.get("payload") or {}} for i in picked]
    return jsonify({"ok": True, "count": len(actions), "actions": actions, "items": picked}), 200



# =============================================================================
# SM V8.0 Desktop Mirror Runtime API
# =============================================================================
# Backend-owned desktop capture surface for AvatarPanel's Desktop Mirror mode.
# The frontend is display-only. Desktop control/autonomy requests are accepted
# only as governed tickets and are not executed here.
# =============================================================================

_DESKTOP_RUNTIME_LOCK = threading.RLock()
_DESKTOP_RUNTIME = None


def _desktop_runtime():
    """Return the singleton SarahMemoryDesktop runtime without hard-failing app.py."""
    global _DESKTOP_RUNTIME
    with _DESKTOP_RUNTIME_LOCK:
        if _DESKTOP_RUNTIME is not None:
            return _DESKTOP_RUNTIME
        try:
            import SarahMemoryDesktop as _SMDesktop  # type: ignore
            _DESKTOP_RUNTIME = _SMDesktop.get_desktop_runtime()
            return _DESKTOP_RUNTIME
        except Exception as exc:
            app_logger.warning("SarahMemoryDesktop runtime unavailable: %s", exc)
            return None


def _desktop_request_allowed() -> bool:
    """Desktop capture is local-first by default because it can expose private screen data."""
    try:
        if str(os.getenv("SARAH_DESKTOP_REMOTE_ALLOWED", "0")).strip().lower() in ("1", "true", "yes", "on"):
            return True
        remote = str(request.remote_addr or "").strip().lower()
        if remote in ("127.0.0.1", "::1", "localhost", ""):
            return True
        # Some local reverse proxies report IPv4-mapped loopback.
        if remote.endswith("127.0.0.1"):
            return True
    except Exception:
        pass
    return False


def _desktop_blocked_response():
    return jsonify({
        "ok": False,
        "error": "desktop_mirror_remote_blocked",
        "message": "Desktop mirror is local-only by default. Set SARAH_DESKTOP_REMOTE_ALLOWED=1 only if you explicitly want LAN/remote browser access.",
        "source": "api.desktop.guard",
    }), 403


@app.route("/api/desktop/status", methods=["GET"])
def api_desktop_status():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.status"}), 503
    return jsonify(rt.status()), 200


@app.route("/api/desktop/start", methods=["POST"])
def api_desktop_start():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.start"}), 503
    payload = request.get_json(silent=True) or {}
    result = rt.start(payload if isinstance(payload, dict) else {})
    return jsonify(result), 200 if result.get("ok") else 503


@app.route("/api/desktop/stop", methods=["POST"])
def api_desktop_stop():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.stop"}), 503
    payload = request.get_json(silent=True) or {}
    result = rt.stop(payload if isinstance(payload, dict) else {})
    return jsonify(result), 200 if result.get("ok") else 500


@app.route("/api/desktop/capture", methods=["GET", "POST"])
def api_desktop_capture():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.capture"}), 503
    payload = request.get_json(silent=True) if request.method == "POST" else {}
    if not isinstance(payload, dict):
        payload = {}
    if request.method == "GET":
        payload["include_image"] = str(request.args.get("include_image") or "1").lower() not in ("0", "false", "no", "off")
        if request.args.get("monitor"):
            payload["monitor"] = request.args.get("monitor")
    result = rt.capture(payload)
    return jsonify(result), 200 if result.get("ok") else 503


@app.route("/api/desktop/latest", methods=["GET"])
def api_desktop_latest():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.latest"}), 503
    include_image = str(request.args.get("include_image") or "1").lower() not in ("0", "false", "no", "off")
    auto_capture = str(request.args.get("capture") or request.args.get("auto_capture") or "0").lower() in ("1", "true", "yes", "on")
    result = rt.latest(include_image=include_image, auto_capture=auto_capture)
    return jsonify(result), 200 if result.get("ok", True) else 503


@app.route("/api/desktop/observe", methods=["GET"])
def api_desktop_observe():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.observe"}), 503
    include_image = str(request.args.get("include_image") or "0").lower() in ("1", "true", "yes", "on")
    result = rt.observe(include_image=include_image)
    return jsonify(result), 200 if result.get("ok") else 503


@app.route("/api/desktop/action/request", methods=["POST"])
def api_desktop_action_request():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.action"}), 503
    payload = request.get_json(silent=True) or {}
    result = rt.request_action(payload if isinstance(payload, dict) else {})
    return jsonify(result), 202 if result.get("ok") else 400


@app.route("/api/desktop/task/request", methods=["POST"])
def api_desktop_task_request():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.task"}), 503
    payload = request.get_json(silent=True) or {}
    result = rt.request_task(payload if isinstance(payload, dict) else {})
    return jsonify(result), 202 if result.get("ok") else 400


@app.route("/api/desktop/mjpeg", methods=["GET"])
@app.route("/api/desktop/stream", methods=["GET"])
@app.route("/api/desktop_mirror", methods=["GET"])
@app.route("/api/desktop_mirror/stream", methods=["GET"])
@app.route("/api/screen/mjpeg", methods=["GET"])
@app.route("/api/screen/stream", methods=["GET"])
def api_desktop_mjpeg_stream():
    if not _desktop_request_allowed():
        return _desktop_blocked_response()
    rt = _desktop_runtime()
    if rt is None:
        return jsonify({"ok": False, "error": "desktop_runtime_unavailable", "source": "api.desktop.mjpeg"}), 503
    try:
        fps = int(request.args.get("fps") or os.getenv("SARAH_DESKTOP_MIRROR_FPS", "6") or 6)
    except Exception:
        fps = 6
    from flask import Response
    return Response(
        rt.mjpeg_stream(fps=fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "X-SarahMemory-Source": "desktop_mirror"},
    )


# =============================================================================
# SM V8.0 Native VR Operator HUD Runtime API
# =============================================================================
# Visual-only runtime manager. MSDC produces the body/display witness; app.py
# owns process lifecycle for SarahMemoryVRHudRenderer.py. Stopping VR does not
# stop appvision, SOBJE, or FacialRecognition background interpretation.
# =============================================================================

_VR_RUNTIME_LOCK = threading.RLock()
_VR_RENDERER_PROC = None
_VR_WATCHER_STARTED = False
_VR_WATCHER_STOP = False

def _vr_settings_dir() -> str:
    try:
        path = getattr(config, "SETTINGS_DIR", None)
        if path:
            return os.path.abspath(str(path))
    except Exception:
        pass
    return os.path.join(DATA_DIR, "settings")

def _vr_runtime_state_path() -> str:
    return os.path.join(_vr_settings_dir(), "vr_runtime_state.json")

def _vr_renderer_config_path() -> str:
    return os.path.join(_vr_settings_dir(), "vr_hud_renderer.json")

def _vr_read_json(path: str, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {} if default is None else default

def _vr_write_json(path: str, payload) -> bool:
    try:
        return _write_json_if_changed(path, payload or {}, ensure_ascii=False)
    except Exception as exc:
        app_logger.warning("VR JSON write failed %s: %s", path, exc)
        return False

def _vr_renderer_alive() -> bool:
    global _VR_RENDERER_PROC
    with _VR_RUNTIME_LOCK:
        proc = _VR_RENDERER_PROC
        if proc is not None:
            try:
                if proc.poll() is None:
                    return True
                _VR_RENDERER_PROC = None
            except Exception:
                _VR_RENDERER_PROC = None
        state = _vr_read_json(_vr_runtime_state_path(), {})
        pid = int(state.get("pid") or 0) if isinstance(state, dict) else 0
        if pid <= 0:
            return False
        try:
            if os.name == "nt":
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            os.kill(pid, 0)
            return True
        except Exception:
            return False

def _vr_import_msdc():
    try:
        import SarahMemoryMSDC as _MSDC  # type: ignore
        return _MSDC
    except Exception as exc:
        app_logger.warning("MSDC import failed for VR runtime: %s", exc)
        return None

def _vr_msdc_probe() -> dict:
    msdc = _vr_import_msdc()
    if msdc is None:
        return {"ok": False, "error": "msdc_unavailable", "source": "api.vr"}
    try:
        if hasattr(msdc, "msdc_vr_probe"):
            return msdc.msdc_vr_probe(include_driver_actions=True)  # type: ignore[attr-defined]
        if hasattr(msdc, "msdc_vr_hud_status"):
            return {"ok": True, "fallback": True, "status": msdc.msdc_vr_hud_status()}  # type: ignore[attr-defined]
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": "api.vr.msdc_probe"}
    return {"ok": False, "error": "msdc_vr_probe_missing", "source": "api.vr"}

def _vr_msdc_surface_request(payload: dict | None = None) -> dict:
    msdc = _vr_import_msdc()
    if msdc is None:
        return {"ok": False, "error": "msdc_unavailable", "source": "api.vr"}
    try:
        if hasattr(msdc, "msdc_vr_surface_request"):
            return msdc.msdc_vr_surface_request(payload or {})  # type: ignore[attr-defined]
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": "api.vr.surface_request"}
    return {"ok": False, "error": "msdc_vr_surface_request_missing", "source": "api.vr"}

def _vr_config_from_surface(surface_request: dict, payload: dict | None = None) -> dict:
    payload = payload or {}
    surface = surface_request.get("surface") if isinstance(surface_request.get("surface"), dict) else {}
    bounds = surface.get("bounds") if isinstance(surface.get("bounds"), dict) else {}
    if not bounds:
        bounds = surface.get("display") if isinstance(surface.get("display"), dict) else {}
    headset = surface_request.get("probe", {}).get("native_profile", {}) if isinstance(surface_request.get("probe"), dict) else {}
    active_profile = headset.get("active_profile") if isinstance(headset.get("active_profile"), dict) else {}
    cfg = {
        "schema": "SMHUD_RENDERER_CONFIG_V1",
        "api_base": str(payload.get("api_base") or "http://127.0.0.1:8000"),
        "endpoints": {
            "frame_latest": "/api/vision/frame/latest",
            "hud_packet": "/api/vision/hud/packet",
            "hud_status": "/api/vision/hud/status",
        },
        "display": {
            "window_title": "SM_A_HUD_DIRECT",
            "x": int(payload.get("x", bounds.get("x", 0)) or 0),
            "y": int(payload.get("y", bounds.get("y", 0)) or 0),
            "width": int(payload.get("width", bounds.get("width", active_profile.get("width", 1920))) or 1920),
            "height": int(payload.get("height", bounds.get("height", active_profile.get("height", 1080))) or 1080),
            "fullscreen": bool(payload.get("fullscreen", True)),
            "borderless": False,
            "move_window": True,
            "target_role": "operator_vr_surface",
            "mirror_x": bool(payload.get("mirror_x", True)),
        },
        "mirror": {
            "enabled": bool(payload.get("mirror_preview", True)),
            "window_title": "SM_A_HUD_MIRROR",
            "x": int(payload.get("mirror_x", 60) or 60),
            "y": int(payload.get("mirror_y", 60) or 60),
            "width": int(payload.get("mirror_width", 960) or 960),
            "height": int(payload.get("mirror_height", 540) or 540),
            "fullscreen": False,
            "move_window": True,
        },
        "headset": {
            "enabled": bool(payload.get("headset_surface", True)),
            "profile_id": str(active_profile.get("profile_id") or headset.get("selected_profile") or "psvr_v1_processor_box"),
            "render_mode": str(active_profile.get("render_mode") or "mono_mirror"),
            "lens_correction": bool(active_profile.get("lens_correction", False)),
            "stereo_split": bool(active_profile.get("stereo_split", False)),
            "auto_start_on_headset_connected": bool(payload.get("auto_start_on_headset_connected", True)),
            "auto_stop_on_headset_disconnected": bool(payload.get("auto_stop_on_headset_disconnected", True)),
        },
        "compositor": {
            "enabled": True,
            "mode": "mirror_plus_headset",
            "fit": "cover",
            "safe_border_px": 0,
            "hud_overlay": True,
        },
        "render": {
            "fps": float(payload.get("fps", 30) or 30),
            "frame_poll_hz": 24,
            "packet_poll_hz": 10,
            "status_poll_hz": 1,
            "filter": str(payload.get("filter") or "mono_crimson"),
            "grid": True,
            "target_brackets": True,
            "telemetry_tapes": True,
            "center_crosshair": True,
            "stale_packet_ms": 2500,
            "no_frame_background": "black",
            "safe_exit_keys": [27, 113],
        },
        "safety": {
            "observe_only": True,
            "movement_locked": True,
            "hud_can_execute_actions": False,
            "hud_can_authorize_movement": False,
            "require_backend_packet_schema": True,
        },
    }
    return cfg

def _vr_start_renderer(payload: dict | None = None, reason: str = "manual") -> dict:
    global _VR_RENDERER_PROC
    payload = payload or {}
    with _VR_RUNTIME_LOCK:
        if _vr_renderer_alive():
            state = _vr_read_json(_vr_runtime_state_path(), {})
            return {"ok": True, "already_running": True, "runtime": state, "source": "api.vr.start"}
        probe = _vr_msdc_probe()
        surface_request = _vr_msdc_surface_request({"api_base": payload.get("api_base") or "http://127.0.0.1:8000"})
        cfg = _vr_config_from_surface(surface_request if surface_request.get("ok") else {"surface": {}, "probe": probe}, payload)
        cfg_path = _vr_renderer_config_path()
        _vr_write_json(cfg_path, cfg)
        renderer_path = os.path.join(BASE_DIR, "SarahMemoryVRHudRenderer.py")
        if not os.path.exists(renderer_path):
            renderer_path = os.path.join(os.getcwd(), "SarahMemoryVRHudRenderer.py")
        if not os.path.exists(renderer_path):
            return {"ok": False, "error": "renderer_file_missing", "renderer_path": renderer_path, "probe": probe}
        cmd = [sys.executable, renderer_path, "--config", cfg_path]
        try:
            _VR_RENDERER_PROC = subprocess.Popen(cmd, cwd=BASE_DIR)
            runtime = {
                "ok": True,
                "running": True,
                "pid": int(_VR_RENDERER_PROC.pid),
                "cmd": cmd,
                "started_ts": time.time(),
                "reason": reason,
                "config_path": cfg_path,
                "renderer_path": renderer_path,
                "movement_lock": True,
                "vision_background_continues_after_stop": True,
                "probe": probe,
                "surface_request": surface_request,
            }
            _vr_write_json(_vr_runtime_state_path(), runtime)
            return runtime
        except Exception as exc:
            return {"ok": False, "error": str(exc), "cmd": cmd, "probe": probe}

def _vr_stop_renderer(reason: str = "manual") -> dict:
    global _VR_RENDERER_PROC
    with _VR_RUNTIME_LOCK:
        stopped = False
        pid = 0
        proc = _VR_RENDERER_PROC
        if proc is not None:
            try:
                pid = int(proc.pid or 0)
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=4)
                    except Exception:
                        proc.kill()
                stopped = True
            except Exception as exc:
                app_logger.warning("VR renderer process stop failed: %s", exc)
            _VR_RENDERER_PROC = None
        state = _vr_read_json(_vr_runtime_state_path(), {})
        state.update({
            "ok": True,
            "running": False,
            "stopped_ts": time.time(),
            "stop_reason": reason,
            "pid": 0,
            "previous_pid": pid or state.get("pid"),
            "vision_background_continues": True,
            "note": "VR display feed stopped; appvision/SOBJE/FacialRecognition remain available for background frame interpretation.",
        })
        _vr_write_json(_vr_runtime_state_path(), state)
        return {"ok": True, "stopped": stopped, "runtime": state, "source": "api.vr.stop"}

def _vr_status_payload(refresh_probe: bool = False) -> dict:
    state = _vr_read_json(_vr_runtime_state_path(), {})
    alive = _vr_renderer_alive()
    probe = _vr_msdc_probe() if refresh_probe else None
    vision = {"ok": True, "endpoint": "/api/vision/hud/status", "background_analysis_continues": True}
    return {
        "ok": True,
        "schema": "SarahMemory.api.vr.status.v1",
        "running": alive,
        "runtime": state if isinstance(state, dict) else {},
        "probe": probe,
        "vision": vision,
        "movement_lock": True,
        "native_runtime": "sarahmemory_native",
        "external_runtime_allowed": False,
        "auto_watcher_started": bool(_VR_WATCHER_STARTED),
    }

def _vr_headset_connected_from_probe(probe: dict) -> bool:
    try:
        r = probe.get("readiness") if isinstance(probe.get("readiness"), dict) else {}
        if bool(r.get("headset_connected")):
            return True
        h = ((probe.get("drivers") or {}).get("headset") or {}) if isinstance(probe.get("drivers"), dict) else {}
        return bool(h.get("connected") or h.get("headset_connected") or (isinstance(h.get("native_hmd"), dict) and h["native_hmd"].get("connected")))
    except Exception:
        return False

def _vr_watcher_loop():
    global _VR_WATCHER_STOP
    while not _VR_WATCHER_STOP:
        try:
            state = _vr_read_json(_vr_runtime_state_path(), {})
            cfg = _vr_read_json(_vr_renderer_config_path(), {})
            headset_cfg = cfg.get("headset") if isinstance(cfg.get("headset"), dict) else {}
            auto_start = bool(headset_cfg.get("auto_start_on_headset_connected", state.get("auto_start_on_headset_connected", True)))
            auto_stop = bool(headset_cfg.get("auto_stop_on_headset_disconnected", state.get("auto_stop_on_headset_disconnected", True)))
            probe = _vr_msdc_probe()
            connected = _vr_headset_connected_from_probe(probe)
            if connected and auto_start and not _vr_renderer_alive():
                _vr_start_renderer({"auto_start_on_headset_connected": True}, reason="headset_connected")
            elif (not connected) and auto_stop and _vr_renderer_alive():
                _vr_stop_renderer(reason="headset_disconnected")
        except Exception as exc:
            app_logger.debug("VR watcher tick failed: %s", exc)
        try:
            _sleep_s = float(os.getenv("SM_VR_WATCHER_INTERVAL_SEC", "5.0") or 5.0)
        except Exception:
            _sleep_s = 5.0
        time.sleep(max(2.0, _sleep_s))

def _vr_ensure_watcher_started() -> None:
    global _VR_WATCHER_STARTED
    if _VR_WATCHER_STARTED:
        return
    try:
        t = threading.Thread(target=_vr_watcher_loop, name="SM_VR_HeadsetWatcher", daemon=True)
        t.start()
        _VR_WATCHER_STARTED = True
    except Exception as exc:
        app_logger.warning("VR watcher start failed: %s", exc)

@app.route("/api/vr/status", methods=["GET"])
def api_vr_status():
    _vr_ensure_watcher_started()
    refresh = str(request.args.get("refresh") or "0").lower() in ("1", "true", "yes", "on")
    return jsonify(_vr_status_payload(refresh_probe=refresh)), 200

@app.route("/api/vr/probe", methods=["POST", "GET"])
def api_vr_probe():
    _vr_ensure_watcher_started()
    probe = _vr_msdc_probe()
    return jsonify({"ok": bool(probe.get("ok", True)), "probe": probe, "running": _vr_renderer_alive(), "source": "api.vr.probe"}), 200

@app.route("/api/vr/start", methods=["POST"])
def api_vr_start():
    _vr_ensure_watcher_started()
    payload = request.get_json(silent=True) or {}
    result = _vr_start_renderer(payload, reason=str(payload.get("reason") or "api_start"))
    return jsonify(result), 200 if result.get("ok") else 400

@app.route("/api/vr/stop", methods=["POST"])
def api_vr_stop():
    payload = request.get_json(silent=True) or {}
    result = _vr_stop_renderer(reason=str(payload.get("reason") or "api_stop"))
    return jsonify(result), 200


# --- SarahMemoryGITtalk (TEMP ADMIN TOOL) ---
try:
    # Only enable when you explicitly turn it on
    if os.environ.get("SARAH_GITTALK_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"):
        mod_path = Path(__file__).resolve().parent / "data" / "mods" / "v800"
        if mod_path.exists() and str(mod_path) not in sys.path:
            sys.path.insert(0, str(mod_path))

        from SarahMemoryGITtalk import create_gittalk_blueprint  # noqa
        app.register_blueprint(create_gittalk_blueprint(url_prefix="/api/gittalk"))
        app_logger.info("SarahMemoryGITtalk blueprint mounted at /api/gittalk")
except Exception as e:
    app_logger.warning(f"SarahMemoryGITtalk not mounted: {e}")
# --- end SarahMemoryGITtalk ---

try:
    from SarahMemoryDatabase import init_database
    init_database()  # ensures ai_learning.db + qa_cache exist
except ImportError:
    app_logger.warning("SarahMemoryDatabase not found. Skipping database initialization.")
except Exception as e:
    app_logger.error(f"DB init failed in app.py: {e}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_sqlite(path: str):
    """Establishes an SQLite database connection with row_factory set to sqlite3.Row."""
    try:
        con = sqlite3.connect(path, timeout=10.0)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA busy_timeout=10000")
            con.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return con
    except sqlite3.Error as e:
        app_logger.error(f"Failed to connect to SQLite DB at {path}: {e}")
        raise # Re-raise to be handled by caller

def _safe_getattr(mod, name, default=None):
    """Safely gets an attribute from a module, returning a default if not found or an error occurs."""
    try:
        return getattr(mod, name)
    except AttributeError:
        # app_logger.debug(f"Attribute '{name}' not found in module {mod.__name__}.")
        return default
    except Exception as e:
        app_logger.error(f"Error accessing attribute '{name}' from module {mod.__name__}: {e}")
        return default

def _ensure_dir(p: str):
    """Ensures a directory exists, logging any errors."""
    try:
        os.makedirs(p, exist_ok=True)
    except OSError as e:
        app_logger.error(f"Failed to create directory {p}: {e}")

# Cache global paths to avoid recalculation on every request
_cached_globals_paths = None
def _globals_paths():
    """
    Locate key SarahMemory paths from SarahMemoryGlobals.py.
    Returns a dict with stable directory keys used by the server and WebUI.
    Discovery of files is NOT activation; these are path hints only.
    """
    global _cached_globals_paths
    if _cached_globals_paths is not None:
        return _cached_globals_paths

    # Defaults (work on PythonAnywhere / headless Linux too)
    root_dir = os.path.abspath(Path(__file__).resolve().parents[2])
    data_dir = os.path.join(root_dir, "data")
    sandbox_dir = os.path.join(root_dir, "sandbox")
    addons_dir = os.path.join(data_dir, "addons")
    mods_dir = os.path.join(root_dir, "mods")
    settings_dir = os.path.join(data_dir, "settings")
    datasets_dir = os.path.join(data_dir, "memory", "datasets")
    documents_dir = os.path.join(data_dir, "documents")
    drivers_dir = os.path.join(data_dir, "drivers")
    core_registry_dir = os.path.join(settings_dir, "core_registry")

    try:
        import SarahMemoryGlobals as smg  # type: ignore
        root_dir = os.path.abspath(getattr(smg, "ROOT_DIR", getattr(smg, "BASE_DIR", root_dir)))
        data_dir = os.path.abspath(getattr(smg, "DATA_DIR", data_dir))
        sandbox_dir = os.path.abspath(getattr(smg, "SANDBOX_DIR", sandbox_dir))
        addons_dir = os.path.abspath(getattr(smg, "ADDONS_DIR", addons_dir))
        mods_dir = os.path.abspath(getattr(smg, "MODS_DIR", mods_dir))
        settings_dir = os.path.abspath(getattr(smg, "SETTINGS_DIR", settings_dir))
        datasets_dir = os.path.abspath(getattr(smg, "DATASETS_DIR", datasets_dir))
        documents_dir = os.path.abspath(getattr(smg, "DOCUMENTS_DIR", documents_dir))
        drivers_dir = os.path.abspath(getattr(smg, "DRIVERS_DIR", drivers_dir))
        core_registry_dir = os.path.abspath(getattr(smg, "CORE_REGISTRY_DIR", core_registry_dir))
    except Exception:
        pass

    # Ensure dirs exist (best-effort)
    for d in (data_dir, sandbox_dir, addons_dir, mods_dir, settings_dir, datasets_dir, documents_dir, drivers_dir, core_registry_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    _cached_globals_paths = {
        "ROOT_DIR": root_dir,
        "DATA_DIR": data_dir,
        "SANDBOX_DIR": sandbox_dir,
        "ADDONS_DIR": addons_dir,
        "MODS_DIR": mods_dir,
        "SETTINGS_DIR": settings_dir,
        "DATASETS_DIR": datasets_dir,
        "DOCUMENTS_DIR": documents_dir,
        "DRIVERS_DIR": drivers_dir,
        "CORE_REGISTRY_DIR": core_registry_dir,
    }
    return _cached_globals_paths


def _globals_dir(key: str, default_rel: str) -> str:
    """Return a string path from _globals_paths()[key].
    Falls back to CWD/default_rel if missing or invalid."""
    try:
        d = _globals_paths()
        if isinstance(d, dict):
            v = d.get(key)
            if isinstance(v, (str, bytes, os.PathLike)):
                return os.fspath(v)
    except Exception:
        pass
    return os.path.join(os.path.abspath(os.getcwd()), default_rel)


def _sm_refresh_core_registry(force: bool = False) -> dict:
    """Best-effort registry warmup. Discovery is not activation."""
    try:
        import SarahMemoryGlobals as G  # type: ignore
        fn = _safe_getattr(G, "sm_refresh_core_registry")
        if callable(fn):
            data = fn(force=force)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        app_logger.warning(f"Core registry refresh failed: {e}")
    return {}


def _sm_core_governance_profile() -> dict:
    try:
        import SarahMemoryGlobals as G  # type: ignore
        fn = _safe_getattr(G, "sm_get_core_governance_profile")
        if callable(fn):
            data = fn()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {
        "dynamic_registration": False,
        "auto_expose_approved": True,
        "contract_validation_required": False,
        "discovery_is_not_activation": True,
    }


def _sm_module_approved(module_name: str, capability: str | None = None) -> bool:
    """Governed activation check. Presence/importability is not acceptance."""
    if not module_name:
        return False
    try:
        _sm_refresh_core_registry(force=False)
        import SarahMemoryGlobals as G  # type: ignore
        fn = _safe_getattr(G, "sm_is_core_module_approved")
        if callable(fn):
            return bool(fn(module_name, capability=capability))
    except Exception as e:
        app_logger.warning(f"Core module approval check failed for {module_name}: {e}")
    return True


def _sm_build_context_packet(payload: dict, text: str, intent: str, tone: str, complexity: str, avatar_request: bool, *, local_only: bool, safe_mode: bool, neoskymatrix: bool, developersmode: bool) -> dict:
    meta_in = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    session_id = _get_or_create_session_id(payload)

    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    if not images and isinstance(meta_in.get("images"), list):
        images = list(meta_in.get("images") or [])
    video = payload.get("video") if isinstance(payload.get("video"), list) else []
    if not video and isinstance(meta_in.get("video"), list):
        video = list(meta_in.get("video") or [])
    files = payload.get("files") if isinstance(payload.get("files"), list) else []
    if not files and isinstance(meta_in.get("files"), list):
        files = list(meta_in.get("files") or [])

    frame_payload = _normalize_vision_frame_payload(payload)
    frame_value = frame_payload.get("frame") if isinstance(frame_payload, dict) else None

    return {
        "text": text,
        "session_id": session_id,
        "user_id": payload.get("user_id") or payload.get("uid"),
        "source": str(payload.get("source") or "api").strip() or "api",
        "mode": str(payload.get("mode") or ("LOCAL" if local_only else "ANY")).strip().upper() or "ANY",
        "intent": intent,
        "tone": tone,
        "complexity": complexity,
        "avatar_request": bool(avatar_request),
        "request_source": "api_chat",
        "ui": str(payload.get("ui") or "webui"),
        "meta": {
            "files": files,
            "images": images,
            "audio": payload.get("audio") or [],
            "video": video,
            "frame": frame_value,
            "latest_frame": frame_value,
            "offline": bool(local_only or payload.get("offline") or payload.get("local_only")),
            "local_only": bool(local_only),
            "safe_mode": bool(safe_mode),
            "diagnostics_ping": bool(payload.get("diagnostics_ping") or payload.get("diag_ping") or False),
            "force_neuron": bool(payload.get("force_neuron") or payload.get("use_neuron") or True),
            "panel": payload.get("panel"),
            "addon": payload.get("addon"),
            "driver": payload.get("driver"),
            "display_requested": bool(payload.get("display_requested") or False),
            "download_requested": bool(payload.get("download_requested") or False),
            "user_consented": bool(payload.get("user_consented") or payload.get("consented") or False),
            "proposed_action": payload.get("proposed_action") if isinstance(payload.get("proposed_action"), dict) else None,
            "mode_flags": {
                "LOCAL_ONLY_MODE": bool(local_only),
                "SAFE_MODE": bool(safe_mode),
                "NEOSKYMATRIX": bool(neoskymatrix),
                "DEVELOPERSMODE": bool(developersmode),
            },
            "ingress_meta": meta_in,
        },
    }


def _sm_present_text(raw_text: str, *, intent: str = "", meta: dict | None = None) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    low_intent = str(intent or (meta or {}).get("intent") or "").strip().lower()
    if low_intent == "math":
        if text.lower().startswith("the answer") or text.lower().startswith("the result"):
            return text
        return f"The answer is {text}."
    if low_intent in {"diagnostics", "system_status", "status"} and text and text[-1] not in ".!?":
        return text + "."
    return text


def _sm_make_outward_bundle(presentation_text: str, *, meta: dict | None = None, artifacts=None, actions=None, errors=None, raw_answer: str | None = None):
    meta = dict(meta or {})
    meta.setdefault("presentation_only", True)
    meta.setdefault("outward_formatter", "app.py")
    meta.pop("raw_answer", None)
    meta.pop("canonical_answer", None)

    try:
        import SarahMemoryReply as R  # type: ignore
        make_bundle = _safe_getattr(R, "_sm_make_outward_bundle")
        if callable(make_bundle):
            bundle = make_bundle(
                presentation_text,
                meta=meta,
                artifacts=artifacts or [],
                actions=actions or [],
                errors=errors or [],
            )
            enforce = _safe_getattr(R, "_sm_enforce_provenance")
            if callable(enforce):
                bundle = enforce(bundle)
            stamp = _safe_getattr(R, "_stamp_bundle")
            if callable(stamp):
                try:
                    bundle = stamp(bundle)
                except Exception:
                    pass
            if isinstance(bundle, dict):
                bundle["ok"] = True
                bundle["reply"] = bundle.get("presentation_reply") or bundle.get("response") or presentation_text
                return _attach_panel_actions_to_bundle(bundle)
    except Exception:
        pass

    return _attach_panel_actions_to_bundle({
        "ok": True,
        "presentation_reply": presentation_text,
        "reply": presentation_text,
        "response": presentation_text,
        "meta": meta,
        "artifacts": list(artifacts or []),
        "actions": list(actions or []),
        "errors": list(errors or []),
    })



def _get_hub_hmac_secret() -> str:
    """Shared secret for node/hub HMAC signing.

    Priority:
      1) env HUB_HMAC_SECRET / SARAH_HUB_HMAC_SECRET
      2) SarahMemoryGlobals.HUB_HMAC_SECRET (if present)
    """
    try:
        import SarahMemoryGlobals as G
        v = getattr(G, "HUB_HMAC_SECRET", "") or ""
        if v:
            return str(v)
    except Exception:
        pass
    return (os.environ.get("HUB_HMAC_SECRET") or os.environ.get("SARAH_HUB_HMAC_SECRET") or "").strip()

def _sign_ok(body: bytes, signature: str) -> bool:
    """Verify X-Sarah-Signature as hex(HMAC_SHA256(secret, body)).

    If no secret is configured, allow ONLY localhost requests (dev-safe fallback).
    """
    secret = _get_hub_hmac_secret()
    sig = (signature or "").strip()
    if not secret:
        # No secret configured — do not expose signature-less auth to the internet.
        # Accept only loopback for local development.
        try:
            ra = request.remote_addr or ""
        except Exception:
            ra = ""
        return ra in ("127.0.0.1", "::1", "localhost")
    if not sig:
        return False
    try:
        mac = hmac.new(secret.encode("utf-8"), body or b"", hashlib.sha256).hexdigest()
        return hmac.compare_digest(mac, sig)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Wallet / Ledger
# ---------------------------------------------------------------------------
def _wallet_path_simple(node: str) -> str:
    safe = "".join(ch for ch in node if ch.isalnum() or ch in ("_", "-")) or "anon"
    return os.path.join(WALLETS_DIR, f"wallet-{safe}.srh")

def ensure_wallet_simple(node: str):
    """Ensure minimal wallet tables exist."""
    con = None
    try:
        con = _connect_sqlite(WALLET_DB)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                balance TEXT DEFAULT '0'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user_id TEXT,
                delta TEXT,
                note TEXT
            )
        """)
        con.commit()
        return True
    except Exception as e:
        logger.exception("ensure_wallet_simple failed: %s", e)
        return False
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass


def get_balance_simple(path: str) -> Decimal:
    balance = Decimal("0")
    con = None
    try:
        con = _connect_sqlite(path)
        cur = con.cursor()
        cur.execute("SELECT balance FROM wallet WHERE id=1")
        row = cur.fetchone()
        balance = Decimal(row) if row and row is not None else Decimal("0")
    except sqlite3.Error as e:
        app_logger.error(f"Failed to get simple wallet balance from {path}: {e}")
    finally:
        if con: con.close()
    return balance

def read_top_nodes(limit=10):
    """Return top nodes for the public leaderboard.

    Preferred source (when enabled): GoogieHost MySQL table `sm_network_nodes`
      - ordered by `trust_score` DESC
      - limited to `limit`

    Fallback source: local SQLite wallet (legacy/demo)
    """
    # --- Cloud MySQL path (preferred) ---
    try:
        cloud_enabled = str(os.getenv("CLOUD_DB_ENABLED", "false")).strip().lower() in ("1", "true", "yes", "on")
        if cloud_enabled:
            # Local import so the server can still boot even if MySQL client isn't installed.
            try:
                import pymysql  # type: ignore
            except Exception:
                pymysql = None

            if pymysql is not None:
                host = os.getenv("CLOUD_DB_HOST") or ""
                name = os.getenv("CLOUD_DB_NAME") or ""
                user = os.getenv("CLOUD_DB_USER") or ""
                pwd = os.getenv("CLOUD_DB_PASSWORD") or ""
                port = int(os.getenv("CLOUD_DB_PORT") or "3306")

                if host and name and user and pwd:
                    con = None
                    try:
                        con = pymysql.connect(
                            host=host,
                            user=user,
                            password=pwd,
                            database=name,
                            port=port,
                            connect_timeout=5,
                            read_timeout=5,
                            write_timeout=5,
                            cursorclass=pymysql.cursors.DictCursor,
                            charset="utf8mb4",
                        )
                        with con.cursor() as cur:
                            cur.execute(
                                """
                                SELECT node_name, node_id, ip_address, is_online, trust_score
                                FROM sm_network_nodes
                                ORDER BY trust_score DESC, id ASC
                                LIMIT %s
                                """,
                                (max(1, int(limit)),),
                            )
                            rows = cur.fetchall() or []
                        leaders = []
                        rank = 1
                        for r in rows:
                            leaders.append(
                                {
                                    "rank": rank,
                                    "name": (r.get("node_name") or r.get("node_id") or "").strip() or f"Node-{rank}",
                                    "org": "SarahMemory Node",
                                    "rep": float(r.get("trust_score") or 0),
                                    "node_id": r.get("node_id"),
                                    "is_online": int(r.get("is_online") or 0),
                                    "ip": r.get("ip_address"),
                                }
                            )
                            rank += 1
                        return leaders
                    except Exception as e:
                        logger.debug("read_top_nodes cloud MySQL failed: %s", e)
                    finally:
                        try:
                            if con is not None:
                                con.close()
                        except Exception:
                            pass
    except Exception as e:
        logger.debug("read_top_nodes cloud config failed: %s", e)

    # --- Local fallback path (SQLite wallet) ---
    ensure_wallet_simple()
    con = None
    try:
        con = _connect_sqlite(WALLET_DB)
        cur = con.cursor()
        cur.execute("SELECT user_id, balance FROM wallet")
        rows = cur.fetchall() or []
        data = []
        for r in rows:
            uid = r[0]
            bal = Decimal(str(r[1] if r[1] is not None else "0"))
            data.append({
                "rank": 0,
                "name": uid,
                "org": "Local Wallet",
                "rep": float(bal),
                "user_id": uid,
                "balance": str(bal),
            })
        data.sort(key=lambda x: Decimal(str(x.get("rep", 0))), reverse=True)
        # fill ranks
        for i, item in enumerate(data[: max(1, int(limit))], start=1):
            item["rank"] = i
        return data[: max(1, int(limit))]
    except Exception as e:
        logger.debug("read_top_nodes sqlite fallback failed: %s", e)
        return []
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass




def ensure_meta_db():
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        # Hub/node tables (for network sync)
        cur.execute("""CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            last_ts REAL,
            meta TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, context_id TEXT, vector TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS contexts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, text TEXT, tags TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS job_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, job_id TEXT, result TEXT
        )""")
        # Knowledge marketplace + receipts
        cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, requester TEXT, topic TEXT, reward TEXT, status TEXT, provider TEXT, answer TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, payload TEXT, sig TEXT, valid INTEGER
        )""")
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to ensure meta DB at {META_DB}: {e}")
    finally:
        if con: con.close()
ensure_meta_db()


# ---------------------------------------------------------------------------
# Core routes (UI + API)
# ---------------------------------------------------------------------------

def _get_runtime_meta_safe():
    """
    Lightweight wrapper around SarahMemoryGlobals.get_runtime_meta (Phase A1).
    Returns a small dict with runtime identity and safety flags that is safe to
    serialize to logs and JSON responses. If SarahMemoryGlobals is missing or
    incomplete, falls back to conservative defaults.
    """
    try:
        import SarahMemoryGlobals as G
        meta_fn = _safe_getattr(G, "get_runtime_meta")
        if callable(meta_fn):
            meta = meta_fn() or {}
        else:
            meta = {}
        # Ensure baseline keys exist even if get_runtime_meta() is older than v7.7.5
        meta.setdefault("project_version", getattr(G, "PROJECT_VERSION", PROJECT_VERSION))
        meta.setdefault("author", getattr(G, "AUTHOR", "Brian Lee Baros"))
        meta.setdefault("revision_start_date", getattr(G, "REVISION_START_DATE", ""))
        meta.setdefault("run_mode", getattr(G, "RUN_MODE", "local"))
        meta.setdefault("device_mode", getattr(G, "DEVICE_MODE", "local_agent"))
        meta.setdefault("device_profile", getattr(G, "DEVICE_PROFILE", "Standard"))
        meta.setdefault("safe_mode", getattr(G, "SAFE_MODE", False))
        meta.setdefault("local_only", getattr(G, "LOCAL_ONLY_MODE", False))
        meta.setdefault("node_name", getattr(G, "NODE_NAME", "SarahMemoryNode"))
        return meta
    except Exception as e:
        app_logger.warning(f"Error getting runtime meta from SarahMemoryGlobals, falling back: {e}")
        # Fail-safe identity snapshot if globals are unavailable.
        return {
            "project_version": PROJECT_VERSION,
            "author": "Brian Lee Baros",
            "revision_start_date": "",
            "run_mode": "local",
            "device_mode": "local_agent",
            "device_profile": "Standard",
            "safe_mode": False,
            "local_only": False,
            "node_name": "SarahMemoryNode",
        }
try:
    import SarahMemoryCognitiveServices as cog
    COG_AVAILABLE = True
except Exception as e:
    app_logger.warning(f"CognitiveServices not available: {e}")
    cog = None
    COG_AVAILABLE = False

@app.before_request
def _cognitive_guard():
    if (not COG_AVAILABLE) or (cog is None):
        return None

    # Only guard API endpoints (avoid slowing static/template hits)
    p = (request.path or "")
    if not p.startswith("/api/"):
        return None

    # Pull a small amount of text to analyze (don’t log secrets)
    data = request.get_json(silent=True) if request.method in ("POST","PUT","PATCH") else None
    msg = ""
    if isinstance(data, dict):
        # common fields
        msg = str(data.get("message") or data.get("text") or data.get("q") or "")[:4000]

    # Example: call a lightweight analyzer (sentiment/risk tagging/etc.)
    # Store result for the endpoint to use (no blocking by default)
    try:
        g.cognitive = {"ok": True, "sentiment": cog.analyze_text(msg) if msg else None}
    except Exception as e:
        g.cognitive = {"ok": False, "error": str(e)}

    return None


@app.get("/api/ui/contracts")
def api_ui_contracts():
    """Read-only UI/backend contract map for the SarahMemory AiOS shell.

    This endpoint lets the frontend discover which backend routes are actually
    registered instead of guessing or calling hardcoded cloud paths. It does not
    execute commands and does not grant authority.
    """
    try:
        rules = []
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r.rule)):
            methods = sorted([m for m in rule.methods if m not in {"HEAD", "OPTIONS"}])
            rules.append({"path": str(rule.rule), "endpoint": str(rule.endpoint), "methods": methods})
        route_paths = sorted({r["path"] for r in rules})
        def has(path: str) -> bool:
            return path in route_paths
        domains = {
            "chat": {"ready": has("/api/chat"), "backend": "api/server/app.py + SarahMemoryNeuron.py"},
            "vision": {"ready": has("/api/vision/policy") and has("/api/vision/frame/status"), "backend": "api/server/appvision.py + SarahMemoryMSDC.py"},
            "media": {"ready": has("/api/media/capabilities") and has("/api/media/job/render"), "backend": "api/server/appmedia.py"},
            "communications": {"ready": has("/api/comm/health") and has("/api/comm/contacts/list"), "backend": "api/server/appcomm.py"},
            "sarahnet": {"ready": has("/api/net2/health") or has("/api/net/ui/status"), "backend": "api/server/appnet.py + appnet2.py"},
            "addons": {"ready": has("/api/store/addons/registry") or has("/api/store/addons/candidates"), "backend": "api/server/appstore.py + SarahMemoryTrustRegistry.py"},
            "terminal": {"ready": has("/api/terminal/status") and has("/api/terminal/execute"), "backend": "SarahMemoryTerminal.py"},
            "dlengine": {"ready": any(p.startswith("/api/avatar/rem") or p.startswith("/api/dl") for p in route_paths), "backend": "SarahMemoryDL.py / REM routes"},
        }
        return jsonify({
            "ok": True,
            "schema": "SarahMemory.ui_contracts.v1",
            "version": PROJECT_VERSION,
            "route_count": len(route_paths),
            "routes": rules,
            "domains": domains,
            "doctrine": {
                "local_first": True,
                "cloud_optional": True,
                "one_way_broker": True,
                "frontend_authority": False,
                "smget_required_for_actions": True,
            },
            "ts": time.time(),
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "schema": "SarahMemory.ui_contracts.v1"}), 500


@app.get("/api/runtime/thrash/status")
def api_runtime_thrash_status():
    """Read-only runtime anti-thrash status for the AiOS System Center."""
    try:
        try:
            from SarahMemoryOptimization import get_runtime_anti_thrash_profile
            profile = get_runtime_anti_thrash_profile()
        except Exception as exc:
            profile = {"ok": False, "error": str(exc), "schema": "SarahMemory.runtime_anti_thrash.v1"}
        return jsonify({
            "ok": True,
            "schema": "SarahMemory.runtime_status.v1",
            "profile": profile,
            "health_state_write_interval_seconds": _HEALTH_STATE_WRITE_INTERVAL_SECONDS,
            "last_health_state_write_ts": _LAST_HEALTH_STATE_WRITE_TS,
            "doctrine": {
                "bounded_loops": True,
                "rotating_logs_preferred": True,
                "batched_writes_preferred": True,
                "subprocess_timeouts_required": True,
                "authority": False,
            },
            "ts": time.time(),
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/status")
def api_status():
    """
    Explicit status endpoint separate from /api/health.
    Returns persisted server_state.json without rewriting it.
    """
    try:
        state = load_state() or {}
        if not isinstance(state, dict):
            state = {}
        return jsonify({
            "ok": True,
            "state": state,
            "version": PROJECT_VERSION,
            "ts": time.time(),
        }), 200
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "version": PROJECT_VERSION,
            "ts": time.time(),
        }), 500



@app.route("/api/session/bootstrap", methods=['POST'])
def api_session_bootstrap():
    """
    Phase A3 — Session Bootstrap API.
    Single canonical handshake endpoint used by Web UI (app.js) at startup.
    Aligns client and server runtime identity and exposes core feature flags.
    """
    try:
        payload = request.get_json(silent=True) or {} # jsonify handles non-JSON, no need for force=True
    except Exception as e:
        app_logger.warning(f"Failed to parse JSON for bootstrap, proceeding with empty payload: {e}")
        payload = {}

    client_info = {
        "env": (payload.get("client_env") or request.args.get("client_env") or "").strip(),
        "platform": (payload.get("platform") or request.args.get("platform") or "").strip(),
        "ui_version": (payload.get("ui_version") or request.args.get("ui_version") or "").strip(),
        "agent_name": (payload.get("agent_name") or request.args.get("agent_name") or "").strip(),
        "bridge": (payload.get("bridge") or request.args.get("bridge") or "").strip(),
    }

    runtime = _get_runtime_meta_safe()

    # Camera/mic/voice toggles (default to False if never touched yet)
    # Using app.config for Flask global state rather than globals()
    camera_enabled = app.config.get("CAMERA_ENABLED", False)
    mic_enabled = app.config.get("MIC_ENABLED", False)
    voice_enabled = app.config.get("VOICE_OUTPUT_ENABLED", False)

    features = {
        "camera": camera_enabled,
        "microphone": mic_enabled,
        "voice_output": voice_enabled,
        "hub_enabled": bool(net_mod is not None),
        "wallet_enabled": True, # Assume wallet is always enabled if META_DB is there
        "ledger_module": bool(ledger_mod is not None),
        "file_transfer": True, # Assume file transfer is always enabled
    }

    env = {
        "api_base": request.host_url.rstrip("/"),
        "web_root": request.host_url.rstrip("/") + "/api/",
    }

    return jsonify({
        "ok": True,
        "version": PROJECT_VERSION,
        "runtime": runtime,
        "client": client_info,
        "features": features,
        "env": env,
        "ts": time.time(),
    })

@app.route("/api/")
def api_index():
    """API root health banner (JSON).

    NOTE:
    - The Ranking SPA is served at "/" (root_index).
    - "/api/" is reserved for programmatic health/status checks used by the frontend heartbeat.
    """
    return jsonify(
        {
            "ok": True,
            "running": True,
            "service": "SarahMemory API",
            "version": PROJECT_VERSION,
        }
    )


def _req_host() -> str:
    """Return host without port, lowercased."""
    try:
        return (request.host or "").split(":", 1)[0].strip().lower()
    except Exception:
        return ""


def _want_ui_for_request() -> bool:
    """Host-based routing for the dual server.

    Local:
      - 127.0.0.1 / localhost -> Web UI
    Cloud:
      - ai.sarahmemory.com    -> Web UI
      - api.sarahmemory.com   -> Network Hub

    Default is hub, unless it matches UI conditions.
    """
    host = _req_host()
    if host in ("127.0.0.1", "localhost"):
        return True
    if host.startswith("ai."):
        return True
    return False

@app.route("/")
def root_index():
    """Serve the Ranking/Web UI (static SPA) at the site root.

    PythonAnywhere serves /assets and /static via static mappings, but "/" must be handled
    by Flask. If the UI build is present, return static/index.html; otherwise fall back
    to the API banner.
    """
    # Prefer the Web UI (Lovable/Vite dist) for local + ai.* host.
    if _want_ui_for_request():
        ui_index = os.path.join(UI_DIST_DIR, "index.html")
        if os.path.isfile(ui_index):
            return send_from_directory(UI_DIST_DIR, "index.html")

    # Otherwise, show the Network Hub landing (legacy /api/server/static/index.html)
    hub_index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(hub_index):
        return send_from_directory(STATIC_DIR, "index.html")
    return redirect("/api/")


# -----------------------------------------------------------------------------
# Web UI dist asset serving (local + ai.sarahmemory.com)
# -----------------------------------------------------------------------------

@app.route("/assets/<path:filename>")
def ui_assets(filename):
    if _want_ui_for_request():
        base = os.path.join(UI_DIST_DIR, "assets")
        if os.path.isdir(base):
            return send_from_directory(base, filename)
    return abort(404)


@app.route("/themes/<path:filename>")
def ui_themes(filename):
    if _want_ui_for_request():
        base = os.path.join(UI_DIST_DIR, "themes")
        if os.path.isdir(base):
            return send_from_directory(base, filename)
    return abort(404)


@app.route("/favicon.ico")
def ui_favicon():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "favicon.ico")):
        return send_from_directory(UI_DIST_DIR, "favicon.ico")
    return abort(404)


@app.route("/robots.txt")
def ui_robots():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "robots.txt")):
        return send_from_directory(UI_DIST_DIR, "robots.txt")
    return abort(404)


@app.route("/placeholder.svg")
def ui_placeholder():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "placeholder.svg")):
        return send_from_directory(UI_DIST_DIR, "placeholder.svg")
    return abort(404)


@app.route("/<path:path>")
def ui_spa_fallback(path):
    """SPA fallback for non-/api routes.

    Vite builds often use client-side routing; unknown paths must return index.html.
    """
    # Never hijack API routes
    if path.startswith("api/"):
        return abort(404)

    if _want_ui_for_request():
        candidate = os.path.join(UI_DIST_DIR, path)
        if os.path.isfile(candidate):
            return send_from_directory(UI_DIST_DIR, path)
        # Client-side route: return index.html
        ui_index = os.path.join(UI_DIST_DIR, "index.html")
        if os.path.isfile(ui_index):
            return send_from_directory(UI_DIST_DIR, "index.html")

    # Fallback to hub static if present
    candidate = os.path.join(STATIC_DIR, path)
    if os.path.isfile(candidate):
        return send_from_directory(STATIC_DIR, path)
    return abort(404)

@app.route("/api/static/<path:filename>")
def static_serv(filename):
    return send_from_directory(STATIC_DIR, filename)

# Loose assets for the hub index (icons, hero image, QR code, etc.)
# This lets relative URLs like "SOFTDEV0_LLC_Logo.png" work from /api/
# by serving them from either STATIC_DIR or the project BASE_DIR.
ASSET_EXTS = {
    "png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "bmp",
    "css", "js", "map", "json", "txt", "xml"
}

@app.route("/api/<path:filename>")
def api_loose_assets(filename: str):
    # Do not interfere with explicit API endpoints like /api/health or /api/leaderboard.
    # Flask prefers static rules (/api/health) over this dynamic one, so those will still win.
    if "." not in filename:
        # No extension: let the real API routes handle it (or 404 there).
        # We just return a 404 JSON so this route doesn't claim it.
        return jsonify({"error": "not an asset"}), 404

    ext = filename.rsplit(".", 1).lower()
    if ext not in ASSET_EXTS:
        return jsonify({"error": "unsupported asset type", "file": filename}), 404

    # Try in /api/server/static first (STATIC_DIR), then in the project root (BASE_DIR).
    # Using iter for potential performance gain if many routes.
    for root in (STATIC_DIR, BASE_DIR):
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return send_from_directory(root, filename)

    return jsonify({"error": "asset not found", "file": filename}), 404

@app.route("/api/leaderboard")
def api_leaderboard():
    cache_key = 'leaderboard:10'
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)
    payload = {'leaders': read_top_nodes(limit=10)}
    _cache_set(cache_key, payload, ttl_s=5.0)
    return jsonify(payload)

def _perform_health_checks():
    """
    Fast + safe health checks.

    Returns: (ok: bool, notes: list[str], main_running: bool)

    Notes are short machine-readable strings so the UI / SarahNet rendezvous can decide
    whether to fall back (CLOUD/LAN/OFF) without crashing the API.
    """
    import json as _json  # local import to avoid boot-time surprises

    notes = []
    ok = True

    # 1) Core modules importability (best-effort)
    for mod_name in ("SarahMemoryGlobals", "SarahMemoryVoice", "SarahMemoryDatabase", "SarahMemoryAPI"):
        try:
            __import__(mod_name)
        except Exception as e:
            ok = False
            notes.append(f"import_failed:{mod_name}:{e}")

    # 2) server_state.json readable (STATE_DB is JSON, not sqlite)
    try:
        if os.path.exists(STATE_DB):
            try:
                with open(STATE_DB, "r", encoding="utf-8") as f:
                    _json.load(f)
            except Exception as e:
                ok = False
                notes.append(f"state_json_invalid:{e}")
        else:
            notes.append("state_json_missing")
    except Exception as e:
        ok = False
        notes.append(f"state_json_check_failed:{e}")

    # 3) meta.db reachable (sqlite)
    try:
        con = _connect_sqlite(META_DB)
        con.execute("SELECT 1")
        con.close()
    except Exception as e:
        ok = False
        notes.append(f"sqlite_meta_db_failed:{e}")

    # 4) Main process running flag (desktop installs). Safe on cloud.
    main_running = False
    try:
        fn = globals().get("_is_running")
        if callable(fn):
            main_running = bool(fn())
    except Exception as e:
        notes.append(f"main_running_check_failed:{e}")

    return bool(ok), (notes if isinstance(notes, list) else []), bool(main_running)


@app.get("/api/health")
def api_health():
    """
    Universal health endpoint.
    - running      → HTTP API is responding
    - main_running → optional desktop launcher process check
    - routing      → LLM/provider metadata for diagnostics + orchestration
    """
    ok, notes, main_running = _perform_health_checks()
    status = "ok" if ok else "down"
    ts = time.time()

    # --- Routing metadata (safe + non-breaking) ---
    routing_meta = {
        "provider": os.getenv("ACTIVE_LLM_PROVIDER", "local"),
        "model": os.getenv("ACTIVE_LLM_MODEL", "auto"),
        "engine_mode": os.getenv("SARAH_AI_MODE", "standard"),
    }

    # Keep persisted server_state.json aligned with live truth, but do not write on every health poll.
    # The Web UI checks /api/health repeatedly; persisting volatile timestamps each poll causes unnecessary disk churn.
    try:
        global _LAST_HEALTH_STATE_WRITE_TS, _LAST_HEALTH_STATE_FINGERPRINT
        state_payload = {
            "ok": bool(ok),
            "notes": notes if isinstance(notes, list) else [],
            "main_running": bool(main_running),
            "running": True,
            "status": status,
            "version": PROJECT_VERSION,
            "source": "api_health_writer",
            "routing": routing_meta,
        }
        fp = _fingerprint_json(state_payload)
        should_write = (fp != _LAST_HEALTH_STATE_FINGERPRINT) or ((time.time() - _LAST_HEALTH_STATE_WRITE_TS) >= _HEALTH_STATE_WRITE_INTERVAL_SECONDS)
        if should_write:
            state = load_state() or {}
            if not isinstance(state, dict):
                state = {}
            state.update(state_payload)
            state["last_health_ts"] = ts
            save_state(state)
            _LAST_HEALTH_STATE_WRITE_TS = time.time()
            _LAST_HEALTH_STATE_FINGERPRINT = fp
    except Exception:
        pass

    return jsonify(
        {
            "ok": ok,
            "status": status,
            "running": True,
            "main_running": main_running,
            "version": PROJECT_VERSION,
            "ts": ts,
            "notes": notes,
            "routing": routing_meta,  # ← required for diagnostics
        }
    ), 200

@app.route("/api/vision/frame", methods=["POST"])
def api_vision_frame():
    """Cache the latest UI vision frame for the current session.

    Intended for Custom / Web UI low-FPS webcam pushes so /api/chat can reuse the
    freshest frame without forcing every non-vision chat message to upload media.
    """
    try:
        payload = request.get_json(silent=True) or {}
        session_id = _get_or_create_session_id(payload)
        frame_rec = _normalize_vision_frame_payload(payload)
        if not frame_rec:
            return jsonify({"ok": False, "error": "Missing frame/image payload.", "session_id": session_id}), 400
        stored = _store_latest_vision_frame(session_id, frame_rec)
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "frame_cached": True,
            "ts": stored.get("ts"),
            "source": stored.get("source"),
            "has_frame": True,
        }), 200
    except Exception as e:
        app_logger.error(f"/api/vision/frame failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/vision/frame/status-legacy")
def api_vision_frame_status_legacy():
    """Small debug/status endpoint for the active session vision cache."""
    try:
        payload = {
            "session_id": request.args.get("session_id") or request.headers.get("X-Session-Id") or request.headers.get("X-Session-ID")
        }
        session_id = _get_or_create_session_id(payload)
        rec = _get_latest_vision_frame(session_id)
        return jsonify({
            "ok": True,
            "source": "app.py.legacy_frame_cache",
            "canonical_endpoint": "/api/vision/frame/status",
            "session_id": session_id,
            "has_frame": bool(rec),
            "ts": (rec or {}).get("ts"),
            "source": (rec or {}).get("source"),
            "width": (rec or {}).get("width"),
            "height": (rec or {}).get("height"),
        }), 200
    except Exception as e:
        app_logger.error(f"/api/vision/frame/status failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500





def _sm_match_quick_system_route(text: str) -> dict | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    t = t.replace("capslock", "caps lock").replace("numlock", "num lock").replace("scrolllock", "scroll lock")
    if any(p in t for p in ("today's date", "todays date", "current date", "what is the date", "what is todays date", "what is today date", "what's today's date", "what day is it", "what time is it", "current time", "date and time")):
        kind = "datetime"
        if "time" in t and "date" not in t and "today" not in t and "day" not in t:
            kind = "time"
        elif any(p in t for p in ("today's date", "todays date", "current date", "what is the date", "what is today date", "what's today's date", "what day is it")):
            kind = "date"
        return {"route_id": "system.datetime.current", "kind": kind}
    if any(k in t for k in ("caps lock", "num lock", "scroll lock")) and any(k in t for k in ("turn", "put", "set", "enable", "disable", "switch")):
        key_name = "caps_lock" if "caps lock" in t else ("num_lock" if "num lock" in t else "scroll_lock")
        state = "off" if any(k in t for k in ("turn off", "switch off", "disable")) else "on"
        return {"route_id": "system.keyboard.key_state", "key_name": key_name, "requested_state": state}
    if "keyboard" in t and any(k in t for k in ("light", "lights", "led", "rgb", "backlight", "color", "colors", "colour", "colours")):
        color = None
        for c in ("red", "green", "blue", "purple", "yellow", "white", "orange", "pink"):
            if c in t:
                color = c
                break
        return {"route_id": "drivers.keyboard.lighting", "device_type": "keyboard", "value": color or "requested", "requested_state": "on" if any(k in t for k in ("turn on", "enable", "activate")) else None}
    return None


def _sm_now_reply(kind: str) -> str:
    now = datetime.now()
    if kind == "time":
        return f"The current time is {now.strftime('%I:%M %p').lstrip('0')}."
    if kind == "datetime":
        return f"Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p').lstrip('0')}."
    return f"Today's date is {now.strftime('%A, %B %d, %Y')}."


def _sm_set_lock_key_state(key_name: str, requested_state: str) -> tuple[bool, str, dict]:
    requested_state = str(requested_state or "on").lower()
    key_name = str(key_name or "caps_lock").lower()
    vk_map = {"caps_lock": 0x14, "num_lock": 0x90, "scroll_lock": 0x91}
    nice_map = {"caps_lock": "Caps Lock", "num_lock": "Num Lock", "scroll_lock": "Scroll Lock"}
    if key_name not in vk_map:
        return False, "Unsupported keyboard lock key.", {"key_name": key_name}
    if os.name != 'nt':
        return False, f"{nice_map.get(key_name, key_name)} control is not yet implemented for this operating system.", {"key_name": key_name, "os": os.name}
    try:
        import ctypes, time as _time
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        vk = vk_map[key_name]
        desired_on = requested_state != 'off'
        KEYEVENTF_EXTENDEDKEY = 0x0001
        KEYEVENTF_KEYUP = 0x0002
        current_on = bool(user32.GetKeyState(vk) & 1)
        changed = False
        for _ in range(4):
            current_on = bool(user32.GetKeyState(vk) & 1)
            if current_on == desired_on:
                break
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            changed = True
            _time.sleep(0.05)
        final_on = bool(user32.GetKeyState(vk) & 1)
        if final_on == desired_on:
            state_word = 'ON' if final_on else 'OFF'
            if changed:
                return True, f"{nice_map.get(key_name, key_name)} turned {state_word}.", {"key_name": key_name, "requested_state": requested_state, "final_state": state_word.lower()}
            return True, f"{nice_map.get(key_name, key_name)} is already {state_word}.", {"key_name": key_name, "requested_state": requested_state, "final_state": state_word.lower()}
        return False, f"Unable to set {nice_map.get(key_name, key_name)} to the requested state.", {"key_name": key_name, "requested_state": requested_state, "final_state": 'on' if final_on else 'off'}
    except Exception as e:
        return False, f"Failed to change {nice_map.get(key_name, key_name)}: {e}", {"key_name": key_name, "requested_state": requested_state, "error": str(e)}


def _sm_try_keyboard_lighting(text: str, quick_route: dict) -> tuple[bool, str, dict]:
    try:
        import shutil as _shutil
        import subprocess as _subprocess
        color = str(quick_route.get('value') or 'requested').strip().lower()
        op = _shutil.which('openrgb') or _shutil.which('OpenRGB')
        if op:
            cmd = [op, '--mode', 'static']
            color_map = {
                'red': 'FF0000', 'green': '00FF00', 'blue': '0000FF', 'purple': '800080',
                'yellow': 'FFFF00', 'white': 'FFFFFF', 'orange': 'FFA500', 'pink': 'FFC0CB',
            }
            hexv = color_map.get(color)
            if hexv:
                cmd.extend(['--color', hexv])
            proc = _subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if proc.returncode == 0:
                return True, f"Keyboard lighting set to {color} through generic OpenRGB control.", {"driver_id": 'openrgb', 'action_id': 'keyboard_rgb_set', 'stdout': proc.stdout[-500:]}
        import appdrivers as _drv  # type: ignore
    except Exception as e:
        return False, f"Keyboard lighting route matched, but no executable runtime is available: {e}", {"route_id": quick_route.get('route_id')}
    try:
        driver_ids = list(_drv._discover_driver_ids()) if hasattr(_drv, '_discover_driver_ids') else []
    except Exception:
        driver_ids = []
    matches = []
    for did in driver_ids:
        try:
            mf = _drv._load_manifest(did) if hasattr(_drv, '_load_manifest') else {}
        except Exception:
            mf = {}
        raw = json.dumps(mf or {}, ensure_ascii=False).lower()
        if 'keyboard' in raw or 'rgb' in raw or 'lighting' in raw or 'backlight' in raw:
            matches.append((did, mf))
    if not matches:
        return False, 'Keyboard lighting route matched, but no governed keyboard lighting driver or OpenRGB runtime was discovered.', {"driver_matches": []}
    preferred_actions = ['set_color', 'set_led_color', 'set_rgb', 'set_backlight', 'lighting_set', 'keyboard_lighting', 'keyboard_rgb_set']
    for did, mf in matches:
        try:
            _drv._driver_discover(did, payload={"source": "api_chat_quick_route", "user_text": text})
        except Exception:
            pass
        cfg = _drv._load_config(did) if hasattr(_drv, '_load_config') else {}
        try:
            _drv._driver_connect(did, cfg=cfg or {}, connect_payload={"source": "api_chat_quick_route", "user_text": text})
        except Exception:
            pass
        mod, err = _drv._load_driver_module(did) if hasattr(_drv, '_load_driver_module') else (None, 'driver loader unavailable')
        if err or mod is None:
            continue
        actions_blob = json.dumps(mf or {}, ensure_ascii=False).lower()
        for action_id in preferred_actions:
            if (action_id.lower() in actions_blob) or hasattr(mod, f'action_{action_id}') or hasattr(mod, 'driver_action'):
                try:
                    context = _drv._build_driver_context(did, instance_id=_drv._session_get(did).get('instance_id') if hasattr(_drv, '_session_get') else None, extra={"action_id": action_id})
                    payload = {"requested_action": action_id, "entities": {"device_type": "keyboard", "value": quick_route.get('value'), "requested_state": quick_route.get('requested_state')}, "user_text": text}
                    if hasattr(mod, 'driver_action'):
                        out = mod.driver_action(action_id=action_id, context=context, payload=payload)
                    else:
                        out = getattr(mod, f'action_{action_id}')(context=context, payload=payload)
                    if isinstance(out, dict) and bool(out.get('ok', True)):
                        color = quick_route.get('value') or 'requested color'
                        return True, f"Keyboard lighting set to {color} through governed driver {did}.", {"driver_id": did, "action_id": action_id, "driver_result": out}
                except Exception:
                    continue
    return False, 'Keyboard lighting route matched and drivers were discovered, but no executable lighting action succeeded.', {"driver_matches": [m[0] for m in matches]}


def _sm_execute_quick_route(text: str) -> tuple[bool, dict | None]:
    route = _sm_match_quick_system_route(text)
    if not route:
        return False, None
    route_id = str(route.get('route_id') or '')
    if route_id == 'system.datetime.current':
        reply = _sm_now_reply(str(route.get('kind') or 'date'))
        bundle = _sm_make_outward_bundle(reply, meta={"source": "quick_system_route", "engine": "local_datetime", "intent": "system", "route_id": route_id, "version": PROJECT_VERSION})
        bundle.setdefault('actions', [])
        bundle['actions'].append({"type": "route_match", "route_id": route_id})
        return True, bundle
    if route_id == 'system.keyboard.key_state':
        ok, reply, details = _sm_set_lock_key_state(str(route.get('key_name') or 'caps_lock'), str(route.get('requested_state') or 'on'))
        bundle = _sm_make_outward_bundle(reply, meta={"source": "quick_system_route", "engine": "keyboard_key_state", "intent": "system", "route_id": route_id, "version": PROJECT_VERSION}, actions=[{"type": "keyboard_key_state", **details}], errors=[] if ok else [details])
        bundle['ok'] = bool(ok)
        return True, bundle
    if route_id == 'drivers.keyboard.lighting':
        ok, reply, details = _sm_try_keyboard_lighting(text, route)
        bundle = _sm_make_outward_bundle(reply, meta={"source": "quick_system_route", "engine": "keyboard_lighting", "intent": "drivers", "route_id": route_id, "version": PROJECT_VERSION}, actions=[{"type": "driver_route", **details}], errors=[] if ok else [details])
        bundle['ok'] = bool(ok)
        return True, bundle
    return False, None

def _sm_ingress_catalog() -> list[dict]:
    return [
        {"route_id": "research.weather.current", "domain": "research", "action": "weather_current", "target_module": "SarahMemoryResearch", "transport_target": "internal_research_lane", "keywords": ["weather", "temperature", "forecast", "rain", "sunny", "humidity", "wind"], "examples": ["what is the temperature right now in nacogdoches texas", "current weather in lufkin texas", "how hot is it outside in dallas"]},
        {"route_id": "research.weather.forecast", "domain": "research", "action": "weather_forecast", "target_module": "SarahMemoryResearch", "transport_target": "internal_research_lane", "keywords": ["forecast", "tomorrow", "next", "day", "days", "weather", "temperature"], "examples": ["what is the weather like tomorrow in nacogdoches texas", "give me the next 3 day forecast in lufkin texas", "forecast this weekend in houston"]},
        {"route_id": "drivers.device.control", "domain": "drivers", "action": "device_control", "target_module": "appdrivers", "transport_target": "/api/drivers", "keywords": ["driver", "device", "mouse", "keyboard", "webcam", "camera", "microphone", "led", "razer", "usb", "bluetooth"], "examples": ["turn my webcam on", "turn my mouse led color to red", "connect to my razer mouse"]},
        {"route_id": "avatar.create.activate", "domain": "avatar", "action": "create_activate_avatar", "target_module": "UnifiedAvatarController", "transport_target": "internal_avatar_lane", "keywords": ["avatar", "3d", "unreal", "blender", "mouth", "eyes", "panel", "character"], "examples": ["make me a red 3d ball with eyes and moving mouth in unreal engine and place it into the avatar panel", "change the system avatar", "load this as my avatar"]},
        {"route_id": "creative.general.generate", "domain": "creative", "action": "generate_creative", "target_module": "SarahMemoryCanvasStudio", "transport_target": "internal_creative_lane", "keywords": ["create image", "generate image", "make image", "draw picture", "create music", "generate song", "create video", "art", "render"], "examples": ["create an image of a cat on a skateboard", "make me a song about texas", "generate a short video intro"]},
        {"route_id": "documents.office.write", "domain": "documents", "action": "write_document", "target_module": "SarahMemorySi", "transport_target": "internal_software_lane", "keywords": ["word", "document", "write", "docx", "report", "essay", "open office", "excel", "spreadsheet", "notepad", "website", "dreamweaver", "edge", "browser"], "examples": ["write me a word document on penguins in the arctic", "open word and create a report", "make a document about safety procedures", "open edge and search for tigers", "open excel and create a checkbook page"]},
        {"route_id": "email.mail.automation", "domain": "email", "action": "mail_automation", "target_module": "appcomm", "transport_target": "/api/comm", "keywords": ["email", "gmail", "outlook", "spam", "unsubscribe", "trash", "mailbox", "inbox"], "examples": ["open my emails and unsubscribe to all known spam messages then empty my spam trash daily", "check my inbox", "delete spam mail"]},
        {"route_id": "reminder.schedule.task", "domain": "reminder", "action": "schedule_task", "target_module": "appcomm", "transport_target": "/api/comm", "keywords": ["remind", "schedule", "daily", "every day", "weekly", "monthly", "calendar", "task"], "examples": ["remind me tomorrow at 5 pm", "empty my spam trash daily", "schedule a recurring cleanup"]},
        {"route_id": "system.application.control", "domain": "system", "action": "application_control", "target_module": "SarahMemorySi", "transport_target": "internal_software_lane", "keywords": ["open", "close", "launch", "start", "stop", "app", "program", "application", "window"], "examples": ["open notepad", "launch unreal engine", "close the browser"]},
        {"route_id": "research.general.web", "domain": "research", "action": "web_research", "target_module": "SarahMemoryResearch", "transport_target": "internal_research_lane", "keywords": ["research", "look up", "find", "search", "internet", "web"], "examples": ["research this topic online", "look this up for me", "find current information on this"]},
        {"route_id": "chat.general", "domain": "chat", "action": "general_reply", "target_module": "SarahMemoryReply", "transport_target": "/api/chat", "keywords": ["chat", "question", "talk", "explain", "help"], "examples": ["hello", "how are you", "explain this to me"]},
    ]


def _sm_ingress_normalize_text(text: str) -> str:
    t = str(text or "").strip().lower()
    replacements = {"temperture": "temperature", "wether": "weather", "camra": "camera", "web cam": "webcam", "mose": "mouse", "lites": "lights", "coler": "color", "unreel": "unreal", "doccument": "document"}
    for bad, good in replacements.items():
        t = t.replace(bad, good)
    return re.sub(r"\s+", " ", t).strip()



def _sm_ingress_extract_entities(text: str, route_id: str) -> dict:
    norm = _sm_ingress_normalize_text(text)
    entities: dict[str, object] = {}
    weather_match = re.search(r"\bin\s+([a-z0-9 .,'-]+)$", norm)
    if route_id.startswith("research.weather") and weather_match:
        entities["location"] = weather_match.group(1).strip(" ?.,")
    if route_id == "research.weather.forecast":
        if "tomorrow" in norm:
            entities["day_offset"] = 1
        m_days = re.search(r"next\s+(\d+)\s+day", norm)
        if m_days:
            entities["days"] = int(m_days.group(1))
        elif "forecast" in norm and "days" not in entities:
            entities["days"] = 1 if "tomorrow" in norm else 3

    surface_task = {}
    try:
        if _sm_module_approved("SarahMemoryPreTokenAnalyzer", capability="classification"):
            import SarahMemoryPreTokenAnalyzer as _PTA  # type: ignore
            analysis = _PTA.analyze_text(text, context_packet={"source": "api_chat", "mode": "LOCAL"}) if hasattr(_PTA, 'analyze_text') else {}
            if isinstance(analysis, dict) and isinstance(analysis.get('surface_task'), dict):
                surface_task = dict(analysis.get('surface_task') or {})
            elif hasattr(_PTA, 'extract_surface_task'):
                data = _PTA.extract_surface_task(text)
                if isinstance(data, dict):
                    surface_task = dict(data)
    except Exception:
        surface_task = {}

    if surface_task:
        entities['surface_task'] = surface_task
        app_exec = str(surface_task.get('requested_app_exec') or surface_task.get('requested_app') or '').strip()
        if app_exec:
            entities['target_app_exec'] = app_exec
            entities['target_app'] = app_exec
        if route_id in {"system.application.control", "documents.office.write"}:
            entities['requested_state'] = 'open'
        task_kind = str(surface_task.get('task_kind') or '').strip().lower()
        if task_kind:
            entities['followup_action'] = task_kind
        for key in ('topic', 'title', 'document_text', 'draw_subject', 'document_name', 'pages', 'template_kind', 'search_query', 'target_url', 'headers'):
            if surface_task.get(key) not in (None, '', [], {}):
                entities[key] = surface_task.get(key)
        if task_kind == 'document_write':
            entities.setdefault('software_hint', 'microsoft_word')
        if task_kind in {'browser_search', 'browser_open_url'}:
            entities.setdefault('software_hint', 'browser')

    if route_id == "system.application.control" and 'target_app' not in entities:
        app_match = re.search(r"\b(?:open|launch|start|close|stop|quit|exit)\s+(.+)$", norm)
        if app_match:
            entities["target_app"] = app_match.group(1).strip(" ?.,")
        if any(k in norm for k in ("open ", "launch ", "start ")):
            entities["requested_state"] = "open"
        elif any(k in norm for k in ("close ", "stop ", "quit ", "exit ")):
            entities["requested_state"] = "close"

    if route_id == "drivers.device.control":
        for device in ("webcam", "camera", "mouse", "keyboard", "microphone"):
            if device in norm:
                entities["device_type"] = "webcam" if device == "camera" else device
                break
        for vendor in ("razer", "logitech", "corsair", "steelseries"):
            if vendor in norm:
                entities["vendor"] = vendor
                break
        color_match = re.search(r"\b(red|green|blue|purple|yellow|white|orange|pink)\b", norm)
        if color_match:
            entities["value"] = color_match.group(1)
        if any(k in norm for k in ("turn on", "activate", "enable", "start")):
            entities["requested_state"] = "on"
        elif any(k in norm for k in ("turn off", "disable", "stop")):
            entities["requested_state"] = "off"

    if route_id == "documents.office.write" and 'surface_task' not in entities:
        topic_match = re.search(r"\b(?:about|on)\s+(.+)$", norm)
        if topic_match:
            entities["topic"] = topic_match.group(1).strip(" ?.")
        if "word" in norm:
            entities["software_hint"] = "microsoft_word"

    if route_id == "email.mail.automation":
        if "daily" in norm or "every day" in norm:
            entities["schedule"] = "daily"
        elif "weekly" in norm or "every week" in norm:
            entities["schedule"] = "weekly"
        elif "monthly" in norm or "every month" in norm:
            entities["schedule"] = "monthly"
        entities["unsubscribe"] = "unsubscribe" in norm
        entities["target_folder"] = "spam" if "spam" in norm else "inbox"

    if route_id == "avatar.create.activate":
        if "unreal" in norm:
            entities["engine_preference"] = "unreal"
        color_match = re.search(r"\b(red|green|blue|purple|yellow|white|orange|pink)\b", norm)
        if color_match:
            entities["color"] = color_match.group(1)
        if "3d ball" in norm or "sphere" in norm or "ball" in norm:
            entities["shape"] = "ball"
        if "eyes" in norm:
            entities["eyes"] = True
        if "mouth" in norm:
            entities["mouth"] = "moving" if "moving mouth" in norm else True
        if "avatar panel" in norm:
            entities["target_surface"] = "avatar_panel"
    return entities


def _sm_build_virtual_ingress_route(text: str, payload: dict | None = None, context_packet: dict | None = None) -> dict:
    payload = payload or {}
    context_packet = context_packet or {}
    original = str(text or "").strip()
    normalized = _sm_ingress_normalize_text(original)
    cards = _sm_ingress_catalog()
    query_vec = None
    embed_fn = None
    cos_fn = None
    try:
        if _sm_module_approved("SarahMemoryAdvCU", capability="classification"):
            import SarahMemoryAdvCU as _AdvCU  # type: ignore
            embed_fn = getattr(_AdvCU, "embed_text", None)
            cos_fn = getattr(_AdvCU, "cosine_similarity", None)
            if callable(embed_fn) and callable(cos_fn):
                qv = embed_fn(normalized)
                if isinstance(qv, list) and qv:
                    query_vec = qv[0]
    except Exception:
        query_vec = None

    best: dict | None = None
    best_score = -1.0
    scored_cards: list[dict] = []
    query_tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    for card in cards:
        texts = [card.get("route_id", "")] + list(card.get("examples", []))
        semantic = 0.0
        if query_vec is not None and callable(embed_fn) and callable(cos_fn):
            try:
                cvecs = embed_fn([_sm_ingress_normalize_text(t) for t in texts])
                semantic = max(float(cos_fn(query_vec, cv)) for cv in cvecs if cv) if cvecs else 0.0
            except Exception:
                semantic = 0.0
        lexical = 0.0
        try:
            keyword_hits = 0.0
            for kw in card.get("keywords", []):
                if kw in normalized:
                    keyword_hits += 1.0
                else:
                    ratio = difflib.SequenceMatcher(None, normalized, kw).ratio()
                    if ratio >= 0.86:
                        keyword_hits += 0.6
            if card.get("keywords"):
                lexical = max(lexical, min(1.0, keyword_hits / max(1.0, len(card.get("keywords", [])) / 2.5)))
            for ex in card.get("examples", []):
                ex_norm = _sm_ingress_normalize_text(ex)
                ex_tokens = set(re.findall(r"[a-z0-9_]+", ex_norm))
                if ex_tokens:
                    overlap = len(query_tokens & ex_tokens) / max(1, len(query_tokens | ex_tokens))
                    lexical = max(lexical, float(overlap))
        except Exception:
            lexical = lexical or 0.0
        score = semantic * 0.72 + lexical * 0.28 if query_vec is not None else lexical
        scored_cards.append({"route_id": card.get("route_id"), "semantic": round(float(semantic), 4), "lexical": round(float(lexical), 4), "score": round(float(score), 4)})
        if score > best_score:
            best_score = float(score)
            best = dict(card)

    best = best or dict(cards[-1])
    route_id = str(best.get("route_id") or "chat.general")
    entities = _sm_ingress_extract_entities(original, route_id)
    surface_task = dict(entities.get('surface_task') or {}) if isinstance(entities.get('surface_task'), dict) else {}
    task_kind = str(surface_task.get('task_kind') or '').strip().lower()
    if task_kind in {'document_write', 'open_named_document', 'spreadsheet_template', 'website_scaffold'}:
        route_id = 'documents.office.write'
        best['domain'] = 'documents'
        best['action'] = 'write_document'
        best['target_module'] = 'SarahMemorySi'
        best['transport_target'] = 'internal_software_lane'
    elif task_kind in {'browser_search', 'browser_open_url'}:
        route_id = 'system.application.control'
        best['domain'] = 'system'
        best['action'] = 'application_control'
        best['target_module'] = 'SarahMemorySi'
        best['transport_target'] = 'internal_software_lane'
    elif route_id == 'creative.general.generate':
        best['domain'] = 'creative'
    intent_hint = str(best.get("domain") or "chat")
    if route_id.startswith("research.weather"):
        intent_hint = "research"
    elif route_id.startswith("avatar."):
        intent_hint = "creative"
    elif route_id.startswith("reminder."):
        intent_hint = "time"
    elif route_id == "creative.general.generate":
        intent_hint = "creative"
    elif route_id.startswith(("drivers.", "system.", "documents.", "email.")):
        intent_hint = "action"
    needs_discovery = bool(route_id in {"avatar.create.activate", "documents.office.write", "drivers.device.control", "system.application.control", "creative.general.generate"})
    return {"ok": True, "route_id": route_id, "domain": str(best.get("domain") or "chat"), "action": str(best.get("action") or "general_reply"), "target_module": str(best.get("target_module") or "SarahMemoryReply"), "transport_target": str(best.get("transport_target") or "/api/chat"), "intent_hint": intent_hint, "confidence": round(max(0.0, min(0.99, best_score if best_score >= 0 else 0.15)), 4), "entities": entities, "normalized_text": normalized, "source": "semantic_ingress_router", "needs_discovery": needs_discovery, "route_trace": scored_cards[:12]}


def _sm_proposed_action_from_ingress(ingress_route: dict) -> dict:
    route_id = str((ingress_route or {}).get("route_id") or "")
    domain = str((ingress_route or {}).get("domain") or "chat")
    entities = dict((ingress_route or {}).get("entities") or {})

    requested_state = str(entities.get("requested_state") or "").strip().lower()
    target = str(
        entities.get("target_app_exec")
        or entities.get("target_app")
        or entities.get("software_hint")
        or entities.get("document_name")
        or entities.get("device_type")
        or entities.get("target_url")
        or ""
    ).strip()

    action = str(entities.get("action") or "").strip().lower()
    action_type = ""
    if route_id in {"system.application.control", "documents.office.write"}:
        if requested_state in {"close", "quit", "exit", "stop"}:
            action, action_type = "close", "close_app"
        elif requested_state in {"focus", "bring"}:
            action, action_type = "focus", "focus_window"
        elif requested_state == "maximize":
            action, action_type = "maximize", "maximize_window"
        elif requested_state == "minimize":
            action, action_type = "minimize", "minimize_window"
        else:
            action, action_type = "open", "open_app"
    elif route_id == "drivers.device.control":
        action = action or requested_state or "control"
        action_type = "device_control"
    elif route_id == "email.mail.automation":
        action = action or "mail_automation"
        action_type = "mail_automation"
    elif route_id == "reminder.schedule.task":
        action = action or "schedule_task"
        action_type = "schedule_task"
    elif route_id == "creative.general.generate":
        action = action or "create"
        action_type = "create_artifact"

    return {
        "intent": domain.upper(),
        "route_id": route_id,
        "action": action,
        "action_type": action_type,
        "target": target,
        "subsystems": [str((ingress_route or {}).get("target_module") or "")],
        "target_files": [],
        "dry_run": False,
        "touches_network": bool(domain in {"research", "email", "network", "store"}),
        "touches_privacy": bool(domain in {"email", "drivers", "system"}),
        "touches_filesystem": bool(domain in {"documents", "avatar", "system", "media"}),
        "sends_data": bool(domain in {"email", "network", "store"}),
        "entities": entities,
    }


def _sm_operatorcore_should_handle(ingress_route: dict | None) -> bool:
    route_id = str((ingress_route or {}).get("route_id") or "")
    return route_id in {"system.application.control", "documents.office.write"}


def _sm_operatorcore_execution_mode(payload: dict | None, *, local_only: bool, safe_mode: bool, require_user: bool) -> str:
    payload = payload or {}
    requested = str(payload.get("execution_mode") or payload.get("operator_mode") or payload.get("smget_mode") or "").strip().lower()
    if requested in {"apply", "simulate", "draft"}:
        return requested
    if safe_mode or require_user:
        return "simulate"
    try:
        if _is_cloud_request():
            return "simulate"
    except Exception:
        pass
    return "apply" if bool(local_only) else "simulate"


def _sm_operatorcore_bundle_from_result(
    operator_packet: dict,
    *,
    ingress_route: dict,
    context_packet: dict,
    gov_decision: str,
    gov_reasons: list,
    local_only: bool,
    developersmode: bool,
) -> dict:
    contract = dict(operator_packet.get("contract") or {})
    result = dict(operator_packet.get("result") or {})
    ok = bool(operator_packet.get("ok"))
    raw_reply = str(result.get("summary") or "").strip()
    if not raw_reply:
        raw_reply = "Governed execution completed." if ok else "Governed execution could not complete the request."

    meta = {
        "source": "operator_core",
        "engine": "SMGET",
        "intent": str((ingress_route or {}).get("domain") or "action"),
        "local_only": bool(local_only),
        "version": PROJECT_VERSION,
        "session_id": context_packet.get("session_id"),
        "governor": {"decision": gov_decision, "reasons": gov_reasons} if developersmode else {"decision": gov_decision},
        "route_id": str((ingress_route or {}).get("route_id") or ""),
        "operator_contract_id": contract.get("contract_id"),
        "operator_audit_id": result.get("audit_id"),
        "operator_state": result.get("state"),
        "operator_mode": result.get("execution_mode") or contract.get("execution_mode"),
        "operator_executor": result.get("executor_name") or contract.get("executor_name"),
    }
    if developersmode:
        meta["smget"] = {
            "contract": contract,
            "result": result,
        }

    actions = [{
        "type": "smget_operator_result",
        "route_id": str((ingress_route or {}).get("route_id") or ""),
        "contract_id": contract.get("contract_id"),
        "audit_id": result.get("audit_id"),
        "state": result.get("state"),
        "execution_mode": result.get("execution_mode") or contract.get("execution_mode"),
        "success": ok,
    }]
    errors = [str(x) for x in (result.get("errors") or [])]
    warnings = [str(x) for x in (result.get("warnings") or [])]
    if warnings:
        actions.append({"type": "smget_operator_warnings", "warnings": warnings[:10]})

    bundle = _sm_make_outward_bundle(
        _sm_present_text(raw_reply, intent=str((ingress_route or {}).get("domain") or "action"), meta=meta),
        meta=meta,
        actions=actions,
        errors=errors,
        raw_answer=raw_reply,
    )
    bundle["ok"] = ok
    return bundle


def _sm_try_operatorcore_request(
    text: str,
    *,
    payload: dict,
    context_packet: dict,
    ingress_route: dict,
    local_only: bool,
    safe_mode: bool,
    gov_decision: str,
    gov_reasons: list,
    gov_require_user: bool,
    developersmode: bool,
) -> dict | None:
    if not _sm_operatorcore_should_handle(ingress_route):
        return None

    try:
        from SarahMemoryOperatorCore import process_action_request as _smget_process_action_request  # type: ignore
    except Exception as e:
        app_logger.warning(f"OperatorCore not available for ingress execution: {e}")
        return None

    execution_mode = _sm_operatorcore_execution_mode(
        payload,
        local_only=local_only,
        safe_mode=safe_mode,
        require_user=gov_require_user,
    )

    proposed_action = dict((context_packet.get("meta") or {}).get("proposed_action") or {})
    proposed_action.setdefault("route_id", str((ingress_route or {}).get("route_id") or ""))
    proposed_action.setdefault("action", str(proposed_action.get("action") or "open"))
    proposed_action.setdefault("action_type", str(proposed_action.get("action_type") or "open_app"))
    proposed_action.setdefault("target", str(proposed_action.get("target") or proposed_action.get("entities", {}).get("target_app") or "").strip())

    op_meta = {
        "session_id": context_packet.get("session_id"),
        "source": "api_chat",
        "surface": str(context_packet.get("ui") or payload.get("ui") or "webui"),
        "source_surface": str(context_packet.get("ui") or payload.get("ui") or "webui"),
        "execution_mode": execution_mode,
        "user_consented": bool((context_packet.get("meta") or {}).get("user_consented")),
        "ingress_route": ingress_route,
        "context_packet": context_packet,
    }

    try:
        operator_packet = _smget_process_action_request(
            text,
            origin="api_chat",
            meta=op_meta,
            proposed_action=proposed_action,
            execution_mode=execution_mode,
        )
    except Exception as e:
        app_logger.error(f"OperatorCore execution failed: {e}", exc_info=True)
        return None

    if not isinstance(operator_packet, dict):
        return None

    return _sm_operatorcore_bundle_from_result(
        operator_packet,
        ingress_route=ingress_route,
        context_packet=context_packet,
        gov_decision=gov_decision,
        gov_reasons=gov_reasons,
        local_only=local_only,
        developersmode=developersmode,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Primary chat endpoint used by the Web UI.

    Hardcoded to the SarahMemory governed flow:
    Ingress -> Context Packet -> Governor -> AdvCU/Neuron -> Compare -> Presentation -> Reply Bundle
    """
    try:
        payload = request.get_json(silent=True) or {}

        intent = str(payload.get("intent") or "").strip()
        tone = str(payload.get("tone") or "").strip()
        complexity = str(payload.get("complexity") or "").strip()
        avatar_request = bool(payload.get("avatar_request") or payload.get("avatar") or False)
        diagnostics_ping = bool(payload.get("diagnostics_ping") or payload.get("diag_ping") or False)
        text = (payload.get("text") or payload.get("message") or payload.get("q") or "").strip()

        try:
            import SarahMemoryGlobals as G  # type: ignore
            payload_local_only = bool(payload.get("local_only") or payload.get("offline") or payload.get("LOCAL_ONLY_MODE") or payload.get("force_local_only"))
            local_only = bool(getattr(G, "LOCAL_ONLY_MODE", False) or payload_local_only)
            payload_safe_mode = bool(payload.get("safe_mode") or payload.get("SAFE_MODE") or payload.get("force_safe_mode"))
            safe_mode = bool(getattr(G, "SAFE_MODE", False) or payload_safe_mode)
            neoskymatrix = bool(getattr(G, "NEOSKYMATRIX", False))
            developersmode = bool(getattr(G, "DEVELOPERSMODE", False))
        except Exception:
            local_only = False
            safe_mode = False
            neoskymatrix = False
            developersmode = False

        context_packet = _sm_build_context_packet(
            payload,
            text,
            intent,
            tone,
            complexity,
            avatar_request,
            local_only=local_only,
            safe_mode=safe_mode,
            neoskymatrix=neoskymatrix,
            developersmode=developersmode,
        )
        ingress_route = _sm_build_virtual_ingress_route(text, payload=payload, context_packet=context_packet)
        context_packet.setdefault("meta", {})["ingress_route"] = ingress_route
        context_packet["meta"]["proposed_action"] = _sm_proposed_action_from_ingress(ingress_route)
        if not intent and str(ingress_route.get("intent_hint") or "").strip():
            intent = str(ingress_route.get("intent_hint") or "").strip()
        context_packet, frame_rec = _attach_cached_or_inline_vision_frame(payload, context_packet, user_text=text)
        _sm_refresh_core_registry(force=False)

        if not text:
            if diagnostics_ping:
                bundle = _sm_make_outward_bundle(
                    "Diagnostics ping acknowledged.",
                    meta={
                        "source": "api",
                        "engine": "diagnostics_ping",
                        "intent": "diagnostics",
                        "local_only": local_only,
                        "version": PROJECT_VERSION,
                    },
                )
                return jsonify(bundle), 200
            return jsonify({
                "ok": False,
                "error": "Missing 'text' in request body.",
                "meta": {"source": "api", "reason": "no_text", "version": PROJECT_VERSION},
            }), 400

        handled, quick_bundle = _sm_execute_quick_route(text)
        if handled and quick_bundle is not None:
            return jsonify(quick_bundle), 200

        selfaware_fact_bundle = _sm_try_selfaware_fact_route(text, source="api_chat")
        if isinstance(selfaware_fact_bundle, dict):
            return jsonify(selfaware_fact_bundle), 200

        if _is_identity_question(text):
            ident = _identity_payload()
            low = text.strip().lower()
            if "version" in low:
                raw_reply = f"My name is {ident['name']} — your {ident['platform']} companion. Server version: {ident['version']}."
            elif any(k in low for k in (
                "who made you", "who created you", "creator", "who built you",
                "who designed you", "designer", "engineer", "who engineered you",
            )):
                raw_reply = f"I was created by {ident['creator']} ({ident['organization']}) as part of {ident['platform']}."
            elif "mission" in low:
                raw_reply = f"My mission is to help you as {ident['platform']} — fast, accurate, and user-controlled."
            elif "brian lee baros" in low:
                raw_reply = f"{ident['creator']} is the creator and lead engineer of the {ident['platform']} project."
            else:
                raw_reply = f"I'm {ident['name']} — your {ident['platform']} companion."
            bundle = _sm_make_outward_bundle(
                _sm_present_text(raw_reply, intent="identity"),
                meta={"source": "identity_guard", "engine": "identity_guard", "intent": "identity", "version": ident["version"]},
            )
            bundle["identity"] = ident
            return jsonify(bundle), 200

        browser_state_bundle = _browser_state_answer_for_text(text)
        if browser_state_bundle is not None:
            return jsonify(browser_state_bundle), 200

        gov = {"allow": True}

        try:
            if _sm_module_approved("SarahMemoryCognitiveServices", capability="governor"):
                from SarahMemoryCognitiveServices import govern_request  # type: ignore
                gov = govern_request(
                    text,
                    caller="api_chat",
                    caller_context=context_packet,
                    user_present=True,
                    user_consented=bool(context_packet["meta"].get("user_consented")),
                    proposed_action=context_packet["meta"].get("proposed_action"),
                )
        except Exception as e:
            app_logger.warning(f"CognitiveServices govern_request failed; continuing with safe defaults: {e}", exc_info=True)
            gov = None

        if not isinstance(gov, dict):
            gov = {
                "ok": False,
                "decision": "ALLOW",
                "allow": True,
                "require_user": False,
                "reasons": ["governor_unavailable"],
                "rationale": "Governor unavailable; proceeding with safe defaults.",
                "routing_policy": {
                    "allowed_tiers": {"tier0": True, "tier1": True, "tier2": True, "tier3": (not local_only)},
                    "budgets": {"latency_ms": 4000, "max_steps": 12, "max_retries": 1},
                    "side_effects": {"tts": True, "db_write": True, "compare": True},
                },
                "trace": {},
            }

        gov_decision = str(gov.get("decision") or ("ALLOW" if bool(gov.get("allow")) else "DEFER")).upper()
        gov_allow = bool(gov.get("allow")) or (gov_decision == "ALLOW")
        gov_require_user = bool(gov.get("require_user")) or (gov_decision == "REQUIRE_USER")
        gov_rationale = str(gov.get("rationale") or "") if isinstance(gov.get("rationale"), str) else ""
        gov_reasons = gov.get("reasons") if isinstance(gov.get("reasons"), list) else []
        gov_trace = gov.get("trace") if isinstance(gov.get("trace"), dict) else {}
        routing_policy = gov.get("routing_policy") if isinstance(gov.get("routing_policy"), dict) else None

        if (not gov_allow) or gov_require_user or gov_decision in ("DENY", "DEFER", "REQUIRE_USER"):
            if gov_decision == "DENY":
                raw_reply = gov_rationale or "Request denied by policy."
                src = "governor:deny"
            elif gov_decision == "REQUIRE_USER" or gov_require_user:
                raw_reply = gov_rationale or "User confirmation required before proceeding."
                src = "governor:require_user"
            else:
                raw_reply = gov_rationale or "Request deferred. Provide more details or confirm intent."
                src = "governor:defer"
            bundle = _sm_make_outward_bundle(
                _sm_present_text(raw_reply, intent="system"),
                meta={
                    "source": src,
                    "engine": "cognitive_governor",
                    "decision": gov_decision,
                    "reasons": gov_reasons,
                    "trace": gov_trace if developersmode else {},
                    "local_only": local_only,
                    "version": PROJECT_VERSION,
                },
            )
            return jsonify(bundle), 200

        op_bundle = _sm_try_operatorcore_request(
            text,
            payload=payload,
            context_packet=context_packet,
            ingress_route=ingress_route,
            local_only=local_only,
            safe_mode=safe_mode,
            gov_decision=gov_decision,
            gov_reasons=gov_reasons,
            gov_require_user=gov_require_user,
            developersmode=developersmode,
        )
        if isinstance(op_bundle, dict):
            return jsonify(op_bundle), 200

        try:
            if _sm_module_approved("SarahMemoryNeuron", capability="router"):
                from SarahMemoryNeuron import neuron_route  # type: ignore
                nres = neuron_route(text, meta={
                    "intent": intent,
                    "tone": tone,
                    "complexity": complexity,
                    "avatar_request": avatar_request,
                    "ui": context_packet.get("ui"),
                    "local_only": local_only,
                    "offline": local_only,
                    "session_id": context_packet.get("session_id"),
                    "frame": context_packet.get("meta", {}).get("frame"),
                    "latest_frame": context_packet.get("meta", {}).get("latest_frame"),
                    "images": context_packet.get("meta", {}).get("images", []),
                    "vision_frame": context_packet.get("meta", {}).get("vision_frame"),
                    "context_packet": context_packet,
                    "mode_flags": context_packet.get("meta", {}).get("mode_flags", {}),
                    "governor": {"decision": gov_decision, "reasons": gov_reasons},
                    "ingress_route": ingress_route,
                }, policy=routing_policy)

                nres_dict = nres.to_dict() if hasattr(nres, "to_dict") else {
                    "ok": getattr(nres, "ok", True),
                    "reply": getattr(nres, "reply", ""),
                    "confidence": getattr(nres, "confidence", None),
                    "intent": getattr(nres, "intent", intent),
                    "source": getattr(nres, "source", "neuron"),
                    "artifacts": getattr(nres, "artifacts", {}) or {},
                    "trace": getattr(nres, "trace", {}) or {},
                }

                raw_reply = str(nres_dict.get("reply") or "")
                resolved_intent = str(nres_dict.get("intent") or intent or "chat")
                source_label = str(nres_dict.get("source") or "neuron")
                meta_out = {
                    "source": source_label,
                    "engine": "neuron_route",
                    "intent": resolved_intent,
                    "confidence": nres_dict.get("confidence"),
                    "governor": {"decision": gov_decision, "reasons": gov_reasons} if developersmode else {"decision": gov_decision},
                    "local_only": local_only,
                    "version": PROJECT_VERSION,
                    "session_id": context_packet.get("session_id"),
                    "vision_frame_attached": bool(frame_rec),
                    "neuron_trace": nres_dict.get("trace") or {},
                }
                artifacts = []
                actions = []
                try:
                    import SarahMemoryReply as _SMReply  # type: ignore
                    art_fn = _safe_getattr(_SMReply, "_sm_creative_artifacts_from_meta")
                    if callable(art_fn):
                        artifacts = art_fn({"source": source_label, "artifacts": nres_dict.get("artifacts") or {}, "neuron_trace": nres_dict.get("trace") or {}}) or []
                except Exception:
                    artifacts = []
                if not artifacts and isinstance(nres_dict.get("artifacts"), dict):
                    for key, value in (nres_dict.get("artifacts") or {}).items():
                        if value in (None, "", [], {}):
                            continue
                        path = value if isinstance(value, str) else json.dumps(value)
                        artifacts.append({"name": key, "type": "file", "path": path, "display_ready": True, "download_ready": True, "source": source_label})
                if isinstance(nres_dict.get("actions"), list):
                    actions = list(nres_dict.get("actions") or [])
                presentation_text = _sm_present_text(raw_reply, intent=resolved_intent, meta=meta_out)
                bundle = _sm_make_outward_bundle(
                    presentation_text,
                    meta=meta_out,
                    artifacts=artifacts,
                    actions=actions,
                    raw_answer=raw_reply,
                )
                bundle["ok"] = bool(nres_dict.get("ok", True))
                return jsonify(bundle), 200
        except Exception as e:
            app_logger.error(f"Neuron route failed: {e}", exc_info=True)

        try:
            import SarahMemoryReply as _SMReply  # type: ignore
            generate_reply = _safe_getattr(_SMReply, "generate_reply")
            if callable(generate_reply):
                rb = generate_reply(None, text)
            else:
                rb = None
        except Exception as e:
            rb = None
            app_logger.error(f"SarahMemoryReply.generate_reply failed: {e}", exc_info=True)

        if isinstance(rb, dict):
            raw_reply = str(rb.get("presentation_reply") or rb.get("response") or rb.get("reply") or rb.get("text") or "").strip()
            meta_out = rb.get("meta") if isinstance(rb.get("meta"), dict) else {}
            artifacts = rb.get("artifacts") if isinstance(rb.get("artifacts"), list) else []
            actions = rb.get("actions") if isinstance(rb.get("actions"), list) else []
            errors = rb.get("errors") if isinstance(rb.get("errors"), list) else []
        else:
            raw_reply = str(rb or "").strip()
            meta_out = {}
            artifacts = []
            actions = []
            errors = []

        if not raw_reply:
            raw_reply = "I’m having trouble generating a response right now."

        meta_out = {
            **(meta_out or {}),
            "source": str((meta_out or {}).get("source") or "sarahmemory_reply"),
            "engine": str((meta_out or {}).get("engine") or "generate_reply"),
            "intent": str((meta_out or {}).get("intent") or intent or "chat"),
            "governor": {"decision": gov_decision},
            "local_only": local_only,
            "version": PROJECT_VERSION,
        }
        presentation_text = _sm_present_text(raw_reply, intent=str(meta_out.get("intent") or intent or "chat"), meta=meta_out)
        bundle = _sm_make_outward_bundle(
            presentation_text,
            meta=meta_out,
            artifacts=artifacts,
            actions=actions,
            errors=errors,
            raw_answer=raw_reply,
        )
        return jsonify(bundle), 200

    except Exception as e:
        app_logger.error(f"Fatal /api/chat error: {e}", exc_info=True)
        meta = {"source": "api", "engine": "api_chat_exception", "version": PROJECT_VERSION}
        bundle = _sm_make_outward_bundle(
            "I’m having trouble processing that request right now.",
            meta=meta,
            errors=[str(e)],
        )
        bundle["ok"] = False
        bundle["error"] = str(e)
        return jsonify(bundle), 500


@app.route("/api/media/job", methods=["POST"])
def api_media_job_submit():
    """Submit a media generation job. Engine execution is handled by mods/add-ons."""
    try:
        payload = request.get_json(silent=True) or {}
        job = payload.get("job") or payload  # allow direct job dict
        import SarahMemoryAiFunctions as F
        job_id = F.submit_media_job(job)
        return jsonify({"ok": True, "job_id": job_id}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_submit failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/media/job/poll", methods=["POST"])
def api_media_job_poll():
    """Poll the next queued media job (for worker/add-on processes)."""
    try:
        import SarahMemoryAiFunctions as F
        job = F.poll_media_job()
        return jsonify({"ok": True, "job": job}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_poll failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/media/result/<job_id>", methods=["GET"])
def api_media_job_result(job_id):
    """Get status/result for a media job."""
    try:
        import SarahMemoryAiFunctions as F
        rec = F.get_media_result(job_id)
        return jsonify({"ok": True, "data": rec}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_result failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 404

@app.route("/api/media/result/<job_id>/store", methods=["POST"])
def api_media_job_store(job_id):
    """Store a media result (for worker/add-on processes)."""
    try:
        payload = request.get_json(silent=True) or {}
        result = payload.get("result") or {}
        status = payload.get("status") or "done"
        import SarahMemoryAiFunctions as F
        F.store_media_result(job_id, result, status=status)
        # Best-effort: if AvatarPanelAPI is active, try to display it
        try:
            from SarahMemoryAvatarPanel import AvatarPanelAPI
            api = AvatarPanelAPI()
            api.display_media_result(result)
        except Exception:
            pass
        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_store failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/request-knowledge", methods=['POST'])
def api_request_knowledge():
    data = request.get_json(silent=True) or {}
    requester = (data.get("requester") or data.get("from") or "").strip()
    topic = (data.get("topic") or data.get("notes") or "").strip()
    amount = data.get("amount") or data.get("reward") or "0" # Keep as string for Decimal conversion

    # Validate inputs
    if not requester:
        return jsonify({"error": "Requester ID is required."}), 400
    if not topic:
        return jsonify({"error": "Knowledge topic is required."}), 400

    try:
        amount_decimal = Decimal(str(amount)) # Ensure convertible to Decimal
        if amount_decimal < 0:
            return jsonify({"error": "Reward amount cannot be negative."}), 400
    except Exception:
        return jsonify({"error": "Invalid reward amount format."}), 400

    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO knowledge_requests(ts, requester, topic, reward, status) VALUES (?,?,?,?,?)",
                    (time.time(), requester, topic, str(amount_decimal), "open"))
        rid = cur.lastrowid
        con.commit()
        ensure_wallet_simple(requester) # Ensure wallet for requester
        return jsonify({"request_id": rid, "status": "open"}), 201
    except sqlite3.Error as e:
        app_logger.error(f"Failed to record knowledge request to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Failed to record knowledge request due to database error."}), 500
    finally:
        if con: con.close()


@app.route("/api/wallet/<node>")
def api_wallet_view(node):
    con = None
    try:
        p = ensure_wallet_simple(node)
        con = _connect_sqlite(p)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT balance, reputation, last_rep_ts, rep_daily FROM wallet WHERE id=1")
        r = cur.fetchone()
        if not r:
            return jsonify({"error": f"Wallet data not found for node: {node}"}), 404

        cur.execute("SELECT ts,delta,memo FROM txs ORDER BY id DESC LIMIT 50")
        txs = [dict(row) for row in cur.fetchall()] if hasattr(cur, "fetchall") else []

        return jsonify({
            "node": node,
            "balance": r["balance"],
            "reputation": float(r["reputation"] or 0.0),
            "last_rep_ts": float(r["last_rep_ts"] or 0.0),
            "rep_daily": float(r["rep_daily"] or 0.0),
            "txs": txs
        })
    except sqlite3.Error as e:
        app_logger.error(f"SQLite error fetching wallet details for node {node}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching wallet details"}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching wallet for node {node}.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

@app.post("/api/hub/ping")
def hub_ping():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify({"ok": True, "now": time.time(), "echo": payload})
    except Exception as e:
        app_logger.error(f"Error processing hub_ping request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


@app.post("/api/hub/job")
def hub_job():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        jid = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest() # Specify encoding

        # optional light persistence for debugging
        jobs_dir = os.path.join(DATA_DIR, "jobs")
        _ensure_dir(jobs_dir)
        try:
            with open(os.path.join(jobs_dir, f"job-{int(time.time())}-{jid}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            app_logger.warning(f"Failed to persist hub job to disk: {e}")

        # TODO: Integrate with SarahMemoryNetwork (net_mod) for proper job handling
        if net_mod and _safe_getattr(net_mod, "process_hub_job"):
             try:
                 net_mod.process_hub_job(jid, payload)
                 app_logger.info(f"Hub job {jid} processed by SarahMemoryNetwork.")
             except Exception as e:
                 app_logger.error(f"Error in SarahMemoryNetwork processing hub job {jid}: {e}", exc_info=True)
                 # Don't fail the hub_job API, just log the internal processing error

        return jsonify({"ok": True, "job_id": jid}), 200
    except Exception as e:
        app_logger.error(f"Error processing hub_job request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


@app.post("/api/hub/reply")
def hub_reply():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        # optional light persistence for debugging
        receipts_dir = os.path.join(DATA_DIR, "receipts")
        _ensure_dir(receipts_dir)
        try:
            reply_id = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            with open(os.path.join(receipts_dir, f"reply-{int(time.time())}-{reply_id}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            app_logger.warning(f"Failed to persist hub reply receipt to disk: {e}")

        # TODO: Integrate with SarahMemoryNetwork (net_mod) for proper reply handling
        if net_mod and _safe_getattr(net_mod, "process_hub_reply"):
             try:
                 net_mod.process_hub_reply(payload)
                 app_logger.info("Hub reply processed by SarahMemoryNetwork.")
             except Exception as e:
                 app_logger.error(f"Error in SarahMemoryNetwork processing hub reply: {e}", exc_info=True)

        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.error(f"Error processing hub_reply request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# API Key guard + Node/Embedding/Context/Jobs endpoints
# ---------------------------------------------------------------------------
SARAH_API_KEY = os.environ.get("SARAH_API_KEY", "") # Keep variable name consistent

def _api_key_auth_ok() -> bool:
    """
    Optional lightweight auth for admin-ish endpoints.
    Accepts either:
      - X-API-Key: <key>
      - Authorization: Bearer <key>
    """
    # allow local / dev with no auth if explicitly configured
    try:
        if config is not None and getattr(config, "ALLOW_NOAUTH_LOCAL", False):
            return True
    except Exception:
        pass

    api_key = (os.environ.get("SARAHMEMORY_API_KEY") or os.environ.get("API_KEY") or "").strip()
    if not api_key:
        # Backward compatible: no key configured => open
        return True

    hdr = (request.headers.get("X-API-Key") or "").strip()
    if hdr and hmac.compare_digest(hdr, api_key):
        return True

    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token and hmac.compare_digest(token, api_key):
            return True

    return False

@app.post("/api/register-node")
def api_register_node():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    # Ensure meta is a JSON string, assume simple dump if already dict
    meta = json.dumps(data.get("meta") or {})
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO nodes(node_id,last_ts,meta) VALUES(?,?,?) "
                    "ON CONFLICT(node_id) DO UPDATE SET last_ts=excluded.last_ts, meta=excluded.meta",
                    (node_id, time.time(), meta))
        con.commit()
        ensure_wallet_simple(node_id)
        _cache_invalidate('leaderboard')
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to register node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error during node registration."}), 500
    finally:
        if con: con.close()


@app.route("/api/receive-embedding", methods=['POST'])
def api_receive_embedding():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    embedding_data = data.get("embedding")
    context_id = data.get("context_id")

    if not embedding_data:
        return jsonify({"error": "Missing 'embedding' data."}), 400
    if not context_id:
        return jsonify({"error": "Missing 'context_id'."}), 400

    vector = json.dumps(embedding_data)
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO embeddings(ts,node_id,context_id,vector) VALUES(?,?,?,?)",
                    (time.time(), node_id, context_id, vector))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to receive embedding for node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error receiving embedding."}), 500
    finally:
        if con: con.close()

@app.route("/api/context-update", methods=['POST'])
def api_context_update():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    text = data.get("text")
    tags_data = data.get("tags")

    if not text:
        return jsonify({"error": "Missing 'text' for context update."}), 400

    tags = json.dumps(tags_data if isinstance(tags_data, list) else [])
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO contexts(ts,node_id,text,tags) VALUES(?,?,?,?)",
                    (time.time(), node_id, text, tags))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to update context for node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error during context update."}), 500
    finally:
        if con: con.close()

@app.route("/api/jobs", methods=['POST'])
def api_jobs_post():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    job_id = (data.get("job_id") or "").strip() or "unknown_job"
    result_data = data.get("result")

    if not result_data:
        return jsonify({"error": "Missing 'result' data for job."}), 400

    result = json.dumps(result_data)
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO job_results(ts,node_id,job_id,result) VALUES(?,?,?,?)",
                    (time.time(), node_id, job_id, result))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to post job results for node {node_id} and job {job_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error posting job results."}), 500
    finally:
        if con: con.close()

# ---------------------------------------------------------------------------
# WebUI helper endpoints (Themes/Voices/Settings/Contacts/Reminders/Cleanup)
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(resp):
    """Attach basic security headers (safe defaults for WebUI + API)."""
    try:
        # Version / identity
        resp.headers["X-SarahMemory-Version"] = str(PROJECT_VERSION)

        # Standard hardening headers
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # NOTE: CSP can be strict; keep it permissive enough for current WebUI.
        # Tighten later once all asset/CDN usage is finalized.
        if "Content-Security-Policy" not in resp.headers:
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
                "connect-src 'self' https://api.sarahmemory.com https://ai.sarahmemory.com; "
                "img-src 'self' data: blob: https:; "
                "media-src 'self' data: blob: https:; "
                "style-src 'self' 'unsafe-inline' https:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:;"
            )
    except Exception as e:
        try:
            app_logger.error(f"Failed to add security headers: {e}")
        except Exception:
            pass

    # Optional FE speech script injection (gated)
    if os.getenv("SARAH_FE_SPEECH", "0") == "1":
        try:
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ct:
                data = resp.get_data(as_text=True)
                if data and "<html" in data.lower() and 'id="sm-fe-speech"' not in data:
                    tag = "\n<script id=\"sm-fe-speech\" src=\"/api/fe/v800/speech.js\" defer></script>\n"
                    lower = data.lower()
                    i = lower.rfind("</head>")
                    if i != -1:
                        resp.set_data(data[:i] + tag + data[i:])
                        resp.headers.pop("Content-Length", None)
        except Exception as e:
            try:
                app_logger.warning(f"Speech script injection failed: {e}")
            except Exception:
                pass

    return resp

# Centralized settings file path (robust for headless/WSGI environments)
# NOTE: Avoid KeyError at import-time if _globals_paths() returns a partial dict during early init.
try:
    _gp = _globals_paths() or {}
    _settings_dir = _gp.get("SETTINGS_DIR") or os.path.join(_gp.get("DATA_DIR", os.path.join(os.getcwd(), "data")), "settings")
    try:
        os.makedirs(_settings_dir, exist_ok=True)
    except Exception:
        pass
    SETTINGS_FILE = os.path.join(_settings_dir, "settings.json")  # SETTINGS_DIR/settings.json
except Exception:
    SETTINGS_FILE = os.path.join(os.getcwd(), "settings.json")

@app.route("/get_user_setting")
def get_user_setting():
    key = request.args.get("key", "").strip()
    if not key:
        return jsonify({"error": "Setting key is required."}), 400

    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            app_logger.error(f"Error reading settings file {SETTINGS_FILE}: {e}")
            data = {} # On error, treat as empty settings

    return jsonify({"value": data.get(key, "")})

@app.route("/set_user_setting", methods=['POST'])
def set_user_setting():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key")
    val = payload.get("value")

    if key is None:
        return jsonify({"status": "error", "error": "Setting key is required."}), 400

    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            app_logger.error(f"Error reading settings file {SETTINGS_FILE} for update: {e}")
            data = {} # If file is corrupted, start fresh with new setting

    data[key] = val
    _ensure_dir(os.path.dirname(SETTINGS_FILE)) # Ensure settings directory exists
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status":"ok"})
    except IOError as e:
        app_logger.error(f"Error writing settings file {SETTINGS_FILE}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": f"Failed to save setting: {e}"}), 500


# ---------------------------------------------------------------------------
# SarahMemory Model Manager API
# ---------------------------------------------------------------------------
# Frontend is a control surface only. SarahMemoryLLM.py owns discovery,
# validation, classification, active model state, and downloads.

def _sm_llm_manager():
    try:
        import SarahMemoryLLM as _SMLLM  # type: ignore
        return _SMLLM
    except Exception as exc:
        app_logger.error("SarahMemoryLLM import failed for model manager API: %s", exc, exc_info=True)
        return None


def _model_payload() -> dict:
    try:
        data = request.get_json(silent=True) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@app.route("/api/models/status", methods=["GET"])
def api_models_status():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        refresh = str(request.args.get("refresh", "1")).strip().lower() not in ("0", "false", "no", "off")
        fn = getattr(mod, "get_model_manager_status", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_manager_status_unavailable"}), 501
        return jsonify(fn(refresh=refresh)), 200
    except Exception as exc:
        app_logger.exception("Model status failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/scan", methods=["POST"])
def api_models_scan():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        fn = getattr(mod, "scan_model_registry", None)
        status_fn = getattr(mod, "get_model_manager_status", None)
        if callable(fn):
            fn(persist=True)
        if callable(status_fn):
            return jsonify(status_fn(refresh=False)), 200
        return jsonify({"ok": True}), 200
    except Exception as exc:
        app_logger.exception("Model scan failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/select", methods=["POST"])
def api_models_select():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "set_active_model", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_select_unavailable"}), 501
        result = fn(
            str(data.get("category") or ""),
            model_id=str(data.get("model_id") or data.get("id") or ""),
            repo=str(data.get("repo") or ""),
        )
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Model select failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/classify", methods=["POST"])
def api_models_classify():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "classify_model", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_classify_unavailable"}), 501
        result = fn(
            model_id=str(data.get("model_id") or data.get("id") or ""),
            category=str(data.get("category") or "unknown"),
            domain=str(data.get("domain") or "general"),
            adapter_type=str(data.get("adapter_type") or ""),
            display_name=str(data.get("display_name") or ""),
        )
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Model classify failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/verify", methods=["POST"])
def api_models_verify():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "verify_model_by_id", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_verify_unavailable"}), 501
        result = fn(str(data.get("model_id") or data.get("id") or ""))
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Model verify failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/external-path", methods=["POST"])
def api_models_external_path():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "add_external_model_path", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "external_path_unavailable"}), 501
        result = fn(str(data.get("path") or data.get("folder") or ""))
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("External model path add failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/reset", methods=["POST"])
def api_models_reset():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "reset_active_model_to_recommended", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_reset_unavailable"}), 501
        result = fn(str(data.get("category") or "reasoning"))
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Model reset failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/download", methods=["POST"])
def api_models_download():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "download_model_to_registry", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "model_download_unavailable"}), 501
        result = fn(
            category=str(data.get("category") or "reasoning"),
            repo=str(data.get("repo") or ""),
            model_id=str(data.get("model_id") or data.get("id") or ""),
        )
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Model download failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


# Themes routes are fine, pathing should be robust now.

@app.route("/get_available_voices")
def get_available_voices():
    """Return available TTS voices for the WebUI.
    Prefer the richer SarahMemoryVoice bridge (v8.0) so we see both
    system voices and any registered custom voices (.pt models).
    Fallback to a direct pyttsx3 probe if that fails.
    """
    # First try the unified SarahMemoryVoice API
    sm_list_voices = None
    try:
        from SarahMemoryVoice import list_voices as sm_list_voices_func
        sm_list_voices = sm_list_voices_func
    except ImportError:
        app_logger.info("SarahMemoryVoice module not found for listing voices.")
    except Exception as e:
        app_logger.error(f"Error importing SarahMemoryVoice.list_voices: {e}", exc_info=True)

    if sm_list_voices:
        try:
            voices = sm_list_voices() or []
            if voices:
                return jsonify(voices)
        except Exception as e:
            app_logger.warning(f" get_available_voices via SarahMemoryVoice failed: {e}", exc_info=True)

    # Fallback: query local OS voices directly via pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        out = []
        for v in voices:
            name_val = getattr(v, "name", "") or getattr(v, "id", "")
            out.append({
                "id": getattr(v, "id", ""),
                "name": name_val
            })
        return jsonify(out)
    except ImportError:
        app_logger.info("pyttsx3 not installed. Cannot get local OS voices.")
    except Exception as e:
        app_logger.error(f"Error getting voices via pyttsx3 fallback: {e}", exc_info=True)

    return jsonify([]) # Return empty list if all methods fail


# Helper function for cleanup routes to reduce repetition
def _call_cleanup_module_func(func_name: str, *args, **kwargs):
    """Helper to call functions from SarahMemoryCleanup and handle responses."""
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, func_name)
        if callable(fn):
            result = fn(*args, **kwargs)
            return jsonify({"status": "ok", "result": str(result)}), 200
        app_logger.warning(f"SarahMemoryCleanup function '{func_name}' not found or not callable.")
        return jsonify({"status": "noop", "error": f"Cleanup function '{func_name}' not found."}), 404
    except ImportError:
        app_logger.error("SarahMemoryCleanup module not found.")
        return jsonify({"status": "error", "error": "SarahMemoryCleanup module not available."}), 503
    except Exception as e:
        app_logger.exception(f"Error in SarahMemoryCleanup function '{func_name}'.")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/cleanup/backup_all")
def cleanup_backup_all():
    return _call_cleanup_module_func("backup_all")

@app.route("/cleanup/restore_latest")
def cleanup_restore_latest():
    return _call_cleanup_module_func("restore_latest")

@app.route("/cleanup/clear_range", methods=['POST'])
def cleanup_clear_range():
    payload = request.get_json(silent=True) or {}
    db_name = payload.get("db", "context_history.db")
    seconds = int(payload.get("seconds", 0) or 0)
    return _call_cleanup_module_func("clear_range", db_name, seconds if seconds > 0 else None)

@app.route("/cleanup/tidy_logs")
def cleanup_tidy_logs():
    return _call_cleanup_module_func("tidy_logs")


# Camera/Mic/Voice toggles
@app.route("/toggle_camera")
def toggle_camera():
    state = request.args.get("state","").lower() in ("true","1","yes","on")
    app.config["CAMERA_ON"] = state # Use app.config for global state
    return jsonify({"status":"ok","camera": state})

@app.route("/toggle_microphone", methods=["POST"])
def toggle_microphone():
    """
    Enable/disable microphone capture for the UI.
    Accepts JSON: { "enabled": true/false }
    """
    try:
        data = request.get_json(silent=True) or {}
        desired = bool(data.get("enabled", True))

        global MIC_ON, MIC_ENABLED
        MIC_ON = desired
        MIC_ON = desired
        MIC_ENABLED = MIC_ON

        try:
            save_state("MIC_ON", bool(desired))
        except Exception:
            pass

        return jsonify({"ok": True, "mic_enabled": bool(desired)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/toggle_voice_output", methods=["POST"])
def toggle_voice_output():
    """
    Enable/disable voice output for the UI.
    Accepts JSON: { "enabled": true/false }
    """
    try:
        data = request.get_json(silent=True) or {}
        desired = bool(data.get("enabled", True))

        global VOICE_OUTPUT_ON, VOICE_OUTPUT_ENABLED
        VOICE_OUTPUT_ON = desired
        TTS_ON = desired
        TTS_ENABLED = TTS_ON
        VOICE_OUTPUT_ON = TTS_ON
        VOICE_OUTPUT_ENABLED = TTS_ON

        try:
            save_state("TTS_ON", bool(desired))
        except Exception:
            pass

        return jsonify({"ok": True, "voice_output_enabled": bool(desired)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/check_call_active")
def check_call_active():
    return jsonify({"active": app.config.get("CALL_ACTIVE", False)}) # Use app.config

@app.route("/initiate_call", methods=['POST'])
def initiate_call():
    data = request.get_json(silent=True) or {}
    number = (data.get("number") or "").strip()
    app.config["CALL_ACTIVE"] = bool(number)  # Use app.config
    return jsonify({"status":"call_started","to":number})

# File transfer / ingest
@app.route("/send_file_to_remote", methods=['POST'])
def send_file_to_remote():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename")
    b64 = payload.get("data")

    if not fname or not b64:
        return jsonify({"status": "error", "error": "Missing filename or data."}), 400

    try:
        data = base64.b64decode(b64.encode("utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid base64 data: {e}"}), 400

    if os.name == "nt":
        out_dir = os.path.join(os.environ.get("USERPROFILE"), "Downloads") if "USERPROFILE" in os.environ else r"C:\Users\Public\Downloads"
    else:
        out_dir = os.path.join(DATA_DIR, "downloads") # Use DATA_DIR for cross-platform and server-safe

    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(fname)) # Use basename to prevent path traversal
    try:
        with open(out_path, "wb") as f:
            f.write(data)
        return jsonify({"message": f"Sent file to remote user (saved locally): {fname}", "path": out_path}), 200
    except IOError as e:
        app_logger.error(f"Failed to save remote file {out_path}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": f"Failed to save file locally: {e}"}), 500


@app.route("/ingest_local_file", methods=['POST'])
def ingest_local_file():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename")
    b64 = payload.get("data")

    if not fname or not b64:
        return jsonify({"status": "error", "error": "Missing filename or data."}), 400

    _paths = _globals_paths()
    DATASETS_DIR = _paths["DATASETS_DIR"]
    DOCUMENTS_DIR = _paths["DOCUMENTS_DIR"]
    try:
        data = base64.b64decode(b64.encode("utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid base64 data: {e}"}), 400

    out_dir = DOCUMENTS_DIR or DATASETS_DIR # Default to DOCUMENTS_DIR if available, else DATASETS_DIR
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(fname)) # Use basename to prevent path traversal
    try:
        with open(out_path, "wb") as f:
            f.write(data)
        return jsonify({"message": f"Stored file in local documents: {fname}", "path": out_path}), 200
    except IOError as e:
        app_logger.error(f"Failed to ingest local file {out_path}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": f"Failed to store file locally: {e}"}), 500


# Contacts
USER_DB_PATH = os.path.join(_globals_dir("DATA_DIR", "data"), "user_data.db")

def _init_contacts_db(db_path):
    """Helper to initialize contacts table."""
    con = None
    try:
        con = _connect_sqlite(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, number TEXT)")
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to initialize contacts database at {db_path}: {e}")
        raise # Re-raise to ensure caller knows about failure
    finally:
        if con: con.close()


@app.route("/get_all_contacts")
def get_all_contacts():
    con = None
    try:
        _init_contacts_db(USER_DB_PATH) # Ensure table exists
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT id, name, number FROM contacts ORDER BY name COLLATE NOCASE")
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"contacts": rows})
    except Exception as e:
        app_logger.exception(f"Error fetching contacts from {USER_DB_PATH}.")
        return jsonify({"error": "Failed to retrieve contacts."}), 500
    finally:
        if con: con.close()

@app.route("/add_contact", methods=['POST'])
def add_contact():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    number = (data.get("number") or "").strip()

    if not name or not number:
        return jsonify({"status":"error", "error":"Name and number are required to add contact."}), 400

    con = None
    try:
        _init_contacts_db(USER_DB_PATH) # Ensure table exists
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        cur.execute("INSERT INTO contacts(name,number) VALUES(?,?)",(name,number))
        con.commit()
        return jsonify({"status":"ok"}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to add contact {name} to {USER_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error adding contact."}), 500
    finally:
        if con: con.close()

@app.route("/delete_contact", methods=['POST'])
def delete_contact():
    data = request.get_json(silent=True) or {}
    rid = data.get("id")
    if not isinstance(rid, int):
        return jsonify({"status": "error", "error": "Invalid contact ID provided."}), 400

    con = None
    try:
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM contacts WHERE id=?", (rid,))
        if cur.rowcount == 0:
            return jsonify({"status": "error", "error": f"Contact with ID {rid} not found."}), 404
        con.commit()
        return jsonify({"status":"deleted", "id": rid}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to delete contact with ID {rid} from {USER_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error deleting contact."}), 500
    finally:
        if con: con.close()

# Reminders
REMINDERS_DB_PATH = os.path.join(_globals_dir("DATA_DIR", "data"), "reminders.db")

def _init_reminders_db(db_path):
    """Helper to initialize reminders table."""
    con = None
    try:
        con = _connect_sqlite(db_path)
        cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, time TEXT, note TEXT)')
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to initialize reminders database at {db_path}: {e}")
        raise # Re-raise to ensure caller knows about failure
    finally:
        if con: con.close()

@app.route("/get_reminders")
def get_reminders():
    con = None
    try:
        _init_reminders_db(REMINDERS_DB_PATH) # Ensure table exists
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute('SELECT id, title, time, note FROM reminders ORDER BY time ASC')
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({'reminders': rows})
    except Exception as e:
        app_logger.exception(f"Error fetching reminders from {REMINDERS_DB_PATH}.")
        return jsonify({"error": "Failed to retrieve reminders."}), 500
    finally:
        if con: con.close()

@app.route("/save_reminder", methods=['POST'])
def save_reminder():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    time_s = (payload.get("time") or "").strip()
    note = payload.get("note") or ""

    if not title or not time_s:
        return jsonify({"status":"error", "error":"Title and time are required to save reminder."}), 400

    con = None
    try:
        _init_reminders_db(REMINDERS_DB_PATH) # Ensure table exists
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        cur.execute('INSERT INTO reminders(title, time, note) VALUES(?,?,?)',(title, time_s, note))
        con.commit()
        rid = cur.lastrowid
        return jsonify({"status":"ok","id":rid}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to save reminder '{title}' to {REMINDERS_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error saving reminder."}), 500
    finally:
        if con: con.close()

@app.route("/delete_reminder", methods=['POST'])
def delete_reminder():
    payload = request.get_json(silent=True) or {}
    rid = payload.get("id")

    if not isinstance(rid, int):
        return jsonify({"status": "error", "error": "Invalid reminder ID provided."}), 400

    con = None
    try:
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        cur.execute('DELETE FROM reminders WHERE id=?', (rid,))
        if cur.rowcount == 0:
            return jsonify({"status": "error", "error": f"Reminder with ID {rid} not found."}), 404
        con.commit()
        return jsonify({"status":"deleted", "id": rid}), 200
    except sqlite3.Error as e:
        app_logger.exception(f"Failed to delete reminder with ID {rid} from {REMINDERS_DB_PATH}.")
        return jsonify({"status":"error", "error": "Database error deleting reminder."}), 500
    finally:
        if con: con.close()

@app.route("/run_automation_trigger", methods=['POST'])
def run_automation_trigger():
    payload = request.get_json(silent=True) or {}
    try:
        import SarahMemoryAiFunctions as F
        run_automation_func = _safe_getattr(F, "run_automation")
        if callable(run_automation_func):
            res = run_automation_func(payload)
            return jsonify({"status":"ok","result":str(res)}), 200
        app_logger.warning("SarahMemoryAiFunctions.run_automation not found or not callable.")
        return jsonify({"status":"noop", "message":"Automation function not available."}), 404
    except ImportError:
        app_logger.error("SarahMemoryAiFunctions module not found for automation trigger.")
        return jsonify({"status":"error", "error":"Automation module not available."}), 503
    except Exception as e:
        app_logger.exception("Error running automation trigger.")
        return jsonify({"status":"error", "error":str(e)}), 500

# Calendar + Chat history (for Web UI)
CHAT_HISTORY_DB_PATH = os.path.join(_globals_dir("DATA_DIR", "data"), "context_history.db")


# ---------------------------------------------------------------------------
# v8 WebUI Compatibility: Conversations API (HistoryScreen.tsx)
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
def api_conversations_list():
    """Return recent conversation threads.

    Response:
      { ok: true, conversations: [ {id,title,preview,timestamp,message_count} ] }
    """
    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Best-effort schema support: we aggregate by conversation id.
        cur.execute(
            """
            SELECT
              id,
              MAX(timestamp) AS timestamp,
              MAX(COALESCE(user_input, '')) AS preview,
              COUNT(1) AS message_count
            FROM conversations
            GROUP BY id
            ORDER BY MAX(timestamp) DESC
            LIMIT 250
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        convs = []
        for r in rows:
            cid = str(r.get('id'))
            convs.append({
                'id': cid,
                'title': f'Conversation {cid[:8]}' if cid else 'Conversation',
                'preview': r.get('preview') or '',
                'timestamp': r.get('timestamp') or '',
                'message_count': int(r.get('message_count') or 0),
            })
        return jsonify({'ok': True, 'conversations': convs}), 200
    except Exception as e:
        app_logger.error(f"/api/conversations failed: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Failed to fetch conversations'}), 500
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@app.get("/api/conversations/<convo_id>")
def api_conversation_get(convo_id):
    """Return one conversation as a message list.

    Response:
      { ok: true, id: <id>, messages: [ {role,content,meta?,timestamp?} ] }
    """
    if not convo_id:
        return jsonify({'ok': False, 'error': 'Conversation ID required'}), 400

    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Order by timestamp when present; otherwise stable rowid.
        try:
            cur.execute(
                """
                SELECT role, text, metadata AS meta, timestamp
                FROM conversations
                WHERE id = ?
                ORDER BY COALESCE(timestamp, '') ASC
                """,
                (convo_id,),
            )
        except Exception:
            cur.execute(
                """
                SELECT role, text, metadata AS meta, NULL AS timestamp
                FROM conversations
                WHERE id = ?
                """,
                (convo_id,),
            )

        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return jsonify({'ok': False, 'error': 'Not found'}), 404

        msgs = []
        for r in rows:
            role = (r.get('role') or '').strip().lower() or 'assistant'
            if role not in ('user', 'assistant', 'system'):
                # fall back if DB stores other values
                role = 'user' if role.startswith('u') else 'assistant'
            msgs.append({
                'role': role,
                'content': r.get('text') or '',
                'meta': r.get('meta') or None,
                'timestamp': r.get('timestamp') or None,
            })

        return jsonify({'ok': True, 'id': convo_id, 'messages': msgs}), 200
    except Exception as e:
        app_logger.error(f"/api/conversations/{convo_id} failed: {e}", exc_info=True)
        return jsonify({'ok': False, 'error': 'Failed to fetch conversation'}), 500
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@app.route("/get_chat_threads_by_date")
def get_chat_threads_by_date():
    date_filter = request.args.get("date", "").strip()  # YYYY-MM-DD
    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        cur = con.cursor()
        q = "SELECT id, timestamp, user_input AS preview FROM conversations"
        params = []
        if date_filter:
            q += " WHERE date(timestamp)=?"
            params.append(date_filter)
        q += " ORDER BY timestamp DESC" # Order by newest first
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(q, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"threads": rows})
    except sqlite3.Error as e:
        app_logger.error(f"Failed to fetch chat threads by date from {CHAT_HISTORY_DB_PATH}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching chat threads."}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching chat threads by date.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

@app.route("/get_conversation_by_id")
def get_conversation_by_id():
    convo_id = request.args.get("id")
    if not convo_id:
        return jsonify({"error": "Conversation ID is required."}), 400

    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        cur = con.cursor()
        # Assuming conversations table has role, text, and metadata
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT role, text, metadata AS meta FROM conversations WHERE id = ?", (convo_id,))
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return jsonify({"error": f"Conversation with ID {convo_id} not found."}), 404
        return jsonify(rows)
    except sqlite3.Error as e:
        app_logger.error(f"Failed to fetch conversation by ID {convo_id} from {CHAT_HISTORY_DB_PATH}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching conversation."}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching conversation by ID {convo_id}.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
@app.get("/get_theme_files") # Use app.get for GET requests
def get_theme_files():
    final_themes_dir = None
    try:
        import SarahMemoryGlobals as G
        # Prioritize checking THEMES_DIR from SarahMemoryGlobals
        if hasattr(G, "THEMES_DIR"):
            final_themes_dir = G.THEMES_DIR
    except Exception:
        pass # Fallback to local logic if SarahMemoryGlobals has issues

    if final_themes_dir is None: # If not found via Globals, use local logic
        # Re-evaluating path for local fallback to ensure accuracy
        base_dir_local = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir_local = os.path.join(base_dir_local, "data")
        themes_dirA_local = os.path.join(data_dir_local, "mods", "themes")
        themes_dirB_local = os.path.join(data_dir_local, "themes")

        if os.path.isdir(themes_dirA_local):
            final_themes_dir = themes_dirA_local
        elif os.path.isdir(themes_dirB_local):
            final_themes_dir = themes_dirB_local
        else:
            final_themes_dir = themes_dirA_local # Default to this even if it doesn't exist yet

    files = []
    if final_themes_dir and os.path.isdir(final_themes_dir):
        for dp, dn, fnames in os.walk(final_themes_dir):
            for f in fnames:
                # Optimized check for file extensions
                if f.lower().endswith((".css", ".json", ".yml", ".yaml", ".toml", ".png", ".jpg", ".jpeg", ".svg", ".ttf", ".otf")):
                    rel = os.path.relpath(os.path.join(dp, f), final_themes_dir).replace("\\", "/")
                    files.append(rel)
    else:
        app_logger.warning(f"Theme directory '{final_themes_dir}' not found or is not a directory.")

    # Determine active_root for jsonify
    # This logic still refers to the old A/B distinction for `active_root`
    # It might be more robust to derive `active_root` from `final_themes_dir` if it's dynamic
    data_dir_for_json_path = DATA_DIR # Use the global DATA_DIR
    themes_dirA_for_json_path = os.path.join(data_dir_for_json_path, "mods", "themes")
    themes_dirB_for_json_path = os.path.join(data_dir_for_json_path, "themes")

    if os.path.isdir(themes_dirB_for_json_path): # Prefer /data/themes if it contains actual themes
        active_root = "/api/data/themes"
    elif os.path.isdir(themes_dirA_for_json_path): # Then /data/mods/themes
        active_root = "/api/data/mods/themes"
    else: # Fallback
        active_root = "/api/data/mods/themes" # Defaulting to the mods path

    return jsonify({"root": active_root, "count": len(files), "files": sorted(files)})

@app.route("/api/data/themes/<path:filename>")
def serve_theme_file_A(filename):
    data_dir_for_serving = DATA_DIR # Use the determined global DATA_DIR
    root = os.path.join(data_dir_for_serving, "themes")
    # Basic path traversal protection
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400
    try:
        return send_from_directory(root, filename)
    except Exception as e:
        app_logger.error(f"Error serving theme file from {root}/{filename}: {e}")
        return jsonify({"error": "Theme file not found or accessible"}), 404


@app.route("/api/data/mods/themes/<path:filename>")
def serve_theme_file_B(filename):
    data_dir_for_serving = DATA_DIR # Use the determined global DATA_DIR
    root = os.path.join(data_dir_for_serving, "mods", "themes")
    # Basic path traversal protection
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400
    try:
        return send_from_directory(root, filename)
    except Exception as e:
        app_logger.error(f"Error serving theme file from {root}/{filename}: {e}")
        return jsonify({"error": "Theme file not found or accessible"}), 404


# --- Boot Launcher / Health (idempotent server-side autostart) ---
import subprocess

PID_FILE = os.path.join(DATA_DIR, "sarahmemory.pid") # Using global DATA_DIR

def _is_running():
    """Checks if SarahMemoryMain process is already running based on PID file."""
    try:
        if not os.path.exists(PID_FILE):
            return False
        with open(PID_FILE, "r") as f:
            pid_s = (f.read() or "").strip()
        if not pid_s:
            return False
        pid = int(pid_s)
        # Best-effort: os.kill(pid, 0) works on POSIX; on Windows, it might just raise an error
        # rather than allowing os.kill(pid, 0) to check existence. subprocess.os.name handles.
        if os.name == "posix": # Linux/macOS
            try:
                os.kill(pid, 0) # Check if process exists
                return True
            except OSError: # Process does not exist
                return False
        elif os.name == "nt": # Windows
            import ctypes
            # Check if PID is active on Windows
            kernel32 = ctypes.WinDLL('kernel32')
            handle = kernel32.OpenProcess(0x1000, False, pid) # PROCESS_QUERY_LIMITED_INFORMATION
            if handle is not None:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            app_logger.warning(f"Unknown OS type '{os.name}'. Cannot reliably check PID {pid}.")
            return False # Conservative default
    except (ValueError, IOError) as e:
        app_logger.debug(f"PID file read error or invalid PID: {e}")
        return False
    except Exception as e:
        app_logger.error(f"Unexpected error in _is_running: {e}", exc_info=True)
        return False

def _write_pid(pid: int):
    """Writes the current process PID to a file."""
    try:
        _ensure_dir(DATA_DIR) # Ensure DATA_DIR exists before writing PID
        with open(PID_FILE, "w") as f:
            f.write(str(pid))
    except (IOError, OSError) as e:
        app_logger.error(f"Failed to write PID file {PID_FILE}: {e}")
    except Exception as e:
        app_logger.error(f"Unexpected error writing PID file: {e}", exc_info=True)


def _start_sarah_main():
    """Spawn the canonical boot chain (SarahMemoryMain.py) in background."""
    try:
        if _is_running():
            app_logger.info("SarahMemoryMain is already running. Skipping new spawn.")
            return True
    except Exception:
        pass

    main_py_path = os.path.join(BASE_DIR, "SarahMemoryMain.py")
    if not os.path.exists(main_py_path):
        app_logger.error(f"SarahMemoryMain.py not found at {main_py_path}. Cannot start main process.")
        return False

    # Prefer the currently running interpreter, then common venv locations, then system python.
    candidates = [
        [sys.executable, main_py_path],
        [os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"), main_py_path],   # Windows venv
        [os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"), main_py_path], # Windows .venv
        [os.path.join(BASE_DIR, "venv", "bin", "python3"), main_py_path],         # Linux/mac venv
        [os.path.join(BASE_DIR, ".venv", "bin", "python3"), main_py_path],
        ["python", main_py_path],
        ["python3", main_py_path],
    ]

    # Filter invalid interpreter paths (except bare commands)
    final_candidates = []
    for cmd in candidates:
        try:
            exe = cmd[0]
            if os.path.isabs(exe) and not os.path.exists(exe):
                continue
            final_candidates.append(cmd)
        except Exception:
            continue

    # Try each candidate until one spawns successfully
    for cmd in final_candidates:
        try:
            app_logger.info(f"Attempting to start SarahMemoryMain: {cmd}")
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            try:
                _write_pid(proc.pid)
            except Exception:
                pass
            return True
        except Exception as e:
            app_logger.warning(f"Failed to start SarahMemoryMain with {cmd}: {e}")

    return False

@app.post("/api/launch")
def api_launch():
    try:
        if _is_running():
            return jsonify({"ok": True, "running": True, "msg": "SarahMemoryMain is already running."}), 200
        ok = _start_sarah_main()
        return jsonify({"ok": bool(ok), "running": bool(ok), "msg": "SarahMemoryMain launched successfully." if ok else "Failed to launch SarahMemoryMain."}), (200 if ok else 500)
    except Exception as e:
        app_logger.exception("Error during launch API call.")
        return jsonify({"ok": False, "error": str(e), "msg": "Internal server error during launch."}), 500


# ============================================================================
# Phase B: Authentication System
# ============================================================================

# JWT Configuration (Variables are kept as per your original file for .env consistency)
JWT_SECRET = os.getenv("SARAH_JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET:
    app_logger.critical("JWT_SECRET is not set. Using default insecure key. THIS IS DANGEROUS FOR PRODUCTION!")
    JWT_SECRET = "change-this-secret-key-in-production"

JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_DAYS = 7

def generate_jwt_token(user_id, email, display_name): # Added display_name
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'display_name': display_name, # Include display_name in token
        'exp': datetime.utcnow() + timedelta(days=JWT_EXP_DELTA_DAYS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        # Basic validation: ensure essential keys are present
        if 'user_id' in payload and 'email' in payload and 'exp' in payload:
            return payload
        app_logger.warning("JWT payload missing essential keys.")
        return None
    except jwt.ExpiredSignatureError:
        app_logger.info("Expired JWT token received.")
        return None
    except jwt.InvalidTokenError:
        app_logger.warning("Invalid JWT token received.")
        return None
    except Exception as e:
        app_logger.error(f"Unexpected error during JWT verification: {e}", exc_info=True)
        return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Authentication required. Token missing.'}), 401

        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Authentication failed. Invalid or expired token.'}), 401

        request.user_id = payload
        request.user_email = payload
        request.user_display_name = payload # Store display_name
        return f(*args, **kwargs)
    return decorated_function


@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    """Phase B: Register new user account."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        pin = data.get('pin', '')
        display_name = data.get('display_name', '') # Keep display_name in input

        # Validate input
        if not email or '@' not in email or '.' not in email: # More robust email check
            return jsonify({'error': 'Invalid email format.'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters.'}), 400
        if not pin or not pin.isdigit() or len(pin) != 4: # Strict 4-digit check
            return jsonify({'error': 'PIN must be exactly 4 digits.'}), 400
        if not display_name: # Ensure display name
            display_name = email.split('@', 1)[0] # Default if not provided

        # Import database functions
        try:
            from SarahMemoryDatabase import sm_get_user_by_email, sm_create_user, _get_cloud_conn, sm_insert_email_verification
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for authentication.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        # Check if user already exists
        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            existing_user = sm_get_user_by_email(email, conn) # Pass connection to avoid re-opening
            if existing_user: # sm_get_user_by_email should return None if not found
                return jsonify({'error': 'Email already registered.'}), 409

            # Hash password and PIN
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            pin_hash = bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Create user in database
            user_id = sm_create_user(email, display_name, password_hash, pin_hash, conn) # Pass connection
            if not user_id:
                raise Exception("Failed to create user in database.")

            # Generate and insert verification code
            verification_code = secrets.token_urlsafe(18)
            sm_insert_email_verification(user_id, email, verification_code, request.remote_addr, request.headers.get('User-Agent', ''), conn)

            # Send verification email
            send_verification_email(email, verification_code)

            return jsonify({
                'success': True,
                'user_id': user_id,
                'message': 'Registration successful. Please check your email for verification code.'
            }), 201

        except Exception as e:
            app_logger.exception(f" Registration failed for {email}.")
            if conn: conn.rollback() # Rollback on error
            return jsonify({'error': f'Registration failed: {str(e)}'}), 500
        finally:
            if conn: conn.close()

    except Exception as e:
        app_logger.exception(f" Unhandled error during register route processing.")
        return jsonify({'error': 'Internal server error during registration.'}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Phase B: Login user with email, password, and PIN."""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        pin = data.get('pin') or ''

        if not email or not password or not pin:
            return jsonify({'error': 'Email, password, and PIN are required.'}), 400

        # Import database function (cloud user auth)
        try:
            from SarahMemoryDatabase import _get_cloud_conn, sm_get_user_auth_data, sm_update_last_login
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for authentication.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            user_auth = sm_get_user_auth_data(email, conn)
            if not user_auth:
                return jsonify({'error': 'Invalid credentials.'}), 401

            # Normalize auth record
            def _field(obj, *names, default=None):
                if isinstance(obj, dict):
                    for n in names:
                        if n in obj and obj[n] is not None:
                            return obj[n]
                try:
                    for n in names:
                        try:
                            v = obj[n]
                            if v is not None:
                                return v
                        except Exception:
                            pass
                except Exception:
                    pass
                return default

            user_id = _field(user_auth, 'user_id', 'id', 'uid', default=email)
            display_name = _field(user_auth, 'display_name', 'name', 'username', default=email.split('@')[0])
            pw_hash = _field(user_auth, 'password_hash', 'pass_hash', 'password', 'pw_hash', default=None)
            pin_hash = _field(user_auth, 'pin_hash', 'pinhash', 'pin', default=None)
            is_active = _field(user_auth, 'is_active', 'active', default=1)

            if str(is_active) in ("0", "false", "False", "no", "NO"):
                return jsonify({'error': 'Account disabled. Please contact support.'}), 403

            if not pw_hash or not bcrypt.checkpw(password.encode('utf-8'), str(pw_hash).encode('utf-8')):
                return jsonify({'error': 'Invalid credentials.'}), 401

            if not pin_hash or not bcrypt.checkpw(pin.encode('utf-8'), str(pin_hash).encode('utf-8')):
                return jsonify({'error': 'Invalid credentials.'}), 401

            try:
                sm_update_last_login(user_id, conn)
            except Exception:
                pass

            token = generate_jwt_token(user_id, email, display_name)
            return jsonify({
                'ok': True,
                'token': token,
                'user': {'user_id': user_id, 'email': email, 'display_name': display_name}
            }), 200

        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    except Exception as e:
        app_logger.error(f"auth_login failed: {e}", exc_info=True)
        return jsonify({'error': 'Login failed.'}), 500

@app.get("/api/auth/verify-email")
def auth_verify_email():
    """Phase B: Verify email with code."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        code = data.get('code', '').strip()

        if not email or not code:
            return jsonify({'error': 'Email and verification code are required.'}), 400

        try:
            from SarahMemoryDatabase import _get_cloud_conn, sm_get_user_by_email, sm_get_verification_entry, sm_verify_user_email
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for email verification.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            user = sm_get_user_by_email(email, conn)
            if not user:
                return jsonify({'error': 'User not found.'}), 404

            verification_entry = sm_get_verification_entry(user, code, conn)

            if not verification_entry:
                return jsonify({'error': 'Invalid or expired verification code.'}), 400

            # Additional check if it's already verified
            if verification_entry.get('verified_at'):
                return jsonify({'error': 'Email already verified. Please try logging in.'}), 409

            # Mark as verified
            sm_verify_user_email(user, verification_entry, conn)

            return jsonify({'success': True, 'message': 'Email verified successfully.'}), 200

        except Exception as e:
            app_logger.exception(f" Email verification failed for {email}.")
            if conn: conn.rollback() # Rollback on error
            return jsonify({'error': f'Verification failed: {str(e)}'}), 500
        finally:
            if conn: conn.close()

    except Exception as e:
        app_logger.exception(f" Unhandled error during email verification route processing.")
        return jsonify({'error': 'Internal server error during email verification.'}), 500

@app.route('/api/user/preferences', methods=['GET', 'PUT', 'POST'])
@require_auth
def user_preferences():
    """Phase B: Get or update user preferences."""
    conn = None
    try:
        from SarahMemoryDatabase import sm_get_user_preferences, sm_update_user_preferences, _get_cloud_conn
        conn = _get_cloud_conn()
        if not conn:
            return jsonify({'error': 'Cloud database connection unavailable.'}), 503

        if request.method == 'GET':
            prefs = sm_get_user_preferences(request.user_id, conn)
            return jsonify(prefs), 200

        elif request.method == 'PUT':
            data = request.json
            success = sm_update_user_preferences(request.user_id, data, conn)
            if success:
                return jsonify({'success': True}), 200
            else:
                return jsonify({'error': 'Failed to update preferences.'}), 500
    except ImportError:
        app_logger.error("SarahMemoryDatabase module not found for user preferences.")
        return jsonify({'error': 'Database module unavailable.'}), 503
    except Exception as e:
        app_logger.exception(f" Preferences operation failed for user {request.user_id}.")
        return jsonify({'error': f'Operation failed: {str(e)}'}), 500
    finally:
        if conn: conn.close()


def send_verification_email(email, code):
    """Phase B: Send verification email with code."""
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    smtp_from = os.getenv('SMTP_FROM_EMAIL', 'noreply@sarahmemory.com')

    if not smtp_user or not smtp_password or not smtp_host:
        app_logger.warning(" SMTP not fully configured (missing host, user, or password). Skipping email to %s.", email)
        return

    msg = MIMEMultipart('alternative')
    msg = 'SarahMemory Email Verification'
    msg = smtp_from
    msg = email

    text = f"""
Welcome to SarahMemory!

Your verification code is: {code}

This code expires in 15 minutes.

If you didn't request this, please ignore this email.
    """

    html = f"""
<html>
  <body style="font-family: Arial, sans-serif;">
    <h2>Welcome to SarahMemory!</h2>
    <p>Your verification code is:</p>
    <h1 style="background: #5f9ef7; color: white; padding: 20px; text-align: center; font-size: 32px; letter-spacing: 5px;">
      {code}
    </h1>
    <p>This code expires in 15 minutes.</p>
    <p style="color: #666; font-size: 12px;">If you didn't request this, please ignore this email.</p>
  </body>
</html>
    """

    msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, email, msg.as_string())
        app_logger.info(" Verification email sent to %s.", email)
    except smtplib.SMTPAuthenticationError:
        app_logger.error(f" SMTP authentication error for user {smtp_user}. Check SMTP_PASSWORD.")
    except smtplib.SMTPException as e:
        app_logger.error(f" SMTP error sending email to {email}: {e}", exc_info=True)
    except Exception as e:
        app_logger.error(f" Unexpected error sending email to {email}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# SarahMemory 2D Avatar Live PNG State / Manifest / Life-Cycle Contract
# ---------------------------------------------------------------------------
# WebUI-facing contract for the Custom AvatarPanel. This exposes the current
# governed 2D still-frame selection while preserving the existing 3D, media,
# desktop mirror, and legacy AvatarPanel API paths.
#
# Design rule:
# - Active states (speaking/listening/thinking/busy/diagnostics) always win.
# - Heartbeat/life motion only controls idle presentation.
# - The manifest is honored first, then the avatar directory is scanned.
# - Any 29_*.png file dropped into resources/avatars/2D/default is discovered
#   automatically and becomes available as state_29 / extra_29 / concept_29.
_AVATAR_LIVE_LOCK = threading.RLock()
_AVATAR_BOOT_TS = time.time()
_AVATAR_LIVE_STATE = {
    "mode": "avatar_2d",
    "expression": "neutral",
    "emotion": "neutral",
    "speaking": False,
    "listening": False,
    "thinking": False,
    "busy": False,
    "diagnostics": False,
    "current_action": "boot_greeting",
    "life_state": "boot_greeting",
    "life_enabled": True,
    "sequence": 0,
    "heartbeat_count": 0,
    "booted_at": _AVATAR_BOOT_TS,
    "updated_at": _AVATAR_BOOT_TS,
    "last_interaction_at": _AVATAR_BOOT_TS,
    "last_life_tick": 0.0,
    "last_random_at": 0.0,
    "locked_until": _AVATAR_BOOT_TS + 6.0,
    "last_success_at": 0.0,
    "last_error_at": 0.0,
}

_AVATAR_ROLE_MAP = {
    "default": "sarah-avatar.png",
    "neutral": "19_neutral_forward.png",
    "ready": "20_soft_smile.png",
    "idle": "19_neutral_forward.png",
    "thinking": "09_listening_thinking.png",
    "listening": "09_listening_thinking.png",
    "speaking_soft": "07_speaking_soft.png",
    "speaking_open": "08_speaking_open.png",
    "happy": "11_happy_open_smile.png",
    "joy": "11_happy_open_smile.png",
    "trust": "20_soft_smile.png",
    "surprise": "13_surprised_open_mouth.png",
    "shocked": "14_shocked_wide_eyes.png",
    "sad": "05_sad_worried.png",
    "sadness": "05_sad_worried.png",
    "concerned": "03_concerned_worried.png",
    "worried": "03_concerned_worried.png",
    "skeptical": "04_skeptical_side_eye.png",
    "frustrated": "10_overwhelmed_frustrated.png",
    "annoyed": "15_annoyed_pout.png",
    "anger": "16_angry_yelling.png",
    "angry": "16_angry_yelling.png",
    "playful": "17_playful_wink_laugh.png",
    "pointing": "18_playful_pointing.png",
    "hello": "12_waving_hello.png",
    "waving": "12_waving_hello.png",
    "wave": "12_waving_hello.png",
    "sleepy": "02_sleepy_half_lidded.png",
    "relaxed": "01_relaxed_closed_eyes.png",
    "asleep": "01_relaxed_closed_eyes.png",
    "thumbs_up": "21_thumbs_up_smile.png",
    "approval": "21_thumbs_up_smile.png",
    "approve": "21_thumbs_up_smile.png",
    "confirmed": "21_thumbs_up_smile.png",
    "good": "21_thumbs_up_smile.png",
    "ok": "21_thumbs_up_smile.png",
    "pleading": "22_pleading_worry.png",
    "please": "22_pleading_worry.png",
    "vulnerable": "22_pleading_worry.png",
    "empathy_worry": "22_pleading_worry.png",
    "staredown": "22_staredown_contest.png",
    "contest": "22_staredown_contest.png",
    "direct": "22_staredown_contest.png",
    "serious_focus": "22_staredown_contest.png",
    "heartfelt": "23_heartfelt_emotional_kindness.png",
    "emotional": "23_heartfelt_emotional_kindness.png",
    "kindness": "23_heartfelt_emotional_kindness.png",
    "compassionate": "23_heartfelt_emotional_kindness.png",
    "supportive": "23_heartfelt_emotional_kindness.png",
    "pondering": "24_pondering_stare.png",
    "pondering_stare": "24_pondering_stare.png",
    "curious_stare": "24_pondering_stare.png",
    "exhausted": "25_exhausted_sleepy.png",
    "tired": "25_exhausted_sleepy.png",
    "fatigue": "25_exhausted_sleepy.png",
    "very_sleepy": "25_exhausted_sleepy.png",
    "victory": "26_victory_celebration.png",
    "celebration": "26_victory_celebration.png",
    "success": "26_victory_celebration.png",
    "win": "26_victory_celebration.png",
    "hello_again": "27_waving_hello_again.png",
    "waving_again": "27_waving_hello_again.png",
    "greeting_energetic": "27_waving_hello_again.png",
    "wondering": "28_wondering_planning_stare.png",
    "planning": "28_wondering_planning_stare.png",
    "wondering_planning": "28_wondering_planning_stare.png",
    "state_29": "29_extra_avatar_state.png",
    "extra_29": "29_extra_avatar_state.png",
    "concept_29": "29_extra_avatar_state.png",
    "random_29": "29_extra_avatar_state.png",
}

_AVATAR_VALID_MODES = {"avatar_2d", "avatar_3d", "desktop_mirror", "media", "idle"}
_AVATAR_IDLE_RANDOM_POOL = (
    "ready", "neutral", "thinking", "pondering", "wondering",
    "skeptical", "playful", "waving_again", "heartfelt", "state_29",
)
_AVATAR_IDLE_NIGHT_POOL = (
    "sleepy", "very_sleepy", "relaxed", "neutral", "pondering", "state_29",
)
_AVATAR_BUSY_POOL = (
    "thinking", "pondering", "wondering", "serious_focus", "concerned",
)
_AVATAR_LONG_IDLE_SECONDS = int(os.getenv("SARAH_AVATAR_LONG_IDLE_SECONDS", "180") or 180)
_AVATAR_ASLEEP_IDLE_SECONDS = int(os.getenv("SARAH_AVATAR_ASLEEP_IDLE_SECONDS", "600") or 600)
_AVATAR_RANDOM_MIN_SECONDS = int(os.getenv("SARAH_AVATAR_RANDOM_MIN_SECONDS", "12") or 12)
_AVATAR_RANDOM_MAX_SECONDS = int(os.getenv("SARAH_AVATAR_RANDOM_MAX_SECONDS", "38") or 38)
_AVATAR_HEARTBEAT_MIN_SECONDS = float(os.getenv("SARAH_AVATAR_HEARTBEAT_MIN_SECONDS", "1.0") or 1.0)

def _avatar_default_dir() -> str:
    try:
        root = _globals_paths().get("ROOT_DIR") or BASE_DIR
    except Exception:
        root = BASE_DIR
    candidates = [
        os.path.join(root, "resources", "avatars", "2D", "default"),
        os.path.join(BASE_DIR, "resources", "avatars", "2D", "default"),
        os.path.join(os.getcwd(), "resources", "avatars", "2D", "default"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]

def _avatar_manifest_path() -> str:
    d = _avatar_default_dir()
    for name in ("avatar-manifest.json", "manifest.json"):
        pth = os.path.join(d, name)
        if os.path.isfile(pth):
            return pth
    return os.path.join(d, "avatar-manifest.json")

def _avatar_read_manifest() -> dict:
    try:
        manifest = _avatar_manifest_path()
        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        app_logger.debug(f"Avatar manifest read failed: {e}")
    return {}

def _avatar_effective_role_map() -> dict:
    role_map = dict(_AVATAR_ROLE_MAP)
    data = _avatar_read_manifest()
    raw = data.get("role_map") if isinstance(data, dict) else {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k or "").strip().lower()
            val = os.path.basename(str(v or "").strip())
            if key and val:
                role_map[key] = val
    for alias in ("state_29", "extra_29", "concept_29", "random_29"):
        role_map.setdefault(alias, "29_extra_avatar_state.png")
    return role_map

def _safe_avatar_files() -> list[str]:
    files: list[str] = []
    try:
        data = _avatar_read_manifest()
        raw = data.get("files") if isinstance(data, dict) else []
        if isinstance(raw, list):
            for item in raw:
                name = os.path.basename(str(item or "").strip())
                if name.lower().endswith(".png") and name not in files:
                    files.append(name)
    except Exception:
        pass
    try:
        d = _avatar_default_dir()
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                safe = os.path.basename(fn)
                if safe.lower().endswith(".png") and safe not in files:
                    files.append(safe)
    except Exception:
        pass
    return files

def _avatar_public_url(filename: str) -> str:
    return f"/api/avatar/2d/{os.path.basename(filename or 'sarah-avatar.png')}"


# ---------------------------------------------------------------------------
# SarahMemory 3D Avatar Runtime Asset Contract
# ---------------------------------------------------------------------------
# The browser cannot load local Windows paths such as S:\SarahMemory\resources
# directly.  This route family exposes only runtime-safe 3D avatar delivery files
# from resources/avatars/3D.  Source/development files such as .blend, .py, and
# logs are intentionally not web-served.
_AVATAR_3D_ALLOWED_EXTENSIONS = {
    ".glb",
    ".gltf",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".json",
}

_AVATAR_3D_MIMETYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".json": "application/json",
}

def _avatar_3d_dir() -> str:
    """Return the local runtime folder for AvatarPanel 3D assets."""
    try:
        root = _globals_paths().get("ROOT_DIR") or BASE_DIR
    except Exception:
        root = BASE_DIR
    candidates = [
        os.path.join(root, "resources", "avatars", "3D"),
        os.path.join(BASE_DIR, "resources", "avatars", "3D"),
        os.path.join(os.getcwd(), "resources", "avatars", "3D"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]

def _avatar_3d_safe_name(filename: str) -> str:
    safe_name = os.path.basename(str(filename or "").strip())
    if not safe_name:
        return ""
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _AVATAR_3D_ALLOWED_EXTENSIONS:
        return ""
    return safe_name

def _safe_avatar_3d_files() -> list[str]:
    """List runtime-safe 3D files.  Development sources are intentionally hidden."""
    files: list[str] = []
    try:
        d = _avatar_3d_dir()
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                safe = _avatar_3d_safe_name(fn)
                if safe and safe not in files:
                    files.append(safe)
    except Exception as e:
        app_logger.debug(f"Avatar 3D file scan failed: {e}")
    return files

def _avatar_3d_manifest_path() -> str:
    d = _avatar_3d_dir()
    for name in (
        "sarahmemory_3d_avatar_manifest.json",
        "avatar_3d_manifest.json",
        "avatar-manifest-3d.json",
        "manifest.json",
    ):
        pth = os.path.join(d, name)
        if os.path.isfile(pth):
            return pth
    return os.path.join(d, "sarahmemory_3d_avatar_manifest.json")

def _avatar_3d_read_manifest() -> dict:
    try:
        manifest = _avatar_3d_manifest_path()
        if os.path.isfile(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        app_logger.debug(f"Avatar 3D manifest read failed: {e}")
    return {}

def _avatar_3d_pick_model() -> str | None:
    """Select the active GLB/GLTF runtime model, preferring the generated SarahMemory GLB."""
    files = set(_safe_avatar_3d_files())
    manifest = _avatar_3d_read_manifest()
    candidates: list[str] = []

    for key in ("glb", "model", "model_file", "modelUrl", "model_url"):
        raw = manifest.get(key) if isinstance(manifest, dict) else None
        if raw:
            candidates.append(os.path.basename(str(raw)))

    candidates.extend([
        "sarahmemory_3d_avatar.glb",
        "sarahmemory_happy_face_ball.glb",
    ])

    for name in sorted(files):
        if name.lower().endswith((".glb", ".gltf")) and name not in candidates:
            candidates.append(name)

    for candidate in candidates:
        safe = _avatar_3d_safe_name(candidate)
        if safe and safe in files and safe.lower().endswith((".glb", ".gltf")):
            return safe
    return None

def _avatar_3d_public_url(filename: str | None) -> str:
    safe = _avatar_3d_safe_name(filename or "")
    return f"/api/avatar/3d/{safe}" if safe else ""

def _avatar_3d_manifest_payload() -> dict:
    manifest = _avatar_3d_read_manifest()
    files = _safe_avatar_3d_files()
    model_file = _avatar_3d_pick_model()
    return {
        "success": True,
        "ok": bool(model_file),
        "base_url": "/api/avatar/3d",
        "asset_dir": _avatar_3d_dir(),
        "manifest_path": _avatar_3d_manifest_path(),
        "manifest": manifest,
        "files": files,
        "model_file": model_file,
        "model_url": _avatar_3d_public_url(model_file),
        "blocked_extensions": [".blend", ".py", ".log"],
        "runtime_only": True,
    }

def _avatar_3d_spec_payload(state: dict | None = None) -> dict:
    """Return the Avatar3D.tsx backend contract used by the dropdown 3D mode."""
    state = dict(state or {})
    model_file = _avatar_3d_pick_model()
    expression = str(state.get("expression") or state.get("emotion") or "neutral")
    speaking = bool(state.get("speaking", False))
    listening = bool(state.get("listening", False))

    if not model_file:
        return {
            "renderMode": "procedural_holo",
            "modelUrl": "",
            "backgroundType": "none",
            "pose": "stand",
            "gesture": "none",
            "expression": expression,
            "speaking": speaking,
            "listening": listening,
            "source": "avatar_3d_missing_model",
        }

    return {
        "renderMode": "gltf_model",
        "modelUrl": _avatar_3d_public_url(model_file),
        "backgroundType": "none",
        "pose": "stand",
        "gesture": "none",
        "lookAt": {"x": 0, "y": 1.4, "z": 0},
        "expression": expression,
        "speaking": speaking,
        "listening": listening,
        "source": "resources/avatars/3D",
        "modelFile": model_file,
        "runtimeOnly": True,
    }


def _avatar_normalize_mode(mode: str | None) -> str:
    m = str(mode or "avatar_2d").strip().lower()
    if m in {"2d", "avatar2d", "avatar_2d", "avatar-2d", "avater_2d"}:
        return "avatar_2d"
    if m in {"3d", "avatar3d", "avatar_3d", "avatar-3d"}:
        return "avatar_3d"
    if m in {"desktop", "mirror", "desktop_mirror", "desktop-mirror"}:
        return "desktop_mirror"
    if m in {"media", "call", "video_conference", "conference"}:
        return "media"
    if m == "idle":
        return "idle"
    return "avatar_2d"

def _avatar_role_candidates(role_or_file: str, available: set[str]) -> list[str]:
    raw = str(role_or_file or "").strip()
    key = raw.lower()
    role_map = _avatar_effective_role_map()
    candidates: list[str] = []

    def add(name: str) -> None:
        safe = os.path.basename(str(name or "").strip())
        if safe and safe not in candidates:
            candidates.append(safe)

    if raw:
        add(raw)
    mapped = role_map.get(key)
    if mapped:
        add(mapped)

    prefixes: list[str] = []
    if key in {"state_29", "extra_29", "concept_29", "random_29", "29"}:
        prefixes.append("29_")
    if key in {"thumbs_up", "approval", "approve", "confirmed", "good", "ok", "21"}:
        prefixes.append("21_")
    if key in {"pleading", "please", "vulnerable", "empathy_worry", "staredown", "contest", "direct", "serious_focus", "22"}:
        prefixes.append("22_")
    if key in {"heartfelt", "emotional", "kindness", "compassionate", "supportive", "23"}:
        prefixes.append("23_")
    if key in {"pondering", "pondering_stare", "curious_stare", "24"}:
        prefixes.append("24_")
    if key in {"exhausted", "tired", "fatigue", "very_sleepy", "25"}:
        prefixes.append("25_")
    if key in {"victory", "celebration", "success", "win", "26"}:
        prefixes.append("26_")
    if key in {"hello_again", "waving_again", "greeting_energetic", "27"}:
        prefixes.append("27_")
    if key in {"wondering", "planning", "wondering_planning", "28"}:
        prefixes.append("28_")

    for prefix in prefixes:
        for name in sorted(available):
            if name.lower().startswith(prefix) and name.lower().endswith(".png"):
                add(name)

    return candidates

def _avatar_select_existing(role_or_file: str, available: set[str]) -> str | None:
    for candidate in _avatar_role_candidates(role_or_file, available):
        if candidate in available:
            return candidate
    return None

def _avatar_pick_image(state: dict | None = None) -> str:
    state = dict(state or {})
    available = set(_safe_avatar_files())

    def choose(role_or_file: str) -> str:
        selected = _avatar_select_existing(role_or_file, available)
        if selected:
            return selected
        if "sarah-avatar.png" in available:
            return "sarah-avatar.png"
        if "19_neutral_forward.png" in available:
            return "19_neutral_forward.png"
        return sorted(available)[0] if available else "sarah-avatar.png"

    if bool(state.get("speaking")):
        return choose("speaking_open" if int(time.monotonic() * 8) % 2 == 0 else "speaking_soft")
    if bool(state.get("listening")):
        return choose("listening")
    if bool(state.get("thinking")):
        return choose("thinking")
    if bool(state.get("diagnostics")):
        return choose("serious_focus")
    if bool(state.get("busy")):
        return choose("pondering")

    action = str(state.get("current_action") or "").lower()
    if any(k in action for k in ("hello", "greet", "wave", "boot")):
        return choose("hello_again" if "again" in action else "hello")
    if any(k in action for k in ("success", "correct", "complete", "done", "victory", "win")):
        return choose("success")
    if any(k in action for k in ("thumb", "approve", "confirmed", "good")):
        return choose("thumbs_up")
    if any(k in action for k in ("error", "fail", "confused")):
        return choose("concerned")
    if any(k in action for k in ("diagnostic", "self_check")):
        return choose("serious_focus")
    if any(k in action for k in ("busy", "process", "work")):
        return choose("pondering")
    if any(k in action for k in ("think", "reason")):
        return choose("thinking")
    if any(k in action for k in ("asleep", "sleep")):
        return choose("very_sleepy")
    if any(k in action for k in ("random_29", "state_29", "extra_29")):
        return choose("state_29")

    expr = str(state.get("expression") or state.get("emotion") or "neutral").lower().strip()
    return choose(expr or "neutral")

def _avatar_is_night_window(now_dt: datetime | None = None) -> bool:
    now_dt = now_dt or datetime.now()
    return now_dt.hour >= 22 or now_dt.hour < 5

def _avatar_life_pick(pool: tuple[str, ...] | list[str], available: set[str]) -> str:
    usable = [r for r in pool if _avatar_select_existing(r, available)]
    if not usable:
        usable = ["ready", "neutral"]
    return random.choice(usable)

def _avatar_life_tick(force: bool = False) -> None:
    now = time.time()
    with _AVATAR_LIVE_LOCK:
        if not bool(_AVATAR_LIVE_STATE.get("life_enabled", True)):
            return
        if not force and (now - float(_AVATAR_LIVE_STATE.get("last_life_tick") or 0.0)) < _AVATAR_HEARTBEAT_MIN_SECONDS:
            return

        _AVATAR_LIVE_STATE["last_life_tick"] = now
        _AVATAR_LIVE_STATE["heartbeat_count"] = int(_AVATAR_LIVE_STATE.get("heartbeat_count") or 0) + 1

        if bool(_AVATAR_LIVE_STATE.get("speaking")):
            _AVATAR_LIVE_STATE["life_state"] = "speaking"
            _AVATAR_LIVE_STATE["current_action"] = "speaking"
            return
        if bool(_AVATAR_LIVE_STATE.get("listening")):
            _AVATAR_LIVE_STATE["life_state"] = "listening"
            _AVATAR_LIVE_STATE["current_action"] = "listening"
            return
        if bool(_AVATAR_LIVE_STATE.get("diagnostics")):
            _AVATAR_LIVE_STATE["life_state"] = "diagnostics"
            _AVATAR_LIVE_STATE["current_action"] = "diagnostics"
            _AVATAR_LIVE_STATE["expression"] = "serious_focus"
            return
        if bool(_AVATAR_LIVE_STATE.get("busy")) or bool(_AVATAR_LIVE_STATE.get("thinking")):
            available = set(_safe_avatar_files())
            expr = _avatar_life_pick(_AVATAR_BUSY_POOL, available)
            _AVATAR_LIVE_STATE["life_state"] = "busy"
            _AVATAR_LIVE_STATE["current_action"] = "busy"
            _AVATAR_LIVE_STATE["expression"] = expr
            _AVATAR_LIVE_STATE["emotion"] = expr
            _AVATAR_LIVE_STATE["sequence"] = int(_AVATAR_LIVE_STATE.get("sequence") or 0) + 1
            _AVATAR_LIVE_STATE["updated_at"] = now
            return

        locked_until = float(_AVATAR_LIVE_STATE.get("locked_until") or 0.0)
        if now < locked_until:
            return

        idle_seconds = max(0.0, now - float(_AVATAR_LIVE_STATE.get("last_interaction_at") or _AVATAR_BOOT_TS))
        available = set(_safe_avatar_files())
        is_night = _avatar_is_night_window()

        if idle_seconds >= _AVATAR_ASLEEP_IDLE_SECONDS:
            expr = "very_sleepy" if _avatar_select_existing("very_sleepy", available) else "sleepy"
            life_state = "idle_asleep"
            action = "asleep"
        elif idle_seconds >= _AVATAR_LONG_IDLE_SECONDS:
            expr = "sleepy" if not is_night else _avatar_life_pick(_AVATAR_IDLE_NIGHT_POOL, available)
            life_state = "idle_long"
            action = "idle_long"
        elif is_night:
            expr = _avatar_life_pick(_AVATAR_IDLE_NIGHT_POOL, available)
            life_state = "sleepy_night"
            action = "sleepy_night"
        else:
            min_wait = max(4, min(_AVATAR_RANDOM_MIN_SECONDS, _AVATAR_RANDOM_MAX_SECONDS))
            max_wait = max(min_wait, _AVATAR_RANDOM_MAX_SECONDS)
            next_due = float(_AVATAR_LIVE_STATE.get("last_random_at") or 0.0) + random.uniform(min_wait, max_wait)
            if not force and now < next_due:
                return
            expr = _avatar_life_pick(_AVATAR_IDLE_RANDOM_POOL, available)
            life_state = "idle_random"
            action = "random_idle_motion"
            _AVATAR_LIVE_STATE["last_random_at"] = now

        _AVATAR_LIVE_STATE["life_state"] = life_state
        _AVATAR_LIVE_STATE["current_action"] = action
        _AVATAR_LIVE_STATE["expression"] = expr
        _AVATAR_LIVE_STATE["emotion"] = expr
        _AVATAR_LIVE_STATE["sequence"] = int(_AVATAR_LIVE_STATE.get("sequence") or 0) + 1
        _AVATAR_LIVE_STATE["updated_at"] = now

def _avatar_manifest_payload() -> dict:
    role_map = _avatar_effective_role_map()
    files = _safe_avatar_files()
    return {
        "success": True,
        "base_url": "/api/avatar/2d",
        "default_file": "sarah-avatar.png",
        "role_map": role_map,
        "files": files,
        "target_dimensions": [1254, 1254],
        "state_count": len([f for f in files if f.lower().endswith(".png")]),
        "supports_dynamic_29": True,
        "manifest_path": _avatar_manifest_path(),
    }

def _avatar_state_payload(extra: dict | None = None) -> dict:
    _avatar_life_tick()
    with _AVATAR_LIVE_LOCK:
        state = dict(_AVATAR_LIVE_STATE)
        if isinstance(extra, dict):
            protected = {
                "mode", "expression", "emotion", "speaking", "listening",
                "thinking", "busy", "diagnostics", "current_action", "life_state",
                "current_image", "avatar_image", "avatar_image_url", "sequence",
                "updated_at", "last_interaction_at", "last_life_tick",
            }
            state.update({k: v for k, v in extra.items() if v is not None and k not in protected})
            state["controller_state"] = {k: v for k, v in extra.items() if v is not None}
        state["mode"] = _avatar_normalize_mode(state.get("mode"))
        state["idle_seconds"] = max(0.0, time.time() - float(state.get("last_interaction_at") or _AVATAR_BOOT_TS))
        state["night_mode"] = _avatar_is_night_window()
        current_file = _avatar_pick_image(state)
        state["current_image"] = current_file
        state["avatar_image"] = current_file
        state["avatar_image_url"] = _avatar_public_url(current_file)
        state["manifest"] = _avatar_manifest_payload()
        state["avatar_3d"] = _avatar_3d_manifest_payload()
        state["spec"] = _avatar_3d_spec_payload(state)
        state["success"] = True
        return state

def _avatar_update_state(**updates) -> dict:
    clean: dict[str, object] = {}
    mark_interaction = False
    lock_seconds = 0.0

    for k, v in updates.items():
        if k == "mode":
            clean[k] = _avatar_normalize_mode(str(v or "avatar_2d"))
        elif k in {"expression", "emotion", "current_action", "life_state"}:
            clean[k] = str(v or "").strip() or _AVATAR_LIVE_STATE.get(k, "neutral")
        elif k in {"speaking", "listening", "thinking", "busy", "diagnostics", "life_enabled"}:
            clean[k] = bool(v)
        elif k in {"event", "result"}:
            event = str(v or "").strip().lower()
            if event in {"boot", "startup", "hello", "greeting"}:
                clean["current_action"] = "boot_greeting"
                clean["expression"] = "hello"
                clean["emotion"] = "hello"
                clean["life_state"] = "boot_greeting"
                lock_seconds = max(lock_seconds, 5.0)
            elif event in {"success", "correct", "complete", "completed", "done", "ok", "approved"}:
                clean["current_action"] = "success"
                clean["expression"] = "success"
                clean["emotion"] = "success"
                clean["life_state"] = "success"
                clean["last_success_at"] = time.time()
                lock_seconds = max(lock_seconds, 4.0)
            elif event in {"thumbs_up", "approval", "confirmed", "good"}:
                clean["current_action"] = "thumbs_up"
                clean["expression"] = "thumbs_up"
                clean["emotion"] = "thumbs_up"
                clean["life_state"] = "success"
                clean["last_success_at"] = time.time()
                lock_seconds = max(lock_seconds, 4.0)
            elif event in {"error", "failed", "failure", "confused"}:
                clean["current_action"] = "error"
                clean["expression"] = "concerned"
                clean["emotion"] = "concerned"
                clean["life_state"] = "error"
                clean["last_error_at"] = time.time()
                lock_seconds = max(lock_seconds, 4.0)
            elif event in {"diagnostics", "diagnostic", "self_check", "self_diagnostics"}:
                clean["diagnostics"] = True
                clean["current_action"] = "diagnostics"
                clean["expression"] = "serious_focus"
                clean["emotion"] = "serious_focus"
                clean["life_state"] = "diagnostics"
                lock_seconds = max(lock_seconds, 3.0)
            elif event in {"busy", "working", "processing"}:
                clean["busy"] = True
                clean["current_action"] = "busy"
                clean["expression"] = "pondering"
                clean["emotion"] = "pondering"
                clean["life_state"] = "busy"
                lock_seconds = max(lock_seconds, 3.0)
            elif event in {"idle", "ready", "reset"}:
                clean["busy"] = False
                clean["diagnostics"] = False
                clean["thinking"] = False
                clean["current_action"] = "idle"
                clean["expression"] = "ready"
                clean["emotion"] = "ready"
                clean["life_state"] = "ready"
                lock_seconds = max(lock_seconds, 1.0)
        elif k in {"touch", "interaction", "user_interaction"} and bool(v):
            mark_interaction = True

    with _AVATAR_LIVE_LOCK:
        if clean.get("speaking") is True:
            clean["listening"] = False
            clean["busy"] = False
            clean["diagnostics"] = False
            clean.setdefault("current_action", "speaking")
            clean.setdefault("life_state", "speaking")
            mark_interaction = True
        if clean.get("listening") is True:
            clean["speaking"] = False
            clean["busy"] = False
            clean["diagnostics"] = False
            clean.setdefault("current_action", "listening")
            clean.setdefault("life_state", "listening")
            mark_interaction = True
        if clean.get("speaking") is False and _AVATAR_LIVE_STATE.get("current_action") == "speaking":
            clean.setdefault("current_action", "ready")
            clean.setdefault("expression", "ready")
            clean.setdefault("emotion", "ready")
            clean.setdefault("life_state", "ready")
            lock_seconds = max(lock_seconds, 1.5)
        if clean.get("listening") is False and _AVATAR_LIVE_STATE.get("current_action") == "listening":
            clean.setdefault("current_action", "ready")
            clean.setdefault("expression", "ready")
            clean.setdefault("emotion", "ready")
            clean.setdefault("life_state", "ready")
            lock_seconds = max(lock_seconds, 1.5)

        _AVATAR_LIVE_STATE.update(clean)
        now = time.time()
        if mark_interaction or clean:
            _AVATAR_LIVE_STATE["last_interaction_at"] = now
        if lock_seconds > 0:
            _AVATAR_LIVE_STATE["locked_until"] = max(float(_AVATAR_LIVE_STATE.get("locked_until") or 0.0), now + lock_seconds)
        _AVATAR_LIVE_STATE["sequence"] = int(_AVATAR_LIVE_STATE.get("sequence") or 0) + 1
        _AVATAR_LIVE_STATE["updated_at"] = now
        return _avatar_state_payload()

@app.route("/api/avatar/manifest", methods=["GET"])
def avatar_live_manifest():
    return jsonify(_avatar_manifest_payload()), 200

@app.route("/api/avatar/heartbeat", methods=["GET", "POST"])
def avatar_live_heartbeat():
    data = request.get_json(silent=True) or {}
    if request.method == "POST" and isinstance(data, dict):
        updates = {k: data.get(k) for k in (
            "mode", "expression", "emotion", "current_action", "life_state",
            "speaking", "listening", "thinking", "busy", "diagnostics",
            "life_enabled", "event", "result", "touch", "interaction", "user_interaction",
        ) if k in data}
        if updates:
            return jsonify(_avatar_update_state(**updates)), 200
    _avatar_life_tick(force=True)
    return jsonify(_avatar_state_payload()), 200

@app.route("/api/avatar/2d/<path:filename>", methods=["GET"])
def avatar_live_asset(filename: str):
    safe_name = os.path.basename(filename or "")
    allowed = set(_safe_avatar_files())
    if safe_name not in allowed and safe_name != "sarah-avatar.png":
        abort(404)
    try:
        return send_from_directory(_avatar_default_dir(), safe_name, mimetype="image/png", max_age=1)
    except Exception:
        abort(404)



@app.route("/api/avatar/3d/manifest", methods=["GET"])
def avatar_live_3d_manifest():
    return jsonify(_avatar_3d_manifest_payload()), 200

@app.route("/api/avatar/3d/spec", methods=["GET"])
def avatar_live_3d_spec():
    return jsonify({
        "success": True,
        "ok": True,
        "spec": _avatar_3d_spec_payload(_avatar_state_payload()),
        "manifest": _avatar_3d_manifest_payload(),
    }), 200

@app.route("/api/avatar/3d/<path:filename>", methods=["GET"])
def avatar_live_3d_asset(filename: str):
    safe_name = _avatar_3d_safe_name(filename)
    if not safe_name:
        abort(404)

    allowed = set(_safe_avatar_3d_files())
    if safe_name not in allowed:
        abort(404)

    ext = os.path.splitext(safe_name)[1].lower()
    mimetype = _AVATAR_3D_MIMETYPES.get(ext, "application/octet-stream")
    try:
        return send_from_directory(_avatar_3d_dir(), safe_name, mimetype=mimetype, max_age=1)
    except Exception:
        abort(404)

@app.route("/api/avatar/state/live", methods=["GET", "POST"])
def avatar_live_state():
    data = request.get_json(silent=True) or {}
    if request.method == "POST" and isinstance(data, dict):
        updates = {k: data.get(k) for k in (
            "mode", "expression", "emotion", "current_action", "life_state",
            "speaking", "listening", "thinking", "busy", "diagnostics",
            "life_enabled", "event", "result", "touch", "interaction", "user_interaction",
        ) if k in data}
        if updates:
            return jsonify(_avatar_update_state(**updates)), 200
    return jsonify(_avatar_state_payload()), 200

@app.route("/api/avatar/speaking", methods=["POST"])
def avatar_live_speaking():
    data = request.get_json(silent=True) or {}
    value = data.get("speaking", data.get("state", data.get("enabled", False)))
    return jsonify(_avatar_update_state(speaking=bool(value))), 200

@app.route("/api/avatar/listening", methods=["POST"])
def avatar_live_listening():
    data = request.get_json(silent=True) or {}
    value = data.get("listening", data.get("state", data.get("enabled", False)))
    return jsonify(_avatar_update_state(listening=bool(value))), 200

@app.route("/api/avatar/event", methods=["POST"])
def avatar_live_event():
    data = request.get_json(silent=True) or {}
    event = data.get("event", data.get("result", "idle"))
    extra = {k: data.get(k) for k in ("mode", "expression", "emotion", "current_action") if k in data}
    extra["event"] = event
    return jsonify(_avatar_update_state(**extra)), 200

# ===========================================================================
# AVATAR PANEL / MULTIMEDIA / VIDEO CONFERENCE API ROUTES
# ===========================================================================
# These routes integrate with SarahMemoryAvatarPanel.py to provide
# multimedia display, avatar animation, desktop mirror, and video conferencing

_avatar_panel_api = None # Global instance for caching the API object


def get_avatar_panel_api():
    """Get or create the Avatar Panel API instance, caching it."""
    global _avatar_panel_api
    if _avatar_panel_api is None:
        try:
            # Prefer importing from UnifiedAvatarController as per AGI spec
            from UnifiedAvatarController import get_panel_api
            _avatar_panel_api = get_panel_api()
            if _avatar_panel_api:
                app_logger.info("Successfully loaded Avatar Panel API via UnifiedAvatarController.")
            else:
                app_logger.warning("UnifiedAvatarController.get_panel_api returned None.")
        except ImportError:
            try: # Fallback to older SarahMemoryAvatarPanel if UnifiedAvatarController is not ready
                from SarahMemoryAvatarPanel import get_panel_api as smap_get_panel_api
                _avatar_panel_api = smap_get_panel_api()
                if _avatar_panel_api:
                    app_logger.info("Successfully loaded Avatar Panel API via SarahMemoryAvatarPanel (fallback).")
                else:
                    app_logger.warning("SarahMemoryAvatarPanel.get_panel_api returned None.")
            except ImportError:
                app_logger.error("Neither UnifiedAvatarController nor SarahMemoryAvatarPanel found. Avatar features disabled.")
            except Exception as e:
                app_logger.error(f" Error loading panel API via SarahMemoryAvatarPanel: {e}", exc_info=True)
        except Exception as e:
            app_logger.error(f" Error loading panel API via UnifiedAvatarController: {e}", exc_info=True)
    return _avatar_panel_api

def _avatar_api_response_wrapper(func):
    """Decorator to standardize responses for avatar panel API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api = get_avatar_panel_api()
        if not api:
            return jsonify({"error": "Avatar panel not available or initialized."}), 503
        try:
            result = func(api, *args, **kwargs)
            return jsonify(result), 200
        except Exception as e:
            app_logger.exception(f"Error in avatar API endpoint '{request.path}'.")
            return jsonify({"error": str(e), "message": "Failed to perform avatar action."}), 500
    return wrapper

@app.route("/api/avatar/state", methods=["GET", "POST"])
def avatar_get_state():
    controller_state = {}
    try:
        api = get_avatar_panel_api()
        if api and hasattr(api, "get_state"):
            raw_state = api.get_state()
            if isinstance(raw_state, dict):
                controller_state = raw_state
    except Exception as e:
        app_logger.debug(f"Avatar controller state unavailable: {e}")
    return jsonify(_avatar_state_payload(controller_state)), 200

@app.route("/api/avatar/mode", methods=["POST"])
def avatar_set_mode():
    data = request.get_json(silent=True) or {}
    mode = _avatar_normalize_mode(data.get("mode", "avatar_2d"))
    controller_result = None
    try:
        api = get_avatar_panel_api()
        if api and hasattr(api, "set_mode"):
            controller_result = api.set_mode(mode)
    except Exception as e:
        controller_result = {"success": False, "error": str(e)}
    state = _avatar_update_state(mode=mode)
    state["controller_result"] = controller_result
    return jsonify(state), 200

@app.route("/api/avatar/emotion", methods=["POST"])
def avatar_set_emotion():
    data = request.get_json(silent=True) or {}
    emotion = str(data.get("emotion", data.get("expression", "neutral")) or "neutral").strip().lower()
    controller_result = None
    try:
        api = get_avatar_panel_api()
        if api and hasattr(api, "set_emotion"):
            controller_result = api.set_emotion(emotion)
    except Exception as e:
        controller_result = {"success": False, "error": str(e)}
    state = _avatar_update_state(emotion=emotion, expression=emotion)
    state["controller_result"] = controller_result
    return jsonify(state), 200

@app.route("/api/avatar/frame", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_get_frame(api):
    width = int(request.args.get("width", 300))
    height = int(request.args.get("height", 300))
    format = request.args.get("format", "base64") # "base64" or "binary" if streaming
    # Consider validating format here
    return api.get_avatar_frame(width, height, format)

@app.route("/api/avatar/lipsync", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_control_lipsync(api):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "start")
    duration = data.get("duration", 0.0)
    if action == "start":
        return api.start_lip_sync(float(duration))
    elif action == "stop":
        return api.stop_lip_sync()
    else:
        return jsonify({"error": "Invalid action for lipsync. Must be 'start' or 'stop'."}), 400

@app.route("/api/avatar/conference/start", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_start(api):
    data = request.get_json(silent=True) or {}
    peer_id = data.get("peer_id", "")
    video = data.get("video", True)
    audio = data.get("audio", True)
    if not peer_id:
        return jsonify({"error": "Peer ID is required to start a conference."}), 400
    return api.start_call(peer_id, video, audio)

@app.route("/api/avatar/conference/answer", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_answer(api):
    data = request.get_json(silent=True) or {}
    peer_id = data.get("peer_id", "")
    if not peer_id:
        return jsonify({"error": "Peer ID is required to answer a conference."}), 400
    return api.answer_call(peer_id)

@app.route("/api/avatar/conference/end", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_end(api):
    return api.end_call()

@app.route("/api/avatar/conference/toggle", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_toggle(api):
    data = request.get_json(silent=True) or {}
    media_type = data.get("type", "video") # "video" or "audio"
    if media_type == "video":
        return api.toggle_call_video()
    elif media_type == "audio":
        return api.toggle_call_audio()
    else:
        return jsonify({"error": "Invalid media type. Must be 'video' or 'audio'."}), 400

@app.route("/api/avatar/conference/info", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_info(api):
    return api.get_call_info()

@app.route("/api/avatar/media/image", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_display_image(api):
    data = request.get_json(silent=True) or {}
    image_path = data.get("path", "")
    if not image_path:
        return jsonify({"error": "Image path is required to display image."}), 400
    return api.display_image(image_path)

@app.route("/api/avatar/media/video", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_display_video(api):
    data = request.get_json(silent=True) or {}
    video_path = data.get("path", "")
    loop = data.get("loop", False)
    if not video_path:
        return jsonify({"error": "Video path is required to display video."}), 400
    return api.display_video(video_path, loop)

@app.route("/api/avatar/media/stop", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_stop_media(api):
    return api.stop_media()

@app.route("/api/avatar/media/info", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_media_info(api):
    return api.get_media_info()

@app.route("/api/avatar/desktop/mirror", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_desktop_mirror(api):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "start")
    if action == "start":
        return api.start_desktop_mirror()
    elif action == "stop":
        return api.stop_desktop_mirror()
    else:
        return jsonify({"error": "Invalid action for desktop mirror. Must be 'start' or 'stop'."}), 400

@app.route("/api/avatar/panel/size", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_set_panel_size(api):
    data = request.get_json(silent=True) or {}
    width = data.get("width", 480)
    height = data.get("height", 360)
    try: # Validate as integers
        width = int(width)
        height = int(height)
    except ValueError:
        return jsonify({"error": "Width and height must be integers."}), 400
    return api.set_panel_size(width, height)

@app.route("/api/avatar/panel/maximize", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_toggle_maximize(api):
    return api.toggle_maximize()

@app.route("/api/avatar/panel/popout", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_toggle_popout(api):
    return api.toggle_popout()

# ---------------- Additional v8.0 API endpoints (merged from app-new.py) ----------------

def get_config_snapshot():
    """Return a small config snapshot that the WebUI can query."""
    try:
        import SarahMemoryGlobals as G
        meta = {}
        meta.setdefault("project_version", getattr(G, "PROJECT_VERSION", PROJECT_VERSION))
        meta.setdefault("author", getattr(G, "AUTHOR", "Brian Lee Baros"))
        meta.setdefault("revision_start_date", getattr(G, "REVISION_START_DATE", ""))
        meta.setdefault("run_mode", getattr(G, "RUN_MODE", "local"))
        meta.setdefault("device_mode", getattr(G, "DEVICE_MODE", "local_agent"))
        meta.setdefault("device_profile", getattr(G, "DEVICE_PROFILE", "Standard"))
        meta.setdefault("safe_mode", getattr(G, "SAFE_MODE", False))
        meta.setdefault("local_only", getattr(G, "LOCAL_ONLY_MODE", False)) # Changed from LOCAL_ONLY for consistency
        meta.setdefault("node_name", getattr(G, "NODE_NAME", "SarahMemory"))
        meta.setdefault("api_root", getattr(G, "API_ROOT", "/api"))
        return meta
    except Exception as e:
        app_logger.warning(f"Error getting config snapshot from SarahMemoryGlobals, falling back: {e}")
        # Minimal fallback identity snapshot if globals are unavailable.
        return {
            "project_version": PROJECT_VERSION,
            "author": "Brian Lee Baros",
            "revision_start_date": "",
            "run_mode": "local",
            "device_mode": "local_agent",
            "device_profile": "Standard",
            "safe_mode": False,
            "local_only": False,
            "node_name": "SarahMemory",
            "api_root": "/api",
        }

@app.route("/api/settings")
def api_settings():
    meta = get_config_snapshot()
    return jsonify({
        "ok": True,
        "settings": meta,
        # WebUI bootstrap hint: the frontend can choose to speak this via its own
        # browser TTS engine. Server-side TTS cannot play in a remote browser.
        "intro": {
            "text": "Hi! I'm Sarah — ready when you are. Try asking me anything.",
            "should_speak": True,
        },
        "ts": time.time(), # Added timestamp for consistency
    })


@app.route("/api/ui/bootstrap", methods=["GET"])
def api_ui_bootstrap():
    """One-call bootstrap for the React/Vite WebUI.

    The WebUI can call this once on page load.
    - Returns identity/config + capability flags.
    - Returns an intro message that the browser can speak.
    - Uses a session cookie to avoid repeating the intro on every refresh.
    """
    meta = get_config_snapshot()

    # Session-based one-time intro flag.
    already = bool(session.get("intro_spoken"))
    if not already:
        session["intro_spoken"] = True

    # Capability detection for the WebUI.
    # NOTE: Do not reference core_speak_text here because the TTS helper block
    # is initialized further down in this file.
    tts_ok = False
    try:
        from SarahMemoryVoice import speak_text as _s
        tts_ok = callable(_s)
    except Exception:
        tts_ok = False
    avatar_ok = True
    try:
        import SarahMemoryAvatar as _A
        avatar_ok = True
    except Exception:
        avatar_ok = False

    return jsonify({
        "ok": True,
        "settings": meta,
        "capabilities": {
            "tts_server": bool(tts_ok),
            "avatar": bool(avatar_ok),
            "media_jobs": True,
        },
        "intro": {
            "text": "Hi! I'm Sarah — ready when you are. Try asking me anything.",
            "should_speak": (not already),
        },
        "ts": time.time(),
    }), 200

# --------------------------- TTS / VOICE HELPERS --------------------------

core_speak_text = None
try:
    from SarahMemoryVoice import speak_text as core_speak_text_func
    core_speak_text = core_speak_text_func
except ImportError:
    app_logger.info("SarahMemoryVoice module not found for TTS.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryVoice.speak_text: {e}", exc_info=True)


@app.route("/api/tts/speak", methods=['POST'])
def api_tts_speak():
    """
    Minimal TTS bridge for the Web UI.
    Expected JSON:
      { "text": "...", "voice": "default", "rate": 1.0 }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = (data.get("voice") or "default").strip()
    rate_str = data.get("rate") # Keep as string/int for initial parsing

    if not text:
        return jsonify({"ok": False, "error": "Missing text for TTS."}), 400

    try:
        rate = float(rate_str) if rate_str is not None else 1.0
        if not (0.1 <= rate <= 5.0): # Example range, adjust as needed
             return jsonify({"ok": False, "error": "Speech rate must be between 0.1 and 5.0."}), 400
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid speech rate format."}), 400


    if core_speak_text is None:
        return jsonify({
            "ok": False,
            "error": "TTS engine not available on this server.",
        }), 501

    try:
        # Assuming core_speak_text can handle these parameters
        core_speak_text(text, voice_name=voice, rate=rate)
        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.exception(f"Error during TTS speak request for text: '{text}...'")
        return jsonify({"ok": False, "error": f"Failed to speak text: {e}"}), 500

@app.route("/api/logs/events")
def api_logs_events():
    """
    Return the last N lines of api_events.log so the Web UI can show them.
    This reads the log file created by Flask's basic logging, not `log_event()`.
    """
    N = int(request.args.get("limit", 200)) # Limit to last N lines
    path = os.path.join(LOGS_DIR, "api_events.log") # Expecting a JSON log file

    if not os.path.exists(path):
        return jsonify({"ok": True, "events": [], "message":f"Log file {os.path.basename(path)} not found."}), 200

    events = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Read all lines and then slice for performance for large files, or use deque.
            # For simplicity, reading all and slicing.
            lines = f.readlines()
            # If the file is very large, consider reading from end-of-file for performance
            # Or use a more sophisticated log reader.

            # This is a bit slow for very large files, but robust for typical usage
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # If a line isn't valid JSON, still append it as raw to show problem
                    events.append({"raw": line, "error": "Invalid JSON format in log line"})
                except Exception as e:
                    app_logger.warning(f"Error parsing log line: {e} | Line: {line}")
                    events.append({"raw": line, "error": f"Parsing error: {str(e)}"})
        return jsonify({"ok": True, "events": events}), 200
    except IOError as e:
        app_logger.error(f"Error reading API events log file {path}: {e}")
        return jsonify({"ok": False, "error": f"Failed to read API events log: {e}"}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error when fetching API events log.")
        return jsonify({"ok": False, "error": str(e)}), 500


# --------------------------- SIMPLE PING ----------------------------------

@app.route("/api/ping")
def api_ping():
    ok, notes, main_running = _perform_health_checks() # Include health check in ping
    return jsonify({
        "ok": True,
        "pong": True,
        "ts": time.time(),
        "version": PROJECT_VERSION,
        "health_status": "ok" if ok else "warning",
        "running": True,
        "main_running": main_running,
    })
# =========================== LOCAL RUNTIME CONTROL ===========================
# one-per-process shutdown latch
if "SM_SHUTDOWN_EVENT" not in globals():
    SM_SHUTDOWN_EVENT = threading.Event()

def _is_localhost_request() -> bool:
    """True only for local desktop installs (never true for PythonAnywhere / public)."""
    try:
        host = (request.host or "").split(":", 1)[0].strip().lower()
        if host in ("127.0.0.1", "localhost"):
            return True
    except Exception:
        pass

    # allow LAN desktop installs if you explicitly run them (optional)
    try:
        ra = (request.remote_addr or "").strip()
        if ra in ("127.0.0.1", "::1"):
            return True
    except Exception:
        pass

    return False

def _is_cloud_request() -> bool:
    """Best-effort: treat ai.* / api.* as cloud, and never honor shutdown there."""
    try:
        host = (request.host or "").split(":", 1)[0].strip().lower()
        if host.startswith("ai.") or host.startswith("api."):
            return True
    except Exception:
        pass
    return False

def _request_main_shutdown(reason: str = "ui_exit") -> dict:
    """
    MODE B contract:
    - Local: set a shutdown flag + persist state so SarahMemoryMain/Synapes/SelfAware can stop.
    - Cloud: NOOP (never kill the shared server).
    """
    # Always persist a flag that the launcher can watch.
    try:
        state = load_state() or {}
        if not isinstance(state, dict):
            state = {}
        state["shutdown_requested"] = True
        state["shutdown_reason"] = str(reason)
        state["shutdown_ts"] = time.time()
        save_state(state)
    except Exception:
        pass

    # In-process latch (for any background workers living inside app.py itself)
    try:
        SM_SHUTDOWN_EVENT.set()
    except Exception:
        pass

    # If running local desktop with a tracked MAIN_PID, request termination by signaling the PID.
    # IMPORTANT: we DO NOT do this in cloud mode.
    killed = False
    pid = None
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r", encoding="utf-8") as f:
                pid_txt = f.read().strip()
            if pid_txt.isdigit():
                pid = int(pid_txt)
    except Exception:
        pid = None

    # Soft signal: write the flag; your Main loop should observe it and shutdown gracefully.
    # Hard signal is optional; leave commented unless you want it.
    #
    # try:
    #     if pid and pid > 1:
    #         os.kill(pid, signal.SIGTERM)
    #         killed = True
    # except Exception:
    #     killed = False

    return {"ok": True, "shutdown_requested": True, "pid": pid, "hard_signal_sent": killed}

@app.get("/api/local/brain")
def api_local_brain():
    """
    MODE B:
    - Local desktop UI can call this to decide whether to keep brain loops running.
    - Cloud always returns enabled=False so a mobile browser close never kills the service.
    """
    if _is_cloud_request():
        return jsonify({"ok": True, "mode": "cloud", "enabled": False, "reason": "shared_service"}), 200

    if not _is_localhost_request():
        return jsonify({"ok": False, "enabled": False, "error": "forbidden_non_local"}), 403

    # If shutdown requested, tell UI/launcher to stop Synapes/SelfAware loops.
    try:
        state = load_state() or {}
        shutting_down = bool(state.get("shutdown_requested"))
    except Exception:
        shutting_down = bool(SM_SHUTDOWN_EVENT.is_set())

    return jsonify({"ok": True, "mode": "local", "enabled": (not shutting_down), "shutdown": shutting_down}), 200

@app.post("/api/ui/exit")
def api_ui_exit():
    """
    MODE B:
    - Local desktop only: called when the LOCAL WebUI closes to trigger coordinated shutdown.
    - Cloud: NOOP (returns ok, but does not shutdown anything).
    """
    # Cloud safety: never shutdown shared server
    if _is_cloud_request():
        return jsonify({"ok": True, "mode": "cloud", "noop": True}), 200

    # Local safety: only accept from localhost installs
    if not _is_localhost_request():
        return jsonify({"ok": False, "error": "forbidden_non_local"}), 403

    payload = {}
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    reason = (payload.get("reason") or "ui_exit").strip()
    result = _request_main_shutdown(reason=reason)

    return jsonify({"ok": True, "mode": "local", **result}), 200

# ========================= LOCAL RUNTIME CONTROL ===========================

@app.route("/api/ledger/top-nodes")
def api_top_nodes():
    limit_str = request.args.get("limit", "10")
    try:
        limit = int(limit_str)
        if not (1 <= limit <= 100): # Reasonable limit
            raise ValueError("Limit must be between 1 and 100.")
    except ValueError as e:
        return jsonify({"ok": False, "error": f"Invalid limit parameter: {e}"}), 400

    leaders = read_top_nodes(limit=limit)
    return jsonify({"ok": True, "leaders": leaders}), 200

# --------------------------- SIMPLE SETTINGS SNAPSHOT ---------------------

@app.route("/api/download/<path:filename>")
def api_download(filename):
    """Download a file that lives under DATA_DIR (safe path enforced)."""
    if not filename:
        return jsonify({"ok": False, "error": "Missing filename"}), 400

    # Normalize and enforce containment within DATA_DIR
    try:
        base = os.path.abspath(DATA_DIR)
        full_path = os.path.abspath(os.path.join(base, filename))
        common_path = os.path.commonpath([base, full_path])
    except Exception:
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    if common_path != base:
        app_logger.warning("Attempted download outside DATA_DIR: %s", full_path)
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return jsonify({"ok": False, "error": "File not found"}), 404

    try:
        # Use send_file so nested paths are fine after the containment check.
        return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))
    except TypeError:
        # Flask <2.0 compatibility: download_name not supported
        return send_file(full_path, as_attachment=True)


# -----------------------------------------------------------------------------
# Optional dependency shim: bleach
# -----------------------------------------------------------------------------
# appsys.py relies on `bleach.clean()` for HTML sanitization. On some minimal
# installs, `bleach` may not be present. To keep APPSYS online without forcing
# extra installs, we provide a conservative fallback implementation.
try:
    import bleach  # type: ignore
except Exception:  # pragma: no cover
    try:
        import types as _types
        import re as _re
        import html as _html
        _bleach_mod = _types.ModuleType("bleach")

        def _fallback_clean(text, tags=None, attributes=None, strip=False, strip_comments=True, **kwargs):
            try:
                s = "" if text is None else str(text)
            except Exception:
                s = ""
            # Remove HTML comments (basic)
            if strip_comments:
                s = _re.sub(r"<!--.*?-->", "", s, flags=_re.DOTALL)
            if strip:
                # Drop all tags
                s = _re.sub(r"<[^>]+>", "", s)
                return s
            # Escape everything (safest default)
            return _html.escape(s, quote=True)

        _bleach_mod.clean = _fallback_clean  # type: ignore
        import sys as _sys
        _sys.modules["bleach"] = _bleach_mod
    except Exception:
        # If even the shim fails, appsys import will raise and be logged.
        pass

# --- v8 local system endpoints (Files / OS utilities) ---
def _ensure_api_import_paths():
    """Make api/server modules importable in all launch modes."""
    try:
        server_dir = os.path.abspath(os.path.dirname(__file__))      # .../api/server
        api_dir = os.path.abspath(os.path.join(server_dir, ".."))   # .../api
        proj_dir = os.path.abspath(os.path.join(api_dir, ".."))     # project root
        for p in (server_dir, api_dir, proj_dir):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass

try:
    _ensure_api_import_paths()
    try:
        # When imported as a package (e.g., `from api.server.app import app`)
        from . import appsys as _appsys  # type: ignore
    except Exception:
        # When executed with api/server on sys.path (e.g., `python api/server/app.py`)
        import appsys as _appsys  # type: ignore

    _appsys.init_app(app)
except Exception as _e:
    try:
        app_logger.error(f"appsys init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- V8 CANVAS STUDIO SUITE ./api/server/appmedia.py mount ---
try:
    _ensure_api_import_paths()
    try:
        from . import appmedia as _appmedia  # type: ignore
    except Exception:
        import appmedia as _appmedia  # type: ignore

    _appmedia.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    app_logger.info("appmedia mounted: /api/media/*")
except Exception as e:
    app_logger.warning(f"appmedia not mounted: {e}")


# --- v8 MCP broker endpoints (SarahNet one-way broker) ---
try:
    _ensure_api_import_paths()
    try:
        from . import appnet as _appnet  # type: ignore
    except Exception:
        import appnet as _appnet  # type: ignore

    _appnet.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
except Exception as _e:
    try:
        app_logger.error(f"appnet init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 appnet2 endpoints (SarahNet Bravo: DNS/Overlay Tunnel/Identity) ---
try:
    # If your app.py has this helper, use it; otherwise no-op.
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appnet2 as _appnet2  # type: ignore
    except Exception:
        import appnet2 as _appnet2  # type: ignore

    _appnet2.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appnet2 mounted: /api/net2/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appnet2 init failed: {_e}", exc_info=True)
    except Exception:
        pass
# --- v8 appstore endpoints (SarahMemory Power StoreFront) ---
try:
    # If your app.py has this helper, use it; otherwise no-op.
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appstore as _appstore  # type: ignore
    except Exception:
        import appstore as _appstore  # type: ignore

    _appstore.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appstore mounted: /api/store/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appstore init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 appcomm endpoints (communications domain) ---
try:
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appcomm as _appcomm  # type: ignore
    except Exception:
        import appcomm as _appcomm  # type: ignore

    _appcomm.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appcomm mounted: /api/comm/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appcomm init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 appdrivers endpoints (governed hardware / driver domain) ---
try:
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appdrivers as _appdrivers  # type: ignore
    except Exception:
        import appdrivers as _appdrivers  # type: ignore

    _appdrivers.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appdrivers mounted: /api/drivers/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appdrivers init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 appvision endpoints (Governed Vision / MSDC camera bridge) ---
try:
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appvision as _appvision  # type: ignore
    except Exception:
        import appvision as _appvision  # type: ignore

    _appvision.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appvision mounted: /api/vision/policy, /api/vision/devices, /api/vision/analyze, /api/vision/local/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appvision init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 appdevbridge endpoints (Developer Bridge / ChatGPT-assisted packet lane) ---
try:
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appdevbridge as _appdevbridge  # type: ignore
    except Exception:
        import appdevbridge as _appdevbridge  # type: ignore

    _appdevbridge.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appdevbridge mounted: /api/devbridge/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appdevbridge init failed: {_e}", exc_info=True)
    except Exception:
        pass


# --- v8 appself endpoints (SelfAware / CognitiveSelf fact-ticket API) ---
try:
    try:
        _ensure_api_import_paths()  # type: ignore[name-defined]
    except Exception:
        pass

    try:
        from . import appself as _appself  # type: ignore
    except Exception:
        import appself as _appself  # type: ignore

    _appself.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
    try:
        app_logger.info("appself mounted: /api/self/*")
    except Exception:
        pass

except Exception as _e:
    try:
        app_logger.error(f"appself init failed: {_e}", exc_info=True)
    except Exception:
        pass

# ============================================================================
# UI Event Speech Support (Opt-in)
# ============================================================================

@app.post("/api/ui/event")
def api_ui_event():
    """
    Programmatic UI event trigger for speech/notifications.
    Body: {"event": "panel_open", "detail": "Files", "speak": "Opening File Manager"}
    """
    try:
        data = request.get_json(silent=True) or {}
        event = (data.get("event") or "unknown").strip() or "unknown"
        detail = (data.get("detail") or "").strip()
        speak = (data.get("speak") or "").strip()

        app_logger.info(f"UI event: {event} | {detail}")

        if speak and os.getenv("SARAH_UI_SPEECH_LOCAL", "0") == "1":
            try:
                from SarahMemoryVoice import speak_text  # type: ignore
                speak_text(speak, blocking=False)
            except Exception:
                pass

        return jsonify({"ok": True, "event": event}), 200
    except Exception as e:
        app_logger.error(f"UI event failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500




# =============================================================================
# SARAH_REM_BRIDGE_ROUTES_V2
# Robust REM + DL Engine visibility bridge for WebUI/DLEngineScreen.tsx.
# =============================================================================
_REM_BRIDGE_CACHE = None
_REM_BRIDGE_LOCK = threading.RLock()


class _REMBridgeAdapter:
    """Adapter around either the UAC module bridge or a controller instance."""
    def __init__(self, target):
        self._target = target

    def get_rem_status(self):
        return self._target.get_rem_status()

    def get_rem_report(self, limit: int = 5):
        return self._target.get_rem_report(limit=limit)

    def start_rem_sleep(self, reason: str = "idle", force: bool = False):
        try:
            return self._target.start_rem_sleep(reason=reason, force=force)
        except TypeError:
            # Legacy controller/module signature did not accept force.
            if force and "manual" not in str(reason).lower():
                reason = f"manual_force:{reason}"
            return self._target.start_rem_sleep(reason=reason)

    def stop_rem_sleep(self, reason: str = "manual"):
        return self._target.stop_rem_sleep(reason=reason)


def _get_rem_bridge():
    """Return UnifiedAvatarController REM bridge. Never raises to Flask.

    Accepts either:
    - module-level functions, or
    - get_unified_avatar_controller(), or
    - UnifiedAvatarController() class instance.
    """
    global _REM_BRIDGE_CACHE
    with _REM_BRIDGE_LOCK:
        if _REM_BRIDGE_CACHE is not None:
            return _REM_BRIDGE_CACHE
        try:
            import UnifiedAvatarController as _uac  # type: ignore
            required = ("get_rem_status", "get_rem_report", "start_rem_sleep", "stop_rem_sleep")
            if all(hasattr(_uac, name) for name in required):
                _REM_BRIDGE_CACHE = _REMBridgeAdapter(_uac)
                return _REM_BRIDGE_CACHE

            get_ctrl = getattr(_uac, "get_unified_avatar_controller", None)
            if callable(get_ctrl):
                ctrl = get_ctrl()
                if all(hasattr(ctrl, name) for name in required):
                    _REM_BRIDGE_CACHE = _REMBridgeAdapter(ctrl)
                    return _REM_BRIDGE_CACHE

            cls = getattr(_uac, "UnifiedAvatarController", None)
            if callable(cls):
                ctrl = cls()
                if all(hasattr(ctrl, name) for name in required):
                    _REM_BRIDGE_CACHE = _REMBridgeAdapter(ctrl)
                    return _REM_BRIDGE_CACHE

            app_logger.error("UnifiedAvatarController imported but no usable REM bridge surface was found.")
            return None
        except Exception as exc:
            app_logger.error(f"UnifiedAvatarController REM bridge import failed: {exc}", exc_info=True)
            _REM_BRIDGE_CACHE = None
            return None


@app.route("/api/avatar/rem/status", methods=["GET"])
def api_avatar_rem_status():
    bridge = _get_rem_bridge()
    if not bridge:
        return jsonify({"ok": False, "error": "UnifiedAvatarController REM bridge unavailable."}), 503
    try:
        return jsonify({"ok": True, "rem": bridge.get_rem_status()}), 200
    except Exception as exc:
        app_logger.error(f"REM status failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/avatar/rem/report", methods=["GET"])
def api_avatar_rem_report():
    bridge = _get_rem_bridge()
    if not bridge:
        return jsonify({"ok": False, "error": "UnifiedAvatarController REM bridge unavailable."}), 503
    try:
        limit = int(request.args.get("limit", "5") or 5)
        return jsonify(bridge.get_rem_report(limit=limit)), 200
    except Exception as exc:
        app_logger.error(f"REM report failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/avatar/rem/start", methods=["POST"])
def api_avatar_rem_start():
    bridge = _get_rem_bridge()
    if not bridge:
        return jsonify({"ok": False, "error": "UnifiedAvatarController REM bridge unavailable."}), 503
    try:
        data = request.get_json(silent=True) or {}
        reason = str(data.get("reason") or "manual_force_sleep")
        force = bool(data.get("force", True))
        result = bridge.start_rem_sleep(reason=reason, force=force)
        return jsonify(result), (200 if result.get("ok") else 409)
    except Exception as exc:
        app_logger.error(f"REM start failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/avatar/rem/stop", methods=["POST"])
def api_avatar_rem_stop():
    bridge = _get_rem_bridge()
    if not bridge:
        return jsonify({"ok": False, "error": "UnifiedAvatarController REM bridge unavailable."}), 503
    try:
        data = request.get_json(silent=True) or {}
        reason = str(data.get("reason") or "manual_wake")
        return jsonify(bridge.stop_rem_sleep(reason=reason)), 200
    except Exception as exc:
        app_logger.error(f"REM stop failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


def _rem_dlengine_derive_from_report(report_payload: dict | None = None) -> tuple[list, list]:
    report_payload = report_payload or {}

    def _is_cycle_like(obj) -> bool:
        return isinstance(obj, dict) and (
            bool(obj.get("dreams"))
            or bool(obj.get("results"))
            or bool(obj.get("subprocesses"))
            or "cycle_number" in obj
        )

    reports = []
    if isinstance(report_payload.get("reports"), list):
        reports = list(report_payload.get("reports") or [])
    if isinstance(report_payload.get("last_report"), dict):
        last_report = report_payload.get("last_report")
        if last_report not in reports:
            reports.append(last_report)
    status_last = ((report_payload.get("status") or {}) if isinstance(report_payload.get("status"), dict) else {}).get("last_report")
    if isinstance(status_last, dict) and status_last not in reports:
        reports.append(status_last)

    thoughts = []
    subjects = []
    for rep in reports[-10:]:
        cycles = rep.get("cycles") if isinstance(rep, dict) else []
        if not isinstance(cycles, list) or not cycles:
            cycles = [rep] if _is_cycle_like(rep) else []
        for cycle in cycles:
            cycle_no = cycle.get("cycle_number", "?")
            subprocesses = cycle.get("subprocesses") or {}
            for lane_name, lane in subprocesses.items():
                lane = lane if isinstance(lane, dict) else {"value": lane}
                ok = bool(lane.get("ok"))
                level = "success" if ok else ("warning" if lane.get("degraded") or lane.get("skipped") else "error")
                thoughts.append({
                    "id": f"rem-lane-{cycle_no}-{lane_name}",
                    "ts": lane.get("ts") or rep.get("finished_at") or rep.get("started_at") or datetime.now().isoformat(),
                    "title": f"REM lane: {lane_name}",
                    "content": str(lane.get("summary") or lane.get("reason") or lane.get("error") or f"{lane_name} lane observed."),
                    "source": f"rem.{lane_name}",
                    "level": level,
                    "tags": ["rem", "lane", lane_name],
                })
            for idx, dream in enumerate(cycle.get("dreams") or []):
                subject_id = str(dream.get("dream_id") or f"dream-{cycle_no}-{idx}")
                subjects.append({
                    "id": subject_id,
                    "title": str(dream.get("title") or "REM dream candidate"),
                    "summary": str((dream.get("proposed_action") or {}).get("description") or dream.get("rationale") or dream.get("category") or "REM candidate generated."),
                    "source": "rem.cognitive_thinker",
                    "stage": "observed",
                    "confidence": 64,
                    "risk": 28 if str(dream.get("risk_tier", "low")).lower() == "low" else 65,
                    "sandboxRecommended": True,
                    "tags": ["rem", "dream", str(dream.get("category") or "self_study")],
                    "updatedAt": rep.get("finished_at") or datetime.now().isoformat(),
                })
            for idx, result in enumerate(cycle.get("results") or []):
                dream = result.get("dream") or {}
                decision = str(result.get("decision") or "review")
                level = "success" if "AUTO" in decision.upper() or "ALLOW" in decision.upper() else "warning" if "STAGE" in decision.upper() else "error" if "REJECT" in decision.upper() or "DENY" in decision.upper() else "thinking"
                thoughts.append({
                    "id": f"rem-result-{cycle_no}-{idx}",
                    "ts": rep.get("finished_at") or datetime.now().isoformat(),
                    "title": f"REM result: {dream.get('title') or 'candidate'}",
                    "content": f"Decision: {decision}. Sandbox: {(result.get('sandbox') or {}).get('passed', 'n/a')}. Assurance: {(result.get('assurance') or {}).get('decision', 'n/a')}.",
                    "source": "rem.assurance",
                    "level": level,
                    "tags": ["rem", "decision", decision.lower()],
                })
    return thoughts[:200], subjects[:200]


@app.route("/api/dlengine/status", methods=["GET"])
def api_dlengine_status():
    try:
        dl_status = None
        try:
            import SarahMemoryDL as _dl  # type: ignore
            fn = getattr(_dl, "get_dlengine_status", None)
            if callable(fn):
                dl_status = fn()
        except Exception as exc:
            app_logger.debug(f"SarahMemoryDL status unavailable: {exc}")
        if not isinstance(dl_status, dict):
            dl_status = {"ok": False, "stats": {}, "jobs": [], "model": {}}
        bridge = _get_rem_bridge()
        rem_status = bridge.get_rem_status() if bridge else {"enabled": False, "phase": "unavailable"}
        rem_report = bridge.get_rem_report(limit=5) if bridge else {"reports": [], "summary": {}}
        stats = dict(dl_status.get("stats") or {})
        model = dict(dl_status.get("model") or {})
        summary = dict(rem_report.get("summary") or {})
        stats.setdefault("thinkingLoad", 100 if rem_status.get("running") else 0)
        stats.setdefault("thinking_load", stats.get("thinkingLoad", 0))
        stats.setdefault("subjectsOpen", int(summary.get("dreams") or 0))
        stats.setdefault("subjects_open", int(summary.get("dreams") or 0))
        return jsonify({
            "ok": True,
            "stats": stats,
            "jobs": dl_status.get("jobs") or [],
            "model": model,
            "rem": rem_status,
            "rem_summary": summary,
            "runtime": dl_status.get("runtime") or {},
            "controls": dl_status.get("controls") or {},
            "weights": dl_status.get("weights") or {},
            "state": dl_status.get("state") or {},
        }), 200
    except Exception as exc:
        app_logger.error(f"DL Engine status failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/thoughts", methods=["GET"])
def api_dlengine_thoughts():
    try:
        bridge = _get_rem_bridge()
        report = bridge.get_rem_report(limit=10) if bridge else {"reports": []}
        thoughts, _subjects = _rem_dlengine_derive_from_report(report)
        return jsonify({"ok": True, "thoughts": thoughts}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "thoughts": []}), 500


@app.route("/api/dlengine/subjects", methods=["GET"])
def api_dlengine_subjects():
    try:
        bridge = _get_rem_bridge()
        report = bridge.get_rem_report(limit=10) if bridge else {"reports": []}
        _thoughts, subjects = _rem_dlengine_derive_from_report(report)
        return jsonify({"ok": True, "subjects": subjects}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "subjects": []}), 500


@app.route("/api/dlengine/subject_action", methods=["POST"])
@app.route("/api/dlengine/ticket_action", methods=["POST"])
def api_dlengine_subject_action():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "accepted": True, "subject": data, "ts": datetime.now().isoformat()}), 200


def _dlengine_module():
    try:
        import SarahMemoryDL as _dl  # type: ignore
        return _dl
    except Exception as exc:
        app_logger.error(f"SarahMemoryDL bridge import failed: {exc}", exc_info=True)
        return None


@app.route("/api/dlengine/controls", methods=["GET", "POST"])
@app.route("/api/dlengine/finetune/config", methods=["POST"])
def api_dlengine_controls():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    if request.method == "GET":
        try:
            if dl and hasattr(dl, "get_dlengine_runtime_state"):
                return jsonify({"ok": True, "state": dl.get_dlengine_runtime_state()}), 200
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, "state": load_state().get("DLENGINE_CONTROLS", {})}), 200
    try:
        result = None
        if dl and hasattr(dl, "set_dlengine_controls"):
            controls_payload = data.get("controls") if isinstance(data.get("controls"), dict) else data
            result = dl.set_dlengine_controls(controls_payload, source="flask:/api/dlengine/controls")
        try:
            save_state("DLENGINE_CONTROLS", data)
        except Exception:
            pass
        return jsonify(result or {"ok": True, "saved": True, "controls": data, "ts": datetime.now().isoformat()}), 200
    except Exception as exc:
        app_logger.error(f"DL Engine controls failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/mode", methods=["GET", "POST"])
@app.route("/api/dlengine/control", methods=["GET", "POST"])
def api_dlengine_mode():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    if request.method == "GET":
        try:
            runtime_state = {}
            status_payload = {}
            if dl and hasattr(dl, "get_dlengine_runtime_state"):
                runtime_state = dl.get_dlengine_runtime_state() or {}
            if dl and hasattr(dl, "get_dlengine_status"):
                status_payload = dl.get_dlengine_status() or {}
            bridge = _get_rem_bridge()
            rem_status = bridge.get_rem_status() if bridge else {"enabled": False, "phase": "unavailable", "running": False}
            mode = str(
                runtime_state.get("mode")
                or (status_payload.get("runtime") or {}).get("mode")
                or load_state().get("DLENGINE_MODE")
                or "auto"
            )
            return jsonify({
                "ok": True,
                "mode": mode,
                "manual": mode == "manual",
                "paused": mode == "paused",
                "runtime_mode": mode,
                "deep_learning_enabled": mode != "paused",
                "controls": runtime_state.get("controls") or status_payload.get("controls") or {},
                "weights": runtime_state.get("weights") or status_payload.get("weights") or {},
                "rem_sleep_running": bool(rem_status.get("running")),
                "rem_phase": rem_status.get("phase"),
                "rem": rem_status,
                "state": runtime_state,
                "status": status_payload,
                "ts": datetime.now().isoformat(),
            }), 200
        except Exception as exc:
            app_logger.error(f"DL Engine mode GET failed: {exc}", exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    mode = data.get("mode") or data.get("state") or "auto"
    try:
        if dl and hasattr(dl, "set_dlengine_mode"):
            return jsonify(dl.set_dlengine_mode(mode, source="flask:/api/dlengine/mode", payload=data)), 200
        save_state("DLENGINE_MODE", str(mode))
        return jsonify({"ok": True, "saved": True, "mode": str(mode)}), 200
    except Exception as exc:
        app_logger.error(f"DL Engine mode failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/start", methods=["POST"])
def api_dlengine_start():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    try:
        if dl and hasattr(dl, "start_dlengine_manual"):
            return jsonify(dl.start_dlengine_manual(data)), 200
        return jsonify({"ok": True, "mode": "manual", "saved": True}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/stop", methods=["POST"])
def api_dlengine_stop():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    try:
        if dl and hasattr(dl, "pause_dlengine"):
            return jsonify(dl.pause_dlengine(data)), 200
        return jsonify({"ok": True, "mode": "paused", "saved": True}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/auto", methods=["POST"])
def api_dlengine_auto():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    try:
        if dl and hasattr(dl, "set_dlengine_auto"):
            return jsonify(dl.set_dlengine_auto(data)), 200
        return jsonify({"ok": True, "mode": "auto", "saved": True}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/weights", methods=["GET", "POST"])
@app.route("/api/dlengine/tuning_weights", methods=["GET", "POST"])
def api_dlengine_weights():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()

    if request.method == "GET":
        try:
            category = str(request.args.get("category") or "reasoning")
            model_id = str(request.args.get("model_id") or request.args.get("id") or "")
            if dl and hasattr(dl, "get_model_weight_profile"):
                return jsonify(dl.get_model_weight_profile(category=category, model_id=model_id, refresh_models=False)), 200
            return jsonify({
                "ok": True,
                "category": category,
                "model_id": model_id,
                "weights": load_state().get("DLENGINE_WEIGHTS", {}),
                "raw_tensor_edit": False,
            }), 200
        except Exception as exc:
            app_logger.error(f"DL Engine weights GET failed: {exc}", exc_info=True)
            return jsonify({"ok": False, "error": str(exc)}), 500

    weights = data.get("weights") if isinstance(data.get("weights"), dict) else data
    category = str(data.get("category") or data.get("model_category") or "reasoning")
    model_id = str(data.get("model_id") or data.get("id") or "")
    dl_context = data.get("context") if isinstance(data.get("context"), dict) else data

    try:
        if dl and hasattr(dl, "set_dlengine_weights"):
            try:
                return jsonify(dl.set_dlengine_weights(
                    weights,
                    source="flask:/api/dlengine/weights",
                    category=category,
                    model_id=model_id,
                    context=dl_context,
                )), 200
            except TypeError:
                return jsonify(dl.set_dlengine_weights(weights, source="flask:/api/dlengine/weights")), 200
        save_state("DLENGINE_WEIGHTS", weights)
        return jsonify({
            "ok": True,
            "saved": True,
            "category": category,
            "model_id": model_id,
            "weights": weights,
            "raw_tensor_edit": False,
        }), 200
    except Exception as exc:
        app_logger.error(f"DL Engine weights failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/dlengine/weights/reset", methods=["POST"])
def api_dlengine_weights_reset():
    data = request.get_json(silent=True) or {}
    dl = _dlengine_module()
    category = str(data.get("category") or data.get("model_category") or "reasoning")
    model_id = str(data.get("model_id") or data.get("id") or "")
    try:
        if dl and hasattr(dl, "reset_model_weight_profile"):
            return jsonify(dl.reset_model_weight_profile(category=category, model_id=model_id, source="flask:/api/dlengine/weights/reset")), 200
        default_weights = {
            "reasoning": 65,
            "coding": 55,
            "memory": 60,
            "research": 55,
            "creativity": 45,
            "safety": 90,
            "autonomy": 35,
            "precision": 70,
            "speed": 50,
        }
        save_state("DLENGINE_WEIGHTS", default_weights)
        return jsonify({"ok": True, "saved": True, "category": category, "model_id": model_id, "weights": default_weights, "raw_tensor_edit": False}), 200
    except Exception as exc:
        app_logger.error(f"DL Engine weights reset failed: {exc}", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500


# --- Terminal API (DEVELOPERSMODE gated by SarahMemoryTerminal) ---
from flask import request, jsonify
import SarahMemoryTerminal as smterm

@app.get("/api/terminal/status")
def api_terminal_status():
    payload = {
        "session_id": request.args.get("session_id", ""),
    }
    result = smterm.terminal_api_status(payload, caller="Flask:/api/terminal/status")
    return jsonify(result), 200

@app.post("/api/terminal/execute")
def api_terminal_execute():
    payload = request.get_json(silent=True) or {}
    result = smterm.terminal_api_execute(payload, caller="Flask:/api/terminal/execute")
    return jsonify(result), (200 if result.get("ok") else 403 if result.get("blocked") else 400)


# =============================================================================
# SM V8.0 Cognitive Living Loop / Emergency Instinct API
# =============================================================================
# These endpoints expose the distributed Cognitive Living Loop and Emergency
# Instinct governance surface. They do not directly actuate hardware; physical
# action still requires SMGET/OperatorCore/MSDC dispatch.
# =============================================================================

@app.get("/api/cognitive/living/status")
def api_cognitive_living_status():
    try:
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.cognitive_living_loop_status()
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.living.status"}), 500


@app.post("/api/cognitive/living/tick")
def api_cognitive_living_tick():
    try:
        payload = request.get_json(silent=True) or {}
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.run_cognitive_living_tick(payload)
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.living.tick"}), 500


@app.post("/api/cognitive/living/start")
def api_cognitive_living_start():
    try:
        payload = request.get_json(silent=True) or {}
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        interval = payload.get("interval_seconds", payload.get("interval"))
        result = _CogServices.start_cognitive_living_loop(
            str(payload.get("reason") or "api_start"),
            interval_seconds=interval,
            daemon=True,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.living.start"}), 500


@app.post("/api/cognitive/living/stop")
def api_cognitive_living_stop():
    try:
        payload = request.get_json(silent=True) or {}
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.stop_cognitive_living_loop(str(payload.get("reason") or "api_stop"))
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.living.stop"}), 500


@app.post("/api/cognitive/instinct/evaluate")
def api_cognitive_instinct_evaluate():
    try:
        payload = request.get_json(silent=True) or {}
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.evaluate_emergency_instinct(payload, caller="Flask:/api/cognitive/instinct/evaluate")
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.instinct.evaluate"}), 500


@app.post("/api/cognitive/instinct/trigger")
def api_cognitive_instinct_trigger():
    try:
        payload = request.get_json(silent=True) or {}
        execute = bool(payload.get("execute", False))
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.run_emergency_instinct(payload, execute=execute, caller="Flask:/api/cognitive/instinct/trigger")
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.instinct.trigger"}), 500


@app.get("/api/cognitive/instinct/logs")
def api_cognitive_instinct_logs():
    try:
        limit = int(request.args.get("limit", "25") or 25)
        incident_id = str(request.args.get("incident_id", "") or "")
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.list_emergency_instinct_logs(limit=limit, incident_id=incident_id)
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "api.cognitive.instinct.logs"}), 500


def _start_autonomous_services():
    try:
        import SarahMemoryGlobals as config
        _neosky = bool(getattr(config, "NEOSKYMATRIX", False))
        _dev = bool(getattr(config, "DEVELOPERSMODE", False))

        if _neosky and _dev:
            import threading
            import SarahMemorySelfAware as _SMA
            if hasattr(_SMA, "run_autonomous_loop"):
                t = threading.Thread(
                    target=_SMA.run_autonomous_loop,
                    name="SM_SelfAware",
                    daemon=True
                )
                t.start()
                app_logger.warning("SelfAware ARMED (API Mode).")
    except Exception as e:
        app_logger.error(f"Autonomous init failed: {e}", exc_info=True)

_start_autonomous_services()
try:
    _vr_ensure_watcher_started()
except Exception:
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    app_logger.info(f"Starting SarahMemory Flask API server on http://0.0.0.0:{port}")
    # Initializing app.config with default values for toggles
    app.config.setdefault("CAMERA_ON", False)
    app.config.setdefault("MIC_ON", False)
    app.config.setdefault("VOICE_OUTPUT_ON", True)
    app.config.setdefault("TELECOM_ENABLED", False)  # For telecom stateub

    # In development, use debug=True for reloader and debugger
    # In production, use a WSGI server like Gunicorn/uWSGI
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    if debug_mode:
        app_logger.warning("Running in DEBUG mode. Do NOT use in production!")

    app.run(host="0.0.0.0", port=port, debug=debug_mode)

# ====================================================================
# END OF app.py v9.0.0
# ====================================================================
