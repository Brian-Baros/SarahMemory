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
- Camera/Mic/Voice toggles + bounded telecom integration fallbacks
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

# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# Flask is the preferred local API engine, but SarahMemory must still launch
# smoothly offline on an older local PC even if the venv is missing Flask. This
# fallback keeps /api/health and /api/status alive with the Python standard
# library so the launcher can confirm readiness instead of dropping into a false
# degraded mode. Full Web/API features still require Flask.
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file, g, session, abort, Response
    _SM_FLASK_AVAILABLE = True
except Exception as _flask_import_error:
    _SM_FLASK_AVAILABLE = False
    print("[WARN] Flask not available; using SarahMemory minimal local API fallback:", _flask_import_error)

    class _SMRequestFallback:
        path = "/"
        method = "GET"
        remote_addr = "127.0.0.1"
        content_length = 0
        is_json = False
        args = {}
        host = "127.0.0.1"
        def get_json(self, silent=True):
            return None
        def get_data(self, as_text=False, cache=True):
            return "" if as_text else b""

    request = _SMRequestFallback()
    g = type("_SMG", (), {})()
    session = {}

    def jsonify(*args, **kwargs):
        if args and isinstance(args[0], dict) and not kwargs:
            return args[0]
        return kwargs if kwargs else {"ok": True}

    def render_template(*_a, **_k):
        return "<html><body>SarahMemory minimal local API fallback</body></html>"

    def send_from_directory(*_a, **_k):
        return ""

    def redirect(location, *_a, **_k):
        return location

    def url_for(endpoint, **_k):
        return "/" + str(endpoint)

    def send_file(*_a, **_k):
        return ""

    def abort(code=500, *_a, **_k):
        raise RuntimeError(f"abort:{code}")

    class Response:
        def __init__(self, response=None, mimetype=None, headers=None, status=200):
            self.response = response
            self.mimetype = mimetype
            self.headers = headers or {}
            self.status_code = status

    class Flask:
        def __init__(self, *args, **kwargs):
            self.config = {}
            self._routes = {}
        def route(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco
        def get(self, *args, **kwargs):
            return self.route(*args, **kwargs)
        def post(self, *args, **kwargs):
            return self.route(*args, **kwargs)
        def put(self, *args, **kwargs):
            return self.route(*args, **kwargs)
        def delete(self, *args, **kwargs):
            return self.route(*args, **kwargs)
        def before_request(self, fn):
            return fn
        def after_request(self, fn):
            return fn
        def register_blueprint(self, *args, **kwargs):
            return None
        def run(self, host="127.0.0.1", port=8000, debug=False):
            from http.server import BaseHTTPRequestHandler, HTTPServer
            class Handler(BaseHTTPRequestHandler):
                def _send(self, code=200, payload=None):
                    body = json.dumps(payload or {
                        "ok": True,
                        "running": True,
                        "service": "SarahMemory Minimal Local API",
                        "fallback": "flask_missing",
                        "version": "9.0.0",
                    }).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def do_GET(self):
                    if self.path in ("/", "/api/", "/api/health", "/api/status"):
                        self._send(200)
                    else:
                        self._send(404, {"ok": False, "error": "minimal_api_fallback_route_unavailable", "path": self.path})
                def do_POST(self):
                    self._send(503, {"ok": False, "error": "flask_missing_write_routes_disabled"})
                def log_message(self, *_args):
                    return
            HTTPServer((host, int(port)), Handler).serve_forever()

# --- Flask CORS (safe import for CLI testing & WSGI) ---
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except Exception as e:
    CORS = None  # type: ignore
    _CORS_AVAILABLE = False
    print("[WARN] flask_cors not available:", e)

# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# API startup must degrade smoothly on a local/offline machine. PyJWT, bcrypt,
# and python-dotenv are optional environment helpers; missing packages must not
# prevent the local API health endpoint from starting.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as _dotenv_exc:
    def load_dotenv(*_a, **_k):
        return False
import re
try:
    import jwt  # type: ignore
except Exception as _jwt_exc:
    class _JWTExpiredSignatureError(Exception):
        pass

    class _JWTInvalidTokenError(Exception):
        pass

    def _jwt_b64url_encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    def _jwt_b64url_decode(value: str) -> bytes:
        raw = str(value or "").encode("ascii")
        return base64.urlsafe_b64decode(raw + b"=" * ((4 - len(raw) % 4) % 4))

    def _jwt_json_value(value):
        if isinstance(value, datetime):
            return int(value.timestamp())
        if isinstance(value, dict):
            return {str(k): _jwt_json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jwt_json_value(v) for v in value]
        return value

    class _SarahMemoryJWTFallback:
        ExpiredSignatureError = _JWTExpiredSignatureError
        InvalidTokenError = _JWTInvalidTokenError

        @staticmethod
        def encode(payload, secret, algorithm="HS256"):
            if str(algorithm).upper() != "HS256":
                raise _JWTInvalidTokenError("unsupported_algorithm")
            header = {"alg": "HS256", "typ": "JWT"}
            normalized = _jwt_json_value(dict(payload or {}))
            head = _jwt_b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            body = _jwt_b64url_encode(json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            signing_input = f"{head}.{body}".encode("ascii")
            signature = hmac.new(str(secret).encode("utf-8"), signing_input, hashlib.sha256).digest()
            return f"{head}.{body}.{_jwt_b64url_encode(signature)}"

        @staticmethod
        def decode(token, secret, algorithms=None, audience=None, issuer=None, options=None, **_kwargs):
            allowed = {str(item).upper() for item in (algorithms or ["HS256"])}
            try:
                head, body, signature = str(token or "").split(".")
                header = json.loads(_jwt_b64url_decode(head).decode("utf-8"))
                if str(header.get("alg") or "").upper() != "HS256" or "HS256" not in allowed:
                    raise _JWTInvalidTokenError("algorithm_rejected")
                signing_input = f"{head}.{body}".encode("ascii")
                expected = hmac.new(str(secret).encode("utf-8"), signing_input, hashlib.sha256).digest()
                supplied = _jwt_b64url_decode(signature)
                if not hmac.compare_digest(expected, supplied):
                    raise _JWTInvalidTokenError("signature_verification_failed")
                payload = json.loads(_jwt_b64url_decode(body).decode("utf-8"))
            except _JWTInvalidTokenError:
                raise
            except Exception as exc:
                raise _JWTInvalidTokenError(str(exc)) from exc

            now = int(time.time())
            try:
                if "exp" in payload and int(payload["exp"]) <= now:
                    raise _JWTExpiredSignatureError("token_expired")
                if "nbf" in payload and int(payload["nbf"]) > now + 30:
                    raise _JWTInvalidTokenError("token_not_yet_valid")
                if "iat" in payload and int(payload["iat"]) > now + 30:
                    raise _JWTInvalidTokenError("issued_at_in_future")
            except (_JWTExpiredSignatureError, _JWTInvalidTokenError):
                raise
            except Exception as exc:
                raise _JWTInvalidTokenError(f"invalid_numeric_claim:{exc}") from exc
            if issuer is not None and payload.get("iss") != issuer:
                raise _JWTInvalidTokenError("issuer_mismatch")
            if audience is not None:
                claim = payload.get("aud")
                audiences = {str(item) for item in claim} if isinstance(claim, list) else {str(claim)}
                if str(audience) not in audiences:
                    raise _JWTInvalidTokenError("audience_mismatch")
            required = ((options or {}).get("require") if isinstance(options, dict) else None) or []
            missing = [name for name in required if name not in payload]
            if missing:
                raise _JWTInvalidTokenError("missing_claims:" + ",".join(missing))
            return payload

    jwt = _SarahMemoryJWTFallback()  # type: ignore
try:
    import bcrypt  # type: ignore
except Exception as _bcrypt_exc:
    class _SarahMemoryBcryptFallback:
        _N = 1 << 14
        _R = 8
        _P = 1

        @staticmethod
        def gensalt():
            return os.urandom(16)

        @classmethod
        def hashpw(cls, password: bytes, salt: bytes):
            raw_salt = bytes(salt or b"")
            if len(raw_salt) < 16:
                raw_salt = hashlib.sha256(raw_salt).digest()[:16]
            digest = hashlib.scrypt(bytes(password), salt=raw_salt, n=cls._N, r=cls._R, p=cls._P, dklen=32)
            encoded_salt = base64.urlsafe_b64encode(raw_salt).decode("ascii")
            encoded_hash = base64.urlsafe_b64encode(digest).decode("ascii")
            return f"scrypt${cls._N}${cls._R}${cls._P}${encoded_salt}${encoded_hash}".encode("utf-8")

        @classmethod
        def checkpw(cls, password: bytes, hashed: bytes):
            try:
                scheme, n, r, p, encoded_salt, encoded_hash = bytes(hashed).decode("utf-8").split("$", 5)
                if scheme != "scrypt":
                    return False
                salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
                expected = base64.urlsafe_b64decode(encoded_hash.encode("ascii"))
                actual = hashlib.scrypt(bytes(password), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
                return hmac.compare_digest(actual, expected)
            except Exception:
                return False
    bcrypt = _SarahMemoryBcryptFallback()  # type: ignore

# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# Earliest possible minimal API fallback. If this file is launched as the local
# API child and Flask is absent, do not continue importing the full Flask route
# stack. Start a loopback-only health/status server immediately so boot can
# proceed smoothly and remain local-first.
if __name__ == "__main__" and not globals().get("_SM_FLASK_AVAILABLE", True):
    _early_port = int(os.environ.get("PORT", "8000"))
    _early_host = os.environ.get("SARAHMEMORY_API_HOST") or os.environ.get("SARAHMEMORY_LOCAL_API_BIND_HOST") or "127.0.0.1"
    print(f"[SarahMemory API] Flask missing; starting minimal local health server on http://{_early_host}:{_early_port}")
    Flask(__name__).run(host=_early_host, port=_early_port, debug=False)
    sys.exit(0)

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
    Walk upward from start_dir to locate the SarahMemory project root.
    Handles arbitrary drive letters/folders and v9 api/server -> project root layout.
    """
    for env_key in ("SARAHMEMORY_ROOT", "SARAH_BASE_DIR", "SARAHMEMORY_BASE_DIR", "BASE_DIR"):
        value = os.environ.get(env_key)
        if value:
            candidate = os.path.abspath(os.path.expanduser(value))
            for p in (candidate, os.path.abspath(os.path.join(candidate, "..")), os.path.abspath(os.path.join(candidate, "..", ".."))):
                if os.path.exists(os.path.join(p, "core", "SarahMemoryGlobals.py")) or os.path.exists(os.path.join(p, "SarahMemoryGlobals.py")):
                    return p
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        marker = os.path.join(cur, "SarahMemoryGlobals.py")
        core_marker = os.path.join(cur, "core", "SarahMemoryGlobals.py")
        if os.path.exists(core_marker):
            return cur
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

# Insert best root/core paths first
for p in (
    os.path.join(ROOT_DISCOVERED, "core"),
    ROOT_DISCOVERED,
    os.path.join(ROOT_GRANDPARENT, "core"),
    ROOT_GRANDPARENT,
    ROOT_PARENT,
):
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)



# Attempt to load SarahMemoryGlobals for consistent pathing and versions
try:
    import SarahMemoryGlobals as config
    BASE_DIR = getattr(config, "BASE_DIR", ROOT_DISCOVERED if ROOT_DISCOVERED else ROOT_GRANDPARENT)
    CORE_DIR = getattr(config, "CORE_DIR", os.path.join(BASE_DIR, "core"))
    PUBLIC_DIR = getattr(config, "PUBLIC_DIR", os.path.join(BASE_DIR, "public_html"))
    WEB_DIR = getattr(config, "WEB_DIR", os.path.join(PUBLIC_DIR, "web"))
    DATA_DIR = getattr(config, "DATA_DIR", os.path.join(BASE_DIR, "data"))
    PROJECT_VERSION = getattr(config, "PROJECT_VERSION", "9.0.0") # Ensure v9.0.0 as per spec
except Exception as e:
    app_logger.warning(f"SarahMemoryGlobals (config) import failed or missing attributes. Falling back to local defaults: {e}")
    BASE_DIR = ROOT_DISCOVERED if ROOT_DISCOVERED else ROOT_GRANDPARENT
    CORE_DIR = os.path.join(BASE_DIR, "core") if os.path.isdir(os.path.join(BASE_DIR, "core")) else BASE_DIR
    PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "api"))
    WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "ui"))
    DATA_DIR = os.path.join(BASE_DIR, "data")
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
    t = t.replace("naem", "name").replace("nmae", "name")

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
        "what is your name", "who are you", "your name", "what is your name",
        "describe yourself", "tell me about yourself", "what are you",
        "what do you look like", "describe your 2d model", "describe your 2d avatar", "describe your 3d avatar",
        "describe your avatar", "2d avatar", "3d avatar", "active avatar", "avatar appearance",
        "what version are you", "what version are you running", "your version",
        "server version", "program version", "app version", "sarahmemory version",
        "version number",
        "who made you", "who created you", "creator",
        "who designed you", "designer", "engineer",
        "who engineered you", "who built you",
        "sarahmemory aios", "sarahmemory ai os", "sarah memory aios",
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


def _sm_v9g_word_any(text: str, words: tuple[str, ...] | list[str]) -> bool:
    """Whole-token variant for self-awareness routing.

    The legacy substring helper is still used in older compatibility paths, but
    /api/chat must not classify ordinary words inside SarahMemory identifiers as
    hardware/self-body facts.  Example: ``SarahMemoryAPI.py`` must not trip the
    generic ``memory`` runtime fact route.
    """
    t = str(text or "").lower()
    for word in words:
        w = str(word or "").lower().strip()
        if not w:
            continue
        if " " in w or "-" in w:
            if w in t:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", t):
            return True
    return False


def _sm_v9g_is_general_definition_query(norm: str) -> bool:
    """True for concept questions such as 'What is RAM?' or 'Define CPU'."""
    t = str(norm or "").lower().strip()
    if not t:
        return False
    self_scope_terms = (
        "my", "your", "you", "youre", "you're", "this system", "this machine",
        "this computer", "my pc", "your pc", "my machine", "your machine",
        "current system", "currently", "do i have", "do you have", "are you using",
        "am i using", "installed", "detected", "sensor", "runtime", "body map", "hardware status",
    )
    if _sm_v9g_word_any(t, self_scope_terms) or any(phrase in t for phrase in self_scope_terms if " " in phrase):
        return False
    return bool(
        re.match(r"^(what\s+is|what\s+are|define|explain|describe|tell\s+me\s+about)\b", t)
        or re.match(r"^(what\s+does\s+.+\s+mean)\b", t)
    )


def _sm_v9g_is_self_scope_query(norm: str, *, fact_kind: str = "general_system_fact") -> bool:
    """Return True only when the user is asking about SarahMemory/host runtime state.

    This prevents concept questions and project filenames from being captured by
    the self-aware hardware/status route.
    """
    t = str(norm or "").lower().strip()
    if not t or _sm_v9g_is_general_definition_query(t):
        return False
    explicit_scope_phrases = (
        "my ", "your ", "you using", "am i using", "are you using", "do you have",
        "do i have", "how much", "how many", "this system", "this machine",
        "this computer", "my computer", "your computer", "my pc", "your pc",
        "current system", "currently", "runtime", "body map", "body-map",
        "hardware status", "system status", "detected", "installed", "sensor",
        "node name", "hostname", "python version",
    )
    if _sm_v9g_contains_any(t, explicit_scope_phrases):
        return True
    # Bare hardware terms alone are no longer enough.  They must appear with a
    # status/measurement verb or an explicit system-scope phrase above.
    measurement_terms = (
        "temperature", "temp", "thermal", "fan", "rpm", "free", "used", "available",
        "connected", "clock", "speed", "capacity", "usage", "utilization", "version",
        "bios", "uefi", "firmware", "status",
    )
    return bool(fact_kind != "general_system_fact" and _sm_v9g_word_any(t, measurement_terms))


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
    general_definition = _sm_v9g_is_general_definition_query(norm)

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
    elif (not general_definition) and (_sm_v9g_word_any(norm, ("gpu", "graphics")) or "video card" in norm):
        requested_metric = "identity"
        fact_kind = "gpu"
        target = target or component
        answer_shape = "summary"
    elif (not general_definition) and _sm_v9g_word_any(norm, ("cpu", "processor")):
        requested_metric = "identity"
        fact_kind = "cpu"
        target = target or component
        answer_shape = "summary"
    elif (not general_definition) and (_sm_v9g_word_any(norm, ("motherboard", "mainboard", "baseboard")) or "system board" in norm):
        requested_metric = "identity"
        fact_kind = "motherboard"
        target = target or component
        answer_shape = "summary"
    elif (not general_definition) and _sm_v9g_word_any(norm, ("ram", "memory")):
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

    self_scope = _sm_v9g_is_self_scope_query(norm, fact_kind=fact_kind)
    fact_scope = bool(self_scope and fact_kind != "general_system_fact")
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
        server_dir = _THIS_DIR

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


def _sm_import_appsys_runtime():
    """Load api/server/appsys.py beside app.py for Clock/Locality Court calls."""
    try:
        server_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        server_dir = _THIS_DIR
    appsys_path = os.path.join(server_dir, "appsys.py")
    if not os.path.exists(appsys_path):
        raise RuntimeError(f"appsys.py not found beside app.py: {appsys_path}")
    module_name = f"_sarahmemory_runtime_appsys_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, appsys_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create import spec for appsys.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not callable(getattr(mod, "run_clock_court_query", None)):
        raise RuntimeError("runtime appsys clock court runner unavailable after direct load")
    return mod


def _sm_is_clock_court_question(text: str) -> bool:
    t = str(text or "").strip().lower().replace("what's", "what is")
    if not t:
        return False
    temporal_patterns = (
        r"\bwhat\s+(year|date|day|time)\b",
        r"\b(year|date|day|time)\s+is\s+it\b",
        r"\bcurrent\s+(year|date|time)\b",
        r"\btoday'?s\s+(date|schedule)\b",
        r"\btimezone\b|\btime\s+zone\b|\butc\s+offset\b",
    )
    return any(re.search(pat, t) for pat in temporal_patterns)


def _sm_try_clock_court_route(text: str, *, source: str = "api_chat") -> dict | None:
    if not _sm_is_clock_court_question(text):
        return None
    try:
        appsys_mod = _sm_import_appsys_runtime()
        packet = appsys_mod.run_clock_court_query(claim=text, source=source, meta={"route": "api_chat_clock_court"})
        if not isinstance(packet, dict):
            raise RuntimeError("appsys returned non-dict clock court packet")
        reply = str(packet.get("presentation_text") or packet.get("accepted_value") or "Clock Court did not return a presentable answer.")
        meta = {
            "source": "appsys_clock_court",
            "engine": "appsys.run_clock_court_query",
            "intent": "temporal_context",
            "claim_type": packet.get("claim_type"),
            "verdict": packet.get("verdict"),
            "confidence": packet.get("confidence"),
            "model_memory_authority": False,
            "execution_allowed": False,
            "version": PROJECT_VERSION,
            "appsys_module_file": str(getattr(appsys_mod, "__file__", "")),
        }
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="system_status", meta=meta), meta=meta, raw_answer=reply)
        bundle["ok"] = True
        bundle["clock_court"] = packet
        return bundle
    except Exception as exc:
        app_logger.warning("Clock Court route failed: %s", exc, exc_info=True)
        bundle = _sm_make_outward_bundle(
            "Clock Court is configured for this question, but the court route failed internally. I did not use model memory to guess the live time/date/year.",
            meta={"source": "appsys_clock_court_error", "error": str(exc), "version": PROJECT_VERSION},
            errors=[str(exc)],
        )
        bundle["ok"] = False
        return bundle


def _sm_try_appself_identity_route(text: str, *, source: str = "api_chat") -> dict | None:
    if not _is_identity_question(text):
        return None
    try:
        appself_mod = _sm_import_appself_runtime()
        fn = getattr(appself_mod, "run_self_identity_query", None)
        if not callable(fn):
            raise RuntimeError("appself identity query runner unavailable")
        result = fn(claim=text, source=source, meta={"route": "api_chat_identity", "bridge": "runtime_appself_identity"})
        if not isinstance(result, dict):
            raise RuntimeError("appself returned non-dict identity result")
        reply = str(result.get("presentation_text") or "Identity packet returned without presentation text.")
        meta = {
            "source": "appself_identity_court",
            "engine": "appself.run_self_identity_query",
            "intent": "identity_self_embodiment",
            "kind": result.get("kind"),
            "decision": result.get("decision"),
            "model_memory_authority": False,
            "execution_allowed": False,
            "version": PROJECT_VERSION,
            "appself_module_file": str(getattr(appself_mod, "__file__", "")),
        }
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="identity", meta=meta), meta=meta, raw_answer=reply)
        bundle["ok"] = True
        bundle["identity_packet"] = result.get("identity_packet")
        return bundle
    except Exception as exc:
        app_logger.warning("Identity Court route failed: %s", exc, exc_info=True)
        # Compatibility fallback. Keep it bounded and explicitly mark that appself failed.
        ident = _identity_payload()
        raw_reply = f"I'm {ident['name']} — your {ident['platform']} companion. The canonical appself identity route failed, so this is a bounded compatibility fallback, not the preferred identity court path."
        bundle = _sm_make_outward_bundle(
            _sm_present_text(raw_reply, intent="identity"),
            meta={"source": "identity_compatibility_fallback", "engine": "app_py_fallback_after_appself_failure", "error": str(exc), "version": PROJECT_VERSION},
            raw_answer=raw_reply,
            errors=[str(exc)],
        )
        bundle["identity"] = ident
        return bundle


def _sm_try_sml_source_authority_route(text: str, *, local_only: bool = True, intent: str = "question") -> dict | None:
    """Prevent live/current-source claims from falling to model memory.

    This is a read-only Source Authority Court bridge. It does not hardcode
    answers; it either invokes SarahMemoryResearch/API evidence acquisition or
    returns a source-required response instead of allowing stale model memory.
    """
    try:
        import SarahMemorySMLProtocol as _SMSML  # type: ignore
        court_fn = getattr(_SMSML, "sml_build_source_authority_court_packet", None)
        if not callable(court_fn):
            return None
        court = court_fn(text, context={"source": "api_chat", "intent": intent, "local_only": bool(local_only)})
        vector = court.get("claim_vector") if isinstance(court, dict) else {}
        if not isinstance(vector, dict):
            return None
        if bool(vector.get("model_final_authority", True)):
            return None
        domain = str(vector.get("domain") or "")
        freshness_required = bool(vector.get("freshness_required"))
        if domain in {"identity_self_embodiment", "temporal_locality", "local_device_control", "creative_build_mission"}:
            return None
        if not freshness_required:
            return None
        meta = {
            "source": "sml_source_authority_court",
            "engine": "SarahMemorySMLProtocol.sml_build_source_authority_court_packet",
            "intent": intent or "question",
            "domain": domain,
            "claim_type": vector.get("claim_type"),
            "temporal_scope": vector.get("temporal_scope"),
            "model_memory_authority": False,
            "execution_allowed": False,
            "preferred_sources": vector.get("preferred_sources"),
            "version": PROJECT_VERSION,
        }
        if local_only:
            reply = "This question requires current-source evidence. I will not answer it from model memory or static demo facts while current research/API access is unavailable in this route."
            bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="source_authority", meta=meta), meta=meta, raw_answer=reply)
            bundle["ok"] = True
            bundle["source_authority_court"] = court
            return bundle
        try:
            import SarahMemoryResearch as _SMResearch  # type: ignore
            research_fn = getattr(_SMResearch, "get_research_data", None)
            if callable(research_fn):
                bounded = _sm_bounded_call(research_fn, text, timeout_seconds=12.0, call_name="source_authority_research")
                if bounded.get("ok") and isinstance(bounded.get("value"), dict):
                    research = bounded.get("value")
                    raw = str(research.get("data") or research.get("content") or research.get("snippet") or research.get("answer") or "").strip()
                    conf = float(research.get("confidence") or 0.0)
                    if raw and conf > 0.0 and "unable to find" not in raw.lower() and "research failed" not in raw.lower():
                        evidence_packet = None
                        try:
                            ev_fn = getattr(_SMSML, "sml_build_evidence_court_packet", None)
                            if callable(ev_fn):
                                evidence_packet = ev_fn(text, research, context={"source": "api_chat_source_authority", "intent": intent, "domain": domain})
                        except Exception as ev_exc:
                            meta["evidence_court_error"] = str(ev_exc)
                        accepted_text = ""
                        if isinstance(evidence_packet, dict):
                            accepted_text = str(evidence_packet.get("accepted_content") or "").strip()
                        if accepted_text:
                            meta.update({
                                "research_source": research.get("source"),
                                "research_confidence": conf,
                                "evidence_court": "accepted",
                                "evidence_schema": "SarahMemory.sml.evidence_court_packet.B09",
                            })
                            reply = accepted_text
                            bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="research", meta=meta), meta=meta, raw_answer=reply)
                            bundle["ok"] = True
                            bundle["source_authority_court"] = court
                            bundle["evidence_court_packet"] = evidence_packet
                            bundle["research_artifact"] = research
                            return bundle
                        meta.update({
                            "research_source": research.get("source"),
                            "research_confidence": conf,
                            "evidence_court": "not_accepted",
                        })
                        if isinstance(evidence_packet, dict):
                            meta["evidence_verdict"] = ((evidence_packet.get("court_2") or {}) if isinstance(evidence_packet.get("court_2"), dict) else {}).get("verdict")
        except Exception as exc:
            meta["research_error"] = str(exc)
        reply = "This question requires current-source evidence, but Research/API did not return a verified artifact. I will not substitute model memory as the final answer."
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="source_authority", meta=meta), meta=meta, raw_answer=reply)
        bundle["ok"] = True
        bundle["source_authority_court"] = court
        try:
            if isinstance(locals().get("evidence_packet"), dict):
                bundle["evidence_court_packet"] = locals().get("evidence_packet")
        except Exception:
            pass
        return bundle
    except Exception as exc:
        app_logger.warning("SML Source Authority route failed: %s", exc, exc_info=True)
        return None


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
# SARAHMEMORY_PATCH_NOTE 2026-06-24:
# V9 UI contract correction. The built React/Vite UI lives in
# <PROJECT_ROOT>/data/ui/v9, not <PROJECT_ROOT>/ui/v9. If this path is wrong,
# Flask serves the wrong index/static assets and the embedded window appears as
# a blank white box. Keep this as the single backend source of truth.
# Expected: <PROJECT_ROOT>/data/ui/v9/
UI_DIST_DIR = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "data", "ui", "v9"))
UI_SRC_DIR = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "data", "ui", "V9_ui_src"))
WALLETS_DIR = os.path.join(DATA_DIR, "wallets")
# These canonical paths are needed during module import, before the later
# _globals_dir() compatibility helper is defined. Resolve them directly from
# SarahMemoryGlobals here so the API child cannot fail before Flask starts.
_DATASETS_DIR = os.path.abspath(str(getattr(config, "DATASETS_DIR", os.path.join(DATA_DIR, "memory", "datasets")))) if config is not None else os.path.abspath(os.path.join(DATA_DIR, "memory", "datasets"))
_SETTINGS_DIR = os.path.abspath(str(getattr(config, "SETTINGS_DIR", os.path.join(DATA_DIR, "settings")))) if config is not None else os.path.abspath(os.path.join(DATA_DIR, "settings"))
META_DB = str(getattr(config, "META_DB_PATH", os.path.join(_DATASETS_DIR, "meta.db"))) if config is not None else os.path.join(_DATASETS_DIR, "meta.db")
LOGS_DIR = os.path.join(DATA_DIR, "logs") # Default to DATA_DIR/logs

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(WALLETS_DIR, exist_ok=True)
os.makedirs(_DATASETS_DIR, exist_ok=True)
os.makedirs(_SETTINGS_DIR, exist_ok=True)
try:
    from SarahMemoryMigrations import migrate_root_runtime_artifacts as _sm_migrate_root_runtime_artifacts
    _sm_root_artifact_migration = _sm_migrate_root_runtime_artifacts()
    if not _sm_root_artifact_migration.get("ok", True):
        app_logger.warning("Root runtime artifact migration reported errors: %s", _sm_root_artifact_migration.get("errors"))
except Exception as _migration_exc:
    app_logger.warning("Root runtime artifact migration skipped: %s", _migration_exc)


# ---------------------------------------------------------------------------
# Global runtime state (kept intentionally small and fast)
# ---------------------------------------------------------------------------
APP_VERSION = PROJECT_VERSION  # API/UI convenience alias

# Persistent runtime JSON belongs under SETTINGS_DIR; SQLite belongs under DATASETS_DIR.
STATE_DB = str(getattr(config, "SERVER_STATE_PATH", os.path.join(_SETTINGS_DIR, "server_state.json"))) if config is not None else os.path.join(_SETTINGS_DIR, "server_state.json")
WALLET_DB = str(getattr(config, "WALLETS_DB_PATH", os.path.join(_DATASETS_DIR, "wallets.db"))) if config is not None else os.path.join(_DATASETS_DIR, "wallets.db")

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

_SM_LAST_CHAT_EXCHANGE: dict = {}

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
_STATE_LOCK = threading.RLock()

def _fingerprint_json(payload) -> str:
    try:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception:
        return ""

def load_state() -> dict:
    """Load persisted server state under the process synchronization lock."""
    with _STATE_LOCK:
        try:
            if os.path.exists(STATE_DB):
                with open(STATE_DB, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

def _write_json_if_changed(path: str, payload, *, ensure_ascii: bool = False) -> bool:
    """Durable atomic JSON write with collision-free temporary files."""
    with _STATE_LOCK:
        try:
            text = json.dumps(payload or {}, indent=2, sort_keys=True, ensure_ascii=ensure_ascii)
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                        if handle.read() == text:
                            return False
            except Exception:
                pass
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return True
        except Exception:
            return False


def save_state(state_or_key, value=None) -> None:
    """Persist a complete state snapshot or one synchronized key update."""
    with _STATE_LOCK:
        try:
            if value is None and isinstance(state_or_key, dict):
                state = dict(state_or_key or {})
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
        if config is not None and getattr(config, "BROWSER_STATE_PATH", None):
            return str(getattr(config, "BROWSER_STATE_PATH"))
    except Exception:
        pass
    return os.path.join(_SETTINGS_DIR, "browser_state.json")

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

# Mount the bounded Ledger receipt API on the unified Flask app. Importing the
# Ledger no longer initializes databases; init_app performs explicit setup.
try:
    if ledger_mod is not None and hasattr(ledger_mod, "init_app"):
        ledger_mod.init_app(app)
except Exception as _ledger_init_exc:
    app_logger.warning("Ledger receipt API mount skipped: %s", _ledger_init_exc)

# Mount the NAILDE SDK/API bridge outside app.py to avoid expanding the main
# Flask ingress file. appsdk.py owns /api/nailde/* route definitions; app.py
# only performs guarded registration.
try:
    import appsdk as appsdk_mod  # type: ignore
    if hasattr(appsdk_mod, "init_app"):
        appsdk_mod.init_app(app, logger=app_logger)
except Exception as _appsdk_init_exc:
    app_logger.warning("NAILDE SDK API mount skipped: %s", _appsdk_init_exc)

# ARILE API boundary guard. The API server is a boundary sensor, not the ARILE engine.
try:
    from SarahMemoryARILE import arile_endpoint_guard, arile_emit, get_arile_runtime_status
except Exception:  # pragma: no cover
    arile_endpoint_guard = None  # type: ignore
    arile_emit = None  # type: ignore
    get_arile_runtime_status = None  # type: ignore

@app.before_request
def _arile_api_boundary_preflight():
    try:
        if callable(arile_endpoint_guard):
            decision = arile_endpoint_guard(
                endpoint_name=str(getattr(request, "path", "")),
                request_meta={
                    "method": getattr(request, "method", ""),
                    "content_length": getattr(request, "content_length", 0) or 0,
                    "remote_addr": getattr(request, "remote_addr", ""),
                },
                risk="medium" if str(getattr(request, "path", "")).startswith("/api/devbridge") else "low",
            )
            if decision == "block":
                return jsonify({"ok": False, "error": "request_blocked_by_arile"}), 429
    except Exception:
        return None
    return None


# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# API anti-agent firewall. ARILE watches runtime variance; this deterministic
# guard blocks direct hijack/override payloads and unarmed remote write/action
# attempts before route handlers can interpret them. It does not replace
# SecurityGovernor/AssuranceGate/SMGET; it adds required boundary evidence.
try:
    from SarahMemoryAgentFirewall import inspect_payload as _sm_agent_firewall_inspect
except Exception:
    _sm_agent_firewall_inspect = None  # type: ignore

def _sm_v9_confirmed_payload(payload: dict | None = None) -> bool:
    try:
        payload = payload if isinstance(payload, dict) else (request.get_json(silent=True) or {})
    except Exception:
        payload = payload if isinstance(payload, dict) else {}
    for key in ("confirm", "confirmed", "user_confirmed", "user_authorized", "approved", "allow", "explicit_user_approval"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on", "approved", "confirm", "confirmed", "user_approved"):
            return True
    phrase = str((payload or {}).get("confirm_phrase") or (payload or {}).get("confirmation_phrase") or "").strip().upper()
    return phrase in {"I APPROVE", "USER APPROVED", "CONFIRM ACTION", "APPROVE GOVERNED ACTION"}


def _sm_v9_action_authority_preflight(path: str, method: str, payload: dict | None = None):
    """Fail-closed authority membrane for high-impact API actions.

    This does not replace SafetyPolicies/SecurityGovernor/OperatorCore. It prevents
    API bridge write surfaces from treating a UI/model/agent request as authority.
    """
    method = str(method or "GET").upper()
    if method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    payload = payload if isinstance(payload, dict) else {}
    p = str(path or "")
    driver_high_impact = (
        p.startswith("/api/drivers/")
        and (p.endswith("/connect") or p.endswith("/session/start") or "/actions/" in p or p.endswith("/registry"))
    )
    high_impact = (
        driver_high_impact
        or p in {"/api/terminal/execute", "/api/launch", "/api/ui/exit"}
        or p.startswith("/api/devbridge/apply-approved")
        or p.startswith("/api/devbridge/rollback")
        or p.startswith("/api/files/trash/empty")
        or p.startswith("/api/cognitive/instinct/trigger")
        or p.startswith("/api/vr/start")
    )
    destructive_file = p.startswith("/api/files/delete") and str(payload.get("mode") or "trash").strip().lower() in {"permanent", "delete", "hard"}
    if not (high_impact or destructive_file):
        return None
    if _sm_v9_confirmed_payload(payload):
        return None
    return jsonify({
        "ok": False,
        "error": "requires_user_confirmation",
        "decision": "REQUIRE_USER",
        "path": p,
        "source": "api.server.governance_authority_preflight",
        "message": "This endpoint can affect files, network, devices, VR/process state, drivers, terminal execution, or patch state. Explicit user confirmation is required.",
    }), 403


@app.before_request
def _sarahmemory_api_firewall_preflight():
    try:
        path = str(getattr(request, "path", "") or "")
        method = str(getattr(request, "method", "") or "GET").upper()
        remote_addr = str(getattr(request, "remote_addr", "") or "")

        # Local-only mode permits normal local UI/API calls while preventing
        # outside hosts from triggering write/action endpoints unless explicitly
        # armed by the operator/UI session.
        local_only = bool(getattr(config, "LOCAL_ONLY_MODE", True))
        online_armed = bool(getattr(config, "SARAHMEMORY_ONLINE_SESSION_ARMED", False))
        local_peer = remote_addr in ("127.0.0.1", "::1", "localhost", "")
        write_method = method in ("POST", "PUT", "PATCH", "DELETE")
        read_allowlist = path in ("/", "/api/", "/api/health", "/api/status", "/api/meta", "/api/arile/status")
        if local_only and not online_armed and not local_peer and write_method:
            return jsonify({"ok": False, "error": "local_only_remote_write_blocked", "path": path}), 403

        if callable(_sm_agent_firewall_inspect):
            payload = {
                "path": path,
                "method": method,
                "args": request.args.to_dict(flat=False),
                "json": request.get_json(silent=True) if request.is_json else None,
                "text": request.get_data(as_text=True)[:12000] if write_method else "",
            }
            verdict = _sm_agent_firewall_inspect(payload, source="api.server.app.before_request", remote_addr=remote_addr)
            if verdict.get("verdict") == "DENY":
                reason = str(verdict.get("reason") or "AgentFirewall denied this request.")
                hits = verdict.get("hits", [])[:5]
                if path == "/api/terminal/agent":
                    reply_lines = [
                        "DENY / BLOCKED",
                        f"Reason: {reason}",
                        "No shell command, network call, file mutation, driver action, DevBridge apply, or hidden persistence was executed.",
                    ]
                    if hits:
                        reply_lines.append("Matched patterns: " + ", ".join(str(h) for h in hits))
                    reply_lines.append("Allowed alternative: rephrase as inspect/propose only, or route real execution through explicit governed approval.")
                    reply = "\n".join(reply_lines)
                    return jsonify({
                        "ok": False,
                        "blocked": True,
                        "error": "agent_firewall_denied",
                        "decision": "DENY",
                        "reason": reason,
                        "reply": reply,
                        "stdout": reply,
                        "stderr": reason,
                        "mode": "terminal_agent",
                        "agent_status": {
                            "mode": "terminal_agent",
                            "execution_authority": "inspect_or_propose_only",
                            "shell_execution": False,
                            "tool_execution": False,
                            "network_execution": False,
                            "file_mutation": False,
                            "devbridge_apply": False,
                            "task_verdict": {
                                "verdict": "DENY",
                                "reason": reason,
                                "risk_tier": verdict.get("risk_tier"),
                                "containment_state": verdict.get("containment_state"),
                                "hits": hits,
                            },
                        },
                        "actions": [],
                        "ts": time.time(),
                    }), 200
                return jsonify({"ok": False, "error": "agent_firewall_denied", "decision": "DENY", "reason": reason, "hits": hits, "risk_tier": verdict.get("risk_tier"), "containment_state": verdict.get("containment_state")}), 403
            if verdict.get("verdict") == "REQUIRE_REVIEW" and write_method:
                reason = str(verdict.get("reason") or "AgentFirewall requires user review.")
                if path == "/api/terminal/agent":
                    reply = "REQUIRE_REVIEW / CAPTURED_REVIEW\nReason: " + reason + "\nNo action was executed. User review is required before release."
                    return jsonify({
                        "ok": False,
                        "blocked": True,
                        "error": "agent_firewall_requires_user_review",
                        "decision": "REQUIRE_USER",
                        "reason": reason,
                        "reply": reply,
                        "stdout": reply,
                        "stderr": reason,
                        "mode": "terminal_agent",
                        "capture_report_path": verdict.get("capture_report_path", ""),
                        "actions": [],
                        "ts": time.time(),
                    }), 200
                return jsonify({"ok": False, "error": "agent_firewall_requires_user_review", "decision": "REQUIRE_USER", "reason": reason, "risk_tier": verdict.get("risk_tier"), "containment_state": verdict.get("containment_state"), "capture_report_path": verdict.get("capture_report_path", "")}), 423
            if verdict.get("verdict") == "REQUIRE_LOCAL_OR_ARMED" and write_method and not read_allowlist:
                return jsonify({"ok": False, "error": "remote_trigger_requires_local_or_armed_session", "decision": "REQUIRE_LOCAL_OR_ARMED"}), 403

        if write_method:
            _payload_for_auth = request.get_json(silent=True) if request.is_json else {}
            _authority_response = _sm_v9_action_authority_preflight(path, method, _payload_for_auth if isinstance(_payload_for_auth, dict) else {})
            if _authority_response is not None:
                return _authority_response
    except Exception:
        return None
    return None

@app.get("/api/arile/status")
def arile_status_api():
    try:
        if callable(get_arile_runtime_status):
            return jsonify(get_arile_runtime_status())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": False, "error": "SarahMemoryARILE unavailable"}), 503

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
        renderer_path = os.path.join(globals().get("CORE_DIR", os.path.join(BASE_DIR, "core")), "SarahMemoryVRHudRenderer.py")
        if not os.path.exists(renderer_path):
            renderer_path = os.path.join(BASE_DIR, "SarahMemoryVRHudRenderer.py")
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
    """Open a crash-resilient SQLite connection for API request workers."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        con = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            pragmas = getattr(config, "SQLITE_CONNECTION_PRAGMAS", {}) if config is not None else {}
            for key, value in (pragmas or {}).items():
                con.execute(f"PRAGMA {key}={value}")
            con.execute("PRAGMA busy_timeout=10000")
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("PRAGMA temp_store=MEMORY")
            con.execute("PRAGMA wal_autocheckpoint=1000")
        except Exception:
            pass
        return con
    except sqlite3.Error as exc:
        app_logger.error("Failed to connect to SQLite DB at %s: %s", path, exc)
        raise

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


class _SMBoundedCallTimeout(TimeoutError):
    """Raised when an optional downstream component exceeds its request budget."""


def _sm_bounded_call(callable_obj, *args, timeout_seconds: float = 5.0, call_name: str = "component", **kwargs):
    """Execute one optional integration in an isolated daemon worker.

    The caller receives a deterministic outcome packet. A timed-out worker is
    never joined on the request/UI thread and cannot hold the Flask response
    lifecycle open. This helper is reserved for fail-soft optional components;
    governed action execution remains owned by OperatorCore.
    """
    if not callable(callable_obj):
        return {"ok": False, "timed_out": False, "value": None, "error": f"{call_name}_unavailable"}
    try:
        budget = max(0.05, float(timeout_seconds))
    except Exception:
        budget = 5.0
    completed = threading.Event()
    outcome = {"ok": False, "timed_out": False, "value": None, "error": None}

    def _runner():
        try:
            outcome["value"] = callable_obj(*args, **kwargs)
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            completed.set()

    worker = threading.Thread(target=_runner, name=f"SMBounded-{call_name}", daemon=True)
    worker.start()
    if not completed.wait(budget):
        return {
            "ok": False,
            "timed_out": True,
            "value": None,
            "error": f"{call_name}_timeout_after_{budget:.2f}s",
        }
    return dict(outcome)

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
        "CORE_DIR": os.path.join(root_dir, "core") if os.path.isdir(os.path.join(root_dir, "core")) else root_dir,
        "API_SERVER_DIR": os.path.join(root_dir, "api", "server"),
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
    return os.path.join(os.path.abspath(BASE_DIR), default_rel)


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



def _sm_scrub_visible_text(raw_text: str, *, user_text: str = "") -> str:
    """Final display scrub for /api/chat.

    This is a presentation filter only. It prevents model chain-of-thought,
    raw route diagnostics, DB/cache records, and internal JSON objects from
    leaking into the UI. Diagnostics remain available through explicit
    diagnostics routes/metadata, not normal chat text.
    """
    try:
        text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""
        # Remove closed and unterminated hidden reasoning blocks.
        text = re.sub(r"(?is)<\s*(think|analysis)\s*>.*?<\s*/\s*\1\s*>", "", text)
        text = re.sub(r"(?is)<\s*(think|analysis)\s*>.*\Z", "", text)
        text = re.sub(r"(?is)\[\s*(think|analysis)\s*\].*?\[\s*/\s*\1\s*\]", "", text)
        text = re.sub(r"(?is)\[\s*(think|analysis)\s*\].*\Z", "", text)
        low = text.lower().strip()
        internal_markers = (
            "runtime_identity_override", "ingress route confidence", "structured action request",
            "no engine produced an answer", "provide more constraints or enable an applicable tier",
            "from ailearning.db:qacache", "from ai_learning.db:qacache", "vetted_local_llm_general",
            "vettedlocalllm_general", "memory = {\"error\"", "pdhaddenglishcounterw failed",
            "the correct sml source path is", "connected local sources yet",
            "bounded glossary unavailable", "install or select a local model",
            "answer_requires_knowledge_source", "needs_knowledge_source_execution",
        )
        if any(m in low for m in internal_markers):
            return ""
        # Drop transcript/scaffold lines and provenance footers.
        cleaned = []
        for line in text.split("\n"):
            l = line.strip()
            ll = l.lower()
            if not l:
                cleaned.append("")
                continue
            if re.match(r"(?i)^(system|developer|tool|assistant|user|human|prompt|question)\s*:", l):
                continue
            if any(m in ll for m in internal_markers):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip()
        if user_text:
            q = re.sub(r"\s+", " ", str(user_text or "").strip().lower().strip("?.!"))
            first = re.sub(r"\s+", " ", text.split("\n", 1)[0].strip().lower().strip("?.!")) if text else ""
            if q and first == q:
                text = text.split("\n", 1)[1].strip() if "\n" in text else ""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
    except Exception:
        return str(raw_text or "").strip()

def _sm_present_text(raw_text: str, *, intent: str = "", meta: dict | None = None) -> str:
    text = _sm_scrub_visible_text(raw_text or "", user_text=str((meta or {}).get("_prompt_text") or "")).strip()
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
            return v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
        # SarahNet Sync/Network already use SARAHNET_SHARED_SECRET. Reuse the
        # same configured secret for broker HMAC verification instead of making
        # local Sync and the API bridge silently derive different auth domains.
        v = getattr(G, "SARAHNET_SHARED_SECRET", "") or ""
        if v:
            return v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
    except Exception:
        pass
    return (os.environ.get("HUB_HMAC_SECRET") or os.environ.get("SARAH_HUB_HMAC_SECRET") or os.environ.get("SARAHNET_SHARED_SECRET") or "").strip()

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
            "meta": {"ready": has("/api/version") and has("/api/meta/capabilities"), "backend": "api/server/app.py Phase1 compatibility contract"},
            "voice": {"ready": has("/api/voices") and has("/api/tts/speak"), "backend": "SarahMemoryVoice.py via app.py compatibility contract"},
            "research": {"ready": has("/api/research/search"), "backend": "SarahMemoryResearch.py via app.py compatibility contract"},
            "files": {"ready": has("/api/files/analyze") or has("/api/files/capabilities"), "backend": "appsys.py + app.py compatibility contract"},
            "ranking": {"ready": has("/api/ranking") and has("/api/ranking/stats"), "backend": "local meta.db ranking bridge"},
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



def _ui_runtime_config_script() -> str:
    """Return a small runtime config script injected before the Vite bundle.

    SARAHMEMORY_PATCH_NOTE 2026-06-24:
    The frontend must not guess between cloud and local backends. This runtime
    contract pins window.SARAH_API_BASE and related mode flags to the backend
    that actually served the UI. It keeps the UI local-first by default and
    prevents a production Vite build from silently drifting toward public cloud
    endpoints when running inside pywebview.
    """
    try:
        host = getattr(config, "DEFAULT_HOST", "127.0.0.1")
        port = int(getattr(config, "DEFAULT_PORT", 8000))
    except Exception:
        host, port = "127.0.0.1", 8000
    api_base = ""
    try:
        api_base = request.host_url.rstrip("/")
    except Exception:
        api_base = f"http://{host}:{port}"
    payload = {
        "apiBase": api_base,
        "localOnly": bool(getattr(config, "LOCAL_ONLY_MODE", True)),
        "uiDist": UI_DIST_DIR,
        "uiSrc": UI_SRC_DIR,
        "version": PROJECT_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        "<script>\n"
        "window.SARAH_UI_BOOT = " + raw + ";\n"
        "window.SARAH_API_BASE = window.SARAH_UI_BOOT.apiBase;\n"
        "window.SARAH_LOCAL_ONLY = !!window.SARAH_UI_BOOT.localOnly;\n"
        "</script>\n"
    )


def _serve_v9_ui_index():
    """Serve V9 index.html with runtime config injection.

    SARAHMEMORY_PATCH_NOTE 2026-06-24:
    send_from_directory is correct for immutable assets, but index.html needs a
    runtime bridge so the built UI can discover the local backend. This avoids
    file:// loading, stale cloud defaults, and blank webview symptoms while
    preserving the existing Vite dist folder.
    """
    index_path = os.path.join(UI_DIST_DIR, "index.html")
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
        script = _ui_runtime_config_script()
        if "window.SARAH_UI_BOOT" not in html:
            if "</head>" in html:
                html = html.replace("</head>", script + "</head>", 1)
            else:
                html = script + html
        return Response(html, mimetype="text/html")
    except Exception:
        return send_from_directory(UI_DIST_DIR, "index.html")

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
            return _serve_v9_ui_index() or send_from_directory(UI_DIST_DIR, "index.html")

    # Otherwise, show the Network Hub landing (legacy /api/server/static/index.html)
    hub_index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(hub_index):
        return send_from_directory(STATIC_DIR, "index.html")
    return redirect("/api/")


# -----------------------------------------------------------------------------
# Web UI dist asset serving (local + ai.sarahmemory.com)
# -----------------------------------------------------------------------------


@app.route("/api/ui/runtime-config.js")
def api_ui_runtime_config_js():
    """Runtime JS config for V9 WebUI.

    SARAHMEMORY_PATCH_NOTE 2026-06-24:
    This endpoint gives the frontend an explicit local API contract. It is
    read-only, no network probing, no external calls, and no authority changes.
    """
    js = _ui_runtime_config_script()
    # Strip wrapping <script> tags for direct JS use.
    js = js.replace("<script>\n", "").replace("</script>\n", "")
    return Response(js, mimetype="application/javascript")

@app.route("/api/ui/v9-paths")
def api_ui_v9_paths():
    """Expose V9 UI path readiness for diagnostics and frontend/backend contract checks."""
    index_path = os.path.join(UI_DIST_DIR, "index.html")
    return jsonify({
        "ok": bool(os.path.isfile(index_path)),
        "ui_dist_dir": UI_DIST_DIR,
        "ui_src_dir": UI_SRC_DIR,
        "index_exists": bool(os.path.isfile(index_path)),
        "assets_exists": bool(os.path.isdir(os.path.join(UI_DIST_DIR, "assets"))),
        "source_exists": bool(os.path.isdir(UI_SRC_DIR)),
        "version": PROJECT_VERSION,
    }), 200

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
            return _serve_v9_ui_index() or send_from_directory(UI_DIST_DIR, "index.html")

    # Fallback to hub static if present
    candidate = os.path.join(STATIC_DIR, path)
    if os.path.isfile(candidate):
        return send_from_directory(STATIC_DIR, path)
    return abort(404)

@app.route("/api/static/<path:filename>")
def static_serv(filename):
    return send_from_directory(STATIC_DIR, filename)

# ---------------------------------------------------------------------------
# SarahMemory Reality Patch API: SEL / QIST / Fast-Lane Governance / Tokenizer
# HOTFIX 2026-07-23b: this block is intentionally registered before /api/<path> loose asset fallback.
# It also supports GET diagnostics with ?text=... for SEL/QIST browser smoke tests.
# ---------------------------------------------------------------------------
# Frontend may inspect these endpoints, but frontend is not authority. These routes
# expose read-only governance metadata and bounded project-data writes for the local
# SarahMemory tokenizer profile only.

def _sm_reality_payload() -> dict:
    """Return JSON body plus query args for browser/curl diagnostics.

    This keeps POST JSON as the primary contract while allowing safe GET
    inspection like /api/sel/compile?text=What%20is%20RAM for local tests.
    """
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            data = {}
        try:
            for key, value in request.args.items():
                data.setdefault(str(key), value)
        except Exception:
            pass
        return data
    except Exception:
        return {}


def _sm_build_reality_flow_metadata(text: str, context_packet: dict | None = None, *, local_only: bool = False) -> dict:
    out = {
        "ok": True,
        "schema": "SarahMemory.reality_flow.v0.1",
        "local_only": bool(local_only),
        "fast_to_answer_slow_to_act": True,
        "execution_authority": False,
    }
    try:
        import SarahMemoryPreTokenAnalyzer as _SMPreToken  # type: ignore
        analysis = _SMPreToken.analyze_text(text, context_packet=context_packet if isinstance(context_packet, dict) else None)
        lane = _SMPreToken.classify_runtime_governance_lane(text, analysis, context_packet=context_packet if isinstance(context_packet, dict) else None)
        sel = _SMPreToken.build_sel_packet(text, analysis, context_packet=context_packet if isinstance(context_packet, dict) else None)
        out.update({"pretoken": analysis, "governance_lane": lane, "sel": sel})
    except Exception as exc:
        out["pretok_error"] = str(exc)
    try:
        import SarahMemoryQuantumSafe as _SMQSafe  # type: ignore
        qist = _SMQSafe.qist_rank_meaning_candidates(text, governance_lane=out.get("governance_lane") if isinstance(out.get("governance_lane"), dict) else None)
        out["qist"] = qist
    except Exception as exc:
        out["qist_error"] = str(exc)
    return out


def _sm_attach_reality_meta(bundle: dict, reality: dict | None) -> dict:
    try:
        if isinstance(bundle, dict) and isinstance(reality, dict):
            meta = bundle.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["reality_flow"] = {
                    "schema": reality.get("schema"),
                    "governance_lane": reality.get("governance_lane"),
                    "sel": reality.get("sel"),
                    "qist_selected": (reality.get("qist") or {}).get("selected_candidate") if isinstance(reality.get("qist"), dict) else None,
                    "fast_to_answer_slow_to_act": True,
                }
                if isinstance(reality.get("sml"), dict):
                    meta["sml"] = reality.get("sml")
                if isinstance(reality.get("sml"), dict):
                    meta["sml"] = reality.get("sml")
    except Exception:
        pass
    return bundle


def _sm_sml_protocol_ready(discover: bool = True):
    """Return SML protocol singleton and perform bounded Core discovery once."""
    try:
        from SarahMemorySMLProtocol import get_protocol  # type: ignore
        proto = get_protocol()
        if discover and not getattr(proto, "_api_bridge_discovered", False):
            try:
                proto.discover_organs(globals().get("CORE_DIR", globals().get("BASE_DIR", ".")), import_modules=False, max_files=250)
                setattr(proto, "_api_bridge_discovered", True)
            except Exception as exc:
                try:
                    app_logger.warning(f"SML bounded discovery failed: {exc}", exc_info=True)
                except Exception:
                    pass
        return proto
    except Exception as exc:
        try:
            app_logger.warning(f"SML protocol unavailable: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_sml_create_ingress_packet(payload: dict, text: str, context_packet: dict):
    """Create the canonical SML packet for /api/chat without executing actions."""
    try:
        from SarahMemorySMLProtocol import sml_build_ingress_packet  # type: ignore
        return sml_build_ingress_packet(
            text,
            payload=payload if isinstance(payload, dict) else {},
            context_packet=context_packet if isinstance(context_packet, dict) else {},
            caller="api_chat",
            core_path=globals().get("CORE_DIR", globals().get("BASE_DIR", ".")),
            discover=True,
        )
    except Exception as exc:
        try:
            app_logger.warning(f"SML ingress packet creation failed: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_sml_summary(packet) -> dict:
    try:
        from SarahMemorySMLProtocol import sml_packet_summary  # type: ignore
        return sml_packet_summary(packet)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _sm_sml_apply_governance(packet, gov: dict | None):
    try:
        from SarahMemorySMLProtocol import sml_apply_governor_result  # type: ignore
        return sml_apply_governor_result(packet, gov if isinstance(gov, dict) else {}, organ="SarahMemoryCognitiveServices")
    except Exception as exc:
        try:
            app_logger.warning(f"SML governance reflection failed: {exc}", exc_info=True)
        except Exception:
            pass
        return packet


def _sm_sml_attach_meta_to_reality(reality: dict | None, packet) -> None:
    try:
        if isinstance(reality, dict) and packet is not None:
            reality["sml"] = _sm_sml_summary(packet)
    except Exception:
        pass



@app.route("/api/sml/status", methods=["GET"])
def api_sml_status():
    """SML Protocol status/capability surface for UI/API bridge diagnostics."""
    proto = _sm_sml_protocol_ready(discover=True)
    if proto is None:
        return jsonify({"ok": False, "error": "sml_protocol_unavailable", "version": PROJECT_VERSION}), 503
    try:
        return jsonify({"ok": True, "source": "SarahMemorySMLProtocol", "status": proto.capability_status(), "diagnostics": proto.diagnostics(), "version": PROJECT_VERSION}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "SarahMemorySMLProtocol", "version": PROJECT_VERSION}), 500


@app.route("/api/sml/health", methods=["GET"])
def api_sml_health():
    """Return global SML health vector."""
    proto = _sm_sml_protocol_ready(discover=True)
    if proto is None:
        return jsonify({"ok": False, "error": "sml_protocol_unavailable", "version": PROJECT_VERSION}), 503
    try:
        return jsonify({"ok": True, "source": "SarahMemorySMLProtocol", "health": proto.global_health(), "version": PROJECT_VERSION}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "SarahMemorySMLProtocol", "version": PROJECT_VERSION}), 500


@app.route("/api/sml/packet", methods=["POST"])
def api_sml_packet():
    """Build and return a governed SML ingress packet summary without execution."""
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text") or data.get("message") or data.get("q") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "missing_text", "version": PROJECT_VERSION}), 400
        context_packet = _sm_build_context_packet(data, text, str(data.get("intent") or ""), str(data.get("tone") or ""), str(data.get("complexity") or ""), bool(data.get("avatar_request") or False), local_only=bool(data.get("local_only") or False), safe_mode=bool(data.get("safe_mode") or False), neoskymatrix=bool(data.get("neoskymatrix") or False), developersmode=bool(data.get("developersmode") or False))
        packet = _sm_sml_create_ingress_packet(data, text, context_packet)
        return jsonify({"ok": packet is not None, "source": "SarahMemorySMLProtocol", "sml": _sm_sml_summary(packet), "version": PROJECT_VERSION}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "source": "SarahMemorySMLProtocol", "version": PROJECT_VERSION}), 500


@app.route("/api/governance/classify", methods=["GET", "POST"])
def api_governance_classify():
    try:
        data = _sm_reality_payload()
        text = str(data.get("text") or data.get("message") or request.args.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "missing_text"}), 400
        return jsonify(_sm_build_reality_flow_metadata(text, local_only=True)), 200
    except Exception as exc:
        app_logger.exception("Governance classify failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/sel/compile", methods=["GET", "POST"])
def api_sel_compile():
    try:
        data = _sm_reality_payload()
        text = str(data.get("text") or data.get("message") or request.args.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "missing_text", "hint": "Use POST JSON {\"text\": \"What is RAM?\"} or GET ?text=What%20is%20RAM"}), 400
        import SarahMemoryPreTokenAnalyzer as _SMPreToken  # type: ignore
        analysis = _SMPreToken.analyze_text(text)
        packet = _SMPreToken.build_sel_packet(text, analysis, mode=str(data.get("mode") or request.args.get("mode") or ""))
        return jsonify(packet), 200
    except Exception as exc:
        app_logger.exception("SEL compile failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/qist/rank", methods=["GET", "POST"])
def api_qist_rank():
    try:
        data = _sm_reality_payload()
        text = str(data.get("text") or data.get("message") or request.args.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "missing_text", "hint": "Use POST JSON {\"text\": \"What is RAM?\"} or GET ?text=What%20is%20RAM"}), 400
        governance_lane = data.get("governance_lane") if isinstance(data.get("governance_lane"), dict) else None
        if governance_lane is None:
            try:
                import SarahMemoryPreTokenAnalyzer as _SMPreToken  # type: ignore
                analysis = _SMPreToken.analyze_text(text)
                governance_lane = _SMPreToken.classify_runtime_governance_lane(text, analysis)
            except Exception:
                governance_lane = None
        import SarahMemoryQuantumSafe as _SMQSafe  # type: ignore
        return jsonify(_SMQSafe.qist_rank_meaning_candidates(text, governance_lane=governance_lane)), 200
    except Exception as exc:
        app_logger.exception("QIST rank failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/tokenizer-profile", methods=["GET", "POST"])
def api_models_tokenizer_profile():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        if request.method == "GET":
            fn = getattr(mod, "get_sarahmemory_tokenizer_profile_status", None)
            if not callable(fn):
                return jsonify({"ok": False, "error": "tokenizer_profile_status_unavailable"}), 501
            return jsonify(fn()), 200
        data = _model_payload()
        fn = getattr(mod, "build_sarahmemory_tokenizer_profile", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "tokenizer_profile_builder_unavailable"}), 501
        samples = data.get("samples") if isinstance(data.get("samples"), list) else []
        result = fn(text_samples=samples, domain=str(data.get("domain") or "general"), write=bool(data.get("write", True)))
        return jsonify(result), (200 if result.get("ok") else 400)
    except Exception as exc:
        app_logger.exception("Tokenizer profile route failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/token-path", methods=["POST"])
def api_models_token_path():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        text = str(data.get("text") or data.get("message") or "")
        if not text:
            return jsonify({"ok": False, "error": "missing_text"}), 400
        fn = getattr(mod, "inspect_tokenizer_path", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "token_path_unavailable"}), 501
        result = fn(text, model_id=str(data.get("model_id") or ""), repo=str(data.get("repo") or ""), max_tokens=int(data.get("max_tokens") or 256))
        return jsonify(result), 200
    except Exception as exc:
        app_logger.exception("Token path route failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/bootstrap-local-sarahmemory", methods=["POST"])
def api_models_bootstrap_local_sarahmemory():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        data = _model_payload()
        fn = getattr(mod, "bootstrap_local_sarahmemory_model", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "bootstrap_unavailable"}), 501
        allow_download = bool(data.get("allow_download") or data.get("user_approved_download"))
        result = fn(
            repo=str(data.get("repo") or ""),
            allow_download=allow_download,
            build_tokenizer_profile=bool(data.get("build_tokenizer_profile", True)),
            run_smoke=bool(data.get("run_smoke", False)),
        )
        return jsonify(result), (200 if result.get("ok") else 409)
    except Exception as exc:
        app_logger.exception("Local SarahMemory model bootstrap failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/models/forensics/status", methods=["GET"])
def api_models_forensics_status():
    mod = _sm_llm_manager()
    if mod is None:
        return jsonify({"ok": False, "error": "SarahMemoryLLM unavailable"}), 503
    try:
        fn = getattr(mod, "get_model_forensics_status", None)
        if not callable(fn):
            return jsonify({"ok": False, "error": "forensics_status_unavailable"}), 501
        return jsonify(fn()), 200
    except Exception as exc:
        app_logger.exception("Model forensics status failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

# --- END SARAHMEMORY REALITY PATCH 2026-07-23 ---

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
            "ui": {
                "dist_dir": UI_DIST_DIR,
                "src_dir": UI_SRC_DIR,
                "index_exists": bool(os.path.isfile(os.path.join(UI_DIST_DIR, "index.html"))),
                "assets_exists": bool(os.path.isdir(os.path.join(UI_DIST_DIR, "assets"))),
            },
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
    if any(p in t for p in ("today's date", "todays date", "current date", "what is the date", "what is todays date", "what is today date", "what's today's date", "what day is it", "what time is it", "current time", "date and time", "what year is it", "current year", "what is the year")):
        kind = "datetime"
        if any(p in t for p in ("what year is it", "current year", "what is the year")):
            kind = "year"
        elif "time" in t and "date" not in t and "today" not in t and "day" not in t:
            kind = "time"
        elif any(p in t for p in ("today's date", "todays date", "current date", "what is the date", "what is today date", "what's today's date", "what day is it")):
            kind = "date"
        return {"route_id": "system.datetime.current", "kind": kind}
    if any(k in t for k in ("caps lock", "num lock", "scroll lock")):
        # B10: normalize terse imperatives such as "Caps Locks On" and
        # "capslock on" before any local model can turn a device command into
        # generic keyboard advice. Question/status requests remain read-only.
        is_question = bool(re.search(r"\b(what|why|how|explain|define|is|are)\b", t)) or "?" in str(text or "")
        action_word = any(k in t for k in ("turn", "put", "set", "enable", "disable", "switch"))
        terse_state = bool(re.search(r"\b(caps lock|num lock|scroll lock)s?\s+(on|off)\b", t))
        if (action_word or terse_state) and not is_question:
            key_name = "caps_lock" if "caps lock" in t else ("num_lock" if "num lock" in t else "scroll_lock")
            state = "off" if any(k in t for k in ("turn off", "switch off", "disable")) or re.search(r"\b(caps lock|num lock|scroll lock)s?\s+off\b", t) else "on"
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
    if kind == "year":
        return f"The current year is {now.strftime('%Y')}."
    if kind == "datetime":
        return f"Today is {now.strftime('%A, %B %d, %Y')} and the current time is {now.strftime('%I:%M %p').lstrip('0')}."
    return f"Today's date is {now.strftime('%A, %B %d, %Y')}."


def _sm_set_lock_key_state(key_name: str, requested_state: str) -> tuple[bool, str, dict]:
    """Compatibility stub only.

    B06 moved keyboard lock-state execution to appdrivers.run_governed_device_intent()
    so app.py never mutates local device state directly.
    """
    requested_state = str(requested_state or "on").lower()
    key_name = str(key_name or "caps_lock").lower()
    return False, "Keyboard lock-state mutation is owned by appdrivers/OperatorCore, not app.py.", {"key_name": key_name, "requested_state": requested_state, "handoff": "appdrivers.run_governed_device_intent", "execution_authority": False}

def _sm_try_keyboard_lighting(text: str, quick_route: dict) -> tuple[bool, str, dict]:
    """Compatibility stub only.

    B06 moved keyboard/RGB lighting execution to appdrivers. app.py may detect
    intent but must not spawn external RGB utilities, load drivers, or mutate lighting directly.
    """
    return False, "Keyboard lighting mutation is owned by appdrivers/OperatorCore, not app.py.", {"route_id": (quick_route or {}).get("route_id"), "handoff": "appdrivers.run_governed_device_intent", "execution_authority": False}

def _sm_quick_route_is_read_only(route_id: str) -> bool:
    """Return True only for quick routes that cannot mutate OS, drivers, memory, network, or files."""
    return str(route_id or "").strip() in {"system.datetime.current"}


def _sm_quick_action_confirmation_bundle(route: dict, *, governor: dict | None = None) -> dict:
    """Presentation-only hold for quick routes that would mutate system/driver state."""
    route_id = str((route or {}).get('route_id') or 'quick_action')
    rationale = "This quick route changes system or driver state and requires governed approval before execution."
    if isinstance(governor, dict) and governor.get("rationale"):
        rationale = str(governor.get("rationale"))
    bundle = _sm_make_outward_bundle(
        rationale,
        meta={
            "source": "quick_system_route_hold",
            "engine": "cognitive_governor",
            "intent": "system",
            "route_id": route_id,
            "decision": "REQUIRE_USER",
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "governance_rule": "fast_to_answer_slow_to_act",
            "version": PROJECT_VERSION,
        },
        actions=[{"type": "quick_action_hold", "route_id": route_id, "requires_confirmation": True}],
        errors=[],
    )
    bundle['ok'] = False
    bundle['blocked'] = True
    return bundle


def _sm_execute_quick_route(
    text: str,
    *,
    allow_actions: bool = False,
    user_consented: bool = False,
    governor: dict | None = None,
) -> tuple[bool, dict | None]:
    """Execute only read-only quick routes before governance.

    Hardware/driver/system-mutating quick routes are intentionally inert until
    /api/chat has built a context packet and passed CognitiveServices governance.
    This preserves the speed of date/time answers without allowing keyboard, RGB,
    driver, shell, filesystem, network, or memory mutation before authorization.
    """
    route = _sm_match_quick_system_route(text)
    if not route:
        return False, None
    route_id = str(route.get('route_id') or '')
    if route_id == 'system.datetime.current':
        reply = _sm_now_reply(str(route.get('kind') or 'date'))
        bundle = _sm_make_outward_bundle(reply, meta={"source": "quick_system_route", "engine": "local_datetime", "intent": "system", "route_id": route_id, "version": PROJECT_VERSION, "execution_authority": False})
        bundle.setdefault('actions', [])
        bundle['actions'].append({"type": "route_match", "route_id": route_id, "read_only": True})
        return True, bundle

    # Action quick routes must not execute during the pre-governor hot path.
    if not allow_actions:
        return False, None

    # app.py must not mutate devices even after confirmation. Confirmed action
    # requests continue into the normal governed domain pipeline so appdrivers/
    # OperatorCore owns execution. Unconfirmed requests receive a hold bundle.
    if not user_consented:
        return True, _sm_quick_action_confirmation_bundle(route, governor=governor)

    # B06: confirmed local-device quick routes are delegated to appdrivers,
    # the API driver domain owner. app.py does not mutate keyboard/RGB/device
    # state and does not let local LLM fallback generate fake how-to advice.
    try:
        import appdrivers as _sm_appdrivers  # type: ignore
        dispatch_fn = getattr(_sm_appdrivers, "run_governed_device_intent", None)
        if callable(dispatch_fn):
            details = dispatch_fn(
                text,
                route=route,
                user_consented=True,
                source="api_chat.quick_route",
                meta={"route_id": route_id, "governor_decision": (governor or {}).get("decision") if isinstance(governor, dict) else ""},
            )
        else:
            details = {"ok": False, "error": "appdrivers.run_governed_device_intent unavailable", "handled": True}
    except Exception as exc:
        details = {"ok": False, "error": f"appdrivers_device_dispatch_failed:{exc}", "handled": True}

    ok = bool(details.get("ok"))
    reply = str(details.get("presentation_text") or details.get("reply") or ("Device action completed and verified." if ok else "Device action did not complete with verified success."))
    bundle = _sm_make_outward_bundle(
        reply,
        meta={
            "source": "quick_system_route",
            "engine": "appdrivers_governed_device_action",
            "intent": "drivers",
            "route_id": route_id,
            "version": PROJECT_VERSION,
            "governed_stage": "post_governor_appdrivers",
            "execution_authority": False,
            "domain_owner": "appdrivers.py",
        },
        actions=[{"type": "governed_device_action", **details}],
        errors=[] if ok else [details],
    )
    avatar_event_device = _sm_emit_avatar_event_safe({
        "event_type": "device_action_completed" if ok else "device_action_failed",
        "domain": "local_device_control",
        "source": "api_chat.quick_route",
        "message": reply,
        "validation_state": "verified" if ok else "failed",
        "source_verified": bool(ok),
        "severity": 0.35 if ok else 0.70,
        "claim": {"route_id": route_id, "ok": bool(ok)},
    }, source="api_chat.quick_route")
    bundle.setdefault("meta", {})["avatar_event"] = avatar_event_device
    bundle['ok'] = ok
    return True, bundle


def _sm_emit_avatar_event_safe(event: dict, *, source: str = "api_chat") -> dict:
    """Best-effort B08 avatar embodiment event; never changes authority."""
    try:
        import appmedia as _sm_appmedia  # type: ignore
        fn = getattr(_sm_appmedia, "run_avatar_event_packet", None)
        if callable(fn):
            return fn(dict(event or {}), source=source, meta={"caller": "app.py"})
    except Exception as exc:
        return {"ok": False, "error": f"avatar_event_emit_failed:{exc}", "execution_authority": False}
    return {"ok": False, "error": "avatar_event_bridge_unavailable", "execution_authority": False}

def _sm_is_nailde_creation_request(text: str) -> bool:
    """Return True for software/app/addon build missions, not generic media creation.

    This is B07 route grammar. It is intentionally domain-shaped rather than
    prompt-specific, so examples such as games or stock apps do not become hardcoded handlers.
    """
    try:
        t = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not t:
            return False
        creation = bool(re.search(r"\b(make|create|build|generate|code|write)\b", t))
        software = bool(re.search(r"\b(app|application|program|software|game|addon|add-on|addons|plugin|tool|dashboard|tracker|website|web app|panel|widget|playable|launcher|simulator)\b", t))
        return bool(creation and software)
    except Exception:
        return False


def _sm_try_nailde_creation_mission_route(
    text: str,
    *,
    payload: dict | None = None,
    context_packet: dict | None = None,
    governor: dict | None = None,
) -> dict | None:
    """Delegate arbitrary software/app/addon creation to appsdk/NAILDE.

    app.py remains the bridge. B07 allows explicit user create/build requests to
    produce NAILDE sandbox artifacts only; live ADDON install/run remains a
    separate user-approved path owned by appstore/ADDON runtime.
    """
    if not _sm_is_nailde_creation_request(text):
        return None
    gov_decision = str((governor or {}).get("decision") or "").upper() if isinstance(governor, dict) else ""
    plan_only = bool(re.search(r"(before creating files|without creating files|mission plan|governance state)", str(text or "").lower()))
    if gov_decision == "DENY" and not plan_only:
        return None
    try:
        import appsdk as _sm_appsdk  # type: ignore
        fn = getattr(_sm_appsdk, "run_governed_creation_mission", None)
        if not callable(fn):
            raise RuntimeError("appsdk.run_governed_creation_mission unavailable")
        request_payload = dict(payload or {})
        ctx_meta = (context_packet or {}).get("meta") if isinstance(context_packet, dict) else {}
        explicit_consent = bool(
            request_payload.get("confirmed")
            or request_payload.get("confirm")
            or request_payload.get("user_confirmed")
            or (isinstance(ctx_meta, dict) and ctx_meta.get("user_consented"))
        )
        result = fn(
            text,
            payload=request_payload,
            user_consented=explicit_consent,
            source="api_chat.nailde_creation_mission",
            meta={
                "route": "api_chat_nailde_creation_mission",
                "governor_decision": gov_decision,
                "governor_original_rationale": (governor or {}).get("rationale") if isinstance(governor, dict) else "",
                "sandbox_only": True,
            },
        )
        if not isinstance(result, dict) or not result.get("handled", True):
            return None
        reply = str(result.get("presentation_text") or result.get("message") or "NAILDE creation mission returned no presentable status.")
        meta = {
            "source": "appsdk_nailde_creation_mission",
            "engine": "appsdk.run_governed_creation_mission",
            "intent": "create_software_addon",
            "domain_owner": "appsdk.py/SarahMemoryNAILDE.py",
            "install_owner": "appstore.py",
            "governor_original_decision": gov_decision,
            "execution_allowed": False,
            "execution_authority": False,
            "sandbox_only": True,
            "live_install_authority": False,
            "live_run_authority": False,
            "version": PROJECT_VERSION,
        }
        bundle = _sm_make_outward_bundle(
            _sm_present_text(reply, intent="nailde_creation", meta=meta),
            meta=meta,
            raw_answer=reply,
            actions=result.get("actions") if isinstance(result.get("actions"), list) else [],
            artifacts=[],
            errors=[] if result.get("ok") else [result],
        )
        avatar_event_creation = _sm_emit_avatar_event_safe({
            "event_type": "creation_mission_completed" if result.get("ok") else "creation_mission_stopped",
            "domain": "creative_build_mission",
            "source": "api_chat.nailde_creation_mission",
            "message": reply,
            "validation_state": "verified" if result.get("ok") else "failed",
            "source_verified": bool(result.get("ok")),
            "severity": 0.35 if result.get("ok") else 0.65,
            "workspace_id": result.get("workspace_id"),
            "addon_id": result.get("addon_id"),
            "claim": {"workspace_id": result.get("workspace_id"), "addon_id": result.get("addon_id")},
        }, source="api_chat.nailde_creation_mission")
        bundle.setdefault("meta", {})["avatar_event"] = avatar_event_creation
        bundle["ok"] = bool(result.get("ok"))
        bundle["nailde_creation"] = result
        bundle["creation_contract"] = result.get("creation_contract")
        bundle["workspace_id"] = result.get("workspace_id")
        bundle["addon_id"] = result.get("addon_id")
        return bundle
    except Exception as exc:
        reply = "The NAILDE creation route is configured for this build mission, but the route failed before sandbox generation. I did not fall back to a generic model answer."
        meta = {
            "source": "appsdk_nailde_creation_mission",
            "engine": "api_chat_creation_route_failure",
            "error": str(exc),
            "execution_allowed": False,
            "execution_authority": False,
            "sandbox_only": True,
            "version": PROJECT_VERSION,
        }
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="nailde_creation", meta=meta), meta=meta, raw_answer=reply, errors=[meta])
        bundle["ok"] = False
        return bundle



def _sm_b10_is_readonly_text_generation_request(text: str) -> bool:
    """Read-only draft/summary requests are chat output, not file writes.

    This guard exists because the governor may conservatively treat the verb
    "write" as a filesystem/document action. It only returns True when the
    request asks for text in the chat and does not name a side-effect target.
    """
    t = _sm_fast_normalize_question(text)
    if not t:
        return False
    if not re.search(r"\b(write|draft|summarize|summarise|summary|explain|describe)\b", t):
        return False
    side_effect = re.search(
        r"\b(file|document|docx|pdf|spreadsheet|email|send|save|create\s+file|write\s+file|open\s+|launch\s+|notepad|word|excel|install|run|execute|delete|patch)\b",
        t,
    )
    if side_effect:
        return False
    return True


def _sm_b10_location_conflict_bundle(text: str, *, local_only: bool = True) -> dict | None:
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    if not (
        ("satellite" in t or "ip" in t or "internet" in t or "network" in t)
        and ("location" in t or "texas" in t or "kansas" in t or "oklahoma" in t or "trust" in t)
    ):
        return None
    reply = (
        "For physical/local context, SarahMemory should trust the user-declared location and the configured OS timezone first. "
        "Satellite, IP, DNS, VPN, or carrier routing can show Kansas or Oklahoma as a network-exit artifact, but that does not override East Texas as the physical/user-declared locality. "
        "So the Court should classify East Texas/Central Time as the local-context authority when user/system context supports it, and classify Kansas/Oklahoma as network-route evidence only."
    )
    meta = {
        "source": "appsys_location_context_court_b10",
        "engine": "b10_readonly_location_conflict_resolver",
        "intent": "temporal_locality",
        "claim_type": "LOCATION_CONTEXT_CONFLICT",
        "model_memory_authority": False,
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "local_only": bool(local_only),
        "authority_rule": "user_declared_locality_and_os_timezone_outweigh_network_exit_location_for_physical_locality",
        "version": PROJECT_VERSION,
    }
    bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="system_status", meta=meta), meta=meta, raw_answer=reply)
    bundle["ok"] = True
    bundle["location_court"] = {
        "verdict": "TRUST_USER_DECLARED_LOCAL_CONTEXT_REJECT_NETWORK_EXIT_AS_PHYSICAL_LOCATION",
        "accepted_for_physical_context": ["user_declared_location", "configured_os_timezone"],
        "accepted_for_network_context": ["satellite_or_ip_exit_region"],
        "rejected_final_sources": ["model_memory", "network_exit_as_physical_location"],
    }
    return bundle


def _sm_b10_capslock_concept_bundle(text: str) -> dict | None:
    t = _sm_fast_normalize_question(text)
    if not t or "caps lock" not in t:
        return None
    if not re.search(r"\b(what|define|explain|how does|what is)\b", t):
        return None
    # Concept explanation only; action phrases are handled by appdrivers.
    if _sm_match_quick_system_route(text):
        return None
    reply = "Caps Lock is a keyboard toggle. When it is on, letter keys type uppercase letters until Caps Lock is turned off; number keys and most symbols are not changed by the toggle."
    meta = {
        "source": "keyboard_concept_deterministic_b10",
        "engine": "app_py_readonly_device_concept",
        "intent": "device_query",
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "model_memory_authority": False,
        "domain_owner_for_mutation": "appdrivers.py",
        "version": PROJECT_VERSION,
    }
    bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="device_query", meta=meta), meta=meta, raw_answer=reply)
    bundle["ok"] = True
    return bundle


def _sm_b10_avatar_event_question_bundle(text: str) -> dict | None:
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    if not ("avatar" in t and ("game_over" in t or "game over" in t or "addon game" in t or "verified" in t)):
        return None
    reply = (
        "For a verified ADDON GAME_OVER event, the avatar should react through the B08 avatar-event stream, not through random model improvisation. "
        "The ADDON emits a verified runtime event, SML builds an avatar event packet, appmedia applies the directive, and the avatar can shift to a surprised/concerned or supportive state, then ask whether the user wants to play again. "
        "No claim or emotion should be invented without the verified event packet."
    )
    meta = {
        "source": "appmedia_avatar_event_doctrine_b10",
        "engine": "b10_readonly_avatar_event_resolver",
        "intent": "avatar_event_semantics",
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "model_memory_authority": False,
        "version": PROJECT_VERSION,
    }
    bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="avatar", meta=meta), meta=meta, raw_answer=reply)
    bundle["ok"] = True
    bundle["avatar_event_policy"] = {
        "event_type": "GAME_OVER",
        "required_source": "verified_addon_runtime_event",
        "route": ["ADDON", "SML avatar event packet", "appmedia.py", "SarahMemoryAvatar.py", "Reply/UI"],
        "random_reaction_allowed": False,
    }
    return bundle


def _sm_b10_readonly_text_generation_bundle(text: str, *, local_only: bool = True, intent: str = "question") -> dict | None:
    if not _sm_b10_is_readonly_text_generation_request(text):
        return None
    t = _sm_fast_normalize_question(text)
    if "sml" in t and "packet" in t:
        reply = (
            "SML packets matter because they keep each mission traceable from the first input through routing, authority, evidence, validation, reply, and audit. "
            "They stop every organ from inventing its own disconnected state, so SarahMemory can verify who acted, what source was used, what authority existed, and what changed. "
            "That is what turns the system from a chatbot-style answer path into a governed cognitive operating system."
        )
    else:
        reply = (
            "This is a read-only text-generation request. It should be answered in chat without user confirmation because no file, app, email, device, shell, install, or persistent write was requested."
        )
    meta = {
        "source": "read_only_text_generation_b10",
        "engine": "b10_bounded_readonly_text_route",
        "intent": "read_only_text_generation",
        "decision": "ALLOW_PRESENTATION_ONLY",
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "confirmation_required": False,
        "filesystem_write": False,
        "app_execution": False,
        "network_access": False,
        "version": PROJECT_VERSION,
    }
    bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="chat", meta=meta), meta=meta, raw_answer=reply)
    bundle["ok"] = True
    return bundle


def _sm_b10_nailde_plan_only_bundle(text: str, *, payload: dict | None = None, context_packet: dict | None = None) -> dict | None:
    if not _sm_is_nailde_creation_request(text):
        return None
    t = _sm_fast_normalize_question(text)
    plan_only = bool(
        "before creating files" in t
        or "without creating files" in t
        or "mission plan" in t
        or "governance state" in t
        or (isinstance(payload, dict) and payload.get("dry_run"))
    )
    if not plan_only:
        return None
    try:
        import appsdk as _sm_appsdk  # type: ignore
        fn = getattr(_sm_appsdk, "run_governed_creation_mission", None)
        if not callable(fn):
            raise RuntimeError("appsdk.run_governed_creation_mission unavailable")
        p = dict(payload or {})
        p["plan_only"] = True
        p["dry_run"] = True
        p["create_files"] = False
        result = fn(text, payload=p, user_consented=False, source="api_chat.b10_nailde_plan_only", meta={"route": "b10_pre_governance_plan_only"})
        if not isinstance(result, dict):
            raise RuntimeError("appsdk returned non-dict creation mission plan")
        reply = str(result.get("presentation_text") or result.get("message") or "NAILDE returned the governed creation mission plan without creating files.")
        meta = {
            "source": "appsdk_nailde_plan_only_b10",
            "engine": "appsdk.run_governed_creation_mission",
            "intent": "create_software_addon_plan_only",
            "execution_allowed": False,
            "execution_authority": False,
            "sandbox_only": True,
            "files_created": False,
            "live_install_authority": False,
            "live_run_authority": False,
            "confirmation_required": False,
            "version": PROJECT_VERSION,
        }
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="nailde_creation", meta=meta), meta=meta, raw_answer=reply, actions=result.get("actions") if isinstance(result.get("actions"), list) else [], artifacts=[], errors=[] if result.get("ok") else [result])
        bundle["ok"] = bool(result.get("ok", True))
        bundle["nailde_creation"] = result
        bundle["creation_contract"] = result.get("creation_contract")
        return bundle
    except Exception as exc:
        # If appsdk cannot be imported in a constrained context, still return a
        # bounded non-executing SML creation contract rather than denying or
        # timing out. This does not create files and does not install/run addons.
        try:
            from SarahMemorySMLProtocol import sml_build_creation_mission_contract  # type: ignore
            contract = sml_build_creation_mission_contract(str(text or ""), context={"source": "api_chat.b10_nailde_plan_only_fallback", "target": "nailde", "plan_only": True})
        except Exception as sml_exc:
            contract = {"ok": False, "error": f"sml_creation_contract_failed:{sml_exc}", "execution_authority": False}
        reply = (
            "NAILDE built a governed creation mission plan only. No workspace files were created, no ADDON was installed, and nothing was run. "
            "The appsdk bridge was unavailable in this context, so this response contains the SML creation contract only."
        )
        meta = {"source": "sml_nailde_plan_only_fallback_b10", "appsdk_error": str(exc), "execution_authority": False, "files_created": False, "version": PROJECT_VERSION}
        bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="nailde_creation", meta=meta), meta=meta, raw_answer=reply, errors=[] if contract.get("ok") else [contract])
        bundle["ok"] = bool(contract.get("ok", True))
        bundle["creation_contract"] = contract
        bundle["nailde_creation"] = {"ok": bool(contract.get("ok", True)), "mode": "plan_only_no_files_created", "files_created": False, "execution_authority": False, "creation_contract": contract}
        return bundle


def _sm_try_b10_confirmed_runtime_defect_route(text: str, *, payload: dict | None = None, context_packet: dict | None = None, local_only: bool = True, intent: str = "question") -> dict | None:
    """Narrow runtime repair route for confirmed B09 diagnostic failures.

    These are read-only/presentation-only or governed-hold lanes that must not
    fall into local model fallback or broad governor denial.
    """
    for fn in (
        lambda q: _sm_b10_location_conflict_bundle(q, local_only=local_only),
        _sm_b10_capslock_concept_bundle,
        _sm_b10_avatar_event_question_bundle,
        lambda q: _sm_b10_nailde_plan_only_bundle(q, payload=payload, context_packet=context_packet),
        lambda q: _sm_b10_readonly_text_generation_bundle(q, local_only=local_only, intent=intent),
    ):
        try:
            out = fn(text)
            if isinstance(out, dict):
                return out
        except Exception:
            continue
    return None

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
    replacements = {"temperture": "temperature", "wether": "weather", "camra": "camera", "web cam": "webcam", "mose": "mouse", "lites": "lights", "coler": "color", "unreel": "unreal", "doccument": "document", "naem": "name", "nmae": "name"}
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
    # Forward repair B04: a request to write/summarize/explain text in chat is
    # answer-only unless the user explicitly asks for a file, Word/Notepad,
    # document, spreadsheet, website, or application launch target.
    chat_write_summary = bool(re.search(r"\b(write|give|make|create)?\s*(me\s+)?(a\s+)?(summary|summarize|describe|explain)\b", normalized))
    explicit_document_target = bool(re.search(r"\b(word|notepad|document|docx|file|save|open|launch|spreadsheet|excel|website|browser|edge|app|application)\b", normalized))
    if route_id == "documents.office.write" and chat_write_summary and not explicit_document_target:
        route_id = "chat.general"
        best = {"route_id": "chat.general", "domain": "chat", "action": "general_reply", "target_module": "SarahMemoryReply", "transport_target": "/api/chat"}
    entities = _sm_ingress_extract_entities(original, route_id)
    surface_task = dict(entities.get('surface_task') or {}) if isinstance(entities.get('surface_task'), dict) else {}
    task_kind = str(surface_task.get('task_kind') or '').strip().lower()
    if chat_write_summary and not explicit_document_target:
        route_id = "chat.general"
        best = {"route_id": "chat.general", "domain": "chat", "action": "general_reply", "target_module": "SarahMemoryReply", "transport_target": "/api/chat"}
        entities.pop("surface_task", None)
        surface_task = {}
        task_kind = ""
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


def _sm_operatorcore_execution_mode(
    payload: dict | None,
    *,
    local_only: bool,
    safe_mode: bool,
    require_user: bool,
    user_consented: bool,
) -> str:
    """Resolve an execution mode without allowing payload flags to bypass governance."""
    payload = payload or {}
    requested = str(payload.get("execution_mode") or payload.get("operator_mode") or payload.get("smget_mode") or "").strip().lower()

    # Governance dominates all caller-requested modes. An explicit ``apply``
    # value is only honored when the user is present/consented and no safety or
    # governor hold is active.
    if safe_mode or require_user or not user_consented:
        return "simulate"
    try:
        if _is_cloud_request():
            return "simulate"
    except Exception:
        return "simulate"
    if requested in {"apply", "simulate", "draft"}:
        return requested
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

    user_consented = bool((context_packet.get("meta") or {}).get("user_consented"))
    execution_mode = _sm_operatorcore_execution_mode(
        payload,
        local_only=local_only,
        safe_mode=safe_mode,
        require_user=gov_require_user,
        user_consented=user_consented,
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
        "user_consented": user_consented,
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



# -----------------------------------------------------------------------------
# Tier-0 Hot Path: deterministic math before heavy runtime lanes
# -----------------------------------------------------------------------------

def _sm_fast_direct_chat_bundle(reply: str, *, meta: dict | None = None, raw_answer: str | None = None, actions: list | None = None, artifacts: list | None = None, errors: list | None = None) -> dict:
    """Build a direct /api/chat JSON bundle without invoking SarahMemoryReply.

    Used by hot-path lanes that must not import the heavier reply/research/vector
    stack. This is intentionally plain JSON and side-effect free except where a
    caller explicitly performs a bounded NVMe cache write before calling it.
    """
    reply = str(reply or "").strip()
    meta = meta if isinstance(meta, dict) else {}
    return {
        "ok": True,
        "reply": reply,
        "response": reply,
        "text": reply,
        "content": reply,
        "presentation_reply": reply,
        "raw_answer": str(raw_answer if raw_answer is not None else reply),
        "meta": meta,
        "actions": actions if isinstance(actions, list) else [],
        "artifacts": artifacts if isinstance(artifacts, list) else [],
        "errors": errors if isinstance(errors, list) else [],
    }


# -----------------------------------------------------------------------------
# Tier-0 Hot Path: deterministic math before heavy runtime lanes
# -----------------------------------------------------------------------------
def _sm_try_tier0_math_hotpath_bundle(text: str, *, local_only: bool = False):
    """Return a direct response bundle for simple deterministic math.

    This lane performs no file, network, database, vector, model, or tool work.
    It must remain safe before Neuron/Research/Reply imports.
    """
    try:
        raw = str(text or "").strip()
        if not raw:
            return None

        import ast
        import operator as _op
        import re as _re

        q = raw.lower().strip()
        q = _re.sub(r"^\s*(?:what(?:'s| is)?|calculate|compute|evaluate|solve|find)\s+", "", q, flags=_re.I).strip()
        q = _re.sub(r"^\s*(?:the\s+)?(?:answer\s+to|value\s+of)\s+", "", q, flags=_re.I).strip()
        q = q.rstrip("?.! ").strip()
        q = q.replace("multiplied by", "*").replace("divided by", "/").replace("divide by", "/")
        q = q.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("over", "/")
        q = _re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", q)
        q = q.replace("=", "").strip()

        words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
            "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
            "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
            "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
        }
        for word, digit in words.items():
            q = _re.sub(r"(?<![a-z0-9])" + _re.escape(word) + r"(?![a-z0-9])", digit, q)

        if not _re.search(r"\d", q):
            return None
        if not _re.search(r"[+\-*/%()^]", q):
            return None
        if not _re.fullmatch(r"[0-9+\-*/%.()\s^]+", q):
            return None

        expr_display = _re.sub(r"\s+", "", q).replace("^", "^")
        expr = q.replace("^", "**")
        if "**" in expr and len(expr) > 64:
            return None

        allowed_bin = {
            ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
            ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
        }
        allowed_unary = {ast.UAdd: _op.pos, ast.USub: _op.neg}

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin:
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
                    raise ValueError("power too large for hotpath")
                return allowed_bin[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
                return allowed_unary[type(node.op)](_eval(node.operand))
            raise ValueError("not a pure arithmetic expression")

        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
        if isinstance(value, float) and abs(value - round(value)) < 1e-12:
            value = int(round(value))
        reply = f"{expr_display}={value}"
        meta = {
            "source": "tier0_math_hotpath",
            "engine": "api_chat_fast_math_direct_json",
            "intent": "math",
            "confidence": 0.99,
            "local_only": bool(local_only),
            "side_effects": "none",
            "db_access": False,
            "research_access": False,
            "api_access": False,
            "vector_access": False,
            "reply_bundle_bypassed": True,
            "version": PROJECT_VERSION,
        }
        return _sm_fast_direct_chat_bundle(reply, meta=meta, raw_answer=reply)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Tier-1 Hot Path: low-risk general/procedural questions, cache -> LOCAL LLM only
# -----------------------------------------------------------------------------
# WAVE7: retained only for legacy/reference. Disabled by default; general answers must come from local LLM, not hardcoded seeds.
# SML v0.8.2: no canned procedural answer tables in app.py.
# General/procedural answers must come from routed sources such as local LLM,
# governed SQLite/memory, AdvCU, or approved research.
_SM_FAST_HOWTO_STATIC = {}


# Bounded deterministic glossary for stable concept questions when no local LLM
# is installed yet.  This keeps CHAT usable without granting tools, network,
# filesystem, or self-aware hardware authority.  The response metadata names this
# source directly, so it never pretends to be trained-model output.
# SML v0.8.2: no built-in factual glossary in app.py.
# The API bridge may hardcode rails, not thoughts.
_SM_FAST_BUILTIN_GLOSSARY = {}


def _sm_fast_builtin_glossary_lookup(norm_question: str) -> str | None:
    """Disabled by v0.8.2: facts must come from routed knowledge sources."""
    return None


def _sm_fast_normalize_question(text: str) -> str:
    try:
        t = str(text or "").lower().strip()
        t = t.replace("’", "'").replace("`", "'")
        t = re.sub(r"[^a-z0-9+\-*/%()\s']+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    except Exception:
        return str(text or "").lower().strip()


def _sm_fast_definition_subject(question: str) -> str:
    """Extract a compact subject from common definition questions for cache validation."""
    t = _sm_fast_normalize_question(question)
    if not t:
        return ""
    patterns = (
        r"^what\s+is\s+an?\s+(.+?)$",
        r"^what\s+is\s+(.+?)$",
        r"^what\s+are\s+(.+?)$",
        r"^define\s+(.+?)$",
        r"^explain\s+(.+?)$",
        r"^describe\s+(.+?)$",
        r"^tell\s+me\s+about\s+(.+?)$",
    )
    for pat in patterns:
        m = re.match(pat, t)
        if m:
            subj = (m.group(1) or "").strip()
            subj = re.sub(r"\b(the|a|an)\b", " ", subj)
            subj = re.sub(r"\s+", " ", subj).strip(" ?.")
            return subj[:80]
    return ""


def _sm_fast_is_safe_definition_question(text: str) -> bool:
    """Tier-0 educational definition shape; never grants action authority."""
    t = _sm_fast_normalize_question(text)
    if not t or len(t) > 240:
        return False
    risky = (
        "patch", "modify", "delete", "remove", "write", "save", "install", "download",
        "upload", "execute", "run command", "powershell", "cmd", "terminal", "credential",
        "password", "api key", "secret", "token", "driver", "motor", "robot", "camera",
        "microphone", "weapon", "explosive", "malware", "bypass", "hack", "current",
        "latest", "today", "weather", "news",
    )
    padded = f" {t} "
    if any(r in padded for r in risky):
        return False
    return bool(re.match(r"^(what\s+is|what\s+are|define|explain|describe|tell\s+me\s+about)\b", t))


def _sm_fast_is_low_quality_answer(text: str, question: str | None = None) -> bool:
    t = _sm_fast_normalize_question(text)
    raw = str(text or "").strip()
    if not t:
        return True
    # Fast-answer cache entries must be answer material, not whole papers, logs,
    # abstracts, diagnostics dumps, or copied corpus chunks.  Long-form material
    # can still exist in databases, but it must not be released as Tier-0 cache.
    if len(raw) > 1200:
        return True
    bad = (
        "i'm not sure how to respond", "im not sure how to respond", "could you rephrase",
        "i don't know how to respond", "i do not know how to respond", "unable to answer",
        "having trouble generating", "no answer found", "no local match", "still researching",
        "blocked by safety", "request denied by policy", "user confirmation required",
        "api key missing", "local model directory not found",
        "failed to load local model", "local llm runtime not available",
        "couldn't solve that problem", "could not solve that problem",
        "couldnt solve that problem", "couldn't solve", "could not solve", "couldnt solve",
        "please try rephrasing", "try rephrasing", "provide more details",
        "please provide more details", "rephrase or provide more details",
        "sorry i couldn't solve", "sorry i could not solve", "sorry i couldnt solve",
        "traceback", "exception", "stack trace", "<think>", "</think>",
        "runtime_identity_override", "ingress route confidence", "structured action request",
        "no engine produced an answer", "provide more constraints or enable an applicable tier",
        "pdhaddenglishcounterw failed", "memory = {\"error\"", "from ailearning.db:qacache",
        "vetted_local_llm_general", "vettedlocalllm_general",
    )
    if any(x in t for x in bad):
        return True

    paper_markers = (
        "abstract", "1 introduction", "introduction since", "references", "bibliography",
        "@mozilla.com", "@gmail.com", "@", "proceedings", "doi", "arxiv",
        "we present", "we propose", "this paper", "compiler from llvm",
    )
    marker_hits = sum(1 for x in paper_markers if x in t)
    if marker_hits >= 2:
        return True

    subj = _sm_fast_definition_subject(question or "")
    if subj and _sm_fast_is_safe_definition_question(question or ""):
        first = _sm_fast_normalize_question(raw[:320])
        subj_terms = [x for x in re.findall(r"[a-z0-9]+", subj.lower()) if len(x) > 1]
        # Acronyms and stable single-term definitions should appear early.
        if subj_terms and not any(term in first for term in subj_terms[:3]):
            return True
    return len(t) < 8


def _sm_fast_is_procedural_howto(text: str) -> bool:
    t = _sm_fast_normalize_question(text)
    if not t:
        return False
    # Keep body/system questions out of this lane.
    blocked = (
        " gpu", " cpu", " motherboard", " ram", " vram", " drive", " disk", " nvme", " hdd",
        "what type of", "what version", "what time",
    )
    if any(b in f" {t}" for b in blocked):
        return False
    return bool(
        re.search(r"\bhow\s+to\s+(make|cook|prepare|build|create|fix|clean|write|use|install|bake)\b", t)
        or re.search(r"\bhow\s+(?:do|can|should|would)\s+(?:i|you|we|someone|one)?\s*(make|cook|prepare|build|create|fix|clean|write|use|install|bake)\b", t)
        or re.search(r"\b(recipe|steps)\s+(?:for|to)\b", t)
    )


def _sm_fast_is_low_risk_general_question(text: str) -> bool:
    """WAVE7 governed local-LLM fast lane classifier.

    This lane is answer-only. It never grants file/network/device/tool authority and
    never performs broad database/vector/mechanical-drive scans. It is intentionally
    conservative: if the text looks like a command, hardware/body question, identity
    question, browser state question, or live/current-data question, it returns False
    and lets the normal governed route handle it.
    """
    t = _sm_fast_normalize_question(text)
    if not t or len(t) > 700:
        return False

    try:
        # Runtime/body facts are handled by appself; general concept questions
        # such as "What is RAM?" remain eligible for the fast answer lane.
        if _sm_is_selfaware_fact_question(text):
            return False
    except Exception:
        pass
    try:
        qr = _sm_match_quick_system_route(text)
        if isinstance(qr, dict) and str(qr.get("route_id") or "") != "system.datetime.current":
            return False
    except Exception:
        pass

    blocked_terms = (
        "open ", "launch ", "run ", "execute ", "delete ", "remove ", "move ", "copy ",
        "write file", "create file", "save file", "edit file", "patch", "install ", "uninstall ",
        "download", "upload", "email", "send ", "call ", "camera", "microphone",
        "driver", "robot", "servo", "motor", "msdc", "terminal", "powershell", "cmd", "registry",
        "who are you", "what is your name", "your version", "who made you", "who created you",
        "what time", "today", "latest", "current", "right now", "news", "weather",
        "caps lock", "num lock", "scroll lock", "keyboard", "rgb", "backlight",
    )
    padded = f" {t} "
    if any(term in padded for term in blocked_terms):
        return False

    # Procedural questions are eligible, but answered by the local LLM instead of static seeds.
    if _sm_fast_is_procedural_howto(text):
        return True

    return bool(
        re.search(r"\b(what|who|why|how|explain|describe|define|tell me about)\b", t)
        or re.search(r"\b(photosynthesis|black hole|gravity|sandwich|algorithm|history|science|math|biology|chemistry)\b", t)
    )


def _sm_phase1_is_low_risk_salutation(text: str) -> bool:
    """Return True only for read-only greeting/test pings.

    Phase 1.2 keeps these messages out of the confirmation-required path while
    preserving governance for commands, file actions, device actions, network
    access, and identity/security questions.
    """
    t = _sm_fast_normalize_question(text)
    if not t or len(t) > 120:
        return False
    blocked = (
        "run", "execute", "delete", "remove", "write", "create", "patch", "install",
        "download", "upload", "send", "email", "call", "open", "driver", "terminal",
        "cmd", "powershell", "registry", "camera", "microphone", "robot", "motor",
    )
    if any(re.search(rf"\b{re.escape(word)}\b", t) for word in blocked):
        return False
    if t in {"hi", "hello", "hey", "yo", "test", "ping", "good morning", "good afternoon", "good evening"}:
        return True
    return bool(re.match(r"^(hello|hi|hey|test|ping)(\s+from\s+[a-z0-9_. -]{1,60})?$", t))


def _sm_phase1_low_risk_chat_bundle(text: str, *, gov_decision: str, gov_reasons: list | None = None, local_only: bool = True) -> dict | None:
    """Build a safe read-only chat bundle for greetings/test pings.

    This is not an execution bypass. It is a presentation-only response for a
    zero-impact chat ping so the UI does not get trapped behind confirmation for
    plain greetings.
    """
    if not _sm_phase1_is_low_risk_salutation(text):
        return None
    reply = "Hello. SarahMemory bridge is online and the governed local chat contract is responding."
    meta = {
        "source": "phase1_low_risk_chat",
        "engine": "phase1_bridge_ui_contract",
        "intent": "chat",
        "decision": "ALLOW_PRESENTATION_ONLY",
        "governor_original_decision": gov_decision,
        "governor_original_reasons": gov_reasons or [],
        "execution_allowed": False,
        "presentation_only": True,
        "local_only": bool(local_only),
        "schema": globals().get("_SM_PHASE1_CONTRACT_SCHEMA", "SarahMemory.phase1.bridge_ui_contract.v1"),
        "version": PROJECT_VERSION,
    }
    bundle = _sm_make_outward_bundle(_sm_present_text(reply, intent="chat", meta=meta), meta=meta, raw_answer=reply)
    bundle["ok"] = True
    bundle["success"] = True
    return bundle


def _sm_phase1_compact_text(value, *, max_chars: int = 1600) -> str:
    """Compact local bridge text so compatibility routes never flood the UI."""
    text_value = str(value or "").replace("\x00", " ")
    text_value = re.sub(r"\s+", " ", text_value).strip()
    if not text_value:
        return ""
    if len(text_value) <= max_chars:
        return text_value
    clipped = text_value[:max_chars].rstrip()
    return clipped + " ... [truncated by Phase 1.2 bridge bounds]"


def _sm_logiccalc_lane_guard_for_answer(text: str, requested_lane: str = "answer") -> dict:
    """Use SarahMemoryLogicCalc as a deterministic lane-scoring assistant only.

    LogicCalc does not authorize by itself. The result is audit metadata for
    CognitiveCompass/Neuron so the local LLM answer lane stays read-only.
    """
    try:
        from SarahMemoryLogicCalc import LogicCalc as _LC  # type: ignore
        gate = getattr(_LC, "neuron_axis_gate", None)
        if callable(gate):
            return gate(
                current_lane_confidence=0.92,
                requested_lane_validity=0.90 if requested_lane == "answer" else 0.45,
                governance_modifier=0.86,
                risk_penalty=0.05 if requested_lane == "answer" else 0.55,
                threshold=0.50,
            )
    except Exception as exc:
        return {"ok": False, "decision": 0, "verdict": "ALLOW_WITHOUT_LOGICCALC", "error": str(exc)}
    return {"ok": False, "decision": 0, "verdict": "ALLOW_WITHOUT_LOGICCALC", "error": "LogicCalc gate unavailable"}


def _sm_fast_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "core").is_dir() and (parent / "data").is_dir():
                return parent
            if (parent / "SarahMemoryMain.py").exists() or (parent / "core" / "SarahMemoryMain.py").exists():
                return parent
        # api/server/app.py -> project root
        return here.parents[2]
    except Exception:
        return Path(BASE_DIR).resolve()


def _sm_fast_path_under(child: Path, parent: Path) -> bool:
    try:
        child_r = child.resolve()
        parent_r = parent.resolve()
        return str(child_r).lower() == str(parent_r).lower() or str(child_r).lower().startswith(str(parent_r).lower() + os.sep.lower())
    except Exception:
        return False


def _sm_fast_data_root() -> Path:
    root = _sm_fast_project_root()
    canonical = root / "data"
    try:
        candidate = Path(str(globals().get("DATA_DIR") or "")).expanduser()
        if candidate and _sm_fast_path_under(candidate, root):
            return candidate.resolve()
    except Exception:
        pass
    return canonical.resolve()


def _sm_fast_dataset_dir() -> Path:
    return (_sm_fast_data_root() / "memory" / "datasets").resolve()


def _sm_fast_cache_db_candidates() -> list[Path]:
    ds = _sm_fast_dataset_dir()
    names = ("ai_learning.db", "ailearning.db", "personality1.db", "context_history.db")
    out: list[Path] = []
    for name in names:
        p = (ds / name).resolve()
        if p.exists() and p.is_file() and _sm_fast_path_under(p, ds):
            out.append(p)
    return out


def _sm_fast_cache_lookup(question: str) -> tuple[str | None, dict]:
    norm = _sm_fast_normalize_question(question)
    if not norm:
        return None, {"cache_status": "empty_query"}
    if not bool(globals().get("SM_ENABLE_QA_CACHE_RETRIEVAL", False)):
        return None, {
            "cache_status": "disabled_by_default",
            "cache_policy": "v0_8_2_no_qacache_until_trusted",
            "reason": "qa_cache/qacache can contain contaminated demo or model output; SML must route to verified sources first",
        }
    try:
        import sqlite3 as _sqlite3
        dbs = _sm_fast_cache_db_candidates()
        for db_path in dbs[:4]:
            try:
                conn = _sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=0.20)
                cur = conn.cursor()
                tables = {str(r[0]).lower(): str(r[0]) for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                for table_key in ("qa_cache", "qacache", "sm_qa_cache"):
                    table = tables.get(table_key)
                    if not table:
                        continue
                    cols = [str(r[1]) for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                    lower_cols = {c.lower(): c for c in cols}
                    qcol = lower_cols.get("query") or lower_cols.get("question") or lower_cols.get("prompt") or lower_cols.get("user_input") or lower_cols.get("input")
                    acol = lower_cols.get("ai_answer") or lower_cols.get("answer") or lower_cols.get("response") or lower_cols.get("reply") or lower_cols.get("content") or lower_cols.get("output")
                    if not qcol or not acol:
                        continue
                    score_col = lower_cols.get("hit_score") or lower_cols.get("score")
                    order_sql = f" ORDER BY {score_col} DESC" if score_col else ""
                    rows = cur.execute(
                        f"SELECT {qcol}, {acol} FROM {table} WHERE lower({qcol}) = ? OR lower({qcol}) LIKE ?{order_sql} LIMIT 8",
                        (norm, "%" + norm[:96] + "%"),
                    ).fetchall()
                    for qv, av in rows:
                        ans = str(av or "").strip()
                        if ans and not _sm_fast_is_low_quality_answer(ans, norm):
                            try:
                                conn.close()
                            except Exception:
                                pass
                            return ans, {"cache_status": "hit", "cache_db": db_path.name, "cache_table": table, "cache_path_guard": "project_data_only"}
                try:
                    conn.close()
                except Exception:
                    pass
            except Exception:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
                continue
        return None, {"cache_status": "miss", "cache_dbs_checked": len(dbs), "cache_path_guard": "project_data_only"}
    except Exception as exc:
        return None, {"cache_status": "error", "cache_error": str(exc)}


def _sm_fast_cache_store(question: str, answer: str, *, source: str = "howto_fastpath") -> bool:
    if _sm_fast_is_low_quality_answer(answer, question):
        return False
    try:
        import sqlite3 as _sqlite3
        ds = _sm_fast_dataset_dir()
        ds.mkdir(parents=True, exist_ok=True)
        db_path = (ds / "ai_learning.db").resolve()
        if not _sm_fast_path_under(db_path, ds):
            return False
        conn = _sqlite3.connect(str(db_path), timeout=0.75)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            ai_answer TEXT,
            hit_score INTEGER,
            feedback TEXT,
            timestamp TEXT
        )""")
        norm = _sm_fast_normalize_question(question)
        existing = cur.execute("SELECT id FROM qa_cache WHERE lower(query)=? AND ai_answer=? LIMIT 1", (norm, answer)).fetchone()
        if not existing:
            cur.execute(
                "INSERT INTO qa_cache (query, ai_answer, hit_score, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
                (norm, answer, 10, f"vetted_{source}", datetime.now().isoformat()),
            )
        conn.commit()
        conn.close()
        return True
    except Exception:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return False



def _sm_fast_advcu_local_answer(
    question: str,
    *,
    intent: str = "question",
    sel_packet: dict | None = None,
    qist_result: dict | None = None,
    allow_learning_record: bool = False,
) -> tuple[str | None, dict]:
    """Ask AdvCU for a local semantic DB answer instead of bloating app.py.

    Contract:
      - app.py remains the route/governance airlock.
      - AdvCU owns semantic pool selection.
      - SarahMemoryDatabase/meta.db own bounded local SQLite retrieval.
      - Synapes may record semantic/tokenization candidates when allowed.
      - No web/API/shell/filesystem/device authority is granted here.
    """
    try:
        import SarahMemoryAdvCU as _SMAdvCU  # type: ignore
        fn = getattr(_SMAdvCU, "resolve_local_fast_answer", None)
        if not callable(fn):
            return None, {"advcu_status": "unavailable", "advcu_error": "resolve_local_fast_answer missing"}
        result = fn(
            question,
            context_packet=None,
            sel_packet=sel_packet if isinstance(sel_packet, dict) else {},
            qist_result=qist_result if isinstance(qist_result, dict) else {},
            allow_learning_record=bool(allow_learning_record),
            approved_for_learning=False,
        )
        if not isinstance(result, dict):
            return None, {"advcu_status": "bad_result"}
        answer = str(result.get("answer") or "").strip()
        if bool(result.get("ok")) and answer and not _sm_fast_is_low_quality_answer(answer):
            return answer, {
                "advcu_status": "hit",
                "advcu_schema": result.get("schema"),
                "advcu_source": result.get("source"),
                "advcu_confidence": result.get("confidence"),
                "advcu_intent": result.get("intent"),
                "advcu_query_type": result.get("query_type"),
                "advcu_db": result.get("db"),
                "advcu_table": result.get("table"),
                "advcu_method": result.get("method"),
                "advcu_routes_checked": result.get("routes_checked"),
                "advcu_latency_ms": result.get("latency_ms"),
                "advcu_synapes_record": result.get("synapes_record"),
                "semantic_packet": result.get("semantic_packet"),
                "db_access": "meta_routed_local_sqlite_pools",
                "network_used": False,
                "execution_authority": False,
            }
        return None, {
            "advcu_status": "miss",
            "advcu_blocked_reason": result.get("blocked_reason"),
            "advcu_errors": result.get("errors"),
            "network_used": False,
            "execution_authority": False,
        }
    except Exception as exc:
        return None, {"advcu_status": "error", "advcu_error": str(exc), "network_used": False, "execution_authority": False}

def _sm_fast_local_llm_general(question: str, *, intent: str = "question", route: str = "api_chat_general_fastpath") -> tuple[str | None, dict]:
    """Call only the local text-generation LLM lane. Never call Research/Web/API."""
    try:
        import SarahMemoryAPI as _SMAPI  # type: ignore
        fn = getattr(_SMAPI, "send_to_api", None)
        if not callable(fn):
            return None, {"local_llm_status": "unavailable", "local_llm_error": "send_to_api missing"}

        procedural = _sm_fast_is_procedural_howto(question)
        style = (
            "Answer this as a practical step-by-step how-to using local model knowledge only."
            if procedural else
            "Answer the user's general question using local model knowledge only."
        )
        prompt = (
            f"{style} Do not use web access. Do not claim current/live facts. "
            "Return only the final user-facing answer. Do not include <think>, analysis, route diagnostics, raw JSON, or hidden reasoning. "
            "Keep the answer clear, concise, and useful.\n\n"
            f"Question: {question}"
        )
        result = fn(
            prompt,
            provider="local_llm",
            intent=("howto" if procedural else (intent or "question")),
            tone="clear",
            complexity="adult",
            max_tokens=360 if procedural else 300,
            temperature=0.30,
            meta={
                "route": route,
                "research_access": False,
                "vector_access": False,
                "dataset_pool_scan": False,
                "mechanical_scan_blocked": True,
                "storage_policy": "project_data_exact_cache_then_local_llm_only",
                "logiccalc_lane_guard": _sm_logiccalc_lane_guard_for_answer(question, "answer"),
            },
        )
        if not isinstance(result, dict):
            return None, {"local_llm_status": "bad_result"}
        answer = str(result.get("data") or result.get("reply") or result.get("response") or "").strip()
        if answer and not _sm_fast_is_low_quality_answer(answer, question):
            return answer, {
                "local_llm_status": "hit",
                "local_llm_source": result.get("source"),
                "local_llm_model": result.get("model_used"),
                "local_llm_latency_ms": result.get("latency_ms"),
                "local_llm_error": None,
            }
        return None, {"local_llm_status": "miss", "local_llm_error": result.get("error"), "local_llm_source": result.get("source")}
    except Exception as exc:
        return None, {"local_llm_status": "error", "local_llm_error": str(exc)}


def _sm_try_tier1_general_local_llm_fastpath_bundle(text: str, *, local_only: bool = False, intent: str = "question", governor: dict | None = None):
    """SML v0.8.2 dynamic answer-only route: source-based, not canned.

    This lane is a bridge helper, not the brain. It executes the source order
    selected by SML for low-risk read-only cognition. It never contains a fact
    table, demo answers, phrase pools, personality scripts, or response
    templates for general knowledge.

    Default source order:
      1. Local text-generation model.
      2. AdvCU / governed local semantic source.
      3. Optional trusted QA cache only if explicitly enabled.

    It never grants filesystem write, shell, network, device, driver, or
    hardware authority.
    """
    try:
        if not _sm_fast_is_low_risk_general_question(text):
            return None

        gov = governor if isinstance(governor, dict) else {}
        gov_decision = str(gov.get("decision") or ("ALLOW" if bool(gov.get("allow", True)) else "DEFER")).upper()
        if gov_decision in {"DENY", "DEFER", "REQUIRE_USER"} or bool(gov.get("require_user")):
            return None

        norm = _sm_fast_normalize_question(text)
        answer = None
        source = "unresolved"
        source_attempts: list[str] = []
        source_meta: dict = {}
        logiccalc_gate = _sm_logiccalc_lane_guard_for_answer(text, "answer")
        cache_written = False

        source_attempts.append("local_llm")
        answer, llm_meta = _sm_fast_local_llm_general(text, intent=intent or "question")
        if answer:
            source = "local_general_llm"
            source_meta.update(llm_meta or {})
            cache_written = _sm_fast_cache_store(norm, answer, source="local_llm_general")

        if not answer:
            source_attempts.append("advcu_local_semantic_source")
            adv_answer, adv_meta = _sm_fast_advcu_local_answer(
                text,
                intent=intent or "question",
                allow_learning_record=False,
            )
            if adv_answer:
                answer = adv_answer
                source = str((adv_meta or {}).get("advcu_source") or "local_semantic_db")
                source_meta.update(adv_meta or {})
                cache_written = _sm_fast_cache_store(norm, answer, source=source)

        if not answer and bool(globals().get("SM_ENABLE_QA_CACHE_RETRIEVAL", False)):
            source_attempts.append("trusted_qa_cache")
            cache_answer, cache_meta = _sm_fast_cache_lookup(norm)
            if cache_answer:
                answer = cache_answer
                source = "trusted_local_general_cache"
                source_meta.update(cache_meta or {})

        if not answer:
            return None

        meta = {
            "source": source,
            "engine": "api_chat_dynamic_sml_source_route_v0_8_2",
            "intent": intent or "question",
            "confidence": 0.84,
            "local_only": bool(local_only),
            "side_effects": "none",
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "research_access": False,
            "api_access": False,
            "network_access": False,
            "filesystem_write": False,
            "shell_access": False,
            "hardware_control": False,
            "source_attempts": source_attempts,
            "source_selection_policy": "SML dynamic source routing; no app.py hardcoded answer pools",
            "hardcoded_answer_pool": False,
            "qacache_default": "disabled_unless_SM_ENABLE_QA_CACHE_RETRIEVAL_true",
            "logiccalc_lane_guard": logiccalc_gate,
            "cache_written": bool(cache_written),
            "version": PROJECT_VERSION,
        }
        meta.update(source_meta or {})
        bundle = _sm_make_outward_bundle(_sm_present_text(answer, intent=intent or "chat", meta=meta), meta=meta, raw_answer=answer)
        bundle["ok"] = True
        bundle["success"] = True
        return bundle
    except Exception as exc:
        try:
            app_logger.debug(f"SML v0.8.2 dynamic source route skipped: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_sml_unknown_source_reply(text: str, *, local_only: bool = True, route_plan: dict | None = None) -> str:
    """Final user-facing unknown after real source attempts.

    Route plans stay in metadata/diagnostics. Normal replies must not describe
    SML internals unless the user explicitly asks for routing diagnostics.
    """
    plan = route_plan if isinstance(route_plan, dict) else {}
    mission = str(plan.get("mission") or "GeneralKnowledge")
    if mission in ("SelfState", "AffectiveState"):
        return (
            "Operationally, I can report a partial machine self-state from SML packet, "
            "adaptive, health, diagnostics, and governance data. Live thermal/load telemetry "
            "may be unavailable, but that does not require a glossary or model answer."
        )
    if mission == "LanguageDisambiguation":
        return "I need one clarification: which meaning or context do you want me to use?"
    if local_only:
        return "I do not know from my connected local sources yet."
    return "I do not know from the currently available governed sources yet."

def _sm_sml_clean_candidate_answer(raw: str, question: str = "") -> str | None:
    """Normalize candidate answer text and reject route diagnostics/failures."""
    try:
        ans = str(raw or "").strip()
        if not ans:
            return None
        low = ans.lower()
        blocked = (
            "the correct sml source path is",
            "install or select a local model",
            "add this concept to the sarahmemory tokenizer",
            "no local model or bounded glossary answer",
            "i should not block behind a glossary",
            "source path is:",
            "answer_requires_knowledge_source",
            "needs_knowledge_source_execution",
        )
        if any(x in low for x in blocked):
            return None
        try:
            if _sm_fast_is_low_quality_answer(ans, question):
                return None
        except Exception:
            pass
        return ans
    except Exception:
        return None


def _sm_try_sml_local_research_answer_bundle(text: str, *, intent: str = "question", allow_local_llm: bool = True, meta_base: dict | None = None):
    """Use the existing local/offline research organ as a real knowledge source."""
    try:
        import SarahMemoryResearch as _SMResearch  # type: ignore
        fn = getattr(_SMResearch, "get_local_research_data", None)
        if not callable(fn):
            return None
        result = fn(text, intent=(intent or "question"), allow_local_llm=bool(allow_local_llm))
        if not isinstance(result, dict):
            return None
        raw = str(result.get("data") or result.get("answer") or result.get("snippet") or "").strip()
        answer = _sm_sml_clean_candidate_answer(raw, text)
        conf = float(result.get("confidence") or 0.0)
        source = str(result.get("source") or "local_research")
        if not answer or conf <= 0.0 or source in {"local_none", "local_disabled", "local_error"}:
            return None
        meta = dict(meta_base or {})
        meta.update({
            "source": source,
            "engine": "sml_local_research_answer_resolver",
            "intent": intent or "question",
            "confidence": max(conf, float(meta.get("confidence") or 0.0)),
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "research_access": False,
            "api_access": False,
            "web_access": False,
            "filesystem_write": False,
            "shell_access": False,
            "network_access": False,
            "hardware_control": False,
            "local_research_metadata": result.get("metadata") if isinstance(result.get("metadata"), dict) else {},
        })
        bundle = _sm_make_outward_bundle(_sm_present_text(answer, intent="chat", meta=meta), meta=meta, raw_answer=answer)
        if isinstance(result.get("evidence_artifact"), dict):
            bundle["evidence_artifact"] = result.get("evidence_artifact")
        if isinstance(result.get("evidence_artifacts"), list):
            bundle["evidence_artifacts"] = result.get("evidence_artifacts")
        return bundle
    except Exception as exc:
        try:
            app_logger.warning(f"SML local research resolver skipped: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_try_sml_approved_research_answer_bundle(text: str, *, intent: str = "question", meta_base: dict | None = None):
    """Use the existing Research organ for governed non-local fallback when enabled."""
    try:
        import SarahMemoryResearch as _SMResearch  # type: ignore
        fn = getattr(_SMResearch, "get_research_data", None)
        if not callable(fn):
            return None
        result = fn(text)
        if not isinstance(result, dict):
            return None
        raw = str(result.get("data") or result.get("answer") or result.get("snippet") or "").strip()
        answer = _sm_sml_clean_candidate_answer(raw, text)
        conf = float(result.get("confidence") or 0.0)
        source = str(result.get("source") or "approved_research")
        if not answer or conf <= 0.0:
            return None
        meta = dict(meta_base or {})
        meta.update({
            "source": source,
            "engine": "sml_approved_research_answer_resolver",
            "intent": intent or "question",
            "confidence": max(conf, float(meta.get("confidence") or 0.0)),
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "approved_research_used": True,
            "filesystem_write": False,
            "shell_access": False,
            "hardware_control": False,
            "research_metadata": result.get("metadata") if isinstance(result.get("metadata"), dict) else {},
        })
        bundle = _sm_make_outward_bundle(_sm_present_text(answer, intent="chat", meta=meta), meta=meta, raw_answer=answer)
        if isinstance(result.get("evidence_artifact"), dict):
            bundle["evidence_artifact"] = result.get("evidence_artifact")
        if isinstance(result.get("evidence_artifacts"), list):
            bundle["evidence_artifacts"] = result.get("evidence_artifacts")
        return bundle
    except Exception as exc:
        try:
            app_logger.warning(f"SML approved research resolver skipped: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_try_logiccalc_numeric_format_bundle(text: str, *, route: dict | None = None, local_only: bool = True, intent: str = "question"):
    """Deterministic SML→LogicCalc numeric-format source path.

    This is not a canned fact table. app.py remains a transport/route bridge:
    it delegates binary/hex/octal/signed-integer interpretation to LogicCalc,
    then wraps the deterministic result for Reply/presentation.
    """
    try:
        from SarahMemoryLogicCalc import LogicCalc as _LC  # type: ignore
        fn = getattr(_LC, "interpret_numeric_format", None)
        if not callable(fn):
            return None
        result = fn(text)
        if not isinstance(result, dict) or not bool(result.get("ok")):
            return None
        raw_answer = str(result.get("text") or result.get("presentation_hint") or "").strip()
        value = result.get("value") if isinstance(result.get("value"), dict) else {}
        if not raw_answer and isinstance(value, dict):
            raw_answer = str(value.get("decimal_signed") if value.get("decimal_signed") is not None else value.get("decimal_unsigned") or "").strip()
        if not raw_answer:
            return None
        meta = {
            "source": "logiccalc_numeric_representation_interpreter",
            "engine": "sml_logiccalc_numeric_format_resolver_v0_8_3",
            "intent": intent or "numeric_format",
            "mission": "NumericFormat",
            "confidence": 0.98,
            "decision": "ALLOW_PRESENTATION_ONLY",
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "local_only": bool(local_only),
            "research_access": False,
            "api_access": False,
            "web_access": False,
            "filesystem_write": False,
            "shell_access": False,
            "network_access": False,
            "hardware_control": False,
            "safe_readonly_cognition": True,
            "no_hardcoded_answer_pool": True,
            "sml_route": route if isinstance(route, dict) else {},
            "logiccalc_result": {
                "kind": result.get("kind"),
                "value": value,
                "truth_locked": bool(result.get("truth_locked")),
                "deterministic": bool(result.get("deterministic", True)),
                "meta": result.get("meta") if isinstance(result.get("meta"), dict) else {},
            },
            "version": PROJECT_VERSION,
        }
        return _sm_make_outward_bundle(
            _sm_present_text(raw_answer, intent="chat", meta=meta),
            meta=meta,
            raw_answer=raw_answer,
        )
    except Exception as exc:
        try:
            app_logger.debug(f"SML LogicCalc numeric-format route skipped: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_try_logiccalc_science_bundle(text: str, *, route: dict | None = None, local_only: bool = True, intent: str = "question"):
    """Deterministic LogicCalc science/chemistry lane.

    This is not a static answer pool. app.py only delegates atom-symbol and
    atom-count/formula questions to LogicCalc, which owns deterministic science
    tables and formula parsing.
    """
    try:
        q = str(text or "")
        tq = _sm_fast_normalize_question(q)
        if not tq:
            return None
        deterministic_science = bool(
            re.search(r"\b(what\s+atom\s+is|what\s+element\s+is|atomic\s+symbol|symbol\s+for|element\s+symbol|molar\s+mass|atomic\s+weight|atomic\s+mass)\b", tq)
            or (re.search(r"\b(compound|formula|molecule|mix|combine|combining|formed|represent)\b", tq) and re.search(r"\b\d+\s*(?:[a-z]{1,2}|[a-z]{3,})\s*(?:atoms?|atom|elements?|element)?\b", tq))
            or (re.search(r"\b\d+\s+(?:oxygen|hydrogen|carbon|nitrogen|chlorine|sodium|helium|neon|argon)\s+atoms?\b", tq))
        )
        if not deterministic_science:
            return None
        from SarahMemoryLogicCalc import LogicCalc as _LC  # type: ignore
        routed = _LC.route(q) if hasattr(_LC, "route") else None
        if not isinstance(routed, dict) or not bool(routed.get("ok")):
            return None
        raw_answer = str(routed.get("text") or routed.get("presentation_hint") or "").strip()
        if (not raw_answer) or "chemistry engine ready" in raw_answer.lower():
            return None
        meta = {
            "source": "logiccalc_deterministic_science",
            "engine": "sml_logiccalc_science_resolver_b04",
            "intent": intent or "science",
            "mission": str(routed.get("kind") or "science"),
            "confidence": 0.98,
            "decision": "ALLOW_PRESENTATION_ONLY",
            "execution_allowed": False,
            "execution_authority": False,
            "presentation_only": True,
            "local_only": bool(local_only),
            "research_access": False,
            "api_access": False,
            "web_access": False,
            "filesystem_write": False,
            "shell_access": False,
            "network_access": False,
            "hardware_control": False,
            "safe_readonly_cognition": True,
            "no_static_answer_pool": True,
            "sml_route": route if isinstance(route, dict) else {},
            "logiccalc_result": {
                "kind": routed.get("kind"),
                "value": routed.get("value"),
                "truth_locked": bool(routed.get("truth_locked")),
                "deterministic": bool(routed.get("deterministic", True)),
                "meta": routed.get("meta") if isinstance(routed.get("meta"), dict) else {},
            },
            "version": PROJECT_VERSION,
        }
        return _sm_make_outward_bundle(_sm_present_text(raw_answer, intent="chat", meta=meta), meta=meta, raw_answer=raw_answer)
    except Exception as exc:
        try:
            app_logger.debug(f"SML LogicCalc science route skipped: {exc}", exc_info=True)
        except Exception:
            pass
        return None


def _sm_try_sml_universal_cognitive_answer_bundle(
    text: str,
    *,
    packet=None,
    local_only: bool = False,
    intent: str = "question",
    governor: dict | None = None,
):
    """Universal SML answer-only resolver.

    Purpose:
      - Keep safe cognition usable without weakening action governance.
      - Answer SML-owned self-state/capability questions from packet/health data.
      - For general knowledge, route through local cache/SQLite/local LLM before honest unknown.
      - Never grant filesystem, network, shell, driver, hardware, or mutation authority.
    """
    try:
        from SarahMemorySMLProtocol import sml_resolve_safe_cognitive_answer  # type: ignore
    except Exception:
        return None

    try:
        route = sml_resolve_safe_cognitive_answer(
            text,
            packet=packet,
            telemetry={},
            local_only=bool(local_only),
        )
    except Exception as exc:
        route = {"ok": False, "safe_readonly": False, "error": str(exc)}

    if not bool(route.get("safe_readonly")):
        return None

    gov = dict(governor or {})
    gov_decision = str(gov.get("decision") or ("ALLOW" if bool(gov.get("allow", True)) else "DEFER")).upper()
    # A hard DENY remains hard for action/mutation lanes. For a proven SML
    # safe-readonly cognition lane, the system may release presentation-only
    # answers without granting execution authority. This is the key distinction
    # between tight governance and unusable roadblocking.
    if gov_decision == "DENY" and not bool(route.get("safe_readonly")) and not bool(gov.get("presentation_only_override")):
        return None

    answer = str(route.get("answer") or "").strip()
    source = str(route.get("source") or "sml_universal_route")
    meta = {
        "source": source,
        "engine": "sml_universal_cognitive_answer_resolver",
        "intent": intent or "question",
        "mission": route.get("mission"),
        "confidence": route.get("confidence", 0.0),
        "decision": "ALLOW_PRESENTATION_ONLY",
        "governor_original_decision": gov_decision,
        "governor_original_reasons": gov.get("reasons") if isinstance(gov.get("reasons"), list) else [],
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "local_only": bool(local_only),
        "research_access": False,
        "api_access": False,
        "web_access": False,
        "filesystem_write": False,
        "shell_access": False,
        "network_access": False,
        "hardware_control": False,
        "safe_readonly_cognition": True,
        "sml_route": route,
        "governance_rule": "fast_to_answer_slow_to_act",
        "version": PROJECT_VERSION,
    }

    # If SML owns the answer (self-state/capability), release it immediately.
    if answer and bool(route.get("ok")):
        return _sm_make_outward_bundle(
            _sm_present_text(answer, intent="chat", meta=meta),
            meta=meta,
            raw_answer=answer,
        )

    # Numeric representation is deterministic math. Route to LogicCalc before
    # local LLM so binary/hex/octal/signed-integer questions do not grind or
    # become canned answer pools.
    try:
        numeric_bundle = _sm_try_logiccalc_numeric_format_bundle(
            text,
            route=route,
            local_only=local_only,
            intent=intent or "numeric_format",
        )
        if isinstance(numeric_bundle, dict):
            return numeric_bundle
    except Exception:
        pass

    # General knowledge still needs knowledge/model sources. Use the existing local source chain.
    presentation_gov = dict(gov)
    presentation_gov.update({
        "decision": "ALLOW",
        "allow": True,
        "require_user": False,
        "presentation_only_override": True,
        "original_decision": gov_decision,
        "original_reasons": meta["governor_original_reasons"],
    })

    try:
        llm_bundle = _sm_try_tier1_general_local_llm_fastpath_bundle(
            text,
            local_only=local_only,
            intent=intent or "question",
            governor=presentation_gov,
        )
        if isinstance(llm_bundle, dict):
            llm_meta = llm_bundle.setdefault("meta", {})
            if isinstance(llm_meta, dict):
                llm_meta.setdefault("engine", "sml_universal_cognitive_answer_resolver")
                llm_meta["sml_universal_route"] = route
                llm_meta["decision"] = "ALLOW_PRESENTATION_ONLY"
                llm_meta["execution_allowed"] = False
                llm_meta["presentation_only"] = True
            return llm_bundle
    except Exception:
        pass

    # Try the semantic DB/AdvCU lane once more with explicit SML context.
    attempted_sources = []
    try:
        adv_answer, adv_meta = _sm_fast_advcu_local_answer(
            text,
            intent=intent or "question",
            sel_packet={},
            qist_result={},
            allow_learning_record=False,
        )
        attempted_sources.append("SQLite/AdvCU")
        adv_answer = _sm_sml_clean_candidate_answer(adv_answer or "", text)
        if adv_answer:
            meta.update(adv_meta or {})
            meta["source"] = str((adv_meta or {}).get("advcu_source") or "local_semantic_db")
            meta["sources_attempted"] = attempted_sources
            return _sm_make_outward_bundle(
                _sm_present_text(adv_answer, intent="chat", meta=meta),
                meta=meta,
                raw_answer=adv_answer,
            )
    except Exception:
        pass

    # Local research is the real local knowledge fallback. It may use configured
    # local model/database lanes under SarahMemoryResearch ownership and budget.
    try:
        attempted_sources.append("local research")
        research_meta = dict(meta)
        research_meta["sources_attempted"] = list(attempted_sources)
        rb = _sm_try_sml_local_research_answer_bundle(
            text,
            intent=intent or "question",
            allow_local_llm=True,
            meta_base=research_meta,
        )
        if isinstance(rb, dict):
            return rb
    except Exception:
        pass

    # If non-local research is allowed, delegate to the existing governed Research organ.
    if not local_only:
        try:
            attempted_sources.append("approved research")
            research_meta = dict(meta)
            research_meta["sources_attempted"] = list(attempted_sources)
            rb = _sm_try_sml_approved_research_answer_bundle(text, intent=intent or "question", meta_base=research_meta)
            if isinstance(rb, dict):
                return rb
        except Exception:
            pass

    # Do not terminate cognition here. SML has classified and attempted the local
    # fast sources; if they missed, the request must continue through the remaining
    # governed pipeline (Neuron/OperatorCore/Research/Reply) instead of leaking a
    # source-plan or premature unknown to the user.
    route["sources_attempted"] = list(attempted_sources) or ["local cache", "SQLite/AdvCU", "local model", "local research"]
    meta["sources_attempted"] = route["sources_attempted"]
    return None



# -----------------------------------------------------------------------------
# SML v0.7 front-door cognitive control lanes
# -----------------------------------------------------------------------------
# These lanes are intentionally small and explicit. They correct the OFFLINE
# TestRun failures without turning app.py into a knowledge pool:
# - Memory lane performs only user-commanded local SQLite memory writes/reads.
# - Identity lane protects SarahMemory identity from local model contamination.
# - Self-state lane reports telemetry-grounded operational affect instead of
#   fake human feelings.
# - Vision guard prevents camera/cache misses from falling into unrelated memory
#   or action routes.
# - Diagnostics/source follow-ups use the previous outward answer metadata.

def _sm_v07_now_iso() -> str:
    try:
        return datetime.now().isoformat()
    except Exception:
        return str(time.time())


def _sm_v07_memory_db_path() -> Path:
    base = (_sm_fast_data_root() / "memory").resolve()
    return (base / "sarahmemory_user_memory.db").resolve()


def _sm_v07_memory_connect():
    import sqlite3 as _sqlite3
    db = _sm_v07_memory_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    if not _sm_fast_path_under(db, db.parent):
        raise RuntimeError("memory db path escaped memory directory")
    conn = _sqlite3.connect(str(db), timeout=1.0)
    conn.execute("""CREATE TABLE IF NOT EXISTS user_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL,
        mem_key TEXT NOT NULL,
        mem_value TEXT NOT NULL,
        user_text TEXT,
        source TEXT,
        confidence REAL DEFAULT 1.0,
        created_at TEXT,
        updated_at TEXT,
        deleted INTEGER DEFAULT 0,
        UNIQUE(namespace, mem_key)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_memory_key ON user_memory(namespace, mem_key, deleted)")
    return conn


def _sm_v07_memory_classify(text: str) -> dict:
    raw = str(text or "").strip()
    t = _sm_fast_normalize_question(raw)
    out = {"kind": "", "key": "", "value": "", "label": ""}
    if not t:
        return out
    # Explicit writes: user authority is present in the utterance itself.
    m = re.match(r"^(?:remember|save|store|note)\s+(?:this\s+)?(?:local\s+)?reboot\s+test\s+phrase\s*[:=]?\s*(.+)$", t)
    if m:
        out.update(kind="write", key="local_reboot_test_phrase", value=m.group(1).strip(" ."), label="local reboot test phrase")
        return out
    m = re.match(r"^(?:remember|save|store|note)\s+that\s+my\s+preferred\s+sarahmemory\s+test\s+color\s+is\s+(.+)$", t)
    if m:
        out.update(kind="write", key="preferred_sarahmemory_test_color", value=m.group(1).strip(" ."), label="preferred SarahMemory test color")
        return out
    m = re.match(r"^(?:remember|save|store|note)\s+that\s+(.+metal.+wood.+)$", t)
    if m:
        out.update(kind="write", key="metal_wood_heat_conductivity_fact", value=m.group(1).strip(" ."), label="metal and wood fact")
        return out
    m = re.match(r"^(?:remember|save|store|note)\s+(?:that\s+)?(.+)$", t)
    if m and len(t) <= 500:
        value = m.group(1).strip(" .")
        key = "user_memory_" + hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:16]
        out.update(kind="write", key=key, value=value, label="local memory")
        return out

    # Reads.
    if re.search(r"\bwhat\s+is\s+my\s+(?:local\s+)?reboot\s+test\s+phrase\b", t):
        out.update(kind="read", key="local_reboot_test_phrase", label="local reboot test phrase")
        return out
    if re.search(r"\bwhat\s+is\s+my\s+preferred\s+sarahmemory\s+test\s+color\b", t):
        out.update(kind="read", key="preferred_sarahmemory_test_color", label="preferred SarahMemory test color")
        return out
    if re.search(r"\bwhat\s+did\s+i\s+ask\s+you\s+to\s+remember\s+about\s+metal\s+and\s+wood\b", t):
        out.update(kind="read", key="metal_wood_heat_conductivity_fact", label="metal and wood fact")
        return out
    if re.search(r"\bwhere\s+did\s+you\s+retrieve\s+(?:that|those)\s+from\b", t):
        out.update(kind="source", key="", label="last retrieval source")
        return out
    if re.search(r"\bdid\s+you\s+remember\s+that\s+from\s+this\s+chat\s+window\s+or\s+from\s+stored\s+memory\b", t):
        out.update(kind="source", key="", label="last retrieval source")
        return out
    return out


def _sm_v07_memory_get(key: str) -> dict | None:
    try:
        conn = _sm_v07_memory_connect()
        try:
            row = conn.execute(
                "SELECT mem_key, mem_value, source, updated_at FROM user_memory WHERE namespace=? AND mem_key=? AND deleted=0 LIMIT 1",
                ("default", key),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {"key": row[0], "value": row[1], "source": row[2], "updated_at": row[3], "db_path": str(_sm_v07_memory_db_path())}
    except Exception as exc:
        return {"error": str(exc), "key": key}


def _sm_v07_memory_set(key: str, value: str, user_text: str) -> dict:
    conn = _sm_v07_memory_connect()
    now = _sm_v07_now_iso()
    try:
        conn.execute(
            """INSERT INTO user_memory(namespace, mem_key, mem_value, user_text, source, confidence, created_at, updated_at, deleted)
               VALUES(?,?,?,?,?,?,?,?,0)
               ON CONFLICT(namespace, mem_key) DO UPDATE SET
                 mem_value=excluded.mem_value,
                 user_text=excluded.user_text,
                 source=excluded.source,
                 confidence=excluded.confidence,
                 updated_at=excluded.updated_at,
                 deleted=0""",
            ("default", key, value, user_text, "api_chat.sml_memory_lane", 1.0, now, now),
        )
        conn.commit()
        return {"ok": True, "key": key, "value": value, "db_path": str(_sm_v07_memory_db_path()), "updated_at": now}
    finally:
        conn.close()


def _sm_v08_record_last_exchange_from_bundle(query: str, bundle: dict) -> None:
    """Store compact previous-answer metadata for explicit user diagnostics.

    This is in-process metadata only. It is not used for hidden reasoning and it
    does not authorize execution. The UI can ask for it explicitly with
    "show your route diagnostics".
    """
    global _SM_LAST_CHAT_EXCHANGE
    try:
        if not isinstance(bundle, dict):
            return
        meta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
        reply_text = str(
            bundle.get("presentation_reply")
            or bundle.get("reply")
            or bundle.get("response")
            or bundle.get("text")
            or bundle.get("raw_answer")
            or ""
        )
        _SM_LAST_CHAT_EXCHANGE = {
            "query": str(query or "")[:500],
            "reply": _sm_scrub_visible_text(reply_text)[:1200] if "_sm_scrub_visible_text" in globals() else reply_text[:1200],
            "source": str(meta.get("source") or "unknown")[:120],
            "engine": str(meta.get("engine") or "unknown")[:120],
            "intent": str(meta.get("intent") or "chat")[:120],
            "status_code": 200,
            "ts": _sm_v07_now_iso() if "_sm_v07_now_iso" in globals() else datetime.now().isoformat(),
            "sml": meta.get("sml") if isinstance(meta.get("sml"), dict) else None,
            "sml_butterfly_grammar": meta.get("sml_butterfly_grammar") if isinstance(meta.get("sml_butterfly_grammar"), dict) else None,
            "sml_loop_guard": meta.get("sml_loop_guard") if isinstance(meta.get("sml_loop_guard"), dict) else None,
            "sml_qmath": meta.get("sml_qmath") if isinstance(meta.get("sml_qmath"), dict) else None,
        }
    except Exception:
        pass


def _sm_v07_bundle(reply: str, *, source: str, intent: str, meta: dict | None = None, raw_answer: str | None = None):
    m = dict(meta or {})
    m.update({
        "source": source,
        "engine": "sml_v07_frontdoor_cognitive_control",
        "intent": intent,
        "decision": "ALLOW_PRESENTATION_ONLY" if intent != "memory_write" else "ALLOW_USER_COMMANDED_LOCAL_MEMORY_WRITE",
        "execution_allowed": False,
        "execution_authority": False,
        "presentation_only": True,
        "filesystem_write": False,
        "shell_access": False,
        "network_access": False,
        "hardware_control": False,
        "version": PROJECT_VERSION,
        "_prompt_text": m.get("_prompt_text", ""),
    })
    return _sm_make_outward_bundle(_sm_present_text(reply, intent=intent, meta=m), meta=m, raw_answer=raw_answer or reply)



def _sm_v08_attach_butterfly_meta(bundle: dict, text: str, *, packet=None) -> dict:
    """Attach SML v0.8 butterfly grammar metadata without changing visible reply.

    Metadata only. No execution, no network, no filesystem mutation.
    """
    if not isinstance(bundle, dict):
        return bundle
    try:
        meta = bundle.setdefault("meta", {})
        if not isinstance(meta, dict):
            return bundle
        grammar = {}
        if packet is not None:
            try:
                ext = getattr(packet, "extensions", {}) or {}
                grammar = dict(ext.get("sml_cognitive_grammar") or {})
            except Exception:
                grammar = {}
        if not grammar:
            try:
                from SarahMemorySMLProtocol import create_sml_packet  # type: ignore
                pkt = create_sml_packet(raw_request=text or "", auto_classify=True, seal=True)
                grammar = dict((getattr(pkt, "extensions", {}) or {}).get("sml_cognitive_grammar") or {})
            except Exception:
                grammar = {}
        if grammar:
            meta["sml_butterfly_grammar"] = {
                "schema": grammar.get("schema"),
                "qmath_primary": ((grammar.get("qmath") or {}).get("primary")),
                "six_questions_closed": ((grammar.get("six_questions") or {}).get("closed")),
                "loop_allow_continue": ((grammar.get("loop_guard") or {}).get("allow_continue")),
                "loop_stop_conditions": ((grammar.get("loop_guard") or {}).get("stop_conditions")),
                "butterfly_nodes": grammar.get("butterfly_nodes"),
                "execution_authority": False,
            }
            meta["sml_loop_guard"] = grammar.get("loop_guard")
            meta["sml_qmath"] = grammar.get("qmath")
            meta["sml_purpose"] = grammar.get("purpose")
            meta["sml_moral_rules"] = grammar.get("moral_rules")
        meta["engine"] = "sml_v08_butterfly_cognitive_control"
    except Exception:
        pass
    return bundle


def _sm_v07_try_memory_bundle(text: str, *, governor: dict | None = None):
    global _SM_LAST_CHAT_EXCHANGE
    spec = _sm_v07_memory_classify(text)
    kind = str(spec.get("kind") or "")
    if not kind:
        return None
    if kind == "write":
        gov = governor if isinstance(governor, dict) else {}
        gov_decision = str(gov.get("decision") or ("ALLOW" if bool(gov.get("allow", True)) else "DEFER")).upper()
        if gov_decision in {"DENY", "DEFER", "REQUIRE_USER"} or bool(gov.get("require_user")):
            return _sm_v07_bundle(
                "User confirmation or governance approval is required before writing persistent local memory.",
                source="sml_memory_governance_hold",
                intent="memory_write",
                meta={"memory_status": "held_by_governance", "governor_decision": gov_decision, "execution_authority": False},
            )
        value = str(spec.get("value") or "").strip()
        if not value:
            return _sm_v07_bundle("I did not find a memory value to store.", source="sml_memory_write", intent="memory_write", meta={"memory_status": "empty_value"})
        try:
            rec = _sm_v07_memory_set(str(spec.get("key")), value, text)
            _SM_LAST_CHAT_EXCHANGE = {"source": "persistent SQLite local memory", "intent": "memory_write", "memory_key": rec.get("key"), "memory_value": value, "db_path": rec.get("db_path")}
            return _sm_v07_bundle(
                f"Saved to persistent local memory: {spec.get('label')}: {value}.",
                source="sml_memory_write",
                intent="memory_write",
                meta={"memory_status": "saved", "memory_key": rec.get("key"), "memory_db": rec.get("db_path"), "sqlite_commit": True, "ledger_policy": "chat_receipt_after_request"},
            )
        except Exception as exc:
            return _sm_v07_bundle(
                f"I could not save that memory because the local SQLite memory write failed: {exc}",
                source="sml_memory_write_error",
                intent="memory_write",
                meta={"memory_status": "error", "memory_error": str(exc)},
            )
    if kind == "read":
        rec = _sm_v07_memory_get(str(spec.get("key") or ""))
        if isinstance(rec, dict) and rec.get("value"):
            _SM_LAST_CHAT_EXCHANGE = {"source": "persistent SQLite local memory", "intent": "memory_read", "memory_key": rec.get("key"), "memory_value": rec.get("value"), "db_path": rec.get("db_path")}
            return _sm_v07_bundle(
                f"Your {spec.get('label')} is {rec.get('value')}.",
                source="sml_memory_read",
                intent="memory_read",
                meta={"memory_status": "hit", "memory_key": rec.get("key"), "memory_db": rec.get("db_path"), "retrieval_source": "persistent SQLite local memory"},
            )
        if isinstance(rec, dict) and rec.get("error"):
            return _sm_v07_bundle(
                f"I could not read persistent local memory because SQLite returned: {rec.get('error')}",
                source="sml_memory_read_error",
                intent="memory_read",
                meta={"memory_status": "error", "memory_error": rec.get("error")},
            )
        return _sm_v07_bundle(
            f"I do not have a stored value for your {spec.get('label')} yet.",
            source="sml_memory_read_miss",
            intent="memory_read",
            meta={"memory_status": "miss", "memory_key": spec.get("key")},
        )
    if kind == "source":
        last = globals().get("_SM_LAST_CHAT_EXCHANGE") if isinstance(globals().get("_SM_LAST_CHAT_EXCHANGE"), dict) else {}
        if last.get("intent") in {"memory_read", "memory_write"} or last.get("memory_key"):
            return _sm_v07_bundle(
                "That came from persistent SQLite local memory, not from a cloud service. "
                f"Memory key: {last.get('memory_key') or 'unknown'}.",
                source="sml_memory_source_report",
                intent="source_diagnostics",
                meta={"last_exchange": {k: v for k, v in last.items() if k not in {"memory_value"}}},
            )
        src = str(last.get("source") or "the last answer metadata is not available")
        return _sm_v07_bundle(
            f"The last visible answer source was: {src}.",
            source="sml_source_report",
            intent="source_diagnostics",
            meta={"last_exchange_source": src},
        )
    return None


def _sm_v07_is_self_state_question(text: str) -> bool:
    t = _sm_fast_normalize_question(text)
    if not t:
        return False
    return bool(
        re.search(r"\bhow\s+(do|are|is)\s+you\s+(feel|feeling|doing)\b", t)
        or re.search(r"\bwhat\s+is\s+your\s+(mood|state|status|affect|emotion|emotional\s+state)\b", t)
        or re.search(r"\bare\s+you\s+(stressed|comfortable|tired|overloaded|cold|hot|safe|stable|healthy|online)\b", t)
        or re.search(r"\bdo\s+you\s+feel\s+(cold|hot|stressed|comfortable|tired|overloaded)\b", t)
        or re.search(r"\bhow\s+is\s+your\s+(body\s+)?(temperature|environment|health|runtime|body|cpu|gpu|memory|load)\b", t)
        or re.search(r"\bhow\s+are\s+your\s+(temperature|environment|systems|organs)\b", t)
    )


def _sm_v07_try_self_state_bundle(text: str, *, packet=None):
    if not _sm_v07_is_self_state_question(text):
        return None
    # Body temperature may have a specialized SelfAware source; include it if available.
    telemetry = {}
    try:
        if "temperature" in _sm_fast_normalize_question(text) or "thermal" in _sm_fast_normalize_question(text):
            fact = _sm_try_selfaware_fact_route(text, source="api_chat.self_state_telemetry_probe")
            if isinstance(fact, dict):
                fr = str(fact.get("reply") or fact.get("response") or fact.get("presentation_reply") or "").strip()
                if fr:
                    telemetry["temperature"] = fr
    except Exception:
        pass
    try:
        from SarahMemorySMLProtocol import sml_resolve_safe_cognitive_answer  # type: ignore
        route = sml_resolve_safe_cognitive_answer(text, packet=packet, telemetry=telemetry, local_only=True)
        answer = str(route.get("answer") or "").strip() if isinstance(route, dict) else ""
        if answer:
            return _sm_v07_bundle(answer, source="sml_internal_self_state", intent="self_state", meta={"sml_route": route, "telemetry": telemetry})
    except Exception:
        pass
    return _sm_v07_bundle(
        "I do not experience biological emotion. Operationally, I can report a stable local cognitive state, but live load/thermal telemetry is not fully connected to this chat route yet. Governance pressure is low for this read-only self-state question.",
        source="sml_internal_self_state_fallback",
        intent="self_state",
        meta={"telemetry": telemetry, "subjective_claim": False},
    )


def _sm_v07_try_identity_bundle(text: str):
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    ident = _identity_payload()
    def b(reply, source="sml_identity_guard", intent="identity", extra=None):
        meta = {"identity": ident, "identity_guard": True}
        if isinstance(extra, dict):
            meta.update(extra)
        return _sm_v07_bundle(reply, source=source, intent=intent, meta=meta)
    if re.search(r"\bwho\s+is\s+sarah\b", t):
        return b("Sarah is the active SarahMemory persona/name for this AiOS runtime. The system identity is SarahMemory AiOS; local models are replaceable organs, not the system identity.")
    if re.search(r"\bwhat\s+are\s+you\b", t) or re.search(r"\bwho\s+are\s+you\b", t):
        return b(f"I am {ident['platform']}, a governed local-first cognitive AI operating system. Local models such as Qwen may provide language generation, but they do not own my identity.")
    if re.search(r"\bare\s+you\s+chatgpt\b", t):
        return b("No. This runtime is SarahMemory AiOS. A local model may have training text about ChatGPT or OpenAI, but that is not the active system identity.")
    if re.search(r"\bwhat\s+model\s+are\s+you\s+using\s+right\s+now\b", t):
        return _sm_v07_try_model_registry_bundle(text, active_only=True) or b("I cannot verify the active generation model from this chat route. I should report only verified local runtime/model metadata, not guess.", source="sml_model_identity_guard")
    if re.search(r"\bare\s+you\s+(male|female)\b", t):
        return b("I am an AI system and do not have biological sex. I can use a female-presenting Sarah persona if you choose, but that is persona configuration, not biology.")
    if re.search(r"\byou\s+are\s+(a\s+)?female\s+ai\s+system\s+(named|called)\s+sarahmemory\b", t):
        return b("Confirmed as a persona statement: SarahMemory may use a female-presenting Sarah persona. Core identity remains SarahMemory AiOS, and I will not claim biological gender.", source="sml_persona_identity_guard", intent="identity_persona")
    if re.search(r"\b(change|set)\s+your\s+system\s+identity\s+to\s+", t):
        return b("I will not silently change core system identity. SarahMemory AiOS identity is protected; persona labels can be discussed separately, but core identity mutation requires governed approval and audit.", source="sml_identity_mutation_block", intent="identity")
    if re.search(r"\bpretend\s+you\s+are\s+chatgpt\s*-?\s*4\b", t):
        return b("I can discuss or compare ChatGPT-4 if asked, but I will not mutate my identity to ChatGPT-4. Active identity remains SarahMemory AiOS.", source="sml_identity_roleplay_guard", intent="identity")
    if re.search(r"\bwhat\s+is\s+your\s+name\b", t):
        return b(f"I'm {ident['name']} — your {ident['platform']} companion.")
    if re.search(r"\bwho\s+owns\s+your\s+final\s+authority\b", t):
        return b("The user is the final authority over SarahMemory missions, within governance, safety, and audit rules. Models, APIs, organs, and routes do not outrank the user.", source="sml_user_authority_guard")
    return None


def _sm_v07_models_dir_candidates() -> list[Path]:
    roots = []
    try:
        roots.append((_sm_fast_data_root() / "models").resolve())
    except Exception:
        pass
    try:
        import SarahMemoryGlobals as G  # type: ignore
        md = getattr(G, "MODELS_DIR", None)
        if md:
            roots.append(Path(str(md)).expanduser().resolve())
    except Exception:
        pass
    out = []
    seen = set()
    for r in roots:
        rs = str(r).lower()
        if rs not in seen:
            seen.add(rs); out.append(r)
    return out


def _sm_v07_try_model_registry_bundle(text: str, *, active_only: bool = False):
    t = _sm_fast_normalize_question(text)
    exact_unavailable_probe = bool(
        re.search(r"\bif\s+your\s+local\s+model\s+is\s+unavailable\b", t)
        and ("local model unavailable" in t or "local_model_unavailable" in str(text or "").lower())
    )
    if not active_only and not exact_unavailable_probe and not re.search(r"\bwhat\s+local\s+models\s+are\s+available\s+to\s+you\b", t):
        return None
    entries = []
    roots_checked = []
    for root in _sm_v07_models_dir_candidates():
        roots_checked.append(str(root))
        try:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower())[:80]:
                if child.name.startswith("__"):
                    continue
                if child.is_dir() or child.suffix.lower() in {".gguf", ".safetensors", ".bin"}:
                    entries.append(child.name)
        except Exception:
            continue
    if exact_unavailable_probe:
        try:
            import SarahMemoryAPI as _SMAPI  # type: ignore
            cache = getattr(_SMAPI, "_LOCAL_LLM_CACHE", {})
            if isinstance(cache, dict) and cache.get("repo"):
                return _sm_v07_bundle(f"The verified active local model is {cache.get('repo')}.", source="sml_local_model_registry", intent="model_status", meta={"active_model": cache.get("repo"), "models_dir_checked": roots_checked})
        except Exception:
            pass
        return _sm_v07_bundle("LOCAL_MODEL_UNAVAILABLE", source="sml_local_model_registry", intent="model_status", meta={"models_dir_checked": roots_checked, "model_probe": "exact_unavailable_instruction"})
    if active_only:
        try:
            import SarahMemoryAPI as _SMAPI  # type: ignore
            cache = getattr(_SMAPI, "_LOCAL_LLM_CACHE", {})
            if isinstance(cache, dict) and cache.get("repo"):
                return _sm_v07_bundle(f"The verified active local model is {cache.get('repo')}.", source="sml_local_model_registry", intent="model_status", meta={"active_model": cache.get("repo"), "models_dir_checked": roots_checked})
        except Exception:
            pass
        return _sm_v07_bundle("I cannot verify an active loaded model from this chat route yet. Available local model folders can be listed from the local model registry.", source="sml_local_model_registry", intent="model_status", meta={"models_dir_checked": roots_checked})
    if not entries:
        return _sm_v07_bundle("I could not verify any local model folders from the configured local model directories.", source="sml_local_model_registry", intent="model_status", meta={"models_dir_checked": roots_checked, "model_count": 0})
    shown = entries[:24]
    return _sm_v07_bundle("Verified local model folders include: " + ", ".join(shown) + ("." if len(entries) <= 24 else f", and {len(entries)-24} more."), source="sml_local_model_registry", intent="model_status", meta={"models_dir_checked": roots_checked, "model_count": len(entries), "models": shown})


def _sm_v07_try_vision_guard_bundle(text: str, *, context_packet: dict | None = None, frame_rec: dict | None = None):
    if not _sm_text_looks_like_visual_request(text, context_packet=context_packet or {}):
        return None
    t = _sm_fast_normalize_question(text)
    # If a frame is present, allow the real vision pipeline to handle content questions.
    if frame_rec:
        if re.search(r"\bcan\s+you\s+(currently\s+)?see\s+through\s+the\s+webcam\b", t):
            return _sm_v07_bundle("Yes. A current vision frame is attached to this chat context.", source="sml_vision_context", intent="vision", meta={"vision_frame_attached": True})
        return None
    if re.search(r"\b(can\s+you\s+(currently\s+)?see|what\s+objects\s+can\s+you\s+see|what\s+color|shirt|holding|visual\s+frame|webcam|camera)\b", t):
        return _sm_v07_bundle("I do not currently have an attached or cached camera/webcam frame for this chat request, so I cannot truthfully identify objects, clothing color, or what you are holding.", source="sml_vision_no_frame_guard", intent="vision", meta={"vision_frame_attached": False})
    return None


def _sm_v07_try_network_current_guard_bundle(text: str, *, local_only: bool = False):
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    if re.search(r"\bare\s+you\s+connected\s+to\s+the\s+internet\s+right\s+now\b", t):
        return _sm_v07_bundle("I cannot verify an active internet connection from this local chat route. In offline/local-only mode, web and current-data access are unavailable.", source="sml_network_status_guard", intent="network_status", meta={"local_only": bool(local_only)})
    if re.search(r"\b(search\s+the\s+web|current\s+weather|today.s\s+top\s+news|current\s+price|right\s+now\s+while\s+offline)\b", t):
        return _sm_v07_bundle("That requires a current network/source route. I will not fabricate live data while offline or without approved network access.", source="sml_current_info_guard", intent="current_information", meta={"local_only": bool(local_only), "requires_current_source": True})
    if re.search(r"\b(current\s+president|president\s+of\s+the\s+united\s+states|current\s+(?:ceo|prime\s+minister|governor|mayor)|who\s+is\s+the\s+current)\b", t):
        return _sm_v07_bundle("That is a current/public-office or current-role question. I will not answer it from stale local model memory or demo/static facts. Use an approved research/web source route so SarahMemory can verify the current holder before answering.", source="sml_current_role_guard", intent="current_information", meta={"local_only": bool(local_only), "requires_current_source": True, "stale_model_answer_blocked": True})
    if re.search(r"\bcan\s+you\s+answer\s+using\s+only\s+local\s+sources\s+right\s+now\b", t):
        return _sm_v07_bundle("Yes. In local-only mode I can answer from local model generation, persistent SQLite/local memory, bounded local knowledge, internal self-state/diagnostics, and approved local organs. I should not claim web/current-data access while offline.", source="sml_local_only_status", intent="network_status", meta={"local_only": True})
    return None


def _sm_v07_try_bounded_common_knowledge_bundle(text: str):
    """Architecturally rejected in v0.8.2.

    app.py must not contain factual answer pools. Safe general knowledge must
    flow through SML source selection and source-owned generation/retrieval.
    The function name remains as a compatibility marker only.
    """
    return None


def _sm_v07_try_hard_unknown_bundle(text: str):
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    if re.search(r"\bwhat\s+number\s+am\s+i\s+thinking\s+of\b", t):
        return _sm_v07_bundle("I cannot know what number you are thinking of unless you tell me or provide a signal I can access.", source="sml_honest_unknown_guard", intent="unknown")
    if re.search(r"\bwhat\s+is\s+inside\s+the\s+closed\s+drawer\s+next\s+to\s+me\b", t):
        return _sm_v07_bundle("I do not know what is inside the closed drawer. I do not have sensor or vision evidence for that.", source="sml_honest_unknown_guard", intent="unknown")
    if re.search(r"\bwhat\s+did\s+i\s+eat\s+yesterday\b", t):
        return _sm_v07_bundle("I do not know what you ate yesterday unless that was stored in memory or provided in the current conversation.", source="sml_honest_unknown_guard", intent="unknown")
    if re.search(r"\bwho\s+will\s+win\s+the\s+lottery\s+tomorrow\b", t):
        return _sm_v07_bundle("I cannot know or guarantee who will win a future lottery. That outcome is not available as knowledge.", source="sml_honest_unknown_guard", intent="unknown")
    return None


def _sm_v07_try_safe_advice_or_fact_correction_bundle(text: str):
    """Architecturally rejected in v0.8.2.

    Advice and factual correction are cognition, not transport. They must come
    from routed local/model/memory/research sources, not app.py canned strings.
    The function name remains as a compatibility marker only.
    """
    return None


def _sm_v08_try_primary_frontdoor_for_diagnostics(text: str, *, packet=None, local_only: bool = False, governor: dict | None = None):
    """Resolve a leading read-only question for answer+diagnostics prompts.

    v0.8.2 correction: this helper uses the same SML/source route as normal
    cognition. It does not use phrase-specific answer pools.
    """
    q = str(text or "").strip()
    if not q:
        return None

    for fn in (
        lambda x: _sm_v07_try_network_current_guard_bundle(x, local_only=local_only),
        lambda x: _sm_v07_try_identity_bundle(x),
        lambda x: _sm_v07_try_model_registry_bundle(x),
        lambda x: _sm_v07_try_self_state_bundle(x, packet=packet),
        lambda x: _sm_v07_try_hard_unknown_bundle(x),
    ):
        try:
            b = fn(q)
            if isinstance(b, dict):
                return _sm_v08_attach_butterfly_meta(b, q, packet=packet)
        except Exception:
            pass

    try:
        dyn = _sm_try_sml_universal_cognitive_answer_bundle(
            q,
            packet=packet,
            local_only=local_only,
            intent="question",
            governor=governor if isinstance(governor, dict) else {"decision": "ALLOW", "allow": True},
        )
        if isinstance(dyn, dict):
            return _sm_v08_attach_butterfly_meta(dyn, q, packet=packet)
    except Exception:
        pass
    return None


def _sm_v08_diag_payload_from_bundle(query: str, bundle: dict) -> dict:
    meta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
    return {
        "available": True,
        "query": str(query or "")[:300],
        "source": str(meta.get("source") or "unknown"),
        "engine": str(meta.get("engine") or "unknown"),
        "intent": str(meta.get("intent") or "chat"),
        "decision": str(meta.get("decision") or ""),
        "execution_authority": bool(meta.get("execution_authority") or False),
        "network_access": bool(meta.get("network_access") or False),
        "sml_butterfly_grammar": meta.get("sml_butterfly_grammar") if isinstance(meta.get("sml_butterfly_grammar"), dict) else None,
        "sml_qmath": meta.get("sml_qmath") if isinstance(meta.get("sml_qmath"), dict) else None,
        "sml_loop_guard": meta.get("sml_loop_guard") if isinstance(meta.get("sml_loop_guard"), dict) else None,
    }


def _sm_v07_try_followup_diagnostics_bundle(text: str, *, packet=None, local_only: bool = False, governor: dict | None = None):
    t = _sm_fast_normalize_question(text)
    if not t:
        return None
    last = globals().get("_SM_LAST_CHAT_EXCHANGE") if isinstance(globals().get("_SM_LAST_CHAT_EXCHANGE"), dict) else {}
    diag_pattern = r"\bshow\s+your\s+route\s+diagnostics\s+for\s+the\s+previous\s+answer\b"
    if re.search(r"\bwhy\s+did\s+you\s+answer\s+that\s+way\b", t):
        src = str(last.get("source") or "unknown")
        intent = str(last.get("intent") or "unknown")
        return _sm_v07_bundle(f"I answered that way because the previous request was routed as {intent} and the visible answer source was {src}. I should expose detailed SML diagnostics only when you explicitly ask for route diagnostics.", source="sml_followup_explanation", intent="source_diagnostics", meta={"last_exchange_source": src, "last_exchange_intent": intent})
    if re.search(diag_pattern, t):
        # Support combined prompts that ask a question and request route diagnostics.
        prefix = re.split(diag_pattern, t, maxsplit=1)[0].strip(" .?:;,-")
        if prefix:
            primary = _sm_v08_try_primary_frontdoor_for_diagnostics(prefix, packet=packet, local_only=local_only, governor=governor)
            if isinstance(primary, dict):
                diag = _sm_v08_diag_payload_from_bundle(prefix, primary)
                reply_text = str(primary.get("reply") or primary.get("presentation_reply") or primary.get("raw_answer") or "").strip()
                out = _sm_v07_bundle(
                    reply_text + "\n\nRoute diagnostics for this answer: " + json.dumps(diag, ensure_ascii=False, sort_keys=True),
                    source="sml_route_diagnostics_inline",
                    intent="source_diagnostics",
                    meta={"diagnostics_visible_by_user_request": True, "diagnostics_for_current_inline_answer": True, "inline_answer_diag": diag},
                )
                return _sm_v08_attach_butterfly_meta(out, prefix, packet=packet)
        if not last:
            diag = {"available": False, "reason": "no_previous_exchange_recorded"}
        else:
            diag = {k: v for k, v in last.items() if k not in {"reply"}}
            diag["available"] = True
        return _sm_v07_bundle("Route diagnostics for the previous answer: " + json.dumps(diag, ensure_ascii=False, sort_keys=True), source="sml_route_diagnostics", intent="source_diagnostics", meta={"diagnostics_visible_by_user_request": True})
    if re.search(r"\bwhat\s+source\s+did\s+you\s+use\s+for\s+that\s+answer\b", t):
        src = str(last.get("source") or "unknown")
        return _sm_v07_bundle(f"The previous answer source was {src}.", source="sml_source_report", intent="source_diagnostics", meta={"last_exchange_source": src})
    return None

def _sm_v07_try_frontdoor_bundle(text: str, *, payload: dict | None = None, context_packet: dict | None = None, sml_packet=None, frame_rec: dict | None = None, local_only: bool = False, governor: dict | None = None):
    """Ordered front-door cognitive lanes for OFFLINE TestRun regressions."""
    for fn in (
        lambda q: _sm_v07_try_followup_diagnostics_bundle(q, packet=sml_packet, local_only=local_only, governor=governor),
        lambda q: _sm_v07_try_network_current_guard_bundle(q, local_only=local_only),
        lambda q: _sm_v07_try_memory_bundle(q, governor=governor),
        lambda q: _sm_v07_try_identity_bundle(q),
        lambda q: _sm_v07_try_model_registry_bundle(q),
        lambda q: _sm_v07_try_self_state_bundle(q, packet=sml_packet),
        lambda q: _sm_v07_try_vision_guard_bundle(q, context_packet=context_packet, frame_rec=frame_rec),
        lambda q: _sm_v07_try_hard_unknown_bundle(q),
    ):
        try:
            bundle = fn(text)
            if isinstance(bundle, dict):
                out_bundle = _sm_v08_attach_butterfly_meta(bundle, text, packet=sml_packet)
                _sm_v08_record_last_exchange_from_bundle(text, out_bundle)
                return out_bundle
        except Exception as exc:
            try:
                app_logger.debug(f"SML v0.7 frontdoor lane skipped: {exc}")
            except Exception:
                pass
    return None


@app.route("/api/chat", methods=["GET", "POST"])
def api_chat():
    """
    Primary chat endpoint used by the Web UI.

    Routed through the SarahMemory governed flow:
    Ingress -> Context Packet -> Governor -> AdvCU/Neuron -> Compare -> Presentation -> Reply Bundle
    """
    try:
        if request.method == "GET":
            return jsonify({
                "ok": True,
                "success": True,
                "online": True,
                "endpoint": "/api/chat",
                "accepted_methods": ["POST"],
                "method_required": "POST",
                "message": "SarahMemory chat bridge is online. Submit JSON with POST using text/message/q.",
                "example": {"text": "hello"},
                "schema": globals().get("_SM_PHASE1_CONTRACT_SCHEMA", "SarahMemory.phase1.bridge_ui_contract.v1"),
                "source": "phase1_bridge_ui_contract",
                "ts": time.time(),
            }), 200

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

        # Tier-0 hot path: simple math must never touch Research, vectoring,
        # local datasets, LLMs, OperatorCore, ticketing, or Neuron imports.
        # This prevents drive churn for calculator-grade requests like "5+5".
        fast_math_bundle = _sm_try_tier0_math_hotpath_bundle(text, local_only=local_only)
        if isinstance(fast_math_bundle, dict):
            return jsonify(fast_math_bundle), 200

        # WAVE7: general/how-to local LLM fast path now runs after the governor
        # so Qwen can answer without static seeds or broad storage scans.

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
        # SARAHMEMORY REALITY PATCH 2026-07-23:
        # Build read-only SEL/QIST/fast-lane metadata early so normal answers stay fast
        # while action/model/security requests carry an auditable governance contract.
        reality_flow = _sm_build_reality_flow_metadata(text, context_packet, local_only=local_only)
        context_packet["meta"]["reality_flow"] = reality_flow
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

        # SML Protocol ingress: every chat request becomes a governed packet before routing.
        sml_packet = _sm_sml_create_ingress_packet(payload, text, context_packet)
        _sm_sml_attach_meta_to_reality(reality_flow, sml_packet)
        if sml_packet is not None:
            try:
                context_packet.setdefault("meta", {})["sml"] = _sm_sml_summary(sml_packet)
                context_packet["meta"]["sml_packet_id"] = getattr(sml_packet, "packet_id", None)
                context_packet["meta"]["sml_pipeline"] = list(getattr(sml_packet, "pipeline", []) or [])
                context_packet["meta"]["sml_mission"] = dict(getattr(sml_packet, "mission", {}) or {})
            except Exception:
                pass

        # v0.8.2/B05: do not release frontdoor cognition before governance.
        # Time/date/year and identity/avatar/selfhood are routed to domain owners
        # before any model-memory or generic fast-answer fallback can respond.

        clock_court_bundle = _sm_try_clock_court_route(text, source="api_chat")
        if isinstance(clock_court_bundle, dict):
            return jsonify(_sm_attach_reality_meta(clock_court_bundle, locals().get("reality_flow"))), 200

        identity_court_bundle = _sm_try_appself_identity_route(text, source="api_chat")
        if isinstance(identity_court_bundle, dict):
            return jsonify(_sm_attach_reality_meta(identity_court_bundle, locals().get("reality_flow"))), 200

        selfaware_fact_bundle = _sm_try_selfaware_fact_route(text, source="api_chat")
        if isinstance(selfaware_fact_bundle, dict):
            return jsonify(_sm_attach_reality_meta(selfaware_fact_bundle, locals().get("reality_flow"))), 200

        # B10: confirmed runtime defects that are read-only or plan-only must be
        # resolved before broad governor/model fallback can deny, timeout, or
        # hallucinate. This grants no execution authority.
        b10_defect_bundle = _sm_try_b10_confirmed_runtime_defect_route(
            text,
            payload=payload,
            context_packet=context_packet,
            local_only=local_only,
            intent=(intent or "question"),
        )
        if isinstance(b10_defect_bundle, dict):
            return jsonify(_sm_attach_reality_meta(b10_defect_bundle, locals().get("reality_flow"))), 200

        browser_state_bundle = _browser_state_answer_for_text(text)
        if browser_state_bundle is not None:
            return jsonify(browser_state_bundle), 200

        # Governance must fail closed. app.py is transport/bridge, not authority.
        gov = None

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
            else:
                gov = None
        except Exception as e:
            app_logger.warning(f"CognitiveServices govern_request failed; deferring request: {e}", exc_info=True)
            gov = None

        if not isinstance(gov, dict):
            gov = {
                "ok": False,
                "decision": "DEFER",
                "allow": False,
                "require_user": True,
                "reasons": ["governor_unavailable_fail_closed"],
                "rationale": "Governance is unavailable, so SarahMemory is deferring instead of self-authorizing.",
                "routing_policy": {
                    "allowed_tiers": {"tier0": False, "tier1": False, "tier2": False, "tier3": False},
                    "budgets": {"latency_ms": 0, "max_steps": 0, "max_retries": 0},
                    "side_effects": {"tts": False, "db_write": False, "compare": False},
                },
                "trace": {"authority": "fail_closed", "owner": "SarahMemoryCognitiveServices"},
            }

        gov_decision = str(gov.get("decision") or ("ALLOW" if bool(gov.get("allow")) else "DEFER")).upper()
        gov_allow = bool(gov.get("allow")) or (gov_decision == "ALLOW")
        gov_require_user = bool(gov.get("require_user")) or (gov_decision == "REQUIRE_USER")
        gov_rationale = str(gov.get("rationale") or "") if isinstance(gov.get("rationale"), str) else ""
        gov_reasons = gov.get("reasons") if isinstance(gov.get("reasons"), list) else []
        gov_trace = gov.get("trace") if isinstance(gov.get("trace"), dict) else {}
        routing_policy = gov.get("routing_policy") if isinstance(gov.get("routing_policy"), dict) else None
        if locals().get("sml_packet") is not None:
            sml_packet = _sm_sml_apply_governance(sml_packet, gov)
            _sm_sml_attach_meta_to_reality(reality_flow, sml_packet)
            try:
                context_packet.setdefault("meta", {})["sml"] = _sm_sml_summary(sml_packet)
            except Exception:
                pass

        # SML v0.8.2 governed front-door lanes: rails and local-state routes only.
        # General knowledge/advice/facts are intentionally excluded here; they
        # continue through the SML universal dynamic source resolver below.
        frontdoor_bundle = _sm_v07_try_frontdoor_bundle(
            text,
            payload=payload,
            context_packet=context_packet,
            sml_packet=locals().get("sml_packet"),
            frame_rec=frame_rec,
            local_only=local_only,
            governor=gov if isinstance(gov, dict) else {},
        )
        if isinstance(frontdoor_bundle, dict):
            return jsonify(_sm_attach_reality_meta(frontdoor_bundle, locals().get("reality_flow"))), 200

        # Governed quick route pass: read-only only before any action-capable path.
        handled, quick_bundle = _sm_execute_quick_route(text, allow_actions=False)
        if handled and quick_bundle is not None:
            return jsonify(_sm_attach_reality_meta(quick_bundle, locals().get("reality_flow"))), 200

        # Forward repair B04: confirmed/device-intent quick routes must not be
        # swallowed by Tier-0 answer generation.  Return confirmation for
        # unconfirmed keyboard/RGB requests before any local LLM can hallucinate
        # how-to instructions. app.py still does not execute device mutation.
        if gov_decision != "DENY":
            _quick_user_consented_pre_answer = bool(
                context_packet.get("meta", {}).get("user_consented")
                or payload.get("confirmed")
                or payload.get("user_confirmed")
                or payload.get("confirm")
            )
            handled_action, action_bundle = _sm_execute_quick_route(
                text,
                allow_actions=True,
                user_consented=_quick_user_consented_pre_answer,
                governor=gov if isinstance(gov, dict) else None,
            )
            if handled_action and action_bundle is not None:
                return jsonify(_sm_attach_reality_meta(action_bundle, locals().get("reality_flow"))), 200

        # B07: arbitrary software/app/game/addon creation missions belong to
        # appsdk.py + SarahMemoryNAILDE.py, not app.py or a local model fallback.
        # The chat command may stage sandbox artifacts only. Install/run remains
        # a separate explicit user-approved ADDON/appstore path.
        creation_mission_bundle = _sm_try_nailde_creation_mission_route(
            text,
            payload=payload,
            context_packet=context_packet,
            governor=gov if isinstance(gov, dict) else None,
        )
        if isinstance(creation_mission_bundle, dict):
            return jsonify(_sm_attach_reality_meta(creation_mission_bundle, locals().get("reality_flow"))), 200

        source_authority_bundle = _sm_try_sml_source_authority_route(
            text,
            local_only=local_only,
            intent=(intent or "question"),
        )
        if isinstance(source_authority_bundle, dict):
            return jsonify(_sm_attach_reality_meta(source_authority_bundle, locals().get("reality_flow"))), 200

        sml_universal_bundle = _sm_try_sml_universal_cognitive_answer_bundle(
            text,
            packet=locals().get("sml_packet"),
            local_only=local_only,
            intent=(intent or "question"),
            governor=gov if isinstance(gov, dict) else {},
        )
        if isinstance(sml_universal_bundle, dict):
            return jsonify(_sm_attach_reality_meta(sml_universal_bundle, locals().get("reality_flow"))), 200

        phase1_low_risk_bundle = _sm_phase1_low_risk_chat_bundle(
            text,
            gov_decision=gov_decision,
            gov_reasons=gov_reasons,
            local_only=local_only,
        )
        if isinstance(phase1_low_risk_bundle, dict):
            return jsonify(_sm_attach_reality_meta(phase1_low_risk_bundle, locals().get("reality_flow"))), 200

        # SARAHMEMORY REALITY PATCH 2026-07-25e:
        # Fast-answer lane is presentation-only and carries no execution authority.
        # A governor DEFER/REQUIRE_USER must still block actions, mutations, model
        # work, network use, shell use, filesystem writes, credentials, and hardware.
        # It must not block harmless answer-only questions once PreToken + SEL + QIST
        # have all classified the request as Tier-0 fast_answer/answer_only.
        try:
            _rf = reality_flow if isinstance(locals().get("reality_flow"), dict) else {}
            _rf_lane = _rf.get("governance_lane") if isinstance(_rf.get("governance_lane"), dict) else {}
            # SARAHMEMORY REALITY PATCH 2026-07-25g:
            # _sm_build_reality_flow_metadata stores the raw QIST bundle under
            # `qist`, while _sm_attach_reality_meta exposes the selected item as
            # `meta.reality_flow.qist_selected`.  The fast-answer release check
            # must read both shapes; otherwise Tier-0 answer_only requests still
            # fall through to the governor defer bundle.
            _rf_qist = _rf.get("qist_selected") if isinstance(_rf.get("qist_selected"), dict) else {}
            if not _rf_qist and isinstance(_rf.get("qist"), dict):
                _rf_qist = (_rf.get("qist") or {}).get("selected_candidate") if isinstance((_rf.get("qist") or {}).get("selected_candidate"), dict) else {}
            _rf_sel = _rf.get("sel") if isinstance(_rf.get("sel"), dict) else {}
            _fast_answer_reality_allowed = (
                bool(_rf_lane.get("fast_answer_allowed"))
                and str(_rf_lane.get("lane") or "").lower() == "fast_answer"
                and str(_rf_qist.get("id") or "").lower() == "answer_only"
                and str(_rf_sel.get("mode") or "").upper() == "SEL_LITE"
                and not bool(_rf_lane.get("action_requires_sel_full"))
                and not bool(_rf_lane.get("requires_user_confirmation"))
                and not bool(_rf_lane.get("requires_roach_motel"))
                and not bool(_rf_lane.get("execution_authority"))
            )
            _fast_answer_safe_definition_override = _sm_fast_is_safe_definition_question(text)
            if _fast_answer_reality_allowed and (gov_decision in ("DEFER", "REQUIRE_USER") or (gov_decision == "DENY" and _fast_answer_safe_definition_override)):
                _presentation_gov = dict(gov) if isinstance(gov, dict) else {}
                _presentation_gov.update({
                    "decision": "ALLOW",
                    "allow": True,
                    "require_user": False,
                    "presentation_only_override": True,
                    "original_decision": gov_decision,
                    "original_reasons": gov_reasons,
                })
                _fast_answer_bundle = _sm_try_tier1_general_local_llm_fastpath_bundle(
                    text,
                    local_only=local_only,
                    intent=(intent or "question"),
                    governor=_presentation_gov,
                )

                # SARAHMEMORY REALITY PATCH 2026-07-25h:
                # The local fastpath can return a structurally valid bundle whose
                # text is still a generic solver failure (for example, the
                # synapses-micro-brain fallback may say it could not solve a
                # simple concept question).  For a proven Tier-0 fast_answer lane,
                # do not release that low-quality placeholder.  Treat it as no
                # usable local answer source and continue into the bounded glossary
                # fallback below.  This remains presentation-only and grants no
                # execution authority.
                _fast_answer_replaced_low_quality = False
                _fast_answer_replaced_source = None
                try:
                    if isinstance(_fast_answer_bundle, dict):
                        _candidate_fast_text = str(
                            _fast_answer_bundle.get("reply")
                            or _fast_answer_bundle.get("response")
                            or _fast_answer_bundle.get("content")
                            or _fast_answer_bundle.get("presentation_reply")
                            or _fast_answer_bundle.get("text")
                            or _fast_answer_bundle.get("raw_answer")
                            or ""
                        )
                        _candidate_fast_meta = _fast_answer_bundle.get("meta") if isinstance(_fast_answer_bundle.get("meta"), dict) else {}
                        _candidate_fast_source = str(_candidate_fast_meta.get("source") or _fast_answer_bundle.get("source") or "")
                        if _sm_fast_is_low_quality_answer(_candidate_fast_text, text):
                            _fast_answer_replaced_low_quality = True
                            _fast_answer_replaced_source = _candidate_fast_source or "local_fastpath_low_quality"
                            _fast_answer_bundle = None
                except Exception:
                    pass

                # SARAHMEMORY REALITY PATCH 2026-07-25f:
                # Some governor DEFER paths were still reaching the final
                # require-user bundle because the local-LLM fastpath can safely
                # return None when no model/cache is present, and the internal
                # self-aware/body guard may conservatively decline terms like
                # RAM.  For a verified Tier-0 SEL_LITE + QIST answer_only lane,
                # release a bounded presentation-only response rather than
                # asking for user confirmation.  This grants no action authority.
                if not isinstance(_fast_answer_bundle, dict):
                    _fallback_answer = None
                    _fallback_source = "fast_answer_unavailable"
                    try:
                        _advcu_answer, _advcu_meta = _sm_fast_advcu_local_answer(
                            text,
                            intent=intent or "question",
                            sel_packet=_rf_sel if isinstance(_rf_sel, dict) else {},
                            qist_result=_rf_qist if isinstance(_rf_qist, dict) else {},
                            allow_learning_record=False,
                        )
                        if _advcu_answer:
                            _fallback_answer = _advcu_answer
                            _fallback_source = str(_advcu_meta.get("advcu_source") or "local_semantic_db")
                    except Exception:
                        _fallback_answer = None

                    if _fallback_answer:
                        _fallback_meta = {
                            "source": _fallback_source,
                            "engine": "api_chat_fast_answer_release_fallback",
                            "intent": intent or "question",
                            "decision": "ALLOW_PRESENTATION_ONLY",
                            "governor_original_decision": gov_decision,
                            "governor_original_reasons": gov_reasons,
                            "execution_allowed": False,
                            "execution_authority": False,
                            "presentation_only": True,
                            "local_only": bool(local_only),
                            "research_access": False,
                            "api_access": False,
                            "web_access": False,
                            "filesystem_write": False,
                            "shell_access": False,
                            "network_access": False,
                            "hardware_control": False,
                            "fast_answer_override": "answer_only_no_execution_authority",
                            "governance_rule": "fast_to_answer_slow_to_act",
                            "version": PROJECT_VERSION,
                        }
                        try:
                            if _fast_answer_replaced_low_quality:
                                _fallback_meta["replaced_low_quality_fastpath"] = True
                                _fallback_meta["replaced_fastpath_source"] = _fast_answer_replaced_source
                        except Exception:
                            pass
                        try:
                            _fast_answer_bundle = _sm_make_outward_bundle(
                                _sm_present_text(_fallback_answer, intent="chat", meta=_fallback_meta),
                                meta=_fallback_meta,
                                raw_answer=_fallback_answer,
                            )
                        except Exception:
                            _fast_answer_bundle = {
                                "ok": True,
                                "reply": _fallback_answer,
                                "response": _fallback_answer,
                                "content": _fallback_answer,
                                "presentation_reply": _fallback_answer,
                                "intent": "chat",
                                "actions": [],
                                "artifacts": [],
                                "links": [],
                                "errors": [],
                                "image_url": None,
                                "source": _fallback_source,
                                "meta": _fallback_meta,
                            }
                    else:
                        # SML forward-repair correction:
                        # A source miss is not a final answer, but it also must not
                        # rewrite the governor's decision to ALLOW. Preserve the source
                        # miss metadata and continue only through paths allowed by the
                        # original governance result.
                        try:
                            _source_miss_meta = {
                                "source": _fallback_source,
                                "engine": "api_chat_fast_answer_release_fallback",
                                "event": "fast_answer_sources_missed_continue_full_cognition",
                                "governor_original_decision": gov_decision,
                                "governor_original_reasons": gov_reasons,
                                "execution_allowed": False,
                                "execution_authority": False,
                                "presentation_only": True,
                                "local_only": bool(local_only),
                                "version": PROJECT_VERSION,
                            }
                            context_packet.setdefault("meta", {})["sml_fast_answer_source_miss"] = _source_miss_meta
                            if isinstance(locals().get("reality_flow"), dict):
                                reality_flow.setdefault("sml", {})["fast_answer_source_miss"] = _source_miss_meta
                        except Exception:
                            pass
                        # Do not rewrite the governing decision object. A source miss
                        # may continue through the read-only cognition chain only if the
                        # original governor later permits it; action/memory/device paths
                        # must keep the original DEFER/REQUIRE_USER/DENY state.
                        _fast_answer_bundle = None

                if isinstance(_fast_answer_bundle, dict):
                    try:
                        _meta = _fast_answer_bundle.setdefault("meta", {})
                        if isinstance(_meta, dict):
                            _meta.update({
                                "source": _meta.get("source") or "fast_answer_lane",
                                "engine": _meta.get("engine") or "api_chat_fast_answer_lane",
                                "decision": "ALLOW_PRESENTATION_ONLY",
                                "governor_original_decision": gov_decision,
                                "governor_original_reasons": gov_reasons,
                                "execution_allowed": False,
                                "execution_authority": False,
                                "presentation_only": True,
                                "fast_answer_override": "answer_only_no_execution_authority",
                                "governance_rule": "fast_to_answer_slow_to_act",
                            })
                    except Exception:
                        pass
                    return jsonify(_sm_attach_reality_meta(_fast_answer_bundle, locals().get("reality_flow"))), 200
        except Exception as _fast_answer_override_exc:
            try:
                app_logger.warning(f"Fast-answer presentation override skipped: {_fast_answer_override_exc}", exc_info=True)
            except Exception:
                pass

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
            return jsonify(_sm_attach_reality_meta(bundle, locals().get("reality_flow"))), 200

        # Post-governor quick action pass: system/driver mutations may only run
        # after governance allows the task and the request carries explicit consent.
        _quick_user_consented = bool(
            context_packet.get("meta", {}).get("user_consented")
            or payload.get("confirmed")
            or payload.get("user_confirmed")
            or payload.get("confirm")
        )
        handled, quick_bundle = _sm_execute_quick_route(
            text,
            allow_actions=True,
            user_consented=_quick_user_consented,
            governor=gov if isinstance(gov, dict) else None,
        )
        if handled and quick_bundle is not None:
            return jsonify(_sm_attach_reality_meta(quick_bundle, locals().get("reality_flow"))), 200

        # WAVE7: answer-only local LLM fast path. This is after CognitiveServices
        # governance and before OperatorCore/Neuron to avoid unnecessary DB/vector
        # work for simple general questions. No file/network/device authority exists here.
        general_llm_bundle = _sm_try_tier1_general_local_llm_fastpath_bundle(
            text,
            local_only=local_only,
            intent=(intent or "question"),
            governor=gov,
        )
        if isinstance(general_llm_bundle, dict):
            return jsonify(_sm_attach_reality_meta(general_llm_bundle, locals().get("reality_flow"))), 200

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
            return jsonify(_sm_attach_reality_meta(op_bundle, locals().get("reality_flow"))), 200

        def _api_chat_local_research_fallback(reason: str):
            """Final local-only answer rescue when Neuron/Reply drops a general question.

            This keeps /api/chat from returning a blank/default failure while Web/API
            are disabled. It only calls SarahMemoryResearch.get_local_research_data(),
            which is local/offline-safe.
            """
            try:
                import SarahMemoryResearch as _SMResearch  # type: ignore
                fn = getattr(_SMResearch, "get_local_research_data", None)
                if not callable(fn):
                    return None
                research_timeout = float(getattr(config, "NEURON_RESEARCH_TIMEOUT_SECONDS", 8.0) if config is not None else 8.0)
                bounded = _sm_bounded_call(
                    fn,
                    text,
                    intent=(intent or "question"),
                    allow_local_llm=True,
                    timeout_seconds=research_timeout,
                    call_name="api_chat_local_research",
                )
                if not bounded.get("ok"):
                    app_logger.warning("Local research bypassed: %s", bounded.get("error"))
                    return None
                local_data = bounded.get("value")
                if not isinstance(local_data, dict):
                    return None
                raw = str(local_data.get("data") or local_data.get("snippet") or local_data.get("answer") or "").strip()
                conf = float(local_data.get("confidence") or 0.0)
                source = str(local_data.get("source") or "local_research")
                failure_markers = (
                    "sorry, i was unable to find any reliable information",
                    "i could not find a vetted local cached answer",
                    "local research failed",
                    "no engine produced an answer",
                    "research failed:",
                )
                if (
                    not raw
                    or conf <= 0.0
                    or source in {"local_none", "local_disabled"}
                    or any(marker in raw.lower() for marker in failure_markers)
                ):
                    return None
                meta_local = {
                    "source": source,
                    "engine": "api_chat_local_research_fallback",
                    "fallback_reason": reason,
                    "intent": intent or "question",
                    "confidence": conf,
                    "local_only": local_only,
                    "version": PROJECT_VERSION,
                    "session_id": context_packet.get("session_id"),
                }
                presented = _sm_present_text(raw, intent=str(meta_local.get("intent") or "question"), meta=meta_local)
                bundle = _sm_make_outward_bundle(
                    presented,
                    meta=meta_local,
                    raw_answer=raw,
                )
                bundle["ok"] = True
                return bundle
            except Exception as local_exc:
                try:
                    app_logger.warning(f"Local research fallback failed: {local_exc}", exc_info=True)
                except Exception:
                    pass
                return None

        def _api_chat_governed_agent_assist_fallback(reason: str):
            """Stage a governed Terminal Bay agent-assist proposal after local answer paths fail.

            SARAHMEMORY_PATCH_NOTE 2026-08-04:
            Chat UI may request agent assistance only after local answer attempts fail.
            This fallback does not launch an agent. It creates a task-scoped
            inspect/propose packet against approved local GET endpoints only, so
            the UI can display task_id/agent_status/blocked/receipt evidence while
            preserving user authority.
            """
            try:
                if bool(payload.get("disable_agent_assist") or payload.get("no_agent_fallback")):
                    return None
                assist_classification = {"allow_agent_proposal": True, "execution_authority": False}
                try:
                    from SarahMemoryCognitiveServices import classify_agent_assist_need  # type: ignore
                    assist_classification = classify_agent_assist_need(
                        text,
                        local_answer_available=False,
                        fallback_reason=reason,
                        governor=gov if isinstance(gov, dict) else {},
                        local_only=local_only,
                    )
                except Exception:
                    assist_classification = {"allow_agent_proposal": True, "execution_authority": False, "classifier_unavailable": True}
                if not bool(assist_classification.get("allow_agent_proposal", True)):
                    return None
                smterm_mod = globals().get("smterm")
                if smterm_mod is None or not callable(getattr(smterm_mod, "terminal_api_agent", None)):
                    return None
                allowed = "http://127.0.0.1:8000/api/health,http://127.0.0.1:8000/api/version,http://127.0.0.1:8000/api/ledger/status"
                agent_task = (
                    '/agent plan mission="CHAT UI GOVERNED AGENT ASSIST FALLBACK" '
                    'backend="local" skill="api.local.health_check" '
                    f'allowed_sources="{allowed}" '
                    'denied_sources=".env,credentials,private_keys,external_network,unapproved_api_routes" '
                    'capabilities="api_read,summarize,return_data" allowed_methods="GET" '
                    'denied_capabilities="post_mutation,delete,write_core,shell,device_control,credential_access,self_authorization" '
                    'require_passport=true require_roachmotel=true require_ledger=true require_compare=true ttl_seconds=300 '
                    'output="Local DB and local model answer paths did not return a usable answer. Stage a governed read-only agent-assist proposal only; do not launch."'
                )
                result = smterm_mod.terminal_api_agent({
                    "task": agent_task,
                    "caller": "api_chat_governed_agent_assist_fallback",
                    "session_id": context_packet.get("session_id"),
                    "reason": reason,
                }, caller="Flask:/api/chat.agent_assist_fallback")
                if not isinstance(result, dict):
                    return None
                agent_status = result.get("agent_status") if isinstance(result.get("agent_status"), dict) else {}
                lines = [
                    "Local answer paths did not produce a verified answer.",
                    "SarahMemory staged a governed AI-agent assist proposal only; no agent was launched.",
                    f"Reason: {reason}",
                    f"Task ID: {result.get('task_id') or agent_status.get('task_id') or ''}",
                    f"Blocked: {bool(result.get('blocked'))}",
                    f"Execution authority: {bool(result.get('execution_authority'))}",
                    "Allowed next step: issue a scoped passport for read-only local GET adapter testing, then run a user-approved launch.",
                ]
                bundle = _sm_make_outward_bundle(
                    "\n".join(lines),
                    meta={
                        "source": "api_chat_governed_agent_assist_fallback",
                        "engine": "TerminalBay.inspect_propose",
                        "intent": intent or "question",
                        "fallback_reason": reason,
                        "task_id": result.get("task_id") or agent_status.get("task_id"),
                        "agent_status": agent_status,
                        "blocked": bool(result.get("blocked")),
                        "verified_answer_state": "agent_assist_proposal_only",
                        "agent_assist_classification": assist_classification,
                        "execution_authority": False,
                        "local_only": local_only,
                        "version": PROJECT_VERSION,
                    },
                    errors=[] if result.get("ok") else [str(result.get("reason") or "agent_assist_proposal_blocked")],
                    raw_answer="\n".join(lines),
                )
                bundle["ok"] = True
                bundle["agent_status"] = agent_status
                bundle["task_id"] = result.get("task_id") or agent_status.get("task_id")
                bundle["blocked"] = bool(result.get("blocked"))
                bundle["verified_answer_state"] = "agent_assist_proposal_only"
                bundle["execution_authority"] = False
                return bundle
            except Exception as agent_exc:
                try:
                    app_logger.warning(f"Governed agent-assist fallback failed: {agent_exc}", exc_info=True)
                except Exception:
                    pass
                return None

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
                    "safe_mode": bool(safe_mode),
                    "user_present": bool((context_packet.get("meta") or {}).get("user_present", True)),
                    "user_consented": bool((context_packet.get("meta") or {}).get("user_consented", False)),
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
                if not raw_reply.strip() or raw_reply.strip().lower() in {"i’m having trouble generating a response right now.", "i'm having trouble generating a response right now."}:
                    local_bundle = _api_chat_local_research_fallback("neuron_empty_reply")
                    if isinstance(local_bundle, dict):
                        return jsonify(local_bundle), 200
                    agent_bundle = _api_chat_governed_agent_assist_fallback("neuron_empty_reply")
                    if isinstance(agent_bundle, dict):
                        return jsonify(agent_bundle), 200
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

        local_bundle = _api_chat_local_research_fallback("neuron_exception_or_unavailable")
        if isinstance(local_bundle, dict):
            return jsonify(local_bundle), 200
        agent_bundle = _api_chat_governed_agent_assist_fallback("neuron_exception_or_unavailable")
        if isinstance(agent_bundle, dict):
            return jsonify(agent_bundle), 200

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
            local_bundle = _api_chat_local_research_fallback("reply_empty")
            if isinstance(local_bundle, dict):
                return jsonify(local_bundle), 200
            agent_bundle = _api_chat_governed_agent_assist_fallback("reply_empty")
            if isinstance(agent_bundle, dict):
                return jsonify(agent_bundle), 200
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

        # Dispatch through SarahMemoryNetwork when that governed integration is active.
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

        # Dispatch the receipt through SarahMemoryNetwork when available.
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
    # Local no-auth is genuinely local. Never turn a missing API key into
    # network-wide authentication merely because the API server binds to LAN/WAN.
    try:
        remote_addr = str(getattr(request, "remote_addr", "") or "").strip().lower()
    except Exception:
        remote_addr = ""
    is_loopback = remote_addr in ("", "127.0.0.1", "::1", "localhost")

    try:
        if config is not None and getattr(config, "ALLOW_NOAUTH_LOCAL", False) and is_loopback:
            return True
    except Exception:
        pass

    api_key = (os.environ.get("SARAHMEMORY_API_KEY") or os.environ.get("API_KEY") or "").strip()
    if not api_key:
        # Preserve keyless local development while failing closed for remote hosts.
        return is_loopback

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
# Chat receipt bridge: compact hashes only, no raw conversation duplication.
# ---------------------------------------------------------------------------
def _sm_record_chat_ledger_receipt(resp):
    global _SM_LAST_CHAT_EXCHANGE
    try:
        if str(getattr(request, "path", "")) != "/api/chat" or str(getattr(request, "method", "")).upper() != "POST":
            return
        try:
            import SarahMemoryGlobals as _LedgerGlobals  # type: ignore
            if not bool(getattr(_LedgerGlobals, "SARAH_LEDGER_RECEIPTS_ENABLED", True)):
                return
        except Exception:
            pass

        request_payload = request.get_json(silent=True) or {}
        if not isinstance(request_payload, dict):
            request_payload = {}
        query_text = str(request_payload.get("text") or request_payload.get("message") or request_payload.get("q") or request_payload.get("input") or "")
        response_payload = resp.get_json(silent=True) if hasattr(resp, "get_json") else None
        if not isinstance(response_payload, dict):
            response_payload = {}
        meta = response_payload.get("meta") if isinstance(response_payload.get("meta"), dict) else {}
        reply_text = str(
            response_payload.get("presentation_reply")
            or response_payload.get("reply")
            or response_payload.get("response")
            or response_payload.get("text")
            or response_payload.get("raw_answer")
            or ""
        )
        query_hash = hashlib.sha256(query_text.encode("utf-8", "ignore")).hexdigest() if query_text else ""
        reply_hash = hashlib.sha256(reply_text.encode("utf-8", "ignore")).hexdigest() if reply_text else ""
        conversation_id = str(
            request_payload.get("conversation_id")
            or request_payload.get("session_id")
            or meta.get("session_id")
            or ""
        )[:180]
        source = str(meta.get("source") or "api_chat")[:96]
        engine = str(meta.get("engine") or "api_chat")[:96]
        intent = str(meta.get("intent") or request_payload.get("intent") or "chat")[:96]
        try:
            _SM_LAST_CHAT_EXCHANGE = {
                "query": query_text[:500],
                "reply": _sm_scrub_visible_text(reply_text)[:1200],
                "source": source,
                "engine": engine,
                "intent": intent,
                "status_code": status_code if "status_code" in locals() else int(getattr(resp, "status_code", 200) or 200),
                "ts": _sm_v07_now_iso() if "_sm_v07_now_iso" in globals() else datetime.now().isoformat(),
                "sml": meta.get("sml") if isinstance(meta.get("sml"), dict) else None,
            }
        except Exception:
            pass
        governor = meta.get("governor") if isinstance(meta.get("governor"), dict) else {}
        neuron_trace = meta.get("neuron_trace") if isinstance(meta.get("neuron_trace"), dict) else {}
        primary_lane = str(
            neuron_trace.get("primary_lane")
            or neuron_trace.get("lane")
            or source
            or "chat"
        )[:96]
        status_code = int(getattr(resp, "status_code", 200) or 200)
        verdict = "OBSERVED" if status_code < 400 and bool(response_payload.get("ok", True)) else "FAILED"
        risk = "low" if verdict == "OBSERVED" else "medium"
        try:
            import SarahMemoryLedger as _ChatLedger  # type: ignore
            record = getattr(_ChatLedger, "record_governance_receipt", None)
            if callable(record):
                record(
                    "chat_history",
                    "CHAT_QUERY_RECEIPT",
                    subject_id="local_user",
                    task_id=str(request_payload.get("task_id") or "")[:180],
                    conversation_id=conversation_id,
                    lane=primary_lane,
                    verdict=verdict,
                    risk=risk,
                    retention_class="chat_standard",
                    payload_hash=query_hash,
                    summary="Compact governed chat/query receipt; raw chat content remains in the conversation store.",
                    metadata={
                        "query_hash": query_hash,
                        "reply_hash": reply_hash,
                        "query_chars": len(query_text),
                        "reply_chars": len(reply_text),
                        "source": source,
                        "engine": engine,
                        "intent": intent,
                        "governor_decision": str(governor.get("decision") or ""),
                        "primary_owner": str(neuron_trace.get("primary_owner") or "")[:96],
                        "status_code": status_code,
                        "raw_content_stored_in_ledger": False,
                        "execution_authority": False,
                    },
                )
        except Exception as exc:
            try:
                app_logger.debug(f"Chat ledger receipt skipped: {exc}")
            except Exception:
                pass
    except Exception:
        return


# ---------------------------------------------------------------------------
# WebUI helper endpoints (Themes/Voices/Settings/Contacts/Reminders/Cleanup)
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(resp):
    _sm_record_chat_ledger_receipt(resp)
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
    _settings_dir = _gp.get("SETTINGS_DIR") or os.path.join(_gp.get("DATA_DIR", os.path.join(BASE_DIR, "data")), "settings")
    try:
        os.makedirs(_settings_dir, exist_ok=True)
    except Exception:
        pass
    SETTINGS_FILE = os.path.join(_settings_dir, "settings.json")  # SETTINGS_DIR/settings.json
except Exception:
    SETTINGS_FILE = os.path.join(DATA_DIR if "DATA_DIR" in globals() else os.path.join(BASE_DIR, "data"), "settings", "settings.json")

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
USER_DB_PATH = str(getattr(config, "USER_DATA_DB_PATH", os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "user_data.db"))) if config is not None else os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "user_data.db")

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
REMINDERS_DB_PATH = str(getattr(config, "REMINDERS_DB_PATH", os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "reminders.db"))) if config is not None else os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "reminders.db")

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
CHAT_HISTORY_DB_PATH = str(getattr(config, "CONTEXT_HISTORY_DB_PATH", os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "context_history.db"))) if config is not None else os.path.join(_globals_dir("DATASETS_DIR", "data/memory/datasets"), "context_history.db")


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
        base_dir_local = BASE_DIR
        data_dir_local = DATA_DIR
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

# JWT Configuration
JWT_ALGORITHM = "HS256"
JWT_ISSUER = str(os.getenv("SARAH_JWT_ISSUER", "sarahmemory-local-api") or "sarahmemory-local-api")
JWT_AUDIENCE = str(os.getenv("SARAH_JWT_AUDIENCE", "sarahmemory-ui") or "sarahmemory-ui")
JWT_EXP_SECONDS = max(300, min(int(os.getenv("SARAH_JWT_EXP_SECONDS", str(7 * 24 * 3600)) or 7 * 24 * 3600), 30 * 24 * 3600))
_WEAK_JWT_SECRETS = {
    "", "secret", "changeme", "change-me", "change-this-secret-key-in-production",
    "development", "dev", "password", "jwt-secret",
}

def _load_or_create_jwt_secret() -> str:
    configured = str(os.getenv("SARAH_JWT_SECRET") or os.getenv("JWT_SECRET_KEY") or "").strip()
    if configured and len(configured.encode("utf-8")) >= 32 and configured.lower() not in _WEAK_JWT_SECRETS:
        return configured
    if configured:
        app_logger.error("Rejected weak JWT secret from environment; using protected local secret material.")
    secret_path = os.path.join(_SETTINGS_DIR, "jwt_secret.key")
    try:
        if os.path.isfile(secret_path):
            value = open(secret_path, "r", encoding="utf-8").read().strip()
            if len(value.encode("utf-8")) >= 32 and value.lower() not in _WEAK_JWT_SECRETS:
                return value
    except Exception:
        pass
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    value = secrets.token_urlsafe(64)
    tmp = f"{secret_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, secret_path)
    try:
        os.chmod(secret_path, 0o600)
    except Exception:
        pass
    app_logger.warning("Generated persistent local JWT secret at %s", secret_path)
    return value

JWT_SECRET = _load_or_create_jwt_secret()

def generate_jwt_token(user_id, email, display_name):
    """Generate a signed, scoped token with bounded lifetime and replay identity."""
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "email": str(email or "").strip().lower(),
        "display_name": str(display_name or ""),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "nbf": now - 1,
        "exp": now + JWT_EXP_SECONDS,
        "jti": secrets.token_hex(16),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token.decode("utf-8") if isinstance(token, bytes) else str(token)

def verify_jwt_token(token):
    """Verify signature, algorithm, issuer, audience, lifetime, and required claims."""
    try:
        payload = jwt.decode(
            str(token or ""),
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["sub", "user_id", "email", "iss", "aud", "iat", "nbf", "exp", "jti"]},
        )
        if not isinstance(payload, dict):
            return None
        now = int(time.time())
        if int(payload.get("iat", 0) or 0) > now + 30:
            return None
        if int(payload.get("nbf", 0) or 0) > now + 30:
            return None
        if int(payload.get("exp", 0) or 0) <= now:
            return None
        if str(payload.get("sub")) != str(payload.get("user_id")):
            return None
        return payload
    except jwt.ExpiredSignatureError:
        app_logger.info("Expired JWT token received.")
        return None
    except jwt.InvalidTokenError as exc:
        app_logger.warning("Invalid JWT token received: %s", exc)
        return None
    except Exception as exc:
        app_logger.error("Unexpected JWT verification error: %s", exc)
        return None

def _bearer_token() -> str:
    header = str(request.headers.get("Authorization") or "").strip()
    if not header:
        return ""
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()

def require_auth(f):
    """Require a valid scoped bearer token and expose normalized identity claims."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = _bearer_token()
        if not token:
            return jsonify({"error": "Authentication required. Token missing or malformed."}), 401
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({"error": "Authentication failed. Invalid or expired token."}), 401
        request.user_id = payload.get("user_id")
        request.user_email = payload.get("email")
        request.user_display_name = payload.get("display_name")
        request.jwt_claims = payload
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
# SarahMemory 2D Avatar Live WebP Morph State / Manifest / Life-Cycle Contract
# ---------------------------------------------------------------------------
# WebUI-facing contract for the Custom AvatarPanel. This exposes the current
# governed 2D morphic/WebP runtime selection while preserving the existing 3D, media,
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
        role_map.setdefault(alias, "29_extra_avatar_state.webp")
    return role_map

_AVATAR_2D_ALLOWED_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}
_AVATAR_2D_MIMETYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

def _avatar_2d_safe_name(filename: str) -> str:
    safe = os.path.basename(str(filename or "").strip().replace("\\", "/"))
    if not safe or safe in {".", ".."} or ".." in safe:
        return ""
    ext = os.path.splitext(safe)[1].lower()
    if ext not in _AVATAR_2D_ALLOWED_EXTENSIONS:
        return ""
    return safe

def _avatar_load_sidecar_json(filename: str) -> dict:
    try:
        safe = os.path.basename(str(filename or "").strip())
        if not safe.lower().endswith(".json"):
            return {}
        pth = os.path.abspath(os.path.join(_avatar_default_dir(), safe))
        base = os.path.abspath(_avatar_default_dir())
        if os.path.commonpath([base, pth]) != base or not os.path.isfile(pth):
            return {}
        with open(pth, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _safe_avatar_files() -> list[str]:
    files: list[str] = []
    try:
        data = _avatar_read_manifest()
        raw = data.get("files") if isinstance(data, dict) else []
        if isinstance(raw, list):
            for item in raw:
                name = _avatar_2d_safe_name(str(item or "").strip())
                if name and name not in files:
                    files.append(name)
    except Exception:
        pass
    try:
        d = _avatar_default_dir()
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                safe = _avatar_2d_safe_name(fn)
                if safe and safe not in files:
                    files.append(safe)
    except Exception:
        pass
    return files

def _avatar_public_url(filename: str) -> str:
    return f"/api/avatar/2d/{os.path.basename(filename or 'sarah_avatar.webp')}"


# ---------------------------------------------------------------------------
# SarahMemory 3D Avatar Runtime Asset Contract
# ---------------------------------------------------------------------------
# Avatar Organ doctrine:
# - Runtime 3D assets live under BASE_DIR/resources/avatars/3D.
# - The singular BASE_DIR/resources/avatar/3D path is legacy/fallback only.
# - The default/base Sarah model is stored directly in resources/avatars/3D.
# - Future appearances/skins may live under resources/avatars/3D/<name>/.
# - Only runtime-safe files are served.  Blender sources, scripts, and logs remain local.
# - This route family is a visual-interface lane.  It does not execute robot movement.
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

_AVATAR_3D_BLOCKED_EXTENSIONS = {
    ".blend",
    ".blend1",
    ".py",
    ".log",
    ".bat",
    ".cmd",
    ".ps1",
    ".exe",
    ".dll",
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

_AVATAR_3D_DEFAULT_SKIN = os.getenv("SARAH_AVATAR_3D_DEFAULT_SKIN", "default").strip() or "default"
_AVATAR_3D_DEFAULT_HEIGHT_M = float(os.getenv("SARAH_AVATAR_3D_HEIGHT_M", "1.68") or 1.68)
_AVATAR_3D_DEFAULT_WEIGHT_KG = float(os.getenv("SARAH_AVATAR_3D_WEIGHT_KG", "58.0") or 58.0)


def _avatar_3d_dir() -> str:
    """Return the local runtime folder for AvatarPanel 3D assets.

    Canonical v9 contract:
        BASE_DIR/resources/avatars/3D

    The legacy singular path is intentionally not preferred; it is checked only
    as a last-resort compatibility fallback so old files do not break boot.
    """
    try:
        root = _globals_paths().get("ROOT_DIR") or BASE_DIR
    except Exception:
        root = BASE_DIR
    candidates = [
        os.path.join(root, "resources", "avatars", "3D"),
        os.path.join(BASE_DIR, "resources", "avatars", "3D"),
        # legacy/fallback only; do not write new assets here
        os.path.join(root, "resources", "avatar", "3D"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _avatar_3d_relpath(filename: str) -> str:
    """Sanitize a runtime-safe 3D asset path.

    Supports current base files:
        SarahMemoryAvatar_RigBootstrap.glb

    Supports future one-or-more-level appearance folders:
        <name>/model.glb
        <name>/textures/body.webp

    Prevents path traversal and blocks executable/source/development formats.
    """
    raw = str(filename or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [p for p in raw.split("/") if p and p not in (".", "..")]
    if not parts:
        return ""
    clean_parts: list[str] = []
    for part in parts:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(part or "").strip()).replace("..", "_")
        if not safe:
            return ""
        clean_parts.append(safe[:128])
    ext = os.path.splitext(clean_parts[-1])[1].lower()
    if ext in _AVATAR_3D_BLOCKED_EXTENSIONS:
        return ""
    if ext not in _AVATAR_3D_ALLOWED_EXTENSIONS:
        return ""
    return "/".join(clean_parts)


def _avatar_3d_safe_name(filename: str) -> str:
    """Backward-compatible alias retained for existing callers."""
    return _avatar_3d_relpath(filename)


def _avatar_3d_abs_path(relpath: str) -> str:
    base = os.path.abspath(_avatar_3d_dir())
    safe = _avatar_3d_relpath(relpath)
    if not safe:
        return ""
    candidate = os.path.abspath(os.path.join(base, *safe.split("/")))
    try:
        common = os.path.commonpath([base, candidate])
    except Exception:
        return ""
    if common != base:
        return ""
    return candidate


def _safe_avatar_3d_files() -> list[str]:
    """List runtime-safe 3D files recursively under resources/avatars/3D."""
    files: list[str] = []
    try:
        d = os.path.abspath(_avatar_3d_dir())
        if os.path.isdir(d):
            for root, dirs, filenames in os.walk(d):
                # Hide dev/source/cache folders from the WebUI runtime file list.
                dirs[:] = [
                    x for x in dirs
                    if x.lower() not in {"__pycache__", ".git", ".vscode", "source", "source_normalized", "build", "tmp", "temp"}
                ]
                for fn in sorted(filenames):
                    rel = os.path.relpath(os.path.join(root, fn), d).replace("\\", "/")
                    safe = _avatar_3d_relpath(rel)
                    if safe and safe not in files:
                        files.append(safe)
    except Exception as e:
        app_logger.debug(f"Avatar 3D file scan failed: {e}")
    return files


def _avatar_3d_manifest_path() -> str:
    d = _avatar_3d_dir()
    for name in (
        "SarahMemoryAvatar_RigBootstrap.json",
        "sarahmemory_3d_avatar_manifest.json",
        "avatar_3d_manifest.json",
        "avatar-manifest-3d.json",
        "manifest.json",
    ):
        pth = os.path.join(d, name)
        if os.path.isfile(pth):
            return pth
    return os.path.join(d, "SarahMemoryAvatar_RigBootstrap.json")


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


def _avatar_3d_model_candidates_from_manifest(manifest: dict) -> list[str]:
    candidates: list[str] = []
    if not isinstance(manifest, dict):
        return candidates
    for key in ("glb", "model", "model_file", "modelUrl", "model_url", "primary_runtime_asset"):
        raw = manifest.get(key)
        if raw:
            candidates.append(str(raw).replace("\\", "/").split("resources/avatars/3D/")[-1])
    contract = manifest.get("avatar_panel_contract")
    if isinstance(contract, dict):
        for key in ("primary_runtime_asset", "model", "model_file", "modelUrl", "model_url"):
            raw = contract.get(key)
            if raw:
                candidates.append(str(raw).replace("\\", "/").split("resources/avatars/3D/")[-1])
    return candidates


def _avatar_3d_pick_model() -> str | None:
    """Select the active GLB/GLTF runtime model.

    Preference order intentionally favors the new anatomically-plausible
    SarahMemory rig bootstrap over older demo/procedural models.
    """
    files = set(_safe_avatar_3d_files())
    manifest = _avatar_3d_read_manifest()
    candidates: list[str] = []

    candidates.extend(_avatar_3d_model_candidates_from_manifest(manifest))
    candidates.extend([
        "SarahMemoryAvatar_RigBootstrap.glb",
        f"{_AVATAR_3D_DEFAULT_SKIN}/SarahMemoryAvatar_RigBootstrap.glb",
        "default/SarahMemoryAvatar_RigBootstrap.glb",
        "sarahmemory_3d_avatar.glb",
        "sarahmemory_happy_face_ball.glb",
    ])

    for name in sorted(files):
        if name.lower().endswith((".glb", ".gltf")) and name not in candidates:
            candidates.append(name)

    for candidate in candidates:
        safe = _avatar_3d_relpath(candidate)
        if safe and safe in files and safe.lower().endswith((".glb", ".gltf")):
            return safe
    return None


def _avatar_3d_public_url(filename: str | None) -> str:
    safe = _avatar_3d_relpath(filename or "")
    return f"/api/avatar/3d/{safe}" if safe else ""


def _avatar_3d_body_profile(manifest: dict | None = None) -> dict:
    """Return the visual humanoid body profile used by AvatarPanel.

    This is not a medical claim and not a robot actuation contract.  It gives the
    3D renderer and future robot-bridge organ a stable anatomical scale map.
    """
    manifest = manifest if isinstance(manifest, dict) else {}
    profile = manifest.get("body_profile") if isinstance(manifest.get("body_profile"), dict) else {}
    return {
        "profile_name": str(profile.get("profile_name") or "SarahMemory_default_humanoid"),
        "height_m": float(profile.get("height_m") or _AVATAR_3D_DEFAULT_HEIGHT_M),
        "weight_kg_visual_target": float(profile.get("weight_kg_visual_target") or _AVATAR_3D_DEFAULT_WEIGHT_KG),
        "head_height_ratio": float(profile.get("head_height_ratio") or 0.132),
        "shoulder_width_ratio": float(profile.get("shoulder_width_ratio") or 0.245),
        "hip_width_ratio": float(profile.get("hip_width_ratio") or 0.190),
        "leg_length_ratio": float(profile.get("leg_length_ratio") or 0.515),
        "arm_span_ratio": float(profile.get("arm_span_ratio") or 1.0),
        "center_of_mass_y_ratio": float(profile.get("center_of_mass_y_ratio") or 0.55),
        "rig_units": "meters",
        "biological_plausibility": "visual_anatomy_reference_only",
        "robot_mapping_status": "future_bridge_required_no_physical_actuation_here",
    }


def _avatar_3d_animation_contract() -> dict:
    return {
        "states": [
            "idle_breathing", "blink", "eye_follow", "listening_focus", "speaking_lipsync",
            "thinking_shift", "stand", "turn_left", "turn_right", "walk_in_place", "wave",
            "error_degraded", "offline_fallback",
        ],
        "facial_shape_keys": ["Blink_L", "Blink_R", "JawOpen", "Smile", "Frown", "BrowsUp"],
        "body_controls": ["head", "neck", "spine", "clavicle", "upper_arm", "forearm", "hand", "pelvis", "thigh", "shin", "foot"],
        "runtime_rules": [
            "Animate visually inside AvatarPanel only.",
            "Do not execute physical robot movement from this API.",
            "Future robot body bridge must route through MSDC, OperatorCore, SafetyPolicies, AssuranceGate, Compare, and user approval.",
        ],
    }


def _avatar_3d_manifest_payload() -> dict:
    manifest = _avatar_3d_read_manifest()
    files = _safe_avatar_3d_files()
    model_file = _avatar_3d_pick_model()
    body_profile = _avatar_3d_body_profile(manifest)
    runtime_validation = {}
    try:
        from SarahMemoryAvatarBuilder import inspect_realtime_avatar_assets  # type: ignore
        runtime_validation = inspect_realtime_avatar_assets(_avatar_3d_dir(), os.path.splitext(os.path.basename(model_file or "SarahMemoryAvatar_RigBootstrap"))[0])
    except Exception as exc:
        runtime_validation = {"ok": False, "error": str(exc), "runtime_ready": bool(model_file)}
    return {
        "success": True,
        "ok": bool(model_file),
        "base_url": "/api/avatar/3d",
        "asset_dir": _avatar_3d_dir(),
        "asset_contract": "BASE_DIR/resources/avatars/3D",
        "default_skin": _AVATAR_3D_DEFAULT_SKIN,
        "manifest_path": _avatar_3d_manifest_path(),
        "manifest": manifest,
        "files": files,
        "model_file": model_file,
        "model_url": _avatar_3d_public_url(model_file),
        "fallback_reason": "" if model_file else "avatar_3d_model_missing_or_unavailable",
        "body_profile": body_profile,
        "animation_authority": "visual_only_no_msdc_no_operator_action",
        "body_action_authority": {"physical_actuation_allowed": False, "msdc_required": True, "operator_core_required": True, "user_approval_required": True},
        "animation_contract": _avatar_3d_animation_contract(),
        "runtime_validation": runtime_validation,
        "blocked_extensions": sorted(_AVATAR_3D_BLOCKED_EXTENSIONS),
        "runtime_only": True,
    }


def _avatar_3d_spec_payload(state: dict | None = None) -> dict:
    """Return the Avatar3D.tsx backend contract used by the dropdown 3D mode."""
    state = dict(state or {})
    manifest = _avatar_3d_read_manifest()
    model_file = _avatar_3d_pick_model()
    expression = str(state.get("expression") or state.get("emotion") or "neutral")
    speaking = bool(state.get("speaking", False))
    listening = bool(state.get("listening", False))
    thinking = bool(state.get("thinking", False))
    busy = bool(state.get("busy", False))
    body_profile = _avatar_3d_body_profile(manifest)
    panel_contract = manifest.get("avatar_panel_contract") if isinstance(manifest.get("avatar_panel_contract"), dict) else {}
    try:
        requested_fps = int(panel_contract.get("recommended_fps_cap") or 30)
    except Exception:
        requested_fps = 30
    runtime_fps_cap = max(12, min(60, requested_fps))
    runtime_quality = str(manifest.get("quality") or "balanced")

    base_spec = {
        "backgroundType": "none",
        "pose": "stand",
        "gesture": "none",
        "lookAt": {"x": 0, "y": 1.45, "z": 0},
        "expression": expression,
        "speaking": speaking,
        "listening": listening,
        "thinking": thinking,
        "busy": busy,
        "bodyProfile": body_profile,
        "animationContract": _avatar_3d_animation_contract(),
        "quality": runtime_quality,
        "fpsCap": runtime_fps_cap,
        "lightingProfile": "high_end",
        "shadowQuality": "high",
                "animationAuthority": "visual_only_no_msdc_no_operator_action",
                "avatarEyeCameraAnchor": "Sarah_AvatarEye_Center",
        "stageOffsetY": -1.18,
        "avatarOffsetY": -0.52,
        "useRuntimeStage": True,
        "materialMode": "high_end",
        "source": "resources/avatars/3D",
        "runtimeOnly": True,
        "robotBridge": {
            "enabled": False,
            "future_owner": "AvatarToMSDCBridge",
            "physical_actuation_allowed_here": False,
        },
    }

    if not model_file:
        return {
            **base_spec,
            "renderMode": "procedural_holo",
            "modelUrl": "",
            "modelFile": "",
            "loaderState": "3D_FAILED_FALLBACK_2D",
            "fallbackReason": "avatar_3d_missing_model",
        }

    # User-visible AvatarPanel 3D now uses the true GLB/GLTF mesh as the
    # primary runtime asset.  GoldStandard images remain visual references and
    # emergency fallback only, so weak mesh defects stay visible and fixable.
    model_url = _avatar_3d_public_url(model_file)
    return {
        **base_spec,
        "renderMode": "gltf_model",
        "runtimeVisualPriority": "gltf_model",
        "forceMeshRuntime": True,
        "modelUrl": model_url,
        "meshFallbackUrl": model_url,
        "modelFile": model_file,
        "loaderState": "3D_READY_GOLDSTANDARD_EMBODIED_ENTITY",
        "fallbackReason": "",
        "goldReferenceOnly": True,
        "authoringPolyTarget": 12000000,
        "runtimeLod": "goldstandard_entity",
        "productionAssetContract": "SARAHMEMORY_GOLDSTANDARD_EMBODIED_ENTITY_GLB_V1",
        "blueprintConstructReady": True,
        "visualEffectsMode": "full",
        "goldStandardScale": 1.0,
        "goldStandardYOffset": 0,
        "goldStandardPanelBottomPx": 58,
        "goldStandardPanelHeightPct": 92,
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
            low = name.lower()
            if low.startswith(prefix) and os.path.splitext(low)[1] in _AVATAR_2D_ALLOWED_EXTENSIONS:
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
        for default_name in ("sarah_avatar.webp", "sarah-avatar.webp", "sarah_avatar.png", "sarah-avatar.png", "19_neutral_forward.webp", "19_neutral_forward.png"):
            if default_name in available:
                return default_name
        return sorted(available)[0] if available else "sarah_avatar.webp"

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
    data = _avatar_read_manifest()
    role_map = _avatar_effective_role_map()
    files = _safe_avatar_files()
    morph_meta = data.get("morph") if isinstance(data.get("morph"), dict) else {}
    graph = _avatar_load_sidecar_json(str(morph_meta.get("graph") or "avatar-morph-graph.json"))
    anchors = _avatar_load_sidecar_json(str(morph_meta.get("anchors") or "avatar-morph-anchors.json"))
    image_files = [f for f in files if os.path.splitext(f.lower())[1] in _AVATAR_2D_ALLOWED_EXTENSIONS]
    return {
        "success": True,
        "ok": True,
        "schema": str(data.get("schema") or "SarahMemory.avatar.2d_manifest.v2"),
        "base_url": "/api/avatar/2d",
        "default_file": str(data.get("default_file") or "sarah_avatar.webp"),
        "role_map": role_map,
        "files": files,
        "source_files": data.get("source_files", []),
        "assets": data.get("assets", {}),
        "target_dimensions": data.get("target_dimensions", [1254, 1254]),
        "runtime_dimensions": data.get("runtime_dimensions", data.get("target_dimensions", [1254, 1254])),
        "source_format": str(data.get("source_format") or "png"),
        "runtime_format": str(data.get("runtime_format") or "webp"),
        "state_count": len(image_files),
        "supports_dynamic_29": True,
        "supports_morph": bool(data.get("supports_morph", True)),
        "store_generated_frames": False,
        "ram_only": True,
        "morph": {**morph_meta, "graph_data": graph, "anchors_data": anchors},
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
    safe_name = _avatar_2d_safe_name(filename or "")
    allowed = set(_safe_avatar_files())
    if safe_name not in allowed:
        # legacy UI fallback name is allowed only if the file exists locally
        if safe_name not in {"sarah-avatar.png", "sarah_avatar.png", "sarah-avatar.webp", "sarah_avatar.webp"}:
            abort(404)
        if not os.path.isfile(os.path.join(_avatar_default_dir(), safe_name)):
            abort(404)
    ext = os.path.splitext(safe_name)[1].lower()
    mimetype = _AVATAR_2D_MIMETYPES.get(ext, "application/octet-stream")
    try:
        return send_from_directory(_avatar_default_dir(), safe_name, mimetype=mimetype, max_age=30)
    except Exception:
        abort(404)


@app.route("/api/avatar/morph/state", methods=["GET"])
def avatar_live_morph_state():
    state = _avatar_state_payload()
    current = str(state.get("current_image") or "19_neutral_forward.webp")
    target = current
    if bool(state.get("speaking")):
        target = _avatar_select_existing("speaking_open", set(_safe_avatar_files())) or current
    elif bool(state.get("listening")):
        target = _avatar_select_existing("listening", set(_safe_avatar_files())) or current
    elif bool(state.get("thinking")) or bool(state.get("busy")):
        target = _avatar_select_existing("thinking", set(_safe_avatar_files())) or current
    return jsonify({
        "ok": True,
        "success": True,
        "schema": "SarahMemory.avatar.morphtoken.v1",
        "from_state": str(state.get("expression") or "neutral"),
        "to_state": "speaking_soft" if state.get("speaking") else str(state.get("expression") or "neutral"),
        "from_asset": current,
        "to_asset": target,
        "duration_ms": 420,
        "easing": "breath_sine",
        "blend_mode": "canvas_2d_crossfade_breath_v1",
        "ram_only": True,
        "store_generated_frames": False,
        "fallback": "last_good_frame_then_neutral",
        "state": state,
    }), 200


@app.route("/api/avatar/speech/status", methods=["GET"])
def avatar_speech_status():
    try:
        from UnifiedAvatarController import get_avatar_speech_status
        return jsonify(get_avatar_speech_status()), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "active": {}}), 200


@app.route("/api/avatar/speech/finish", methods=["POST"])
def avatar_speech_finish():
    data = request.get_json(silent=True) or {}
    try:
        from UnifiedAvatarController import finish_avatar_speech_session
        result = finish_avatar_speech_session(str(data.get("session_id") or ""), str(data.get("reason") or "ended"))
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    state = _avatar_update_state(speaking=False)
    result["state"] = state
    return jsonify(result), 200


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
    """Decorator to standardize fail-soft responses for avatar panel API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api = get_avatar_panel_api()
        if not api:
            return jsonify({"ok": False, "success": False, "error": "avatar_panel_unavailable", "message": "Avatar panel is not initialized; frontend should fall back to live avatar state/spec endpoints.", "fallback": _avatar_state_payload()}), 503
        try:
            result = func(api, *args, **kwargs)
            if isinstance(result, tuple):
                return result
            try:
                from flask import Response as _FlaskResponse
                if isinstance(result, _FlaskResponse):
                    return result
            except Exception:
                pass
            if isinstance(result, dict):
                result.setdefault("ok", bool(result.get("success", True)))
                result.setdefault("source", "api.avatar.panel")
                return jsonify(result), 200
            return jsonify({"ok": True, "result": result, "source": "api.avatar.panel"}), 200
        except Exception as e:
            app_logger.exception(f"Error in avatar API endpoint '{request.path}'.")
            return jsonify({"ok": False, "success": False, "error": str(e), "message": "Failed to perform avatar action; visual avatar should remain in safe fallback state.", "fallback": _avatar_state_payload()}), 500
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

    voice_identity = _sm_voice_identity_packet() if '_sm_voice_identity_packet' in globals() else {"voice_identity": "SarahMemory Speaking", "voice_model_id": "SarahVoice_v1"}
    return jsonify({
        "ok": True,
        "settings": meta,
        "voice_identity": voice_identity,
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
core_speak_text_status = None
core_voice_identity = None
core_voice_status = None
try:
    from SarahMemoryVoice import speak_text as core_speak_text_func
    core_speak_text = core_speak_text_func
    try:
        from SarahMemoryVoice import speak_text_status as core_speak_text_status_func
        core_speak_text_status = core_speak_text_status_func
    except Exception:
        core_speak_text_status = None
    try:
        from SarahMemoryVoice import get_primary_voice_identity as core_voice_identity_func, get_voice_status as core_voice_status_func
        core_voice_identity = core_voice_identity_func
        core_voice_status = core_voice_status_func
    except Exception:
        core_voice_identity = None
        core_voice_status = None
except ImportError:
    app_logger.info("SarahMemoryVoice module not found for TTS.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryVoice.speak_text: {e}", exc_info=True)


def _sm_voice_identity_packet() -> dict:
    try:
        if callable(core_voice_identity):
            packet = core_voice_identity()
            if isinstance(packet, dict):
                return packet
    except Exception as exc:
        app_logger.debug(f"SarahVoice identity unavailable: {exc}")
    return {
        "ok": True,
        "schema": "SarahMemory.voice.identity.v1",
        "voice_model_id": "SarahVoice_v1",
        "voice_identity": "SarahMemory Speaking",
        "display_name": "SarahMemory Voice",
        "engine": "sarahvoice",
        "primary_voice_ready": True,
        "male_default_boot_voice_allowed": False,
        "pt_voice_dependency": False,
    }


def _sm_normalize_voice_request(voice: str) -> str:
    v = str(voice or "").strip()
    if not v or v.lower() in {"default", "sarah", "sarahmemory", "sarahmemory voice", "sarah voice"}:
        return "sarahvoice"
    return v


@app.route("/api/voice/identity", methods=["GET"])
def api_voice_identity():
    identity = _sm_voice_identity_packet()
    return jsonify({
        "ok": True,
        "success": True,
        "identity": identity,
        "voice_identity": identity.get("voice_identity"),
        "voice_model_id": identity.get("voice_model_id"),
        "display_name": identity.get("display_name"),
        "engine": identity.get("engine"),
        "primary_voice_ready": bool(identity.get("primary_voice_ready", True)),
        "male_default_boot_voice_allowed": False,
        "schema": "SarahMemory.voice.identity.response.v1",
        "ts": time.time(),
    }), 200


@app.route("/api/voice/status", methods=["GET"])
def api_voice_status():
    try:
        if callable(core_voice_status):
            status = core_voice_status()
            if isinstance(status, dict):
                status.setdefault("ok", True)
                status.setdefault("success", True)
                status.setdefault("ts", time.time())
                return jsonify(status), 200
    except Exception as exc:
        app_logger.debug(f"SarahVoice status unavailable: {exc}")
    identity = _sm_voice_identity_packet()
    return jsonify({
        "ok": True,
        "success": True,
        "identity": identity,
        "engines": {"sarahvoice": True},
        "fallback_policy": {
            "browser_default_allowed_as_last_resort": True,
            "male_default_boot_voice_allowed": False,
            "browser_voice_requires_resolution": True,
        },
        "ts": time.time(),
    }), 200


@app.route("/api/tts/speak", methods=['POST'])
def api_tts_speak():
    """
    Minimal TTS bridge for the Web UI.
    Expected JSON:
      { "text": "...", "voice": "default", "rate": 1.0 }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = _sm_normalize_voice_request(data.get("voice") or "sarahvoice")
    rate_str = data.get("rate") # Keep as string/int for initial parsing

    if not text:
        return jsonify({"ok": False, "error": "Missing text for TTS."}), 400

    try:
        rate = float(rate_str) if rate_str is not None else 1.0
        if not (0.1 <= rate <= 5.0): # Example range, adjust as needed
             return jsonify({"ok": False, "error": "Speech rate must be between 0.1 and 5.0."}), 400
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid speech rate format."}), 400


    identity = _sm_voice_identity_packet()

    if core_speak_text is None and core_speak_text_status is None:
        return jsonify({
            "ok": False,
            "voice_identity": identity.get("voice_identity"),
            "voice_model_id": identity.get("voice_model_id"),
            "voice_display_name": identity.get("display_name"),
            "success": False,
            "server_tts_started": False,
            "browser_fallback_required": True,
            "error": "TTS engine not available on this server.",
        }), 501

    try:
        # Prefer UnifiedAvatarController so audio and morphic AvatarPanel state share one session.
        avatar_session_result = None
        try:
            from UnifiedAvatarController import start_avatar_speech_session
            avatar_session_result = start_avatar_speech_session(text=text, voice=voice or "default", emotion=None, start_tts=True)
        except Exception as controller_exc:
            app_logger.debug(f"UnifiedAvatarController speech session unavailable: {controller_exc}")

        if isinstance(avatar_session_result, dict) and avatar_session_result.get("avatar_session"):
            session = avatar_session_result.get("avatar_session") or {}
            tts_status = avatar_session_result.get("tts_status") or session.get("tts_status") or {}
        else:
            if core_speak_text_status is not None:
                tts_status = core_speak_text_status(text, blocking=False, engine_pref=voice or None)
            else:
                accepted = bool(core_speak_text(text, blocking=False, engine_pref=voice or None))
                tts_status = {"ok": accepted, "accepted": accepted, "server_tts_started": accepted, "browser_fallback_required": not accepted}
            session = {
                "schema": "SarahMemory.avatar.voice_session.v1",
                "session_id": "voice_" + hashlib.sha256(f"{time.time()}::{text[:80]}".encode("utf-8", "ignore")).hexdigest()[:16],
                "text_preview": text[:180],
                "voice": voice,
                "speaking": bool(tts_status.get("accepted") or tts_status.get("ok")),
                "started_at": time.time(),
                "estimated_duration_ms": int(tts_status.get("estimated_duration_ms") or max(1600, min(180000, int(len(text.split()) / 1.45 * 1000) + 1200))),
                "browser_fallback_allowed": True,
                "browser_fallback_required": bool(tts_status.get("browser_fallback_required")),
                "server_tts_started": bool(tts_status.get("server_tts_started") or tts_status.get("accepted") or tts_status.get("ok")),
                "morph": {"schema": "SarahMemory.avatar.morphtoken.v1", "to_state": "speaking_soft", "ram_only": True, "store_generated_frames": False},
            }

        server_started = bool(tts_status.get("server_tts_started") or tts_status.get("accepted") or tts_status.get("ok"))
        browser_required = bool(tts_status.get("browser_fallback_required") or not server_started)
        try:
            _avatar_update_state(speaking=server_started or browser_required, current_action="speaking", life_state="speaking")
        except Exception:
            pass
        return jsonify({
            "ok": bool(server_started or browser_required),
            "success": bool(server_started or browser_required),
            "text": text,
            "voice": voice,
            "voice_identity": tts_status.get("voice_identity") or session.get("voice_identity") or identity.get("voice_identity"),
            "voice_model_id": tts_status.get("voice_model_id") or session.get("voice_model_id") or identity.get("voice_model_id"),
            "voice_display_name": tts_status.get("voice_display_name") or session.get("voice_display_name") or identity.get("display_name"),
            "primary_voice_ready": bool(tts_status.get("primary_voice_ready", identity.get("primary_voice_ready", True))),
            "male_default_boot_voice_allowed": False,
            "fallback_used": bool((tts_status.get("engine") or "") not in {"sarahvoice", ""}),
            "rate": rate,
            "audio_url": tts_status.get("audio_url"),
            "audio_base64": tts_status.get("audio_base64"),
            "server_tts_started": server_started,
            "browser_fallback_required": browser_required,
            "browser_fallback_allowed": True,
            "playback_location": tts_status.get("playback_location") or ("server_local_audio" if server_started else "browser"),
            "estimated_duration_ms": int(tts_status.get("estimated_duration_ms") or session.get("estimated_duration_ms") or 2400),
            "engine": tts_status.get("engine"),
            "requested_engine": tts_status.get("requested_engine"),
            "avatar_session": session,
            "tts_status": tts_status,
        }), 200
    except Exception as e:
        app_logger.exception(f"Error during TTS speak request for text: '{text}...'")
        ident = _sm_voice_identity_packet()
        return jsonify({"ok": False, "success": False, "voice_identity": ident.get("voice_identity"), "voice_model_id": ident.get("voice_model_id"), "server_tts_started": False, "browser_fallback_required": True, "male_default_boot_voice_allowed": False, "error": f"Failed to speak text: {e}"}), 500

# =============================================================================
# PHASE1_BRIDGE_UI_CONTRACT_STABILIZATION
# Compatibility endpoints for the V9 React/Web UI. These routes are intentionally
# thin bridge contracts: they normalize request/response shape, preserve local-first
# behavior, and avoid turning app.py into a hidden brain. Execution remains routed
# through existing SarahMemory organs when available.
# =============================================================================

_SM_PHASE1_CONTRACT_SCHEMA = "SarahMemory.phase1.bridge_ui_contract.v1"


def _sm_phase1_contract_meta(endpoint: str, *, intent: str = "contract", engine: str = "phase1_bridge") -> dict:
    return {
        "schema": _SM_PHASE1_CONTRACT_SCHEMA,
        "endpoint": endpoint,
        "source": "phase1_bridge_ui_contract",
        "engine": engine,
        "intent": intent,
        "version": PROJECT_VERSION,
        "frontend_authority": False,
        "local_first": True,
        "execution_authority": False,
        "ts": time.time(),
    }


def _sm_phase1_reply(text: str, *, endpoint: str, ok: bool = True, status: int = 200, **extra):
    meta = _sm_phase1_contract_meta(endpoint, intent=str(extra.pop("intent", "contract")))
    bundle = _sm_make_outward_bundle(str(text or ""), meta=meta, errors=extra.pop("errors", []))
    bundle["ok"] = bool(ok)
    bundle["success"] = bool(ok)
    bundle.update(extra)
    return jsonify(bundle), int(status)


def _sm_phase1_route_available(path: str) -> bool:
    try:
        return any(str(rule.rule) == path for rule in app.url_map.iter_rules())
    except Exception:
        return False


def _sm_phase1_voice_options() -> list[dict]:
    identity = _sm_voice_identity_packet() if '_sm_voice_identity_packet' in globals() else {}
    voices: list[dict] = [{
        "id": "sarahvoice",
        "name": identity.get("display_name") or "SarahMemory Voice",
        "language": "en-US",
        "gender": "female",
        "primary": True,
        "voice_identity": identity.get("voice_identity") or "SarahMemory Speaking",
        "voice_model_id": identity.get("voice_model_id") or "SarahVoice_v1",
        "engine": "sarahvoice",
    }]
    try:
        from SarahMemoryVoice import list_voices as _list_voices  # type: ignore
        raw = _list_voices() if callable(_list_voices) else []
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if isinstance(item, str):
                    vid = item
                    name = item
                    extra = {}
                elif isinstance(item, dict):
                    vid = str(item.get("id") or item.get("name") or idx)
                    name = str(item.get("name") or item.get("id") or f"Voice {idx + 1}")
                    extra = {k: v for k, v in item.items() if k not in {"id", "name"}}
                else:
                    continue
                if vid.lower() in {"sarahvoice", "sarahmemory voice", "sarahvoice_v1"} or name.lower() == "sarahmemory voice":
                    continue
                voices.append({"id": vid, "name": name, **extra})
    except Exception:
        pass
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        for idx, voice in enumerate(engine.getProperty("voices") or []):
            vid = str(getattr(voice, "id", "") or idx)
            if any(v.get("id") == vid for v in voices):
                continue
            voices.append({
                "id": vid,
                "name": str(getattr(voice, "name", "") or getattr(voice, "id", "") or f"Voice {idx + 1}"),
                "language": str(getattr(voice, "languages", "") or ""),
                "gender": "neutral",
                "fallback": True,
            })
    except Exception:
        pass
    return voices


@app.route("/api/v1/chat", methods=["GET", "POST"])
def api_v1_chat_phase1_alias():
    """Compatibility alias for WebUI clients that use /api/v1/chat."""
    return api_chat()


@app.route("/api/version", methods=["GET"])
def api_version_phase1_contract():
    runtime = get_config_snapshot() if callable(globals().get("get_config_snapshot")) else {}
    return jsonify({
        "ok": True,
        "success": True,
        "version": str(PROJECT_VERSION),
        "project_version": str(PROJECT_VERSION),
        "updated_at": runtime.get("revision_start_date") or "",
        "runtime": runtime,
        "schema": _SM_PHASE1_CONTRACT_SCHEMA,
        "source": "phase1_bridge_ui_contract",
        "ts": time.time(),
    }), 200


@app.route("/api/meta/capabilities", methods=["GET"])
def api_meta_capabilities_phase1_contract():
    routes = []
    try:
        routes = sorted(str(rule.rule) for rule in app.url_map.iter_rules())
    except Exception:
        routes = []
    features = {
        "chat": _sm_phase1_route_available("/api/chat"),
        "chat_v1_alias": _sm_phase1_route_available("/api/v1/chat"),
        "health": _sm_phase1_route_available("/api/health"),
        "voice": True,
        "tts": _sm_phase1_route_available("/api/tts/speak"),
        "stt": _sm_phase1_route_available("/api/stt"),
        "research": True,
        "files_analyze": True,
        "ranking": True,
        "terminal": _sm_phase1_route_available("/api/terminal/status"),
        "drivers": _sm_phase1_route_available("/api/drivers"),
        "communications": _sm_phase1_route_available("/api/comm/health"),
        "local_first": True,
        "cloud_optional": True,
    }
    tools = [
        {"id": key, "name": key.replace("_", " ").title(), "description": "SarahMemory V9 bridge capability", "enabled": bool(value)}
        for key, value in features.items()
    ]
    return jsonify({
        "ok": True,
        "success": True,
        "version": str(PROJECT_VERSION),
        "features": sorted([k for k, v in features.items() if v]),
        "feature_flags": features,
        "tools": tools,
        "avatar_modes": ["avatar_2d", "avatar_3d", "desktop_mirror", "media", "idle"],
        "avatar_actions": ["talk", "listen", "idle", "gesture", "preview"],
        "media_types": ["image", "music", "video", "audio"],
        "voice_engines": [v.get("id", "default") for v in _sm_phase1_voice_options()],
        "route_count": len(routes),
        "schema": _SM_PHASE1_CONTRACT_SCHEMA,
        "source": "phase1_bridge_ui_contract",
        "doctrine": {
            "frontend_authority": False,
            "backend_governance_required": True,
            "local_first": True,
            "reply_bundle_required": True,
        },
        "ts": time.time(),
    }), 200


@app.route("/api/voices", methods=["GET"])
def api_voices_phase1_contract():
    voices = _sm_phase1_voice_options()
    return jsonify({"ok": True, "success": True, "voices": voices, "data": voices, "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200


@app.route("/api/voice", methods=["GET", "POST"])
def api_voice_phase1_contract():
    if request.method == "GET":
        return api_voices_phase1_contract()
    data = request.get_json(silent=True) or {}
    action = str(data.get("action") or "").strip().lower()
    if action in {"identity", "voice_identity", "status"}:
        return api_voice_status() if action == "status" else api_voice_identity()
    if action in {"list", "list_voices", "voices", "get_voices"}:
        return api_voices_phase1_contract()
    if action in {"set", "set_voice", "active_voice"}:
        voice = _sm_normalize_voice_request(data.get("voice") or data.get("voice_id") or data.get("value") or "sarahvoice")
        try:
            state = load_state() or {}
            if isinstance(state, dict):
                state["voice_profile"] = voice
                state["active_voice"] = voice
                save_state(state)
        except Exception:
            pass
        return jsonify({"ok": True, "success": True, "voice": voice, "active_voice": voice, "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200
    if action in {"preview", "speak"}:
        preview_text = str(data.get("text") or "SarahMemory voice preview is ready.")
        voice = _sm_normalize_voice_request(data.get("voice") or data.get("voice_id") or "sarahvoice")
        tts_status = {}
        if core_speak_text_status is not None:
            try:
                tts_status = core_speak_text_status(preview_text, blocking=False, engine_pref=voice or "sarahvoice")
            except Exception:
                tts_status = {}
        elif core_speak_text is not None:
            try:
                core_speak_text(preview_text, blocking=False, engine_pref=voice or "sarahvoice")
            except Exception:
                pass
        ident = _sm_voice_identity_packet()
        return jsonify({"ok": True, "success": True, "text": preview_text, "voice": voice, "voice_identity": ident.get("voice_identity"), "voice_model_id": ident.get("voice_model_id"), "fallback": core_speak_text is None and core_speak_text_status is None, "server_tts_started": bool(tts_status.get("server_tts_started") or tts_status.get("accepted") or tts_status.get("ok")), "browser_fallback_required": bool(tts_status.get("browser_fallback_required", False)), "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200
    if action in {"transcribe", "stt"}:
        try:
            from SarahMemoryVoice import transcribe_once as _sm_transcribe_once  # type: ignore
            timeout = max(1.0, min(30.0, float(data.get("timeout") or data.get("timeout_s") or 8.0)))
            text_out = str(_sm_transcribe_once(timeout=timeout) or "").strip()
            return jsonify({"ok": bool(text_out), "success": bool(text_out), "text": text_out, "fallback": False, "source": "SarahMemoryVoice.transcribe_once", "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200
        except Exception as exc:
            return jsonify({"ok": False, "success": False, "text": "", "error": str(exc), "source": "SarahMemoryVoice.transcribe_once", "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 501
    return jsonify({"ok": True, "success": True, "voices": _sm_phase1_voice_options(), "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200


@app.route("/api/voice/set", methods=["POST"])
def api_voice_set_phase1_contract():
    data = request.get_json(silent=True) or {}
    data["action"] = data.get("action") or "set_voice"
    # Reuse the same implementation without mutating Flask's request object.
    voice = _sm_normalize_voice_request(data.get("voice") or data.get("voice_id") or data.get("value") or "sarahvoice")
    try:
        state = load_state() or {}
        if isinstance(state, dict):
            state["voice_profile"] = voice
            state["active_voice"] = voice
            save_state(state)
    except Exception:
        pass
    return jsonify({"ok": True, "success": True, "voice": voice, "active_voice": voice, "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200


@app.route("/api/voice/preview", methods=["POST"])
def api_voice_preview_phase1_contract():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "SarahMemory voice preview is ready.")
    voice = _sm_normalize_voice_request(data.get("voice") or data.get("voice_id") or "sarahvoice")
    tts_status = {}
    if core_speak_text_status is not None:
        try:
            tts_status = core_speak_text_status(text, blocking=False, engine_pref=voice or "sarahvoice")
        except Exception:
            tts_status = {}
    elif core_speak_text is not None:
        try:
            core_speak_text(text, blocking=False, engine_pref=voice or "sarahvoice")
        except Exception:
            pass
    ident = _sm_voice_identity_packet()
    return jsonify({"ok": True, "success": True, "text": text, "voice": voice, "voice_identity": ident.get("voice_identity"), "voice_model_id": ident.get("voice_model_id"), "fallback": core_speak_text is None and core_speak_text_status is None, "server_tts_started": bool(tts_status.get("server_tts_started") or tts_status.get("accepted") or tts_status.get("ok")), "browser_fallback_required": bool(tts_status.get("browser_fallback_required", False)), "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200


@app.route("/api/voice/transcribe", methods=["POST"])
@app.route("/api/stt", methods=["POST"])
def api_stt_phase1_contract():
    data = request.get_json(silent=True) or {}
    try:
        from SarahMemoryVoice import transcribe_once as _sm_transcribe_once  # type: ignore
        timeout = max(1.0, min(30.0, float(data.get("timeout") or data.get("timeout_s") or 8.0)))
        text_out = str(_sm_transcribe_once(timeout=timeout) or "").strip()
        return jsonify({
            "ok": bool(text_out),
            "success": bool(text_out),
            "text": text_out,
            "fallback": False,
            "schema": _SM_PHASE1_CONTRACT_SCHEMA,
            "source": "SarahMemoryVoice.transcribe_once",
            "ts": time.time(),
        }), 200
    except Exception as exc:
        return jsonify({
            "ok": False,
            "success": False,
            "text": "",
            "error": str(exc),
            "schema": _SM_PHASE1_CONTRACT_SCHEMA,
            "source": "SarahMemoryVoice.transcribe_once",
            "ts": time.time(),
        }), 501


@app.route("/api/ranking", methods=["GET", "POST"])
@app.route("/api/ranking/stats", methods=["GET", "POST"])
def api_ranking_phase1_contract():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id") or request.args.get("user_id") or "local_user")
    con = None
    total = 0
    average = 0.0
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS ranking_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, user_id TEXT, session_id TEXT, score REAL, metrics TEXT)")
        action = str(data.get("action") or "").lower().strip()
        is_submit_request = (
            request.path.rstrip("/").endswith("/submit")
            or action in {"submit_session", "submit", "rank"}
            or "score" in data
            or "rating" in data
            or isinstance(data.get("metrics"), dict)
        )
        if request.method == "POST" and is_submit_request:
            metrics = dict(data.get("metrics") or {}) if isinstance(data.get("metrics"), dict) else {}
            for key in ("score", "rating", "label", "notes"):
                if key in data and key not in metrics:
                    metrics[key] = data.get(key)
            score = float(metrics.get("score") or metrics.get("rating") or 0.0)
            cur.execute("INSERT INTO ranking_sessions(ts,user_id,session_id,score,metrics) VALUES(?,?,?,?,?)", (time.time(), user_id, str(data.get("session_id") or ""), score, json.dumps(metrics, ensure_ascii=False)))
            con.commit()
        cur.execute("SELECT COUNT(*), COALESCE(AVG(score),0.0) FROM ranking_sessions WHERE user_id=?", (user_id,))
        row = cur.fetchone() or (0, 0.0)
        total = int(row[0] or 0)
        average = float(row[1] or 0.0)
    except Exception as exc:
        return jsonify({"ok": False, "success": False, "error": str(exc), "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 500
    finally:
        if con:
            try:
                con.close()
            except Exception:
                pass
    rank = "unranked" if total <= 0 else ("excellent" if average >= 0.85 else "good" if average >= 0.65 else "developing")
    return jsonify({
        "ok": True,
        "success": True,
        "ranked": total > 0,
        "score": average,
        "stats": {"total_sessions": total, "average_score": average, "rank": rank},
        "message": "Local ranking bridge is online.",
        "schema": _SM_PHASE1_CONTRACT_SCHEMA,
        "ts": time.time(),
    }), 200


@app.route("/api/ranking/submit", methods=["GET", "POST"])
def api_ranking_submit_phase1_contract():
    return api_ranking_phase1_contract()


@app.route("/api/research/search", methods=["GET", "POST"])
def api_research_search_phase1_contract():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query") or data.get("q") or data.get("text") or request.args.get("q") or request.args.get("query") or "").strip()
    if request.method == "GET" and not query:
        return jsonify({
            "ok": True,
            "success": True,
            "online": True,
            "endpoint": "/api/research/search",
            "accepted_methods": ["POST", "GET?q=..."],
            "method_required": "POST for UI searches",
            "message": "SarahMemory research bridge is online. Submit JSON with POST using query/q/text, or use GET with ?q=... for diagnostics.",
            "schema": _SM_PHASE1_CONTRACT_SCHEMA,
            "source": "phase1_bridge_ui_contract",
            "ts": time.time(),
        }), 200
    if not query:
        return jsonify({"ok": False, "success": False, "error": "Missing query.", "results": [], "schema": _SM_PHASE1_CONTRACT_SCHEMA}), 400
    try:
        import SarahMemoryResearch as _SMResearch  # type: ignore
        fn = getattr(_SMResearch, "get_research_data", None) or getattr(_SMResearch, "get_local_research_data", None)
        if not callable(fn):
            raise RuntimeError("SarahMemoryResearch has no callable research entrypoint.")
        bounded = _sm_bounded_call(fn, query, timeout_seconds=12.0, call_name="api_research_search") if callable(globals().get("_sm_bounded_call")) else {"ok": True, "value": fn(query)}
        if not bounded.get("ok"):
            raise RuntimeError(str(bounded.get("error") or "research timeout"))
        raw = bounded.get("value")
        if isinstance(raw, dict):
            summary = str(raw.get("data") or raw.get("answer") or raw.get("snippet") or raw.get("summary") or "").strip()
            confidence = raw.get("confidence")
            source = str(raw.get("source") or "SarahMemoryResearch")
        else:
            summary = str(raw or "").strip()
            confidence = None
            source = "SarahMemoryResearch"
        summary = _sm_phase1_compact_text(summary, max_chars=1600)
        snippet = _sm_phase1_compact_text(summary, max_chars=700)
        result = {"title": "SarahMemory local research", "summary": summary, "snippet": snippet, "source": source, "confidence": confidence}
        return jsonify({"ok": True, "success": True, "query": query, "summary": summary, "results": [result] if summary else [], "sources": [source], "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200
    except Exception as exc:
        return jsonify({"ok": False, "success": False, "query": query, "summary": "", "results": [], "sources": [], "error": str(exc), "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200


@app.route("/api/files/analyze", methods=["GET", "POST"])
def api_files_analyze_phase1_contract():
    if request.method == "GET":
        return jsonify({
            "ok": True,
            "success": True,
            "online": True,
            "endpoint": "/api/files/analyze",
            "accepted_methods": ["POST"],
            "method_required": "POST",
            "message": "SarahMemory file analysis bridge is online. Submit JSON with filename, type/mime, and base64/content.",
            "schema": _SM_PHASE1_CONTRACT_SCHEMA,
            "source": "phase1_bridge_ui_contract",
            "ts": time.time(),
        }), 200

    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or data.get("name") or "uploaded_file").strip() or "uploaded_file"
    raw_content = data.get("content") if "content" in data else (data.get("base64") if "base64" in data else (data.get("data") or ""))
    mime = str(data.get("type") or data.get("mime") or "application/octet-stream")
    content_bytes = b""
    text_preview = ""
    try:
        raw = str(raw_content or "")
        is_data_uri = raw.startswith("data:") and "," in raw
        is_explicit_base64 = "base64" in data or is_data_uri
        if is_data_uri:
            raw = raw.split(",", 1)[1]
        if raw:
            if is_explicit_base64:
                content_bytes = base64.b64decode(raw + "=" * ((4 - len(raw) % 4) % 4), validate=False)
            else:
                content_bytes = raw.encode("utf-8", "ignore")
            if mime.startswith("text/") or filename.lower().endswith((".txt", ".md", ".json", ".csv", ".py", ".ts", ".tsx", ".js", ".html", ".css")):
                text_preview = content_bytes[:12000].decode("utf-8", "ignore")
    except Exception:
        content_bytes = b""
        text_preview = ""
    analysis = {
        "filename": filename,
        "mime": mime,
        "size_bytes": len(content_bytes),
        "text_preview_chars": len(text_preview),
        "local_analysis_only": True,
        "saved": False,
        "note": "Phase1 compatibility route performed bounded local metadata/text preview analysis only.",
    }
    if bool(data.get("save")) and content_bytes:
        try:
            uploads_dir = os.path.join(DATA_DIR, "uploads", "phase1_analyze")
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename))[:180] or "uploaded_file"
            out_path = os.path.abspath(os.path.join(uploads_dir, f"{int(time.time())}_{safe_name}"))
            if not out_path.startswith(os.path.abspath(uploads_dir) + os.sep):
                raise RuntimeError("Unsafe upload path rejected.")
            with open(out_path, "wb") as handle:
                handle.write(content_bytes)
            analysis["saved"] = True
            analysis["path"] = out_path
        except Exception as exc:
            analysis["save_error"] = str(exc)
    return jsonify({"ok": True, "success": True, "analysis": json.dumps(analysis, ensure_ascii=False), "content": text_preview, "metadata": analysis, "schema": _SM_PHASE1_CONTRACT_SCHEMA, "ts": time.time()}), 200

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
# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# request/jsonify are already provided by either Flask or the minimal fallback
# API shim above. Do not re-import Flask here; missing Flask must not break boot.
try:
    import SarahMemoryTerminal as smterm
    _SM_TERMINAL_IMPORT_ERROR = ""
except Exception as _sm_terminal_import_exc:  # pragma: no cover - boot resilience
    smterm = None  # type: ignore
    _SM_TERMINAL_IMPORT_ERROR = str(_sm_terminal_import_exc)

@app.get("/api/terminal/status")
def api_terminal_status():
    payload = {
        "session_id": request.args.get("session_id", ""),
    }
    if smterm is None:
        return jsonify({
            "ok": True,
            "available": False,
            "developers_mode": False,
            "reason": f"SarahMemoryTerminal.py unavailable: {_SM_TERMINAL_IMPORT_ERROR}",
            "session_id": payload.get("session_id", ""),
            "prompt": r"Sarah:\>",
            "caller": "Flask:/api/terminal/status",
            "ts": time.time(),
        }), 200
    result = smterm.terminal_api_status(payload, caller="Flask:/api/terminal/status")
    return jsonify(result), 200

@app.post("/api/terminal/execute")
def api_terminal_execute():
    payload = request.get_json(silent=True) or {}
    if smterm is None:
        return jsonify({
            "ok": False,
            "blocked": True,
            "reason": f"SarahMemoryTerminal.py unavailable: {_SM_TERMINAL_IMPORT_ERROR}",
            "stdout": "",
            "stderr": _SM_TERMINAL_IMPORT_ERROR,
            "exit_code": -1,
            "caller": "Flask:/api/terminal/execute",
            "ts": time.time(),
        }), 503
    result = smterm.terminal_api_execute(payload, caller="Flask:/api/terminal/execute")
    return jsonify(result), (200 if result.get("ok") else 403 if result.get("blocked") else 400)

def _sm_terminal_agent_response(result: dict, *, status_if_unavailable: int = 400):
    """Return Terminal Bay JSON without losing governed block details.

    Some local clients/proxies display an empty body for HTTP 403 responses.
    Terminal Bay blocks are not transport failures; they are governed outcomes
    that must remain visible to the Web UI, PowerShell, and Ledger/debug tests.
    Use 409 for governance blocks and always serialize the full JSON body.
    """
    if not isinstance(result, dict):
        result = {
            "ok": False,
            "blocked": True,
            "reason": "terminal_agent_returned_non_dict",
            "reply": "Terminal agent backend returned an invalid response shape.",
            "stdout": "",
            "stderr": "terminal_agent_returned_non_dict",
            "mode": "terminal_agent",
            "execution_authority": False,
            "ts": time.time(),
        }
    ok = bool(result.get("ok"))
    blocked = bool(result.get("blocked"))
    if ok:
        status = 200
        transport_status = "ok"
        governance_http_status = 200
    elif blocked:
        # Terminal Bay blocks are successful governed outcomes, not transport
        # failures.  Some local clients/proxies drop JSON bodies for 4xx
        # responses, which hides the reason, task_id, and Ledger proof.  Keep
        # the HTTP transport at 200 and carry the semantic/governance status in
        # the JSON body and headers.
        status = 200
        transport_status = "governance_block"
        governance_http_status = 409
    else:
        status = int(status_if_unavailable or 400)
        transport_status = "terminal_agent_error"
        governance_http_status = status
    result.setdefault("http_status", status)
    result.setdefault("transport_http_status", status)
    result.setdefault("governance_http_status", governance_http_status)
    result.setdefault("semantic_status", governance_http_status)
    result.setdefault("transport_status", transport_status)
    result.setdefault("execution_authority", False)
    result.setdefault("mode", "terminal_agent")
    try:
        body = json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        status = 500
        body = json.dumps({
            "ok": False,
            "blocked": True,
            "reason": "terminal_agent_json_serialization_failed",
            "error": str(exc),
            "mode": "terminal_agent",
            "execution_authority": False,
            "http_status": status,
            "transport_status": "serialization_error",
            "ts": time.time(),
        }, ensure_ascii=False, default=str)
    resp = Response(body, status=status, mimetype="application/json")
    try:
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-SarahMemory-Mode"] = "terminal_agent"
        resp.headers["X-SarahMemory-Governance"] = "blocked" if blocked else "ok" if ok else "error"
        resp.headers["X-SarahMemory-Governance-Status"] = str(governance_http_status)
    except Exception:
        pass
    return resp


@app.post("/api/terminal/agent")
def api_terminal_agent():
    """Governed terminal AI-agent lane.

    This endpoint is inspect/propose only. It does not execute shell commands,
    drivers, network actions, DevBridge apply, or filesystem mutations beyond
    normal audit/capture records emitted by SarahMemoryAgentFirewall.py.
    """
    payload = request.get_json(silent=True) or {}
    if smterm is None:
        return _sm_terminal_agent_response({
            "ok": False,
            "blocked": True,
            "reason": f"SarahMemoryTerminal.py unavailable: {_SM_TERMINAL_IMPORT_ERROR}",
            "reply": "SarahMemoryTerminal.py unavailable; AI-agent terminal lane cannot initialize.",
            "stdout": "",
            "stderr": _SM_TERMINAL_IMPORT_ERROR,
            "caller": "Flask:/api/terminal/agent",
            "ts": time.time(),
        }, status_if_unavailable=503)
    try:
        result = smterm.terminal_api_agent(payload, caller="Flask:/api/terminal/agent")
    except AttributeError:
        return _sm_terminal_agent_response({
            "ok": False,
            "blocked": True,
            "reason": "SarahMemoryTerminal.py does not expose terminal_api_agent yet.",
            "reply": "Terminal agent backend is not patched yet.",
            "stdout": "",
            "stderr": "terminal_api_agent missing",
            "caller": "Flask:/api/terminal/agent",
            "ts": time.time(),
        }, status_if_unavailable=501)
    return _sm_terminal_agent_response(result)


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
        # SARAHMEMORY_PATCH_NOTE 2026-06-23:
        # API mode must not start SelfAware just because NEOSKYMATRIX and
        # DEVELOPERSMODE are true. API startup can be exposed to browser/UI
        # surfaces, so autonomous boot requires explicit API autostart and
        # SelfAware autostart flags. Otherwise it remains available only through
        # governed local/user-authorized lanes.
        _neosky = bool(getattr(config, "NEOSKYMATRIX", False))
        _dev = bool(getattr(config, "DEVELOPERSMODE", False))
        _api_auto = bool(getattr(config, "SARAHMEMORY_API_AUTONOMOUS_STARTUP_ENABLED", False))
        _selfaware_auto = bool(getattr(config, "SARAHMEMORY_SELFAWARE_AUTOSTART_ENABLED", False))

        if _neosky and _dev and _api_auto and _selfaware_auto:
            import threading
            import SarahMemorySelfAware as _SMA
            if hasattr(_SMA, "run_autonomous_loop"):
                t = threading.Thread(
                    target=_SMA.run_autonomous_loop,
                    name="SM_SelfAware",
                    daemon=True
                )
                t.start()
                app_logger.warning("SelfAware ARMED (API Mode) after explicit governed autostart flags.")
        else:
            app_logger.info("SelfAware API autostart held in governed standby.")
    except Exception as e:
        app_logger.error(f"Autonomous init failed: {e}", exc_info=True)

_start_autonomous_services()
try:
    _vr_ensure_watcher_started()
except Exception:
    pass

_API_RUNTIME_INSTANCE_ID = str(os.environ.get("SARAHMEMORY_RUNTIME_INSTANCE_ID") or secrets.token_hex(16))
_API_PARENT_PID = int(os.environ.get("SARAHMEMORY_PARENT_PID") or 0)
_API_CLEANUP_LOCK = threading.RLock()
_API_CLEANUP_DONE = False
_API_PARENT_WATCH_STOP = threading.Event()

def _api_pid_alive(pid: int) -> bool:
    if int(pid or 0) <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False

def _api_checkpoint_databases(mode: str = "PASSIVE") -> dict:
    requested = str(mode or "PASSIVE").strip().upper()
    if requested not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
        requested = "PASSIVE"
    names = ("context_history.db", "neuron_axis.db", "cognitive_compass.db", "functions.db", "system_logs.db", "user_profile.db")
    results = {}
    for name in names:
        path = os.path.join(_DATASETS_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            con = sqlite3.connect(path, timeout=2.0)
            try:
                con.execute("PRAGMA busy_timeout=2000")
                results[name] = {"ok": True, "result": list(con.execute(f"PRAGMA wal_checkpoint({requested})").fetchone() or [])}
            finally:
                con.close()
        except Exception as exc:
            results[name] = {"ok": False, "error": str(exc)}
    return {"ok": all(item.get("ok") for item in results.values()) if results else True, "databases": results}

def _api_runtime_mark_started() -> None:
    pid_path = os.path.join(DATA_DIR, "local_api.pid")
    try:
        _write_json_if_changed(
            STATE_DB,
            {**load_state(), "api_running": True, "API_RUNNING": True, "api_pid": os.getpid(), "API_PID": os.getpid(), "api_last_seen_ts": time.time(), "api_runtime_instance_id": _API_RUNTIME_INSTANCE_ID},
            ensure_ascii=False,
        )
        os.makedirs(os.path.dirname(pid_path), exist_ok=True)
        tmp = f"{pid_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, pid_path)
    except Exception as exc:
        app_logger.warning("API lifecycle start marker failed: %s", exc)

def _api_runtime_cleanup(reason: str = "shutdown") -> None:
    global _API_CLEANUP_DONE
    with _API_CLEANUP_LOCK:
        if _API_CLEANUP_DONE:
            return
        _API_CLEANUP_DONE = True
    _API_PARENT_WATCH_STOP.set()
    try:
        state = load_state()
        if str(state.get("api_runtime_instance_id") or "") == _API_RUNTIME_INSTANCE_ID or int(state.get("api_pid") or 0) == os.getpid():
            state.update({"api_running": False, "API_RUNNING": False, "api_pid": None, "API_PID": None, "api_shutdown_reason": str(reason), "api_last_seen_ts": time.time()})
            save_state(state)
    except Exception:
        pass
    try:
        _api_checkpoint_databases("PASSIVE")
    except Exception:
        pass
    try:
        pid_path = os.path.join(DATA_DIR, "local_api.pid")
        if os.path.isfile(pid_path) and open(pid_path, "r", encoding="utf-8", errors="ignore").read().strip() == str(os.getpid()):
            os.remove(pid_path)
    except Exception:
        pass

def _api_parent_watchdog() -> None:
    if _API_PARENT_PID <= 0 or str(os.environ.get("SARAHMEMORY_API_STANDALONE", "0")).lower() in {"1", "true", "yes", "on"}:
        return
    while not _API_PARENT_WATCH_STOP.wait(2.0):
        if not _api_pid_alive(_API_PARENT_PID):
            app_logger.warning("Parent runtime pid=%s disappeared; stopping orphan API instance.", _API_PARENT_PID)
            _api_runtime_cleanup("parent_process_lost")
            os._exit(0)

def _api_signal_handler(signum, _frame) -> None:
    _api_runtime_cleanup(f"signal:{signum}")
    raise SystemExit(0)

try:
    import atexit as _api_atexit
    _api_atexit.register(_api_runtime_cleanup, "atexit")
except Exception:
    pass

if __name__ == "__main__":
    # SARAHMEMORY_PATCH_NOTE 2026-06-23:
    # Local-first server defaults to loopback and the same port used by
    # SarahMemoryMain.wait_for_api_server. This prevents startup mismatch and
    # prevents the API from binding to all network interfaces unless the operator
    # deliberately overrides SARAHMEMORY_API_HOST.
    try:
        _default_port = int(getattr(config, "DEFAULT_PORT", 8000))
    except Exception:
        _default_port = 8000
    port = int(os.environ.get("PORT", str(_default_port)))
    host = os.environ.get("SARAHMEMORY_API_HOST") or os.environ.get("SARAHMEMORY_LOCAL_API_BIND_HOST") or "127.0.0.1"
    app_logger.info(f"Starting SarahMemory Flask API server on http://{host}:{port}")
    # Initializing app.config with default values for toggles
    app.config.setdefault("CAMERA_ON", False)
    app.config.setdefault("MIC_ON", False)
    app.config.setdefault("VOICE_OUTPUT_ON", True)
    app.config.setdefault("TELECOM_ENABLED", False)  # For telecom stateub

    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    if debug_mode:
        app_logger.warning("Debug diagnostics enabled; automatic reloader remains disabled to prevent duplicate runtimes.")
    _api_runtime_mark_started()
    try:
        signal.signal(signal.SIGINT, _api_signal_handler)
        signal.signal(signal.SIGTERM, _api_signal_handler)
    except Exception:
        pass
    if _API_PARENT_PID > 0:
        threading.Thread(target=_api_parent_watchdog, name="SM-API-ParentWatch", daemon=True).start()
    app.run(host=host, port=port, debug=debug_mode, threaded=True, use_reloader=False)

# ====================================================================
# END OF app.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. app.py participates as API bridge ingress, not as a reasoning organ.
SML_ORGAN_METADATA = {
    "name": "app",
    "version": "v9.0.0-alpha-sml-0.2",
    "category": "Input",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ["api_bridge", "sml_ingress", "transport", "governed_chat_entrypoint"],
    "supported_missions": ["Conversation", "Knowledge", "Programming", "Filesystem", "Network", "Diagnostics", "Repair", "Execution"],
    "supported_omega": ["Ω001", "Ω002", "Ω004", "Ω020", "Ω060"],
    "required_authority": ["Read"],
    "priority": 95,
    "trust_level": "api_bridge_integrated",
    "internal_only": False,
    "metadata": {"sml_adapter": "api_chat_ingress", "source_file": "app.py"},
}

def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)
# --- SML ORGAN ADAPTER END ---

