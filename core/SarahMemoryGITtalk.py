#!/usr/bin/env python3

"""--==The SarahMemory Project==--
File: SarahMemoryGITtalk.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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
===============================================================================

SarahMemoryGITtalk.py  (MASTER monkey patch - FRONTEND ONLY)

TEMP placement (as you requested):
<PROJECT_ROOT>/data/mods/v900/SarahMemoryGITtalk.py

What this file does (when present on PythonAnywhere):
- Exposes secure API endpoints (Flask Blueprint) so you can:
1) chat/plan a patch (OpenAI → unified diff)
2) preview the patch in the WebUI
3) apply the patch (git apply → commit → PUSH to sarah-s-dashboard)
4) automatically run SarahMemoryUIupdater.py (build/deploy WebUI)
- ALSO supports CLI modes (optional), but you can run everything via WebUI.

STRICT RULES:
- ONLY operates on frontend checkout:
<PROJECT_ROOT>/ui/V9_ui_src
- PUSHES ONLY to sarah-s-dashboard repo (origin url must contain "sarah-s-dashboard")
- NEVER touches sarahmemory repo
- NEVER touches sarahmemory-api repo

SECURITY:
- All API endpoints require header:
X-Sarah-Admin-Key: <secret>
Secret must be set server-side in .env as SARAH_ADMIN_KEY (or SARAH_WEBUI_ADMIN_KEY).
- No secrets are ever sent to the browser except via your own backend.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "frontend_patch_tool"
# CATEGORY = "frontend_patch_operations"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "sandbox"
# API_DOMAIN = "webui_admin"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "gittalk"
# FAMILY = "ui_governance"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Restricted frontend-only patch planning/apply/push tool for sarah-s-dashboard with admin-key gating, git safety checks, and optional UI rebuild trigger."
# --- SARAHMETA END ---

import os
import sys
import re
import json
import time
import uuid
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict, Tuple

# ARILE protected cleanup/update guard. Protected immune-system files are never
# cleanup/update targets from automated maintenance lanes.
try:
    from SarahMemoryARILE import arile_is_protected_core_file, arile_emit
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    def arile_is_protected_core_file(path_value):
        try:
            return Path(path_value).name.lower() in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}
        except Exception:
            return False

def _arile_block_if_protected_target(path_value: str, operation: str = "maintenance") -> None:
    if arile_is_protected_core_file(path_value):
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, kind="protected_core_variance", failure_type="maintenance_target_blocked", severity=0.92, confidence=0.90, risk="critical", summary=f"Blocked {operation} against protected core file.", requires_governance=True, retention="security_audit", data={"target": str(path_value)})
        except Exception:
            pass
        raise PermissionError(f"Protected core maintenance target blocked: {path_value}")


# OpenAI import is optional at runtime (AI endpoints/mode only).
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Flask blueprint support (only used if your backend imports + registers it)
try:
    from flask import Blueprint, request, jsonify  # type: ignore
except Exception:
    Blueprint = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore


# =============================================================================
# PATHS (auto-detected; no hard-coded machine paths)
# =============================================================================

def _detect_core_dir() -> Path:
    """Best-effort project root detection.

    Works when this file is placed either:
      - BASE_DIR/SarahMemoryGITtalk.py
      - BASE_DIR/data/mods/v800/SarahMemoryGITtalk.py

    We walk upward looking for markers like app.py and data/ directory.
    Env override (highest priority):
      - SARAH_BASE_DIR
      - SARAHMEMORY_BASE_DIR
      - BASE_DIR
    """
    for env_key in ("SARAHMEMORY_ROOT", "SARAH_BASE_DIR", "SARAHMEMORY_BASE_DIR", "BASE_DIR"):
        v = os.getenv(env_key)
        if v:
            try:
                p = Path(v).expanduser().resolve()
                if p.exists():
                    return p
            except Exception:
                pass

    here = Path(__file__).expanduser().resolve()
    for parent in [here.parent] + list(here.parents):
        # stop if too high
        try:
            if parent.name.lower() == "core" and (parent / "SarahMemoryGlobals.py").exists():
                return parent.parent
            if (parent / "core" / "SarahMemoryGlobals.py").exists():
                return parent
            if (parent / "app.py").exists() and (parent / "data").exists():
                return parent
            if (parent / "SarahMemoryGlobals.py").exists() and (parent / "data").exists():
                return parent
        except Exception:
            continue

    # fallback: directory containing this file
    return here.parent

CORE_DIR = _detect_core_dir()

# Frontend checkout used by SarahMemoryUIupdater.py
REPO_DIR = CORE_DIR / "ui" / "V9_ui_src"

# After push, run updater to build/deploy
UI_UPDATER = CORE_DIR / "core" / "SarahMemoryUIupdater.py"

# Patch inbox (NOT inside any repo)
PATCH_DIR = CORE_DIR / "data" / "mods" / "v900" / "patches"

# State files for server-side persistence
STATE_DIR = CORE_DIR / "data" / "mods" / "v900"
AI_STATE_FILE = STATE_DIR / "gittalk_state.json"
RUN_STATE_FILE = STATE_DIR / "gittalk_run_state.json"
PENDING_DIR = STATE_DIR / "gittalk_pending"  # stores patch_id -> patch text

# Git safety
ALLOWED_REMOTE = "origin"
ALLOWED_BRANCH = "main"
ALLOWED_REPO_KEYWORD = "sarah-s-dashboard"

# GitHub token (optional if origin remote is SSH deploy key)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GITHUB_PAT")

# Optional HTTPS repo URL (without token). Only used if you WANT to re-point origin.
# Example: https://github.com/<owner>/sarah-s-dashboard.git
GIT_HTTPS_URL = os.getenv("SARAH_GIT_HTTPS_URL") or os.getenv("UI_REPO_URL")

# WebUI admin gate (must be set on PythonAnywhere, never in frontend)
SARAH_ADMIN_KEY = (os.getenv("SARAH_ADMIN_KEY") or os.getenv("SARAH_WEBUI_ADMIN_KEY") or "").strip()


# =============================================================================
# OpenAI embedded "bridge" (AI patch generator)
# =============================================================================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

AI_SYSTEM_PROMPT = """You are SarahMemory GitTalk.

STRICT RULES:
- Output ONLY a unified git diff patch.
- Target ONLY the sarah-s-dashboard frontend repo.
- NEVER modify backend, APIs, infra, deployment files, or server config.
- NO markdown. NO explanations. NO code fences.
- If clarification is needed, respond exactly:
  ASK: <one short question>
"""


# =============================================================================
# Small helpers
# =============================================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_unix() -> float:
    return time.time()


def _print(title: str, body: str = "") -> None:
    bar = "=" * 72
    msg = f"\n{bar}\n[{_ts()}] {title}\n{bar}"
    if body:
        msg += f"\n{body}\n"
    else:
        msg += "\n"
    print(msg)


def _mask_token(url: str) -> str:
    try:
        if not url:
            return url
        # mask https://TOKEN@github.com/...
        return re.sub(r"(https?://)([^/@:]+)(@github\.com/)", r"\1***\3", url)
    except Exception:
        return url



def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Bounded Git subprocess; no shell, no secrets in outputs."""
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=check, timeout=30)


def _read_json(path: Path, default: dict) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _set_run_state(**kwargs: Any) -> None:
    state = _read_json(RUN_STATE_FILE, default={})
    state.update(kwargs)
    state["updated_ts"] = _now_unix()
    _write_json(RUN_STATE_FILE, state)


def _get_run_state() -> dict:
    return _read_json(RUN_STATE_FILE, default={
        "status": "idle",
        "last_error": "",
        "last_ok": "",
        "last_patch_id": "",
        "last_commit": "",
        "last_push": "",
        "last_build": "",
        "updated_ts": 0,
    })


def _ensure_dirs() -> None:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Security gate for HTTP
# =============================================================================
def _require_admin(req) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    Requires SARAH_ADMIN_KEY and matching header X-Sarah-Admin-Key.
    """
    if not SARAH_ADMIN_KEY:
        return False, "Server misconfig: SARAH_ADMIN_KEY not set."

    hdr = (req.headers.get("X-Sarah-Admin-Key") or "").strip()
    if not hdr:
        return False, "Missing X-Sarah-Admin-Key header."
    if not __import__("hmac").compare_digest(hdr, SARAH_ADMIN_KEY):
        return False, "Invalid admin key."
    return True, ""


# =============================================================================
# Safety gates for repo
# =============================================================================
def _ensure_repo() -> None:
    if not REPO_DIR.exists():
        raise RuntimeError(f"Frontend repo folder not found: {REPO_DIR}")
    if not (REPO_DIR / ".git").exists():
        raise RuntimeError(f"Not a git repo (missing .git): {REPO_DIR}")
    if not (REPO_DIR / "package.json").exists():
        raise RuntimeError("Safety gate: package.json missing. This doesn't look like the frontend checkout.")



def _ensure_correct_remote() -> str:
    """Require exact GitHub repository basename on origin (owner may vary)."""
    from urllib.parse import urlsplit
    raw = (_run(["git", "remote", "get-url", ALLOWED_REMOTE], cwd=REPO_DIR).stdout or "").strip()
    if raw.startswith("git@github.com:"):
        tail = raw.split(":", 1)[1]
    else:
        u = urlsplit(raw)
        if u.scheme != "https" or u.hostname != "github.com" or u.username or u.password or u.port:
            raise RuntimeError("Untrusted frontend origin host or credentials")
        tail = u.path.lstrip("/")
    parts = tail.rstrip("/").split("/")
    approved_owner = os.getenv("SARAH_GITTALK_REPO_OWNER", "").strip()
    if (not approved_owner or not re.fullmatch(r"[A-Za-z0-9_.-]+", approved_owner)
            or len(parts) != 2 or parts[0] != approved_owner
            or parts[1] not in ("sarah-s-dashboard", "sarah-s-dashboard.git")):
        raise RuntimeError("Frontend remote must match an explicitly approved GitHub owner and repository")
    return raw



def _configure_https_remote_if_needed() -> None:
    """Do not persist an embedded Git access token in repository configuration."""
    if GIT_HTTPS_URL:
        raise PermissionError("Credential-bearing remote rewrite disabled; use a preconfigured secure Git credential helper")


# =============================================================================
# Git ops
# =============================================================================
def git_pull() -> str:
    _configure_https_remote_if_needed()
    cp = _run(["git", "pull", ALLOWED_REMOTE, ALLOWED_BRANCH], cwd=REPO_DIR, check=True)
    return cp.stdout or ""


def git_status_porcelain() -> str:
    cp = _run(["git", "status", "--porcelain=v1"], cwd=REPO_DIR, check=True)
    return (cp.stdout or "").strip()



def apply_patch_text(patch_text: str) -> str:
    ok, why = validate_unified_diff(patch_text)
    if not ok:
        raise ValueError(why)
    raise PermissionError("Git apply requires an authenticated per-patch OperatorCore deployment receipt")



def commit_and_push(message: str) -> str:
    raise PermissionError("Commit and push require an authenticated per-patch OperatorCore deployment receipt")


def get_head_commit() -> str:
    cp = _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True)
    return (cp.stdout or "").strip()


# =============================================================================
# Build/deploy
# =============================================================================

def run_ui_updater() -> str:
    raise PermissionError("UI deployment requires an authenticated per-patch OperatorCore deployment receipt")


# =============================================================================
# OpenAI state + generation (persistent via previous_response_id)
# =============================================================================
def _load_ai_state() -> dict:
    return _read_json(AI_STATE_FILE, default={})


def _save_ai_state(state: dict) -> None:
    _write_json(AI_STATE_FILE, state)


def ai_generate_patch(user_request: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai python package not installed. Install in venv: pip install openai")
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_KEY)

    state = _load_ai_state()
    prev_id = state.get("previous_response_id") or None

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ],
        # store enables server-side retention; previous_response_id chains the conversation
        "store": False,
    }
    if prev_id:
        kwargs["previous_response_id"] = prev_id

    resp = client.responses.create(**kwargs)

    # persist chain id
    try:
        state["previous_response_id"] = getattr(resp, "id", None) or prev_id
        state["updated_ts"] = _now_unix()
        _save_ai_state(state)
    except Exception:
        pass

    # Preferred: SDK often provides output_text
    out_text = getattr(resp, "output_text", None)
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()

    # Fallback parse
    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "output_text":
            chunks.append(getattr(item, "text", ""))
    return "\n".join(chunks).strip()


# =============================================================================
# Patch storage (so WebUI can preview/apply without bash)
# =============================================================================
def _safe_patch_id(patch_id: str) -> str:
    if not isinstance(patch_id, str) or not re.fullmatch(r"patch_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}", patch_id):
        raise ValueError("invalid_patch_id")
    return patch_id


def _new_patch_id() -> str:
    return f"patch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_pending_patch(patch_text: str, patch_id: Optional[str] = None) -> str:
    _ensure_dirs()
    if not patch_id:
        patch_id = _new_patch_id()
    path = PENDING_DIR / f"{_safe_patch_id(patch_id)}.patch"
    ok, why = validate_unified_diff(patch_text)
    if not ok:
        raise ValueError(why)
    if path.exists() or path.is_symlink():
        raise FileExistsError("patch_id_already_exists")
    with path.open("x", encoding="utf-8") as f:
        f.write(patch_text)
    return patch_id


def load_pending_patch(patch_id: str) -> str:
    path = PENDING_DIR / f"{_safe_patch_id(patch_id)}.patch"
    if not path.exists():
        raise RuntimeError(f"Pending patch not found: {patch_id}")
    return path.read_text(encoding="utf-8", errors="replace")


def delete_pending_patch(patch_id: str) -> None:
    path = PENDING_DIR / f"{_safe_patch_id(patch_id)}.patch"
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass



def validate_unified_diff(patch_text: str) -> Tuple[bool, str]:
    """Inspect every Git diff header and reject protected/escaping paths.

    Validation alone is not a permission to apply a patch.
    """
    if not isinstance(patch_text, str) or len(patch_text.encode("utf-8")) > 1024 * 1024:
        return False, "empty or oversized patch"
    s = patch_text.strip()
    if not s.startswith("diff --git ") or "\x00" in s:
        return False, "missing Git unified diff header"
    paths = []
    for line in s.splitlines():
        if line.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            if not match or match.group(1) != match.group(2):
                return False, "invalid/rename patch header"
            paths.append(match.group(1))
        if line.startswith(("rename from ", "rename to ", "copy from ", "copy to ", "GIT binary patch", "Binary files ", "Submodule ")):
            return False, "rename, copy, binary and submodule patches require dedicated review"
    if not paths or len(paths) > 40:
        return False, "missing or excessive changed paths"
    banned_dirs = {".git", ".github", "api", "server", "core", "infra", "deployment", "scripts", "node_modules", "dist", "build"}
    banned_names = {".env", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "vite.config.ts", "vite.config.js", "package.json"}
    for p in paths:
        parts = p.replace("\\", "/").split("/")
        if (p.startswith("/") or ":" in p or any(x in ("", ".", "..") for x in parts)
                or parts[0].lower() in banned_dirs
                or any(x.lower() in banned_dirs or x.lower() in banned_names for x in parts)
                or not p.lower().endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".html", ".svg", ".json"))):
            return False, "protected, escaping or unsupported frontend path"
    return True, ""


# =============================================================================
# Core apply pipeline (non-interactive)
# =============================================================================
def apply_patch_id_and_deploy(patch_id: str) -> dict:
    """Review-only transition: no authenticated approval receipt in legacy contract.

    The admin header authenticates the API caller but does not authorize an
    independently verified Git commit, push, or production deployment.
    """
    started = _now_unix()
    try:
        patch = load_pending_patch(patch_id)
        good, reason = validate_unified_diff(patch)
        if not good:
            raise ValueError(reason)
        message = "OperatorCore per-patch deployment receipt unavailable; proposal retained for review"
        _set_run_state(status="review_required", last_error=message, last_patch_id=patch_id)
        return {"ok": False, "patch_id": patch_id, "status": "review_required",
                "steps": [{"name": "validation", "out": "passed"}],
                "error": message, "started_ts": started, "finished_ts": _now_unix()}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _set_run_state(status="blocked", last_error=message)
        return {"ok": False, "patch_id": patch_id, "status": "blocked",
                "steps": [], "error": message, "started_ts": started, "finished_ts": _now_unix()}


# =============================================================================
# Flask Blueprint (Monkey Patch API surface)
# =============================================================================
def create_gittalk_blueprint(url_prefix: str = "/api/gittalk"):
    """
    Call this from app.py/appnet.py (ONLY when this mod file exists) to mount endpoints.
    """
    if Blueprint is None:
        raise RuntimeError("Flask is not available in this environment.")

    # Ensure patch inbox exists
    try:
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    bp = Blueprint("sarah_gittalk", __name__, url_prefix=url_prefix)
    @bp.route("/ping", methods=["GET"])
    def ping():
        """Lightweight liveness probe.

        Public: returns minimal info without requiring admin key.
        If admin key is provided, returns extra diagnostics (paths, repo state).
        """
        is_admin, _ = _require_admin(request)
        payload = {
            "ok": True,
            "service": "gittalk",
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        if is_admin:
            payload.update({
                "core_dir": str(CORE_DIR),
                "repo_dir": str(REPO_DIR),
                "repo_exists": REPO_DIR.exists(),
                "patch_dir": str(PATCH_DIR),
                "patch_dir_exists": PATCH_DIR.exists(),
            })
        return jsonify(payload)


    @bp.route("/status", methods=["GET"])
    def status():
        ok, reason = _require_admin(request)
        if not ok:
            return jsonify({"ok": False, "error": reason}), 401
        return jsonify({"ok": True, "state": _get_run_state()}), 200

    @bp.route("/plan", methods=["POST"])
    def plan():
        ok, reason = _require_admin(request)
        if not ok:
            return jsonify({"ok": False, "error": reason}), 401

        payload = request.get_json(silent=True) or {}
        user_request = (payload.get("request") or "").strip()
        if not user_request:
            return jsonify({"ok": False, "error": "Missing 'request'"}), 400

        _ensure_dirs()
        _set_run_state(status="planning", last_error="", last_ok="", last_patch_id="")

        try:
            patch_text = ai_generate_patch(user_request)

            # If the model asks a question, return it (no patch stored)
            if patch_text.lstrip().startswith("ASK:"):
                _set_run_state(status="idle", last_ok="AI requested clarification.", last_error="")
                return jsonify({"ok": True, "ask": patch_text.strip()}), 200

            ok2, why = validate_unified_diff(patch_text)
            if not ok2:
                _set_run_state(status="idle", last_error=f"Invalid diff: {why}")
                return jsonify({"ok": False, "error": f"AI did not return valid diff: {why}", "raw": patch_text[:4000]}), 400

            patch_id = save_pending_patch(patch_text)
            _set_run_state(status="idle", last_ok="Patch planned.", last_patch_id=patch_id)

            # Return patch for preview (you can show in WebUI)
            return jsonify({
                "ok": True,
                "patch_id": patch_id,
                "diff": patch_text,
            }), 200

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            _set_run_state(status="idle", last_error=err)
            return jsonify({"ok": False, "error": err}), 500

    @bp.route("/preview/<patch_id>", methods=["GET"])
    def preview(patch_id: str):
        ok, reason = _require_admin(request)
        if not ok:
            return jsonify({"ok": False, "error": reason}), 401
        try:
            txt = load_pending_patch(patch_id)
            return jsonify({"ok": True, "patch_id": patch_id, "diff": txt}), 200
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 404

    @bp.route("/apply", methods=["POST"])
    def apply():
        ok, reason = _require_admin(request)
        if not ok:
            return jsonify({"ok": False, "error": reason}), 401

        payload = request.get_json(silent=True) or {}
        patch_id = (payload.get("patch_id") or "").strip()
        if not patch_id:
            return jsonify({"ok": False, "error": "Missing 'patch_id'"}), 400

        # Run synchronously (simplest + most reliable)
        # If you later want async, we can fork a subprocess and return job id.
        logs = apply_patch_id_and_deploy(patch_id)
        code = 200 if logs.get("ok") else 500
        return jsonify(logs), code

    return bp


# =============================================================================
# CLI (optional)
# =============================================================================
def cli_ai_plan(request_text: str, out_name: str = "") -> int:
    _ensure_dirs()
    patch_text = ai_generate_patch(request_text)

    if patch_text.lstrip().startswith("ASK:"):
        _print("AI Needs Clarification", patch_text.strip())
        return 0

    ok, why = validate_unified_diff(patch_text)
    if not ok:
        _print("ERROR", f"AI did not return a valid diff: {why}\n\n{patch_text[:2000]}")
        return 1

    patch_id = save_pending_patch(patch_text)
    if out_name:
        # also write to PATCH_DIR for convenience
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.patch", out_name):
            raise ValueError("invalid_output_filename")
        out_path = PATCH_DIR / out_name
        with out_path.open("x", encoding="utf-8") as output:
            output.write(patch_text)

    _print("AI Patch Planned", f"patch_id={patch_id}\nSaved: {PENDING_DIR / (patch_id + '.patch')}")
    return 0


def cli_apply_patch_id(patch_id: str) -> int:
    logs = apply_patch_id_and_deploy(patch_id)
    if logs.get("ok"):
        _print("DONE", "Applied + pushed + built successfully.")
        return 0
    _print("FAILED", json.dumps(logs, indent=2))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ai", dest="ai_request", nargs="+", help='Plan a patch via OpenAI. Example: --ai "Fix theme dropdown z-index"')
    parser.add_argument("--apply-id", dest="apply_id", default=None, help="Apply a pending patch_id (non-interactive).")
    parser.add_argument("--out", dest="out_name", default="", help="Also write AI diff into PATCH_DIR/<out> (optional).")
    parser.add_argument("--print-state", action="store_true", help="Print run state json.")
    args = parser.parse_args()

    if args.print_state:
        _print("RUN STATE", json.dumps(_get_run_state(), indent=2))
        return 0

    if args.ai_request is not None:
        req = " ".join(args.ai_request).strip()
        if not req:
            _print("ERROR", "Empty --ai request.")
            return 1
        return cli_ai_plan(req, out_name=args.out_name or "")

    if args.apply_id:
        _print("REVIEW REQUIRED", "No verified OperatorCore deployment receipt is available for CLI promotion")
        return 1

    _print("INFO", "No CLI action specified. This file is primarily intended to be mounted as a Flask blueprint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryGITtalk.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryGITtalk',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Unknown',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": [],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 40,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryGITtalk.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return __import__("copy").deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Report only observed SML registration state; no fabricated performance scores."""
    return {
        "status": "unverified_runtime", "availability": None,
        "integrity": None, "performance": None, "reliability": None,
        "confidence": 0.0, "latency_ms": None, "stability": None,
        "compatibility": None, "notes": ["SML registration present; runtime health not measured"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryGITtalk',
        "sml_adapter": True,
        "metadata": __import__("copy").deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryGITtalk', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

