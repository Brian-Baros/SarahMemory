#!/usr/bin/env python3

"""--==The SarahMemory Project==--
File: SarahMemoryGITtalk.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
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
# VALIDATION_DATE = "2026-07-11"
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
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
    )


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
    if hdr != SARAH_ADMIN_KEY:
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
    cp = _run(["git", "remote", "get-url", ALLOWED_REMOTE], cwd=REPO_DIR, check=True)
    remote_url = (cp.stdout or "").strip()
    if ALLOWED_REPO_KEYWORD not in remote_url.lower():
        raise RuntimeError(
            f"Safety gate failed: {ALLOWED_REMOTE} URL does not contain '{ALLOWED_REPO_KEYWORD}'.\n"
            f"origin = {remote_url}"
        )
    return remote_url


def _configure_https_remote_if_needed() -> None:
    """
    If using SSH deploy keys: do nothing.
    If using HTTPS+token and you set SARAH_GIT_HTTPS_URL/UI_REPO_URL: re-point origin.
    """
    if not GIT_HTTPS_URL:
        return  # likely SSH origin already configured

    if ALLOWED_REPO_KEYWORD not in GIT_HTTPS_URL.lower():
        raise RuntimeError("Safety gate: SARAH_GIT_HTTPS_URL/UI_REPO_URL is not sarah-s-dashboard.")

    if not GITHUB_TOKEN:
        raise RuntimeError("HTTPS remote requested but no token found in env (GITHUB_TOKEN/GH_TOKEN/GITHUB_PAT).")

    safe_url = GIT_HTTPS_URL.replace("https://", "https://x-access-token:***@")
    token_url = GIT_HTTPS_URL.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@")

    _run(["git", "remote", "set-url", ALLOWED_REMOTE, token_url], cwd=REPO_DIR, check=True)
    _print("Remote configured (HTTPS token)", f"Set {ALLOWED_REMOTE} to {_mask_token(safe_url)}")


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
    """
    Applies a unified diff patch text using `git apply` from stdin.
    """
    _ensure_repo()
    _ensure_correct_remote()
    _configure_https_remote_if_needed()

    p = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=str(REPO_DIR),
        input=patch_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if p.returncode != 0:
        raise RuntimeError(f"git apply failed:\n{p.stdout}")
    return p.stdout or "Patch applied."


def commit_and_push(message: str) -> str:
    _configure_https_remote_if_needed()

    _run(["git", "add", "-A"], cwd=REPO_DIR, check=True)

    # commit (handle nothing to commit)
    commit_out = ""
    try:
        cp_commit = _run(["git", "commit", "-m", message], cwd=REPO_DIR, check=True)
        commit_out = cp_commit.stdout or ""
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "")
        if "nothing to commit" in out.lower():
            commit_out = out
        else:
            raise

    cp_push = _run(["git", "push", ALLOWED_REMOTE, ALLOWED_BRANCH], cwd=REPO_DIR, check=True)
    return (commit_out or "") + "\n" + (cp_push.stdout or "")


def get_head_commit() -> str:
    cp = _run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True)
    return (cp.stdout or "").strip()


# =============================================================================
# Build/deploy
# =============================================================================
def run_ui_updater() -> str:
    if not UI_UPDATER.exists():
        raise RuntimeError(f"SarahMemoryUIupdater.py not found: {UI_UPDATER}")

    cp = _run([sys.executable, str(UI_UPDATER)], cwd=CORE_DIR, check=True)
    return cp.stdout or ""


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
        "store": True,
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
def _new_patch_id() -> str:
    return f"patch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_pending_patch(patch_text: str, patch_id: Optional[str] = None) -> str:
    _ensure_dirs()
    if not patch_id:
        patch_id = _new_patch_id()
    path = PENDING_DIR / f"{patch_id}.patch"
    path.write_text(patch_text, encoding="utf-8")
    return patch_id


def load_pending_patch(patch_id: str) -> str:
    path = PENDING_DIR / f"{patch_id}.patch"
    if not path.exists():
        raise RuntimeError(f"Pending patch not found: {patch_id}")
    return path.read_text(encoding="utf-8", errors="replace")


def delete_pending_patch(patch_id: str) -> None:
    path = PENDING_DIR / f"{patch_id}.patch"
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def validate_unified_diff(patch_text: str) -> Tuple[bool, str]:
    t = (patch_text or "").lstrip()
    if t.startswith("ASK:"):
        return False, "Model requested clarification (ASK: ...)."
    if not t.startswith("diff --git"):
        return False, "Not a unified diff (must start with 'diff --git')."
    # cheap safety: must mention frontend-ish paths; not perfect but reduces nonsense
    if "diff --git" not in t:
        return False, "Missing diff headers."
    return True, ""


# =============================================================================
# Core apply pipeline (non-interactive)
# =============================================================================
def apply_patch_id_and_deploy(patch_id: str) -> dict:
    """
    Non-interactive path used by WebUI endpoint:
    - git pull
    - git apply (patch text)
    - commit
    - PUSH to sarah-s-dashboard
    - run UI updater build/deploy
    """
    _ensure_dirs()
    _set_run_state(status="running", last_error="", last_ok="", last_patch_id=patch_id)

    logs: dict = {
        "patch_id": patch_id,
        "steps": [],
        "ok": False,
        "started_ts": _now_unix(),
        "finished_ts": None,
    }

    def step(name: str, out: str) -> None:
        logs["steps"].append({"name": name, "out": out})

    try:
        _ensure_repo()
        origin = _ensure_correct_remote()
        step("repo_verify", f"REPO_DIR={REPO_DIR}\norigin={_mask_token(origin)}")

        out = git_pull()
        step("git_pull", out)

        patch_text = load_pending_patch(patch_id)
        ok, why = validate_unified_diff(patch_text)
        if not ok:
            raise RuntimeError(f"Patch validation failed: {why}")

        out = apply_patch_text(patch_text)
        step("git_apply", out)

        status = git_status_porcelain()
        step("git_status", status or "(clean)")

        if not status:
            # nothing changed; do not push/build
            logs["ok"] = True
            logs["note"] = "Patch applied cleanly but produced no changes; nothing to commit/push/build."
            _set_run_state(status="idle", last_ok=logs["note"], last_error="")
            logs["finished_ts"] = _now_unix()
            return logs

        msg = f"[sarah-s-dashboard] apply {patch_id}"
        out = commit_and_push(msg)
        step("commit_push", out)

        head = get_head_commit()
        step("head_commit", head)
        _set_run_state(last_commit=head, last_push="ok")

        out = run_ui_updater()
        step("ui_updater", out)
        _set_run_state(last_build="ok")

        logs["ok"] = True
        logs["finished_ts"] = _now_unix()
        _set_run_state(status="idle", last_ok="Deployed successfully.", last_error="")

        # optional: remove pending patch after success
        delete_pending_patch(patch_id)

        return logs

    except Exception as e:
        logs["ok"] = False
        logs["finished_ts"] = _now_unix()
        err = f"{type(e).__name__}: {e}"
        step("error", err)
        _set_run_state(status="idle", last_error=err)
        return logs


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
        (PATCH_DIR / out_name).write_text(patch_text, encoding="utf-8")

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
        return cli_apply_patch_id(args.apply_id.strip())

    _print("INFO", "No CLI action specified. This file is primarily intended to be mounted as a Flask blueprint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryGITtalk.py v9.0.0
# ====================================================================
