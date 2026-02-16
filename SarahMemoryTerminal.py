"""--== SarahMemory Project ==--
File: SarahMemoryTerminal.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-01-28
Author: © 2025 Brian Lee Baros. All Rights Reserved.
https://www.sarahmemory.com
===============================================================================

PURPOSE:
- Enterprise-grade Developer Terminal execution service (server-side).
- HARD GATED by DEVELOPERSMODE (SarahMemoryGlobals.py OR env var).
- Cross-platform:
    - Windows commands via cmd.exe (default on Windows)
    - Bash commands via /bin/bash on Linux/macOS
    - Bash on Windows via WSL (wsl.exe) when available
- NO UI here. This module is a backend capability provider for WebUI.

SECURITY MODEL:
- Disabled unless DEVELOPERSMODE == True.
- Default sandboxing:
    - Working directory scoped to BASE_DIR (or BASE_DIR/data by default)
    - Optional allowlist/denylist controls
    - Timeouts, output caps, and audit logging
- This is a developer tool. Keep it OFF for end-users.

===============================================================================
"""

from __future__ import annotations

import os
import json
import time
import shlex
import sqlite3
import logging
import platform
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import SarahMemoryGlobals as config

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryTerminal")
logger.setLevel(logging.DEBUG)
_null = logging.NullHandler()
_null.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_null)

# -----------------------------------------------------------------------------
# Developer mode gate
# -----------------------------------------------------------------------------
_DEVMODE_CACHE: Optional[bool] = None


def developers_mode_enabled() -> bool:
    """
    Gate reads config.DEVELOPERSMODE first, then env var.
    """
    global _DEVMODE_CACHE
    if _DEVMODE_CACHE is not None:
        return bool(_DEVMODE_CACHE)

    v = getattr(config, "DEVELOPERSMODE", None)
    if v is None:
        v = os.getenv("DEVELOPERSMODE", None)

    if isinstance(v, bool):
        _DEVMODE_CACHE = v
        return bool(_DEVMODE_CACHE)

    s = str(v or "").strip().lower()
    _DEVMODE_CACHE = s in ("1", "true", "yes", "on", "enabled")
    return bool(_DEVMODE_CACHE)


# -----------------------------------------------------------------------------
# Paths + logging (portable)
# -----------------------------------------------------------------------------
def _datasets_dir() -> str:
    try:
        return getattr(
            config,
            "DATASETS_DIR",
            os.path.join(getattr(config, "DATA_DIR", os.getcwd()), "memory", "datasets"),
        )
    except Exception:
        return os.path.join(os.getcwd(), "data", "memory", "datasets")


def _system_logs_db() -> str:
    return os.path.join(_datasets_dir(), "system_logs.db")


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def _ensure_tables() -> None:
    con: Optional[sqlite3.Connection] = None
    try:
        con = _connect(_system_logs_db())
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS terminal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                severity TEXT,
                event TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as e:
        logger.debug("Terminal DB ensure failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def log_terminal_event(
    event: str,
    details: str,
    *,
    severity: str = "INFO",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _ensure_tables()
        con = _connect(_system_logs_db())
        cur = con.cursor()
        ts = datetime.now().isoformat()
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False)
        except Exception:
            meta_json = "{}"
        cur.execute(
            "INSERT INTO terminal_events (ts, severity, event, details, meta_json) VALUES (?, ?, ?, ?, ?)",
            (ts, str(severity), str(event), str(details), meta_json),
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.debug("Failed to log terminal event: %s", e)


# -----------------------------------------------------------------------------
# Session management (in-memory, TTL)
# -----------------------------------------------------------------------------
_SESS_LOCK = threading.RLock()
_SESS_TTL_S = 60 * 60 * 2  # 2 hours
_SESS_MAX = 64
_SESS: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return float(time.time())


def _prune_sessions() -> None:
    now = _now()
    with _SESS_LOCK:
        # TTL prune
        dead = []
        for sid, rec in _SESS.items():
            ts = float(rec.get("last_epoch", rec.get("created_epoch", 0.0)) or 0.0)
            if ts and (now - ts) > _SESS_TTL_S:
                dead.append(sid)
        for sid in dead:
            _SESS.pop(sid, None)

        # size prune oldest first
        if len(_SESS) > _SESS_MAX:
            items = sorted(_SESS.items(), key=lambda kv: float(kv[1].get("last_epoch", 0.0) or 0.0))
            for sid, _ in items[: max(0, len(_SESS) - _SESS_MAX)]:
                _SESS.pop(sid, None)


def get_or_create_session(session_id: Optional[str], *, base_workdir: str) -> str:
    _prune_sessions()
    sid = (session_id or "").strip()

    with _SESS_LOCK:
        if sid and sid in _SESS:
            _SESS[sid]["last_epoch"] = _now()
            return sid

        # Create new session
        sid = sid if sid else _new_session_id()
        _SESS[sid] = {
            "id": sid,
            "created_epoch": _now(),
            "last_epoch": _now(),
            "cwd": base_workdir,
            "env": {},
        }
        return sid


def _new_session_id() -> str:
    # avoid uuid import to keep module light
    return f"term_{int(_now() * 1000)}_{os.getpid()}"


def get_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    _prune_sessions()
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _SESS_LOCK:
        rec = _SESS.get(sid)
        return dict(rec) if isinstance(rec, dict) else None


def update_session_cwd(session_id: str, cwd: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    with _SESS_LOCK:
        if sid in _SESS:
            _SESS[sid]["cwd"] = cwd
            _SESS[sid]["last_epoch"] = _now()


# -----------------------------------------------------------------------------
# Safety controls (enterprise guardrails)
# -----------------------------------------------------------------------------
def _base_dir() -> str:
    return str(getattr(config, "BASE_DIR", os.getcwd()) or os.getcwd())


def _default_workdir() -> str:
    # keep it inside BASE_DIR by default
    bd = _base_dir()
    wd = os.path.join(bd, "data")
    return wd if os.path.isdir(wd) else bd


def _realpath(p: str) -> str:
    return os.path.realpath(os.path.abspath(p))


def _is_within_base_dir(path: str) -> bool:
    bd = _realpath(_base_dir())
    rp = _realpath(path)
    try:
        return os.path.commonpath([bd, rp]) == bd
    except Exception:
        return False


def _sanitize_workdir(workdir: Optional[str]) -> str:
    wd = (workdir or "").strip()
    if not wd:
        wd = _default_workdir()
    # If user tries to escape BASE_DIR, clamp
    if not _is_within_base_dir(wd):
        wd = _default_workdir()
    os.makedirs(wd, exist_ok=True)
    return wd


# Hard denylist (minimize catastrophic operator error)
_DENY_PATTERNS = [
    # destructive disk/system actions (high blast radius)
    r"\brm\s+-rf\s+/\b",
    r"\bmkfs(\.|_)?\b",
    r"\bdd\s+if=\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bformat\s+[a-zA-Z]:\b",
    r"\bdiskpart\b",
    r"\bdel\s+/s\b",
    r"\brd\s+/s\b",
    # escalation / persistence patterns (tighten as needed)
    r"\bsudo\b",
]


def _matches_denylist(cmd: str) -> Optional[str]:
    import re
    t = (cmd or "").strip().lower()
    for pat in _DENY_PATTERNS:
        try:
            if re.search(pat, t, flags=re.IGNORECASE):
                return pat
        except Exception:
            continue
    return None


# -----------------------------------------------------------------------------
# Execution backends
# -----------------------------------------------------------------------------
def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _wsl_available() -> bool:
    if not _is_windows():
        return False
    try:
        p = subprocess.run(["wsl.exe", "--status"], capture_output=True, text=True, timeout=3)
        return p.returncode == 0
    except Exception:
        return False


def _build_command(mode: str, command: str) -> Tuple[list, str]:
    """
    Returns (argv, engine_label).
    mode: auto | windows | bash | powershell
    """
    cmd = (command or "").strip()
    m = (mode or "auto").strip().lower()

    if m == "auto":
        if _is_windows():
            return (["cmd.exe", "/c", cmd], "cmd")
        return (["/bin/bash", "-lc", cmd], "bash")

    if m == "windows":
        return (["cmd.exe", "/c", cmd], "cmd")

    if m == "powershell":
        # keep it explicit; no profile load
        return (["powershell.exe", "-NoProfile", "-Command", cmd], "powershell")

    if m == "bash":
        if _is_windows():
            if _wsl_available():
                # -e runs command directly; wrap with bash -lc inside WSL for consistent behavior
                return (["wsl.exe", "bash", "-lc", cmd], "wsl-bash")
            # fallback: block
            return ([], "bash-unavailable")
        return (["/bin/bash", "-lc", cmd], "bash")

    # Unknown -> auto
    if _is_windows():
        return (["cmd.exe", "/c", cmd], "cmd")
    return (["/bin/bash", "-lc", cmd], "bash")


def _cap_text(s: str, limit: int) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...<output_truncated>..."


def execute_terminal_command(
    *,
    command: str,
    mode: str = "auto",
    session_id: Optional[str] = None,
    workdir: Optional[str] = None,
    timeout_s: int = 12,
    max_output_chars: int = 20000,
    caller: str = "unknown",
) -> Dict[str, Any]:
    """
    Executes a command in a constrained, developer-mode-only context.

    Returns:
        {
          ok: bool,
          blocked: bool,
          reason: str | None,
          session_id: str,
          engine: "cmd"|"bash"|"wsl-bash"|...,
          cwd: str,
          exit_code: int,
          stdout: str,
          stderr: str,
          duration_ms: int,
          ts: iso
        }
    """
    ts = datetime.now().isoformat()

    if not developers_mode_enabled():
        return {
            "ok": False,
            "blocked": True,
            "reason": "DEVELOPERSMODE is OFF; terminal is disabled.",
            "session_id": session_id or "",
            "engine": None,
            "cwd": None,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_ms": 0,
            "ts": ts,
        }

    cmd = (command or "").strip()
    if not cmd:
        return {
            "ok": False,
            "blocked": True,
            "reason": "Empty command.",
            "session_id": session_id or "",
            "engine": None,
            "cwd": None,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_ms": 0,
            "ts": ts,
        }

    deny_hit = _matches_denylist(cmd)
    if deny_hit:
        log_terminal_event(
            "TerminalBlocked",
            "Command blocked by denylist.",
            severity="WARN",
            meta={"caller": caller, "mode": mode, "deny_pattern": deny_hit, "command": cmd[:500]},
        )
        return {
            "ok": False,
            "blocked": True,
            "reason": f"Command blocked by policy (denylist match: {deny_hit}).",
            "session_id": session_id or "",
            "engine": None,
            "cwd": None,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_ms": 0,
            "ts": ts,
        }

    base_wd = _sanitize_workdir(workdir)
    sid = get_or_create_session(session_id, base_workdir=base_wd)
    state = get_session_state(sid) or {}
    cwd = _sanitize_workdir(state.get("cwd") or base_wd)

    argv, engine = _build_command(mode, cmd)
    if not argv:
        return {
            "ok": False,
            "blocked": True,
            "reason": "Requested shell backend unavailable (bash on Windows requires WSL).",
            "session_id": sid,
            "engine": engine,
            "cwd": cwd,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_ms": 0,
            "ts": ts,
        }

    t0 = time.time()
    try:
        # NOTE: shell=False by design; we pass through the chosen shell executable explicitly.
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
            shell=False,
        )
        duration_ms = int((time.time() - t0) * 1000)

        stdout = _cap_text(proc.stdout or "", int(max_output_chars))
        stderr = _cap_text(proc.stderr or "", int(max_output_chars))

        # Heuristic: allow simple 'cd <path>' style session cwd updates
        # (cmd/bash have different semantics; treat as best-effort UX)
        _maybe_update_cwd_from_command(sid, cmd, cwd)

        log_terminal_event(
            "TerminalExecuted",
            "Command executed.",
            severity="INFO",
            meta={
                "caller": caller,
                "mode": mode,
                "engine": engine,
                "cwd": cwd,
                "exit_code": proc.returncode,
                "duration_ms": duration_ms,
                "command": cmd[:800],
            },
        )

        return {
            "ok": True,
            "blocked": False,
            "reason": None,
            "session_id": sid,
            "engine": engine,
            "cwd": cwd,
            "exit_code": int(proc.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
            "ts": ts,
        }

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - t0) * 1000)
        log_terminal_event(
            "TerminalTimeout",
            "Command timed out.",
            severity="WARN",
            meta={"caller": caller, "mode": mode, "engine": engine, "cwd": cwd, "duration_ms": duration_ms, "command": cmd[:800]},
        )
        return {
            "ok": False,
            "blocked": False,
            "reason": f"Command timed out after {timeout_s}s.",
            "session_id": sid,
            "engine": engine,
            "cwd": cwd,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Timeout after {timeout_s}s",
            "duration_ms": duration_ms,
            "ts": ts,
        }
    except Exception as e:
        duration_ms = int((time.time() - t0) * 1000)
        log_terminal_event(
            "TerminalError",
            "Command execution error.",
            severity="ERROR",
            meta={"caller": caller, "mode": mode, "engine": engine, "cwd": cwd, "duration_ms": duration_ms, "error": str(e), "command": cmd[:800]},
        )
        return {
            "ok": False,
            "blocked": False,
            "reason": "Execution error.",
            "session_id": sid,
            "engine": engine,
            "cwd": cwd,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration_ms": duration_ms,
            "ts": ts,
        }


def _maybe_update_cwd_from_command(session_id: str, cmd: str, current_cwd: str) -> None:
    """
    Best-effort: interpret 'cd <path>' and clamp within BASE_DIR.
    """
    t = (cmd or "").strip()
    if not t:
        return

    low = t.lower().strip()

    # bash style: cd path
    if low.startswith("cd "):
        target = t[3:].strip().strip('"').strip("'")
        _apply_cwd_update(session_id, target, current_cwd)
        return

    # cmd style: cd /d path OR cd path
    if low.startswith("cd"):
        parts = shlex.split(t, posix=False)
        if len(parts) >= 2:
            # drop '/d' if present
            rest = [p for p in parts[1:] if p.lower() != "/d"]
            if rest:
                target = " ".join(rest).strip().strip('"').strip("'")
                _apply_cwd_update(session_id, target, current_cwd)


def _apply_cwd_update(session_id: str, target: str, current_cwd: str) -> None:
    if not target:
        return

    # resolve relative path
    if not os.path.isabs(target):
        candidate = os.path.join(current_cwd, target)
    else:
        candidate = target

    candidate = _realpath(candidate)

    if _is_within_base_dir(candidate) and os.path.isdir(candidate):
        update_session_cwd(session_id, candidate)


# -----------------------------------------------------------------------------
# Flask adapter helper (optional)
# -----------------------------------------------------------------------------
def terminal_api_execute(payload: Dict[str, Any], *, caller: str = "api") -> Dict[str, Any]:
    """
    Thin adapter for a Flask route:
      POST /api/terminal/execute
      body: { command, mode, session_id, workdir, timeout_s, max_output_chars }

    HARD GATED by DEVELOPERSMODE (always).
    """
    payload = payload or {}
    return execute_terminal_command(
        command=str(payload.get("command") or ""),
        mode=str(payload.get("mode") or "auto"),
        session_id=payload.get("session_id"),
        workdir=payload.get("workdir"),
        timeout_s=int(payload.get("timeout_s") or 12),
        max_output_chars=int(payload.get("max_output_chars") or 20000),
        caller=str(payload.get("caller") or caller),
    )
