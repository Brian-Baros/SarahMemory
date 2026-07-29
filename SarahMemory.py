"""--==The SarahMemory Project==--
File: SarahMemory.py
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

===============================================================================
SarahMemory v9.0.0 Root Launcher
Thin outside trigger for the SarahMemory AiOS core runtime.
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "root_launcher"
# CATEGORY = "launch_trigger"
# USER_FACING = True
# UI_EXPOSURE = "root_launcher"
# DEPLOYMENT_TARGET = "root"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem_process"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "sarahmemory_root_launcher"
# FAMILY = "boot"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Thin cross-platform ignition script. Resolves the project root, enters ./core, and executes SarahMemoryMain.py without owning runtime logic."
# --- SARAHMETA END ---


import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


APP_NAME = "SarahMemory"
CORE_DIR_NAME = "core"
CORE_ENTRY_FILE = "SarahMemoryMain.py"


def _project_root() -> Path:
    """Return the directory containing this root launcher."""
    return Path(__file__).resolve().parent


def _candidate_venv_python(root: Path) -> Path | None:
    """Return a local virtual-environment Python executable when present."""
    if os.name == "nt":
        candidate = root / "venv" / "Scripts" / "python.exe"
    else:
        candidate = root / "venv" / "bin" / "python"

    if candidate.is_file():
        return candidate
    return None


def _select_python(root: Path) -> str:
    """
    Select the Python interpreter for launching the core runtime.

    Preference order:
      1. Project-local venv Python, if present.
      2. Current interpreter, preserving the active environment.
    """
    venv_python = _candidate_venv_python(root)
    if venv_python is not None:
        return str(venv_python)
    return sys.executable


def _build_environment(root: Path, core_dir: Path) -> dict[str, str]:
    """Build a small launch environment without mutating global process state."""
    env = os.environ.copy()
    env["SARAHMEMORY_ROOT"] = str(root)
    env["SARAHMEMORY_CORE"] = str(core_dir)
    env["SARAHMEMORY_LAUNCHER"] = "SarahMemory.py"

    existing_pythonpath = env.get("PYTHONPATH", "")
    path_parts = [str(core_dir), str(root)]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return env


def launch_core(args: Sequence[str] | None = None) -> int:
    """Launch ./core/SarahMemoryMain.py and return its exit code."""
    root = _project_root()
    core_dir = root / CORE_DIR_NAME
    core_entry = core_dir / CORE_ENTRY_FILE

    if not core_dir.is_dir():
        print(f"[{APP_NAME} Launcher] ERROR: Missing core directory: {core_dir}", file=sys.stderr)
        return 2

    if not core_entry.is_file():
        print(f"[{APP_NAME} Launcher] ERROR: Missing core entry file: {core_entry}", file=sys.stderr)
        return 3

    python_exe = _select_python(root)
    env = _build_environment(root, core_dir)
    forwarded_args = list(args or [])

    command = [python_exe, str(core_entry), *forwarded_args]

    try:
        return subprocess.call(command, cwd=str(core_dir), env=env)
    except KeyboardInterrupt:
        print(f"\n[{APP_NAME} Launcher] Interrupted by user.", file=sys.stderr)
        return 130
    except FileNotFoundError:
        print(f"[{APP_NAME} Launcher] ERROR: Python executable not found: {python_exe}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"[{APP_NAME} Launcher] ERROR: Failed to launch core runtime: {exc}", file=sys.stderr)
        return 5


def main() -> int:
    """Root launcher entrypoint."""
    return launch_core(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemory.py v9.0.0
# ====================================================================
