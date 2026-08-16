"""--==The SarahMemory Project==--
File: SarahMemoryTree.py
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

===============================================================================

SarahMemoryTree.py (PythonAnywhere-safe)
- Generates a directory tree for ~/SarahMemory
- Writes to tree_YYYYMMDD-HHMMSS.txt (never overwrites to 0 bytes)
- Flushes output as it goes (so partial output survives interrupts)
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "D"
# ROLE = "utility_tool"
# CATEGORY = "project_inventory"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "standalone_tool"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "tree"
# FAMILY = "utilities"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Project directory tree/report generator that writes timestamped tree outputs safely without overwriting prior output."
# --- SARAHMETA END ---

import os
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode",
    "venv", "env", ".env",
    "node_modules", ".npm", ".cache",
    ".pytest_cache", ".mypy_cache",
}

EXCLUDE_FILES = {".DS_Store"}

def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:,.1f}{unit}"
        num /= 1024.0
    return f"{num:,.1f}PB"

def safe_getsize(path: str) -> str:
    try:
        return sizeof_fmt(os.path.getsize(path))
    except Exception:
        return "?"

def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(ROOT_DIR, f"tree_{ts}.txt")

    total_files = 0
    total_dirs = 0

    print(f"Root: {ROOT_DIR}")
    print(f"Writing: {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("SarahMemory Project Directory Tree\n")
        out.write("=" * 60 + "\n")
        out.write(
            f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
        )
        out.write(f"Root Path: {ROOT_DIR}\n")
        out.write("=" * 60 + "\n\n")
        out.flush()

        for root, dirs, files in os.walk(ROOT_DIR, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            level = root.replace(ROOT_DIR, "").count(os.sep)
            indent = "│   " * level
            folder_name = os.path.basename(root) or root

            out.write(f"{indent}📁 {folder_name}/\n")
            total_dirs += 1

            subindent = "│   " * (level + 1)
            for f in sorted(files):
                if f in EXCLUDE_FILES:
                    continue
                fp = os.path.join(root, f)
                out.write(f"{subindent}📄 {f} ({safe_getsize(fp)})\n")
                total_files += 1

            out.flush()

        out.write("\n" + "=" * 60 + "\n")
        out.write(f"Total Directories: {total_dirs}\n")
        out.write(f"Total Files: {total_files}\n")
        out.write("=" * 60 + "\n")
        out.flush()

    print("✅ Done.")
    print(f"Output: {out_path}")

if __name__ == "__main__":
    main()

# ====================================================================
# END OF SarahMemoryTree.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryTree',
    "version": "v9.0.0-alpha-sml-0.2",
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
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryTree.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryTree',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryTree', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

