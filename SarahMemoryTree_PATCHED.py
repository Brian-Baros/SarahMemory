"""--==The SarahMemory Project==--
File: SarahMemoryTree_PATCHED.py
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

SarahMemoryTree.py (PythonAnywhere-safe)
- Generates focused directory inventory reports for SarahMemory.
- Default mode focuses on driver bodies:
./data/boot/drivers
./data/drivers
- Excludes noisy/heavy trees such as venv, __pycache__, node_modules, models,
backups, docs, ui, generated dist/build output, cache folders, and logs.
- Writes timestamped tree reports without overwriting prior output.
- Flushes output as it goes so partial output survives interrupts.

Usage:
python SarahMemoryTree.py
python SarahMemoryTree.py --mode drivers
python SarahMemoryTree.py --mode project
python SarahMemoryTree.py --include-ui
python SarahMemoryTree.py --max-depth 6
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
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Focused project directory tree/report generator. Defaults to driver inventory under data/boot/drivers and data/drivers while excluding noisy/heavy runtime trees."
# --- SARAHMETA END ---

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parent

# Directory names excluded anywhere in the tree.
EXCLUDE_DIR_NAMES: Set[str] = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".vs",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    ".npm",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "coverage",
    ".coverage",
    "htmlcov",
    "logs",
    "log",
    "tmp",
    "temp",
    "cache",
    "caches",
    "backups",
    "backup",
    "archive",
    "archives",
    "documents",
    "docs",
    "doc",
    "documentation",
    "models",
    "model",
    "weights",
    "checkpoints",
    "checkpoint",
    "datasets",
    "dataset",
    "downloads",
    "download",
}

# Relative paths excluded from a full project tree.
# Keep data/boot/drivers and data/drivers visible.
EXCLUDE_REL_PATHS: Set[str] = {
    "ui",
    "data/models",
    "data/model",
    "data/backups",
    "data/backup",
    "data/archive",
    "data/archives",
    "data/documents",
    "data/docs",
    "data/downloads",
    "data/memory/datasets",
    "public_html/web/static",
    "public_html/web/assets",
}

EXCLUDE_FILE_NAMES: Set[str] = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

EXCLUDE_FILE_SUFFIXES: Set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".temp",
    ".bak",
    ".old",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".pt",
    ".pth",
    ".onnx",
    ".safetensors",
    ".bin",
    ".gguf",
    ".ckpt",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".mp3",
    ".flac",
}

DRIVER_FOCUS_PATHS: Tuple[str, ...] = (
    "data/boot/drivers",
    "data/drivers",
)


def _normalize_rel(path: Path) -> str:
    """Return a portable lowercase relative path from ROOT_DIR."""
    try:
        rel = path.resolve().relative_to(ROOT_DIR)
    except Exception:
        try:
            rel = path.relative_to(ROOT_DIR)
        except Exception:
            rel = path
    return rel.as_posix().strip("/").lower()


def _is_relative_path_excluded(path: Path, include_ui: bool = False) -> bool:
    rel = _normalize_rel(path)
    if include_ui and rel == "ui":
        return False
    if include_ui and rel.startswith("ui/"):
        return False

    for blocked in EXCLUDE_REL_PATHS:
        blocked_norm = blocked.strip("/").lower()
        if rel == blocked_norm or rel.startswith(blocked_norm + "/"):
            return True
    return False


def _is_dir_excluded(path: Path, include_ui: bool = False) -> bool:
    name = path.name
    if name in EXCLUDE_DIR_NAMES:
        return True
    return _is_relative_path_excluded(path, include_ui=include_ui)


def _is_file_excluded(path: Path) -> bool:
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDE_FILE_SUFFIXES:
        return True
    # Exclude generated reports to keep future reports clean.
    if path.name.startswith("tree_") and path.suffix.lower() == ".txt":
        return True
    return False


def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:,.1f}{unit}"
        num /= 1024.0
    return f"{num:,.1f}PB"


def safe_getsize(path: Path) -> str:
    try:
        return sizeof_fmt(path.stat().st_size)
    except Exception:
        return "?"


def _safe_listdir(path: Path) -> Tuple[List[Path], List[Path]]:
    try:
        children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        return [], []

    dirs = [p for p in children if p.is_dir() and not p.is_symlink()]
    files = [p for p in children if p.is_file()]
    return dirs, files


def _write_tree(
    out,
    start_path: Path,
    *,
    root_label: Optional[str] = None,
    include_ui: bool = False,
    max_depth: Optional[int] = None,
    start_depth: int = 0,
) -> Tuple[int, int, int]:
    """Write a filtered tree rooted at start_path.

    Returns:
        (directory_count, file_count, skipped_count)
    """
    total_dirs = 0
    total_files = 0
    skipped = 0

    if not start_path.exists():
        out.write(f"⚠️  Missing: {start_path}\n")
        out.flush()
        return total_dirs, total_files, skipped + 1

    root_label = root_label or start_path.name
    base_depth = len(start_path.parts)

    for root, dirs, files in os.walk(start_path, topdown=True, followlinks=False):
        root_path = Path(root)

        rel_depth = max(0, len(root_path.parts) - base_depth)
        if max_depth is not None and rel_depth > max_depth:
            dirs[:] = []
            skipped += 1
            continue

        filtered_dirs = []
        for d in sorted(dirs, key=str.lower):
            dpath = root_path / d
            if _is_dir_excluded(dpath, include_ui=include_ui):
                skipped += 1
                continue
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs

        level = start_depth + rel_depth
        indent = "│   " * level

        if rel_depth == 0:
            folder_name = root_label
        else:
            folder_name = root_path.name

        out.write(f"{indent}📁 {folder_name}/\n")
        total_dirs += 1

        subindent = "│   " * (level + 1)
        for f in sorted(files, key=str.lower):
            fp = root_path / f
            if _is_file_excluded(fp):
                skipped += 1
                continue
            out.write(f"{subindent}📄 {f} ({safe_getsize(fp)})\n")
            total_files += 1

        out.flush()

    return total_dirs, total_files, skipped


def _driver_focus_roots() -> List[Path]:
    return [ROOT_DIR / rel for rel in DRIVER_FOCUS_PATHS]


def _project_roots() -> List[Path]:
    return [ROOT_DIR]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a filtered SarahMemory directory tree report.",
    )
    parser.add_argument(
        "--mode",
        choices=("drivers", "project"),
        default="drivers",
        help="drivers = only data/boot/drivers and data/drivers. project = filtered whole project.",
    )
    parser.add_argument(
        "--include-ui",
        action="store_true",
        help="Include ui when using --mode project. Default excludes ui.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional max depth below each scanned root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to SarahMemory root.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"tree_{args.mode}_{ts}.txt"

    roots = _driver_focus_roots() if args.mode == "drivers" else _project_roots()

    total_files = 0
    total_dirs = 0
    total_skipped = 0

    print(f"Root: {ROOT_DIR}")
    print(f"Mode: {args.mode}")
    print(f"Writing: {out_path}")

    with out_path.open("w", encoding="utf-8") as out:
        out.write("SarahMemory Project Directory Tree\n")
        out.write("=" * 72 + "\n")
        out.write(f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}\n")
        out.write(f"Root Path: {ROOT_DIR}\n")
        out.write(f"Mode: {args.mode}\n")
        out.write(f"Include UI: {bool(args.include_ui)}\n")
        out.write(f"Max Depth: {args.max_depth if args.max_depth is not None else 'unlimited'}\n")
        out.write("Excluded: venv, __pycache__, node_modules, models, backups, docs, logs, caches, ui by default\n")
        out.write("=" * 72 + "\n\n")
        out.flush()

        if args.mode == "drivers":
            out.write("Focused Driver Inventory\n")
            out.write("- data/boot/drivers\n")
            out.write("- data/drivers\n")
            out.write("\n")
            out.flush()

        for idx, root in enumerate(roots):
            if idx:
                out.write("\n")
            label = _normalize_rel(root) if root != ROOT_DIR else ROOT_DIR.name
            dirs, files, skipped = _write_tree(
                out,
                root,
                root_label=label,
                include_ui=bool(args.include_ui),
                max_depth=args.max_depth,
            )
            total_dirs += dirs
            total_files += files
            total_skipped += skipped

        out.write("\n" + "=" * 72 + "\n")
        out.write(f"Total Directories Listed: {total_dirs}\n")
        out.write(f"Total Files Listed: {total_files}\n")
        out.write(f"Skipped Entries: {total_skipped}\n")
        out.write("=" * 72 + "\n")
        out.flush()

    print("✅ Done.")
    print(f"Output: {out_path}")
    print(f"Directories listed: {total_dirs}")
    print(f"Files listed: {total_files}")
    print(f"Skipped entries: {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryTree_PATCHED.py v9.0.0
# ====================================================================
