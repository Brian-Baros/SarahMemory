#!/usr/bin/env python3
"""
SarahMemoryTree.py (PythonAnywhere-safe)
- Generates a directory tree for ~/SarahMemory
- Writes to tree_YYYYMMDD-HHMMSS.txt (never overwrites to 0 bytes)
- Flushes output as it goes (so partial output survives interrupts)
"""

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
