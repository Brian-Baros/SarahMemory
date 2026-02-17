"""=== SarahMemory Project ===
File: SarahMemoryUIupdater.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================

SarahMemoryUIupdater.py
SarahMemory Web UI Auto Updater (PythonAnywhere, .env-driven)

Behavior:
    - Loads GITHUB_TOKEN (and optional UI_REPO_URL) from .env
    - Ensures the following directory layout under the SarahMemory project:

        /home/Softdev0/SarahMemory/
            data/
                ui/
                    app.js
                    index.html
                    styles.css
                    V8/          <-- built static UI (served to users)
                    V8_ui_src/   <-- GitHub repo source (sarah-s-dashboard)

    - Clones or updates the sarah-s-dashboard repo into:
        data/ui/V8_ui_src

    - Runs `npm install` (first time) and `npm run build` in V8_ui_src
    - Copies the built `dist/` (or `build/`) contents into:
        data/ui/V8

    - Creates missing folders:
        data/ui, data/ui/V8, data/ui/V8_ui_src
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import argparse

# Try to load .env (if python-dotenv is available)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception as e:
    print(f"[WARN] python-dotenv unavailable or failed, .env not loaded: {e}")
import warnings
warnings.filterwarnings("error", category=SyntaxWarning)

# ---------------------------------------------------------------------------
# CONFIG / DEFAULTS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_UI_DIR = BASE_DIR / "data" / "ui"
SRC_DIR = DATA_UI_DIR / "V8_ui_src"   # where the repo lives & builds
TARGET_DIR = DATA_UI_DIR / "V8"       # where the built UI is deployed

# Load GitHub token from .env (preferred)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Default repo URL if not overridden via env
if GITHUB_TOKEN:
    DEFAULT_REPO_URL = (
        f"https://{GITHUB_TOKEN}@github.com/Brian-Baros/sarah-s-dashboard.git"
    )
else:
    # Works if repo is public; private needs token
    DEFAULT_REPO_URL = "https://github.com/Brian-Baros/sarah-s-dashboard.git"

REPO_URL = os.getenv("UI_REPO_URL", DEFAULT_REPO_URL)

# The repo itself will live directly in SRC_DIR
REPO_DIR = SRC_DIR.resolve()

# Build directories to look for (Vite/Lovable uses dist; CRA uses build)
DEFAULT_BUILD_DIR_NAME = "dist"

# Backups (optional): old UI from V8 can be copied to backups/ if desired
BACKUP_ROOT = DATA_UI_DIR / "backups"


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
# NOTE:
# Some UI repos are "source-only" (need npm build) while others ship pre-built static assets.
# This updater supports BOTH:
#   - If a package.json is present (either at repo root or a nested subdir), we run npm.
#   - If no package.json exists, we skip npm and deploy prebuilt dist/build or raw static files.

_NODE_CWD: Path | None = None

def _detect_node_project_dir() -> Path | None:
    """Best-effort: find the directory that contains package.json.

    Supports repos that keep the React/Vite project in a nested folder.
    """
    global _NODE_CWD
    if _NODE_CWD is not None:
        return _NODE_CWD
    try:
        root_pkg = REPO_DIR / "package.json"
        if root_pkg.exists():
            _NODE_CWD = REPO_DIR
            return _NODE_CWD

        # Search shallowly (avoid crawling node_modules if present)
        max_depth = 4
        for pkg in REPO_DIR.rglob("package.json"):
            try:
                rel = pkg.relative_to(REPO_DIR)
            except Exception:
                continue
            if "node_modules" in rel.parts:
                continue
            if len(rel.parts) <= max_depth:
                _NODE_CWD = pkg.parent
                return _NODE_CWD
    except Exception:
        pass
    _NODE_CWD = None
    return None

def _mask_token(url: str) -> str:
    """Avoid printing raw GitHub tokens in console logs."""
    try:
        if not url:
            return url
        # mask https://TOKEN@github.com/...
        return re.sub(r"(https?://)([^/@:]+)(@github\.com/)", r"\1***\3", url)
    except Exception:
        return url

def run_cmd(cmd: str, cwd: Path | None = None, check: bool = True) -> int:
    """
    Run a shell command and stream output to console.
    Raises CalledProcessError if check=True and command fails.
    """
    print(f"\n[CMD] {cmd} (cwd={cwd})")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def ensure_base_dirs() -> None:
    """
    Ensure data/ui, V8, and V8_ui_src directories exist.
    """
    print(f"[INFO] Ensuring base directories under: {DATA_UI_DIR}")
    DATA_UI_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] DATA_UI_DIR = {DATA_UI_DIR}")
    print(f"[INFO] SRC_DIR     = {SRC_DIR}")
    print(f"[INFO] TARGET_DIR  = {TARGET_DIR}")


def ensure_repo() -> None:
    """
    Ensure the frontend repo exists inside SRC_DIR.
    - If not present, git clone it into SRC_DIR.
    - If present, git pull latest from origin main/master.
    """
    git_dir = REPO_DIR / ".git"

    if not git_dir.exists():
        print(f"[INFO] Frontend repo not found at {REPO_DIR}. Cloning...")
        # If directory exists but isn't a git repo (partial/corrupt), clear it first to allow clone.
        if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
            print(f"[WARN] {REPO_DIR} exists but is not a git repo. Clearing directory before re-clone...")
            for item in REPO_DIR.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"[WARN] Failed to remove {item}: {e}")
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        run_cmd(f"git clone {REPO_URL} {REPO_DIR}")
    else:
        print(f"[INFO] Frontend repo found at {REPO_DIR}. Pulling latest...")
        run_cmd("git fetch --all", cwd=REPO_DIR)
        # Try main first, fallback to master
        try:
            run_cmd("git pull origin main", cwd=REPO_DIR)
        except subprocess.CalledProcessError:
            print("[WARN] 'main' branch pull failed, trying 'master'...")
            run_cmd("git pull origin master", cwd=REPO_DIR)

    print("[INFO] Repository sync complete.")


def ensure_node_modules(skip_install: bool = False) -> None:
    """
    Install npm dependencies if node_modules is missing or incomplete.

    Fix for common headless/PythonAnywhere issues:
      - node_modules may exist but devDependencies (vite) are missing, causing: `vite: not found`
      - npm may have been run previously with production/omit=dev settings
    """
    pkg_json = REPO_DIR / "package.json"
    if not pkg_json.exists():
        print("[INFO] No package.json found; this repo does not appear to be a Node project. Skipping npm install.")
        return

    node_modules = REPO_DIR / "node_modules"
    vite_bin = node_modules / ".bin" / ("vite.cmd" if os.name == "nt" else "vite")

    if skip_install and node_modules.exists() and vite_bin.exists():
        print("[INFO] Skipping npm install (requested) and node_modules looks complete.")
        return

    if node_modules.exists() and not vite_bin.exists():
        print("[WARN] node_modules exists but Vite is missing. Re-installing dependencies with dev deps included...")
        lock = REPO_DIR / "package-lock.json"
        if lock.exists():
            try:
                run_cmd("npm ci --include=dev", cwd=REPO_DIR)
                print("[INFO] npm ci complete.")
                return
            except Exception as e:
                print(f"[WARN] npm ci failed, falling back to npm install: {e}")

        run_cmd("npm install --include=dev", cwd=REPO_DIR)
        print("[INFO] npm install complete.")
        return

    if node_modules.exists():
        print("[INFO] node_modules already exists. Skipping npm install.")
        return

    if skip_install:
        print("[WARN] --skip-install specified but node_modules is missing; build will likely fail.")
        return

    print("[INFO] node_modules missing. Running npm install (this may take a while)...")
    lock = REPO_DIR / "package-lock.json"
    if lock.exists():
        try:
            run_cmd("npm ci --include=dev", cwd=REPO_DIR)
            print("[INFO] npm ci complete.")
            return
        except Exception as e:
            print(f"[WARN] npm ci failed, falling back to npm install: {e}")

    run_cmd("npm install --include=dev", cwd=REPO_DIR)
    print("[INFO] npm install complete.")


def build_frontend(skip_build: bool = False, build_script: str = "build") -> None:
    """
    Run npm run <build_script> (default: build).
    """
    if skip_build:
        print("[INFO] Skipping npm run build (requested).")
        return

    node_cwd = _detect_node_project_dir()
    if node_cwd is None:
        print("[INFO] No package.json detected in repo. Skipping npm build (static UI repo).")
        return

    print(f"[INFO] Running npm run {build_script} in {node_cwd}...")
    run_cmd(f"npm run {build_script}", cwd=node_cwd)
    print("[INFO] Frontend build complete.")


def locate_build_output(custom_build_dir: str | None = None) -> Path:
    """
    Locate the built static output directory (dist or build or custom) inside repo.

    If the repo does not contain a Node project (no package.json), we will:
      1) Prefer a prebuilt dist/ or build/ directory if present
      2) Otherwise treat the repo root as static output if it contains index.html
    """
    candidates: list[Path] = []

    node_cwd = _detect_node_project_dir()
    base = node_cwd if node_cwd is not None else REPO_DIR

    if custom_build_dir:
        candidates.append(base / custom_build_dir)

    # Standard Vite/Lovable output
    candidates.append(base / DEFAULT_BUILD_DIR_NAME)
    # Standard CRA output
    candidates.append(base / "build")

    # Also consider dist/build at repo root (in case base is nested or node_cwd is None)
    if base != REPO_DIR:
        candidates.append(REPO_DIR / DEFAULT_BUILD_DIR_NAME)
        candidates.append(REPO_DIR / "build")

    for c in candidates:
        if c.exists() and c.is_dir():
            print(f"[INFO] Using build output from: {c}")
            return c

    # Static-site fallback: deploy repo root if it looks like a static UI
    static_index = REPO_DIR / "index.html"
    if static_index.exists():
        print(f"[INFO] No dist/build directory found; deploying static files from repo root: {REPO_DIR}")
        return REPO_DIR

    raise FileNotFoundError(
        "Could not find build output directory. "
        "Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + " (or index.html at repo root for static deployment)"
    )


def backup_existing_target(skip_backup: bool = False) -> None:
    """
    Optional: backup current TARGET_DIR contents (if any) into a timestamped folder.
    Currently used but can be disabled via flag.
    """
    if skip_backup:
        print("[INFO] Skipping backup of existing UI (requested).")
        return

    if not TARGET_DIR.exists() or not any(TARGET_DIR.iterdir()):
        print("[INFO] No existing UI in TARGET_DIR to backup.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_dir = BACKUP_ROOT / f"V8_{timestamp}"

    print(f"[INFO] Backing up existing UI from {TARGET_DIR} to {backup_dir}...")
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    shutil.copytree(TARGET_DIR, backup_dir)
    print(f"[INFO] Backup complete: {backup_dir}")


def clear_target_dir() -> None:
    """
    Delete all files and directories in TARGET_DIR (but keep the directory itself).
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Clearing existing contents in {TARGET_DIR}...")
    for item in TARGET_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {item}: {e}")
    print("[INFO] Target directory cleared.")


def deploy_build(build_dir: Path) -> None:
    """
    Copy built files from build_dir into TARGET_DIR.
    """
    print(f"[INFO] Deploying build from {build_dir} to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for item in build_dir.iterdir():
        dest = TARGET_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print("[INFO] Deployment complete.")
    print(f"[INFO] Web UI is now updated at: {TARGET_DIR}")


def clear_src_dir() -> None:
    """
    Optional option to delete all files and directories in SRC_DIR (V8_ui_src) to save space,
    but keep the SRC_DIR folder itself so future runs can re-clone as needed.
    This is especially helpful on constrained environments like PythonAnywhere.
    Notice This OPTION is CURRENTLY DISABLED AND ENABLED USED a '#" on LINE 475
    """
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Clearing source checkout in {SRC_DIR}...")
    for item in SRC_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {item}: {e}")
    print("[INFO] Source directory cleared.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SarahMemory Web UI Updater (using .env, V8/V8_ui_src layout)"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip 'npm install' even if node_modules is missing."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip 'npm run build' (useful if build is done elsewhere)."
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup of existing UI (data/ui/V8)."
    )
    parser.add_argument(
        "--build-script",
        default="build",
        help="Name of npm build script (default: build)."
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Override build output directory name (e.g., 'dist', 'build')."
    )

    args = parser.parse_args()

    print("============================================================")
    print("         SarahMemory AiOS v8.0.0 Web UI Updater             ")
    print("============================================================")
    print(f"[INFO] BASE_DIR    = {BASE_DIR}")
    print(f"[INFO] DATA_UI_DIR = {DATA_UI_DIR}")
    print(f"[INFO] SRC_DIR     = {SRC_DIR}")
    print(f"[INFO] TARGET_DIR  = {TARGET_DIR}")
    print(f"[INFO] REPO_DIR    = {REPO_DIR}")
    print(f"[INFO] REPO_URL    = {_mask_token(REPO_URL)}")
    print("------------------------------------------------------------")

    try:
        ensure_base_dirs()
        ensure_repo()
        ensure_node_modules(skip_install=args.skip_install)
        build_frontend(skip_build=args.skip_build, build_script=args.build_script)
        build_dir = locate_build_output(custom_build_dir=args.build_dir)

        backup_existing_target(skip_backup=args.skip_backup)
        clear_target_dir()
        deploy_build(build_dir)

        # After a successful deployment, reclaim space by wiping the
        # source checkout (V8_ui_src). On the next run the repo will
        # simply should be cloned again if needed but doesn't.
        # 'OPTIONAL' just remove the '#' on Line 475 but if you do you have to manually download the SRC each time
        # you update as of 12/24/2025
        
        #clear_src_dir()

        print("\n[OK] Web UI update finished successfully.")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {e.cmd} (exit code {e.returncode})")
        return e.returncode
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# ====================================================================
# END OF SarahMemoryUIupdater.py v8.0.0
# ====================================================================