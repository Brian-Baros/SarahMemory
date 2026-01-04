# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_health_rootdir_selfrepair_patch.py
# Patch: v8.0.0 app.py RootDir + /api/health Self-Repair (Collision-Safe)
"""
Purpose:
- Fix app.py root/data directory resolution bug caused by os.getcwd() under WSGI (PythonAnywhere)
- Ensure /api/health exists and NEVER crashes import-time
- Prevent Flask endpoint collision: "View function mapping is overwriting an existing endpoint function: api_health"
- Minimal, surgical text edits. Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path


def _log(msg: str) -> None:
    try:
        print(f"[v800_app_selfrepair] {msg}")
    except Exception:
        pass


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _write(p: Path, s: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(s, encoding="utf-8")
    os.replace(tmp, p)


def _ensure_import_pathlib(src: str) -> tuple[str, bool]:
    if re.search(r"^\s*from\s+pathlib\s+import\s+Path\s*$", src, re.M):
        return src, False

    m = re.search(r"^(import\s+[^\n]+\n)+", src, re.M)
    if m:
        insert_at = m.end()
        return src[:insert_at] + "from pathlib import Path\n" + src[insert_at:], True

    return "from pathlib import Path\n" + src, True


def _patch_globals_paths_rootdir(src: str) -> tuple[str, bool]:
    """
    In app.py, inside _globals_paths():
      root_dir = os.path.abspath(os.getcwd())
    is wrong under WSGI. Replace with stable BASE_DIR derived from app.py location.
    app.py: <BASE>/api/server/app.py  => parents[2] == <BASE>
    """
    pat = r"root_dir\s*=\s*os\.path\.abspath\(\s*os\.getcwd\(\)\s*\)"
    repl = "root_dir = os.path.abspath(Path(__file__).resolve().parents[2])  # v800 patch: stable BASE_DIR"

    if re.search(pat, src):
        return re.sub(pat, repl, src, count=1), True

    if "Path(__file__).resolve().parents[2]" in src:
        return src, False

    return src, False


def _force_unique_health_endpoint_if_present(src: str) -> tuple[str, bool]:
    """
    If /api/health exists but was declared as:
      @app.route("/api/health")
      def api_health():
    it will collide if *any* other endpoint named api_health exists.
    Fix by changing to:
      @app.route("/api/health", endpoint="sm_api_health")
      def api_health_v800():
    Also, if decorator exists without endpoint=..., add endpoint=...
    Only touches the FIRST /api/health route declaration (the one we care about).
    """
    changed = False

    # Case A: exact collision pattern
    pat = r'@app\.route\(\s*[\'"]\/api\/health[\'"]\s*\)\s*\n\s*def\s+api_health\s*\(\s*\)\s*:'
    repl = '@app.route("/api/health", endpoint="sm_api_health")\ndef api_health_v800():'
    if re.search(pat, src):
        src = re.sub(pat, repl, src, count=1)
        changed = True

    # Case B: decorator exists but without endpoint kw
    if not changed:
        pat2 = r'@app\.route\(\s*[\'"]\/api\/health[\'"]\s*\)'
        if re.search(pat2, src) and 'endpoint="sm_api_health"' not in src:
            src = re.sub(pat2, '@app.route("/api/health", endpoint="sm_api_health")', src, count=1)
            changed = True

    # Case C: ensure function name matches our unique endpoint intent (best effort)
    # If we injected endpoint but function is still def api_health(): then rename function
    if 'endpoint="sm_api_health"' in src and "def api_health_v800" not in src:
        # rename ONLY the first "def api_health():" that follows the /api/health decorator
        pat3 = r'(@app\.route\(\s*[\'"]\/api\/health[\'"]\s*,\s*endpoint\s*=\s*[\'"]sm_api_health[\'"]\s*\)\s*\n)\s*def\s+api_health\s*\(\s*\)\s*:'
        repl3 = r'\1def api_health_v800():'
        if re.search(pat3, src):
            src = re.sub(pat3, repl3, src, count=1)
            changed = True

    return src, changed


def _insert_api_health_route_if_missing(src: str) -> tuple[str, bool]:
    """
    Insert /api/health route if it doesn't exist.
    IMPORTANT: Insert collision-safe endpoint + unique function name.
    """
    if re.search(r"@app\.route\(\s*[\"']\/api\/health[\"']\s*", src):
        return src, False

    health_block = r'''
@app.route("/api/health", endpoint="sm_api_health")
def api_health_v800():
    """
    Health endpoint required by UI + SarahNet rendezvous checks.
    Must NEVER throw; always return JSON.
    """
    try:
        ok, notes, main_running = _perform_health_checks()
        return jsonify({
            "ok": bool(ok),
            "status": "ok" if ok else "warning",
            "notes": notes if isinstance(notes, list) else [],
            "running": True,
            "main_running": bool(main_running),
            "ts": time.time(),
            "version": PROJECT_VERSION,
        })
    except Exception as e:
        try:
            app_logger.error(f"/api/health exception: {e}")
        except Exception:
            pass
        return jsonify({
            "ok": False,
            "status": "error",
            "notes": ["health_exception", str(e)],
            "running": True,
            "main_running": False,
            "ts": time.time(),
            "version": PROJECT_VERSION,
        }), 200
'''.lstrip("\n")

    ping_route = '@app.route("/api/ping")'
    idx = src.find(ping_route)
    if idx != -1:
        after_ping = src.find("\n@app.route(", idx + len(ping_route))
        if after_ping != -1:
            return src[:after_ping] + "\n" + health_block + "\n" + src[after_ping:], True
        return src + "\n\n" + health_block + "\n", True

    m = re.search(r"app\s*=\s*Flask\([^\)]*\)\s*\n", src)
    if m:
        insert_at = m.end()
        return src[:insert_at] + "\n" + health_block + "\n" + src[insert_at:], True

    return src + "\n\n" + health_block + "\n", True


def apply_patch():
    # Locate BASE_DIR from this patch path: <BASE>/data/mods/v800/this_file.py
    patch_file = Path(__file__).resolve()
    base_dir = patch_file.parents[3]
    app_py = base_dir / "api" / "server" / "app.py"

    if not app_py.exists():
        _log(f"FAIL: app.py not found at {app_py}")
        return

    src0 = _read(app_py)
    src = src0

    changed = False
    notes: list[str] = []

    # Ensure Path import
    src, did = _ensure_import_pathlib(src)
    if did:
        changed = True
        notes.append("added pathlib Path import")

    # Patch root_dir drift
    src, did = _patch_globals_paths_rootdir(src)
    if did:
        changed = True
        notes.append("patched _globals_paths root_dir to Path(__file__).parents[2]")

    # FIRST: if /api/health exists, force unique endpoint to avoid import crash
    src, did = _force_unique_health_endpoint_if_present(src)
    if did:
        changed = True
        notes.append("repaired /api/health endpoint collision (unique endpoint + function)")

    # THEN: if /api/health missing, insert collision-safe version
    src, did = _insert_api_health_route_if_missing(src)
    if did:
        changed = True
        notes.append("inserted /api/health route (collision-safe)")

    if not changed:
        _log("No changes needed (already patched).")
        return

    backup = app_py.with_suffix(".py.bak_" + str(int(time.time())))
    try:
        backup.write_text(src0, encoding="utf-8")
        _log(f"Backup created: {backup}")
    except Exception as e:
        _log(f"WARNING: backup failed: {e}")

    _write(app_py, src)
    _log("PATCHED app.py successfully:")
    for n in notes:
        _log(f" - {n}")


# Auto-run on import by mods loader
apply_patch()
