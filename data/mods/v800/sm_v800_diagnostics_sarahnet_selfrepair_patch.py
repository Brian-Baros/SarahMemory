"""
--== SarahMemory MonkeyPatch ==--
Patch: sm_v800_diagnostics_sarahnet_selfrepair_patch.py
Target: SarahMemoryDiagnostics.py (menu extension + self-repair routine)
Version: v8.0.0 (v800)
Owner: Brian Lee Baros / SOFTDEV0 LLC

PURPOSE
- Add a "SarahNet / API Self-Repair" option to SarahMemoryDiagnostics.py menu
- Repair directory drift (BAD: ../api/* state dirs ; GOOD: ../data/*)
- Enforce correct BASE_DIR/DATA_DIR/API_DIR at runtime WITHOUT editing SarahMemoryGlobals.py
- Validate sarahnet.comms.json existence and shape
- Attempt safe import of api/server/app.py to capture root traceback for 500 errors
"""

from __future__ import annotations

import os
import json
import time
import shutil
import traceback
import importlib.util
from pathlib import Path


def _now_iso() -> str:
    try:
        import datetime as _dt
        return _dt.datetime.now().isoformat(timespec="seconds")
    except Exception:
        return str(time.time())


def _safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _write_json(path: Path, data: dict) -> None:
    _safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _log(diag_mod, tag: str, msg: str) -> None:
    # Prefer Diagnostics append function if present, otherwise print
    try:
        if hasattr(diag_mod, "append_diag_log"):
            diag_mod.append_diag_log(tag, msg)
        if hasattr(diag_mod, "logger"):
            try:
                diag_mod.logger.info(f"[{tag}] {msg}")
            except Exception:
                pass
    except Exception:
        pass
    try:
        print(f"[{tag}] {msg}")
    except Exception:
        pass


def _force_paths(config_mod, patch_file: Path) -> dict:
    """
    Force correct directory structure based on patch location:
      .../data/mods/v800/<this_file>.py  => BASE_DIR = parents[3]
    """
    base_dir = patch_file.resolve().parents[3]
    data_dir = base_dir / "data"
    api_dir = base_dir / "api"

    # Force common globals used across project
    # NOTE: Some parts of project treat these as strings, some as Paths.
    config_mod.BASE_DIR = os.fspath(base_dir)
    config_mod.DATA_DIR = os.fspath(data_dir)
    config_mod.API_DIR = os.fspath(api_dir)

    # Keep commonly referenced dirs consistent if they exist in config
    try:
        if hasattr(config_mod, "DATASETS_DIR"):
            # Only override if it's clearly wrong or points inside api/
            cur = str(getattr(config_mod, "DATASETS_DIR"))
            if "/api/" in cur.replace("\\", "/"):
                config_mod.DATASETS_DIR = os.fspath(data_dir / "memory" / "datasets")
        else:
            config_mod.DATASETS_DIR = os.fspath(data_dir / "memory" / "datasets")
    except Exception:
        pass

    return {
        "base_dir": os.fspath(base_dir),
        "data_dir": os.fspath(data_dir),
        "api_dir": os.fspath(api_dir),
    }


def _move_tree(src: Path, dst: Path, notes: list[str]) -> None:
    """
    Move a directory tree into destination.
    If dst exists, merge contents.
    """
    if not src.exists():
        return

    _safe_mkdir(dst)

    # Merge children
    for item in src.iterdir():
        target = dst / item.name
        try:
            if item.is_dir():
                _move_tree(item, target, notes)
                # remove empty dir after merge
                try:
                    if item.exists() and not any(item.iterdir()):
                        item.rmdir()
                except Exception:
                    pass
            else:
                if target.exists():
                    # Avoid overwrite; keep newest with suffix
                    suffix = f".migrated_{int(time.time())}"
                    target2 = target.with_name(target.name + suffix)
                    shutil.move(os.fspath(item), os.fspath(target2))
                    notes.append(f"MERGE: {item} -> {target2} (dst existed)")
                else:
                    shutil.move(os.fspath(item), os.fspath(target))
                    notes.append(f"MOVE: {item} -> {target}")
        except Exception as e:
            notes.append(f"ERROR moving {item} -> {target}: {e}")

    # Try remove src if empty
    try:
        if src.exists() and src.is_dir() and not any(src.iterdir()):
            src.rmdir()
    except Exception:
        pass


def _repair_api_orphans(config_mod, diag_mod) -> dict:
    """
    Ensure nothing like api/wallets or api/settings exists.
    Move those into data/*.
    """
    base_dir = Path(getattr(config_mod, "BASE_DIR", Path.cwd()))
    data_dir = Path(getattr(config_mod, "DATA_DIR", base_dir / "data"))
    api_dir = Path(getattr(config_mod, "API_DIR", base_dir / "api"))

    notes: list[str] = []
    repaired = []
    found = []

    # Anything that belongs in data/, not api/
    orphan_dirs = [
        "addons", "documents", "logs", "memory", "meta", "settings",
        "wallet", "wallets", "mods", "sandbox"
    ]
    orphan_files = ["server_state.json", "meta.db", "server_meta.db"]

    # Migrate directories
    for d in orphan_dirs:
        src = api_dir / d
        if src.exists() and src.is_dir():
            found.append(os.fspath(src))
            dst = data_dir / d
            _log(diag_mod, "SELF_REPAIR", f"Orphan dir detected: {src}  ->  {dst}")
            _move_tree(src, dst, notes)
            repaired.append(f"dir:{d}")

    # Migrate known orphan files
    for fn in orphan_files:
        srcf = api_dir / fn
        if srcf.exists() and srcf.is_file():
            found.append(os.fspath(srcf))
            dstf = data_dir / fn
            _safe_mkdir(dstf.parent)
            try:
                if dstf.exists():
                    dst2 = dstf.with_name(dstf.name + f".migrated_{int(time.time())}")
                    shutil.move(os.fspath(srcf), os.fspath(dst2))
                    notes.append(f"MOVEFILE: {srcf} -> {dst2} (dst existed)")
                else:
                    shutil.move(os.fspath(srcf), os.fspath(dstf))
                    notes.append(f"MOVEFILE: {srcf} -> {dstf}")
                repaired.append(f"file:{fn}")
            except Exception as e:
                notes.append(f"ERROR moving file {srcf} -> {dstf}: {e}")

    return {
        "ok": True,
        "api_dir": os.fspath(api_dir),
        "data_dir": os.fspath(data_dir),
        "found": found,
        "repaired": repaired,
        "notes": notes,
    }


def _ensure_sarahnet_comms(config_mod, diag_mod) -> dict:
    base_dir = Path(getattr(config_mod, "BASE_DIR", Path.cwd()))
    data_dir = Path(getattr(config_mod, "DATA_DIR", base_dir / "data"))
    comms_path = data_dir / "settings" / "sarahnet.comms.json"

    default_doc = {
        "mode": "CLOUD_LAN",  # CLOUD_ONLY | LAN_ONLY | CLOUD_LAN | OFF
        "cloud": {
            "enabled": True,
            "rendezvous_base": "https://api.sarahmemory.com",
            "health_path": "/api/health",
            "timeout_ms": 2500,
            "presence_ttl_sec": 600
        },
        "lan": {
            "enabled": True,
            "auto_discovery": True
        },
        "fallback": {
            "on_no_internet": True,
            "fallback_to_lan": True,
            "fallback_to_off": True
        },
        "last_resolved": {
            "effective_mode": "OFF",
            "reason": "not_resolved_yet",
            "ts": None
        },
        "persona": {
            "screen_name": "SarahMemory_AI",
            "avatar_name": "SarahMemory Node",
            "status": "Online",
            "discoverable": True,
            "allow_inbound_from": "TRUSTED_ONLY"
        }
    }

    created = False
    fixed = False
    notes: list[str] = []

    if not comms_path.exists():
        _write_json(comms_path, default_doc)
        created = True
        _log(diag_mod, "SELF_REPAIR", f"Created missing {comms_path}")
    else:
        # Validate minimal keys; patch forward-compat without breaking user config
        try:
            raw = json.loads(comms_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("sarahnet.comms.json is not a JSON object")

            # Ensure keys exist
            for k, v in default_doc.items():
                if k not in raw:
                    raw[k] = v
                    fixed = True
                    notes.append(f"Added missing key: {k}")

            # Ensure mode is sane
            valid_modes = {"CLOUD_ONLY", "LAN_ONLY", "CLOUD_LAN", "OFF"}
            m = str(raw.get("mode", "CLOUD_LAN")).upper().strip()
            if m not in valid_modes:
                raw["mode"] = "CLOUD_LAN"
                fixed = True
                notes.append(f"Normalized invalid mode -> CLOUD_LAN (was {m!r})")
            else:
                raw["mode"] = m

            if fixed:
                _write_json(comms_path, raw)
                _log(diag_mod, "SELF_REPAIR", f"Repaired structure of {comms_path}")
        except Exception as e:
            # If JSON is corrupt, back it up and restore defaults
            try:
                backup = comms_path.with_suffix(".json.bad_" + str(int(time.time())))
                shutil.copy2(os.fspath(comms_path), os.fspath(backup))
                notes.append(f"Backed up corrupt comms to {backup}")
            except Exception:
                pass
            _write_json(comms_path, default_doc)
            fixed = True
            notes.append(f"Reset comms file due to parse error: {e}")

    return {
        "ok": True,
        "path": os.fspath(comms_path),
        "created": created,
        "fixed": fixed,
        "notes": notes,
    }


def _safe_import_app_py(config_mod, diag_mod) -> dict:
    """
    Try importing api/server/app.py in isolation (via file loader) to capture
    the real exception that is causing PythonAnywhere 500.
    """
    base_dir = Path(getattr(config_mod, "BASE_DIR", Path.cwd()))
    api_dir = Path(getattr(config_mod, "API_DIR", base_dir / "api"))
    app_path = api_dir / "server" / "app.py"
    if not app_path.exists():
        return {"ok": False, "error": f"Missing app.py at {app_path}"}

    try:
        spec = importlib.util.spec_from_file_location("SM_API_SERVER_APP_DIAG", os.fspath(app_path))
        if spec is None or spec.loader is None:
            return {"ok": False, "error": "spec_from_file_location failed"}
        mod = importlib.util.module_from_spec(spec)
        # Execute module
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return {"ok": True, "app_path": os.fspath(app_path), "result": "import_ok"}
    except Exception as e:
        tb = traceback.format_exc()
        _log(diag_mod, "API_500_TRACE", f"Import failed for {app_path}: {e}\n{tb}")
        return {"ok": False, "app_path": os.fspath(app_path), "error": str(e), "traceback": tb}


def run_sarahnet_api_self_repair() -> dict:
    """
    Public entry point used by Diagnostics menu extension.
    """
    import SarahMemoryGlobals as config
    import SarahMemoryDiagnostics as SMD

    patch_file = Path(__file__).resolve()

    report: dict = {
        "ok": True,
        "ts": _now_iso(),
        "steps": [],
        "paths": {},
        "orphans": {},
        "comms": {},
        "app_import": {},
    }

    try:
        report["paths"] = _force_paths(config, patch_file)
        report["steps"].append("forced_paths_ok")
        _log(SMD, "SELF_REPAIR", f"Forced paths: {report['paths']}")
    except Exception as e:
        report["ok"] = False
        report["steps"].append("forced_paths_failed")
        _log(SMD, "SELF_REPAIR", f"Failed forcing paths: {e}")

    try:
        report["orphans"] = _repair_api_orphans(config, SMD)
        report["steps"].append("repair_api_orphans_ok")
    except Exception as e:
        report["ok"] = False
        report["steps"].append("repair_api_orphans_failed")
        _log(SMD, "SELF_REPAIR", f"Orphan repair failed: {e}")

    try:
        report["comms"] = _ensure_sarahnet_comms(config, SMD)
        report["steps"].append("ensure_comms_ok")
    except Exception as e:
        report["ok"] = False
        report["steps"].append("ensure_comms_failed")
        _log(SMD, "SELF_REPAIR", f"Comms repair failed: {e}")

    try:
        report["app_import"] = _safe_import_app_py(config, SMD)
        report["steps"].append("safe_import_app_py_done")
        if not report["app_import"].get("ok"):
            report["ok"] = False
    except Exception as e:
        report["ok"] = False
        report["steps"].append("safe_import_app_py_failed")
        _log(SMD, "SELF_REPAIR", f"Safe import attempt crashed: {e}")

    # Persist a self-repair report in logs for SarahMemoryEvolution to harvest
    try:
        data_dir = Path(getattr(config, "DATA_DIR", Path(getattr(config, "BASE_DIR", Path.cwd())) / "data"))
        out = data_dir / "reports" / "v800" / "selfrepair_last.json"
        _write_json(out, report)
        _log(SMD, "SELF_REPAIR", f"Wrote self-repair report: {out}")
    except Exception:
        pass

    return report


def apply():
    """
    Monkeypatch entrypoint expected by SarahMemory patch loader.
    This wraps the Diagnostics menu and adds option 10.
    """
    import SarahMemoryDiagnostics as SMD

    if getattr(SMD, "_SM_V800_SARAHNET_SELFREPAIR_PATCHED", False):
        return

    original_menu = getattr(SMD, "diagnostics_menu", None)

    def patched_menu() -> None:
        # If original menu isn't present, just run self-repair directly.
        if not callable(original_menu):
            print("Diagnostics menu not found; running SarahNet/API self-repair directly.")
            run_sarahnet_api_self_repair()
            return

        while True:
            print("\n===========================")
            print(" SarahMemory Diagnostics")
            print("===========================")
            print("1) Run Full System Diagnostics")
            print("2) Database Diagnostics")
            print("3) UI Diagnostics")
            print("4) API Diagnostics")
            print("5) Network Diagnostics")
            print("6) Hardware Diagnostics")
            print("7) Cloud Sync Diagnostics")
            print("8) Security Diagnostics")
            print("9) Run & Log FULL Diagnostics")
            print("10) SarahNet / API Self-Repair (Fix paths + capture 500 traceback)")
            print("0) Exit")

            choice = input("Select option: ").strip()
            if choice == "0":
                print("Exiting diagnostics.")
                break

            # Let original menu handle 1-9 exactly as before by temporarily
            # calling its logic (we re-dispatch to keep behavior identical).
            if choice in {str(i) for i in range(1, 10)}:
                # Call the original menu but "simulate" by invoking the known functions
                # (copying original behavior avoids recursion).
                if choice == "1":
                    print("\n[Full] Running core self-check + system + network + API.")
                    base = SMD.run_self_check()
                    _ = base
                    sys_report = SMD.run_system_diagnostics()
                    net_report = SMD.run_network_diagnostics()
                    api_report = SMD.run_api_diagnostics()
                    SMD._print_system_report(sys_report)
                    SMD._print_network_report(net_report)
                    SMD._print_api_report(api_report)
                elif choice == "2":
                    print("\n[DB] Running database diagnostics.")
                    diag = SMD.DatabaseDiagnosticsSuperTask()
                    db_report = diag.run()
                    SMD._print_database_report(db_report)
                elif choice == "3":
                    print("\n[UI] Running WebUI bridge + UI/JS diagnostics.")
                    w_report = SMD.run_webui_bridge_diagnostics()
                    SMD._print_webui_bridge_report(w_report)
                    ui_report = SMD.run_ui_diagnostics()
                    SMD._print_ui_report(ui_report)
                elif choice == "4":
                    print("\n[API] Running API diagnostics.")
                    api_report = SMD.run_api_diagnostics()
                    SMD._print_api_report(api_report)
                elif choice == "5":
                    print("\n[NET] Running network diagnostics.")
                    net_report = SMD.run_network_diagnostics()
                    SMD._print_network_report(net_report)
                elif choice == "6":
                    print("\n[HW] Running hardware diagnostics.")
                    hw_report = SMD.run_hardware_diagnostics()
                    SMD._print_hardware_report(hw_report)
                elif choice == "7":
                    print("\n[SYNC] Running cloud sync diagnostics.")
                    sync_report = SMD.run_sync_diagnostics()
                    SMD._print_sync_report(sync_report)
                elif choice == "8":
                    print("\n[SEC] Running security diagnostics.")
                    sec_report = SMD.run_security_diagnostics()
                    SMD._print_security_report(sec_report)
                elif choice == "9":
                    print("\n[FULL+LOG] Running full diagnostics and logging summary.")
                    _ = SMD.run_full_diagnostics_suite(write_aggregate_log=True)
                continue

            if choice == "10":
                print("\n[SELF-REPAIR] Running SarahNet/API self-repair now...")
                rep = run_sarahnet_api_self_repair()
                print("\n=== Self-Repair Summary ===")
                print(json.dumps(rep, indent=2))
                continue

            print("Invalid choice, please select 0-10.")

    # Patch in
    SMD.diagnostics_menu = patched_menu
    SMD.run_sarahnet_api_self_repair = run_sarahnet_api_self_repair  # optional external call
    SMD._SM_V800_SARAHNET_SELFREPAIR_PATCHED = True


# Auto-apply on import (safe)
try:
    apply()
except Exception:
    pass
