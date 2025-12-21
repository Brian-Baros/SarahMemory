"""=== SarahMemory Project ===
File: SarahMemoryADDONLCHR.py
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

ADDON LAUNCHER ENGINE
======================================================

PURPOSE:
--------
This module serves as the launching platform for MODS and Additional Modules that an END-UUSER may want to add
to the SarahMemory Platform, The Ai should be able to Learn and adapt to other Modules and 
to user preferences, patterns, timing, and operational habits. Making SarahMemory the dynamic functional foundation.
"""

import os
import sys
import subprocess
import time
import tkinter as tk
from tkinter import ttk, messagebox
import SarahMemoryGlobals as config
from SarahMemoryGlobals import BASE_DIR, ADDONS_DIR, log_gui_event

class AddonLauncher:
    def __init__(self, parent):
        self.parent = parent
        self.selected_addon = tk.StringVar()
        self.selected_file = tk.StringVar()
        self.addons_window = None
        self.launch_window = None
        self.running_addons = []  # list of tuples: (Popen, logfile, folder)
        self.loaded_manifest_addons = {}  # addon_id -> {"manifest":..., "hooks":..., "context":..., "mode":...}

    def open_addons(self):
        self.addons_window = tk.Toplevel(self.parent)
        self.addons_window.title("Add-ons")
        self.addons_window.geometry("400x350")
        self.addons_window.protocol("WM_DELETE_WINDOW", self.close_addons)

        self.refresh_button = ttk.Button(self.addons_window, text="Refresh List", command=self.refresh_addon_list)
        self.refresh_button.pack(pady=5)

        self.addon_dropdown_label = ttk.Label(self.addons_window, text="Select Add-on Folder")
        self.addon_dropdown_label.pack()

        self.dropdown = None
        self.refresh_addon_list()

        self.addon1_button = ttk.Button(self.addons_window, text="LOAD", command=self.addon1_LOADOPTIONS)
        self.addon1_button.pack(pady=10)

        self.addon3_button = ttk.Button(self.addons_window, text="Shutdown all Add-ons", command=self.addon3_SHUTDOWNADDONS)
        self.addon3_button.pack(pady=10)

    def refresh_addon_list(self):
        addon_base = ADDONS_DIR
        os.makedirs(addon_base, exist_ok=True)

        self.addon_subdirs = [
            d for d in os.listdir(addon_base)
            if os.path.isdir(os.path.join(addon_base, d))
        ]

        if self.dropdown:
            try:
                self.dropdown.destroy()
            except Exception:
                pass
            self.dropdown = None

        if not self.addon_subdirs:
            ttk.Label(self.addons_window, text="No add-ons found.").pack()
            return

        # Build display names with status tags (Manifest / Loaded / Running)
        display_options = []
        for d in self.addon_subdirs:
            tags = []
            if self._has_manifest(os.path.join(addon_base, d)):
                tags.append("Manifest")
            if self._is_manifest_loaded(d):
                tags.append("Loaded")
            if self.is_addon_running(d):
                tags.append("Running")
            suffix = f" ({', '.join(tags)})" if tags else ""
            display_options.append(f"{d}{suffix}")

        # Default selection
        self.selected_addon.set(display_options[0])

        self.dropdown = ttk.OptionMenu(
            self.addons_window,
            self.selected_addon,
            display_options[0],
            *display_options
        )
        self.dropdown.pack(pady=10)

    def is_addon_running(self, folder_name):
        # Only counts subprocess-launched addons as "Running"
        for item in list(self.running_addons or []):
            try:
                proc = item[0]
                folder = item[2] if len(item) > 2 else None
                if folder and folder != folder_name:
                    continue
                if proc and proc.poll() is None:
                    return True
            except Exception:
                continue
        return False

    def addon1_LOADOPTIONS(self):
        folder = self._addon_folder_from_display(self.selected_addon.get())
        addon_path = os.path.join(ADDONS_DIR, folder)

        self.available_py = []
        try:
            self.available_py = [f for f in os.listdir(addon_path) if f.endswith(".py")]
        except Exception:
            self.available_py = []

        has_manifest = self._has_manifest(addon_path)

        self.launch_window = tk.Toplevel(self.addons_window)
        self.launch_window.title(f"{folder} Add-on Options")
        self.launch_window.geometry("360x240")

        # Manifest-first workflow (Option 3: Hybrid)
        if has_manifest:
            ttk.Label(self.launch_window, text="Manifest detected (manifest.json).").pack(pady=6)
            ttk.Button(self.launch_window, text="LAUNCH (Manifest Add-on)", command=self.addon2_LAUNCH).pack(pady=6)

            # Optional legacy runner inside same folder
            if self.available_py:
                ttk.Separator(self.launch_window, orient="horizontal").pack(fill="x", pady=6)
                ttk.Label(self.launch_window, text="Legacy scripts in this folder:").pack()
                self.selected_file.set(self.available_py[0])
                self.dropdown_file = ttk.OptionMenu(self.launch_window, self.selected_file, self.available_py[0], *self.available_py)
                self.dropdown_file.pack(pady=8)
                ttk.Button(self.launch_window, text="LAUNCH (Legacy Script)", command=self._launch_selected_legacy).pack(pady=6)
            return

        # Legacy-only workflow (no manifest.json)
        ttk.Label(self.launch_window, text="Choose a Python file:").pack(pady=6)
        if self.available_py:
            self.selected_file.set(self.available_py[0])
            self.dropdown_file = ttk.OptionMenu(self.launch_window, self.selected_file, self.available_py[0], *self.available_py)
            self.dropdown_file.pack(pady=10)

            self.addon2_button = ttk.Button(self.launch_window, text="LAUNCH", command=self.addon2_LAUNCH)
            self.addon2_button.pack(pady=10)
        else:
            ttk.Label(self.launch_window, text="No .py files found in this folder.").pack(pady=10)

    def addon2_LAUNCH(self):
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            addon_root = os.path.join(ADDONS_DIR, folder)

            # If this folder has a manifest.json, treat it as a package add-on
            if self._has_manifest(addon_root):
                self._launch_manifest_addon(addon_root)
                self.refresh_addon_list()
                return

            # Legacy script launch path
            file = (self.selected_file.get() or "").strip()
            if not file:
                raise ValueError("No add-on file selected.")
            full_path = os.path.join(addon_root, file)
            self._launch_legacy_script(full_path, folder=folder, display_name=file)
            self.refresh_addon_list()

        except Exception as e:
            log_gui_event("Addon Launch Failed", str(e))
            messagebox.showerror("Error", f"Could not launch add-on.\n{e}")

    def addon3_SHUTDOWNADDONS(self):
        confirm = messagebox.askyesno("Confirm Shutdown", "Are you sure you want to terminate all running add-ons?")
        if not confirm:
            return

        # Stop subprocess-launched addons
        for item in list(self.running_addons or []):
            try:
                proc = item[0]
                logfile = item[1] if len(item) > 1 else None
                if proc and proc.poll() is None:
                    proc.terminate()
                if logfile:
                    try:
                        logfile.close()
                    except Exception:
                        pass
            except Exception:
                pass
        self.running_addons.clear()

        # Stop in-process manifest addons (best effort)
        for addon_id, rec in list(getattr(self, "loaded_manifest_addons", {}).items()):
            try:
                hooks = rec.get("hooks") or {}
                shutdown_fn = hooks.get("shutdown")
                if callable(shutdown_fn):
                    shutdown_fn(rec.get("context") or {})
            except Exception:
                pass
        try:
            getattr(self, "loaded_manifest_addons", {}).clear()
        except Exception:
            pass

        messagebox.showinfo("Shutdown", "All launched add-ons have been shut down.")
        self.refresh_addon_list()

    

    # ------------------------------------------------------------------
    # Hybrid Add-on Support (Option 3) - Manifest + Legacy
    # ------------------------------------------------------------------

    def _addon_folder_from_display(self, display_value: str) -> str:
        """Normalize dropdown label into folder name."""
        s = (display_value or "").strip()
        if " (" in s:
            s = s.split(" (", 1)[0].strip()
        return s

    def _has_manifest(self, addon_root: str) -> bool:
        try:
            return os.path.isfile(os.path.join(addon_root, "manifest.json"))
        except Exception:
            return False

    def _is_manifest_loaded(self, folder_name: str) -> bool:
        try:
            addon_root = os.path.join(ADDONS_DIR, folder_name)
            man = self._read_manifest(addon_root)
            if not man:
                return False
            addon_id = man.get("addon_id") or folder_name
            return addon_id in (self.loaded_manifest_addons or {})
        except Exception:
            return False

    def _read_manifest(self, addon_root: str):
        import json
        try:
            path = os.path.join(addon_root, "manifest.json")
            if not os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return json.load(f)
        except Exception as e:
            log_gui_event("Addon Manifest Read Failed", str(e))
            return None

    def _addon_exec_mode_from_manifest(self, manifest: dict) -> str:
        """
        Decide execution mode.
        Hybrid policy:
          - If SAFE_MODE: force subprocess (safer containment)
          - If manifest sets execution.mode: honor (inprocess/subprocess/auto)
          - Else: default auto => subprocess unless explicitly trusted
        """
        try:
            safe_mode = bool(getattr(config, "SAFE_MODE", False))
        except Exception:
            safe_mode = False
        if safe_mode:
            return "subprocess"

        exec_cfg = (manifest or {}).get("execution") or {}
        mode = (exec_cfg.get("mode") or "auto").strip().lower()
        if mode in ("inprocess", "subprocess"):
            return mode

        # Auto mode: in-process only if marked trusted
        trusted = False
        try:
            trusted = bool(((manifest or {}).get("security") or {}).get("trusted", False))
        except Exception:
            trusted = False
        return "inprocess" if trusted else "subprocess"

    def _launch_selected_legacy(self):
        """Helper for manifest folders that also contain legacy scripts."""
        try:
            folder = self._addon_folder_from_display(self.selected_addon.get())
            file = (self.selected_file.get() or "").strip()
            if not file:
                raise ValueError("No legacy file selected.")
            full_path = os.path.join(ADDONS_DIR, folder, file)
            self._launch_legacy_script(full_path, folder=folder, display_name=file)
            self.refresh_addon_list()
        except Exception as e:
            log_gui_event("Legacy Addon Launch Failed", str(e))
            messagebox.showerror("Error", f"Could not launch legacy add-on.\n{e}")

    def _launch_legacy_script(self, full_path: str, folder: str, display_name: str = ""):
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        python_executable = sys.executable
        log_path = os.path.join(ADDONS_DIR, folder, f"{os.path.basename(full_path)}_launch.log")
        logfile = open(log_path, "w", encoding="utf-8", errors="ignore")

        proc = subprocess.Popen(
            [python_executable, full_path],
            cwd=os.path.dirname(full_path),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            stdout=logfile,
            stderr=logfile
        )
        self.running_addons.append((proc, logfile, folder))

        time.sleep(0.5)
        if proc.poll() is not None:
            try:
                logfile.close()
            except Exception:
                pass
            crash_output = ""
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    crash_output = f.read()
            except Exception:
                pass
            log_gui_event("Addon Crash Detected", crash_output or "Unknown crash output")
            messagebox.showerror("Add-on Crash", f"{display_name or full_path} crashed immediately.\n\nError Output:\n{(crash_output or '')[:500]}")
            return

        log_gui_event("Addon Launch", f"Launched {display_name or os.path.basename(full_path)} from {folder}")
        messagebox.showinfo("Add-on Launched", f"{display_name or os.path.basename(full_path)} is now running.")

    def _launch_manifest_addon(self, addon_root: str):
        manifest = self._read_manifest(addon_root)
        if not manifest:
            raise ValueError("manifest.json missing or invalid.")

        mode = self._addon_exec_mode_from_manifest(manifest)
        if mode == "inprocess":
            self._launch_manifest_inprocess(addon_root, manifest)
        else:
            self._launch_manifest_subprocess(addon_root, manifest)

    def _launch_manifest_inprocess(self, addon_root: str, manifest: dict):
        entry = (manifest or {}).get("entrypoint") or {}
        module_name = (entry.get("module") or "").strip()
        callable_name = (entry.get("callable") or "").strip()
        if not module_name or not callable_name:
            raise ValueError("Invalid manifest entrypoint (module/callable missing).")

        if addon_root not in sys.path:
            sys.path.insert(0, addon_root)

        mod = __import__(module_name)
        fn = getattr(mod, callable_name, None)
        if not callable(fn):
            raise ValueError(f"Entrypoint callable not found: {module_name}.{callable_name}")

        addon_id = (manifest.get("addon_id") or os.path.basename(addon_root)).strip()

        context = {
            "platform_version": getattr(config, "PROJECT_VERSION", "8.0.0"),
            "addon_path": addon_root,
            "permissions": (manifest.get("permissions") or {}),
            "run_mode": getattr(config, "RUN_MODE", "local"),
            "device_mode": getattr(config, "DEVICE_MODE", "local_agent"),
        }

        hooks = fn(context) or {}
        self.loaded_manifest_addons[addon_id] = {
            "manifest": manifest,
            "hooks": hooks,
            "context": context,
            "mode": "inprocess",
        }

        log_gui_event("Addon Loaded", f"{manifest.get('name', addon_id)} v{manifest.get('version','')}")
        messagebox.showinfo("Add-on Loaded", f"{manifest.get('name', addon_id)} is now active (in-process).")

    def _launch_manifest_subprocess(self, addon_root: str, manifest: dict):
        entry = (manifest or {}).get("entrypoint") or {}
        module_name = (entry.get("module") or "").strip()
        callable_name = (entry.get("callable") or "").strip()
        if not module_name or not callable_name:
            raise ValueError("Invalid manifest entrypoint (module/callable missing).")

        addon_id = (manifest.get("addon_id") or os.path.basename(addon_root)).strip()
        folder = os.path.basename(addon_root)

        python_executable = sys.executable
        log_path = os.path.join(addon_root, "manifest_launch.log")
        logfile = open(log_path, "w", encoding="utf-8", errors="ignore")

        ctx = {
            "platform_version": getattr(config, "PROJECT_VERSION", "8.0.0"),
            "addon_path": addon_root,
            "permissions": (manifest.get("permissions") or {}),
            "run_mode": getattr(config, "RUN_MODE", "local"),
            "device_mode": getattr(config, "DEVICE_MODE", "local_agent"),
        }
        import json
        ctx_json = json.dumps(ctx)

        code = (
            "import json,sys; "
            "sys.path.insert(0, r\"%s\"); " % addon_root.replace('\\', '\\\\') +
            f"m=__import__('{module_name}'); "
            f"fn=getattr(m,'{callable_name}',None); "
            "ctx=json.loads(sys.argv[1]); "
            "fn(ctx) if callable(fn) else None"
        )

        proc = subprocess.Popen(
            [python_executable, "-c", code, ctx_json],
            cwd=addon_root,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            stdout=logfile,
            stderr=logfile
        )
        self.running_addons.append((proc, logfile, folder))

        log_gui_event("Addon Launch", f"{manifest.get('name', addon_id)} launched (subprocess)")
        messagebox.showinfo("Add-on Launched", f"{manifest.get('name', addon_id)} is now running (subprocess).")
def close_addons(self):
        if self.launch_window:
            self.launch_window.destroy()
        if self.addons_window:
            self.addons_window.destroy()

# ====================================================================
# END OF SarahMemoryADDONLCHR.py v8.0.0
# ====================================================================