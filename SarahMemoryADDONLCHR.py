"""=== SarahMemory Project ===
File: SarahMemoryADDONLCHR.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
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
        self.running_addons = []

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
        self.addon_subdirs = [d for d in os.listdir(addon_base) if os.path.isdir(os.path.join(addon_base, d))]

        if self.dropdown:
            self.dropdown.destroy()

        if self.addon_subdirs:
            self.selected_addon.set(self.addon_subdirs[0])
            display_options = [f"{d} {'(Running)' if self.is_addon_running(d) else ''}" for d in self.addon_subdirs]
            self.dropdown = ttk.OptionMenu(self.addons_window, self.selected_addon, display_options[0], *display_options)
            self.dropdown.pack(pady=10)
        else:
            ttk.Label(self.addons_window, text="No add-ons found.").pack()

    def is_addon_running(self, folder_name):
        for proc, _ in self.running_addons:
            if proc.poll() is None:
                return True
        return False

    def addon1_LOADOPTIONS(self):
        folder = self.selected_addon.get().replace(" (Running)", "").strip()
        addon_path = os.path.join(ADDONS_DIR, folder)
        self.available_py = [f for f in os.listdir(addon_path) if f.endswith(".py")]

        self.launch_window = tk.Toplevel(self.addons_window)
        self.launch_window.title(f"{folder} Add-on Files")
        self.launch_window.geometry("320x200")

        if self.available_py:
            self.selected_file.set(self.available_py[0])
            ttk.Label(self.launch_window, text="Choose a Python file:").pack()
            self.dropdown_file = ttk.OptionMenu(self.launch_window, self.selected_file, self.available_py[0], *self.available_py)
            self.dropdown_file.pack(pady=10)

            self.addon2_button = ttk.Button(self.launch_window, text="LAUNCH", command=self.addon2_LAUNCH)
            self.addon2_button.pack(pady=10)
        else:
            ttk.Label(self.launch_window, text="No .py files found in this folder.").pack()

    def addon2_LAUNCH(self):
        try:
            folder = self.selected_addon.get().replace(" (Running)", "").strip()
            file = self.selected_file.get()
            full_path = os.path.join(ADDONS_DIR, folder, file)

            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")

            python_executable = sys.executable
            log_path = os.path.join(ADDONS_DIR, folder, f"{file}_launch.log")
            logfile = open(log_path, "w")

            proc = subprocess.Popen(
                [python_executable, full_path],
                cwd=os.path.dirname(full_path),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
                stdout=logfile,
                stderr=logfile
            )
            self.running_addons.append((proc, logfile, folder))

            time.sleep(1)
            if proc.poll() is not None:
                logfile.close()
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    crash_output = f.read()
                log_gui_event("Addon Crash Detected", crash_output)
                messagebox.showerror("Add-on Crash", f"{file} crashed immediately.\n\nError Output:\n{crash_output[:500]}")
                return

            log_gui_event("Addon Launch", f"Launched {file} from {folder}")
            messagebox.showinfo("Add-on Launched", f"{file} is now running in its own window.")

        except Exception as e:
            log_gui_event("Addon Launch Failed", str(e))
            messagebox.showerror("Error", f"Could not launch {file}.\n{e}")

    def addon3_SHUTDOWNADDONS(self):
        confirm = messagebox.askyesno("Confirm Shutdown", "Are you sure you want to terminate all running add-ons?")
        if not confirm:
            return
        for item in self.running_addons:
            proc, logfile = (item[0], item[1])
            try:
                proc.terminate()
                logfile.close()
            except Exception:
                pass
        self.running_addons.clear()
        messagebox.showinfo("Shutdown", "All launched add-ons have been shut down.")
        self.refresh_addon_list()

    def close_addons(self):
        if self.launch_window:
            self.launch_window.destroy()
        if self.addons_window:
            self.addons_window.destroy()