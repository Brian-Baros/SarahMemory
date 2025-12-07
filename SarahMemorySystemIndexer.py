"""--==The SarahMemory Project==--
File: SarahMemorySystemIndexer.py
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
"""

import os
import sqlite3
import time
import datetime
import hashlib
import psutil
import winreg
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

# Import global configuration flags where available
try:
    import SarahMemoryGlobals as config  # type: ignore
    from SarahMemoryGlobals import SAFE_MODE, LOCAL_ONLY_MODE, is_offline  # type: ignore
except Exception:
    config = None  # Fallback if globals not available
    SAFE_MODE = False
    LOCAL_ONLY_MODE = False
    def is_offline() -> bool:
        return False

FILE_EXTENSIONS = {
    '.doc', '.docx', '.pdf', '.txt', '.jpg', '.png',
    '.mp3', '.wav', '.mp4', '.md', '.csv', '.json', '.xml',
    '.yaml', '.yml', '.odt', '.jpeg', '.bmp', '.tif', '.tiff',
    '.ogg', '.flac', '.m4a', '.mkv', '.mov', '.avi',
    '.py', '.ipynb', '.js', '.html', '.css', '.php',
    '.asp', '.bat', '.sh', '.sql'
}
REGISTRY_ROOTS = [
    (winreg.HKEY_CURRENT_USER, "Software"),
    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE")
]
MAX_REGISTRY_DEPTH = 3
# Dynamically derive the path to the system index database using SarahMemoryGlobals when available.
if config:
    DB_PATH = os.path.join(config.DATASETS_DIR, "system_index.db")
else:
    # Fallback to the hard-coded default when config is not available
    DB_PATH = os.path.join("c:\\SarahMemory", "data", "memory", "datasets", "system_index.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            file_type TEXT,
            file_size INTEGER,
            modified TEXT,
            sha256 TEXT,
            indexed_at TEXT,
    tagged_topic TEXT DEFAULT NULL        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS registry_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_key TEXT,
            key_path TEXT,
            value_name TEXT,
            value_data TEXT,
            value_type TEXT,
            indexed_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_file_record(file_path, file_type, file_size, modified, sha256):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO file_index
        (file_path, file_type, file_size, modified, sha256, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_path, file_type, file_size, modified, sha256, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def registry_entry_already_indexed(root_key, key_path, value_name, value_data):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM registry_index WHERE root_key = ? AND key_path = ? AND value_name = ? AND value_data = ?",
                (root_key, key_path, value_name, str(value_data)))
    exists = cur.fetchone()
    conn.close()
    return exists is not None

def insert_registry_record(root_key, key_path, value_name, value_data, value_type):
    if registry_entry_already_indexed(root_key, key_path, value_name, value_data):
        print(f"[Skipped] Already indexed: {key_path} - {value_name}")
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO registry_index
        (root_key, key_path, value_name, value_data, value_type, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (root_key, key_path, value_name, str(value_data), value_type, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def compute_sha256(filepath, block_size=65536):
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        print(f"[Error] Could not compute SHA256 for {filepath}: {e}")
        return None

def file_already_indexed(file_path, sha256):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM file_index WHERE file_path = ? AND sha256 = ?", (file_path, sha256))
    exists = cur.fetchone()
    conn.close()
    return exists is not None

def scan_filesystem_gui(selected_drive, selected_ext):
    print("[Indexer] Starting GUI-based file scan...")
    drives = [selected_drive] if selected_drive != "ALL" else [part.mountpoint for part in psutil.disk_partitions(all=False) if "cdrom" not in part.opts and part.fstype]
    extensions = [selected_ext] if selected_ext != "ALL" else list(FILE_EXTENSIONS)

    for mount_point in drives:
        for root, dirs, files in os.walk(mount_point):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, file)
                    try:
                        stat = os.stat(full_path)
                        file_size = stat.st_size
                        modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                        file_hash = compute_sha256(full_path)
                        if not file_already_indexed(full_path, file_hash):
                            insert_file_record(full_path, ext, file_size, modified, file_hash)
                            print(f"[File Indexed] {full_path}")
                        else:
                            print(f"[Skipped] Already indexed: {full_path}")
                    except Exception as e:
                        print(f"[Error] Indexing file {full_path}: {e}")
    print("[Indexer] File system scan complete.")

def scan_registry():
    print("[Indexer] Starting registry scan...")
    # Scan original registry hives
    for root, base_path in REGISTRY_ROOTS:
        try:
            scan_registry_key(root, base_path, depth=0)
        except Exception as e:
            print(f"[Error] Scanning registry root {base_path}: {e}")

    # NEW: Scan for executable app paths from App Paths key
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") as app_paths:
            for i in range(0, winreg.QueryInfoKey(app_paths)[0]):
                subkey = winreg.EnumKey(app_paths, i)
                subkey_path = f"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{subkey}"
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey_path) as sub:
                        try:
                            value, _ = winreg.QueryValueEx(sub, None)
                            insert_registry_record("HKLM", subkey_path, "exe_path", value, "AppPath")
                        except FileNotFoundError:
                            pass
                except Exception as e:
                    print(f"[AppPaths] Failed subkey {subkey_path}: {e}")
    except Exception as e:
        print(f"[AppPaths] Could not scan App Paths: {e}")

    # NEW: Scan Uninstall keys for friendly app names
    uninstall_keys = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
    ]
    for root, path in uninstall_keys:
        try:
            with winreg.OpenKey(root, path) as uninstall_root:
                for i in range(winreg.QueryInfoKey(uninstall_root)[0]):
                    subkey = winreg.EnumKey(uninstall_root, i)
                    subkey_path = f"{path}\{subkey}"
                    try:
                        with winreg.OpenKey(root, subkey_path) as app_key:
                            try:
                                display_name, _ = winreg.QueryValueEx(app_key, "DisplayName")
                                display_icon, _ = winreg.QueryValueEx(app_key, "DisplayIcon")
                                insert_registry_record(str(root), subkey_path, "friendly_name", display_name, "string")
                                insert_registry_record(str(root), subkey_path, "exe_path", display_icon, "string")
                            except FileNotFoundError:
                                continue
                    except Exception as e:
                        print(f"[Uninstall] Failed key {subkey_path}: {e}")
        except Exception as e:
            print(f"[Uninstall] Failed path {path}: {e}")

    print("[Indexer] Registry scan complete.")

def scan_registry_key(root, key_path, depth=0):
    if depth > MAX_REGISTRY_DEPTH:
        return
    if any(bad in key_path.lower() for bad in [".vob", ".ifo", ".bup", "video_ts"]):
        return
    try:
        with winreg.OpenKey(root, key_path) as key:
            for i in range(winreg.QueryInfoKey(key)[1]):
                try:
                    value_name, value_data, value_type = winreg.EnumValue(key, i)
                    insert_registry_record(str(root), key_path, value_name, value_data, str(value_type))
                except Exception as e:
                    print(f"[Error] Reading value in {key_path}: {e}")
            for j in range(winreg.QueryInfoKey(key)[0]):
                try:
                    subkey_name = winreg.EnumKey(key, j)
                    scan_registry_key(root, os.path.join(key_path, subkey_name), depth + 1)
                except Exception as e:
                    print(f"[Error] Subkey in {key_path}: {e}")
    except Exception as e:
        print(f"[Error] Opening key {key_path}: {e}")

def launch_indexer_gui():
    root = tk.Tk()
    root.title("SarahMemory Indexer v7.0 by Brian Baros")
    root.geometry("450x250")

    tk.Label(root, text="Select Drive:").pack(pady=5)
    drives = ["ALL"] + [part.mountpoint for part in psutil.disk_partitions(all=False) if "cdrom" not in part.opts and part.fstype]
    drive_var = tk.StringVar(value=drives[0])
    ttk.Combobox(root, textvariable=drive_var, values=drives).pack()

    tk.Label(root, text="Select File Type:").pack(pady=5)
    exts = ["ALL"] + sorted(FILE_EXTENSIONS)
    ext_var = tk.StringVar(value=exts[0])
    ttk.Combobox(root, textvariable=ext_var, values=exts).pack()

    registry_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Include Registry Scan", variable=registry_var).pack(pady=5)

    def start_indexing():
        """
        Kick off the indexing process. Respect SAFE_MODE and LOCAL_ONLY_MODE flags so
        that heavy indexing does not occur during safe or local-only operation.
        """
        # If the system is in safe mode or local-only mode, skip indexing entirely
        try:
            if SAFE_MODE or LOCAL_ONLY_MODE:
                messagebox.showwarning(
                    "Indexing Disabled",
                    "System indexing is disabled while SAFE_MODE or LOCAL_ONLY_MODE is enabled."
                )
                return
        except Exception:
            # If flags are missing, proceed normally
            pass
        init_db()
        scan_filesystem_gui(drive_var.get(), ext_var.get())
        if registry_var.get():
            scan_registry()
        messagebox.showinfo("Indexing Complete", "System Indexing is complete!")

    tk.Button(root, text="START INDEXING", command=start_indexing).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    launch_indexer_gui()