
"""--== SarahMemory Project ==--
File: SarahMemoryCleanup.py
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
"""

import os, sqlite3, shutil, time, traceback
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk  # optional for icons
except Exception:
    Image = ImageTk = None

try:
    import SarahMemoryGlobals as config
except Exception:
    class config:
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MEMORY_DIR = os.path.join(DATA_DIR, "memory")
        DATASETS_DIR = os.path.join(MEMORY_DIR, "datasets")
        LOGS_DIR = os.path.join(DATA_DIR, "logs")

BACKUP_DIR = os.path.join(config.DATASETS_DIR, "_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

DBS = {
    "context_history.db": {
        "path": os.path.join(config.DATASETS_DIR, "context_history.db"),
        "ranges": [("context_history","timestamp")]
    },
    "ai_learning.db": {
        "path": os.path.join(config.DATASETS_DIR, "ai_learning.db"),
        "ranges": [("intent_logs","timestamp")]
    },
    "personality1.db": {
        "path": os.path.join(config.DATASETS_DIR, "personality1.db"),
        "ranges": [("emotion_states","timestamp")]  # `responses` may lack timestamp; cleared on ALL only
    },
    "functions.db": {
        "path": os.path.join(config.DATASETS_DIR, "functions.db"),
        "ranges": [("dl_cache","ts")]
    },
    "system_logs.db": {
        "path": os.path.join(config.DATASETS_DIR, "system_logs.db"),
        "ranges": [("events","timestamp"), ("response","timestamp"), ("responses","timestamp")]
    },
}

RANGES = [
    ("5 minutes", 5*60),
    ("10 minutes", 10*60),
    ("30 minutes", 30*60),
    ("1 hour", 60*60),
    ("3 hours", 3*60*60),
    ("5 hours", 5*60*60),
    ("12 hours", 12*60*60),
    ("1 day", 24*60*60),
    ("3 days", 3*24*60*60),
    ("1 week", 7*24*60*60),
    ("1 month (~30d)", 30*24*60*60),
    ("3 months", 90*24*60*60),
    ("6 months", 180*24*60*60),
    ("1 year", 365*24*60*60),
    ("ALL DATA", None),
]

def backup_all():
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    bundle = os.path.join(BACKUP_DIR, f"backup-{ts}")
    os.makedirs(bundle, exist_ok=True)
    for name, meta in DBS.items():
        src = meta["path"]
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bundle, name))
    # copy logs too
    if os.path.isdir(config.LOGS_DIR):
        zipname = os.path.join(bundle, "logs.zip")
        shutil.make_archive(zipname[:-4], "zip", config.LOGS_DIR)
    messagebox.showinfo("Backup", f"Backup created: {bundle}")
    return bundle

def restore_backup():
    folder = filedialog.askdirectory(initialdir=BACKUP_DIR, title="Select backup folder")
    if not folder:
        return
    restored = []
    for name, meta in DBS.items():
        src = os.path.join(folder, name)
        dst = meta["path"]
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(name)
    messagebox.showinfo("Restore", "Restored: " + ", ".join(restored) if restored else "No DB files in chosen backup.")

def clear_range(seconds=None):
    now = datetime.utcnow()
    cutoff = None if seconds is None else now - timedelta(seconds=seconds)
    for dbname, meta in DBS.items():
        path = meta["path"]
        if not os.path.exists(path):
            continue
        try:
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                if seconds is None:
                    # Clear everything safely
                    for table, _tscol in meta["ranges"]:
                        try:
                            cur.execute(f"DELETE FROM {table}")
                        except Exception:
                            pass
                    # handle responses in personality if present
                    try:
                        cur.execute("DELETE FROM responses")
                    except Exception:
                        pass
                else:
                    for table, tscol in meta["ranges"]:
                        try:
                            # Only delete where timestamp column exists
                            cur.execute(f"PRAGMA table_info({table})")
                            cols = [r[1] for r in cur.fetchall()]
                            if tscol in cols:
                                cur.execute(f"DELETE FROM {table} WHERE {tscol} >= ?", (cutoff.isoformat(),))
                        except Exception:
                            pass
                con.commit()
                con.execute("VACUUM")
        except Exception as e:
            print("[Cleanup] Failed clearing", dbname, ":", e)
    messagebox.showinfo("Cleanup", "Cleanup completed.")

def tidy_logs():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    for fn in os.listdir(config.LOGS_DIR):
        p = os.path.join(config.LOGS_DIR, fn)
        try:
            if os.path.isfile(p) and (fn.lower().endswith(".log") or fn.lower().endswith(".txt")):
                # Truncate if > 5MB
                if os.path.getsize(p) > 5*1024*1024:
                    with open(p, "rb+") as f:
                        f.seek(-1024*1024, os.SEEK_END)
                        tail = f.read()
                        f.seek(0); f.truncate()
                        f.write(b"...[truncated]\n" + tail)
        except Exception as e:
            print("[Cleanup] tidy_logs:", e)
    messagebox.showinfo("Logs", "Logs tidied.")

def launch_cleanup_gui():
    root = tk.Tk()
    root.title("SarahMemory Cleanup & Restore")
    root.geometry("520x520")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Select a time range to clear across DBs").pack(pady=6)

    range_var = tk.StringVar(value=RANGES[0][0])
    combo = ttk.Combobox(frm, textvariable=range_var, values=[r[0] for r in RANGES], state="readonly")
    combo.pack(pady=6, fill="x")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=10)
    ttk.Button(btns, text="Create Backup", command=backup_all).pack(side="left", padx=4)
    ttk.Button(btns, text="Restore Backup", command=restore_backup).pack(side="left", padx=4)
    ttk.Button(btns, text="Tidy Logs", command=tidy_logs).pack(side="left", padx=4)

    def on_clear():
        label = range_var.get()
        seconds = next((s for lbl, s in RANGES if lbl == label), None)
        if messagebox.askyesno("Confirm", f"Proceed to clear: {label}?"):
            clear_range(seconds)

    ttk.Button(frm, text="CLEAR SELECTED RANGE", command=on_clear).pack(pady=16, fill="x")

    ttk.Label(frm, text="DB Directory: " + config.DATASETS_DIR).pack(anchor="w", pady=6)
    ttk.Label(frm, text="Logs Directory: " + config.LOGS_DIR).pack(anchor="w")

    root.mainloop()

if __name__ == "__main__":
    launch_cleanup_gui()
# ====================================================================
# END OF SarahMemoryCleanup.py v8.0.0
# ====================================================================