"""
--==The SarahMemory Project==--
File: SarahMemoryGUI.py
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

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception as e:
    print(f"[WARN] python-dotenv unavailable or failed, .env not loaded: {e}")

import re
import tkinter as tk
import os, sys

def _sm_has_display():
    if os.environ.get('SARAH_FORCE_HEADLESS','').lower() in ('1','true','yes'):
        return False
    if os.environ.get('PYTHONANYWHERE_DOMAIN') or os.environ.get('PA_HOME'):
        return False
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    if sys.platform.startswith('ios') or 'android' in sys.platform.lower():
        return bool(os.environ.get('DISPLAY'))
    return True

import asyncio
def _update_network_state_thread():
    try:
        asyncio.run(async_update_network_state())
    except Exception as e:
        try:
            logger.exception('Network state update failed: %s', e)
        except Exception:
            print('[WARN] Network state update failed:', e)

# --- Sync wrapper for async network check (surgical patch) ---
def async_update_network_state_sync():
    try:
        import asyncio
        return asyncio.run(async_update_network_state())
    except Exception as e:
        return {"online": False, "reason": f"error: {e}"}
from tkinter import BOTH, LEFT, RIGHT, BOTTOM, X, Y, TOP, NSEW
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading
import logging
import os
import time
import random
# Optional desktop automation / screenshot helper.
# Wrapped in a try/except so headless environments (no DISPLAY) don't crash on import.
try:
    import pyautogui  # type: ignore
    _SM_HAVE_PYAUTOGUI = True
except Exception:
    pyautogui = None  # type: ignore
    _SM_HAVE_PYAUTOGUI = False
import io
import json
import shutil
from SarahMemoryGUI2 import WorldClassGUIEnhancer

# --- v7.7.4 hotfix: TkinterWeb thread-safety guard ---------------------------------
try:
    import threading as _sm_th
    try:
        from tkinterweb.htmlwidgets import HtmlFrame as _SM_HtmlFrame
        _SM_MAIN_TID = _sm_th.get_ident()
        _SM_ORIG_CONT = getattr(_SM_HtmlFrame, "_continue_loading", None)
        if _SM_ORIG_CONT is not None:
            def _sm_safe_continue_loading(self, *a, **k):
                # Tkinter is not thread-safe; background threads must not call Tk methods.
                # When invoked off the main thread, degrade by skipping threaded UI work.
                if _sm_th.get_ident() != _SM_MAIN_TID:
                    return  # no-op to avoid "main thread is not in main loop"
                try:
                    return _SM_ORIG_CONT(self, *a, **k)
                except Exception:
                    return
            _SM_HtmlFrame._continue_loading = _sm_safe_continue_loading
    except Exception:
        pass
except Exception:
    pass
# -------------------------------------------------------------------------------

# === Mini Browser Imports (added) ===
try:
    from tkinterweb import HtmlFrame  # pip install tkinterweb
    _SM_WEBVIEW_AVAILABLE = True
except Exception:
    _SM_WEBVIEW_AVAILABLE = False
import webbrowser as _sm_webbrowser
from urllib.parse import urljoin as _sm_urljoin
import base64 as _sm_base64

import numpy as np
import asyncio
import datetime
# import openai
import socket
import glob
from PyQt5 import QtWidgets, QtGui, QtCore, QtOpenGL
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtOpenGL import QGLWidget
# import bpy # if using Blender scripting context
# import unreal # if calling Unreal from Python - only works in Unreal's embedded Python
# Try to import torch for DLSS simulation; if unavailable, set to None.
try:
    import torch
except ImportError:
    torch = None

import SarahMemoryGlobals as config # Global configuration module
import SarahMemoryVoice as voice # Voice synthesis module
import SarahMemoryAvatar as avatar # Avatar module
# Testing import SarahMemoryAPI as oai # OpenAI API module
from SarahMemoryGlobals import run_async # Async function to run tasks
from SarahMemoryHi import async_update_network_state  # Async network function
import SarahMemorySOBJE as sobje # Object detection module
from SarahMemoryADDONLCHR import AddonLauncher # Addon launcher module
from SarahMemoryResearch import get_research_data #get data from intent statements given by the user in the text box or by voice
from PIL import Image, ImageTk
import trimesh

UNREAL_AVAILABLE = True
UNREAL_HOST = '127.0.0.1'
UNREAL_PORT = 7777
RENDER_LOOP_PATH = r"C:\SarahMemory\resources\Unreal Projects\SarahMemory\Saved\MovieRenders"
RENDER_PATTERN = "3D_MotionDesign.*.jpeg"
FRAME_RATE = 24

try:
    import psmove
    PSMOVE_ENABLED = True
except ImportError:
    PSMOVE_ENABLED = False


# Import UnifiedAvatarController; if unavailable, define a dummy.
try:
    from UnifiedAvatarController import UnifiedAvatarController


except ImportError:
    class UnifiedAvatarController:
        def show_avatar(self):
            return None

# Import intent classification and personality integration functions.
try:
    from SarahMemoryAdvCU import classify_intent
except ImportError:
    def classify_intent(text):
        return "statement"

try:
    from SarahMemoryPersonality import process_interaction, integrate_with_personality
except ImportError:

    def integrate_with_personality(text):
        return "I am here to help you."

# Set up a dedicated asyncio event loop.

# ==================================================
# 🚨 QUERY RESEARCH PATH LOGGER SETUP
# Logs the research/debug path of every query issued by the GUI
# ==================================================
research_log_path = os.path.join(config.BASE_DIR, "data", "logs", "research.log")
research_path_logger = logging.getLogger("ResearchPathLogger")
research_path_logger.setLevel(logging.DEBUG)

research_file_handler = logging.FileHandler(research_log_path, mode='a', encoding='utf-8')
research_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

if not research_path_logger.hasHandlers():
    research_file_handler = logging.FileHandler(research_log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    research_file_handler.setFormatter(formatter)
    research_path_logger.addHandler(research_file_handler)
    research_path_logger.setLevel(logging.INFO)

def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True)
    async_thread.start()

# ------------------------- Global Variables -------------------------
SETTINGS_FILE = os.path.join(config.SETTINGS_DIR, "settings.json")
NETWORK_STATE = "red"  # red, yellow, or green
MIC_STATUS = "On"
CAMERA_STATUS = "On"
# ------------------------- Shared Frame Buffer -------------------------
shared_frame = None
shared_lock = threading.Lock()

# ------------------------- Theme Loader -------------------------
theme_logger = logging.getLogger("ThemeLoader")
theme_logger.setLevel(logging.DEBUG)
if not theme_logger.hasHandlers():
    th = logging.NullHandler()
    th.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    theme_logger.addHandler(th)

def get_active_sentence_model():
    try:
        # Prefer local-only to avoid network stalls when HF token isn’t present
        return SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    except Exception:
        # Last-resort lightweight embedder to keep UI responsive
        class _LiteEmbedder:
            def encode(self, text):
                # tiny, deterministic embedding; enough for routing
                import hashlib, numpy as np
                h = hashlib.sha1(text.encode('utf-8')).digest()
                return np.frombuffer(h[:64], dtype=np.uint8).astype('float32')
        return _LiteEmbedder()
def apply_theme_from_choice(css_filename):
    """
    Applies the chosen theme by parsing the CSS file and applying styles to ttk widgets.
    """
    mods_dir = config.THEMES_DIR
    css_path = os.path.join(mods_dir, css_filename)
    if not os.path.exists(css_path):
        theme_logger.error("CSS file not found: " + css_path)
        return
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        theme_logger.error("Failed to open CSS file: " + str(e))
        return

    style = ttk.Style()
    pattern = r'([.\w-]+)\s*\{([^}]+)\}'
    matches = re.findall(pattern, css_content)
    css_to_ttk = {
        "background-color": "background",
        "background": "background",
        "color": "foreground",
        "foreground": "foreground",
        "font": "font",
        "borderwidth": "borderwidth",
        "relief": "relief"
    }
    for selector, properties in matches:
        style_name = selector.lstrip('.')
        props = {}
        for declaration in properties.split(';'):
            declaration = declaration.strip()
            if not declaration:
                continue
            if ':' in declaration:
                key, value = declaration.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                ttk_key = css_to_ttk.get(key, key)
                props[ttk_key] = value
        if props:
            try:
                style.configure(style_name, **props)
                theme_logger.info(f"Applied theme '{css_filename}' to style '{style_name}' with properties: {props}")
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                theme_logger.error(f"Error applying theme for style '{style_name}': {e}")

# ------------------------- Logger Setup -------------------------
logger = logging.getLogger("SarahMemoryGUI")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    sh = logging.NullHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

# ------------------------- Utility Functions -------------------------
def log_gui_event(event: str, details: str) -> None:
    try:
        db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        import sqlite3
        from datetime import datetime
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gui_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event TEXT,
                    details TEXT
                )
            """)
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO gui_events (timestamp, event, details) VALUES (?, ?, ?)",
                           (timestamp, event, details))
            conn.commit()
        logger.info(f"Logged GUI event: {event} - {details}")
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        logger.error(f"Error logging GUI event: {e}")

# ------------------------- Extended Avatar Controller -------------------------
class ExtendedAvatarController(UnifiedAvatarController):
    def show_avatar(self):
        # ✅ Try primary Blender-rendered avatar
        rendered_avatar_path = os.path.join(config.AVATAR_DIR, "avatar_rendered.jpg")
        try:
            if os.path.exists(rendered_avatar_path):
                img = Image.open(rendered_avatar_path).convert("RGB")
                img = img.resize((640, 800))
                logger.info("Loaded rendered 3D avatar image.")
            else:
                raise FileNotFoundError("avatar_rendered.jpg not found.")
        except Exception as e:
            logger.warning(f"⚠️ Rendered avatar load failed: {e}")
            # 🔄 Fallback to a random static image from avatar folder
            avatar_files = [f for f in os.listdir(config.AVATAR_DIR) if f.lower().endswith(".jpg")]
            if not avatar_files:
                logger.error("No fallback avatars found in avatar folder.")
                return None
            fallback_path = os.path.join(config.AVATAR_DIR, random.choice(avatar_files))
            img = Image.open(fallback_path).convert("RGB")
            img = img.resize((640, 800))
            logger.info("Loaded fallback static avatar.")

        try:
            if torch is not None:
                arr = np.array(img)
                tensor = torch.tensor(arr).float()
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                    noise = torch.randn_like(tensor) * config.NOISE_SCALE
                    tensor = torch.clamp(tensor + noise, 0, 255).cpu()
                else:
                    noise = torch.randn_like(tensor) * config.NOISE_SCALE
                    tensor = torch.clamp(tensor + noise, 0, 255)
                animated_img = Image.fromarray(tensor.byte().numpy())
                photo = ImageTk.PhotoImage(animated_img, master=tk._default_root)
                logger.info("PyTorch-augmented avatar displayed.")
            else:
                photo = ImageTk.PhotoImage(img, master=tk._default_root)
                logger.info("Static avatar displayed (no PyTorch).")
            return photo
        except Exception as e:
            logger.error(f"Avatar display/render error: {e}")
            return None

# ------------------------- Connection Panel (Top Bar) -------------------------
class ConnectionPanel:
    def __init__(self, parent, settings_callback):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, padx=5, pady=5)
        HOST_ROOM_TEXT = "Host Room"
        JOIN_ROOM_TEXT = "Join Room"
        SEND_FILE_TEXT = "Send File"
        MIC_STATUS_TEXT = "Mic Mute"
        CAM_STATUS_TEXT = "Camera On/Off"
        SCAN_STATUS_TEXT = "Scan Item"
        ADDON_SELECT_TEXT = "Open Add-ons"
        MEMORY_REFRESH_TEXT = "Memory AutoCorrect"
        EXIT_TEXT = "Exit"

        self.host_button = ttk.Button(self.frame, text=HOST_ROOM_TEXT, command=self.host_room)
        self.join_button = ttk.Button(self.frame, text=JOIN_ROOM_TEXT, command=self.join_room)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(self.frame, text=SEND_FILE_TEXT, command=self.send_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        # Call open_settings_window without expecting an extra argument.
        self.host_button = ttk.Button(self.frame, text=MIC_STATUS_TEXT, command=self.toggle_mic)
        self.host_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=CAM_STATUS_TEXT, command=self.toggle_camera)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=SCAN_STATUS_TEXT, command=self.scan_item)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=ADDON_SELECT_TEXT, command=self.open_addons)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.join_button = ttk.Button(self.frame, text=MEMORY_REFRESH_TEXT, command=self.memory_autocorrect)
        self.join_button.pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(self.frame, text=EXIT_TEXT, command=self.exit_app)

        self.file_button.pack(side=tk.LEFT, padx=5)
    def host_room(self):
        pwd = simpledialog.askstring("Host Room", "Enter a password for your room:")
        if pwd:
            logger.info("Hosting room with provided password.")
            messagebox.showinfo("Host Room", "Room hosted successfully.")
        else:
            logger.warning("Host room cancelled.")

    def join_room(self):
        pwd = simpledialog.askstring("Join Room", "Enter the room password:")
        if pwd:
            logger.info("Joining room with provided password.")
            messagebox.showinfo("Join Room", "Joined room successfully.")
        else:
            logger.warning("Join room cancelled.")

    def send_file(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to send")
        if file_paths:
            for f in file_paths:
                _, ext = os.path.splitext(f)
                ext = ext.lower()
                mapping = {
                    ".csv": config.DATASETS_DIR,
                    ".json": config.DATASETS_DIR,
                    ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
                    ".png": os.path.join(config.PROJECTS_DIR, "images"),
                    ".py": config.IMPORTS_DIR
                }
                target = mapping.get(ext, config.ADDONS_DIR)
                os.makedirs(target, exist_ok=True)
                try:
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    messagebox.showerror("Error", f"Failed to send file {f}: {e}")
            messagebox.showinfo("Send File", "Files sent successfully!")
    def scan_item(self):
    # Placeholder: call scanning function from avatar module (if implemented)
        try:
            from SarahMemoryAvatar import perform_scan_capture
            perform_scan_capture()
            log_gui_event("Scan Item", "Scan triggered.")
        except Exception as e:
            logger.error(f"Scan error: {e}")
            messagebox.showerror("Error", f"Scan failed: {e}")

    def toggle_mic(self):
        global MIC_STATUS
        MIC_STATUS = "Off" if MIC_STATUS == "On" else "On"
        log_gui_event("Mic Toggle", f"Mic status set to {MIC_STATUS}")

    def toggle_camera(self):
        global CAMERA_STATUS
        CAMERA_STATUS = "Off" if CAMERA_STATUS == "On" else "On"
        log_gui_event("Camera Toggle", f"Camera status set to {CAMERA_STATUS}")
    #--------------------------------OPEN ADDONS MODULE-----------------------------
    def open_addons(self):
        self.addon_launcher = AddonLauncher(parent=self.frame)
        self.addon_launcher.open_addons()
    #-------------------------------MEMORY AUTOCORRECT FUNCTION---------------------
    def memory_autocorrect(self):
        from SarahMemorySystemLearn import memory_autocorrect
        memory_autocorrect()
        messagebox.showinfo("Memory AutoCorrect", "Intent overrides updated from system logs.")

    #-------------------------------END ADDONS MODULE-------------------------------
    def exit_app(self):
        import sys
        log_gui_event("Shutdown", "User clicked Exit button.")
        self.shutdown()


# ------------------------- Video Panel (Left Column) -------------------------
class VideoPanel:
    def _try_hw_focus(self, cap):
        """Best-effort hardware autofocus/autoexposure toggles across backends.
        Safe on devices that don't support these properties."""
        try:
            import cv2
            # Known property IDs across backends
            props = [getattr(cv2, 'CAP_PROP_AUTOFOCUS', None), 39]
            for p in props:
                if p is None:
                    continue
                try:
                    cap.set(p, 1)
                except Exception:
                    pass
            # Nudge manual focus if exposed (some cams need a 'poke' to refocus)
            for p in [getattr(cv2, 'CAP_PROP_FOCUS', None), 28]:
                if p is None:
                    continue
                try:
                    cur = cap.get(p)
                    cap.set(p, cur)
                except Exception:
                    pass
        except Exception:
            pass

    def _soft_focus_score(self, frame):
        try:
            import cv2, numpy as np
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(fm)
        except Exception:
            return 0.0

    def __del__(self):
        try:
            self.release_resources()
        except Exception:
            pass

    def __init__(self, parent):
        self.frame = ttk.Frame(parent, relief=tk.GROOVE, borderwidth=1)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        self.local_label = tk.Label(self.frame, text="Local Camera", bg="black")
        self.local_label.grid(row=0, column=0, sticky="nsew")
        self.remote_label = tk.Label(self.frame, text="Remote Camera", bg="black")
        self.remote_label.grid(row=1, column=0, sticky="nsew")
        self.screen_preview = tk.Label(self.frame, text="Desktop Mirror", bg="black")
        self.screen_preview.grid(row=2, column=0, sticky="nsew")

        self.local_camera = cv2.VideoCapture(0)
        if not self.local_camera.isOpened():
            logger.error("Failed to open local camera.")
            self.local_camera = None

        self.remote_camera = None
        self.local_photo = None
        self.remote_photo = None
        self.update_video()

        # Start background object and facial detection loop
        if config.ASYNC_PROCESSING_ENABLED:
            from SarahMemorySOBJE import ultra_detect_objects
            from SarahMemoryFacialRecognition import detect_faces_dnn

            def vision_processing_loop():
                global shared_frame
                while True:
                    with shared_lock:
                        frame = shared_frame.copy() if shared_frame is not None else None
                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        tags = ultra_detect_objects(frame)
                        faces = detect_faces_dnn(frame)
                        if hasattr(config, 'status_bar'):
                            try:
                                text_summary = f"Tags: {', '.join(tags[:3])} | Faces: {len(faces)}"
                                config.status_bar.set_status(text_summary)
                            except Exception as e:
                                logger.warning(f"Status bar update failed: {e}")
                    time.sleep(3)
            if config.OBJECT_DETECTION_ENABLED:
                threading.Thread(target=vision_processing_loop, daemon=True).start()


    def update_video(self):
        global shared_frame
        try:
            if self.local_camera and self.local_camera.isOpened():
                ret, frame = self.local_camera.read()
                if not ret:
                    logger.warning("Camera read failed; disabling camera to keep UI responsive.")
                    try:
                        self.local_camera.release()
                    except Exception:
                        pass
                    self.local_camera = None
                    # backoff before next open attempt
                    self._next_cam_retry_ts = time.time() + 2.0

                    # Local placeholder (build once)
                    if not hasattr(self, "_local_placeholder"):
                        blank_local = np.zeros((240, 320, 3), dtype=np.uint8)
                        self._local_placeholder = ImageTk.PhotoImage(Image.fromarray(blank_local))
                    self.local_label.configure(image=self._local_placeholder)
                    self.local_label.image = self._local_placeholder

                    # Slightly slower reschedule right after failure
                    if self.frame.winfo_exists():
                        self.frame.after(500, self.update_video)
                    return

                # Good frame path
                frame = cv2.flip(frame, 1)
                # Measure focus; if too blurry for a while, poke autofocus
                try:
                    score = self._soft_focus_score(frame)
                    blurry = score < 120.0
                    count = getattr(self, '_blur_count', 0)
                    self._blur_count = count + 1 if blurry else 0
                    if self._blur_count >= 8 and self.local_camera is not None:
                        self._try_hw_focus(self.local_camera)
                        self._blur_count = 0
                except Exception:
                    pass
                resized = cv2.resize(frame, (320, 240))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self.local_photo = ImageTk.PhotoImage(img)
                self.local_label.configure(image=self.local_photo)
                self.local_label.image = self.local_photo
                with shared_lock:
                    shared_frame = frame.copy()

            else:
                # Throttle camera reopen attempts
                now = time.time()
                retry_due = getattr(self, "_next_cam_retry_ts", 0)
                if now >= retry_due:
                    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                    if not (cap and cap.isOpened()):
                        # Try DirectShow as fallback
                        if cap:
                            try: cap.release()
                            except Exception: pass
                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

                    if cap and cap.isOpened():
                        ok, _ = cap.read()
                        if not ok:
                            cap.release()
                            cap = None
                            # wait a bit longer before retry
                            self._next_cam_retry_ts = now + 2.0
                        else:
                            # camera looks good; reset retry cadence
                            self._next_cam_retry_ts = now + 0.25
                    else:
                        # couldn’t open at all; wait before retry
                        self._next_cam_retry_ts = now + 2.0

                    self.local_camera = cap
                    try:
                        self._try_hw_focus(cap)
                    except Exception:
                        pass

                # Local placeholder (build once)
                if not hasattr(self, "_local_placeholder"):
                    blank_local = np.zeros((240, 320, 3), dtype=np.uint8)
                    self._local_placeholder = ImageTk.PhotoImage(Image.fromarray(blank_local))
                self.local_label.configure(image=self._local_placeholder)
                self.local_label.image = self._local_placeholder

            # Remote placeholder (build once)
            if not hasattr(self, "_remote_placeholder"):
                blank_remote = np.zeros((240, 320, 3), dtype=np.uint8)
                self._remote_placeholder = ImageTk.PhotoImage(Image.fromarray(blank_remote))
            self.remote_label.configure(image=self._remote_placeholder, text="")
            self.remote_label.image = self._remote_placeholder

        except Exception as e:
            logger.warning(f"update_video error: {e}")

        # Always keep the loop alive (only if widget still exists)
        if self.frame.winfo_exists():
            self.frame.after(30, self.update_video)




    def update_desktop_mirror(self):

        try:
        # If we're headless or pyautogui failed to import, skip.
            if not globals().get("_SM_HAVE_PYAUTOGUI", False) or pyautogui is None:
                return

            screenshot = pyautogui.screenshot()
            screenshot = screenshot.resize((320, 240))

            photo = ImageTk.PhotoImage(screenshot, master=tk._default_root)
            self.desktop_label.configure(image=photo)
            self.desktop_label.image = photo

        except Exception as e:
            logger.error(f"Desktop mirror update failed: {e}")

    # Schedule the next update
        self.frame.after(5000, self.update_desktop_mirror)

    def host_room(self):
        pwd = simpledialog.askstring("Host Room", "Enter a password for your room:")
        if pwd:
            logger.info("Hosting room with provided password.")
            messagebox.showinfo("Host Room", "Room hosted successfully.")
        else:
            logger.warning("Host room cancelled.")

    def join_room(self):
        pwd = simpledialog.askstring("Join Room", "Enter the room password:")
        if pwd:
            logger.info("Joining room with provided password.")
            messagebox.showinfo("Join Room", "Joined room successfully.")
        else:
            logger.warning("Join room cancelled.")

    def send_file(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to send")
        if file_paths:
            for f in file_paths:
                _, ext = os.path.splitext(f)
                ext = ext.lower()
                mapping = {
                    ".csv": config.DATASETS_DIR,
                    ".json": config.DATASETS_DIR,
                    ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
                    ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
                    ".png": os.path.join(config.PROJECTS_DIR, "images"),
                    ".py": config.IMPORTS_DIR
                }
                target = mapping.get(ext, config.ADDONS_DIR)
                os.makedirs(target, exist_ok=True)
                try:
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    messagebox.showerror("Error", f"Failed to send file {f}: {e}")
            messagebox.showinfo("Send File", "Files sent successfully!")

    def release_resources(self):
        if self.local_camera and hasattr(self.local_camera, 'isOpened') and self.local_camera.isOpened():
            self.local_camera.release()
        if self.remote_camera and hasattr(self.remote_camera, 'isOpened') and self.remote_camera.isOpened():
            self.remote_camera.release()
        # screen_preview is a Label, so we skip checking isOpened
# ------------------------- Chat Panel (Middle Column) -------------------------
# ------------------------- Chat Panel (Middle Column) -------------------------
class ChatPanel:
    def __init__(self, parent, gui_instance, avatar_controller):
        # Store GUI and avatar references
        self.gui = gui_instance
        self.avatar_controller = avatar_controller

        # Main container frame setup
        self.frame = ttk.Frame(parent, relief=tk.GROOVE, borderwidth=2)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.frame.columnconfigure(0, weight=1)

        self.frame.columnconfigure(1, weight=0)




        self.frame.rowconfigure(0, weight=4)
        self.frame.rowconfigure(1, weight=0)
        self.frame.rowconfigure(2, weight=0)
        self.frame.rowconfigure(3, weight=3)
        self.frame.rowconfigure(4, weight=0)
        # Output display widget for chat history
        self.chat_output = tk.Text(self.frame, height=20, wrap=tk.WORD, state=tk.DISABLED,
                                   bg='#ffffff', fg='#333333', relief=tk.FLAT)
        self.chat_output.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # === Mini Browser Reply Pane (added) ===
        try:
            self._reply_toolbar = ttk.Frame(self.frame)
            self._reply_toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0,4))

            # toolbar buttons (packed into the reply toolbar)
            self._btn_back = ttk.Button(self._reply_toolbar, text="◀ Back", command=lambda: self._reply_nav("back"))
            self._btn_forward = ttk.Button(self._reply_toolbar, text="Forward ▶", command=lambda: self._reply_nav("forward"))
            self._btn_reload = ttk.Button(self._reply_toolbar, text="⟳ Reload", command=lambda: self._reply_nav("reload"))
            self._btn_open = ttk.Button(self._reply_toolbar, text="Open in Browser", command=lambda: self._reply_nav("open_ext"))
            self._btn_toggle = ttk.Button(self._reply_toolbar, text="Toggle HTML/Text", command=lambda: self._reply_nav("toggle"))
            for _b in (self._btn_back, self._btn_forward, self._btn_reload, self._btn_open, self._btn_toggle):
                _b.pack(side="left", padx=4)

            # --- Mini Browser Address Bar (added) ---
            self._addr_frame = ttk.Frame(self.frame)
            self._addr_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(2,6))
            self._addr_var = tk.StringVar()
            self._addr_entry = ttk.Entry(self._addr_frame, textvariable=self._addr_var)
            self._addr_entry.pack(side="left", fill="x", expand=True, padx=(0,6))
            self._addr_entry.bind("<Return>", lambda e: self._reply_go())
            self._addr_go = ttk.Button(self._addr_frame, text="Go", command=self._reply_go)
            self._addr_go.pack(side="left")
        except Exception:
            pass

        self._reply_stack = ttk.Frame(self.frame)
        self._reply_stack.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=(0,5))
        try:
            # Placeholder for future reply stack initialization (no operation)
            pass
        except Exception:
            pass

        self._reply_browser = None
        if _SM_WEBVIEW_AVAILABLE:
            try:
                self._reply_browser = HtmlFrame(self._reply_stack, messages_enabled=False, vertical_scrollbar=True)
                self._reply_browser.pack(fill="both", expand=True)
            except Exception:
                self._reply_browser = None

        self._reply_text = tk.Text(self._reply_stack, wrap="word", state="disabled", bg="#ffffff", fg="#333333", relief=tk.FLAT)
        if self._reply_browser:
            self._reply_text.pack_forget()
        else:
            self._reply_text.pack(fill="both", expand=True)

        self._reply_mode_html = bool(self._reply_browser)
        self._reply_base_url = "file:///{}/".format(os.path.abspath(".").replace("\\", "/"))
        self._seed_google_lite()
# Define custom tags for colouring chat output. These tags provide
        # separate colours and font weights for the user and Sarah, and a
        # neutral colour for system information.  If tag configuration fails
        # (e.g. in older Tk versions), the defaults remain.
        try:
            self.chat_output.tag_configure('user', foreground='#006400', font=('Helvetica', 11, 'bold'))
            self.chat_output.tag_configure('sarah', foreground='#1f3faf', font=('Helvetica', 11))
            self.chat_output.tag_configure('info', foreground='#666666', font=('Helvetica', 10, 'italic'))
        except Exception as e:
            logger.warning(f"Tag configuration failed: {e}")

        # Scrollbar for the output box
        self.scrollbar = ttk.Scrollbar(self.frame, command=self.chat_output.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.chat_output.configure(yscrollcommand=self.scrollbar.set)
        # User input box
        self.chat_input = tk.Text(self.frame, height=3)
        self.chat_input.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        self.chat_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ttk.Button(self.frame, text="Send", command=self.send_message)
        self.send_button.grid(row=4, column=1, padx=5, pady=5)

        # Store images embedded in chat
        self.chat_images = []

    def insert_avatar_image(self, photo):
        self.chat_output.configure(state=tk.NORMAL)
        self.chat_images.append(photo)
        self.chat_output.image_create(tk.END, image=photo)
        self.chat_output.insert(tk.END, "\n")
        self.chat_output.see(tk.END)
        self.chat_output.configure(state=tk.DISABLED)

    def scan_item(self):
        from SarahMemoryAvatar import perform_scan_capture
        perform_scan_capture()
        log_gui_event("Manual Scan", "User clicked scan button.")

    def capture_desktop_loop(self):
        import time
        while True:
            try:
                screenshot = pyautogui.screenshot()
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                small = cv2.resize(img, (100, 60))
                img_pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(img_pil)
                self.screen_preview.configure(image=photo)
                self.screen_preview.image = photo
                from SarahMemoryAvatar import ultra_detect_objects
                tags = ultra_detect_objects(img)
                if hasattr(config, 'status_bar'):
                    config.status_bar.set_status(f"Desktop View: {', '.join(tags[:3])}")
            except Exception as e:
                logger.warning(f"Screen capture error: {e}")
            time.sleep(15)

    def send_message(self, event=None):
        # -- exit guard hf4 --
        try:
            _txt = (self.input_box.get() if hasattr(self, 'input_box') else '').strip()
        except Exception:
            _txt = ''
        if _txt.lower() == 'exit':
            try:
                if hasattr(self, 'show_exit_popup'):
                    self.show_exit_popup()
                    return
            except Exception:
                pass
            # -- exit guard begin --
        try:
            _txt = (self.input_box.get() if hasattr(self, 'input_box') else '').strip()
        except Exception:
            _txt = ''
        if _txt.lower() == 'exit':
            try:
                if hasattr(self, 'show_exit_popup'):
                    self.show_exit_popup()
                    return
            except Exception:
                pass
        # -- exit guard end --
        from SarahMemoryGlobals import INTERRUPT_KEYWORDS, INTERRUPT_FLAG
        from SarahMemoryPersonality import get_emotion_response

        # ✅ Correct access to data mode settings
        override_mode = getattr(self.gui, 'override_data_mode', 'api').lower()
        config.LOCAL_DATA_ENABLED = override_mode in ["local", "any"]
        config.API_RESEARCH_ENABLED = override_mode in ["api", "web", "any"]
        config.WEB_RESEARCH_ENABLED = override_mode in ["web", "any"]

        if event:
            event.widget.mark_set("insert", "end")
            event.widget.tag_remove("sel", "1.0", "end")

        text = self.chat_input.get("1.0", tk.END).strip()
        research_path_logger.debug(f"[GUI] Received user input: '{text}' at send_message()")
        if not text:
            return "break"

        self.chat_input.delete("1.0", tk.END)
        self.append_message("You: " + text)
        log_gui_event("Chat Sent", text)
        self.gui.status_bar.set_intent_light("yellow")

        if any(word in text.lower() for word in INTERRUPT_KEYWORDS):
            import SarahMemoryGlobals
            SarahMemoryGlobals.INTERRUPT_FLAG = True
            emotion = "frustration"
            response = get_emotion_response(emotion)
            self.append_message("Sarah: " + response + " I’ve stopped the request as you asked.")
            None  # (TTS disabled; handled by Reply layer)
            return "break"

        if "exit" in text.lower():
            self.exit_chat()
            return "break"

        if "show me your avatar" in text.lower():
            self.gui.update_avatar_window()
            return "break"

        if "create your avatar" in text.lower() or "change your" in text.lower():
            run_async(self.avatar_controller.create_avatar, text)
            return "break"

        run_async(self.generate_response, text)
        return "break"

    def generate_response(self, user_text):
        from SarahMemoryReply import generate_reply
        from SarahMemoryCompare import compare_reply
        from SarahMemoryGlobals import COMPARE_VOTE
        from SarahMemoryDatabase import record_qa_feedback
        research_path_logger.debug(f"[GUI] Forwarding to generate_reply() from GUI input: '{user_text}'")
        result_bundle = generate_reply(self, user_text)  # LOGGING STARTS HERE
        # When generate_reply returns a dictionary, unpack and display the reply
        if isinstance(result_bundle, dict):
            response = (result_bundle.get("response") or result_bundle.get("data") or "").strip()
            meta = result_bundle.get("meta") or {}
            source = meta.get("source", result_bundle.get("source", "unknown"))
            intent = meta.get("intent", result_bundle.get("intent", "undetermined"))
#---------------------------------------------------------------------------------------
            # Update last interaction timestamp to prevent idle optimization
            self.gui.last_user_interaction = time.time()

#---------------------------------------------------------------------------------------
        # Clean inline provenance artifacts emitted by downstream layers.
        # Always strip trailing bare brackets like " []"

            text = (response or "").strip()

            if text:
                # If REPLY_STATUS is OFF, strip inline debug artifacts before showing
                try:
                    if not getattr(config, "REPLY_STATUS", False):
                        text = _re.sub(r"\s*\[\]\s*$", "", text)                      # trailing []
                        text = _re.sub(r"\s*\[Source:[^\]]+\]\s*", "", text)          # inline [Source: ...]
                        text = _re.sub(r"\s*\(Intent:[^)]+\)\s*", "", text)           # inline (Intent: ...)
                except Exception:
                    pass

                self.append_message("Sarah: " + text)

                # Optional GUI catch-all TTS (OFF by default; Reply layer is primary speaker)
                try:
                    if getattr(config, "GUI_CATCHALL_TTS", False):
                        meta = (result_bundle.get("meta") or {}) if isinstance(result_bundle, dict) else {}
                        if not meta.get("spoken", False):                # avoid double-speak if Reply already spoke
                            self._speak_if_needed(result_bundle)
                except Exception:
                    pass
            else:
                # Only show fallback when truly empty
                self.append_message("Sarah: [No response]")

            # --- Optional provenance line (separate, clean; respects REPLY_STATUS) ---
            try:
                if getattr(config, "REPLY_STATUS", False):
                    source_display = source or "unknown"
                    intent_display = intent or "undetermined"
                    self.append_message(f"[Source: {source_display}] (Intent: {intent_display})")
            except Exception:
                pass
                        # Optional comparison and voting logic
            if config.API_RESPONSE_CHECK_TRAINER:
                compare_result = compare_reply(user_text, response)
                if compare_result and isinstance(compare_result, dict):
                    conf = compare_result.get("similarity_score", 'N/A')
                    status = compare_result.get("status", 'N/A')
                    comp_source = compare_result.get("source", 'unknown')
                    comp_intent = compare_result.get("intent", 'verification')
                    self.append_message(f"[Comparison] Status: {status} | Confidence: {conf} | [Source: {comp_source}] (Intent: {comp_intent})")
                    if COMPARE_VOTE:
                        try:
                            from tkinter import messagebox
                            vote = messagebox.askyesno("Feedback", "Was this a helpful response?")
                            vote_label = "Yes" if vote else "No"
                            record_qa_feedback(user_text, score=1 if vote else 0, feedback=f"UserVote: {vote_label}")
                            self.append_message(f"[User Vote] You said: {vote_label}")
                        except Exception as e:
                            logger.warning(f"User feedback prompt failed: {e}")
        else:
            # Fallback error if generate_reply did not return a dict
            self.append_message("[ERROR] Failed to get a valid response from AI pipeline.")

    def compare_responses(self, user_text, generated_response):
        from SarahMemoryCompare import compare_reply
        result = compare_reply(user_text, generated_response)
        if result and isinstance(result, dict):
            self.append_message(f"[Comparison Result] Status: {result['status']} | Confidence: {result.get('similarity_score', 'N/A')}")
        else:
            self.append_message("[Comparison Result] No feedback returned.")

    def exit_chat(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            log_gui_event("Exit Command", "User typed 'exit' in chat.")
            self.gui.shutdown()


    def _reply_nav(self, action):
        """Handle Mini Browser toolbar actions: back, forward, reload, open_ext, toggle."""
        try:
            if action == "back" and self._reply_browser:
                try:
                    self._reply_browser.html.backward()
                except Exception:
                    pass
            elif action == "forward" and self._reply_browser:
                try:
                    self._reply_browser.html.forward()
                except Exception:
                    pass
            elif action == "reload" and self._reply_browser:
                try:
                    self._reply_browser.on_reload()
                except Exception:
                    pass
            elif action == "open_ext":
                try:
                    current = self._addr_var.get().strip()
                    if current:
                        import webbrowser as _wb
                        _wb.open(current)
                except Exception:
                    pass
            elif action == "toggle":
                try:
                    if self._reply_browser and self._reply_text:
                        if self._reply_text.winfo_ismapped():
                            self._reply_text.pack_forget()
                            self._reply_browser.pack(fill="both", expand=True)
                        else:
                            self._reply_browser.pack_forget()
                            self._reply_text.pack(fill="both", expand=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _reply_go(self):
        url = self._addr_var.get().strip() if hasattr(self, "_addr_var") else ""
        if not url:
            self._seed_google_lite()
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url
            self._addr_var.set(url)
        try:
            if self._reply_browser:
                try:
                    self._reply_browser.load_website(url)
                except Exception:
                    self._reply_text.configure(state="normal")
                    self._reply_text.delete("1.0", "end")
                    self._reply_text.insert("end", f"Open in browser: {url}\n")
                    self._reply_text.configure(state="disabled")
            else:
                self._reply_text.configure(state="normal")
                self._reply_text.delete("1.0", "end")
                self._reply_text.insert("end", f"Open in browser: {url}\n")
                self._reply_text.configure(state="disabled")
        except Exception:
            pass

    def _seed_google_lite(self):
        html = """<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>Google (Lite Demo)</title>
<style>
  html, body { margin:0; padding:0; font-family: Arial, sans-serif; background:#ffffff; }
  .wrap { display:flex; flex-direction:column; align-items:center; padding:16px; }
  .card { width: 100%; max-width: 640px; border:1px solid #e6e6e6; border-radius:10px; padding:24px; box-sizing:border-box; }
  .logo { display:flex; justify-content:center; margin:12px 0 18px; }
  .bar { display:flex; gap:8px; }
  .bar input { flex:1; padding:10px 12px; border:1px solid #d0d0d0; border-radius:20px; outline:none; }
  .bar button { padding:10px 16px; border:1px solid #d0d0d0; background:#f5f5f5; border-radius:20px; cursor:pointer; }
  .hint { color:#666; font-size:12px; margin-top:12px; text-align:center; }
  .scaled { transform: scale(0.85); transform-origin: top center; }
  .link { text-align:center; margin-top:10px; }
  a { color:#1a73e8; text-decoration:none; }
</style>
</head>
<body>
  <div class=\"wrap scaled\">
    <div class=\"card\">
      <div class=\"logo\">
        <img src=\"https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png\" alt=\"Google\"/>
      </div>
      <div class=\"bar\">
        <input type=\"text\" placeholder=\"Search Google (demo box)\" />
        <button onclick=\"window.open('https://www.google.com','_blank')\">Google</button>
      </div>
      <div class=\"hint\">This is a lightweight demo rendered by SarahMemory's Mini Browser (tkinterweb).<br/>
      Use the address bar above the viewer to load a site. The styling is scaled for compact preview.</div>
      <div class=\"link\"><a href=\"https://www.google.com\" target=\"_blank\">Open real google.com</a></div>
    </div>
  </div>
</body>
</html>"""
        try:
            if self._reply_browser:
                try:
                    self._reply_browser.set_content(html)
                except Exception:
                    pass
            else:
                self._reply_text.configure(state="normal")
                self._reply_text.delete("1.0","end")
                self._reply_text.insert("end", "Mini Browser (text fallback): Google Lite Demo\\nVisit: https://www.google.com\\n")
                self._reply_text.configure(state="disabled")
        except Exception:
            pass


# ------------------------- Avatar Panel (Right Column) -------------------------
    def append_message(self, message: str) -> None:
            """Append a message to the chat output with colouring based on speaker.

            Messages starting with "You:" are styled with the 'user' tag, "Sarah:" with
            the 'sarah' tag, and other bracketed lines (e.g. [Comparison] or
            [Source:]) with the 'info' tag.  Messages without a recognised prefix
            default to the base style.
            """
            self.chat_output.configure(state=tk.NORMAL)
            # Determine tag based on prefix
            tag = None
            if message.startswith("You:"):
                tag = 'user'
            elif message.startswith("Sarah:"):
                tag = 'sarah'
            elif message.startswith("["):
                tag = 'info'
            try:
                if tag:
                    self.chat_output.insert(tk.END, message + "\n", tag)
                else:
                    self.chat_output.insert(tk.END, message + "\n")
            except Exception:
                # Fallback if tag insertion fails
                self.chat_output.insert(tk.END, message + "\n")
            self.chat_output.configure(state=tk.DISABLED)
            self.chat_output.see(tk.END)

class AvatarPanel(QGLWidget):

    def __init__(self, parent=None):
        if not config.ENABLE_AVATAR_PANEL:
                return  # Exit if avatar panel is disabled
        super(AvatarPanel, self).__init__(parent)
        self.setMinimumSize(640, 800)
        self.frames = []
        self.current_frame = 0
        self.video_cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.mode = "static"
        self.static_frame = None
        self.controller_position = (320, 400)
        self.last_path = r"C:\SarahMemory\resources\Unreal Projects\SarahMemory\Saved\MovieRenders"

        self.selector_widget = QWidget(self)
        self.selector_widget.setGeometry(0, 0, 640, 40)
        self.selector_layout = QHBoxLayout()
        self.selector_button = QPushButton("Select Avatar Media")
        self.selector_layout.addWidget(self.selector_button)
        self.selector_widget.setLayout(self.selector_layout)
        self.selector_button.clicked.connect(self.select_folder)

        if PSMOVE_ENABLED:
            self.ps_move = psmove.PSMove()
            self.controller_thread = threading.Thread(target=self.poll_controller)
            self.controller_thread.daemon = True
            self.controller_thread.start()

        self.load_media(self.last_path)
        self.timer.start(int(1000 / 24))

    def poll_controller(self):
        while True:
            if self.ps_move.poll():
                x, y, _ = self.ps_move.get_accelerometer_frame(psmove.Frame.Last)
                self.controller_position = (
                    int(320 + x * 100),
                    int(400 + y * 100)
                )
            time.sleep(0.01)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder or Media")
        if folder:
            self.last_path = folder
            self.cleanup_previous()
            self.load_media(folder)

    def cleanup_previous(self):
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()
        self.frames.clear()
        self.static_frame = None
        self.mode = "static"

    def load_media(self, path):
        try:
            if os.path.isdir(path):
                image_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
                if len(image_files) > 1:
                    self.mode = "frame_sequence"
                    self.frames = [cv2.imread(img) for img in image_files]
                    logger.info(f"Loaded {len(self.frames)} frames for looping sequence.")
                elif len(image_files) == 1:
                    self.mode = "static"
                    self.static_frame = cv2.imread(image_files[0])
                    logger.info(f"Loaded single frame: {image_files[0]}")
                else:
                    video_files = glob.glob(os.path.join(path, "*.mp4"))
                    if video_files:
                        self.mode = "video"
                        self.video_cap = cv2.VideoCapture(video_files[0])
                        logger.info(f"Loaded MP4 video: {video_files[0]}")
                    else:
                        model_files = glob.glob(os.path.join(path, "*.glb")) + glob.glob(os.path.join(path, "*.stl"))
                        if model_files:
                            self.mode = "model_3d"
                            self.static_frame = self.render_model_preview(model_files[0])
                            logger.info(f"Loaded 3D model: {model_files[0]}")
            else:
                logger.warning("Provided path is not valid.")
        except Exception as e:
            logger.error(f"Media load error: {e}")

    def update_frame(self):
        if self.mode == "frame_sequence":
            if self.frames:
                self.current_frame = (self.current_frame + 1) % len(self.frames)
                frame = self.frames[self.current_frame]
                self.render_image(frame)
        elif self.mode == "static":
            if self.static_frame is not None:
                self.render_image(self.static_frame)
        elif self.mode == "video":
            if self.video_cap and self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    self.render_image(frame)
                else:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.mode == "model_3d":
            if self.static_frame is not None:
                self.render_image(self.static_frame)

    def render_model_preview(self, model_path):
        try:
            mesh = trimesh.load(model_path)
            preview = mesh.scene().save_image(resolution=(640, 800))
            nparr = np.frombuffer(preview, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"3D model preview failed: {e}")
            return None

    def render_image(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 40, qimg)
            if PSMOVE_ENABLED:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                painter.drawEllipse(self.controller_position[0], self.controller_position[1], 30, 30)
            painter.end()
        except Exception as e:
            logger.error(f"Image rendering failed: {e}")

    def update_virtual_object(self, object_description):
        try:
            logger.info(f"3D Engine: Creating object: {object_description}")
            # Unreal integration: Blueprint call
            if UNREAL_AVAILABLE:
                try:
                    with socket.create_connection((UNREAL_HOST, UNREAL_PORT), timeout=5) as sock:
                        command = json.dumps({"command": "spawn_object", "name": object_description})
                        sock.sendall(command.encode('utf-8'))
                        response = sock.recv(1024).decode('utf-8')
                        logger.info(f"Unreal Engine responded: {response}")
                        return
                except Exception as ue_err:
                    logger.error(f"Socket to Unreal failed: {ue_err}")
                try:
                    unreal.log(f"Request to generate: {object_description}")
                    world = ue.get_editor_world()
                    actor = world.actor_spawn(ue.find_class('StaticMeshActor'), ue.FVector(0, 0, 200))
                    system_lib = unreal.SystemLibrary()
                    actor_util = unreal.EditorLevelLibrary()
                    new_actor = actor_util.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), location=(0, 0, 100))
                    logger.info("Unreal object spawned.")
                    return
                except Exception as e:
                    pass
        except Exception as ue:
            logger.warning(f"Unreal fallback to Blender: {ue}")
            try:
                bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1, depth=2, location=(0, 0, 0))
                bpy.context.active_object.name = object_description.replace(" ", "_")
                bpy.ops.render.opengl(animation=False)
                logger.info("Blender object generated and rendered.")
            except Exception as be:
                logger.error(f"Blender object error: {be}")# MAIN LOGIC: Create item from voice/text command and animate it in world
        try:
            logger.info(f"3D Engine: Creating object: {object_description}")
            from UnifiedAvatarController import UnifiedAvatarController
            controller = UnifiedAvatarController()
            controller.create_avatar(object_description)  # Handles Blender/Unreal command pipe
            logger.info("Object creation dispatched to engine controller.")
        except Exception as e:
            logger.error(f"Error generating 3D object: {e}")

    def control_camera(self, direction):
        x, y, z = self.camera_position
        if direction == "left": x -= 1
        elif direction == "right": x += 1
        elif direction == "forward": z -= 1
        elif direction == "back": z += 1
        self.camera_position = (x, y, z)
        logger.info(f"Camera moved to: {self.camera_position}")
        try:
            bpy.context.scene.camera.location = (x, y, z)
            logger.info("Blender camera moved.")
        except Exception as e:
            logger.warning(f"Camera control failed: {e}")

    def enable_vr_mirror(self):
        try:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.context.window_manager.xr_session_settings.base_pose_type = 'CUSTOM'
            bpy.ops.wm.xr_session_start()
            logger.info("VR session started with viewport mirror.")
        except Exception as vr:
            logger.error(f"Failed to start VR session: {vr}")




        self.update_avatar()  # Auto-update on startup

    def load_avatar_image(self):
        try:
            avatar_path = os.path.join(globals_module.AVATAR_DIR, "avatar_rendered.jpg")
            if not os.path.exists(avatar_path):
                avatar_path = os.path.join(globals_module.AVATAR_DIR, "default_avatar.jpg")

            photo = self.avatar_controller.show_avatar()
            if photo:
                self.photo = photo
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
                logger.info("Avatar loaded using avatar_controller.")
            else:
                image = Image.open(avatar_path)
                image = image.resize((300, 300), Image.ANTIALIAS)
                self.avatar_image = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.avatar_image)
                self.image_label.image = self.avatar_image
                logger.warning("Fallback: Avatar image loaded directly.")
        except Exception as e:
            logger.error(f"Failed to load avatar image: {e}")

    def update_avatar(self):
        def _update():
            try:
                # Attempt to load avatar image (if this method does not raise, we exit early)
                self.load_avatar_image()

            except Exception as primary_exception:
                # If load_avatar_image fails, fallback to showing a random avatar
                try:
                    photo = self.avatar_controller.show_avatar()
                    if photo:
                        self.photo = photo
                        self.avatar_label.configure(image=self.photo, text="")
                        self.avatar_label.image = self.photo
                        log_gui_event("Avatar Update", "Avatar Panel updated with avatar image.")
                    else:
                        self.avatar_label.configure(text="No Avatar Available", image="")
                        log_gui_event("Avatar Update", "No avatar available for display.")

                except Exception as e:
                    # Attempt to update visual warning light (optional)
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")

                    logger.error(f"Error updating avatar: {e}")
                    log_gui_event("Avatar Update Error", str(e))

            # If load_avatar_image succeeds, log that event
            else:
                log_gui_event("Avatar Update", "Avatar loaded successfully via load_avatar_image().")

        _update()

        # Schedule the next update after 10 seconds (10000 milliseconds)
        self.frame.after(10000, self.update_avatar)
        # Alternative scheduling method from globals (commented out as requested)
        # self.after(globals_module.AVATAR_REFRESH_RATE * 1000, self.update_avatar)

# ------------------------- Files Panel (Files Tab) -------------------------
class FilesPanel:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.files_display = tk.Listbox(self.frame, height=20)
        self.files_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.options_frame = ttk.Frame(self.frame)
        self.options_frame.pack(fill=tk.X, padx=10, pady=5)
        self.upload_button = ttk.Button(self.options_frame, text="Upload Files", command=self.add_files)
        self.upload_button.pack(side=tk.LEFT, padx=5)
        self.refresh_button = ttk.Button(self.options_frame, text="Refresh List", command=self.refresh_list)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        self.uploaded_files = []

    def add_files(self):
        file_paths = filedialog.askopenfilenames(title="Select file(s) to upload")
        if file_paths:
            uploaded_count = 0
            failed_count = 0
            for f in file_paths:
                try:
                    target = self.categorize_file(f)
                    dest_path = os.path.join(target, os.path.basename(f))
                    shutil.copy(f, dest_path)
                    self.uploaded_files.append(f"{os.path.basename(f)} -> {dest_path}")
                    log_gui_event("File Sent", f"Copied {f} to {dest_path}")
                    uploaded_count += 1
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    logger.error(f"Failed to upload {f}: {e}")
                    messagebox.showerror("Error", f"Failed to upload {f}: {e}")
                    failed_count += 1
            messagebox.showinfo("Files Uploaded", f"{uploaded_count} files uploaded successfully!\n{failed_count} files failed.")
            self.refresh_list()
        else:
            logger.warning("No files selected for upload.")
            messagebox.showwarning("Upload Files", "No files selected.")

    def refresh_list(self):
        if not self.uploaded_files:
            messagebox.showinfo("Refresh List", "No files available to display.")
            logger.info("File list refresh attempted with no files.")
        self.files_display.delete(0, tk.END)
        for entry in self.uploaded_files:
            self.files_display.insert(tk.END, entry)
        log_gui_event("Files Refreshed", f"{len(self.uploaded_files)} files listed.")

    def categorize_file(self, file_path: str) -> str:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        mapping = {
            ".csv": config.DATASETS_DIR,
            ".json": config.DATASETS_DIR,
            ".pdf": os.path.join(config.DOCUMENTS_DIR, "docs"),
            ".txt": os.path.join(config.DOCUMENTS_DIR, "docs"),
            ".jpg": os.path.join(config.PROJECTS_DIR, "images"),
            ".png": os.path.join(config.PROJECTS_DIR, "images"),
            ".py": config.IMPORTS_DIR
        }
        target = mapping.get(ext, config.ADDONS_DIR)
        os.makedirs(target, exist_ok=True)
        return target

# ------------------------- Settings Panel (Settings Tab) -------------------------
class SettingsPanel:
    def __init__(self, parent, gui_instance):
        # Re-enforce Voice Profile after GUI load
        try:
            from SarahMemoryVoice import set_voice_profile, set_pitch, set_bass, set_treble

            settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    data = json.load(f)
                if "voice_profile" in data:
                    set_voice_profile(data["voice_profile"])
                    if "pitch" in data:
                        set_pitch(data["pitch"])
                    if "bass" in data:
                        set_bass(data["bass"])
                    if "treble" in data:
                        set_treble(data["treble"])
                    logger.info(f"🔊 Re-applied voice profile after GUI load: {data['voice_profile']}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to re-apply voice profile in GUI: {e}")

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.gui = gui_instance
        self.settings = self.load_settings()

        tk.Label(self.frame, text="AI Mode:").pack(pady=5)
        self.api_mode = tk.StringVar(value=self.settings.get("api_mode", "Any"))
        self.data_mode_var = self.api_mode  # ✅ Ensure variable exists
        ttk.Combobox(self.frame, textvariable=self.api_mode, values=["Any", "Local", "Web", "API"]).pack()

        tk.Label(self.frame, text="Voice Profile:").pack(pady=5)
        self.voice_profile = tk.StringVar(value=self.settings.get("voice_profile", "Default"))
        ttk.Combobox(self.frame, textvariable=self.voice_profile, values=voice.get_voice_profiles()).pack(pady=5)

        for label, var_name in [("Pitch", "pitch"), ("Bass", "bass"), ("Treble", "treble")]:
            tk.Label(self.frame, text=label).pack(pady=5)
            var = tk.DoubleVar(value=self.settings.get(var_name, 1.0))
            setattr(self, var_name, var)
            tk.Scale(self.frame, variable=var, from_=0.5, to=2.0, resolution=0.1, orient="horizontal").pack(fill=tk.X)

        tk.Button(self.frame, text="Import Custom Voice", command=self.import_voice).pack(pady=5)

        tk.Label(self.frame, text="Avatar Settings", font=("Arial", 12, "bold")).pack(pady=10)
        self.avatar_selection = tk.StringVar(value=self.settings.get("avatar_file", ""))
        self.avatar_dropdown = ttk.Combobox(self.frame, textvariable=self.avatar_selection, values=self.get_uploaded_avatars())
        self.avatar_dropdown.pack(pady=5)
        tk.Button(self.frame, text="Upload Avatar", command=self.upload_avatar).pack(pady=5)

        tk.Label(self.frame, text="3D Engine:").pack(pady=5)
        self.engine_selection = tk.StringVar(value=self.settings.get("engine_selection", "Auto"))
        ttk.Combobox(self.frame, textvariable=self.engine_selection, values=["Microsoft3DViewer", "Blender", "Unreal", "Auto"]).pack(pady=5)

        tk.Label(self.frame, text="Theme:").pack(pady=5)
        self.theme_selection = tk.StringVar(value=self.settings.get("theme", ""))
        theme_files = []
        if os.path.exists(config.THEMES_DIR):
            theme_files = [f for f in os.listdir(config.THEMES_DIR) if f.lower().endswith(".css")]
        if not self.theme_selection.get() and theme_files:
            self.theme_selection.set(theme_files[0])
        ttk.Combobox(self.frame, textvariable=self.theme_selection, values=theme_files).pack(pady=5)

        tk.Button(self.frame, text="Save Settings", command=self.save_settings).pack(pady=10)

    def load_settings(self):
        if os.path.exists(config.SETTINGS_FILE):
            try:
                with open(config.SETTINGS_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error("Failed to load settings: " + str(e))

        return {
            "api_mode": "Local",
            "voice_profile": "female",
            "pitch": 1.0,
            "bass": 1.0,
            "treble": 1.0,
            "avatar_file": "",
            "engine_selection": "Auto",
            "theme": ""
        }

    def get_uploaded_avatars(self):
        avatars = []
        if os.path.exists(config.AVATAR_DIR):
            avatars = [file for file in os.listdir(config.AVATAR_DIR) if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
        return avatars

    def upload_avatar(self):
        path = filedialog.askopenfilename(title="Select Avatar Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
        if path:
            try:
                dest = os.path.join(config.AVATAR_DIR, os.path.basename(path))
                shutil.copy(path, dest)
                messagebox.showinfo("Avatar Upload", "Avatar uploaded successfully!")
                log_gui_event("Avatar Upload", f"Copied avatar {path} to {dest}")
                self.avatar_dropdown['values'] = self.get_uploaded_avatars()
                self.avatar_selection.set(os.path.basename(path))
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error(f"Failed to upload avatar: {e}")
                messagebox.showerror("Error", f"Failed to upload avatar: {e}")

    def import_voice(self):
        path = filedialog.askopenfilename(title="Select Voice Profile")
        if path:
            try:
                voice.import_custom_voice_profile(path)
                log_gui_event("Import Voice", f"Imported voice profile from: {path}")
                messagebox.showinfo("Voice Import", "Voice profile imported successfully!")
            except Exception as e:
                try:
                    if hasattr(config, 'vision_canvas'):
                        config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
                logger.error(f"Failed to import voice profile: {e}")
                messagebox.showerror("Error", f"Failed to import voice profile: {e}")

    def save_settings(self):
        from SarahMemoryVoice import set_voice_profile, set_pitch, set_bass, set_treble, save_voice_settings

        data = {
            "api_mode": self.api_mode.get(),
            "voice_profile": self.voice_profile.get(),
            "pitch": self.pitch.get(),
            "bass": self.bass.get(),
            "treble": self.treble.get(),
            "avatar_file": self.avatar_selection.get(),
            "engine_selection": self.engine_selection.get(),
            "theme": self.theme_selection.get()
        }

        selected_mode = self.api_mode.get().lower()
        self.gui.override_data_mode = selected_mode
        try:
            import SarahMemoryGlobals as config
            if selected_mode == "any":
                config.LOCAL_DATA_ENABLED = True
                config.WEB_RESEARCH_ENABLED = True
                config.API_RESEARCH_ENABLED = True
                config.ROUTE_MODE = "Any"
            elif selected_mode == "local":
                config.LOCAL_DATA_ENABLED = True
                config.WEB_RESEARCH_ENABLED = False
                config.API_RESEARCH_ENABLED = False
                config.ROUTE_MODE = "Local"
            elif selected_mode == "web":
                config.LOCAL_DATA_ENABLED = False
                config.WEB_RESEARCH_ENABLED = True
                config.API_RESEARCH_ENABLED = False
                config.ROUTE_MODE = "Web"
            elif selected_mode == "api":
                config.LOCAL_DATA_ENABLED = False
                config.WEB_RESEARCH_ENABLED = False
                config.API_RESEARCH_ENABLED = True
                config.ROUTE_MODE = "API"
        except Exception as _e:
            logger.warning(f"Mode flag update failed: {_e}")


        log_gui_event("Mode Override", f"Mode changed to: {selected_mode.upper()}")

        try:
            with open(config.SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=4)

            voice.set_voice_profile(data["voice_profile"])
            voice.set_pitch(data["pitch"])
            voice.set_bass(data["bass"])
            voice.set_treble(data["treble"])

            if data["theme"]:
                apply_theme_from_choice(data["theme"])

            save_voice_settings()

            self.gui.update_status()

            messagebox.showinfo("Settings", "Settings saved successfully!")
            log_gui_event("Settings Saved", str(data))

        except Exception as e:
            try:
                if hasattr(config, 'vision_canvas'):
                    config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
            except Exception as ce:
                logger.warning(f"Vision light update failed: {ce}")

            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

# ------------------------- Status Bar -------------------------

class StatusBar:
    def __init__(self, parent):
        # Initialize status bar with three equal-sized status lights: intent, object, network.
        self.frame = ttk.Frame(parent, style="StatusBar.TFrame")
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        # status text
        self.status_label = ttk.Label(self.frame, text="", anchor=tk.W, style="StatusBar.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Intent status light
        self.intent_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.intent_canvas.pack(side=tk.RIGHT, padx=5)
        self.intent_light = self.intent_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Object detection status light
        self.object_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.object_canvas.pack(side=tk.RIGHT, padx=5)
        self.object_light = self.object_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Network status light
        self.network_canvas = tk.Canvas(self.frame, width=20, height=20, highlightthickness=0)
        self.network_canvas.pack(side=tk.RIGHT, padx=5)
        self.network_indicator = self.network_canvas.create_oval(2, 2, 18, 18, fill=config.STATUS_LIGHTS['green'])
        # Start periodic status updates
        self.update_status()

    def set_intent_light(self, color):
        # color: 'green', 'yellow', or 'red'
        self.intent_canvas.itemconfig(self.intent_light, fill=config.STATUS_LIGHTS[color])

    def set_object_light(self, color):
        self.object_canvas.itemconfig(self.object_light, fill=config.STATUS_LIGHTS[color])
    def set_status(self, text):
        """Update the status bar text from other modules (Phase B: shared GUI status hook)."""
        try:
            self.status_label.config(text=text)
        except Exception as e:
            logger.warning(f"[StatusBar] Failed to set status text: {e}")


    def update_status(self):
        try:
            settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                mode = settings.get("api_mode", "Local")
            else:
                mode = "Local"
        except Exception as e:
            logger.warning(f"⚠️ Failed to read settings for status bar: {e}")
            mode = "Local"

        # Phase A/B: runtime identity & safety flags surfaced in the GUI
        run_mode = getattr(config, "RUN_MODE", "local")
        device_mode = getattr(config, "DEVICE_MODE", "local_agent")
        device_profile = getattr(config, "DEVICE_PROFILE", "Standard")
        safe_mode = bool(getattr(config, "SAFE_MODE", False))
        local_only = bool(getattr(config, "LOCAL_ONLY_MODE", False))

        voice_profile = voice.get_voice_profiles()[0] if voice.get_voice_profiles() else "Default"
        avatar_file = "None"
        global MIC_STATUS, CAMERA_STATUS
        status_text = (
            f"Mode: {mode} | Runtime: {run_mode}/{device_mode}/{device_profile} | "
            f"Safe:{'On' if safe_mode else 'Off'} LocalOnly:{'On' if local_only else 'Off'} | "
            f"Voice: {voice_profile} | Avatar: {avatar_file} | Mic: {MIC_STATUS} | Camera: {CAMERA_STATUS}"
        )
        self.status_label.config(text=status_text)
        #self.status_bar.display_message("© 2025 Brian Lee Baros.")

        try:
            network_state = async_update_network_state_sync()
        except Exception as e:
            try:
                if hasattr(config, 'vision_canvas'):
                    config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
            except Exception as ce:
                logger.warning(f"Vision light update failed: {ce}")
            network_state = "red"

        color = {"red": "red", "yellow": "yellow", "green": "green"}.get(network_state, "green")
        self.network_canvas.itemconfig(self.network_indicator, fill=color)

        self.frame.after(1000, self.update_status)


# ------------------------- Main Unified GUI -------------------------
class SarahMemoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=0)  # Tabs
        self.root.rowconfigure(1, weight=0)  # Top button bar
        self.root.rowconfigure(2, weight=1)  # Main content
        self.root.rowconfigure(3, weight=0)  # Status bar
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Set application title and geometry
        self.root.title("SarahMemory AI-Bot Companion Platform")
        # A slightly smaller default size for better cross‑platform compatibility
        self.root.geometry("1600x1000")
        # ---------------------------------------------------------------------------------
        # [Updater] Apply a cohesive ttk style to give the GUI a more professional look.
        # This style uses a light background, consistent fonts, and subtle accent colours.
        # If ttk.Style is not available or fails, the defaults will remain unaffected.
        try:
            style = ttk.Style(self.root)
            # Base frame styling
            style.configure('TFrame', background='#f5f5f5')
            style.configure('TLabel', background='#f5f5f5', foreground='#333333', font=('Helvetica', 11))
            style.configure('TButton', font=('Helvetica', 11), padding=(6, 3))
            # Notebook styling
            style.configure('TNotebook', background='#f5f5f5', borderwidth=0)
            style.configure('TNotebook.Tab', padding=(10, 5), font=('Helvetica', 11, 'bold'))
            # Status bar styling
            style.configure('StatusBar.TFrame', background='#eaeaea')
            style.configure('StatusBar.TLabel', background='#eaeaea', foreground='#333333', font=('Helvetica', 10))
        except Exception as e:
            logger.warning(f"Style configuration failed: {e}")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.override_data_mode = "api"  # Default value



        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.video_chat_tab = ttk.Frame(self.notebook)
        self.files_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.video_chat_tab, text="Video Chat")
        self.notebook.add(self.files_tab, text="Files")
        self.notebook.add(self.settings_tab, text="Settings")

        self.connection_panel = ConnectionPanel(self.video_chat_tab, open_settings_window)

        self.main_frame = ttk.Frame(self.video_chat_tab)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=2)
        self.main_frame.rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(self.main_frame)



        self.avatar_controller = ExtendedAvatarController()  # Create a single instance here
        self.chat_panel = ChatPanel(self.main_frame, self, self.avatar_controller)
        #self.avatar_panel = AvatarPanel(self.main_frame)
        #globals_module.avatar_panel_instance = self.avatar_panel  # ✅ Allow external update after avatar render

        self.video_panel.frame.grid(row=0, column=0, sticky="nsew")
        self.chat_panel.frame.grid(row=0, column=1, sticky="nsew")
        #self.avatar_panel.frame.grid(row=0, column=2, sticky="nsew")

        self.files_panel = FilesPanel(self.files_tab)
        self.settings_panel = SettingsPanel(self.settings_tab, self)

        # Bottom frame for Control Panel and Status Bar.
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Control Panel: Theme selection drop-down and Apply button.
        self.control_panel = ttk.Frame(self.bottom_frame)
        self.control_panel.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.theme_var = tk.StringVar()
        if os.path.exists(config.THEMES_DIR):
            theme_files = [f for f in os.listdir(config.THEMES_DIR) if f.lower().endswith(".css")]
        else:
            theme_files = []
        self.theme_var.set(theme_files[0] if theme_files else "")
        theme_dropdown = ttk.Combobox(self.control_panel, textvariable=self.theme_var, values=theme_files)
        theme_dropdown.pack(side=tk.LEFT, padx=5)
        apply_theme_button = ttk.Button(self.control_panel, text="Apply Theme", command=lambda: apply_theme_from_choice(self.theme_var.get()))
        apply_theme_button.pack(side=tk.LEFT, padx=5)

        # Status Bar: Using the new StatusBar class with ttk styling.
        self.status_bar = StatusBar(self.bottom_frame)
        # Phase B: expose status bar globally so backend modules can update it.
        try:
            config.status_bar = self.status_bar
        except Exception:
            pass

        self.start_voice_recognition_loop()
        self.start_SuperObjectEngine()


        self.last_user_interaction = time.time()
        self.check_idle_thread = threading.Thread(target=self.monitor_idle_time, daemon=True)
        self.check_idle_thread.start()


        self.root.after(3500, self.intro_greeting)
        log_gui_event("GUI Init", "SarahMemoryGUI initialized with previous settings.")

        # 🔁 Avatar Panel GPU Launcher
        import subprocess
        try:
            self.avatar_proc = subprocess.Popen(["python", "SarahMemoryAvatarPanel.py"])
            #subprocess.Popen(["python", "SarahMemoryAvatarPanel.py"])
            logger.info("✅ AvatarPanel GPU window launched in parallel.")
        except Exception as launch_error:
            logger.error(f"❌ Failed to launch AvatarPanel: {launch_error}")


    def intro_greeting(self):
        # Safe logger
        try:
            import logging
            _log = logging.getLogger(__name__)
        except Exception:
            class _Null:
                def error(self, *a, **k): pass
            _log = _Null()

        # Default fallback greeting
        greeting_text = "Well Hello there! How may be of assistance today?"

        # Try to get a personalized greeting
        try:
            from SarahMemoryPersonality import get_greeting_response
            g = get_greeting_response()
            if isinstance(g, str) and g.strip():
                greeting_text = g.strip()
        except Exception as e:
            _log.error("[Greeting] get_greeting_response failed: %s", e)

        # Show it in the chat
        try:
            self.chat_panel.append_message("Sarah: " + greeting_text)
        except Exception as e:
            _log.error("[Greeting] append_message failed: %s", e)

        # Speak once at boot (Reply layer is not invoked for the greeting)
        try:
            import SarahMemoryGlobals as config
            if getattr(config, "VOICE_FEEDBACK_ENABLED", True):
                from SarahMemoryAiFunctions import synthesize_voice
                synthesize_voice(greeting_text)
        except Exception as e:
            _log.error("[Greeting] synthesize_voice failed: %s", e)

        # Log the greeting event using the exact text shown
        try:
            from SarahMemoryDatabase import log_ai_functions_event
            log_ai_functions_event("Greeting", greeting_text)
        except Exception as e:
            _log.error("[Greeting] log_ai_functions_event failed: %s", e)

        # Ensure avatar flag is reset
        try:
            import SarahMemoryGlobals as config  # re-import safe
            config.AVATAR_IS_SPEAKING = False
        except Exception:
            pass

    def update_avatar_window(self):
        self.avatar_panel.update_avatar()

    def start_voice_recognition_loop(self):
        def voice_loop():
            while True:
                try:
                    # Listen for voice input using the AI module's combined listen_and_process function.
                    recognized = voice.listen_and_process()
                    if recognized:
                        logger.info(f"Voice input received: {recognized}")
                        # Check for avatar commands first.
                        lower_text = recognized.lower()
                        if "create your avatar" in lower_text or "change your" in lower_text:
                            logger.info(f"Avatar command detected: {recognized}")
                            if "create your avatar" in lower_text:
                                self.create_avatar(recognized)
                            elif "change your" in lower_text:
                                self.modify_avatar(recognized)
                        else:
                            # Otherwise, treat as chat input.
                            self.chat_panel.append_message("You (voice): " + recognized)
                            run_async(self.chat_panel.generate_response, recognized)
                except Exception as e:
                    logger.error(f"Voice recognition error: {e}")
                    time.sleep(5)
        run_async(voice_loop)


    def start_SuperObjectEngine(self):
        def detection_loop():
            while True:
                try:
                    with shared_lock:
                        frame = shared_frame.copy() if shared_frame is not None else None
                        if frame is not None:
                            objects = sobje.ultra_detect_objects(frame)
                            if objects:
                                log_gui_event("Object Detection", f"Detected objects: {', '.join(objects)}")
                    time.sleep(3)
                except Exception as e:
                    try:
                        if hasattr(config, 'vision_canvas'):
                            config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                    except Exception as ce:
                        logger.warning(f"Vision light update failed: {ce}")
                    logger.error(f"Object detection error: {e}")
                    time.sleep(20)
        run_async(detection_loop)

    def monitor_idle_time(self):
        while True:
            idle_time = time.time() - self.last_user_interaction
            if idle_time > config.DL_IDLE_TIMER:  # Idle Timer is set in SarahMemoryGlobals.py
                try:
                    from SarahMemoryOptimization import run_idle_optimization_tasks
                    run_idle_optimization_tasks()
                except Exception as e:
                    logger.warning(f"[Idle Optimization Error] {e}")
            time.sleep(300)  # Check every 5 minutes

    def shutdown(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            log_gui_event("Shutdown", "User exited the GUI.")
            self.video_panel.release_resources()
            voice.shutdown_tts()
            if hasattr(self, 'avatar_proc') and self.avatar_proc.poll() is None:
                self.avatar_proc.terminate()
            self.root.destroy()
            try:
                root.after_cancel(update_video_job_id)
                root.after_cancel(update_status_job_id)
                root.after_cancel(update_avatar_job_id)
                root.after_cancel(self.video_update_job)
            except Exception:
                pass
            try:
                update_video_job_id = root.after(1000, update_video)
                update_status_job_id = root.after(1000, update_status)
                update_avatar_job_id = root.after(1000, update_avatar)
                root.after_cancel(self.status_update_job)
            except Exception:
                pass
            logger.info("SarahMemory GUI shutdown successfully.")
        else:
            logger.info("Shutdown cancelled by user.")



# ------------------------- Settings Window -------------------------
def open_settings_window():
    try:
        win = tk.Toplevel()
        win.title("Settings")
        win.geometry("320x240")
        SettingsPanel(win, win)
        logger.info("Settings window opened successfully.")
    except Exception as e:
        try:
            if hasattr(config, 'vision_canvas'):
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
        logger.error(f"Error opening settings window: {e}")
        messagebox.showerror("Error", "Failed to open the settings window.")

#================----------Place SUPER OBJECT DETECTION ENGINE HERE----------================
class SuperObjectEngine:
    def __init__(self):
        from SarahMemorySOBJE import ultra_detect_objects
        self.detect = ultra_detect_objects

    def detect_objects(self, frame):
        try:
            return self.detect(frame)
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
#==========================END OF SUPER OBJECT DETECTION ENGINE HERE===========
# ------------------------- Main Execution -------------------------
def run_gui():
    import SarahMemoryGlobals as config

    # Headless guard: if there's no graphical display, don't attempt to launch GUI/webview.
    if not _sm_has_display():
        print("[SarahMemoryGUI] No DISPLAY detected — running in headless mode, GUI will not launch.")
        return

    # Prefer modern Web UI if enabled in globals
    ui_mode = getattr(config, "UI_MODE", "classic")
    if bool(getattr(config, "USE_WEBVIEW", False)) and ui_mode in ("web", "custom"):
        try:
            import webview  # pywebview
        except Exception as e:
            print("[WebUI] pywebview not available → falling back to legacy Tk GUI.", e)
        else:
            import os, json, io, base64, time
            from PIL import Image

            # Decide which HTML file to load based on UI_MODE
            html_path = None

            if ui_mode == "custom":
                # New React/Vite V8 build (same path used by SarahMemoryUIupdater.py)
                base_dir = getattr(config, "BASE_DIR", os.getcwd())
                html_path = os.path.join(base_dir, "data", "ui", "V8", "index.html")
            elif ui_mode == "web":
                # Original app.js-based WebUI
                html_path = getattr(config, "WEBUI_HTML_PATH", None)
                if not html_path:
                    # Fallback: root index.html in the project directory
                    base_dir = getattr(config, "BASE_DIR", os.getcwd())
                    html_path = os.path.join(base_dir, "index.html")

            # Resolve URL:
            # For Web-based UIs (legacy/custom), prefer the local Flask server URL.
            # This avoids file:/// loading (Vite builds commonly go blank due to /assets paths).
            url = None
            try:
                host = getattr(config, "DEFAULT_HOST", "127.0.0.1")
                port = int(getattr(config, "DEFAULT_PORT", 5500))
                if ui_mode in ("web", "custom"):
                    url = f"http://{host}:{port}/"
            except Exception:
                url = None

            # Fallback to local file if needed (e.g., API server not running yet)
            if not url:
                if html_path and os.path.exists(html_path):
                    # Normalized file:// URL (⚠ backslashes must be escaped)
                    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
                else:
                    # If the local file is missing, fall back to the hosted SarahMemory HTML
                    url = "https://www.sarahmemory.com/api/data/ui/SarahMemory.html"

            # --- JS Bridge expected by /data/ui/app.js ---
            class _WebBridge:
                def get_status(self):
                    # Intent/Network/Vision LEDs for WebUI
                    import time as _t
                    try:
                        from SarahMemoryGUI import async_update_network_state_sync as _net
                        ns = _net()
                        net = 'green' if (isinstance(ns, dict) and ns.get('online', True)) else 'red'
                    except Exception:
                        net = 'yellow'
                    intent = 'yellow' if (getattr(self, '_last_send_ts', 0) and
                                          _t.time() - getattr(self, '_last_send_ts', 0) < 2.0) else 'green'
                    vision = 'green'
                    return {'intent': intent, 'vision': vision, 'network': net, 'text': 'Ready'}

                def speak(self, text):
                    try:
                        if not text:
                            return False
                        from SarahMemoryVoice import synthesize_voice
                        synthesize_voice(text)
                        return True
                    except Exception:
                        return False

                def get_greeting(self):
                    try:
                        from SarahMemoryPersonality import get_greeting_response
                        g = get_greeting_response()
                        return {'text': g}
                    except Exception:
                        return {'text': "Hello! I'm Sarah — ready when you are."}

                # Bridge methods intentionally match calls found in app.js:
                # get_boot_state, set_flag, send_message, list_threads_for_date,
                # list_reminders, create_reminder, get_snapshot
                def get_boot_state(self):
                    try:
                        import datetime as _dt
                        d = {
                            'reply_status': bool(getattr(config, 'REPLY_STATUS', True)),
                            'compare_trainer': bool(getattr(config, 'API_RESPONSE_CHECK_TRAINER', False)),
                            'route_mode': str(getattr(config, 'ROUTE_MODE', 'Any')),
                        }
                        d['REPLY_STATUS'] = d['reply_status']
                        d['API_RESPONSE_CHECK_TRAINER'] = d['compare_trainer']
                        d['today'] = _dt.date.today().isoformat()
                        return d
                    except Exception:
                        return {"reply_status": True, "compare_trainer": False, "route_mode": "Any"}

                def set_flag(self, name, value):
                    try:
                        if name in ("reply_status", "REPLY_STATUS"):
                            setattr(config, "REPLY_STATUS", bool(value))
                        elif name in ("compare_trainer", "API_RESPONSE_CHECK_TRAINER"):
                            setattr(config, "API_RESPONSE_CHECK_TRAINER", bool(value))
                        elif name in ("route_mode", "ROUTE_MODE"):
                            setattr(config, "ROUTE_MODE", str(value))
                        return True
                    except Exception:
                        return False

                def send_message(self, text, file_path=None):
                    # Minimal integration with existing Reply pipeline
                    try:
                        from SarahMemoryReply import generate_reply
                        bundle = generate_reply(None, text)  # ChatPanel 'self' is unused in v7.7.4+
                        # Ensure plain dict with expected keys
                        resp = {
                            "response": (bundle.get("response") if isinstance(bundle, dict) else str(bundle)),
                            "links": (bundle.get("links") if isinstance(bundle, dict) else []),
                            "meta": (bundle.get("meta") if isinstance(bundle, dict) else {}),
                        }
                        try:
                            from SarahMemoryGlobals import VOICE_FEEDBACK_ENABLED
                            if VOICE_FEEDBACK_ENABLED:
                                from SarahMemoryVoice import synthesize_voice
                                synthesize_voice(resp.get('response', ''))
                        except Exception:
                            pass
                        return resp
                    except Exception as e:
                        return {
                            "response": f"[ERROR] {e}",
                            "links": [],
                            "meta": {"intent": "error", "source": "local"},
                        }

                def list_threads_for_date(self, iso_date):
                    # Optional: return empty list to keep UI responsive without DB
                    try:
                        from SarahMemoryDatabase import fetch_conversation_threads_for_date
                        return fetch_conversation_threads_for_date(iso_date)  # if implemented
                    except Exception:
                        return []

                def list_reminders(self):
                    # Optional: use settings.json 'reminders' field if present
                    try:
                        settings_file = os.path.join(config.SETTINGS_DIR, "settings.json")
                        if os.path.exists(settings_file):
                            with open(settings_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            return data.get("reminders", [])
                    except Exception:
                        pass
                    return []

                def create_reminder(self, title, when, note=None):
                    # Best-effort: append to settings.json 'reminders' list (non-breaking)
                    try:
                        settings_file = os.path.join(config.SETTINGS_DIR, "settings.json")
                        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
                        data = {}
                        if os.path.exists(settings_file):
                            with open(settings_file, "r", encoding="utf-8") as f:
                                data = json.load(f) or {}
                        rem = data.get("reminders", [])
                        rem.append({"title": title, "when": when, "note": note or ""})
                        data["reminders"] = rem
                        with open(settings_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        return True
                    except Exception:
                        return False

                def get_snapshot(self):
                    """Provide a small screenshot or placeholder as data URL."""
                    try:
                        # Headless-safe: if pyautogui is unavailable or no display, make a solid placeholder
                        if (not globals().get("_SM_HAVE_PYAUTOGUI", False) or
                                pyautogui is None or
                                not _sm_has_display()):
                            img = Image.new("RGB", (640, 160), (10, 12, 16))
                        else:
                            img = pyautogui.screenshot()
                    except Exception:
                        img = Image.new("RGB", (640, 160), (10, 12, 16))

                    try:
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        return {"data_url": "data:image/png;base64," + b64}
                    except Exception:
                        return {"data_url": None}

                def get_themes(self):
                    try:
                        if os.path.isdir(config.THEMES_DIR):
                            return [f for f in os.listdir(config.THEMES_DIR) if f.lower().endswith(".css")]
                        return []
                    except Exception:
                        return []

                def set_theme(self, css_name: str):
                    """Copy the chosen theme from /data/mods/themes into the WebUI as unified-theme.css."""
                    try:
                        src = os.path.join(config.THEMES_DIR, css_name)
                        dst = os.path.join(config.BASE_DIR, "data", "ui", "unified-theme.css")
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copyfile(src, dst)
                        return True
                    except Exception:
                        return False

                def open_avatar_panel(self):
                    """Launch the separate GPU Avatar Panel window."""
                    try:
                        exe = sys.executable or "python"
                        subprocess.Popen([exe, "SarahMemoryAvatarPanel.py"], close_fds=True)
                        return True
                    except Exception:
                        return False

                def list_voices(self):
                    """Return available TTS voices if supported by SarahMemoryVoice; otherwise []"""
                    try:
                        from SarahMemoryVoice import list_voices
                        return list_voices() or []
                    except Exception:
                        return []

                def set_voice(self, opts=None):
                    try:
                        opts = opts or {}
                        from SarahMemoryVoice import configure_voice
                        configure_voice(opts)
                        return True
                    except Exception:
                        return False

                def open_thread(self, thread_id):
                    # TODO: integrate with DB; placeholder
                    try:
                        return True
                    except Exception:
                        return False

                def save_contact(self, name, addr):
                    try:
                        path = os.path.join(config.SETTINGS_DIR, "contacts.json")
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        existing = []
                        if os.path.exists(path):
                            with open(path, "r", encoding="utf-8") as f:
                                existing = json.load(f) or []
                        existing.append({"name": name, "addr": addr})
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=2)
                        return True
                    except Exception:
                        return False

                def start_network_chat(self, addr):
                    try:
                        from SarahMemoryNetwork import start_chat_with
                        start_chat_with(addr)
                        return True
                    except Exception:
                        return False

                def transcribe_once(self, timeout_sec=10):
                    """Fallback speech capture via Python if browser SR is unavailable."""
                    try:
                        from SarahMemoryVoice import transcribe_once as _tr
                        return {"text": _tr(timeout=timeout_sec) or ""}
                    except Exception:
                        return {"text": ""}

            # Launch window
            webview.create_window(
                "SarahMemory — Web UI",
                url,
                js_api=_WebBridge(),
                width=1180,
                height=840,
                resizable=True
            )
            try:
                # Serve files over localhost to grant persistent media permissions
                webview.start(http_server=True)  # choose best backend automatically
            except Exception:
                pass
            return  # do not launch legacy Tk when Web UI is active

    # --- Legacy Tk GUI fallback ---

    root = tk.Tk() if _sm_has_display() else None if _sm_has_display() else None
    app = SarahMemoryGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    logger.info("Starting unified GUI mainloop.")
    root.mainloop() if root is not None else None if root is not None else None

if __name__ == '__main__':
    run_gui()
try:
    from PIL import Image, ImageTk
    import io, urllib.request
except Exception:
    Image = None

def _sm_download_image(url: str):
    if not Image: return None
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read()
        return Image.open(io.BytesIO(data))
    except Exception:
        return None

def insert_bundle(self, bundle):
    txt = bundle.get("response") or bundle.get("text") or ""
    if txt:
        try:
            self.add_assistant_text(txt + "\n")
        except Exception:
            try:
                self.chat_text.insert("end", txt + "\n"); self.chat_text.see("end")
            except Exception:
                pass
    img_url = bundle.get("image_url")
    if img_url and Image:
        img = _sm_download_image(img_url)
        if img:
            try:
                img.thumbnail((512,512))
                photo = ImageTk.PhotoImage(img)
                if not hasattr(self, "_img_refs"): self._img_refs = []
                self._img_refs.append(photo)
                self.chat_text.image_create("end", image=photo)
                self.chat_text.insert("end", "\n"); self.chat_text.see("end")
            except Exception:
                pass
# ============================================================================

# ================== Mini Browser Panel ==================
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
class MiniBrowser:
    """A minimal HTML viewer using tkinterweb if available, else fallback Text with clickable links."""
    def __init__(self, parent):
        self.parent = parent
        self.history = []
        self.index = -1
        self.addr_var = tk.StringVar() if tk else None

        self.frame = ttk.Frame(parent)
        top = ttk.Frame(self.frame)
        top.pack(fill="x")
        self.addr = ttk.Entry(top, textvariable=self.addr_var, width=80) if tk else None
        if self.addr:
            self.addr.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        ttk.Button(top, text="Go", command=self._go).pack(side="left", padx=2)
        ttk.Button(top, text="Back", command=self.back).pack(side="left", padx=2)
        ttk.Button(top, text="Forward", command=self.forward).pack(side="left", padx=2)
        ttk.Button(top, text="Open Externally", command=self.open_external).pack(side="left", padx=2)

        # Viewer
        self.html_widget = None
        try:
            from tkinterweb import HtmlFrame  # pip install tkinterweb
            self.html_widget = HtmlFrame(self.frame, horizontal_scrollbar="auto")
            self.html_widget.pack(fill="both", expand=True)
        except Exception:
            # fallback simple Text
            self.text = tk.Text(self.frame, wrap="word", height=18)
            self.text.pack(fill="both", expand=True)
            self.text.insert("end", "MiniBrowser fallback ready. Install 'tkinterweb' for richer HTML.")

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def _go(self):
        if not self.addr: return
        url = self.addr_var.get().strip()
        if url:
            self.open(url)

    def open(self, url_or_html):
        """If looks like URL, load; else treat as raw HTML."""
        if self.addr and (url_or_html.startswith("http://") or url_or_html.startswith("https://")):
            self.addr_var.set(url_or_html)
        self.history = self.history[:self.index+1] + [url_or_html]
        self.index += 1
        self._render(url_or_html)

    def _render(self, url_or_html):
        try:
            if self.html_widget:
                if url_or_html.startswith("http://") or url_or_html.startswith("https://"):
                    self.html_widget.load_website(url_or_html)
                else:
                    self.html_widget.set_content(url_or_html)
            else:
                self.text.delete("1.0","end")
                self.text.insert("end", url_or_html)
        except Exception as e:
            pass

    def back(self):
        if self.index > 0:
            self.index -= 1
            self._render(self.history[self.index])

    def forward(self):
        if self.index + 1 < len(self.history):
            self.index += 1
            self._render(self.history[self.index])

    def open_external(self):
        try:
            import webbrowser
            if self.index >= 0:
                val = self.history[self.index]
                if val.startswith("http://") or val.startswith("https://"):
                    webbrowser.open(val)
        except Exception:
            pass
#========================== Unified communications (Pro) ==========================
class UnifiedCommsProPanel:

    def __init__(self, parent, video_panel: 'VideoPanel' = None):
        self.parent = parent
        self.video_panel = video_panel
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.EXPORTS_DIR, exist_ok=True)
        self._ensure_tables()

        # Mesh/telephony/voice/video wiring
        self._mesh_ok = False
        self._attach_network_services()

        # Runtime state
        self.current_peer_id = None
        self.call_active = False
        self.call_start_ts = None
        self.call_recording = False
        self.call_on_hold = False
        self.local_mute = False
        self.output_volume = 1.0
        self._audio_rec_chunks = []
        self._audio_rec_path = None
        self._qos_last = {"rtt_ms": 0, "jitter_ms": 0, "loss": 0.0}
        self._last_remote_snapshot = None
        self._device_lists = {"mic": [], "speaker": [], "camera": []}

        # UI: notebook with five tabs
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.nb = ttk.Notebook(self.frame); self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_calls = ttk.Frame(self.nb)
        self.tab_msgs = ttk.Frame(self.nb)
        self.tab_contacts = ttk.Frame(self.nb)
        self.tab_keypad = ttk.Frame(self.nb)
        self.tab_devices = ttk.Frame(self.nb)
        self.nb.add(self.tab_calls, text="Recent Calls")
        self.nb.add(self.tab_msgs, text="Messages")
        self.nb.add(self.tab_contacts, text="Contacts")
        self.nb.add(self.tab_keypad, text="Keypad")
        self.nb.add(self.tab_devices, text="Devices/QoS")

        # Build each tab
        self._build_calls_tab()
        self._build_messages_tab()
        self._build_contacts_tab()
        self._build_keypad_tab()
        self._build_devices_tab()

        # Bottom call controls bar
        self._build_call_controls_bar()

        # Periodic network/QoS updates
        self._poll_job = self.frame.after(1200, self._poll_network_state)
        self._qos_job = self.frame.after(1500, self._poll_qos_metrics)

        # Load initial lists
        self._load_contacts()
        self._load_calls()

    # -----------------------------------------------------------------------------
    # DB schema and helpers
    # -----------------------------------------------------------------------------
    def _ensure_tables(self):
        con = sqlite3.connect(self.DB_PATH)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, email TEXT, phone TEXT,
                website TEXT, addr TEXT,
                image_path TEXT, notes TEXT,
                ts REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, peer TEXT, direction TEXT,
                status TEXT, duration REAL,
                recording_path TEXT, rtt_ms REAL, jitter_ms REAL, loss REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, peer TEXT, direction TEXT,
                text TEXT, delivered INTEGER DEFAULT 1,
                reactions TEXT   -- JSON: [{emoji:"👍", ts:...}, ...]
            )
        """)
        con.commit(); con.close()

    def _db(self):
        return sqlite3.connect(self.DB_PATH)

    # -----------------------------------------------------------------------------
    # Network attach and services
    # -----------------------------------------------------------------------------
    def _attach_network_services(self):
        try:
            from SarahMemoryNetwork import get_default_server_connection, attach_extended_services, NetworkProtocol
            self.server = get_default_server_connection()
            self.node = getattr(self.server, "mesh", None)
            if self.node:
                attach_extended_services(self.node, allow_external_sip=False)
                self._mesh_ok = True
            else:
                self._mesh_ok = False
        except Exception as e:
            logger.warning(f"[CommsPro] Mesh attach failed: {e}")
            self.server = None
            self.node = None
            self._mesh_ok = False

    # -----------------------------------------------------------------------------
    # Calls tab
    # -----------------------------------------------------------------------------
    def _build_calls_tab(self):
        # Toolbar
        tbar = ttk.Frame(self.tab_calls); tbar.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(tbar, text="Refresh", command=self._load_calls).pack(side=tk.LEFT)
        ttk.Button(tbar, text="Export CSV", command=self._export_calls_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(tbar, text="Export JSON", command=self._export_calls_json).pack(side=tk.LEFT, padx=6)

        # List with scrollbar
        wrap = ttk.Frame(self.tab_calls); wrap.pack(fill=tk.BOTH, expand=True)
        self.calls_list = tk.Listbox(wrap, height=14)
        sb = ttk.Scrollbar(wrap, command=self.calls_list.yview)
        self.calls_list.configure(yscrollcommand=sb.set)
        self.calls_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.calls_list.bind("<<ListboxSelect>>", self._on_call_select)

        # Details
        self.call_details = tk.Text(self.tab_calls, height=6, state="disabled", wrap="word")
        self.call_details.pack(fill=tk.X, padx=8, pady=(0,8))

    def _load_calls(self):
        self.calls_list.delete(0, tk.END)
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT id, ts, peer, direction, status, duration FROM calls ORDER BY ts DESC LIMIT 300")
        for cid, ts, peer, direction, status, duration in cur.fetchall():
            t = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            self.calls_list.insert(tk.END, f"{cid} | {t} | {direction} → {peer} | {status} | {int(duration or 0)}s")
        con.close()

    def _on_call_select(self, _evt=None):
        sel = self.calls_list.curselection()
        if not sel: return
        line = self.calls_list.get(sel[0])
        try:
            cid = int(line.split("|")[0].strip())
        except Exception:
            return
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT ts, peer, direction, status, duration, recording_path, rtt_ms, jitter_ms, loss FROM calls WHERE id=?", (cid,))
        r = cur.fetchone(); con.close()
        if not r: return
        ts, peer, direction, status, duration, rec, rtt, jitter, loss = r
        self.call_details.configure(state="normal"); self.call_details.delete("1.0","end")
        self.call_details.insert("end", f"Peer: {peer}\nTime: {datetime.fromtimestamp(ts)}\nDirection: {direction}\nStatus: {status}\nDuration: {int(duration or 0)}s\nRecording: {rec or '[none]'}\nRTT: {int(rtt or 0)} ms | Jitter: {int(jitter or 0)} ms | Loss: {round(loss or 0, 3)}\n")
        self.call_details.configure(state="disabled")

    def _export_calls_csv(self):
        path = os.path.join(self.EXPORTS_DIR, f"calls-{int(time.time())}.csv")
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT ts, peer, direction, status, duration, recording_path, rtt_ms, jitter_ms, loss FROM calls ORDER BY ts DESC")
        rows = cur.fetchall(); con.close()
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts","peer","direction","status","duration","recording_path","rtt_ms","jitter_ms","loss"])
            for r in rows: w.writerow(r)
        messagebox.showinfo("Export", f"Calls exported to {path}")

    def _export_calls_json(self):
        path = os.path.join(self.EXPORTS_DIR, f"calls-{int(time.time())}.json")
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT ts, peer, direction, status, duration, recording_path, rtt_ms, jitter_ms, loss FROM calls ORDER BY ts DESC")
        rows = cur.fetchall(); con.close()
        data = [{"ts": r[0], "peer": r[1], "direction": r[2], "status": r[3], "duration": r[4], "recording_path": r[5], "rtt_ms": r[6], "jitter_ms": r[7], "loss": r[8]} for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Export", f"Calls exported to {path}")

    # -----------------------------------------------------------------------------
    # Messages tab
    # -----------------------------------------------------------------------------
    def _build_messages_tab(self):
        # Toolbar
        tbar = ttk.Frame(self.tab_msgs); tbar.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(tbar, text="Peer ID:").pack(side=tk.LEFT)
        self.msg_peer = tk.StringVar()
        tk.Entry(tbar, textvariable=self.msg_peer, width=28).pack(side=tk.LEFT, padx=6)
        ttk.Button(tbar, text="Load", command=self._load_messages).pack(side=tk.LEFT, padx=6)
        ttk.Button(tbar, text="Export JSON", command=self._export_messages_json).pack(side=tk.LEFT, padx=6)

        # Viewer
        self.msg_view = tk.Text(self.tab_msgs, wrap="word", state="disabled")
        self.msg_view.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Send bar
        sendbar = ttk.Frame(self.tab_msgs); sendbar.pack(fill=tk.X, padx=8, pady=6)
        self.msg_input = tk.Entry(sendbar)
        self.msg_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sendbar, text="Send", command=self._send_message_text).pack(side=tk.LEFT, padx=6)
        ttk.Button(sendbar, text="😊", command=lambda: self._add_reaction("👍")).pack(side=tk.LEFT)
        ttk.Button(sendbar, text="📎", command=self._attach_file_to_message).pack(side=tk.LEFT)

    def _load_messages(self):
        peer = (self.msg_peer.get() or "").strip()
        self.msg_view.configure(state="normal"); self.msg_view.delete("1.0", "end")
        if not peer:
            self.msg_view.insert("end", "[Select a peer ID to load messages]\n")
            self.msg_view.configure(state="disabled"); return
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT ts, direction, text, delivered, reactions FROM messages WHERE peer=? ORDER BY ts ASC", (peer,))
        for ts, direction, text, delivered, reactions in cur.fetchall():
            t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            delivered_icon = "✔" if (delivered or 0) else "⏳"
            rx = ""
            try:
                arr = json.loads(reactions) if reactions else []
                rx = " " + "".join([item.get("emoji","") for item in arr])
            except Exception:
                pass
            self.msg_view.insert("end", f"{t} {direction}: {text} {delivered_icon}{rx}\n")
        self.msg_view.configure(state="disabled")
        con.close()

    def _export_messages_json(self):
        peer = (self.msg_peer.get() or "").strip()
        if not peer:
            messagebox.showwarning("Export", "Select a peer ID first."); return
        path = os.path.join(self.EXPORTS_DIR, f"messages-{peer}-{int(time.time())}.json")
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT ts, direction, text, delivered, reactions FROM messages WHERE peer=? ORDER BY ts ASC", (peer,))
        rows = cur.fetchall(); con.close()
        data = [{"ts": r[0], "direction": r[1], "text": r[2], "delivered": bool(r[3]), "reactions": (json.loads(r[4]) if r[4] else [])} for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Export", f"Messages exported to {path}")

    def _send_message_text(self):
        peer = (self.msg_peer.get() or "").strip()
        text = (self.msg_input.get() or "").strip()
        if not peer or not text:
            messagebox.showwarning("Message", "Peer and text are required."); return
        ts = time.time()
        con = self._db(); cur = con.cursor()
        cur.execute("INSERT INTO messages(ts, peer, direction, text, delivered, reactions) VALUES(?,?,?,?,?,?)", (ts, peer, "out", text, 1, json.dumps([])))
        con.commit(); con.close()
        try:
            if self._mesh_ok:
                self.node._u.send_text(peer, text)
        except Exception as e:
            logger.warning(f"[CommsPro] send_text error: {e}")
        self.msg_input.delete(0, tk.END)
        self._load_messages()

    def _add_reaction(self, emoji="👍"):
        peer = (self.msg_peer.get() or "").strip()
        if not peer:
            return
        # attach reaction to last outgoing message
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT id, reactions FROM messages WHERE peer=? AND direction='out' ORDER BY ts DESC LIMIT 1", (peer,))
        row = cur.fetchone()
        if not row:
            con.close(); return
        mid, reactions = row
        arr = []
        try:
            arr = json.loads(reactions) if reactions else []
        except Exception:
            arr = []
        arr.append({"emoji": emoji, "ts": time.time()})
        cur.execute("UPDATE messages SET reactions=? WHERE id=?", (json.dumps(arr), mid))
        con.commit(); con.close()
        self._load_messages()

    def _attach_file_to_message(self):
        peer = (self.msg_peer.get() or "").strip()
        if not peer:
            messagebox.showwarning("Attach", "Select a peer first."); return
        paths = filedialog.askopenfilenames(title="Select file(s)")
        if not paths: return
        msg = f"[Attached {len(paths)} file(s)]"
        ts = time.time()
        con = self._db(); cur = con.cursor()
        cur.execute("INSERT INTO messages(ts, peer, direction, text, delivered, reactions) VALUES(?,?,?,?,?,?)", (ts, peer, "out", msg, 1, json.dumps([])))
        con.commit(); con.close()
        try:
            if self._mesh_ok:
                for p in paths:
                    with open(p, "rb") as f:
                        blob = f.read()
                    self.node._u.send_file(peer, blob, os.path.basename(p))
        except Exception as e:
            logger.warning(f"[CommsPro] send_file error: {e}")
        self._load_messages()

    # -----------------------------------------------------------------------------
    # Contacts tab
    # -----------------------------------------------------------------------------
    def _build_contacts_tab(self):
        pane = ttk.PanedWindow(self.tab_contacts, orient=tk.HORIZONTAL); pane.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pane); right = ttk.Frame(pane)
        pane.add(left, weight=1); pane.add(right, weight=1)

        # Search bar
        sbar = ttk.Frame(left); sbar.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(sbar, text="Search").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        tk.Entry(sbar, textvariable=self.search_var, width=24).pack(side=tk.LEFT, padx=6)
        ttk.Button(sbar, text="Go", command=self._search_contacts).pack(side=tk.LEFT)
        ttk.Button(sbar, text="Reset", command=self._load_contacts).pack(side=tk.LEFT, padx=6)

        # List
        wrap = ttk.Frame(left); wrap.pack(fill=tk.BOTH, expand=True)
        self.contacts_list = tk.Listbox(wrap, height=16)
        sb = ttk.Scrollbar(wrap, command=self.contacts_list.yview)
        self.contacts_list.configure(yscrollcommand=sb.set)
        self.contacts_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.contacts_list.bind("<<ListboxSelect>>", self._on_contact_select)

        # Import/Export
        ibar = ttk.Frame(left); ibar.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(ibar, text="Import CSV", command=self._import_contacts_csv).pack(side=tk.LEFT)
        ttk.Button(ibar, text="Export CSV", command=self._export_contacts_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(ibar, text="Export JSON", command=self._export_contacts_json).pack(side=tk.LEFT, padx=4)

        # Details
        form = ttk.Frame(right); form.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.c_name  = tk.StringVar(); self.c_email = tk.StringVar()
        self.c_phone = tk.StringVar(); self.c_site  = tk.StringVar()
        self.c_addr  = tk.StringVar(); self.c_notes = tk.StringVar()
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_name).grid(row=0, column=1, sticky="ew")
        ttk.Label(form, text="Email").grid(row=1, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_email).grid(row=1, column=1, sticky="ew")
        ttk.Label(form, text="Phone").grid(row=2, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_phone).grid(row=2, column=1, sticky="ew")
        ttk.Label(form, text="Website").grid(row=3, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_site).grid(row=3, column=1, sticky="ew")
        ttk.Label(form, text="Address (user@host or id)").grid(row=4, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_addr).grid(row=4, column=1, sticky="ew")
        ttk.Label(form, text="Notes").grid(row=5, column=0, sticky="w"); tk.Entry(form, textvariable=self.c_notes).grid(row=5, column=1, sticky="ew")
        form.grid_columnconfigure(1, weight=1)

        # Avatar
        self.avatar_canvas = tk.Label(form, text="[Avatar]")
        self.avatar_canvas.grid(row=0, column=2, rowspan=6, padx=8, pady=6)

        # Actions
        btns = ttk.Frame(form); btns.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(btns, text="Save contact", command=self._save_contact).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete", command=self._delete_selected_contact).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Set avatar", command=self._set_avatar).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Call", command=self._call_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Message", command=self._message_selected).pack(side=tk.LEFT, padx=4)

    def _load_contacts(self):
        self.contacts_list.delete(0, tk.END)
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT id, name, phone, email, addr FROM contacts ORDER BY ts DESC")
        for cid, name, phone, email, addr in cur.fetchall():
            show = name or email or phone or addr or f"ID:{cid}"
            self.contacts_list.insert(tk.END, f"{cid} | {show}")
        con.close()

    def _search_contacts(self):
        q = (self.search_var.get() or "").strip().lower()
        self.contacts_list.delete(0, tk.END)
        con = self._db(); cur = con.cursor()
        like = f"%{q}%"
        cur.execute("""SELECT id, name, phone, email, addr FROM contacts
                       WHERE LOWER(name) LIKE ? OR LOWER(email) LIKE ? OR LOWER(phone) LIKE ? OR LOWER(addr) LIKE ?
                       ORDER BY ts DESC""", (like, like, like, like))
        for cid, name, phone, email, addr in cur.fetchall():
            show = name or email or phone or addr or f"ID:{cid}"
            self.contacts_list.insert(tk.END, f"{cid} | {show}")
        con.close()

    def _on_contact_select(self, _evt=None):
        sel = self.contacts_list.curselection()
        if not sel: return
        line = self.contacts_list.get(sel[0])
        try:
            cid = int(line.split("|")[0].strip())
        except Exception:
            return
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT name,email,phone,website,addr,image_path,notes FROM contacts WHERE id=?", (cid,))
        row = cur.fetchone(); con.close()
        if not row: return
        name,email,phone,site,addr,img_path,notes = row
        self.c_name.set(name or ""); self.c_email.set(email or ""); self.c_phone.set(phone or "")
        self.c_site.set(site or ""); self.c_addr.set(addr or ""); self.c_notes.set(notes or "")
        self._load_avatar(img_path)

    def _load_avatar(self, path):
        try:
            if not path or not os.path.isfile(path):
                self.avatar_canvas.configure(image="", text="[Avatar]"); return
            img = Image.open(path).convert("RGB")
            img.thumbnail((140, 140))
            photo = ImageTk.PhotoImage(img)
            self.avatar_canvas.configure(image=photo, text="")
            self.avatar_canvas.image = photo
        except Exception as e:
            logger.warning(f"[CommsPro] avatar load failed: {e}")
            self.avatar_canvas.configure(image="", text="[Avatar]")

    def _save_contact(self):
        name,email,phone,site,addr,notes = self.c_name.get(), self.c_email.get(), self.c_phone.get(), self.c_site.get(), self.c_addr.get(), self.c_notes.get()
        if not (name or email or phone or addr):
            messagebox.showwarning("Contact", "Provide at least a name, email, phone or address."); return
        # de-dup (same addr)
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT COUNT(1) FROM contacts WHERE addr=?", (addr,))
        if (cur.fetchone() or [0])[0] > 0 and addr:
            if not messagebox.askyesno("Duplicate", "A contact with this address exists. Save anyway?"):
                con.close(); return
        cur.execute("INSERT INTO contacts(name,email,phone,website,addr,image_path,notes,ts) VALUES(?,?,?,?,?,?,?,?)",
                    (name,email,phone,site,addr,None,notes,time.time()))
        con.commit(); con.close()
        self._load_contacts()

    def _delete_selected_contact(self):
        sel = self.contacts_list.curselection()
        if not sel: return
        line = self.contacts_list.get(sel[0])
        try: cid = int(line.split("|")[0].strip())
        except Exception: return
        if not messagebox.askyesno("Delete", "Delete selected contact?"): return
        con = self._db(); cur = con.cursor()
        cur.execute("DELETE FROM contacts WHERE id=?", (cid,))
        con.commit(); con.close()
        self._load_contacts()

    def _set_avatar(self):
        sel = self.contacts_list.curselection()
        if not sel: return
        line = self.contacts_list.get(sel[0])
        try: cid = int(line.split("|")[0].strip())
        except Exception: return
        path = filedialog.askopenfilename(title="Select avatar image",
                                          filetypes=[("Images","*.png;*.jpg;*.jpeg;*.gif")])
        if not path: return
        try:
            os.makedirs(self.IMAGES_DIR, exist_ok=True)
            dst = os.path.join(self.IMAGES_DIR, f"contact-{cid}-{int(time.time())}{os.path.splitext(path)[1].lower()}")
            shutil.copyfile(path, dst)
            con = self._db(); cur = con.cursor()
            cur.execute("UPDATE contacts SET image_path=? WHERE id=?", (dst, cid))
            con.commit(); con.close()
            self._load_avatar(dst)
        except Exception as e:
            logger.error(f"[CommsPro] set avatar error: {e}")
            messagebox.showerror("Avatar", f"Failed to set avatar: {e}")

    def _import_contacts_csv(self):
        path = filedialog.askopenfilename(title="Import contacts (CSV)")
        if not path: return
        import csv
        con = self._db(); cur = con.cursor()
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                cur.execute("INSERT INTO contacts(name,email,phone,website,addr,image_path,notes,ts) VALUES(?,?,?,?,?,?,?,?)",
                            (row.get("name"),row.get("email"),row.get("phone"),row.get("website"),row.get("addr"),row.get("image_path"),row.get("notes"),time.time()))
                count += 1
        con.commit(); con.close()
        self._load_contacts()
        messagebox.showinfo("Import", f"Imported {count} contacts.")

    def _export_contacts_csv(self):
        path = os.path.join(self.EXPORTS_DIR, f"contacts-{int(time.time())}.csv")
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT name,email,phone,website,addr,image_path,notes,ts FROM contacts ORDER BY ts DESC")
        rows = cur.fetchall(); con.close()
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name","email","phone","website","addr","image_path","notes","ts"])
            for r in rows: w.writerow(r)
        messagebox.showinfo("Export", f"Contacts exported to {path}")

    def _export_contacts_json(self):
        path = os.path.join(self.EXPORTS_DIR, f"contacts-{int(time.time())}.json")
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT name,email,phone,website,addr,image_path,notes,ts FROM contacts ORDER BY ts DESC")
        rows = cur.fetchall(); con.close()
        data = [{"name": r[0], "email": r[1], "phone": r[2], "website": r[3], "addr": r[4], "image_path": r[5], "notes": r[6], "ts": r[7]} for r in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Export", f"Contacts exported to {path}")

    def _call_selected(self):
        sel = self.contacts_list.curselection()
        if not sel: return
        line = self.contacts_list.get(sel[0])
        try: cid = int(line.split("|")[0].strip())
        except Exception: return
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT addr FROM contacts WHERE id=?", (cid,))
        r = cur.fetchone(); con.close()
        if not r or not r[0]:
            messagebox.showwarning("Call", "Contact has no peer address."); return
        self._start_call(r[0])

    def _message_selected(self):
        sel = self.contacts_list.curselection()
        if not sel: return
        line = self.contacts_list.get(sel[0])
        try: cid = int(line.split("|")[0].strip())
        except Exception: return
        con = self._db(); cur = con.cursor()
        cur.execute("SELECT addr FROM contacts WHERE id=?", (cid,))
        r = cur.fetchone(); con.close()
        if r and r[0]:
            self.msg_peer.set(r[0])
            self.nb.select(self.tab_msgs)
            self._load_messages()

    # -----------------------------------------------------------------------------
    # Keypad tab
    # -----------------------------------------------------------------------------
    def _build_keypad_tab(self):
        grid = ttk.Frame(self.tab_keypad); grid.pack(padx=10, pady=10)
        self.key_display = tk.Entry(self.tab_keypad, font=("Segoe UI", 14))
        self.key_display.pack(fill=tk.X, padx=10, pady=6)
        keys = [
            ("1",""),("2","ABC"),("3","DEF"),
            ("4","GHI"),("5","JKL"),("6","MNO"),
            ("7","PQRS"),("8","TUV"),("9","WXYZ"),
            ("*",""),("0","+"),("#","")
        ]
        r=c=0
        for digit,label in keys:
            btn = ttk.Button(grid, text=f"{digit}\n{label}", width=8, command=lambda d=digit: self._dial_press(d))
            btn.grid(row=r, column=c, padx=4, pady=4)
            c += 1
            if c>=3: c=0; r+=1
        bar = ttk.Frame(self.tab_keypad); bar.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(bar, text="Dial", command=self._dial_number).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Hangup", command=self._hangup).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Hold", command=self._toggle_hold).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Mute", command=self._toggle_mute).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Transfer", command=self._transfer_call).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="Volume").pack(side=tk.LEFT, padx=8)
        self.vol_var = tk.DoubleVar(value=1.0)
        tk.Scale(bar, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", variable=self.vol_var,
                 command=lambda v: self._set_volume(float(v))).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Speed dials
        sp = ttk.Frame(self.tab_keypad); sp.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(sp, text="Speed dials").pack(side=tk.LEFT)
        ttk.Button(sp, text="Support", command=lambda: self._start_call("support@mesh")).pack(side=tk.LEFT, padx=4)
        ttk.Button(sp, text="Ops", command=lambda: self._start_call("ops@mesh")).pack(side=tk.LEFT, padx=4)

    def _dial_press(self, symbol: str):
        self.key_display.insert(tk.END, symbol)
        if self.call_active and self.current_peer_id and self._mesh_ok:
            try:
                self.node._tel.send_dtmf(self.current_peer_id, symbol)
            except Exception as e:
                logger.warning(f"[CommsPro] DTMF send failed: {e}")

    def _dial_number(self):
        number = (self.key_display.get() or "").strip()
        if not number:
            messagebox.showwarning("Dial", "Enter a number or peer id."); return
        self._start_call(number)

    def _toggle_hold(self):
        self.call_on_hold = not self.call_on_hold
        messagebox.showinfo("Call", f"Hold: {'ON' if self.call_on_hold else 'OFF'}")

    def _toggle_mute(self):
        self.local_mute = not self.local_mute
        messagebox.showinfo("Call", f"Mute: {'ON' if self.local_mute else 'OFF'}")

    def _set_volume(self, v: float):
        self.output_volume = max(0.0, min(1.0, v))

    def _transfer_call(self):
        if not self.call_active or not self.current_peer_id:
            messagebox.showwarning("Transfer", "No active call."); return
        target = simpledialog.askstring("Transfer", "Transfer to peer ID:")
        if not target: return
        # Best-effort: send INFO message, then hangup and re‑dial target
        try:
            if self._mesh_ok:
                self.node._u.send_text(self.current_peer_id, "[Transferring call]")
        except Exception:
            pass
        self._hangup()
        self._start_call(target)

    # -----------------------------------------------------------------------------
    # Devices/QoS tab
    # -----------------------------------------------------------------------------
    def _build_devices_tab(self):
        pane = ttk.PanedWindow(self.tab_devices, orient=tk.VERTICAL); pane.pack(fill=tk.BOTH, expand=True)
        devs = ttk.Frame(pane); pane.add(devs, weight=1)
        qos  = ttk.Frame(pane); pane.add(qos,  weight=1)

        # Device selection
        ttk.Label(devs, text="Microphone").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(devs, text="Speaker").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Label(devs, text="Camera").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.sel_mic = tk.StringVar(); self.sel_spk = tk.StringVar(); self.sel_cam = tk.StringVar()
        self.mic_box = ttk.Combobox(devs, textvariable=self.sel_mic, values=self._list_mics()); self.mic_box.grid(row=0, column=1, sticky="ew", padx=8)
        self.spk_box = ttk.Combobox(devs, textvariable=self.sel_spk, values=self._list_speakers()); self.spk_box.grid(row=1, column=1, sticky="ew", padx=8)
        self.cam_box = ttk.Combobox(devs, textvariable=self.sel_cam, values=self._list_cameras()); self.cam_box.grid(row=2, column=1, sticky="ew", padx=8)
        devs.grid_columnconfigure(1, weight=1)
        ttk.Button(devs, text="Apply", command=self._apply_device_selection).grid(row=3, column=0, columnspan=2, pady=10)

        # QoS viewer
        ttk.Label(qos, text="QoS / IDS metrics").pack(anchor="w", padx=8, pady=(6,2))
        self.qos_text = tk.Text(qos, height=12, state="disabled", wrap="word")
        self.qos_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # IDS Audit viewer
        aud = ttk.Frame(qos); aud.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(aud, text="View IDS Audit", command=self._view_ids_audit).pack(side=tk.LEFT)

    def _list_mics(self):
        # placeholder; real enumeration requires OS-specific APIs
        self._device_lists["mic"] = ["Default mic", "USB mic", "BT mic"]
        return self._device_lists["mic"]

    def _list_speakers(self):
        self._device_lists["speaker"] = ["Default speakers", "USB audio", "BT headset"]
        return self._device_lists["speaker"]

    def _list_cameras(self):
        self._device_lists["camera"] = ["Default camera", "External cam"]
        return self._device_lists["camera"]

    def _apply_device_selection(self):
        m = self.sel_mic.get() or "Default mic"
        s = self.sel_spk.get() or "Default speakers"
        c = self.sel_cam.get() or "Default camera"
        messagebox.showinfo("Devices", f"Mic: {m}\nSpeaker: {s}\nCamera: {c}\n(Selection applied best effort)")

    def _view_ids_audit(self):
        try:
            # Show last ~50 audit lines from NetworkIDS
            self.qos_text.configure(state="normal"); self.qos_text.delete("1.0", "end")
            if self._mesh_ok and hasattr(self.node, "ids"):
                for line in list(self.node.ids.audit_log)[-50:]:
                    self.qos_text.insert("end", line + "\n")
            else:
                self.qos_text.insert("end", "[No IDS audit available]\n")
            self.qos_text.configure(state="disabled")
        except Exception:
            pass

    # -----------------------------------------------------------------------------
    # Call controls bar
    # -----------------------------------------------------------------------------
    def _build_call_controls_bar(self):
        bar = ttk.Frame(self.frame); bar.pack(fill=tk.X, padx=8, pady=(0,8))
        ttk.Button(bar, text="Start Call", command=self._start_call_prompt).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Hangup", command=self._hangup).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Record", command=self._toggle_record).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Snapshot", command=self._snapshot_remote).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Share Screen", command=self._share_screen_placeholder).pack(side=tk.LEFT, padx=4)

    def _start_call_prompt(self):
        peer = simpledialog.askstring("Call", "Peer ID:")
        if peer: self._start_call(peer)

    # -----------------------------------------------------------------------------
    # Call lifecycle
    # -----------------------------------------------------------------------------
    def _start_call(self, peer_id: str):
        if not self._mesh_ok:
            messagebox.showerror("Call", "Mesh not initialized."); return
        try:
            self.current_peer_id = peer_id
            self.call_active = True
            self.call_start_ts = time.time()
            self.call_on_hold = False
            self.local_mute = False
            self._audio_rec_chunks = []
            self._audio_rec_path = None

            # Telephony INVITE
            self.node._tel.dial_p2p(peer_id)

            # UI: activate remote feed on VideoPanel
            self._set_remote_feed_active(True)

            # Log call start (duration will be patched on hangup)
            self._log_call(peer_id, "out", "connected", 0, rec_path=None)
        except Exception as e:
            logger.error(f"[CommsPro] start_call error: {e}")
            messagebox.showerror("Call", f"Start call failed: {e}")

    def _hangup(self):
        if not self.call_active:
            return
        duration = max(0, time.time() - (self.call_start_ts or time.time()))
        # finalize recording path if any
        rec = self._audio_rec_path
        self._log_call(self.current_peer_id or "", "out", "completed", duration, rec_path=rec)
        self.call_active = False
        self.current_peer_id = None
        self.call_start_ts = None
        self._set_remote_feed_active(False)
        self._stop_recording()

    def _toggle_record(self):
        self.call_recording = not self.call_recording
        if self.call_recording:
            self._start_recording()
        else:
            self._stop_recording()
        messagebox.showinfo("Record", f"Recording: {'ON' if self.call_recording else 'OFF'}")

    def _start_recording(self):
        # best effort: create WAV; here we just append chunks symbolically
        try:
            ts = int(time.time())
            self._audio_rec_path = os.path.join(self.EXPORTS_DIR, f"call-audio-{ts}.wav")
            self._audio_rec_chunks = []
        except Exception:
            self._audio_rec_path = None

    def _stop_recording(self):
        try:
            if self._audio_rec_path and self._audio_rec_chunks:
                # write placeholder silence (real PCM capture would be needed)
                with open(self._audio_rec_path, "wb") as f:
                    f.write(b"")  # placeholder
        except Exception:
            pass

    def _snapshot_remote(self):
        try:
            if not self.call_active:
                messagebox.showwarning("Snapshot", "No active call."); return
            # best-effort remote snapshot capture (WebUI handles this via bridge/network_state)
            self._last_remote_snapshot = time.time()
            messagebox.showinfo("Snapshot", "Remote snapshot captured (placeholder).")
        except Exception:
            pass

    def _share_screen_placeholder(self):
        try:
            if not self.call_active:
                messagebox.showwarning("Share", "No active call."); return
            messagebox.showinfo("Share", "Screen sharing started (placeholder).")
        except Exception:
            pass

    # -----------------------------------------------------------------------------
    # Network/QoS polling and UI integration
    # -----------------------------------------------------------------------------
    def _poll_network_state(self):
        try:
            self._set_remote_feed_active(bool(self.call_active))
        except Exception:
            pass
        finally:
            if self.frame and self.frame.winfo_exists():
                self._poll_job = self.frame.after(1200, self._poll_network_state)

    def _poll_qos_metrics(self):
        try:
            if self._mesh_ok and hasattr(self.node, "_voip"):
                # Pull recent stats (placeholder; extend with actual RTCP-lite)
                self._qos_last = {"rtt_ms": self._qos_last.get("rtt_ms", 0),
                                  "jitter_ms": self._qos_last.get("jitter_ms", 0),
                                  "loss": self._qos_last.get("loss", 0.0)}
                self._refresh_qos_text()
        except Exception:
            pass
        finally:
            if self.frame and self.frame.winfo_exists():
                self._qos_job = self.frame.after(1500, self._poll_qos_metrics)

    def _refresh_qos_text(self):
        try:
            self.qos_text.configure(state="normal"); self.qos_text.delete("1.0", "end")
            self.qos_text.insert("end", f"RTT: {int(self._qos_last.get('rtt_ms', 0))} ms\n")
            self.qos_text.insert("end", f"Jitter: {int(self._qos_last.get('jitter_ms', 0))} ms\n")
            self.qos_text.insert("end", f"Loss: {round(self._qos_last.get('loss', 0.0), 3)}\n")
            self.qos_text.configure(state="disabled")
        except Exception:
            pass

    # -----------------------------------------------------------------------------
    # Helpers: UI remote feed toggle and logging
    # -----------------------------------------------------------------------------
    def _set_remote_feed_active(self, active: bool):
        try:
            if not self.video_panel:
                return
            if active:
                # Hide desktop mirror label text; keep remote tile primary
                try:
                    self.video_panel.screen_preview.configure(text="")
                except Exception:
                    pass
            else:
                try:
                    self.video_panel.screen_preview.configure(text="Desktop Mirror")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[CommsPro] remote feed toggle failed: {e}")

    def _log_call(self, peer: str, direction: str, status: str, duration_sec: float, rec_path: str | None):
        try:
            rtt = self._qos_last.get("rtt_ms", 0)
            jit = self._qos_last.get("jitter_ms", 0)
            loss = self._qos_last.get("loss", 0.0)
            con = self._db(); cur = con.cursor()
            cur.execute("""INSERT INTO calls(ts, peer, direction, status, duration, recording_path, rtt_ms, jitter_ms, loss)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (time.time(), peer, direction, status, float(duration_sec or 0), rec_path, rtt, jit, loss))
            con.commit(); con.close()
            self._load_calls()
            log_gui_event("Call", f"{direction} {peer} {status} {int(duration_sec)}s")
        except Exception as e:
            logger.warning(f"[CommsPro] log_call failed: {e}")


def _display_model_response(self, bundle):
    """Expected to be called after generate_reply()."""
    try:
        text = bundle.get("response","")
        html = bundle.get("html")
        if hasattr(self, "browser") and html:
            self.browser.open(html)
        # Always append text to chat area too
        if hasattr(self, "chat_text"):
            self.chat_text.insert("end", f"Sarah: {text}\n")
            self.chat_text.see("end")
    except Exception:
        pass

# ======================= SarahMemory GUI Agent Enhancements (append-only, safe) =======================
def _sm_gui_agent_enhance():
    """Append-only monkey-patch to enhance ChatPanel without renaming existing defs."""
    try:
        Chat = globals().get("ChatPanel")
        if Chat is None:
            return

        # -------- Helper methods to bind --------
        def _open_pro_browser(self, url=None):
            """Launch a Pro Browser (PyQt5 QtWebEngine) for full JS/video; fallback to system browser."""
            try:
                import subprocess, sys, tempfile, textwrap
                if not url:
                    url = "https://www.google.com"
                runner = tempfile.NamedTemporaryFile(delete=False, suffix="_sarah_probrowser.py", mode="w", encoding="utf-8")
                runner.write(textwrap.dedent(f"""
                    import sys
                    from PyQt5 import QtWidgets
                    from PyQt5.QtWebEngineWidgets import QWebEngineView
                    from PyQt5.QtCore import QUrl
                    app = QtWidgets.QApplication(sys.argv)
                    v = QWebEngineView()
                    v.setWindowTitle('SarahMemory — Pro Browser')
                    v.resize(1100, 800)
                    v.setUrl(QUrl("{(url or 'https://www.google.com').replace('"','%22')}"))
                    v.show()
                    sys.exit(app.exec_())
                """))
                runner.close()
                subprocess.Popen([sys.executable, runner.name], close_fds=True)
            except Exception:
                try:
                    import webbrowser
                    webbrowser.open(url or "https://www.google.com")
                except Exception:
                    pass

        def _open_show_me(self):
            """Open first saved reference link, or search last topic in the embedded browser."""
            try:
                if getattr(self, "_last_links", None):
                    self._addr_var.set(self._last_links[0])
                    self._reply_go()
                    return
                topic = getattr(self, "_last_topic", "").strip()
                if topic:
                    self._addr_var.set("https://www.google.com/search?q=" + topic.replace(" ","+"))
                    self._reply_go()
                    return
            except Exception:
                pass
            try:
                self._seed_google_lite()
            except Exception:
                pass

        def _smart_media_request(self, text):
            """
            Detect quick intents and navigate the mini-browser immediately.
            - 'show me an image of X' -> Google Images
            - 'play/show me video of X' -> YouTube search
            - 'deals on amazon' -> Amazon Gold Box
            Returns True if handled.
            """
            try:
                import re as _re
                t = (text or "").strip().lower()
                if not t:
                    return False
                m = _re.search(r"show me (an )?image of (.+)", t)
                if m:
                    q = m.group(2).strip()
                    self._addr_var.set("https://www.google.com/search?tbm=isch&q=" + q.replace(" ","+"))
                    self._reply_go(); return True
                m2 = _re.search(r"(show me|play) (a )?video (of|about) (.+)", t)
                if m2:
                    q = m2.group(4).strip()
                    self._addr_var.set("https://www.youtube.com/results?search_query=" + q.replace(" ","+"))
                    self._reply_go(); return True
                if ("deals" in t) and ("amazon" in t or "amazon.com" in t):
                    self._addr_var.set("https://www.amazon.com/gp/goldbox")
                    self._reply_go(); return True
                if t.startswith("[show me"):
                    self._open_show_me(); return True
            except Exception:
                pass
            return False

        def _start_idle_learning_loop(self):
            """Idle hook to call SarahMemoryDL.idle_deep_learn() periodically when user is idle."""
            try:
                import threading, time
                def _loop():
                    idle = 0
                    while True:
                        time.sleep(5)
                        try:
                            idle += 5
                            if idle >= 30:
                                try:
                                    from SarahMemoryDL import idle_deep_learn
                                    idle_deep_learn()
                                except Exception:
                                    pass
                                idle = 0
                        except Exception:
                            pass
                threading.Thread(target=_loop, daemon=True).start()
            except Exception:
                pass

        # -------- Bind helpers if not present --------
        if not hasattr(Chat, "_open_pro_browser"): Chat._open_pro_browser = _open_pro_browser
        if not hasattr(Chat, "_open_show_me"): Chat._open_show_me = _open_show_me
        if not hasattr(Chat, "_smart_media_request"): Chat._smart_media_request = _smart_media_request
        if not hasattr(Chat, "_start_idle_learning_loop"): Chat._start_idle_learning_loop = _start_idle_learning_loop

        # -------- Wrap __init__ to add state, toolbar button, and idle learning --------
        if not getattr(Chat, "_sm_init_wrapped", False):
            _orig_init = Chat.__init__
            def __init__(self, *a, **kw):
                _orig_init(self, *a, **kw)
                try:
                    if not hasattr(self, "_last_links"): self._last_links = []
                    if not hasattr(self, "_last_topic"): self._last_topic = ""
                except Exception:
                    pass
                # Add Pro Browser button if toolbar exists
                try:
                    if hasattr(self, "_reply_toolbar"):
                        import tkinter as tk
                        from tkinter import ttk
                        self._btn_pro = ttk.Button(self._reply_toolbar, text="Pro Browser",
                            command=lambda: self._open_pro_browser(self._addr_var.get() if hasattr(self,"_addr_var") else "https://www.google.com"))
                        self._btn_pro.pack(side="left", padx=4)
                except Exception:
                    pass
                # Start idle learning
                try:
                    self._start_idle_learning_loop()
                except Exception:
                    pass
            Chat.__init__ = __init__
            Chat._sm_init_wrapped = True

        # -------- Wrap send_message to intercept quick intents --------
        if hasattr(Chat, "send_message") and not getattr(Chat, "_sm_send_wrapped", False):
            _orig_send = Chat.send_message
            def send_message(self, event=None):
                try:
                    raw = self.chat_input.get("1.0", "end").strip()
                except Exception:
                    raw = ""
                try:
                    if self._smart_media_request(raw):
                        return "break"
                except Exception:
                    pass
                return _orig_send(self, event)
            Chat.send_message = send_message
            Chat._sm_send_wrapped = True

        # -------- Wrap generate_response to capture links/topic and auto-open first reference --------
        if hasattr(Chat, "generate_response") and not getattr(Chat, "_sm_gen_wrapped", False):
            _orig_gen = Chat.generate_response
            def generate_response(self, user_text):
                out = _orig_gen(self, user_text)
                # Best-effort: try to pull links from 'last_result_bundle' if your pipeline sets it.
                try:
                    bundle = getattr(self, "last_result_bundle", None)
                    if isinstance(bundle, dict):
                        links = (bundle.get("links") or [])[:5]
                        if links:
                            self._last_links = links
                    self._last_topic = (user_text or "").strip()
                    if getattr(self, "_last_links", None):
                        try:
                            self._addr_var.set(self._last_links[0])
                            self._reply_go()
                        except Exception:
                            pass
                except Exception:
                    pass
                return out
            Chat.generate_response = generate_response
            Chat._sm_gen_wrapped = True

    except Exception:
        pass

# Apply at import
try:
    _sm_gui_agent_enhance()
except Exception:
    pass


def _sm_hotfix_install():
    try:
        Chat = globals().get("ChatPanel")
        if Chat is None:
            return

        def _reply_load_url(self, url: str):
            try:
                if hasattr(self, "_addr_var") and url:
                    self._addr_var.set(url)
                if hasattr(self, "_reply_go"):
                    self._reply_go()
            except Exception:
                pass

        if not hasattr(Chat, "_reply_load_url"):
            Chat._reply_load_url = _reply_load_url

        if not getattr(Chat, "_smhf_init_wrapped", False):
            _orig_init = Chat.__init__
            def __init__(self, *a, **kw):
                _orig_init(self, *a, **kw)
                try:
                    if hasattr(self, "_btn_pro") and self._btn_pro.winfo_exists():
                        try:
                            self._btn_pro.destroy()
                        except Exception:
                            pass
                    try:
                        self._reply_load_url("https://duckduckgo.com/")
                    except Exception:
                        try:
                            if hasattr(self, "_reply_browser") and self._reply_browser:
                                self._reply_browser.set_content("<h3>Mini Browser Ready</h3><a href='https://duckduckgo.com/'>Open DuckDuckGo</a>")
                        except Exception:
                            pass
                    if not hasattr(self, "_last_links"):
                        self._last_links = []
                    if not hasattr(self, "_last_topic"):
                        self._last_topic = ""
                except Exception:
                    pass
            Chat.__init__ = __init__
            Chat._smhf_init_wrapped = True

        if hasattr(Chat, "send_message") and not getattr(Chat, "_smhf_send_wrapped", False):
            _orig_send = Chat.send_message
            import re as _re
            def send_message(self, event=None):
                try:
                    if event is not None:
                        event.widget.mark_set("insert", "end")
                        event.widget.tag_remove("sel", "1.0", "end")
                except Exception:
                    pass
                try:
                    text = self.chat_input.get("1.0", "end").strip()
                except Exception:
                    text = ""
                if not text:
                    return "break"
                try:
                    self.append_message("You: " + text)
                    self.chat_input.delete("1.0", "end")
                except Exception:
                    pass

                low = text.lower().strip()
                try:
                    if low in ("[show me]", "show me", "show me more", "[show me] more"):
                        if getattr(self, "_last_links", None):
                            try:
                                self._reply_load_url(self._last_links[0])
                                return "break"
                            except Exception:
                                pass
                        topic = getattr(self, "_last_topic", "").strip()
                        if topic:
                            try:
                                self._reply_load_url("https://en.wikipedia.org/wiki/Special:Search?search=" + topic.replace(" ", "+"))
                                return "break"
                            except Exception:
                                pass
                        try:
                            self._reply_load_url("https://duckduckgo.com/")
                            return "break"
                        except Exception:
                            pass
                    m = _re.search(r"show me (an )?image of (.+)", low)
                    if m and m.group(2).strip():
                        q = m.group(2).strip()
                        self._reply_load_url("https://en.wikipedia.org/wiki/Special:Search?search=" + q.replace(" ", "+") + "&iax=images&ia=images")
                        return "break"
                    m2 = _re.search(r"(show me|play) (a )?video (of|about) (.+)", low)
                    if m2 and m2.group(4).strip():
                        q = m2.group(4).strip()
                        self._reply_load_url("https://www.youtube.com/results?search_query=" + q.replace(" ", "+"))
                        return "break"
                    if ("deals" in low) and ("amazon" in low or "amazon.com" in low):
                        self._reply_load_url("https://www.amazon.com/gp/goldbox")
                        return "break"
                except Exception:
                    pass
                try:
                    return _orig_send(self, event)
                except Exception:
                    try:
                        self.append_message("Sarah: I'm thinking...")
                    except Exception:
                        pass
                    return "break"
            Chat.send_message = send_message
            Chat._smhf_send_wrapped = True

        if hasattr(Chat, "generate_response") and not getattr(Chat, "_smhf_gen_wrapped", False):
            _orig_gen = Chat.generate_response
            def generate_response(self, user_text):
                try:
                    bundle = _orig_gen(self, user_text)
                except Exception as e:
                    try:
                        self.append_message(f"[ERROR] {e}")
                    except Exception:
                        pass
                    return

                try:
                    if isinstance(bundle, dict):
                        lnks = bundle.get("links") or []
                        self._last_links = lnks[:]
                        self._last_topic = (user_text or "").strip() or (bundle.get("response","").split(".")[0])
                        html = bundle.get("html")
                        if html and hasattr(self, "_reply_browser") and self._reply_browser:
                            try:
                                self._reply_browser.set_content(html)
                            except Exception:
                                pass
                        elif lnks:
                            try:
                                self._reply_load_url(lnks[0])
                            except Exception:
                                pass
                        else:
                            if self._last_topic:
                                try:
                                    self._reply_load_url("https://en.wikipedia.org/wiki/Special:Search?search=" + self._last_topic.replace(" ", "+"))
                                except Exception:
                                    pass
                except Exception:
                    pass
                return bundle
            Chat.generate_response = generate_response
            Chat._smhf_gen_wrapped = True

        def _seed_duck_lite(self):
            html = """
            <!doctype html><html><head>
            <meta charset='utf-8'/>
            <meta name='viewport' content='width=device-width, initial-scale=1'/>
            <title>DuckDuckGo (Lite Preview)</title>
            <style>
              body{font-family:Arial,Helvetica,sans-serif;background:#fff;margin:0}
              .wrap{padding:14px;display:flex;justify-content:center}
              .card{max-width:680px;width:100%;border:1px solid #e6e6e6;border-radius:10px;padding:20px}
              .logo{display:flex;justify-content:center;margin:10px 0 18px}
              .bar{display:flex;gap:8px}
              .bar input{flex:1;padding:10px 12px;border:1px solid #d0d0d0;border-radius:18px}
              .bar button{padding:10px 16px;border:1px solid #d0d0d0;background:#f5f5f5;border-radius:18px;cursor:pointer}
              .hint{color:#666;font-size:12px;margin-top:10px;text-align:center}
              .scaled{transform:scale(.9);transform-origin:top center}
            </style></head><body>
              <div class='wrap scaled'><div class='card'>
                <div class='logo'><img alt='DuckDuckGo' src='https://duckduckgo.com/assets/logo_homepage.normal.v108.svg' height='48'/></div>
                <div class='bar'>
                  <input id='q' placeholder='Search DuckDuckGo (demo input)'/>
                  <button onclick="window.open('https://en.wikipedia.org/wiki/Special:Search?search='+encodeURIComponent(document.getElementById('q').value),'_blank')">Search</button>
                </div>
                <div class='hint'>Mini Browser is ready. Use the address bar above the viewer to load sites.</div>
              </div></div>
            </body></html>
            """.strip()
            try:
                if hasattr(self, "_reply_browser") and self._reply_browser:
                    self._reply_browser.set_content(html)
                elif hasattr(self, "_reply_text"):
                    self._reply_text.configure(state="normal"); self._reply_text.delete("1.0","end")
                    self._reply_text.insert("end", "Mini Browser (text fallback): DuckDuckGo Lite\nVisit: https://duckduckgo.com\n")
                    self._reply_text.configure(state="disabled")
            except Exception:
                pass

        if hasattr(Chat, "_seed_google_lite"):
            Chat._seed_google_lite = _seed_duck_lite
        else:
            Chat._seed_google_lite = _seed_duck_lite

    except Exception:
        pass

try:
    _sm_hotfix_install()
except Exception:
    pass

def _sm_hotfix_install_hf3():
    try:
        Chat = globals().get("ChatPanel")
        if Chat is None:
            return
        if hasattr(Chat, "send_message") and not getattr(Chat, "_smhf3_wrapped", False):
            import re as _re
            from SarahMemoryGlobals import run_async as _run_async
            _orig = Chat.send_message
            def send_message(self, event=None):
                # 1) capture text BEFORE any clearing
                try:
                    text = self.chat_input.get("1.0", "end").strip()
                except Exception:
                    text = ""
                if not text:
                    return "break"
                # 2) echo + clear
                try:
                    self.append_message("You: " + text)
                    self.chat_input.delete("1.0", "end")
                except Exception:
                    pass
                # 3) status hint
                try:
                    if hasattr(self, "gui") and hasattr(self.gui, "status_bar"):
                        self.gui.status_bar.set_intent_light("yellow")
                except Exception:
                    pass
                # 4) preserve last query for 'show me'
                try:
                    self._last_topic = text
                except Exception:
                    pass
                # 5) quick intents
                low = text.lower()
                try:
                    if low in ("[show me]", "show me", "show me more", "[show me] more"):
                        if getattr(self, "_last_links", None):
                            try:
                                self._reply_load_url(self._last_links[0])
                                return "break"
                            except Exception:
                                pass
                        topic = getattr(self, "_last_topic", "").strip()
                        if topic:
                            try:
                                self._reply_load_url("https://en.wikipedia.org/wiki/Special:Search?search=" + topic.replace(" ", "+"))
                                return "break"
                            except Exception:
                                pass
                        try:
                            self._reply_load_url("https://duckduckgo.com/")
                            return "break"
                        except Exception:
                            pass
                    m = _re.search(r"show me (an )?image of (.+)", low)
                    if m and m.group(2).strip():
                        q = m.group(2).strip()
                        try:
                            self._reply_load_url("https://en.wikipedia.org/wiki/Special:Search?search=" + q.replace(" ","+") + "&iax=images&ia=images")
                            return "break"
                        except Exception:
                            pass
                    m2 = _re.search(r"(show me|play) (a )?video (of|about) (.+)", low)
                    if m2 and m2.group(4).strip():
                        q = m2.group(4).strip()
                        try:
                            self._reply_load_url("https://www.youtube.com/results?search_query=" + q.replace(" ","+"))
                            return "break"
                        except Exception:
                            pass
                    if ("deals" in low) and ("amazon" in low or "amazon.com" in low):
                        try:
                            self._reply_load_url("https://www.amazon.com/gp/goldbox")
                            return "break"
                        except Exception:
                            pass
                except Exception:
                    pass
                # 6) normal flow: run async generate_response directly with captured text
                try:
                    _run_async(self.generate_response, text)
                except Exception:
                    # As a fallback, try original (in case future versions depend on it)
                    try:
                        return _orig(self, event)
                    except Exception:
                        try:
                            self.append_message("Sarah: I'm processing that...")
                        except Exception:
                            pass
                return "break"
            Chat.send_message = send_message
            Chat._smhf3_wrapped = True
    except Exception:
        pass

try:
    _sm_hotfix_install_hf3()
except Exception:
    pass


# --- injected: on-demand ensure table for `response` ---
def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
try:
    _ensure_response_table()
except Exception:
    pass

# v7.7.4: camera lifecycle guard
import atexit as _sm_atexit
def _sm_release_cameras():
    try:
        # Add explicit release if you keep a global/singleton VideoPanel in Main
        # e.g., if hasattr(config, "video_panel"): config.video_panel.release_all()
        pass
    except Exception:
        pass
_sm_atexit.register(_sm_release_cameras)

# v7.7.4: defensive CSS key whitelist (used by apply_theme_from_choice)
try:
    _SM_CSS_KEYS = {'background','background-color','color','foreground','font','borderwidth','relief'}
except Exception:
    _SM_CSS_KEYS = set()


# === [APPEND] WebView API: visual query endpoint ===
try:
    import webview
except Exception:
    webview = None

# === pywebview bridge method ===
def visual_query(self, text: str):
    """
    Called from app.js (__SM_visualOCRQuery). Uses the latest shared_frame (BGR).
    Returns a dict with 'response' and 'meta' (keeps GUI contract).
    """
    import SarahMemorySOBJE as sobje
    try:
        with shared_lock:
            frame = shared_frame.copy() if shared_frame is not None else None
        if frame is None:
            return {"response":"I don't have a camera frame yet.", "meta":{"source":"local","intent":"ocr"}}

        out = sobje.answer_visual_question(text, frame)
        reply = out.get("answer") if isinstance(out, dict) else str(out)
        return {"response": reply or "(no OCR result)", "meta":{"source":"local","intent":"ocr"}}
    except Exception as e:
        return {"response": f"OCR error: {e}", "meta":{"source":"local","intent":"ocr"}}