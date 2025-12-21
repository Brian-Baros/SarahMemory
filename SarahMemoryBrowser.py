
"""--==The SarahMemory Project==--
File: SarahMemoryBrowser.py
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

# Description:
# A) Legacy lightweight embedded browser (HtmlFrame when available; Text fallback).
# B) New WebUI wrapper that prefers pywebview for modern HTML/JS/CSS and JS↔Python bridge.
#    Falls back to HtmlFrame if pywebview is unavailable. No def renames of existing parts.
from __future__ import annotations
import os, sys, threading, time, base64, io, logging, webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, List

import SarahMemoryGlobals as config
try:
    # Some codepaths expect WEB_HOMEPAGE from globals
    from SarahMemoryGlobals import WEB_HOMEPAGE  # type: ignore
except Exception:  # fallback if not set
    WEB_HOMEPAGE = "api.sarahmemory.com"

# ---------- Optional HTML fallback viewer ----------
try:
    from tkinterweb import HtmlFrame  # type: ignore
    _HAS_HTMLFRAME = True
except Exception:
    HtmlFrame = None  # type: ignore
    _HAS_HTMLFRAME = False

# ---------- Preferred modern webview ----------
try:
    import webview  # pywebview
    _HAS_WEBVIEW = True
except Exception:
    webview = None  # type: ignore
    _HAS_WEBVIEW = False

logger = logging.getLogger("SarahMemoryBrowser")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class SarahMemoryBrowser:
    """
    Legacy mini-browser widget with toolbar.
    Keeps backward compatibility for existing GUI code.
    """
    def __init__(self, parent, home_url: Optional[str] = None):
        self.parent = parent
        eff = (home_url or WEB_HOMEPAGE or "").strip()
        if eff and not (eff.startswith("http://") or eff.startswith("https://")):
            eff = "https://" + eff
        self.home_url = eff or "https://api.sarahmemory.com"

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        # Toolbar
        tbar = ttk.Frame(self.frame)
        tbar.pack(fill="x", padx=5, pady=(5, 2))

        self.addr_var = tk.StringVar(value=self.home_url)
        self.addr_entry = ttk.Entry(tbar, textvariable=self.addr_var)
        self.addr_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.addr_entry.bind("<Return>", lambda e: self.go())

        ttk.Button(tbar, text="Go", command=self.go).pack(side="left", padx=2)
        ttk.Button(tbar, text="◀", command=self.back).pack(side="left", padx=2)
        ttk.Button(tbar, text="▶", command=self.forward).pack(side="left", padx=2)
        ttk.Button(tbar, text="⟳", command=self.reload).pack(side="left", padx=2)
        ttk.Button(tbar, text="Open", command=self.open_external).pack(side="left", padx=2)

        # Viewer
        self.viewer = None
        self.text = None
        if _HAS_HTMLFRAME:
            try:
                self.viewer = HtmlFrame(self.frame, messages_enabled=False, vertical_scrollbar=True)
                self.viewer.pack(fill="both", expand=True)
                try:
                    if self.home_url:
                        self.viewer.load_website(self.home_url)
                except Exception:
                    pass
            except Exception:
                self.viewer = None

        if self.viewer is None:
            self.text = tk.Text(self.frame, wrap="word", state="normal", relief=tk.FLAT)
            self.text.pack(fill="both", expand=True)
            self.text.insert("end", f"Browser fallback active. Open externally: {self.home_url}\n")
            self.text.configure(state="disabled")

    @property
    def widget(self):
        return self.frame

    def go(self):
        url = self.addr_var.get().strip()
        if not url:
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url
            self.addr_var.set(url)
        self.load_url(url)

    def load_url(self, url: str):
        if self.viewer:
            try:
                self.viewer.load_website(url)
            except Exception:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
        else:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("end", f"Open in browser: {url}\n")
            self.text.configure(state="disabled")

    def set_html(self, html: str):
        if self.viewer:
            try:
                self.viewer.set_content(html)
            except Exception:
                pass
        else:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("end", html or "[No HTML]")
            self.text.configure(state="disabled")

    def back(self):
        if self.viewer:
            try:
                self.viewer.html.backward()
            except Exception:
                pass

    def forward(self):
        if self.viewer:
            try:
                self.viewer.html.forward()
            except Exception:
                pass

    def reload(self):
        if self.viewer:
            try:
                self.viewer.on_reload()
            except Exception:
                pass

    def open_external(self):
        url = self.addr_var.get().strip()
        if url:
            try:
                webbrowser.open(url)
            except Exception:
                pass


def show_browser_page(query: str) -> None:
    print(f"[SarahMemoryBrowser] Showing page for: {query}")
    try:
        q = (query or "").strip().replace(" ", "+")
        url = f"https://www.bing.com/search?q={q}"
        webbrowser.open(url)
    except Exception as e:
        print("[SarahMemoryBrowser] Failed to open browser query:", e)


# Drag-and-drop ingest window (legacy helper)
def launch_drop_ingest_window():
    """
    Opens a small window where users can drag & drop files.
    Falls back to file dialog if tkdnd is unavailable.
    Copies files into DATASETS_DIR and triggers embedding.
    """
    files_collected: List[str] = []

    def _ingest(paths):
        try:
            import shutil
            from SarahMemoryGlobals import DATASETS_DIR, extract_text  # type: ignore
            from SarahMemoryDatabase import embed_and_store_dataset_sentences  # type: ignore

            os.makedirs(DATASETS_DIR, exist_ok=True)
            added = 0
            for p in paths or []:
                p = (p or "").strip().strip("{}")
                if not p or not os.path.exists(p):
                    continue
                dest = os.path.join(DATASETS_DIR, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dest):
                    try:
                        shutil.copy2(p, dest)
                    except Exception:
                        pass
                try:
                    _ = extract_text(dest)  # probe readability
                except Exception:
                    pass
                added += 1
            try:
                embed_and_store_dataset_sentences()
            except Exception:
                pass
            messagebox.showinfo("Ingest Complete", f"Processed {added} file(s).")
        except Exception as e:
            try:
                messagebox.showerror("Ingest Error", str(e))
            except Exception:
                print("[Browser] Ingest error:", e)

    try:
        root = tk.Tk()
        root.title("SarahMemory — Drop to Ingest")
        root.geometry("420x180")
        lab = tk.Label(root, text="Drop files here to ingest\n(or Click to choose)",
                       relief="groove", width=40, height=6)
        lab.pack(padx=12, pady=12, fill="both", expand=True)
        lab.bind("<Button-1>", lambda e: _ingest(filedialog.askopenfilenames(title="Select files")))
        try:
            root.drop_target_register('DND_Files')
            root.dnd_bind('<<Drop>>', lambda e: _ingest(e.data.split()))
        except Exception:
            btn = tk.Button(root, text="Select Files…",
                            command=lambda: _ingest(filedialog.askopenfilenames(title="Select files")))
            btn.pack(pady=6)
        root.mainloop()
    except Exception as e:
        try:
            messagebox.showerror("UI Error", f"Failed to open drag-and-drop window: {e}")
        except Exception:
            print("[Browser] UI open failed:", e)


# -------------------- New WebUI wrapper (pywebview preferred) --------------------

@dataclass
class _Handlers:
    on_event: Optional[Callable[[str, Dict], None]] = None

class WebUI:
    """
    High-level web UI surface for the center pane.
    Prefers pywebview; falls back to HtmlFrame if not available.
    Note: pywebview opens its own window (Tk backend). For now it may be a child window.
    """
    def __init__(self, master_tk, width: int = 920, height: int = 720):
        self.master = master_tk
        self.handlers = _Handlers()
        self.mode = "fallback"
        self.window = None
        self.frame = None
        self._api_obj = None
        self._init_ui(width, height)

    # Public API
    def set_handler(self, on_event: Callable[[str, Dict], None]) -> None:
        self.handlers.on_event = on_event

    def load_html(self, path: Optional[str] = None, html: Optional[str] = None) -> None:
        if self.mode == "webview" and self.window is not None:
            if html is not None:
                try:
                    self.window.load_html(html)
                except Exception:
                    pass
            elif path:
                try:
                    url = "file:///" + path.replace("\\", "/")
                    self.window.load_url(url)
                except Exception:
                    pass
        elif self.mode == "htmlframe" and self.frame is not None:
            try:
                if html is not None:
                    self.frame.set_html(html)
                elif path:
                    with open(path, "r", encoding="utf-8") as f:
                        self.frame.set_html(f.read())
            except Exception:
                pass

    def eval_js(self, script: str) -> None:
        if self.mode == "webview" and self.window is not None:
            try:
                self.window.evaluate_js(script)
            except Exception:
                pass
        elif self.mode == "htmlframe" and self.frame is not None and hasattr(self.frame, "evaluate_js"):
            try:
                self.frame.evaluate_js(script)
            except Exception:
                pass

    # Internal
    def _init_ui(self, width: int, height: int) -> None:
        if config.USE_WEBVIEW and _HAS_WEBVIEW:
            # Launch a webview window (Tk backend)
            self.mode = "webview"
            self.window = webview.create_window(
                title="SarahMemory WebUI",
                url="about:blank",
                width=width,
                height=height,
                resizable=True,
                frameless=False,
                easy_drag=False,
                on_top=False,
                confirm_close=False,
            )

            # JS→Python bridge
            class _Api:
                def __init__(self, outer: "WebUI"):
                    self._outer = outer
                def post(self, action: str, payload: Optional[Dict] = None):
                    # Restrict origins (local file and whitelisted domains)
                    try:
                        if not config.origin_allowed("file://"):
                            return False
                    except Exception:
                        pass
                    handler = outer.handlers.on_event if (outer := self._outer) else None
                    if handler:
                        try:
                            handler(action, payload or {})
                        except Exception:
                            pass
                    return True

            self._api_obj = _Api(self)
            try:
                # Expose API and start loop in a background thread if not already running
                self.window.expose(self._api_obj)  # type: ignore[attr-defined]
            except Exception:
                pass

            def _start():
                try:
                    webview.start(gui="tk")
                except Exception:
                    pass
            if not webview.windows:
                threading.Thread(target=_start, daemon=True).start()

        elif _HAS_HTMLFRAME:
            # Fallback: embed HtmlFrame inside the provided Tk container
            self.mode = "htmlframe"
            self.frame = HtmlFrame(self.master, messages_enabled=False)
            self.frame.pack(fill="both", expand=True)
        else:
            # No HTML backend available; remain in fallback mode (no-op surface)
            self.mode = "fallback"
            ph = ttk.Frame(self.master)
            ph.pack(fill="both", expand=True)
            lbl = ttk.Label(ph, text="No Web UI backend available (pywebview/tkinterweb missing).")
            lbl.pack(padx=12, pady=12)
import os, base64, threading, time as _time
from typing import Any, Dict, List, Optional

# Pull globals (safe fallback if import fails)
try:
    import SarahMemoryGlobals as config
except Exception:
    class config:  # minimal fallback
        BASE_DIR = os.getcwd()
        WEBUI_HTML_PATH = os.path.join(BASE_DIR, "data", "ui", "SarahMemory.html")
        GUI_MODE = "classic"
        DEBUG_MODE = False
        REPLY_STATUS = False
        API_RESPONSE_CHECK_TRAINER = False

# ---------- optional webcam helpers (non-fatal if OpenCV missing) ----------
_CAM_LOCK = threading.Lock()
_CAM = None

def _b64_jpeg_from_frame(frame) -> Optional[str]:
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None

def _try_capture_once() -> Optional[str]:
    try:
        import cv2
    except Exception:
        return None
    global _CAM
    with _CAM_LOCK:
        if _CAM is None:
            try:
                _CAM = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-friendly
                _CAM.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                _CAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            except Exception:
                _CAM = None
        cap = _CAM
    try:
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return _b64_jpeg_from_frame(frame)
    except Exception:
        return None

# ------------------------------ JS Bridge ------------------------------
class WebUIBridge:
    """
    Methods exposed to the WebView front-end (window.pywebview.api in app.js).
    Keep names STABLE.
    """
    def __init__(self, gui=None):
        # gui is the classic Tk GUI object if you created it; optional.
        self.gui = gui

    # ----- Boot / flags -----
    def get_boot_state(self) -> Dict[str, Any]:
        today = _time.strftime("%Y-%m-%d", _time.localtime())
        return {
            "REPLY_STATUS": bool(getattr(config, "REPLY_STATUS", True)),
            "API_RESPONSE_CHECK_TRAINER": bool(getattr(config, "API_RESPONSE_CHECK_TRAINER", False)),
            "today": today
        }

    def set_flag(self, name: str, value: Any) -> bool:
        """
        Toggle a top-level boolean in SarahMemoryGlobals at runtime.
        Example: set_flag('REPLY_STATUS', True)
        """
        try:
            if not hasattr(config, name):
                return False
            v = value
            if isinstance(v, str):
                v = v.strip().lower() in ("1","true","yes","on")
            setattr(config, name, bool(v))
            return True
        except Exception:
            return False

    # ----- Threads / history (best-effort; returns empty if helper not present) -----
    def list_threads_for_date(self, day_iso: str) -> List[Dict[str, Any]]:
        try:
            from SarahMemoryDatabase import list_threads_for_date  # optional helper
            rows = list_threads_for_date(day_iso) or []
            out = []
            for r in rows:
                if isinstance(r, dict):
                    out.append({"title": r.get("title","(untitled)"), "timestamp": r.get("timestamp","")})
                elif isinstance(r, (list,tuple)) and len(r) >= 2:
                    out.append({"title": str(r[0]), "timestamp": str(r[1])})
            return out
        except Exception:
            return []

    # ----- Reminders -----
    def list_reminders(self) -> List[Dict[str, Any]]:
        try:
            from SarahMemoryReminder import list_reminders as _lr
            items = _lr() or []
            out = []
            for it in items:
                if isinstance(it, dict):
                    out.append({"title": it.get("title","(no title)"), "when": it.get("when",""), "note": it.get("note","")})
            return out
        except Exception:
            return []

    def create_reminder(self, title: str, when: str, note: str = "") -> bool:
        try:
            from SarahMemoryReminder import create_reminder as _cr
            _cr(title, when, note)
            return True
        except Exception:
            return False

    # ----- Webcam snapshot -----
    def get_snapshot(self) -> Dict[str, Any]:
        return {"data_url": _try_capture_once()}

    # ----- Messaging (re-use existing pipeline; no doubles) -----
    def send_message(self, text: str, blobs=None) -> Dict[str, Any]:
        """
        Routes message through ChatPanel.generate_response when available
        (so all your existing logic is kept), and returns a clean object:
        { "response": "...", "meta": { "source": "...", "intent": "..." } }
        """
        # Prefer the established GUI method if present
        try:
            if self.gui and getattr(self.gui, "chat_panel", None):
                fn = getattr(self.gui.chat_panel, "generate_response", None)
                if callable(fn):
                    result = fn(text)  # recent versions return dict
                    if isinstance(result, dict):
                        meta = result.get("meta") or {}
                        src = meta.get("source", result.get("source", "unknown"))
                        intent = meta.get("intent", result.get("intent", "undetermined"))
                        resp = (result.get("response") or result.get("data") or "").strip()
                        return {"response": resp, "meta": {"source": src, "intent": intent}}
        except Exception:
            pass

        # Fallback: call Reply directly
        try:
            from SarahMemoryReply import generate_reply
            result = generate_reply(self.gui or self, text)
            if isinstance(result, dict):
                meta = result.get("meta") or {}
                src = meta.get("source", result.get("source", "unknown"))
                intent = meta.get("intent", result.get("intent", "undetermined"))
                resp = (result.get("response") or result.get("data") or "").strip()
                return {"response": resp, "meta": {"source": src, "intent": intent}}
            return {"response": str(result), "meta": {"source": "unknown", "intent": "undetermined"}}
        except Exception as e:
            return {"response": f"[ERROR] {e}", "meta": {"source": "error", "intent": "error"}}

# ------------------------------ launcher ------------------------------
def launch_webui(gui=None, html_path: Optional[str] = None, title: str = "SarahMemory"):
    try:
        import webview
    except Exception as e:
        raise RuntimeError("pywebview is not installed. Run: pip install pywebview") from e

    base_dir = getattr(config, "BASE_DIR", os.getcwd())
    hpath = html_path or getattr(config, "WEBUI_HTML_PATH", None) or os.path.join(base_dir, "data", "ui", "SarahMemory.html")
    if not os.path.isabs(hpath):
        hpath = os.path.join(base_dir, hpath)
    if not os.path.exists(hpath):
        raise FileNotFoundError(f"WebUI HTML not found: {hpath}")

    bridge = WebUIBridge(gui=gui)
    webview.create_window(title, hpath, js_api=bridge, width=1200, height=800, resizable=True)
    webview.start(debug=bool(getattr(config, "DEBUG_MODE", False)))

# --------------------------- pywebview launcher ---------------------------
def _resolve_ui_url() -> str:
    """Return a file:// URL to the local UI if it exists, else fall back to remote."""
    try:
        ui_dir = Path(getattr(config, "UI_DIR", r"C:\SarahMemory\data\ui"))
        index = getattr(config, "UI_INDEX_FILE", "SarahMemory.html")
        local_index = ui_dir / index
        if local_index.exists():
            return local_index.resolve().as_uri()
    except Exception:
        pass
    # Remote fallback (hosted copy)
    return "https://www.sarahmemory.com/api/data/ui/SarahMemory.html"

# A tiny in‑memory reminder store (kept process‑local)
_REMINDERS: List[Dict[str, Any]] = []

def _png_1x1_data_url() -> str:
    # Transparent 1x1 PNG
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAJ2A1BsAAAAASUVORK5CYII="
    return f"data:image/png;base64,{b64}"

class WebUIApi:
    """Functions exposed to the Web UI (window.pywebview.api)."""

    # --- Chat ---
    def send_text(self, text: str) -> Dict[str, Any]:
        """Receive a text prompt from the UI and return a simple response.
        NOTE: This is a safe stub. It can be wired into SarahMemory reply pipeline later.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "empty"}
        # TODO: integrate with SarahMemoryReply.generate_response or GUI bridge
        reply = f"SarahMemory received: {text}"
        return {"ok": True, "reply": reply, "meta": {"source": "local", "ts": time.time()}}

    # --- Status / Snapshot ---
    def get_snapshot(self) -> Dict[str, Any]:
        # Placeholder still image; can be replaced with a webcam capture in future
        return {"ok": True, "data_url": _png_1x1_data_url(), "ts": time.time()}

    def get_stats(self) -> Dict[str, Any]:
        try:
            import psutil  # optional
            cpu = psutil.cpu_percent(interval=0.05)
            mem = psutil.virtual_memory().percent
            return {"ok": True, "cpu": cpu, "mem": mem}
        except Exception:
            return {"ok": True, "cpu": None, "mem": None}

    # --- Threads / History (sample placeholders) ---
    def list_threads(self, date_iso: str | None = None) -> Dict[str, Any]:
        return {"ok": True, "items": [{"id": "demo", "title": "Welcome thread", "date": date_iso or "today"}]}

    # --- Reminders ---
    def get_reminders(self) -> Dict[str, Any]:
        return {"ok": True, "items": list(_REMINDERS)}

    def add_reminder(self, title: str, when: str, note: str | None = None) -> Dict[str, Any]:
        item = {"title": title, "when": when, "note": note or "", "ts": time.time()}
        _REMINDERS.append(item)
        return {"ok": True, "item": item}

    # --- Toggles ---
    def toggle_reply(self, enabled: bool) -> Dict[str, Any]:
        return {"ok": True, "reply_enabled": bool(enabled)}

    def toggle_compare(self, enabled: bool) -> Dict[str, Any]:
        return {"ok": True, "compare_enabled": bool(enabled)}

def launch_web_ui_detached() -> bool:
    """Create a pywebview window pointing to the local/remote UI and return True if opened.
    Runs in a separate thread so it won't block the main Tk loop.
    """
    try:
        import webview  # import inside for environments where pywebview isn't installed
    except Exception as e:
        logger.warning("pywebview not available: %s", e)
        return False

    url = _resolve_ui_url()
    backend = getattr(config, "WEBVIEW_BACKEND", None)
    api = WebUIApi()

    def _run():
        try:
            if backend:
                webview.config.gui = backend  # hint backend (e.g., 'edgechromium', 'qt', 'cef')
            window = webview.create_window("SarahMemory — Web UI", url, width=1280, height=820, resizable=True, js_api=api)
            # Start the loop; debug=False to avoid console noise in prod
            webview.start(debug=False)
        except Exception as e:
            logger.error("Web UI failed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="SarahWebUI")
    t.start()
    logger.info("Web UI launched at %s", url)
    return True


# ---- Telecom bridge into SarahMemoryGUI UnifiedCommsProPanel ----
def _get_comms():
    try:
        from SarahMemoryGUI import get_comms_bridge, init_unified_comms
        c = get_comms_bridge()
        if c is None:
            try: init_unified_comms(None, None)
            except Exception: c = get_comms_bridge()
        return get_comms_bridge()
    except Exception as e:
        return None

class WebUIApi(WebUIApi):  # extend
    def telecom_get_contacts(self, payload: dict | None = None):
        c = _get_comms()
        return c.telecom_get_contacts(payload) if c else []

    def telecom_add_contact(self, payload: dict):
        c = _get_comms()
        return c.telecom_add_contact(payload) if c else {"ok": False}

    def telecom_delete_contact(self, payload: dict):
        c = _get_comms()
        return c.telecom_delete_contact(payload) if c else {"ok": False}

    def telecom_list_recents(self, payload: dict | None = None):
        c = _get_comms()
        return c.telecom_list_recents(payload) if c else []

    def telecom_send_message(self, payload: dict):
        c = _get_comms()
        return c.telecom_send_message(payload) if c else {"ok": False}

    def telecom_start_call(self, payload: dict):
        c = _get_comms()
        return c.telecom_start_call(payload) if c else {"ok": False}

    def telecom_end_call(self, payload: dict | None = None):
        c = _get_comms()
        return c.telecom_end_call(payload) if c else {"ok": False}

    def telecom_get_remote_frame(self, payload: dict | None = None):
        c = _get_comms()
        return c.telecom_get_remote_frame(payload) if c else None

# ====================================================================
# END OF SarahMemoryBrowser.py v8.0.0
# ====================================================================