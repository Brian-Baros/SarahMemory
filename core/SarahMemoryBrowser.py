"""--==The SarahMemory Project==--
File: SarahMemoryBrowser.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "ui_app"
# CATEGORY = "browser_and_webui"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "classic_ui"
# API_DOMAIN = "webui_bridge"
# HARDWARE_DOMAIN = "camera_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "browser"
# FAMILY = "research"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = True
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "User-facing browser and WebUI wrapper with legacy embedded browser, pywebview bridge, drag-drop ingest helper, and JS/Python interface."
# --- SARAHMETA END ---

# Description:
# A) Legacy lightweight embedded browser (HtmlFrame when available; Text fallback).
# B) New WebUI wrapper that prefers pywebview for modern HTML/JS/CSS and JS↔Python bridge.
#    Falls back to HtmlFrame if pywebview is unavailable. No def renames of existing parts.
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
    # SARAHMEMORY_PATCH_NOTE 2026-06-24:
    # Legacy browser wrapper must resolve through the governed UI path contract.
    # Prefer config-provided V9/legacy entries under BASE_DIR/data/ui instead of
    # stale BASE_DIR/ui/web fallback paths.
    hpath = html_path or getattr(config, "CUSTOM_UI_INDEX", None) or getattr(config, "WEBUI_HTML_PATH", None) or os.path.join(base_dir, "data", "ui", "v9", "index.html")
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
        # SARAHMEMORY_PATCH_NOTE 2026-06-24:
        # UI_DIR is owned by SarahMemoryGlobals and points to BASE_DIR/data/ui/v9.
        # The stale root-level UI fallback was removed because it
        # can open an empty/stale page on portable installs.
        ui_dir = Path(getattr(config, "UI_DIR", Path(getattr(config, "BASE_DIR", ".")) / "data" / "ui" / "v9"))
        index = getattr(config, "UI_INDEX_FILE", "SarahMemory.html")
        local_index = ui_dir / index
        if local_index.exists():
            return local_index.resolve().as_uri()
    except Exception:
        pass
    # Remote fallback (hosted copy)
    return "https://www.sarahmemory.com/api/data/ui/SarahMemory.html"

class WebUIApi(WebUIBridge):
    """Bounded pywebview API backed by the governed SarahMemory bridges."""

    _worker_slots = threading.BoundedSemaphore(4)

    def __init__(self, gui=None):
        super().__init__(gui=gui)

    def _bounded(self, fn: Callable[[], Any], *, timeout: float, operation: str) -> Dict[str, Any]:
        if not self._worker_slots.acquire(blocking=False):
            return {"ok": False, "error": f"{operation}_capacity_exhausted", "value": None}
        completed = threading.Event()
        result: Dict[str, Any] = {"ok": False, "error": None, "value": None}

        def _runner() -> None:
            try:
                result["value"] = fn()
                result["ok"] = True
            except BaseException as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                completed.set()
                self._worker_slots.release()

        worker = threading.Thread(target=_runner, name=f"SM-WebUI-{operation}", daemon=True)
        worker.start()
        if not completed.wait(max(0.05, float(timeout))):
            return {"ok": False, "error": f"{operation}_timeout", "value": None}
        return dict(result)

    def send_text(self, text: str) -> Dict[str, Any]:
        prompt = str(text or "").strip()
        if not prompt:
            return {"ok": False, "error": "empty_prompt"}
        timeout = float(getattr(config, "WEBUI_BRIDGE_TIMEOUT_SECONDS", 20.0) or 20.0)
        call = self._bounded(lambda: self.send_message(prompt), timeout=timeout, operation="send_text")
        if not call.get("ok"):
            return {"ok": False, "error": call.get("error"), "reply": ""}
        payload = call.get("value") if isinstance(call.get("value"), dict) else {}
        reply = str(payload.get("response") or payload.get("reply") or "").strip()
        return {
            "ok": bool(reply),
            "reply": reply,
            "meta": dict(payload.get("meta") or {"source": "webui_bridge", "intent": "chat"}),
            "error": None if reply else "empty_reply",
        }

    def get_snapshot(self) -> Dict[str, Any]:
        call = self._bounded(lambda: WebUIBridge.get_snapshot(self), timeout=2.0, operation="snapshot")
        if not call.get("ok"):
            return {"ok": False, "data_url": None, "error": call.get("error"), "ts": time.time()}
        payload = call.get("value") if isinstance(call.get("value"), dict) else {}
        data_url = payload.get("data_url")
        return {"ok": bool(data_url), "data_url": data_url, "error": None if data_url else "frame_unavailable", "ts": time.time()}

    def get_stats(self) -> Dict[str, Any]:
        try:
            import psutil  # type: ignore
            return {
                "ok": True,
                "cpu": float(psutil.cpu_percent(interval=0.05)),
                "mem": float(psutil.virtual_memory().percent),
            }
        except Exception as exc:
            return {"ok": False, "cpu": None, "mem": None, "error": str(exc)}

    def list_threads(self, date_iso: str | None = None) -> Dict[str, Any]:
        day = str(date_iso or time.strftime("%Y-%m-%d", time.localtime()))
        items = self.list_threads_for_date(day)
        return {"ok": True, "items": items, "date": day}

    def get_reminders(self) -> Dict[str, Any]:
        return {"ok": True, "items": self.list_reminders()}

    def add_reminder(self, title: str, when: str, note: str | None = None) -> Dict[str, Any]:
        title_s = str(title or "").strip()
        when_s = str(when or "").strip()
        if not title_s or not when_s:
            return {"ok": False, "error": "title_and_when_required"}
        ok = self.create_reminder(title_s, when_s, str(note or ""))
        return {"ok": bool(ok), "item": {"title": title_s, "when": when_s, "note": str(note or "")}, "error": None if ok else "reminder_backend_unavailable"}

    def toggle_reply(self, enabled: bool) -> Dict[str, Any]:
        ok = self.set_flag("REPLY_STATUS", bool(enabled))
        return {"ok": bool(ok), "reply_enabled": bool(enabled) if ok else None}

    def toggle_compare(self, enabled: bool) -> Dict[str, Any]:
        ok = self.set_flag("API_RESPONSE_CHECK_TRAINER", bool(enabled))
        return {"ok": bool(ok), "compare_enabled": bool(enabled) if ok else None}

    def _telecom_call(self, method_name: str, payload: Any, default: Any) -> Any:
        comms = _get_comms()
        method = getattr(comms, method_name, None) if comms is not None else None
        if not callable(method):
            return default
        call = self._bounded(lambda: method(payload), timeout=3.0, operation=method_name)
        return call.get("value") if call.get("ok") else default

    def telecom_get_contacts(self, payload: dict | None = None):
        return self._telecom_call("telecom_get_contacts", payload, [])

    def telecom_add_contact(self, payload: dict):
        return self._telecom_call("telecom_add_contact", payload, {"ok": False, "error": "telecom_unavailable"})

    def telecom_delete_contact(self, payload: dict):
        return self._telecom_call("telecom_delete_contact", payload, {"ok": False, "error": "telecom_unavailable"})

    def telecom_list_recents(self, payload: dict | None = None):
        return self._telecom_call("telecom_list_recents", payload, [])

    def telecom_send_message(self, payload: dict):
        return self._telecom_call("telecom_send_message", payload, {"ok": False, "error": "telecom_unavailable"})

    def telecom_start_call(self, payload: dict):
        return self._telecom_call("telecom_start_call", payload, {"ok": False, "error": "telecom_unavailable"})

    def telecom_end_call(self, payload: dict | None = None):
        return self._telecom_call("telecom_end_call", payload, {"ok": False, "error": "telecom_unavailable"})

    def telecom_get_remote_frame(self, payload: dict | None = None):
        return self._telecom_call("telecom_get_remote_frame", payload, None)

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


# ====================================================================
# END OF SarahMemoryBrowser.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryBrowser',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Input',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['browser', 'input'],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω004'],
    "required_authority": ['Read'],
    "priority": 60,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryBrowser.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryBrowser',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryBrowser', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

