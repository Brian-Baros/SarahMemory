"""--==The SarahMemory Project==--
File: SarahMemoryBrowser.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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
# VALIDATION_DATE = "2026-11-01"
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
import copy, hashlib, re, shutil, tempfile, uuid, json
from urllib.parse import urlsplit, urlencode
from contextlib import suppress
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

# Local-first v9.1.0a trust/resource policy. A browser document is NOT an
# authentication receipt, and remote HTML must never inherit local Python APIs.
MODULE_VERSION = "9.1.0a"
MAX_BRIDGE_TEXT = 8192
MAX_EVENT_BYTES = 16384
MAX_MEDIA_BYTES = 2 * 1024 * 1024
MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_DATASET_BATCH = 16
_ALLOWED_FLAG_NAMES = frozenset({"REPLY_STATUS", "API_RESPONSE_CHECK_TRAINER"})
_ALLOWED_INGEST_EXTS = frozenset({".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".html", ".htm"})
_UI_EVENT_ACTIONS = frozenset({"ui_ready", "navigation", "tab_change", "view_changed"})
_BROWSER_RUNTIME = {"webview_started": False, "webview_failed": False, "last_error": "", "last_action": "", "last_action_ok": None}
_BROWSER_LOCK = threading.RLock()

def _note_operation(action: str, ok: bool, error: str = "") -> None:
    with _BROWSER_LOCK:
        _BROWSER_RUNTIME.update(last_action=str(action)[:64], last_action_ok=bool(ok), last_error=str(error)[:256])

def _normalize_http_url(url: str, *, allow_search: bool = False) -> str:
    """Allow ordinary public browsing, never OS/JS/file URI schemes."""
    value = str(url or "").strip()
    if not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
        raise ValueError("invalid_url")
    if allow_search and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) and not value.startswith("//"):
        value = "https://" + value
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("unsafe_url_scheme")
    if re.search(r"\s", parsed.netloc) or parsed.username is not None or parsed.password is not None:
        raise ValueError("url_credentials_not_allowed")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_url_port") from exc
    return value

def _ui_root() -> Path:
    base = Path(str(getattr(config, "BASE_DIR", Path(__file__).resolve().parent))).resolve()
    return (base / "data" / "ui").resolve()

def _trusted_local_ui(path_or_url: str) -> bool:
    """An exact local HTML file under the installed UI tree, not a scheme-prefix."""
    try:
        from urllib.parse import unquote, urlparse
        value = str(path_or_url or "")
        if value.startswith("file:"):
            parts = urlparse(value)
            if parts.scheme != "file" or parts.netloc not in {"", "localhost"} or parts.query or parts.fragment:
                return False
            path = Path(unquote(parts.path))
            # Windows file:///C:/... conversion, even under Unix test runners.
            if os.name == "nt" and str(path).startswith("/") and re.match(r"^/[A-Za-z]:", str(path)):
                path = Path(str(path)[1:])
        else:
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
                return False
            path = Path(value)
        root = _ui_root()
        target = path.resolve(strict=True)
        return target.is_file() and target.suffix.lower() in {".html", ".htm"} and target.is_relative_to(root)
    except Exception:
        return False

def _current_window_url(window: Any) -> str:
    """Never infer caller provenance from the requested URL alone."""
    try:
        getter = getattr(window, "get_current_url", None)
        return str(getter() if callable(getter) else "")
    except Exception:
        return ""

def _verified_window(window: Any, pinned_url: str) -> bool:
    try:
        current = _current_window_url(window)
        if not (current and pinned_url and _trusted_local_ui(pinned_url) and _trusted_local_ui(current)):
            return False
        return Path(_file_path_from_url(current)).resolve(strict=True) == Path(_file_path_from_url(pinned_url)).resolve(strict=True)
    except Exception:
        return False

def _file_path_from_url(value: str) -> str:
    from urllib.parse import urlparse, unquote
    parts = urlparse(str(value))
    path = unquote(parts.path) if parts.scheme == "file" else str(value)
    return path[1:] if os.name == "nt" and re.match(r"^/[A-Za-z]:", path) else path

def _strict_bool(value: Any) -> bool:
    if type(value) is bool: return value
    if type(value) is int and value in (0, 1): return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("true", "1", "yes", "on"): return True
        if val in ("false", "0", "no", "off"): return False
    raise ValueError("invalid_boolean")

def _authorize_action(goal: str, action: str, target: str, *, user_consented: bool = False) -> dict:
    """Call existing governance owner; missing/ambiguous decisions deny."""
    if not user_consented:
        return {"ok": False, "decision": "REQUIRE_USER", "allow": False}
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
        out = authorize_external_domain_action(
            goal, origin="SarahMemoryBrowser",
            proposed_action={"action_type": action, "target": target, "target_ref": target,
                             "capability_name": "browser", "risk_level": "medium",
                             "required_permissions": ["user_consent", "local_device_access"]},
            meta={"source": "local_browser_ui", "visual_only": False},
            user_consented=user_consented)
        return out if isinstance(out, dict) and out.get("ok") is True and out.get("allow") is True and out.get("decision") == "ALLOW" and out.get("domain_execution_authorized") is True else {"ok": False, "decision": "DENY", "allow": False}
    except Exception:
        return {"ok": False, "decision": "DENY", "allow": False}

def _report_action(auth: dict, *, action: str, ok: bool, reason: str = "") -> None:
    try:
        from SarahMemoryOperatorCore import record_external_domain_action_result
        record_external_domain_action_result(auth, {"ok": bool(ok), "operation": action, "reason": reason[:160]}, verified=bool(ok), origin="SarahMemoryBrowser")
    except Exception:
        logger.warning("Governed %s result could not be recorded", action)

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
        if not url: return
        try:
            normalized = _normalize_http_url(url, allow_search=True)
            self.load_url(normalized)
            self.addr_var.set(normalized)
        except ValueError as exc:
            _note_operation("browser_navigation", False, str(exc))

    def load_url(self, url: str):
        try:
            normalized = _normalize_http_url(url, allow_search=True)
        except ValueError as exc:
            _note_operation("browser_navigation", False, str(exc))
            return
        if self.viewer:
            try:
                self.viewer.load_website(normalized)
            except Exception:
                # Never turn a failed embedded load into unrequested external navigation.
                _note_operation("browser_navigation", False, "embedded_load_failed")
        elif self.text is not None:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("end", f"Open in browser: {normalized}\n")
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
        try:
            url = _normalize_http_url(self.addr_var.get(), allow_search=True)
            webbrowser.open(url)
            _note_operation("browser_external", True)
        except Exception as exc:
            _note_operation("browser_external", False, type(exc).__name__)

def show_browser_page(query: str) -> None:
    """Open user-requested search with encoded query parameters."""
    try:
        term = str(query or "").strip()
        if not term or len(term) > MAX_BRIDGE_TEXT: raise ValueError("invalid_query")
        url = "https://www.bing.com/search?" + urlencode({"q": term})
        webbrowser.open(url)
        _note_operation("browser_search", True)
    except Exception as exc:
        _note_operation("browser_search", False, type(exc).__name__)

# Drag-and-drop ingest window (legacy helper)
def _ingest_dataset_files(paths: Any, *, embed: bool = True, user_consented: bool = False) -> dict:
    """Copy a small explicitly selected batch without overwriting prior evidence.

    The existing Database organ still owns embedding. No recursive directory walk.
    """
    items = list(paths or [])[:MAX_DATASET_BATCH + 1]
    if not items:
        return {"ok": False, "added": 0, "rejected": 0, "embedding": "not_requested", "error": "empty_selection"}
    if len(items) > MAX_DATASET_BATCH:
        return {"ok": False, "added": 0, "error": "batch_limit_exceeded"}
    auth = _authorize_action("Ingest explicitly selected local dataset files", "dataset_ingest", "local_datasets",
                             user_consented=user_consented)
    if not auth.get("allow"):
        return {"ok": False, "added": 0, "error": "governance_denied", "embedding": "not_requested"}
    target = Path(str(getattr(config, "DATASETS_DIR", "")))
    if not str(target) or str(target) == ".":
        return {"ok": False, "added": 0, "error": "dataset_dir_unconfigured"}
    added = 0
    rejects = []
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {"ok": False, "added": 0, "error": "dataset_dir_unavailable"}
    for item in items:
        dest = None
        created = False
        try:
            raw = str(item or "").strip()
            src = Path(raw).expanduser()
            if not src.is_file() or src.is_symlink() or src.suffix.lower() not in _ALLOWED_INGEST_EXTS:
                raise ValueError("invalid_file")
            length = src.stat().st_size
            if not 0 < length <= MAX_DATASET_BYTES:
                raise ValueError("file_size_limit")
            # Unique, exclusive destination prevents same-name overwrite and symlink collision.
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", src.name)[:96]
            if safe_name in {"", ".", ".."}: raise ValueError("invalid_name")
            dest = target / f"{uuid.uuid4().hex[:12]}_{safe_name}"
            with src.open("rb") as reader, dest.open("xb") as writer:
                created = True
                header = reader.read(4096)
                if src.suffix.lower() == ".pdf" and not header.startswith(b"%PDF-"):
                    raise ValueError("invalid_pdf_header")
                if src.suffix.lower() == ".docx" and not header.startswith(b"PK\x03\x04"):
                    raise ValueError("invalid_docx_header")
                if src.suffix.lower() in {".txt", ".md", ".csv", ".json", ".html", ".htm"} and b"\x00" in header:
                    raise ValueError("invalid_text_content")
                writer.write(header)
                total = len(header)
                while True:
                    part = reader.read(64 * 1024)
                    if not part: break
                    total += len(part)
                    if total > MAX_DATASET_BYTES: raise ValueError("file_size_changed")
                    writer.write(part)
            added += 1
        except (OSError, ValueError) as exc:
            rejects.append(type(exc).__name__ if not isinstance(exc, ValueError) else str(exc))
            with suppress(Exception):
                if created and dest is not None and dest.parent == target:
                    dest.unlink(missing_ok=True)
    embed_result = "not_requested"
    if added and embed:
        try:
            from SarahMemoryDatabase import embed_and_store_dataset_sentences
            # Database supports bounded local refresh, not an unbounded global scan.
            result = embed_and_store_dataset_sentences(include_imported_data=True, include_sqlite_pool=False,
                                                       max_items=150, time_budget_sec=15.0)
            embed_result = "completed" if isinstance(result, dict) and result.get("ok") is True else "unverified"
        except Exception:
            embed_result = "unavailable"
    out = {"ok": bool(added) and not rejects and (not embed or embed_result == "completed"),
           "added": added, "rejected": len(rejects), "errors": rejects[:8], "embedding": embed_result}
    _note_operation("dataset_ingest", out["ok"], embed_result)
    _report_action(auth, action="dataset_ingest", ok=out["ok"], reason=embed_result)
    return out

def launch_drop_ingest_window():
    """Open a local, user-operated file chooser. No silent file-system scans."""
    try:
        root = tk.Tk()
        root.title("SarahMemory — Drop to Ingest")
        root.geometry("420x180")
        def _pick():
            selected = filedialog.askopenfilenames(title="Select Files", parent=root)
            if not selected: return
            out = _ingest_dataset_files(selected, user_consented=True)
            if out["ok"]:
                messagebox.showinfo("Ingest Complete", f"Copied {out['added']} file(s).", parent=root)
            else:
                messagebox.showwarning("Ingest Incomplete", f"Copied {out['added']} file(s). Embedding: {out['embedding']}. Rejected: {out['rejected']}.", parent=root)
        btn = ttk.Button(root, text="Select or Drop Files to Ingest", command=_pick)
        btn.pack(padx=12, pady=24, fill="both", expand=True)
        if hasattr(root, "drop_target_register") and hasattr(root, "dnd_bind"):
            with suppress(Exception):
                root.drop_target_register("DND_Files")
                def _drop(event):
                    # Tk's native Tcl list parser handles paths containing spaces,
                    # unlike the legacy split() implementation.
                    selected = root.tk.splitlist(str(event.data or ""))
                    if selected:
                        out = _ingest_dataset_files(selected, user_consented=True)
                        if out.get("ok"):
                            messagebox.showinfo("Ingest Complete", f"Copied {out['added']} file(s).", parent=root)
                        else:
                            messagebox.showwarning("Ingest Incomplete", str(out.get("error") or out.get("embedding") or "rejected"), parent=root)
                root.dnd_bind("<<Drop>>", _drop)
        root.mainloop()
    except Exception as exc:
        _note_operation("dataset_ui", False, type(exc).__name__)

# -------------------- New WebUI wrapper (pywebview preferred) --------------------

@dataclass
class _Handlers:
    on_event: Optional[Callable[[str, Dict], None]] = None

class WebUI:
    """Optional pywebview window with presentation-only local event bridge."""
    def __init__(self, master_tk, width: int = 920, height: int = 720):
        self.master = master_tk
        self.handlers = _Handlers()
        self.mode = "fallback"
        self.window = None
        self.frame = None
        self._api_obj = None
        self._pinned_url = ""
        self._bridge_armed = False
        self._init_ui(width, height)

    def set_handler(self, on_event: Callable[[str, Dict], None]) -> None:
        self.handlers.on_event = on_event

    def load_html(self, path: Optional[str] = None, html: Optional[str] = None) -> None:
        # Dynamically supplied HTML never acquires bridge authority.
        self._bridge_armed = False
        if self.mode == "webview" and self.window is not None:
            if html is not None:
                self.window.load_html(str(html)[:MAX_EVENT_BYTES])
            elif path and _trusted_local_ui(path):
                self._pinned_url = Path(_file_path_from_url(path)).resolve().as_uri()
                self.window.load_url(self._pinned_url)
                self._bridge_armed = True
        elif self.mode == "htmlframe" and self.frame is not None:
            try:
                if html is not None:
                    self.frame.set_html(str(html)[:MAX_EVENT_BYTES])
                elif path and _trusted_local_ui(path):
                    self.frame.set_html(Path(_file_path_from_url(path)).read_text(encoding="utf8", errors="replace")[:MAX_EVENT_BYTES])
            except Exception:
                _note_operation("load_html", False, "htmlframe_load_failed")

    def eval_js(self, script: str) -> None:
        # Compatibility: only the local Python UI can evaluate script on the same
        # verified pinned document, never on remote/unverified content.
        if (self.mode == "webview" and self.window is not None and
                bool(getattr(self, "_bridge_armed", False)) and _verified_window(self.window, getattr(self, "_pinned_url", "")) and
                isinstance(script, str) and len(script.encode("utf8")) <= MAX_EVENT_BYTES):
            try:
                self.window.evaluate_js(script)
                return
            except Exception:
                pass
        _note_operation("eval_js", False, "unverified_document_or_script")

    def _init_ui(self, width: int, height: int) -> None:
        if bool(getattr(config, "USE_WEBVIEW", False)) and _HAS_WEBVIEW:
            self.mode = "webview"
            self.window = webview.create_window(title="SarahMemory WebUI", url="about:blank",
                                                width=max(320, min(3840, int(width))),
                                                height=max(240, min(2160, int(height))),
                                                resizable=True, frameless=False, easy_drag=False,
                                                on_top=False, confirm_close=False)
            class _Api:
                def __init__(self, outer: "WebUI"):
                    self._outer = outer
                def post(self, action: str, payload: Optional[Dict] = None):
                    outer = self._outer
                    if not (outer._bridge_armed and _verified_window(outer.window, outer._pinned_url)):
                        return False
                    if not isinstance(action, str) or action not in _UI_EVENT_ACTIONS or not isinstance(payload, (dict, type(None))):
                        return False
                    try:
                        data = json.dumps(payload or {}, allow_nan=False)
                        if len(data.encode("utf8")) > MAX_EVENT_BYTES: return False
                        handler = outer.handlers.on_event
                        if not callable(handler): return False
                        handler(action, payload or {})
                        return True
                    except Exception:
                        return False
            self._api_obj = _Api(self)
            # webview.start() requires GUI ownership; no implicit background thread
            # is started from inside an existing Tk mainloop.
            try:
                self.window.expose(self._api_obj)
            except Exception:
                _note_operation("webview_expose", False, "bridge_exposure_unavailable")
        elif _HAS_HTMLFRAME:
            self.mode = "htmlframe"
            self.frame = HtmlFrame(self.master, messages_enabled=False)
            self.frame.pack(fill="both", expand=True)
        else:
            self.mode = "fallback"
            ph = ttk.Frame(self.master)
            ph.pack(fill="both", expand=True)
            ttk.Label(ph, text="No Web UI backend available (pywebview/tkinterweb missing).").pack(padx=12, pady=12)

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
    """Legacy entry point: fail closed without an authenticated permission receipt."""
    return None

def _capture_local_snapshot_with_consent(*, user_consented: bool = False) -> Optional[str]:
    """For a separately verified local GUI action, not exposed to JavaScript."""
    auth = _authorize_action("Capture one user-approved webcam frame", "camera_snapshot", "camera:0", user_consented=user_consented)
    if not auth.get("allow"):
        _note_operation("camera_snapshot", False, "authorization_denied")
        return None
    value = None
    cap = None
    try:
        import cv2
        with _CAM_LOCK:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not cap or not cap.isOpened():
                raise RuntimeError("camera_unavailable")
            with suppress(Exception): cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            with suppress(Exception): cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            ok, frame = cap.read()
            if ok and frame is not None:
                value = _b64_jpeg_from_frame(frame)
                if value and len(value) > MAX_MEDIA_BYTES * 2: value = None
    except Exception as exc:
        _note_operation("camera_snapshot", False, type(exc).__name__)
    finally:
        if cap is not None:
            with suppress(Exception): cap.release()
        _report_action(auth, action="camera_snapshot", ok=bool(value))
    return value

# ------------------------------ JS Bridge ------------------------------
class WebUIBridge:
    """Legacy JS bridge. Existing public methods retain their signatures.

    Enterprise WebUIApi instances additionally enforce the pinned-page policy.
    """
    def __init__(self, gui=None):
        self.gui = gui

    def _source_allowed(self) -> bool:
        verifier = getattr(self, "_bridge_allowed", None)
        return bool(verifier()) if callable(verifier) else True

    def get_boot_state(self) -> Dict[str, Any]:
        if not self._source_allowed():
            return {"REPLY_STATUS": False, "API_RESPONSE_CHECK_TRAINER": False, "today": "", "error": "untrusted_document"}
        return {"REPLY_STATUS": bool(getattr(config, "REPLY_STATUS", False)),
                "API_RESPONSE_CHECK_TRAINER": bool(getattr(config, "API_RESPONSE_CHECK_TRAINER", False)),
                "today": _time.strftime("%Y-%m-%d", _time.localtime())}

    def set_flag(self, name: str, value: Any) -> bool:
        if not self._source_allowed() or str(name) not in _ALLOWED_FLAG_NAMES:
            return False
        try:
            parsed = _strict_bool(value)
            if not hasattr(config, name): return False
            # A browser script may not disable the Compare quality gate.
            # Authorized changes to that governance setting belong to its owner.
            if name == "API_RESPONSE_CHECK_TRAINER" and not parsed:
                _note_operation("ui_flag", False, "compare_disable_requires_approval")
                return False
            setattr(config, name, parsed)
            _note_operation("ui_flag", True)
            return True
        except (TypeError, ValueError, AttributeError):
            _note_operation("ui_flag", False, "invalid_flag_value")
            return False

    def list_threads_for_date(self, day_iso: str) -> List[Dict[str, Any]]:
        if not self._source_allowed(): return []
        try:
            from datetime import date
            date.fromisoformat(str(day_iso))
            from SarahMemoryDatabase import list_threads_for_date
            rows = list_threads_for_date(day_iso) or []
            out = []
            for r in list(rows)[:100]:
                if isinstance(r, dict):
                    out.append({"title": str(r.get("title") or "(untitled)")[:256], "timestamp": str(r.get("timestamp") or "")[:64]})
                elif isinstance(r, (tuple, list)) and len(r) >= 2:
                    out.append({"title": str(r[0])[:256], "timestamp": str(r[1])[:64]})
            return out
        except Exception:
            return []

    def list_reminders(self) -> List[Dict[str, Any]]:
        if not self._source_allowed(): return []
        try:
            from SarahMemoryReminder import list_reminders
            out = []
            for item in list(list_reminders() or [])[:100]:
                if isinstance(item, dict):
                    out.append({"title": str(item.get("title") or "(no title)")[:256],
                                "when": str(item.get("when") or "")[:128],
                                "note": str(item.get("note") or "")[:1024]})
            return out
        except Exception:
            return []

    def create_reminder(self, title: str, when: str, note: str = "") -> bool:
        if not self._source_allowed(): return False
        # A JS call is not evidence of human consent. No approval receipt is
        # currently forwarded by the frontend; refuse to invent one.
        _note_operation("reminder_create", False, "explicit_approval_path_required")
        return False

    def get_snapshot(self) -> Dict[str, Any]:
        # Never use the webcam merely because a JS caller invoked this method.
        _note_operation("camera_snapshot", False, "local_device_approval_required")
        return {"data_url": None}

    def send_message(self, text: str, blobs=None) -> Dict[str, Any]:
        """Delegate strictly to the existing reply owner; no duplicate cognition."""
        if not self._source_allowed():
            return {"response": "", "meta": {"source": "denied", "intent": "untrusted_document"}}
        if blobs:
            return {"response": "", "meta": {"source": "denied", "intent": "attachments_require_governed_ingest"}}
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf8")) > MAX_BRIDGE_TEXT:
            return {"response": "", "meta": {"source": "denied", "intent": "invalid_prompt"}}
        try:
            if self.gui and getattr(self.gui, "chat_panel", None):
                fn = getattr(self.gui.chat_panel, "generate_response", None)
                if callable(fn):
                    result = fn(text)
                    if isinstance(result, dict):
                        meta = result.get("meta") or {}
                        return {"response": str(result.get("response") or result.get("data") or "")[:65536],
                                "meta": {"source": str(meta.get("source", result.get("source", "unknown")))[:100],
                                         "intent": str(meta.get("intent", result.get("intent", "undetermined")))[:100]}}
            from SarahMemoryReply import generate_reply
            result = generate_reply(self.gui or self, text)
            if isinstance(result, dict):
                meta = result.get("meta") or {}
                return {"response": str(result.get("response") or result.get("data") or "")[:65536],
                        "meta": {"source": str(meta.get("source", result.get("source", "unknown")))[:100],
                                 "intent": str(meta.get("intent", result.get("intent", "undetermined")))[:100]}}
            return {"response": str(result)[:65536], "meta": {"source": "unknown", "intent": "undetermined"}}
        except Exception as exc:
            _note_operation("chat", False, type(exc).__name__)
            return {"response": "", "meta": {"source": "error", "intent": "backend_unavailable"}}

# ------------------------------ launcher ------------------------------
def launch_webui(gui=None, html_path: Optional[str] = None, title: str = "SarahMemory"):
    try:
        import webview
    except Exception as exc:
        raise RuntimeError("pywebview_unavailable") from exc
    base_dir = getattr(config, "BASE_DIR", os.getcwd())
    hpath = html_path or getattr(config, "CUSTOM_UI_INDEX", None) or getattr(config, "WEBUI_HTML_PATH", None) or os.path.join(base_dir, "data", "ui", "v9", "index.html")
    if not os.path.isabs(hpath): hpath = os.path.join(base_dir, hpath)
    if not _trusted_local_ui(hpath):
        raise PermissionError("unverified_local_ui_document")
    url = Path(hpath).resolve().as_uri()
    api = WebUIApi(gui=gui)
    window = webview.create_window(title, url, js_api=api, width=1200, height=800, resizable=True)
    api._bind_window(window, url)
    webview.start(debug=False)
    _note_operation("webui", True)

# --------------------------- pywebview launcher ---------------------------
def _resolve_ui_url() -> str:
    """Resolve a verified local UI file, otherwise an unprivileged remote page."""
    try:
        root = _ui_root()
        index = getattr(config, "UI_INDEX_FILE", "SarahMemory.html")
        if not isinstance(index, str) or not index: raise ValueError("invalid_index")
        ui_dir = Path(getattr(config, "UI_DIR", root / "v9"))
        if not ui_dir.is_absolute(): ui_dir = root / ui_dir
        candidate = ui_dir / index
        if _trusted_local_ui(str(candidate)):
            return candidate.resolve().as_uri()
    except Exception:
        pass
    return "https://www.sarahmemory.com/api/data/ui/SarahMemory.html"

class WebUIApi(WebUIBridge):
    """Bounded bridge. Remote and unverified documents have zero Python authority."""
    _worker_slots = threading.BoundedSemaphore(4)

    def __init__(self, gui=None):
        super().__init__(gui=gui)
        self._bridge_window = None
        self._pinned_url = ""

    def _bind_window(self, window: Any, pinned_url: str) -> None:
        self._bridge_window = window
        self._pinned_url = str(pinned_url or "") if _trusted_local_ui(pinned_url) else ""

    def _bridge_allowed(self) -> bool:
        return _verified_window(self._bridge_window, self._pinned_url)

    def _bounded(self, fn: Callable[[], Any], *, timeout: float, operation: str) -> Dict[str, Any]:
        if not self._bridge_allowed():
            return {"ok": False, "error": "untrusted_document", "value": None}
        if not self._worker_slots.acquire(blocking=False):
            return {"ok": False, "error": f"{operation}_capacity_exhausted", "value": None}
        done = threading.Event()
        result: Dict[str, Any] = {"ok": False, "error": None, "value": None}
        def _runner() -> None:
            try:
                result["value"] = fn()
                result["ok"] = True
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {str(exc)[:128]}"
            finally:
                done.set()
                self._worker_slots.release()
        try:
            threading.Thread(target=_runner, name=f"SM-WebUI-{str(operation)[:30]}", daemon=True).start()
        except Exception as exc:
            self._worker_slots.release()
            return {"ok": False, "error": "thread_start_failed", "value": None}
        try:
            wait = max(0.05, min(30.0, float(timeout)))
        except (ValueError, OverflowError, TypeError):
            wait = 10.0
        if not done.wait(wait):
            # The call is STILL RUNNING. Never imply cancellation or no side effects.
            _note_operation(operation, False, "pending_after_timeout")
            return {"ok": False, "error": f"{operation}_pending_after_timeout", "value": None, "pending": True}
        return dict(result)

    def send_text(self, text: str) -> Dict[str, Any]:
        if not self._bridge_allowed(): return {"ok": False, "error": "untrusted_document", "reply": ""}
        if not isinstance(text, str) or len(text.encode("utf8")) > MAX_BRIDGE_TEXT or not text.strip():
            return {"ok": False, "error": "invalid_prompt", "reply": ""}
        timeout = getattr(config, "WEBUI_BRIDGE_TIMEOUT_SECONDS", 20.0)
        call = self._bounded(lambda: self.send_message(text), timeout=timeout, operation="send_text")
        if not call.get("ok"):
            return {"ok": False, "error": call.get("error"), "reply": "", "pending": bool(call.get("pending"))}
        payload = call.get("value") if isinstance(call.get("value"), dict) else {}
        reply = str(payload.get("response") or payload.get("reply") or "").strip()
        return {"ok": bool(reply), "reply": reply, "meta": dict(payload.get("meta") or {"source": "webui_bridge", "intent": "chat"}),
                "error": None if reply else "empty_reply"}

    def get_snapshot(self) -> Dict[str, Any]:
        return {"ok": False, "data_url": None, "error": "local_camera_approval_required", "ts": time.time()}

    def get_stats(self) -> Dict[str, Any]:
        if not self._bridge_allowed(): return {"ok": False, "cpu": None, "mem": None, "error": "untrusted_document"}
        try:
            import psutil
            return {"ok": True, "cpu": float(psutil.cpu_percent(interval=None)), "mem": float(psutil.virtual_memory().percent)}
        except Exception:
            return {"ok": False, "cpu": None, "mem": None, "error": "metrics_unavailable"}

    def list_threads(self, date_iso: str | None = None) -> Dict[str, Any]:
        if not self._bridge_allowed(): return {"ok": False, "items": [], "error": "untrusted_document"}
        day = str(date_iso or time.strftime("%Y-%m-%d", time.localtime()))
        return {"ok": True, "items": self.list_threads_for_date(day), "date": day}

    def get_reminders(self) -> Dict[str, Any]:
        if not self._bridge_allowed(): return {"ok": False, "items": [], "error": "untrusted_document"}
        return {"ok": True, "items": self.list_reminders()}

    def add_reminder(self, title: str, when: str, note: str | None = None) -> Dict[str, Any]:
        if not self._bridge_allowed(): return {"ok": False, "error": "untrusted_document"}
        return {"ok": False, "error": "explicit_approval_path_required"}

    def toggle_reply(self, enabled: bool) -> Dict[str, Any]:
        ok = self.set_flag("REPLY_STATUS", enabled)
        return {"ok": bool(ok), "reply_enabled": bool(getattr(config, "REPLY_STATUS", False)) if ok else None}

    def toggle_compare(self, enabled: bool) -> Dict[str, Any]:
        ok = self.set_flag("API_RESPONSE_CHECK_TRAINER", enabled)
        return {"ok": bool(ok), "compare_enabled": bool(getattr(config, "API_RESPONSE_CHECK_TRAINER", False)) if ok else None}

    def _telecom_call(self, method_name: str, payload: Any, default: Any) -> Any:
        if not self._bridge_allowed(): return default
        # Contacts/recents are read-only. Any outward/state-changing action requires
        # a separate authenticated consent receipt not supplied by this bridge.
        if method_name not in {"telecom_get_contacts", "telecom_list_recents", "telecom_get_remote_frame"}:
            _note_operation(method_name, False, "explicit_approval_path_required")
            return default
        comms = _get_comms()
        method = getattr(comms, method_name, None) if comms is not None else None
        if not callable(method): return default
        call = self._bounded(lambda: method(payload), timeout=3.0, operation=method_name)
        return call.get("value") if call.get("ok") else default

    def telecom_get_contacts(self, payload: dict | None = None):
        return self._telecom_call("telecom_get_contacts", payload, [])

    def telecom_add_contact(self, payload: dict):
        return self._telecom_call("telecom_add_contact", payload, {"ok": False, "error": "explicit_approval_path_required"})

    def telecom_delete_contact(self, payload: dict):
        return self._telecom_call("telecom_delete_contact", payload, {"ok": False, "error": "explicit_approval_path_required"})

    def telecom_list_recents(self, payload: dict | None = None):
        return self._telecom_call("telecom_list_recents", payload, [])

    def telecom_send_message(self, payload: dict):
        return self._telecom_call("telecom_send_message", payload, {"ok": False, "error": "explicit_approval_path_required"})

    def telecom_start_call(self, payload: dict):
        return self._telecom_call("telecom_start_call", payload, {"ok": False, "error": "explicit_approval_path_required"})

    def telecom_end_call(self, payload: dict | None = None):
        return self._telecom_call("telecom_end_call", payload, {"ok": False, "error": "explicit_approval_path_required"})

    def telecom_get_remote_frame(self, payload: dict | None = None):
        return self._telecom_call("telecom_get_remote_frame", payload, None)

def launch_web_ui_detached() -> bool:
    """Start a separate pywebview, confirming a loaded page before returning True.

    A failed/not-yet-loaded backend is never advertised as successfully opened.
    """
    try:
        import webview
    except Exception:
        _note_operation("webui_detached", False, "pywebview_unavailable")
        return False
    url = _resolve_ui_url()
    api = WebUIApi() if _trusted_local_ui(url) else None
    ready = threading.Event()
    failed = threading.Event()
    abort = threading.Event()
    state = {"error": ""}

    def _run():
        window = None
        try:
            window = webview.create_window("SarahMemory — Web UI", url, width=1280, height=820,
                                           resizable=True, js_api=api)
            if api is not None: api._bind_window(window, url)
            # The loaded event is the only reliable startup evidence available
            # without depending on a particular pywebview GUI backend.
            loaded = getattr(getattr(window, "events", None), "loaded", None)
            if loaded is None:
                raise RuntimeError("webview_load_event_unavailable")
            def _on_loaded(*_args, **_kwargs):
                if abort.is_set():
                    if api is not None: api._pinned_url = ""
                    with suppress(Exception): window.destroy()
                else:
                    ready.set()
            loaded += _on_loaded
            if abort.is_set():
                if api is not None: api._pinned_url = ""
                return
            webview.start(debug=False)
            if not ready.is_set():
                raise RuntimeError("webview_closed_before_ready")
        except Exception as exc:
            state["error"] = type(exc).__name__
            failed.set()
            with _BROWSER_LOCK:
                _BROWSER_RUNTIME["webview_failed"] = True

    try:
        thread = threading.Thread(target=_run, daemon=True, name="SarahWebUI")
        thread.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (ready.is_set() or failed.is_set()):
            ready.wait(0.05)
        if ready.is_set():
            with _BROWSER_LOCK:
                _BROWSER_RUNTIME["webview_started"] = True
            _note_operation("webui_detached", True)
            return True
        abort.set()
        if api is not None: api._pinned_url = ""
        _note_operation("webui_detached", False, state["error"] or "startup_unverified")
        return False
    except Exception as exc:
        abort.set()
        _note_operation("webui_detached", False, type(exc).__name__)
        return False


# ---- Telecom bridge into SarahMemoryGUI UnifiedCommsProPanel ----
def _get_comms():
    """Return an already owned communications service; never initialize implicitly."""
    try:
        from SarahMemoryGUI import get_comms_bridge
        return get_comms_bridge()
    except Exception:
        return None


# ====================================================================
# END OF SarahMemoryBrowser.py v9.1.0a
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": "SarahMemoryBrowser",
    "version": "v9.1.0a",
    "category": "Input",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ["browser", "input"],
    "supported_missions": ["Conversation"],
    "supported_omega": ["Ω001", "Ω002", "Ω004"],
    "required_authority": ["Read"],
    "priority": 60,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": "SarahMemoryBrowser.py"},
}

def sml_get_metadata():
    return copy.deepcopy(SML_ORGAN_METADATA)

def sml_health():
    with _BROWSER_LOCK:
        state = dict(_BROWSER_RUNTIME)
    return {"status": "Degraded" if state["webview_failed"] or state["last_action_ok"] is False else "Unknown",
            "availability": None, "integrity": None, "performance": None,
            "reliability": None, "confidence": 0.0, "latency_ms": None,
            "stability": None, "compatibility": None,
            "notes": ["Live UI availability and browser isolation not measured"],
            "webview_started": bool(state["webview_started"]),
            "last_action": state["last_action"], "last_error": state["last_error"]}

def sml_diagnostics():
    return {"status": "OBSERVED", "component": "SarahMemoryBrowser",
            "sml_adapter": True, "metadata": sml_get_metadata(), "health": sml_health()}

def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Observation-only SML adapter; never interpret packet as UI authorization."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(sml_get_metadata())
        return sml_touch_packet(packet, organ="SarahMemoryBrowser", action="observe",
                                note=note or "organ observed packet", updates=None)
    except Exception:
        return packet

# --- SML ORGAN ADAPTER END ---

