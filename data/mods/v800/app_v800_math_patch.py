# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_math_patch.py
# Patch: v8.0.0 Math Routing Fix (LOCAL-first for calculator via SarahMemoryWebSYM.py)

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

from flask import request, jsonify

# ---------------------------------------------------------------------
# Locate the SAME core module instance used by WSGI.
# WSGI imports: from app import app as application
# So we must patch THAT module, not api.server.app (which can be a second load).
# ---------------------------------------------------------------------
core_mod = None

# 1) If WSGI already imported it, it will be in sys.modules
if "app" in sys.modules:
    core_mod = sys.modules["app"]

# 2) Otherwise try importing it as "app"
if core_mod is None:
    try:
        import app as core_mod  # type: ignore
    except Exception:
        core_mod = None

# 3) Last resort: try api.server.app (only if your deployment truly uses that path)
if core_mod is None:
    try:
        from api.server import app as core_mod  # type: ignore
    except Exception:
        core_mod = None

if core_mod is None or not hasattr(core_mod, "app"):
    raise RuntimeError("v800 math patch: could not locate core Flask 'app' instance to patch.")

app = core_mod.app  # the live Flask app instance


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "local")


def _detect_local_mode(payload: Dict[str, Any]) -> bool:
    candidates = [
        payload.get("local"),
        payload.get("LOCAL"),
        payload.get("local_only"),
        payload.get("LOCAL_ONLY"),
        payload.get("use_local"),
        payload.get("useLocal"),
        payload.get("offline"),
        payload.get("run_mode"),
        payload.get("mode"),
        payload.get("engine"),
        payload.get("provider"),
        payload.get("source"),
    ]
    for c in candidates:
        if _truthy(c):
            return True
        if isinstance(c, str) and c.strip().lower() in ("local", "offline", "websym", "internal"):
            return True

    # respect server-side global if present
    try:
        cfg = getattr(core_mod, "config", None)
        if cfg is not None and getattr(cfg, "LOCAL_ONLY_MODE", False):
            return True
    except Exception:
        pass

    return False


def _extract_text(payload: Dict[str, Any]) -> str:
    for k in ("text", "message", "prompt", "query", "q", "input", "user_text", "user"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    msgs = payload.get("messages")
    if isinstance(msgs, list) and msgs:
        for item in reversed(msgs):
            if isinstance(item, dict) and item.get("role") == "user":
                c = item.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()

    return ""


def _websym_math_answer(text: str) -> Optional[str]:
    try:
        from SarahMemoryWebSYM import WebSemanticSynthesizer, is_math_expression  # type: ignore
    except Exception:
        return None

    q = (text or "").strip()
    if not q:
        return None

    try:
        # check if math
        try:
            if hasattr(WebSemanticSynthesizer, "is_math_query"):
                if not WebSemanticSynthesizer.is_math_query(q):
                    if not (callable(is_math_expression) and is_math_expression(q)):
                        return None
            else:
                if not (callable(is_math_expression) and is_math_expression(q)):
                    return None
        except Exception:
            pass

        if hasattr(WebSemanticSynthesizer, "sarah_calculator"):
            ans = WebSemanticSynthesizer.sarah_calculator(q, original_query=q)
            ans = str(ans or "").strip()
            return ans or None
        return None
    except Exception:
        return None


def _find_api_chat_endpoint_name() -> Optional[str]:
    try:
        for rule in app.url_map.iter_rules():
            if rule.rule.rstrip("/") == "/api/chat" and ("POST" in getattr(rule, "methods", set())):
                return rule.endpoint
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------
# Apply patch (idempotent)
# ---------------------------------------------------------------------
_PATCH_GUARD = "_V800_MATH_PATCH_APPLIED"

if not getattr(core_mod, _PATCH_GUARD, False):
    endpoint_name = _find_api_chat_endpoint_name()
    original_handler = app.view_functions.get(endpoint_name) if endpoint_name else None

    def api_chat_math_patched(*args, **kwargs):
        try:
            payload = request.get_json(silent=True) or {}
            text = _extract_text(payload)
            local_mode = _detect_local_mode(payload)

            websym_ans = _websym_math_answer(text)

            if websym_ans is not None:
                return jsonify(
                    {
                        "ok": True,
                        "reply": websym_ans,
                        "meta": {
                            "source": "websym_math",
                            "engine": "SarahMemoryWebSYM",
                            "local_mode": bool(local_mode),
                            "ts": time.time(),
                            "patched_endpoint": endpoint_name or "unknown",
                        },
                    }
                ), 200

            # fallback to core handler
            if callable(original_handler):
                return original_handler(*args, **kwargs)

            return jsonify(
                {
                    "ok": False,
                    "error": "Could not locate core /api/chat handler to wrap.",
                    "meta": {"endpoint_name": endpoint_name},
                }
            ), 500

        except Exception:
            if callable(original_handler):
                return original_handler(*args, **kwargs)
            return jsonify({"ok": False, "error": "v800 math patch error"}), 500

    # Patch the real endpoint
    if endpoint_name and endpoint_name in app.view_functions:
        app.view_functions[endpoint_name] = api_chat_math_patched

    # Status endpoint
    @app.get("/api/mods/v800/math_patch_status")
    def _math_patch_status():
        return jsonify(
            {
                "ok": True,
                "mod": "v800",
                "patch": "math",
                "applied": True,
                "endpoint_name": endpoint_name,
                "ts": time.time(),
            }
        )

    setattr(core_mod, _PATCH_GUARD, True)
