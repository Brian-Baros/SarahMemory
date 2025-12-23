# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_avatar_lipsync_patch.py
# Patch: v8.0.0 Avatar Lip-Sync / Speaking Cues
#
# Goal:
# - When /api/chat returns a reply, signal the Avatar system to "speak":
#   - In LOCAL desktop mode: call SarahMemoryAvatar.simulate_lip_sync_async(duration)
#   - In WebUI: attach meta.avatar_speech with duration + basic mouth cues so the front-end can animate
#
# Notes:
# - No edits to ../api/server/app.py
# - Safe to load multiple times (guarded)
# - Headless/cloud environments won’t crash if avatar modules aren’t available.

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

from flask import jsonify

# ---------------------------------------------------------------------
# Locate live Flask app instance (same object WSGI exports)
# ---------------------------------------------------------------------
core_mod = sys.modules.get("app")
if core_mod is None:
    try:
        import app as core_mod  # type: ignore
    except Exception:
        core_mod = None
if core_mod is None or not hasattr(core_mod, "app"):
    raise RuntimeError("avatar_lipsync patch: could not locate live Flask app")

app = core_mod.app

_PATCH_GUARD = "_V800_AVATAR_LIPSYNC_PATCH_APPLIED"


def _estimate_speech_duration_seconds(text: str) -> float:
    """
    Rough duration estimate so avatar can animate even without audio timestamps.
    2.2 words/sec is a reasonable conversational pace.
    """
    t = (text or "").strip()
    if not t:
        return 0.8
    words = max(1, len(t.split()))
    secs = words / 2.2
    # clamp to reasonable bounds
    if secs < 0.8:
        secs = 0.8
    if secs > 12.0:
        secs = 12.0
    return float(secs)


def _basic_mouth_cues(text: str, duration_s: float) -> list:
    """
    Simple mouth open/close cues (not real visemes).
    Frontend can treat these as amplitude envelopes.
    """
    steps = max(6, min(30, int(duration_s * 10)))  # ~10 updates/sec
    cues = []
    if steps <= 1:
        return [{"t": 0, "v": 0.6}, {"t": int(duration_s * 1000), "v": 0.0}]
    for i in range(steps):
        t_ms = int((i / (steps - 1)) * duration_s * 1000)
        # mild oscillation pattern without importing math
        v = 0.15 + (0.55 if (i % 2 == 0) else 0.30)
        cues.append({"t": t_ms, "v": v})
    cues.append({"t": int(duration_s * 1000), "v": 0.0})
    return cues


def _try_local_avatar_lipsync(duration_s: float) -> None:
    """
    Only triggers local python avatar lip-sync if module exists.
    Non-fatal in headless environments.
    """
    try:
        import SarahMemoryGlobals as config  # type: ignore
        run_mode = getattr(config, "RUN_MODE", "cloud")
        # Only attempt actual python-side avatar animation in local mode
        if str(run_mode).lower() not in ("local", "desktop"):
            return
    except Exception:
        # If config not available, do not assume local
        return

    try:
        import SarahMemoryAvatar as Avatar  # type: ignore
        if hasattr(Avatar, "simulate_lip_sync_async"):
            Avatar.simulate_lip_sync_async(duration=duration_s)
    except Exception:
        return


def _find_api_chat_endpoint_name() -> Optional[str]:
    try:
        for rule in app.url_map.iter_rules():
            if rule.rule.rstrip("/") == "/api/chat" and ("POST" in getattr(rule, "methods", set())):
                return rule.endpoint
    except Exception:
        pass
    return None


if not getattr(core_mod, _PATCH_GUARD, False):
    endpoint_name = _find_api_chat_endpoint_name()
    original_handler = app.view_functions.get(endpoint_name) if endpoint_name else None

    def api_chat_avatar_lipsync_patched(*args, **kwargs):
        # Call the existing /api/chat handler first (do not break routing)
        if not callable(original_handler):
            return jsonify({"ok": False, "error": "Core /api/chat handler missing"}), 500

        resp = original_handler(*args, **kwargs)

        # Normalize Flask response -> dict
        data: Optional[Dict[str, Any]] = None
        status = 200

        try:
            if isinstance(resp, tuple) and len(resp) >= 1:
                r0 = resp[0]
                if len(resp) >= 2 and isinstance(resp[1], int):
                    status = resp[1]
                if hasattr(r0, "get_json"):
                    data = r0.get_json(silent=True)
                elif isinstance(r0, dict):
                    data = r0
            else:
                if hasattr(resp, "get_json"):
                    data = resp.get_json(silent=True)
        except Exception:
            data = None

        # If we can't read JSON, just return as-is
        if not isinstance(data, dict):
            return resp

        reply_text = str(data.get("reply") or data.get("response") or "").strip()
        if not reply_text:
            return resp

        # Estimate duration + cues
        dur_s = _estimate_speech_duration_seconds(reply_text)
        cues = _basic_mouth_cues(reply_text, dur_s)

        # Trigger local python avatar lip-sync (best effort)
        _try_local_avatar_lipsync(dur_s)

        # Attach cues for WebUI 3D avatar to animate
        meta = data.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        meta["avatar_speech"] = {
            "speak": True,
            "duration_ms": int(dur_s * 1000),
            "cues": cues,  # mouth amplitude envelope
            "ts": time.time(),
        }

        data["meta"] = meta

        return jsonify(data), status

    # Apply patch
    if endpoint_name and endpoint_name in app.view_functions:
        app.view_functions[endpoint_name] = api_chat_avatar_lipsync_patched

    @app.get("/api/mods/v800/avatar_lipsync_patch_status")
    def _avatar_lipsync_patch_status():
        return jsonify(
            {
                "ok": True,
                "mod": "v800",
                "patch": "avatar_lipsync",
                "applied": True,
                "endpoint_name": endpoint_name,
                "ts": time.time(),
            }
        )

    setattr(core_mod, _PATCH_GUARD, True)
