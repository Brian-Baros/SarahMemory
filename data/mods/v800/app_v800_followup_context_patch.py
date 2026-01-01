# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_followup_context_patch.py
# Patch: v8.0.0 Follow-up Context + Identity Fallback Safety + Math Routing + Avatar Speech Cues
#
# Goals:
# 1) Improve follow-up interaction (yes/no/ok refers to previous answer).
# 2) Replace awkward "Should I dig deeper on that?" behavior with natural prompts.
# 3) Identity safety: if local identity glitches, fall back to safe SarahMemory identity string.
# 4) Math routing: detect calculator-style queries and answer locally via SarahMemoryWebSYM when possible.
# 5) Avatar speech cues: attach meta.avatar_speech so WebUI can animate; trigger local python avatar lipsync best-effort.
#
# - Patches the live Flask endpoint for POST /api/chat.
# - IMPORTANT: this patch intentionally becomes the SINGLE wrapper for /api/chat.
#   Remove/disable other /api/chat wrappers (math_patch, avatar_lipsync_patch) to avoid collisions.

from __future__ import annotations

import re
import sys
import time
from typing import Any, Dict, Optional

from flask import request, jsonify

# ------------------------------
# Attach to live Flask app module
# ------------------------------
core_mod = sys.modules.get("app")
if core_mod is None:
    try:
        import app as core_mod  # type: ignore
    except Exception:
        core_mod = None
if core_mod is None or not hasattr(core_mod, "app"):
    raise RuntimeError("followup context patch: could not locate live Flask app instance")

app = core_mod.app

# ------------------------------
# Small context cache (per session)
# ------------------------------
# key -> {"q": str, "a": str, "ts": float, "topic": str}
_CONTEXT: Dict[str, Dict[str, Any]] = {}

# ------------------------------
# Regex + constants
# ------------------------------
YES_RE = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|do it|go on|continue|tell me more)\b", re.I)
NO_RE = re.compile(r"^(no|nope|nah|never mind|nevermind|stop|cancel|forget it|no thanks)\b", re.I)

IDENTITY_RE = re.compile(
    r"\b(what\s+is\s+your\s+name|who\s+are\s+you|who\s+(made|created|built|developed)\s+you)\b",
    re.I
)

BAD_UI_TRAIL_RE = re.compile(r"(?:\s*,?\s*\[\s*\]\s*)+$")
BAD_DIG_DEEPER_RE = re.compile(r"\n?\s*•\s*Should\s+I\s+dig\s+deeper\s+on\s+that\?\s*(?:\[\s*\])?\s*$", re.I)

# ------------------------------
# Helpers (existing + merged)
# ------------------------------
def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "local")


def _detect_local_mode(payload: Dict[str, Any]) -> bool:
    """
    Matches existing conventions from earlier patches:
    Accepts explicit local hints in request payload OR server config.LOCAL_ONLY_MODE.
    """
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


def _session_key(payload: Dict[str, Any]) -> str:
    # Prefer explicit session ids if present
    for k in ("session_id", "sessionId", "sid", "chat_id", "chatId", "client_id", "clientId"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fall back to remote addr + user-agent (best effort)
    ua = request.headers.get("User-Agent", "")[:120]
    ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
    return f"{ip}|{ua}"


def _clean_reply_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    out = s
    out = out.replace("ðŸ™‚", "🙂").replace("ðŸ˜Š", "😊")
    out = BAD_DIG_DEEPER_RE.sub("", out)
    out = BAD_UI_TRAIL_RE.sub("", out)
    out = re.sub(r"^(The answer is\s+)+", "", out, flags=re.I)
    return out.strip()


def _topic_from_exchange(q: str, a: str) -> str:
    qn = (q or "").strip()
    if qn and len(qn) <= 80:
        return qn
    return (qn[:80] or "that topic").strip()


def _should_fallback_identity(local_answer: str) -> bool:
    a = (local_answer or "").strip().lower()
    if not a:
        return True
    if len(a) < 10:
        return True
    if "openai" in a:
        return True
    if "[]" in a or "ðŸ" in a:
        return True
    return False


def _find_api_chat_endpoint_name() -> Optional[str]:
    try:
        for rule in app.url_map.iter_rules():
            if rule.rule.rstrip("/") == "/api/chat" and ("POST" in getattr(rule, "methods", set())):
                return rule.endpoint
    except Exception:
        pass
    return None


# ------------------------------
# Math routing (merged)
# ------------------------------
def _websym_math_answer(text: str) -> Optional[str]:
    """
    If SarahMemoryWebSYM can confidently treat the text as math, return calculator answer.
    Otherwise return None and let the normal pipeline handle it.
    """
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


# ------------------------------
# Avatar speech cues (merged)
# ------------------------------
def _estimate_speech_duration_seconds(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.8
    words = max(1, len(t.split()))
    secs = words / 2.2
    if secs < 0.8:
        secs = 0.8
    if secs > 12.0:
        secs = 12.0
    return float(secs)


def _basic_mouth_cues(text: str, duration_s: float) -> list:
    steps = max(6, min(30, int(duration_s * 10)))  # ~10 updates/sec
    cues = []
    if steps <= 1:
        return [{"t": 0, "v": 0.6}, {"t": int(duration_s * 1000), "v": 0.0}]
    for i in range(steps):
        t_ms = int((i / (steps - 1)) * duration_s * 1000)
        v = 0.15 + (0.55 if (i % 2 == 0) else 0.30)
        cues.append({"t": t_ms, "v": v})
    cues.append({"t": int(duration_s * 1000), "v": 0.0})
    return cues


def _try_local_avatar_lipsync(duration_s: float) -> None:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        run_mode = getattr(config, "RUN_MODE", "cloud")
        if str(run_mode).lower() not in ("local", "desktop"):
            return
    except Exception:
        return

    try:
        import SarahMemoryAvatar as Avatar  # type: ignore
        if hasattr(Avatar, "simulate_lip_sync_async"):
            Avatar.simulate_lip_sync_async(duration=duration_s)
    except Exception:
        return


# ------------------------------
# Patch /api/chat (single wrapper)
# ------------------------------
_PATCH_GUARD = "_V800_FOLLOWUP_CONTEXT_PATCH_APPLIED"

if not getattr(core_mod, _PATCH_GUARD, False):
    endpoint_name = _find_api_chat_endpoint_name()
    original_handler = app.view_functions.get(endpoint_name) if endpoint_name else None

    def api_chat_followup_patched(*args, **kwargs):
        payload = request.get_json(silent=True) or {}
        user_text = _extract_text(payload)
        key = _session_key(payload)

        # --------------------------
        # Follow-up: yes/no handling
        # --------------------------
        prev = _CONTEXT.get(key)
        if prev and user_text:
            if YES_RE.match(user_text):
                topic = prev.get("topic") or _topic_from_exchange(prev.get("q", ""), prev.get("a", ""))
                user_text = f"Go deeper on this and explain more clearly: {topic}"
                payload["text"] = user_text
            elif NO_RE.match(user_text):
                _CONTEXT.pop(key, None)
                return jsonify({
                    "ok": True,
                    "reply": "No problem — what would you like to talk about next?",
                    "meta": {
                        "engine": "followup_context_patch",
                        "source": "followup_no",
                        "ts": time.time(),
                    }
                }), 200

        # --------------------------
        # Math routing (early return)
        # --------------------------
        try:
            local_mode = _detect_local_mode(payload)
        except Exception:
            local_mode = False

        websym_ans = _websym_math_answer(user_text)
        if websym_ans is not None:
            # Store context for follow-ups too (so "yes" after a math answer can elaborate)
            _CONTEXT[key] = {
                "q": user_text,
                "a": websym_ans,
                "topic": _topic_from_exchange(user_text, websym_ans),
                "ts": time.time(),
            }

            # Add avatar cues for Web UI even on math fast-path
            dur_s = _estimate_speech_duration_seconds(websym_ans)
            cues = _basic_mouth_cues(websym_ans, dur_s)
            _try_local_avatar_lipsync(dur_s)

            return jsonify({
                "ok": True,
                "reply": websym_ans,
                "meta": {
                    "source": "websym_math",
                    "engine": "SarahMemoryWebSYM",
                    "local_mode": bool(local_mode),
                    "ts": time.time(),
                    "patched_endpoint": endpoint_name or "unknown",
                    "avatar_speech": {
                        "speak": True,
                        "duration_ms": int(dur_s * 1000),
                        "cues": cues,
                        "ts": time.time(),
                    },
                    "followup_hint": "If you want more detail, reply: “yes” or ask a specific angle (examples, pros/cons).",
                },
            }), 200

        # --------------------------
        # Normal pipeline
        # --------------------------
        if callable(original_handler):
            resp = original_handler(*args, **kwargs)
        else:
            return jsonify({"ok": False, "error": "Core /api/chat handler missing"}), 500

        # Normalize response object (Flask can return tuple)
        try:
            data = None
            status = 200

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

            if not isinstance(data, dict):
                return resp

            reply_text = _clean_reply_text(str(data.get("reply") or data.get("response") or ""))

            # Identity fallback safety
            if user_text and IDENTITY_RE.search(user_text):
                if _should_fallback_identity(reply_text):
                    reply_text = "My name is Sarah — your SarahMemory AI companion (SarahMemory AiOS)."

            # Store context for next turn
            if user_text and reply_text:
                _CONTEXT[key] = {
                    "q": user_text,
                    "a": reply_text,
                    "topic": _topic_from_exchange(user_text, reply_text),
                    "ts": time.time(),
                }

            # Attach avatar cues
            if reply_text:
                dur_s = _estimate_speech_duration_seconds(reply_text)
                cues = _basic_mouth_cues(reply_text, dur_s)
                _try_local_avatar_lipsync(dur_s)

                meta = data.get("meta")
                if not isinstance(meta, dict):
                    meta = {}

                meta["avatar_speech"] = {
                    "speak": True,
                    "duration_ms": int(dur_s * 1000),
                    "cues": cues,
                    "ts": time.time(),
                }

                meta["followup_hint"] = "If you want more detail, reply: “yes” or ask a specific angle (history, examples, pros/cons)."
                data["meta"] = meta

            data["reply"] = reply_text
            data["response"] = reply_text

            return jsonify(data), status

        except Exception:
            # Never break chat if our patch parsing fails
            return resp

    # Apply patch
    if endpoint_name and endpoint_name in app.view_functions:
        app.view_functions[endpoint_name] = api_chat_followup_patched

    @app.get("/api/mods/v800/followup_context_patch_status")
    def _followup_patch_status():
        return jsonify({
            "ok": True,
            "mod": "v800",
            "patch": "followup_context_combined",
            "applied": True,
            "endpoint_name": endpoint_name,
            "ts": time.time(),
        })

    setattr(core_mod, _PATCH_GUARD, True)
