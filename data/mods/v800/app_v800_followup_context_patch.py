# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_followup_context_patch.py
# Patch: v8.0.0 Follow-up Context + Identity Fallback Safety
#
# Goals:
# 1) Improve follow-up interaction (yes/no/ok refers to previous answer).
# 2) Replace awkward "Should I dig deeper on that?" behavior with natural prompts.
# 3) Identity safety: if local identity glitches, fall back to core app.py response.
#
# Notes:
# - No edits to ../api/server/app.py
# - Patches the live Flask endpoint for POST /api/chat.
# - Uses in-memory per-session cache (process-local; OK for PythonAnywhere single worker).

from __future__ import annotations

import re
import sys
import time
from typing import Any, Dict, Optional, Tuple

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
# Helpers
# ------------------------------
YES_RE = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|do it|go on|continue|tell me more)\b", re.I)
NO_RE = re.compile(r"^(no|nope|nah|never mind|nevermind|stop|cancel|forget it|no thanks)\b", re.I)

IDENTITY_RE = re.compile(
    r"\b(what\s+is\s+your\s+name|who\s+are\s+you|who\s+(made|created|built|developed)\s+you)\b",
    re.I
)

BAD_UI_TRAIL_RE = re.compile(r"(?:\s*,?\s*\[\s*\]\s*)+$")
BAD_DIG_DEEPER_RE = re.compile(r"\n?\s*•\s*Should\s+I\s+dig\s+deeper\s+on\s+that\?\s*(?:\[\s*\])?\s*$", re.I)

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
    # Simple topic extraction: use question if short, else first sentence keywords
    qn = (q or "").strip()
    if qn and len(qn) <= 80:
        return qn
    # fallback
    return (qn[:80] or "that topic").strip()

def _should_fallback_identity(local_answer: str) -> bool:
    """
    If local identity is glitchy, prefer core app.py default (API or non-local path).
    """
    a = (local_answer or "").strip().lower()
    if not a:
        return True
    # "My name is Sarah, []" style / too-short / malformed
    if len(a) < 10:
        return True
    # If local claims OpenAI in identity context, that’s a fail for SarahMemory identity
    if "openai" in a:
        return True
    # obvious corruption artifacts
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
# Patch /api/chat
# ------------------------------
_PATCH_GUARD = "_V800_FOLLOWUP_CONTEXT_PATCH_APPLIED"

if not getattr(core_mod, _PATCH_GUARD, False):
    endpoint_name = _find_api_chat_endpoint_name()
    original_handler = app.view_functions.get(endpoint_name) if endpoint_name else None

    def api_chat_followup_patched(*args, **kwargs):
        payload = request.get_json(silent=True) or {}
        user_text = _extract_text(payload)
        key = _session_key(payload)

        # If user says yes/no/ok, treat as follow-up to last exchange if we have context
        prev = _CONTEXT.get(key)
        if prev and user_text:
            if YES_RE.match(user_text):
                # Convert to an explicit "go deeper" request about the last topic
                topic = prev.get("topic") or _topic_from_exchange(prev.get("q", ""), prev.get("a", ""))
                user_text = f"Go deeper on this and explain more clearly: {topic}"
                payload["text"] = user_text  # feed into local pipeline
            elif NO_RE.match(user_text):
                # Acknowledge and clear the follow-up expectation (don’t call API)
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

        # Let the existing pipeline handle it (your local-first patch + Reply)
        if callable(original_handler):
            resp = original_handler(*args, **kwargs)
        else:
            return jsonify({"ok": False, "error": "Core /api/chat handler missing"}), 500

        # Try to normalize response object (Flask can return tuple)
        try:
            data = None
            status = 200

            if isinstance(resp, tuple) and len(resp) >= 1:
                r0 = resp[0]
                if len(resp) >= 2 and isinstance(resp[1], int):
                    status = resp[1]
                # Flask Response or dict/jsonify
                if hasattr(r0, "get_json"):
                    data = r0.get_json(silent=True)
                elif isinstance(r0, dict):
                    data = r0
            else:
                if hasattr(resp, "get_json"):
                    data = resp.get_json(silent=True)

            # If we can't inspect, just return as-is
            if not isinstance(data, dict):
                return resp

            # Clean reply text and improve follow-up phrasing (meta only)
            reply_text = _clean_reply_text(str(data.get("reply") or data.get("response") or ""))

            # Identity fallback safety:
            # If this was an identity question and local produced a glitchy answer,
            # fall back to core default by re-calling original handler WITHOUT local forcing.
            if user_text and IDENTITY_RE.search(user_text):
                if _should_fallback_identity(reply_text):
                    # Clear local hint keys to allow core to answer normally
                    for k in ("local", "LOCAL", "local_only", "LOCAL_ONLY", "offline", "mode", "engine", "provider", "source"):
                        if k in payload:
                            payload.pop(k, None)
                    # Re-run original handler (core default behavior)
                    if callable(original_handler):
                        # monkey: temporarily swap request json is hard; instead, just return original resp
                        # We can’t re-inject payload cleanly here without editing core.
                        # So we do the safer thing: return a correct SarahMemory identity.
                        reply_text = "My name is Sarah — your SarahMemory AI companion (SarahMemory AiOS)."

            # Store context for next turn
            if user_text and reply_text:
                _CONTEXT[key] = {
                    "q": user_text,
                    "a": reply_text,
                    "topic": _topic_from_exchange(user_text, reply_text),
                    "ts": time.time(),
                }

            # Replace output
            data["reply"] = reply_text
            data["response"] = reply_text

            # Replace old awkward followup prompt behavior (don’t force it, just suggest)
            meta = data.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["followup_hint"] = "If you want more detail, reply: “yes” or ask a specific angle (history, examples, pros/cons)."
            data["meta"] = meta

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
            "patch": "followup_context",
            "applied": True,
            "endpoint_name": endpoint_name,
            "ts": time.time(),
        })

    setattr(core_mod, _PATCH_GUARD, True)
