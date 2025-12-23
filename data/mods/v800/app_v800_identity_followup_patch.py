# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_identity_followup_patch.py
# Patch: v8.0.0 LOCAL Identity + Follow-up Repair (SarahMemoryReply)
#
# Fixes:
# - Identity drift in LOCAL mode (prevents "developed by OpenAI" + generic assistant identity)
# - Removes "• Should I dig deeper on that? []" and any trailing "[]"
# - Provides better follow-up suggestions (no empty brackets)
# - Makes yes/no followups work by caching last topic per-session (best effort)

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

# Patch the LOCAL reply router
import SarahMemoryReply as reply  # type: ignore

# ---------------------------------------------
# Guards
# ---------------------------------------------
_PATCH_GUARD = "_V800_IDENTITY_FOLLOWUP_PATCH_APPLIED"

# ---------------------------------------------
# Per-session lightweight cache (process-local)
# ---------------------------------------------
_LAST: Dict[str, Dict[str, Any]] = {}  # sid -> {"q": str, "a": str, "ts": float, "topic": str}

# ---------------------------------------------
# Patterns
# ---------------------------------------------
IDENTITY_PAT = re.compile(
    r"\b("
    r"what\s+is\s+your\s+name|"
    r"who\s+are\s+you|"
    r"what\s+are\s+you|"
    r"who\s+(made|created|built|developed|engineered|designed)\s+you|"
    r"who\s+is\s+your\s+(creator|developer|engineer|designer)"
    r")\b",
    re.I,
)

YES_PAT = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|go on|continue|tell me more)\b", re.I)
NO_PAT = re.compile(r"^(no|nope|nah|never mind|nevermind|stop|cancel|forget it|no thanks)\b", re.I)

# UI corruption cleanup
TRAILING_EMPTY_LIST_PAT = re.compile(r"(?:\s*,?\s*\[\s*\]\s*)+$")
DIG_DEEPER_LINE_PAT = re.compile(
    r"\n?\s*•\s*Should\s+I\s+dig\s+deeper\s+on\s+that\?\s*(?:\[\s*\])?\s*$",
    re.I,
)

BAD_ENC = {
    "ðŸ™‚": "🙂",
    "ðŸ˜Š": "😊",
    "ðŸ˜€": "😀",
    "ðŸ˜¢": "😢",
}

def _clean_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    out = s
    for bad, good in BAD_ENC.items():
        out = out.replace(bad, good)
    out = DIG_DEEPER_LINE_PAT.sub("", out)
    out = TRAILING_EMPTY_LIST_PAT.sub("", out)
    # remove stray standalone brackets
    out = out.replace("[]", "").strip()
    # remove duplicate "The answer is" prefixing
    out = re.sub(r"^(The answer is\s+)+", "", out, flags=re.I).strip()
    return out

def _identity_answer(question: str) -> str:
    q = (question or "").strip().lower()
    if "name" in q:
        return "My name is Sarah — your SarahMemory AI companion."
    if "who developed" in q or "who made" in q or "who created" in q or "who built" in q:
        return "I was created and engineered for the SarahMemory AiOS project by Brian Lee Baros (SOFTDEV0 LLC)."
    if "what are you" in q or "who are you" in q:
        return "I’m Sarah — the SarahMemory AI companion (SarahMemory AiOS), built to help you locally first, then web, then APIs when needed."
    return "I’m Sarah — your SarahMemory AI companion."

def _topic_hint(user_text: str, response_text: str) -> str:
    # Keep it simple, stable
    t = (user_text or "").strip()
    if t:
        return t[:120]
    return (response_text or "").strip()[:120] or "that topic"

def _followups_for(topic: str) -> list:
    # Natural options (no empty [])
    return [
        f"Want a deeper explanation of: {topic}?",
        "Want examples or a quick summary?",
        "Should I compare it to something similar?",
    ]

def _get_session_id(kwargs: Dict[str, Any]) -> str:
    # SarahMemoryReply may accept session_id in kwargs; fall back to stable default
    sid = kwargs.get("session_id") or kwargs.get("sessionId") or kwargs.get("sid")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return "default"

# ---------------------------------------------
# Patch route_reply (idempotent)
# ---------------------------------------------
if not getattr(reply, _PATCH_GUARD, False):
    _orig_route_reply = reply.route_reply

    def route_reply_patched(*args, **kwargs):
        """
        Wrap SarahMemoryReply.route_reply:
        - Clean broken UI artifacts
        - Handle identity questions deterministically (never OpenAI attribution)
        - Interpret yes/no as follow-up to prior topic (session cache)
        - Provide better follow-up suggestions
        """
        # Extract user_text
        user_text = kwargs.get("user_text")
        if user_text is None and args:
            user_text = args[0]
        user_text = user_text or ""
        sid = _get_session_id(kwargs)

        # Yes/no follow-up handling (LOCAL)
        prev = _LAST.get(sid)
        if prev and isinstance(user_text, str):
            if YES_PAT.match(user_text.strip()):
                # Expand into explicit request about previous topic
                topic = prev.get("topic") or _topic_hint(prev.get("q", ""), prev.get("a", ""))
                user_text = f"Please go deeper and explain more about: {topic}"
                kwargs["user_text"] = user_text
            elif NO_PAT.match(user_text.strip()):
                _LAST.pop(sid, None)
                return {
                    "ok": True,
                    "response": "No problem — what would you like to talk about next?",
                    "reply": "No problem — what would you like to talk about next?",
                    "meta": {
                        "intent": "followup_no",
                        "source": "identity_followup_patch",
                        "followups": ["Ask me anything.", "Give me a topic you’re curious about.", "Or say: help"],
                    },
                }

        # Identity override (LOCAL always wins here)
        if IDENTITY_PAT.search(str(user_text)):
            ans = _identity_answer(str(user_text))
            ans = _clean_text(ans)
            out = {
                "ok": True,
                "response": ans,
                "reply": ans,
                "meta": {
                    "intent": "identity",
                    "source": "identity_followup_patch",
                    "followups": ["Want the short version or the full backstory?", "Want to change my name/voice settings?", "Want help with a task?"],
                },
            }
            _LAST[sid] = {"q": str(user_text), "a": ans, "topic": _topic_hint(str(user_text), ans), "ts": time.time()}
            return out

        # Otherwise run original
        bundle = _orig_route_reply(*args, **kwargs)

        # Normalize bundle
        if not isinstance(bundle, dict):
            return bundle

        # Clean response text
        resp = bundle.get("response") or bundle.get("reply") or ""
        resp = _clean_text(resp)

        # If response still claims OpenAI for identity-ish prompts, correct it
        if IDENTITY_PAT.search(str(user_text)) and re.search(r"\bopenai\b", resp, re.I):
            resp = _identity_answer(str(user_text))
            resp = _clean_text(resp)

        bundle["response"] = resp
        bundle["reply"] = resp

        # Fix followups: remove empty, remove brackets, replace “dig deeper” with usable options
        meta = bundle.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        fu = meta.get("followups")

        # If followups missing or garbage, provide good ones
        if not fu or (isinstance(fu, list) and len([x for x in fu if isinstance(x, str) and x.strip()]) == 0):
            topic = _topic_hint(str(user_text), resp)
            meta["followups"] = _followups_for(topic)
        else:
            # Clean followups
            cleaned = []
            if isinstance(fu, list):
                for x in fu:
                    if isinstance(x, str):
                        cx = _clean_text(x)
                        if cx:
                            cleaned.append(cx)
            meta["followups"] = cleaned or _followups_for(_topic_hint(str(user_text), resp))

        # Never include the old dig-deeper line in meta
        meta["source"] = meta.get("source") or "local"
        bundle["meta"] = meta

        # Update cache for follow-up yes/no
        if str(user_text).strip() and resp.strip():
            _LAST[sid] = {"q": str(user_text), "a": resp, "topic": _topic_hint(str(user_text), resp), "ts": time.time()}

        return bundle

    reply.route_reply = route_reply_patched  # type: ignore
    setattr(reply, _PATCH_GUARD, True)
