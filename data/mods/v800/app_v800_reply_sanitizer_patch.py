# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_reply_sanitizer_patch.py
# Patch: v8.0.0 LOCAL Reply Sanitizer & Intent Isolation
#
# Purpose:
# - Fix corrupted LOCAL replies from SarahMemoryReply.py
# - Prevent stale math / wrong intent bleed
# - Clean follow-ups & encoding
# - Guarantee one intent -> one coherent response

from __future__ import annotations

import sys
import re
from typing import Dict, Any

# Locate core app module (already loaded by WSGI)
core_mod = sys.modules.get("app")
if core_mod is None:
    raise RuntimeError("reply sanitizer patch: core app module not loaded")

# Import reply module
import SarahMemoryReply as reply  # type: ignore


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
MATH_ONLY_RE = re.compile(r"^[\d\.\s\+\-\*\/\(\)=eE]+$")

def _looks_like_math(text: str) -> bool:
    return bool(MATH_ONLY_RE.match(text.strip()))

def _sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Fix broken encoding artifacts
    s = s.replace("ðŸ™‚", "🙂")
    s = s.replace("ðŸ˜Š", "😊")
    # Remove duplicated prefixes
    s = re.sub(r"^(The answer is\s+)+", "", s, flags=re.I)
    return s.strip()


def _sanitize_followups(meta: Dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        return
    fu = meta.get("followups")
    if not fu or not isinstance(fu, list):
        meta.pop("followups", None)
        return
    # remove empty / junk followups
    meta["followups"] = [f for f in fu if isinstance(f, str) and f.strip()]
    if not meta["followups"]:
        meta.pop("followups", None)


# ---------------------------------------------------------------------
# Patch route_reply (idempotent)
# ---------------------------------------------------------------------
_PATCH_GUARD = "_V800_REPLY_SANITIZER_APPLIED"

if not getattr(reply, _PATCH_GUARD, False):

    original_route_reply = reply.route_reply

    def route_reply_sanitized(*args, **kwargs):
        user_text = kwargs.get("user_text") or (args[0] if args else "")
        bundle = original_route_reply(*args, **kwargs)

        if not isinstance(bundle, dict):
            return bundle

        response = bundle.get("response") or bundle.get("reply") or ""
        meta = bundle.get("meta") or {}

        response = _sanitize_text(response)

        # Prevent math-only answers for non-math questions
        if not _looks_like_math(user_text) and _looks_like_math(response):
            response = "I can explain that in words if you'd like — could you clarify what you'd like to know?"

        # Clean followups
        _sanitize_followups(meta)

        bundle["response"] = response
        bundle["reply"] = response
        bundle["meta"] = meta

        return bundle

    reply.route_reply = route_reply_sanitized  # type: ignore
    setattr(reply, _PATCH_GUARD, True)
