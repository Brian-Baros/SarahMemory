# --==The SarahMemory Project==--
# File: ../data/mods/v800/sm_v800_voice_patch.py
# Patch: v8.0.0 Voice (Consolidated)
#
# Consolidates the following patch files into ONE:
# - sm_v800_voice_pyttsx_patch.py
# - sm_v800_voice_voice_tts_patch.py
# - sm_v800_dl_voice_tts_patch.py
# - sm_v800_globals_voice_tts_patch.py
# - sm_v800_sobje_voice_tts_patch.py
#
# Goals:
# - Keep ALL changes as monkey-patches (no core edits).
# - Improve pyttsx3 voice selection (prefer female / less robotic) + gentle tuning.
# - Keep headless-safe (Linux/PythonAnywhere) + Windows-safe.
# - Reduce common “TTS / mic init” crash paths by wrapping in fail-soft guards
#   ONLY when those call-sites exist (verified-at-runtime).
#
# IMPORTANT:
# - This file is designed to be conservative: it will not invent missing functions.
# - If a target module/function doesn't exist, this patch skips it quietly.

from __future__ import annotations

import logging
import traceback
from typing import Optional

logger = logging.getLogger("sm_v800_voice_patch")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_APPLIED = False

# =============================================================================
# Part 1 — pyttsx3 “female voice + soft tuning” (merged from sm_v800_voice_pyttsx_patch)
# =============================================================================

PREFERRED_KEYWORDS = (
    "zira",     # Microsoft Zira
    "aria",     # Microsoft Aria
    "jenny",    # Microsoft Jenny
    "eva",
    "emma",
    "hazel",
    "susan",
    "natasha",
    "female",
)

SOFT_RATE = 170
SOFT_VOLUME = 0.95


def _pick_female_voice(engine) -> Optional[str]:
    """Select a female voice by keyword; fallback to any voice that doesn't look explicitly male."""
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        voices = []

    # Pass 1: keyword match
    for v in voices:
        try:
            name = (getattr(v, "name", "") or "").lower()
            vid = (getattr(v, "id", "") or "").lower()
            if any(k in name or k in vid for k in PREFERRED_KEYWORDS):
                engine.setProperty("voice", v.id)
                return getattr(v, "name", v.id)
        except Exception:
            continue

    # Pass 2: avoid explicit "male" if possible
    for v in voices:
        try:
            name = (getattr(v, "name", "") or "").lower()
            if "male" not in name:
                engine.setProperty("voice", v.id)
                return getattr(v, "name", v.id)
        except Exception:
            continue

    return None


def _apply_engine_tuning(SarahMemoryVoice) -> None:
    """Apply voice selection + basic tuning directly to SarahMemoryVoice.engine (pyttsx3)."""
    engine = getattr(SarahMemoryVoice, "engine", None)
    if engine is None:
        return

    selected = _pick_female_voice(engine)
    if selected:
        logger.info("[v800 voice] Selected TTS voice: %s", selected)

    try:
        engine.setProperty("rate", SOFT_RATE)
    except Exception:
        pass

    try:
        engine.setProperty("volume", SOFT_VOLUME)
    except Exception:
        pass


# =============================================================================
# Part 2 — Fail-soft wrappers (merged intent of other voice TTS patch stubs)
# =============================================================================

def _safe_wrap(module, func_name: str, wrapper_factory):
    """Wrap module.func_name using wrapper_factory(original) -> wrapped. No-ops if missing."""
    try:
        original = getattr(module, func_name, None)
        if not callable(original):
            return False
        # Avoid double-wrapping
        if getattr(original, "_sm_v800_wrapped", False):
            return True
        wrapped = wrapper_factory(original)
        try:
            setattr(wrapped, "_sm_v800_wrapped", True)
        except Exception:
            pass
        setattr(module, func_name, wrapped)
        return True
    except Exception:
        return False


def _patch_sarahmemory_voice() -> None:
    """Patch SarahMemoryVoice in a minimal, safe way."""
    try:
        import SarahMemoryVoice  # core module
    except Exception as e:
        logger.warning("[v800 voice] Could not import SarahMemoryVoice: %s", e)
        return

    # Apply now (covers engine already initialized)
    try:
        _apply_engine_tuning(SarahMemoryVoice)
    except Exception:
        pass

    # Wrap synthesize_voice so tuning re-applies each call (covers engine re-init).
    def factory(original):
        def synthesize_voice_patched(text: str, emotion: str = None, engine_pref: str = None):
            try:
                _apply_engine_tuning(SarahMemoryVoice)
            except Exception:
                pass
            try:
                return original(text, emotion=emotion, engine_pref=engine_pref)
            except TypeError:
                # If signature differs, attempt best-effort
                return original(text)
        return synthesize_voice_patched

    ok = _safe_wrap(SarahMemoryVoice, "synthesize_voice", factory)
    if ok:
        logger.info("[v800 voice] Patched SarahMemoryVoice.synthesize_voice (tuning + female voice preference).")


def _patch_microphone_init_failsoft() -> None:
    """If SarahMemoryDL has mic init helpers, wrap them to fail-soft on headless / missing deps."""
    try:
        import SarahMemoryDL  # likely contains mic/audio init paths (based on patch stub)
    except Exception:
        return

    def factory(original):
        def wrapped(*a, **kw):
            try:
                return original(*a, **kw)
            except Exception as e:
                # Fail-soft: don't crash whole program because mic can't init
                logger.warning("[v800 voice] Microphone init failure (suppressed): %s", e)
                logger.debug(traceback.format_exc())
                return None
        return wrapped

    # We do not assume exact function names. We wrap only if they exist.
    for fn in ("init_microphone", "initialize_microphone", "mic_init", "setup_microphone", "start_microphone"):
        if _safe_wrap(SarahMemoryDL, fn, factory):
            logger.info("[v800 voice] Patched SarahMemoryDL.%s (fail-soft).", fn)


def _reduce_comtypes_noise_best_effort() -> None:
    """Best-effort reduce comtypes debug spam if SarahMemoryVoice uses it; never crash if unavailable."""
    # This does NOT “fix” comtypes internals; it only reduces log noise in common setups.
    try:
        import logging as _logging
        _logging.getLogger("comtypes").setLevel(_logging.WARNING)
        _logging.getLogger("comtypes.client").setLevel(_logging.WARNING)
        _logging.getLogger("comtypes._comobject").setLevel(_logging.WARNING)
        _logging.getLogger("comtypes._vtbl").setLevel(_logging.WARNING)
    except Exception:
        pass


def _patch_sobje_failsoft() -> None:
    """If SarahMemorySOBJE touches voice settings, keep it fail-soft (no assumptions)."""
    try:
        import SarahMemorySOBJE
    except Exception:
        return

    def factory(original):
        def wrapped(*a, **kw):
            try:
                return original(*a, **kw)
            except Exception as e:
                logger.warning("[v800 voice] SarahMemorySOBJE error (suppressed): %s", e)
                logger.debug(traceback.format_exc())
                return None
        return wrapped

    # Wrap only if present
    for fn in ("apply_voice_settings", "set_voice_profile", "init_voice", "initialize_voice"):
        if _safe_wrap(SarahMemorySOBJE, fn, factory):
            logger.info("[v800 voice] Patched SarahMemorySOBJE.%s (fail-soft).", fn)


# =============================================================================
# Main entry
# =============================================================================

def apply() -> bool:
    """Apply the monkey patch (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        _reduce_comtypes_noise_best_effort()
        _patch_sarahmemory_voice()
        _patch_microphone_init_failsoft()
        _patch_sobje_failsoft()

        _APPLIED = True
        return True

    except Exception as e:
        logger.error("Patch apply failed: %s", e)
        logger.debug(traceback.format_exc())
        return False


# Self-apply on import (fallback)
try:
    apply()
except Exception:
    pass
