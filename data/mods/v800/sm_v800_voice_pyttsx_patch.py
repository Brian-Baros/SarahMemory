"""
sm_v800_voice_pyttsx_patch.py
MonkeyPatch for SarahMemory v8.0.0 (NO core edits)

Objective:
- Force a female / less-robotic pyttsx3 voice when pyttsx3 is the active engine.

Placement:
- ../data/mods/v800/sm_v800_voice_pyttsx_patch.py

Notes:
- This patch expects a mod loader to import it OR call apply().
- It also self-applies on import as a fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("SarahMemoryMonkeyPatch.v800.voice_pyttsx")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# Prioritized keywords for common Windows female voices (best-effort).
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

# “De-robot” baseline tuning (leave emotion logic intact; just improve the default).
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
        logger.info("[v800 voice patch] No pyttsx3 engine present; skipping engine tuning.")
        return

    selected = _pick_female_voice(engine)
    if selected:
        logger.info(f"[v800 voice patch] Selected TTS voice: {selected}")
    else:
        logger.info("[v800 voice patch] No preferred female voice found; leaving default voice.")

    try:
        engine.setProperty("rate", SOFT_RATE)
    except Exception:
        pass

    try:
        engine.setProperty("volume", SOFT_VOLUME)
    except Exception:
        pass


def apply() -> bool:
    """
    Mod-loader entry point.
    Returns True if patch applied, False otherwise.
    """
    try:
        import SarahMemoryVoice  # core module
    except Exception as e:
        logger.error(f"[v800 voice patch] Could not import SarahMemoryVoice: {e}")
        return False

    # Apply now (in case engine already exists)
    _apply_engine_tuning(SarahMemoryVoice)

    # Wrap synthesize_voice so tuning re-applies each call (covers engine re-init).
    original = getattr(SarahMemoryVoice, "synthesize_voice", None)
    if not callable(original):
        logger.error("[v800 voice patch] SarahMemoryVoice.synthesize_voice not callable; aborting.")
        return False

    def synthesize_voice_patched(text: str, emotion: str = None, engine_pref: str = None):
        try:
            # Only meaningful when pyttsx3 is in play; harmless otherwise.
            _apply_engine_tuning(SarahMemoryVoice)
        except Exception:
            pass
        return original(text, emotion=emotion, engine_pref=engine_pref)

    SarahMemoryVoice.synthesize_voice = synthesize_voice_patched  # type: ignore[attr-defined]
    logger.info("[v800 voice patch] Patched SarahMemoryVoice.synthesize_voice (female voice enforcement).")
    return True


# Fallback: auto-apply if imported without an explicit apply() call
try:
    _AUTO_APPLIED = apply()
except Exception:
    _AUTO_APPLIED = False
