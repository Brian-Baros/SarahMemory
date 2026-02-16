# --==The SarahMemory Project==--
# File: SarahMemoryVoice.py
# Part of the SarahMemory Companion AI-bot Platform
# Version: v8.0.0
# Date: 2026-01-16
# Author: © 2025-2026 Brian Lee Baros. All Rights Reserved.
#
# SarahMemory v8.0 - Voice & Sound Synthesis Module
#
# CORE GOALS
# - One authoritative backend voice pipeline (no FE speech injection required).
# - Avoid mid-sentence cutoffs by serializing TTS requests through a single worker.
# - Remain headless-safe on cloud servers (PythonAnywhere) and Windows-safe locally.
# - Preserve existing public APIs (synthesize_voice, transcribe_once, etc.) while
#   adding a robust queue-based speak_text().
#
# NOTES
# - WebUI speech should call backend endpoints (e.g., /api/ui/event) which in turn
#   call speak_text() in this file. This eliminates the need for FE_v800_app_speech.py
#   and any speech.js concept.

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import queue
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# GLOBAL CONFIG (safe import)
# =============================================================================
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    # Allow module import in isolated tests
    class _Cfg:  # pragma: no cover
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(os.getcwd(), "data")
        SETTINGS_DIR = os.path.join(os.getcwd(), "data", "settings")
        DOWNLOADS_DIR = os.path.join(os.getcwd(), "data", "downloads")
        AVATAR_IS_SPEAKING = False

    config = _Cfg()  # type: ignore

# Optional helpers from globals (safe if missing)
try:
    from SarahMemoryGlobals import load_user_settings, SAFE_MODE, LOCAL_ONLY_MODE  # type: ignore
except Exception:
    load_user_settings = None  # type: ignore
    SAFE_MODE = False  # type: ignore
    LOCAL_ONLY_MODE = False  # type: ignore

# Load settings early if available
if callable(load_user_settings):
    try:
        load_user_settings()
    except Exception:
        pass

# Mirror AVATAR_IS_SPEAKING on config
if not hasattr(config, "AVATAR_IS_SPEAKING"):
    config.AVATAR_IS_SPEAKING = False

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("SarahMemoryVoice")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)

# Reduce common Windows comtypes spam if present (no hard dependency)
try:
    logging.getLogger("comtypes").setLevel(logging.WARNING)
    logging.getLogger("comtypes.client").setLevel(logging.WARNING)
    logging.getLogger("comtypes._comobject").setLevel(logging.WARNING)
    logging.getLogger("comtypes._vtbl").setLevel(logging.WARNING)
except Exception:
    pass

# =============================================================================
# OPTIONAL LIBS
# =============================================================================
# SpeechRecognition / microphone
try:  # pragma: no cover
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore

recognizer = sr.Recognizer() if sr is not None else None
if recognizer is not None:
    recognizer.dynamic_energy_threshold = True

# pyttsx3 TTS (Primary)
try:  # pragma: no cover
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore

# gTTS (Alternative TTS)
try:
    from gtts import gTTS  # type: ignore
    _HAS_GTTS = True
except Exception:
    gTTS = None  # type: ignore
    _HAS_GTTS = False

# edge-tts (Optional)
try:
    import edge_tts  # type: ignore
    _HAS_EDGE_TTS = True
except Exception:
    edge_tts = None  # type: ignore
    _HAS_EDGE_TTS = False

# Optional audio playback fallback
try:
    import pygame  # type: ignore
    _HAS_PYGAME = True
except Exception:
    pygame = None  # type: ignore
    _HAS_PYGAME = False

# =============================================================================
# PATH HELPERS (consistent BASE/DATA directories)
# =============================================================================
def _base_dir() -> Path:
    bd = getattr(config, "BASE_DIR", None)
    if bd:
        return Path(bd)
    return Path(os.getcwd())


def _data_dir() -> Path:
    dd = getattr(config, "DATA_DIR", None)
    if dd:
        return Path(dd)
    return _base_dir() / "data"


def _settings_dir() -> Path:
    sd = getattr(config, "SETTINGS_DIR", None)
    if sd:
        return Path(sd)
    return _data_dir() / "settings"


def _downloads_dir() -> Path:
    dld = getattr(config, "DOWNLOADS_DIR", None)
    if dld:
        return Path(dld)
    return _data_dir() / "downloads"


def _ensure_dirs() -> None:
    try:
        _data_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _settings_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _downloads_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


_ensure_dirs()

# =============================================================================
# SETTINGS / PROFILES
# =============================================================================
VOICE_PROFILES: Dict[str, str] = {
    "Default": "female",
    "Female": "female",
    "Male": "male",
}

# Soft voice preference keywords (from v800 monkey patch; now core)
PREFERRED_KEYWORDS: Tuple[str, ...] = (
    "zira",
    "aria",
    "jenny",
    "eva",
    "emma",
    "hazel",
    "susan",
    "natasha",
    "female",
)

# Emotion-based prosody defaults
EMOTION_PROSODY: Dict[str, Dict[str, float]] = {
    "joy": {"rate_delta": +12, "pitch_delta": +0.15, "volume": 1.0},
    "excitement": {"rate_delta": +18, "pitch_delta": +0.2, "volume": 1.0},
    "trust": {"rate_delta": +6, "pitch_delta": +0.05, "volume": 1.0},
    "surprise": {"rate_delta": +16, "pitch_delta": +0.25, "volume": 1.0},
    "sadness": {"rate_delta": -14, "pitch_delta": -0.1, "volume": 0.9},
    "fear": {"rate_delta": -6, "pitch_delta": +0.1, "volume": 0.95},
    "anger": {"rate_delta": +10, "pitch_delta": +0.05, "volume": 1.0},
    "calm": {"rate_delta": -8, "pitch_delta": -0.05, "volume": 0.95},
    "neutral": {"rate_delta": 0, "pitch_delta": 0.0, "volume": 1.0},
}

custom_audio_settings: Dict[str, float] = {
    "pitch": 1.0,
    "bass": 1.0,
    "treble": 1.0,
    "reverb": 0.0,
    "echo": 0.0,
    "volume_boost": 1.0,
}

current_settings: Dict[str, Any] = {
    "speech_rate": "Normal",  # Slow/Normal/Fast
    "voice_profile": "Female",
    "emotion": "neutral",
    "tts_engine": "pyttsx3",  # pyttsx3|gtts|edge
    "language": "en",
}

active_voice_profile: str = "Female"

# =============================================================================
# DATABASE LOGGING
# =============================================================================
def log_voice_event(event: str, details: str) -> None:
    """
    Log voice-related events to system_logs.db. Never crashes the caller.
    """
    try:
        db_path = _data_dir() / "memory" / "datasets" / "system_logs.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event TEXT,
                    details TEXT,
                    engine TEXT,
                    emotion TEXT
                )
                """
            )
            ts = datetime.utcnow().isoformat() + "Z"
            cur.execute(
                "INSERT INTO voice_recognition_events (timestamp, event, details, engine, emotion) VALUES (?, ?, ?, ?, ?)",
                (
                    ts,
                    event,
                    details,
                    str(current_settings.get("tts_engine", "pyttsx3")),
                    str(current_settings.get("emotion", "neutral")),
                ),
            )
            conn.commit()
    except Exception:
        # never crash the caller
        pass

# =============================================================================
# TTS ENGINE (pyttsx3) STATE
# =============================================================================
_engine = None
_engine_voices: List[Any] = []
_engine_ready = False

# STOP flag used to interrupt current speech (prevents overlap/cutoffs)
_TTS_STOP_FLAG = threading.Event()

# Single worker thread for ALL TTS output (fixes mid-sentence cutoffs)
_TTS_QUEUE: "queue.Queue[_TTSTask]" = queue.Queue()
_TTS_WORKER_STARTED = False
_TTS_WORKER_LOCK = threading.Lock()

# =============================================================================
# TEXT SPLITTING (prevents very long utterances from choking engines)
# =============================================================================
_SENTENCE_RE = re.compile(r"(?<=[\.\!\?])\s+")
_WS_RE = re.compile(r"\s+")

def _split_text_for_tts(text: str, max_chars: int = 350) -> List[str]:
    """
    Split into sentence-ish chunks, capped by max_chars.
    Prevents pyttsx3/voices from choking on very long strings.
    """
    s = _WS_RE.sub(" ", (text or "").strip())
    if not s:
        return []

    parts = _SENTENCE_RE.split(s)
    out: List[str] = []
    buf: List[str] = []
    ln = 0

    def flush():
        nonlocal buf, ln
        if buf:
            out.append(" ".join(buf).strip())
            buf = []
            ln = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if ln + len(p) + 1 <= max_chars:
            buf.append(p)
            ln += len(p) + 1
        else:
            flush()
            if len(p) <= max_chars:
                buf.append(p)
                ln = len(p)
            else:
                # hard split long sentence
                start = 0
                while start < len(p):
                    out.append(p[start : start + max_chars].strip())
                    start += max_chars
    flush()
    return [x for x in out if x]

# =============================================================================
# PYTTSX3 HELPERS
# =============================================================================
def _headless_safe() -> bool:
    """
    Returns True if we should avoid attempting local audio playback/pyttsx3
    (cloud/headless modes). This is conservative: we still allow Windows/local.
    """
    if bool(getattr(config, "SAFE_MODE", False)):
        return True
    if platform.system().lower() != "windows":
        # on Linux, treat missing DISPLAY as a hint for headless
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False


def _ensure_pyttsx3_engine() -> bool:
    """
    Ensure pyttsx3 engine is initialized. Returns False if unavailable/disabled.
    """
    global _engine_ready, _engine, _engine_voices

    if _headless_safe():
        return False
    if pyttsx3 is None:
        return False
    if _engine_ready and _engine is not None:
        return True

    try:
        _engine = pyttsx3.init()
        try:
            _engine_voices = _engine.getProperty("voices") or []
        except Exception:
            _engine_voices = []
        _engine_ready = True
        return True
    except Exception as e:
        logger.warning("[Voice] pyttsx3 init failed: %s", e)
        _engine_ready = False
        _engine = None
        _engine_voices = []
        return False


def _pick_preferred_voice(profile: str) -> Optional[str]:
    """
    Prefer female-ish voices unless a male profile requested.
    Returns selected voice id (or None).
    """
    if not _engine_voices:
        return None

    want = (VOICE_PROFILES.get(profile) or profile or "female").lower()
    voices = _engine_voices

    # Pass 1: keyword match for preferred female voices
    if want != "male":
        for v in voices:
            try:
                name = (getattr(v, "name", "") or "").lower()
                vid = (getattr(v, "id", "") or "").lower()
                if any(k in name or k in vid for k in PREFERRED_KEYWORDS):
                    return getattr(v, "id", None)
            except Exception:
                continue

        # Pass 2: avoid explicit male if possible
        for v in voices:
            try:
                name = (getattr(v, "name", "") or "").lower()
                if "male" not in name:
                    return getattr(v, "id", None)
            except Exception:
                continue

    # Male requested: look for 'male' keyword
    if want == "male":
        for v in voices:
            try:
                name = (getattr(v, "name", "") or "").lower()
                if "male" in name or "david" in name or "guy" in name:
                    return getattr(v, "id", None)
            except Exception:
                continue

    # fallback: first voice
    try:
        return getattr(voices[0], "id", None)
    except Exception:
        return None


def _rate_value() -> int:
    label = str(current_settings.get("speech_rate", "Normal"))
    if label == "Slow":
        return 145
    if label == "Fast":
        return 220
    return 175


def _apply_engine_tuning(profile: str, emotion: str) -> None:
    """
    Apply voice selection + rate/volume based on settings and emotion.
    """
    if _engine is None:
        return
    # voice selection
    vid = _pick_preferred_voice(profile)
    if vid:
        try:
            _engine.setProperty("voice", vid)
        except Exception:
            pass

    # rate / volume
    pros = EMOTION_PROSODY.get(emotion or "neutral", EMOTION_PROSODY["neutral"])
    base_rate = _rate_value()
    try:
        _engine.setProperty("rate", int(base_rate + int(pros.get("rate_delta", 0))))
    except Exception:
        pass

    vol = float(pros.get("volume", 1.0))
    try:
        _engine.setProperty("volume", max(0.0, min(1.0, vol)))
    except Exception:
        pass


def _play_audio_file(filepath: str) -> None:
    """
    Best-effort audio playback for gTTS/edge outputs.
    In headless environments this should fail-soft.
    """
    if _headless_safe():
        return

    sys = platform.system().lower()
    try:
        if sys == "windows":
            try:
                import winsound  # type: ignore
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
                return
            except Exception:
                pass
        if sys == "darwin":
            try:
                subprocess.run(["afplay", filepath], check=False)
                return
            except Exception:
                pass
        # Linux (try aplay/paplay)
        try:
            subprocess.run(["aplay", filepath], check=False)
            return
        except Exception:
            pass
        try:
            subprocess.run(["paplay", filepath], check=False)
            return
        except Exception:
            pass

        # pygame fallback
        if _HAS_PYGAME and pygame is not None:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception:
                pass
    except Exception:
        pass


def _speak_with_pyttsx3(text: str, profile: str, emotion: str) -> None:
    if not _ensure_pyttsx3_engine():
        return

    # If a stop was requested, honor it before starting.
    if _TTS_STOP_FLAG.is_set():
        return

    _apply_engine_tuning(profile, emotion)

    chunks = _split_text_for_tts(text)
    if not chunks:
        return

    # IMPORTANT:
    # - Queue all chunks, then runAndWait once.
    # - This avoids mid-sentence cutoffs that happen when repeated runAndWait
    #   calls overlap or engine is re-entered.
    try:
        for c in chunks:
            if _TTS_STOP_FLAG.is_set():
                try:
                    _engine.stop()  # type: ignore
                except Exception:
                    pass
                return
            _engine.say(c)  # type: ignore

        _engine.runAndWait()  # type: ignore
    except Exception as e:
        # If engine got into a bad state, reset it once.
        logger.warning("[Voice] pyttsx3 speak failed, resetting engine: %s", e)
        try:
            try:
                _engine.stop()  # type: ignore
            except Exception:
                pass
        finally:
            # full reset
            _reset_pyttsx3_engine()


def _reset_pyttsx3_engine() -> None:
    """Force a clean re-init of pyttsx3 on next use (fixes stuck/cutoff engines)."""
    global _engine_ready, _engine, _engine_voices
    _engine_ready = False
    _engine = None
    _engine_voices = []


def _speak_with_gtts(text: str, lang: str = "en") -> None:
    if not _HAS_GTTS or gTTS is None:
        return

    # Headless OK: this generates an mp3 and attempts playback; on servers it may no-op.
    tmp = _downloads_dir() / f"sm_tts_{uuid.uuid4().hex}.mp3"
    try:
        tts = gTTS(text=text, lang=lang, slow=(str(current_settings.get("speech_rate")) == "Slow"))
        tts.save(str(tmp))
        _play_audio_file(str(tmp))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


async def _edge_tts_async(text: str, voice: str, rate: str, out_path: Path) -> None:
    if not _HAS_EDGE_TTS or edge_tts is None:
        return
    comm = edge_tts.Communicate(text, voice, rate=rate)
    await comm.save(str(out_path))


def _speak_with_edge_tts(text: str, profile: str) -> None:
    if not _HAS_EDGE_TTS or edge_tts is None:
        return
    if _headless_safe():
        return

    # Voice selection
    voice = "en-US-AriaNeural"
    if (VOICE_PROFILES.get(profile) or "").lower() == "male":
        voice = "en-US-GuyNeural"

    # Rate label
    rate = "+0%"
    sr_label = str(current_settings.get("speech_rate", "Normal"))
    if sr_label == "Slow":
        rate = "-25%"
    elif sr_label == "Fast":
        rate = "+25%"

    out = _downloads_dir() / f"sm_edge_{uuid.uuid4().hex}.mp3"
    try:
        import asyncio
        asyncio.run(_edge_tts_async(text, voice, rate, out))
        _play_audio_file(str(out))
    finally:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass

# =============================================================================
# TTS QUEUE / WORKER (this is the main cutoff fix)
# =============================================================================
@dataclass
class _TTSTask:
    text: str
    blocking_event: Optional[threading.Event]
    emotion: str
    engine_pref: str
    voice_profile: str
    lang: str


def _start_tts_worker() -> None:
    global _TTS_WORKER_STARTED
    with _TTS_WORKER_LOCK:
        if _TTS_WORKER_STARTED:
            return

        t = threading.Thread(target=_tts_worker_loop, name="SarahMemoryVoiceTTS", daemon=True)
        t.start()
        _TTS_WORKER_STARTED = True


def _tts_worker_loop() -> None:
    while True:
        task: _TTSTask = _TTS_QUEUE.get()
        try:
            if not task or not task.text.strip():
                continue

            # Mark speaking (used by avatar/UI)
            config.AVATAR_IS_SPEAKING = True
            _TTS_STOP_FLAG.clear()

            engine = (task.engine_pref or "pyttsx3").strip().lower()
            emotion = (task.emotion or "neutral").strip().lower()
            profile = (task.voice_profile or "Female").strip()

            try:
                if engine == "edge" and _HAS_EDGE_TTS:
                    _speak_with_edge_tts(task.text, profile)
                elif engine == "gtts" and _HAS_GTTS:
                    _speak_with_gtts(task.text, lang=task.lang or "en")
                else:
                    # default: pyttsx3 if available
                    _speak_with_pyttsx3(task.text, profile, emotion)
            except Exception as e:
                logger.warning("[Voice] TTS task failed: %s", e)
                logger.debug(traceback.format_exc())
            finally:
                # Always clear speaking flag
                config.AVATAR_IS_SPEAKING = False

                # Signal blocking callers
                if task.blocking_event is not None:
                    try:
                        task.blocking_event.set()
                    except Exception:
                        pass

                # Log
                try:
                    log_voice_event("TTS", f"engine={engine} emotion={emotion} text={task.text[:120]}")
                except Exception:
                    pass

        finally:
            try:
                _TTS_QUEUE.task_done()
            except Exception:
                pass

# =============================================================================
# PUBLIC API: SPEAK / STOP
# =============================================================================
def stop_speaking() -> None:
    """
    Immediately attempt to stop speech playback.
    Safe to call from anywhere.
    """
    _TTS_STOP_FLAG.set()
    try:
        if _engine is not None:
            _engine.stop()  # type: ignore
    except Exception:
        pass


def speak_text(text: str, blocking: bool = True, emotion: Optional[str] = None, engine_pref: Optional[str] = None) -> bool:
    """
    Primary backend TTS entrypoint.

    FIXES MID-SENTENCE CUTOFFS by:
      - sending ALL TTS through a single worker thread
      - calling pyttsx3.runAndWait() once per task after queuing chunks

    Args:
        text: text to speak
        blocking: if True, wait until speech completes
        emotion: optional emotion label (neutral/joy/etc)
        engine_pref: 'pyttsx3'|'gtts'|'edge'

    Returns:
        True if accepted for speech, False if skipped (e.g., SAFE_MODE/headless/no text).
    """
    if not text or not str(text).strip():
        return False

    # In SAFE_MODE we still allow silent operation (skip TTS)
    if SAFE_MODE:
        return False

    _start_tts_worker()

    ev: Optional[threading.Event] = threading.Event() if blocking else None
    task = _TTSTask(
        text=str(text).strip(),
        blocking_event=ev,
        emotion=(emotion or str(current_settings.get("emotion", "neutral"))),
        engine_pref=(engine_pref or str(current_settings.get("tts_engine", "pyttsx3"))),
        voice_profile=str(current_settings.get("voice_profile", active_voice_profile or "Female")),
        lang=str(current_settings.get("language", "en") or "en"),
    )
    _TTS_QUEUE.put(task)

    if ev is not None:
        ev.wait()
    return True


# Backwards compatible alias
def synthesize_voice(text: str, emotion: str = None, engine_pref: str = None) -> None:
    """
    Backwards compatible wrapper for legacy callers.
    """
    speak_text(text, blocking=True, emotion=emotion, engine_pref=engine_pref)


# Convenience wrapper used by some web bridges
def speak_text_async(text: str, emotion: Optional[str] = None, engine_pref: Optional[str] = None) -> bool:
    return speak_text(text, blocking=False, emotion=emotion, engine_pref=engine_pref)

# =============================================================================
# VOICE SETTINGS MANAGEMENT
# =============================================================================
def save_voice_settings() -> None:
    """
    Persist the current voice configuration into settings.json.
    """
    try:
        settings_path = _settings_dir() / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {}
        if settings_path.exists():
            try:
                data = json.loads(settings_path.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}

        data["voice_profile"] = str(current_settings.get("voice_profile", active_voice_profile))
        data["pitch"] = float(custom_audio_settings.get("pitch", 1.0))
        data["bass"] = float(custom_audio_settings.get("bass", 1.0))
        data["treble"] = float(custom_audio_settings.get("treble", 1.0))
        data["reverb"] = float(custom_audio_settings.get("reverb", 0.0))
        data["speech_rate"] = str(current_settings.get("speech_rate", "Normal"))
        data["emotion"] = str(current_settings.get("emotion", "neutral"))
        data["tts_engine"] = str(current_settings.get("tts_engine", "pyttsx3"))
        data["language"] = str(current_settings.get("language", "en"))

        settings_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    except Exception as e:
        logger.warning("[Voice] Failed to save voice settings: %s", e)


def load_voice_settings() -> None:
    """
    Load voice configuration from settings.json.
    """
    try:
        settings_path = _settings_dir() / "settings.json"
        if not settings_path.exists():
            return
        data = json.loads(settings_path.read_text(encoding="utf-8")) or {}

        if "voice_profile" in data:
            set_voice_profile(str(data["voice_profile"]))
        if "pitch" in data:
            set_pitch(float(data["pitch"]))
        if "bass" in data:
            set_bass(float(data["bass"]))
        if "treble" in data:
            set_treble(float(data["treble"]))
        if "reverb" in data:
            set_reverb(float(data["reverb"]))
        if "speech_rate" in data:
            set_speech_rate(str(data["speech_rate"]))
        if "emotion" in data:
            set_emotion(str(data["emotion"]))
        if "tts_engine" in data:
            set_tts_engine(str(data["tts_engine"]))
        if "language" in data:
            current_settings["language"] = str(data["language"])
    except Exception as e:
        logger.warning("[Voice] Failed to load voice settings: %s", e)

# =============================================================================
# SETTERS / GETTERS (WebUI + core)
# =============================================================================
def get_voice_profiles() -> List[str]:
    out = list(VOICE_PROFILES.keys())
    # Include system voices if pyttsx3 is available
    if _ensure_pyttsx3_engine():
        try:
            for v in _engine_voices:
                nm = getattr(v, "name", None)
                if nm and nm not in out:
                    out.append(nm)
        except Exception:
            pass
    return out


def set_voice_profile(profile_name: str) -> None:
    global active_voice_profile
    if not profile_name:
        return
    active_voice_profile = profile_name
    current_settings["voice_profile"] = profile_name


def set_pitch(value: float) -> None:
    custom_audio_settings["pitch"] = float(value)


def set_bass(value: float) -> None:
    custom_audio_settings["bass"] = float(value)


def set_treble(value: float) -> None:
    custom_audio_settings["treble"] = float(value)


def set_reverb(value: float) -> None:
    custom_audio_settings["reverb"] = float(value)


def set_emotion(emotion: str) -> None:
    if not emotion:
        return
    current_settings["emotion"] = emotion


def set_tts_engine(engine_name: str) -> None:
    if not engine_name:
        return
    current_settings["tts_engine"] = engine_name


def set_speech_rate(rate_label: str) -> None:
    if rate_label not in ("Slow", "Normal", "Fast"):
        rate_label = "Normal"
    current_settings["speech_rate"] = rate_label


def list_voices() -> List[Dict[str, Any]]:
    if not _ensure_pyttsx3_engine():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for v in _engine_voices:
            out.append(
                {
                    "id": getattr(v, "id", ""),
                    "name": getattr(v, "name", ""),
                    "lang": (getattr(v, "languages", [""])[0] if hasattr(v, "languages") else ""),
                }
            )
    except Exception:
        pass
    return out


def configure_voice(opts: Dict[str, Any]) -> None:
    """
    Apply a bundle of voice settings from a dict, used by the Web UI bridge.
    """
    try:
        profile = opts.get("profile") or opts.get("voice_profile")
        if profile:
            set_voice_profile(str(profile))

        rate = opts.get("rate") or opts.get("speech_rate")
        if isinstance(rate, str):
            set_speech_rate(rate)
        elif isinstance(rate, (int, float)):
            if float(rate) < 150:
                set_speech_rate("Slow")
            elif float(rate) > 210:
                set_speech_rate("Fast")
            else:
                set_speech_rate("Normal")

        if "pitch" in opts:
            set_pitch(float(opts["pitch"]))
        if "bass" in opts:
            set_bass(float(opts["bass"]))
        if "treble" in opts:
            set_treble(float(opts["treble"]))
        if "reverb" in opts:
            set_reverb(float(opts["reverb"]))
        if "emotion" in opts:
            set_emotion(str(opts["emotion"]))
        if "tts_engine" in opts:
            set_tts_engine(str(opts["tts_engine"]))
        if "language" in opts:
            current_settings["language"] = str(opts["language"])

        save_voice_settings()
    except Exception as e:
        logger.warning("[Voice] configure_voice failed: %s", e)

# =============================================================================
# MICROPHONE / RECOGNITION
# =============================================================================
mic = None

def initialize_microphone():
    """
    Initialize and cache a Microphone object, or None if unavailable.
    Fail-soft by design (cloud/headless safety).
    """
    global mic
    if mic is not None:
        return mic
    if SAFE_MODE or LOCAL_ONLY_MODE:
        return None
    if sr is None or recognizer is None:
        return None
    try:
        mic = sr.Microphone()
        log_voice_event("Microphone Initialized", "Microphone object created successfully.")
        return mic
    except Exception as e:
        log_voice_event("Microphone Initialization Error", f"Error: {e}")
        mic = None
        return None


def _recognize_chunk(audio: "sr.AudioData") -> Optional[str]:
    if sr is None or recognizer is None:
        return None
    try:
        txt = recognizer.recognize_google(audio)
        return (txt or "").strip()
    except Exception:
        return None


def listen_and_process(timeout: Optional[float] = None, phrase_time_limit: Optional[float] = None) -> Optional[str]:
    if SAFE_MODE or LOCAL_ONLY_MODE:
        return None
    mic_obj = initialize_microphone()
    if mic_obj is None or sr is None or recognizer is None:
        return None

    timeout = float(timeout if timeout is not None else 5.0)
    phrase_time_limit = float(phrase_time_limit if phrase_time_limit is not None else 10.0)

    try:
        with mic_obj as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
            except Exception:
                pass
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        txt = _recognize_chunk(audio)
        if txt:
            log_voice_event("Voice Input Recognized", txt)
        return txt
    except Exception:
        return None


def transcribe_once(timeout: float = 10.0) -> str:
    txt = listen_and_process(timeout=timeout)
    return txt or ""

# =============================================================================
# TTS SHUTDOWN
# =============================================================================
def shutdown_tts() -> None:
    """
    Shut down the TTS engine safely.
    """
    stop_speaking()
    try:
        if _engine is not None:
            _engine.stop()  # type: ignore
    except Exception:
        pass
    _reset_pyttsx3_engine()

# =============================================================================
# MODULE INIT
# =============================================================================
try:
    load_voice_settings()
except Exception:
    pass

# =============================================================================
# MODULE SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("SARAHMEMORY v8.0 - VOICE MODULE SELF-TEST")
    print("=" * 78)
    print("TTS engine available:", pyttsx3 is not None and not _headless_safe())
    print("gTTS available:", _HAS_GTTS)
    print("edge-tts available:", _HAS_EDGE_TTS)
    print("Profiles:", get_voice_profiles()[:10])
    print("Speaking a test sentence...")
    speak_text("Hello. This is SarahMemory voice. The cutoff bug should be fixed.", blocking=True)
    print("Done.")

# =============================================================================
# END OF SarahMemoryVoice.py v8.0.0
# =============================================================================
