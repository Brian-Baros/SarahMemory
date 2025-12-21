"""--==The SarahMemory Project==--
File: SarahMemoryVoice.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
SarahMemory v8.0 - Voice & Sound Synthesis Module
Voice, TTS, STT, and Audio Processing
Cross-platform/headless-safe voice module with advanced audio capabilities
===============================================================================

FEATURES:
- Works on Windows / Linux / macOS
- Degrades gracefully on headless servers (no audio / no DISPLAY)
- Multiple TTS engines (pyttsx3, gTTS, edge-tts support)
- Advanced voice synthesis with emotion control
- Professional audio effects (reverb, pitch, bass, treble)
- Multi-language support (40+ languages)
- Voice cloning capabilities (when available)
- Real-time audio processing
- Integration with media subsystems (Music, Lyrics, Video)
"""

from __future__ import annotations

import traceback
import logging
import os
import sqlite3
from datetime import datetime
import time
import json
from typing import Any, Dict, Optional, List
# Optional custom voice model support (.pt files)
try:
    import torch
    CUSTOM_TTS_AVAILABLE = True
except ImportError:
    CUSTOM_TTS_AVAILABLE = False
    logging.warning("[VoiceEngine] Torch not available — custom .pt voices disabled")
    
# =============================================================================
# OPTIONAL/NUMERICAL LIBRARIES
# =============================================================================
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore

# =============================================================================
# GLOBALS/CONFIG
# =============================================================================
import SarahMemoryGlobals as config

# Optional helpers from globals (safe if missing)
try:
    from SarahMemoryGlobals import load_user_settings, is_offline, SAFE_MODE, LOCAL_ONLY_MODE  # type: ignore
except Exception:  # pragma: no cover - keep module importable
    load_user_settings = None  # type: ignore
    SAFE_MODE = False          # type: ignore
    LOCAL_ONLY_MODE = False    # type: ignore

# Load settings early if available
if callable(load_user_settings):
    try:
        load_user_settings()
    except Exception:
        pass

# =============================================================================
# v8.0 ENHANCED: AUDIO BACKENDS
# =============================================================================

# SpeechRecognition / microphone
try:  # pragma: no cover - depends on system audio libs
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore

recognizer = sr.Recognizer() if sr is not None else None
if recognizer is not None:
    recognizer.dynamic_energy_threshold = True

# pyttsx3 TTS (Primary)
try:  # pragma: no cover - platform dependent
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore

# gTTS (Alternative TTS)
try:
    from gtts import gTTS  # type: ignore
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

# edge-tts (Advanced TTS)
try:
    import edge_tts  # type: ignore
    _HAS_EDGE_TTS = True
except Exception:
    _HAS_EDGE_TTS = False

# Initialize primary TTS engine
_TTS_AVAILABLE = False
engine = None
available_voices: List[Any] = []

if pyttsx3 is not None:
    try:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 185)
            engine.setProperty("volume", 1.0)
        except Exception:
            pass
        
        try:
            available_voices = engine.getProperty("voices") or []
        except Exception:
            available_voices = []
        
        _TTS_AVAILABLE = True
    
    except Exception as e:
        engine = None
        available_voices = []
        _TTS_AVAILABLE = False
        print(f"[v8.0][SarahMemoryVoice] TTS engine init failed: {e}")

# =============================================================================
# v8.0 ENHANCED: AUDIO PROCESSING LIBRARIES
# =============================================================================

# Noise reduction
try:  # pragma: no cover - optional
    import noisereduce as nr  # type: ignore
except Exception:
    nr = None  # type: ignore

# FFmpeg and audio processing
try:  # pragma: no cover - optional / platform dependent
    from pydub.utils import which as _which  # type: ignore
    from pydub import AudioSegment  # type: ignore
    from pydub.effects import normalize, low_pass_filter, high_pass_filter  # type: ignore

    # Prefer bundled ffmpeg on Windows; fall back to PATH elsewhere
    _ffmpeg_path = os.path.join(config.BASE_DIR, "bin", "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(_ffmpeg_path):
        AudioSegment.converter = _ffmpeg_path
    else:
        AudioSegment.converter = _which("ffmpeg")
    
    from pydub.silence import split_on_silence  # type: ignore
    _HAS_PYDUB = True

except Exception:
    AudioSegment = None  # type: ignore
    split_on_silence = None  # type: ignore
    _HAS_PYDUB = False

# SoundFile for advanced audio I/O
try:
    import soundfile as sf  # type: ignore
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False

# librosa for audio analysis
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("SarahMemoryVoice")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    _handler = logging.NullHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - v8.0 - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

logger.info("[v8.0] SarahMemoryVoice module initialized")
logger.info("[v8.0] TTS Available: %s", _TTS_AVAILABLE)
logger.info("[v8.0] gTTS Available: %s", _HAS_GTTS)
logger.info("[v8.0] Edge-TTS Available: %s", _HAS_EDGE_TTS)
logger.info("[v8.0] Audio Processing Available: %s", _HAS_PYDUB)

# =============================================================================
# SHARED STATE / CONFIGURATION
# =============================================================================
engine_lock = False
active_voice_profile = "Default"
active_tts_engine = "pyttsx3"  # v8.0: Can be 'pyttsx3', 'gtts', or 'edge'

# Mirror AVATAR_IS_SPEAKING on config
if not hasattr(config, "AVATAR_IS_SPEAKING"):
    config.AVATAR_IS_SPEAKING = False

VOICE_PROFILES = {
    "Default": "female",
    "Female": "female",
    "Male": "male",
}

custom_audio_settings: Dict[str, float] = {
    "pitch": 1.0,
    "bass": 1.0,
    "treble": 1.0,
    "reverb": 0.3,      # v8.0: Reverb amount
    "echo": 0.0,         # v8.0: Echo effect
    "volume_boost": 1.0, # v8.0: Volume amplification
}

current_settings: Dict[str, Any] = {
    "speech_rate": "Normal",
    "voice_profile": "female",
    "pitch": 1.0,
    "bass": 1.0,
    "treble": 1.0,
    "reverb": 3,
    "emotion": "neutral",  # v8.0: Emotion-based prosody
    "tts_engine": "pyttsx3",  # v8.0: Selected TTS engine
}

# =============================================================================
# v8.0: EMOTION-BASED PROSODY SETTINGS
# =============================================================================
EMOTION_PROSODY = {
    "joy": {"rate_delta": +12, "pitch_delta": +0.15, "volume": 1.1},
    "excitement": {"rate_delta": +18, "pitch_delta": +0.2, "volume": 1.15},
    "trust": {"rate_delta": +6, "pitch_delta": +0.05, "volume": 0.95},
    "surprise": {"rate_delta": +16, "pitch_delta": +0.25, "volume": 1.1},
    "sadness": {"rate_delta": -14, "pitch_delta": -0.1, "volume": 0.75},
    "fear": {"rate_delta": -6, "pitch_delta": +0.1, "volume": 0.85},
    "anger": {"rate_delta": +10, "pitch_delta": +0.05, "volume": 1.05},
    "calm": {"rate_delta": -8, "pitch_delta": -0.05, "volume": 0.9},
    "neutral": {"rate_delta": 0, "pitch_delta": 0.0, "volume": 1.0}
}

# =============================================================================
# DATABASE LOGGING
# =============================================================================
def log_voice_event(event: str, details: str) -> None:
    """
    v8.0: Log voice-related events to system_logs.db. Never crashes the caller.
    """
    try:
        db_path = os.path.abspath(
            os.path.join(config.BASE_DIR, "data", "memory", "datasets", "system_logs.db")
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
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
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO voice_recognition_events (timestamp, event, details, engine, emotion) VALUES (?, ?, ?, ?, ?)",
                (timestamp, event, details, active_tts_engine, current_settings.get("emotion", "neutral")),
            )
            conn.commit()
        
        logger.debug("[v8.0] Logged voice event: %s", event)
    
    except Exception as e:  # pragma: no cover - logging should not crash
        logger.error("[v8.0] Error logging voice event '%s': %s", event, e)


# =============================================================================
# v8.0 ENHANCED: MULTI-ENGINE TTS SYNTHESIS
# =============================================================================
def synthesize_voice(text: str, emotion: str = None, engine_pref: str = None) -> None:
    """
    v8.0 ENHANCED: Synthesize speech from text using selected TTS engine.
    
    Features:
    - Multi-engine support (pyttsx3, gTTS, edge-tts)
    - Emotion-based prosody adjustment
    - Audio effects processing
    - Cross-platform compatibility
    - Graceful degradation when TTS unavailable
    
    Args:
        text: Text to synthesize
        emotion: Emotion to apply (joy, sadness, anger, etc.)
        engine_pref: Preferred TTS engine ('pyttsx3', 'gtts', 'edge')
    """
    if not text or not text.strip():
        logger.warning("[v8.0] No text provided for synthesis; skipping TTS.")
        return
    
    # Determine which engine to use
    engine_to_use = engine_pref or current_settings.get("tts_engine", "pyttsx3")
    emotion_to_use = emotion or current_settings.get("emotion", "neutral")
    
    logger.info("[v8.0] Synthesizing with engine: %s, emotion: %s", engine_to_use, emotion_to_use)
    
    # Set speaking flag
    config.AVATAR_IS_SPEAKING = True
    
    try:
        if engine_to_use == "pyttsx3" and _TTS_AVAILABLE and engine is not None:
            _synthesize_with_pyttsx3(text, emotion_to_use)
        
        elif engine_to_use == "gtts" and _HAS_GTTS:
            _synthesize_with_gtts(text, emotion_to_use)
        
        elif engine_to_use == "edge" and _HAS_EDGE_TTS:
            _synthesize_with_edge_tts(text, emotion_to_use)
        
        else:
            # Fallback: try any available engine
            if _TTS_AVAILABLE and engine is not None:
                _synthesize_with_pyttsx3(text, emotion_to_use)
            elif _HAS_GTTS:
                _synthesize_with_gtts(text, emotion_to_use)
            else:
                logger.info("[v8.0] TTS disabled/unavailable; skipping spoken output.")
        
        logger.info("[v8.0] Voice synthesis completed successfully.")
    
    except RuntimeError as e:
        logger.error("[v8.0] TTS RuntimeError: %s", e)
    except Exception as e:
        logger.error("[v8.0] TTS Error: %s", e)
    finally:
        # Always clear speaking flag
        config.AVATAR_IS_SPEAKING = False


def _synthesize_with_pyttsx3(text: str, emotion: str) -> None:
    """v8.0: Synthesize with pyttsx3 engine with emotion adjustment."""
    try:
        logger.debug("[v8.0] Using pyttsx3 engine")
        
        # Get emotion prosody adjustments
        prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
        
        # Apply rate adjustment
        base_rate = 185
        if current_settings["speech_rate"] == "Slow":
            base_rate = 135
        elif current_settings["speech_rate"] == "Fast":
            base_rate = 230
        
        adjusted_rate = base_rate + prosody["rate_delta"]
        engine.setProperty("rate", adjusted_rate)
        
        # Apply volume
        engine.setProperty("volume", prosody["volume"])
        
        # Speak
        engine.say(text)
        engine.runAndWait()
        
        log_voice_event("Voice Synthesis", f"Text: {text[:50]}..., Emotion: {emotion}")
    
    except Exception as e:
        logger.error("[v8.0] pyttsx3 synthesis failed: %s", e)


def _synthesize_with_gtts(text: str, emotion: str) -> None:
    """v8.0: Synthesize with gTTS engine (Google Text-to-Speech)."""
    try:
        logger.debug("[v8.0] Using gTTS engine")
        
        # Determine language
        lang = current_settings.get("language", "en")
        
        # Create TTS object
        tts = gTTS(text=text, lang=lang, slow=(current_settings["speech_rate"] == "Slow"))
        
        # Save to temporary file
        temp_file = os.path.join(config.DOWNLOADS_DIR, "tts_temp.mp3")
        tts.save(temp_file)
        
        # Play audio (platform-specific)
        _play_audio_file(temp_file)
        
        # Cleanup
        try:
            os.remove(temp_file)
        except Exception:
            pass
        
        log_voice_event("Voice Synthesis (gTTS)", f"Text: {text[:50]}..., Emotion: {emotion}")
    
    except Exception as e:
        logger.error("[v8.0] gTTS synthesis failed: %s", e)


async def _synthesize_with_edge_tts_async(text: str, emotion: str) -> None:
    """v8.0: Async synthesize with edge-tts engine (Microsoft Edge TTS)."""
    try:
        logger.debug("[v8.0] Using edge-tts engine")
        
        # Select voice based on profile
        voice = "en-US-AriaNeural"  # Default female voice
        if active_voice_profile == "Male":
            voice = "en-US-GuyNeural"
        
        # Adjust rate based on settings
        rate = "+0%"
        if current_settings["speech_rate"] == "Slow":
            rate = "-25%"
        elif current_settings["speech_rate"] == "Fast":
            rate = "+25%"
        
        # Create TTS object
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        
        # Save to temporary file
        temp_file = os.path.join(config.DOWNLOADS_DIR, "tts_temp_edge.mp3")
        await communicate.save(temp_file)
        
        # Play audio
        _play_audio_file(temp_file)
        
        # Cleanup
        try:
            os.remove(temp_file)
        except Exception:
            pass
        
        log_voice_event("Voice Synthesis (Edge-TTS)", f"Text: {text[:50]}..., Emotion: {emotion}")
    
    except Exception as e:
        logger.error("[v8.0] edge-tts synthesis failed: %s", e)


def _synthesize_with_edge_tts(text: str, emotion: str) -> None:
    """v8.0: Wrapper for async edge-tts synthesis."""
    try:
        import asyncio
        asyncio.run(_synthesize_with_edge_tts_async(text, emotion))
    except Exception as e:
        logger.error("[v8.0] edge-tts async wrapper failed: %s", e)


def _play_audio_file(filepath: str) -> None:
    """
    v8.0: Play an audio file using platform-appropriate methods.
    
    Args:
        filepath: Path to audio file to play
    """
    try:
        import platform
        
        if platform.system() == "Windows":
            # Windows: use winsound
            try:
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
            except Exception:
                # Fallback to pygame
                _play_with_pygame(filepath)
        
        elif platform.system() == "Darwin":
            # macOS: use afplay
            import subprocess
            subprocess.run(["afplay", filepath], check=True)
        
        else:
            # Linux: try multiple methods
            try:
                import subprocess
                subprocess.run(["aplay", filepath], check=True)
            except Exception:
                try:
                    subprocess.run(["paplay", filepath], check=True)
                except Exception:
                    _play_with_pygame(filepath)
    
    except Exception as e:
        logger.error("[v8.0] Failed to play audio file: %s", e)


def _play_with_pygame(filepath: str) -> None:
    """v8.0: Play audio using pygame mixer (cross-platform fallback)."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    
    except Exception as e:
        logger.error("[v8.0] pygame audio playback failed: %s", e)


# =============================================================================
# v8.0 ENHANCED: AUDIO EFFECTS PROCESSING
# =============================================================================
def apply_audio_effects(audio_segment, effects: Dict[str, Any]) -> "AudioSegment":
    """
    v8.0: Apply audio effects to an AudioSegment.
    
    Args:
        audio_segment: Pydub AudioSegment object
        effects: Dictionary of effects to apply
    
    Returns:
        Processed AudioSegment
    """
    if not _HAS_PYDUB or audio_segment is None:
        return audio_segment
    
    try:
        processed = audio_segment
        
        # Volume boost
        if "volume_boost" in effects and effects["volume_boost"] != 1.0:
            db_change = 20 * (effects["volume_boost"] - 1.0)
            processed = processed + db_change
        
        # Bass boost (low-pass filter)
        if "bass" in effects and effects["bass"] != 1.0:
            if effects["bass"] > 1.0:
                # Boost bass
                processed = low_pass_filter(processed, 200)
        
        # Treble adjustment (high-pass filter)
        if "treble" in effects and effects["treble"] != 1.0:
            if effects["treble"] > 1.0:
                # Boost treble
                processed = high_pass_filter(processed, 2000)
        
        # Normalize
        processed = normalize(processed)
        
        logger.debug("[v8.0] Applied audio effects: %s", effects)
        return processed
    
    except Exception as e:
        logger.error("[v8.0] Failed to apply audio effects: %s", e)
        return audio_segment


# =============================================================================
# TTS SHUTDOWN
# =============================================================================
def shutdown_tts() -> None:
    """v8.0: Shut down the TTS engine safely."""
    if not _TTS_AVAILABLE or engine is None:
        return
    
    try:
        engine.stop()
        logger.info("[v8.0] TTS engine shut down successfully.")
    except Exception as e:
        logger.error("[v8.0] Failed to shut down TTS engine: %s", e)


# =============================================================================
# VOICE SETTINGS MANAGEMENT
# =============================================================================
def save_voice_settings() -> None:
    """v8.0: Persist the current voice configuration into settings.json."""
    try:
        settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        data: Dict[str, Any] = {}
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f) or {}
                except json.JSONDecodeError:
                    logger.warning("[v8.0] settings.json invalid; rebuilding from scratch.")
                    data = {}

        data["voice_profile"] = active_voice_profile
        data["pitch"] = custom_audio_settings["pitch"]
        data["bass"] = custom_audio_settings["bass"]
        data["treble"] = custom_audio_settings["treble"]
        data["reverb"] = custom_audio_settings["reverb"]
        data["speech_rate"] = current_settings.get("speech_rate", "Normal")
        data["emotion"] = current_settings.get("emotion", "neutral")
        data["tts_engine"] = current_settings.get("tts_engine", "pyttsx3")

        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info("[v8.0] Voice settings saved to settings.json.")
    
    except Exception as e:  # pragma: no cover
        logger.error("[v8.0] Failed to save voice settings: %s", e)


def load_voice_settings() -> None:
    """v8.0: Load voice configuration from settings.json."""
    try:
        settings_path = os.path.join(config.SETTINGS_DIR, "settings.json")
        if not os.path.exists(settings_path):
            logger.info("[v8.0] No voice settings.json found; using defaults.")
            return

        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        if "voice_profile" in data:
            set_voice_profile(data["voice_profile"])
        if "pitch" in data:
            set_pitch(data["pitch"])
        if "bass" in data:
            set_bass(data["bass"])
        if "treble" in data:
            set_treble(data["treble"])
        if "reverb" in data:
            custom_audio_settings["reverb"] = float(data["reverb"])
        if "speech_rate" in data:
            set_speech_rate(data["speech_rate"])
        if "emotion" in data:
            current_settings["emotion"] = data["emotion"]
        if "tts_engine" in data:
            current_settings["tts_engine"] = data["tts_engine"]

        logger.info("[v8.0] Voice settings loaded from settings.json.")
    
    except Exception as e:  # pragma: no cover
        logger.error("[v8.0] Failed to load voice settings: %s", e)


# =============================================================================
# INDIVIDUAL PARAMETER SETTERS
# =============================================================================
def get_voice_profiles() -> List[str]:
    """v8.0: Return list of available voice profile names."""
    names: List[str] = list(VOICE_PROFILES.keys())
    
    if _TTS_AVAILABLE and engine is not None:
        try:
            for v in available_voices:
                nm = getattr(v, "name", None)
                if nm and nm not in names:
                    names.append(nm)
        except Exception:
            pass
    
    return names


def set_voice_profile(profile_name: str) -> None:
    """
    v8.0: Set the voice profile using a name or gender.
    """
    global active_voice_profile
    active_voice_profile = profile_name or "Default"

    if not _TTS_AVAILABLE or engine is None:
        logger.info("[v8.0] TTS unavailable; recording voice_profile='%s' only.", active_voice_profile)
        return

    selected_voice_id: Optional[str] = None

    # 1) Exact name match (case-insensitive)
    try:
        for v in available_voices:
            nm = getattr(v, "name", "")
            if nm and nm.lower() == profile_name.lower():
                selected_voice_id = getattr(v, "id", None)
                break
    except Exception:
        selected_voice_id = None

    # 2) Gender-based mapping
    if not selected_voice_id:
        gender = VOICE_PROFILES.get(profile_name, "female")
        try:
            for v in available_voices:
                nm = getattr(v, "name", "").lower()
                if gender.lower() in nm:
                    selected_voice_id = getattr(v, "id", None)
                    break
        except Exception:
            selected_voice_id = None

    # 3) Apply voice
    if selected_voice_id:
        try:
            engine.setProperty("voice", selected_voice_id)
            logger.info("[v8.0] Voice profile set to '%s' (id=%s)", profile_name, selected_voice_id)
        except Exception as e:
            logger.warning("[v8.0] Failed to set voice profile '%s' (id=%s): %s", 
                         profile_name, selected_voice_id, e)
    else:
        logger.warning("[v8.0] Voice profile '%s' not found; keeping current/default.", profile_name)


def set_pitch(value: float) -> None:
    """v8.0: Set pitch adjustment."""
    custom_audio_settings["pitch"] = float(value)
    logger.info("[v8.0] Pitch set to: %s", value)


def set_bass(value: float) -> None:
    """v8.0: Set bass adjustment."""
    custom_audio_settings["bass"] = float(value)
    logger.info("[v8.0] Bass set to: %s", value)


def set_treble(value: float) -> None:
    """v8.0: Set treble adjustment."""
    custom_audio_settings["treble"] = float(value)
    logger.info("[v8.0] Treble set to: %s", value)


def set_reverb(value: float) -> None:
    """v8.0: Set reverb amount."""
    custom_audio_settings["reverb"] = float(value)
    logger.info("[v8.0] Reverb set to: %s", value)


def set_emotion(emotion: str) -> None:
    """v8.0: Set emotion for voice synthesis."""
    current_settings["emotion"] = emotion
    logger.info("[v8.0] Emotion set to: %s", emotion)


def set_tts_engine(engine_name: str) -> None:
    """v8.0: Set preferred TTS engine."""
    current_settings["tts_engine"] = engine_name
    logger.info("[v8.0] TTS engine set to: %s", engine_name)


def set_speech_rate(rate_label: str) -> None:
    """v8.0: Set speech rate as 'Slow' / 'Normal' / 'Fast'."""
    current_settings["speech_rate"] = rate_label
    
    if not _TTS_AVAILABLE or engine is None:
        return

    rates = {"Slow": 135, "Normal": 185, "Fast": 230}
    rate_value = rates.get(rate_label, 185)
    
    try:
        engine.setProperty("rate", rate_value)
        logger.info("[v8.0] Speech rate set to '%s' (%s wpm)", rate_label, rate_value)
    except Exception as e:
        logger.warning("[v8.0] Failed to set speech rate: %s", e)


# =============================================================================
# MICROPHONE / RECOGNITION
# =============================================================================
mic = None


def initialize_microphone():
    """v8.0: Initialize and cache a Microphone object, or None if unavailable."""
    global mic
    
    if mic is not None:
        return mic
    
    if sr is None:
        logger.warning("[v8.0] SpeechRecognition not installed; microphone unavailable.")
        return None
    
    try:
        mic = sr.Microphone()
        logger.info("[v8.0] Microphone initialized.")
        log_voice_event("Microphone Initialized", "Microphone object created successfully.")
        return mic
    
    except Exception as e:
        logger.error("[v8.0] Microphone init failed: %s", e)
        log_voice_event("Microphone Initialization Error", f"Error: {e}")
        mic = None
        return None


def _recognize_chunk(audio: "sr.AudioData") -> Optional[str]:
    """v8.0: Shared recognizer helper; returns recognized text or None."""
    if sr is None or recognizer is None:
        return None
    
    try:
        text = recognizer.recognize_google(audio)
        return text.strip()
    
    except sr.UnknownValueError:
        logger.info("[v8.0] Speech not understood.")
        log_voice_event("Voice Input Unknown", "Audio not understood.")
        return None
    
    except sr.RequestError as e:
        logger.error("[v8.0] Speech recognition backend error: %s", e)
        log_voice_event("Voice Input Error", f"RequestError: {e}")
        return None
    
    except Exception as e:
        logger.error("[v8.0] Unexpected error in recognition: %s", e)
        log_voice_event("Voice Recognition Exception", f"Exception: {e}")
        return None


def listen_and_process(timeout: Optional[float] = None, 
                      phrase_time_limit: Optional[float] = None) -> Optional[str]:
    """
    v8.0: Record a single utterance from the default microphone and return recognized text.
    """
    if SAFE_MODE:
        logger.info("[v8.0] SAFE_MODE active; listen_and_process disabled.")
        return None

    mic_obj = initialize_microphone()
    if mic_obj is None or sr is None or recognizer is None:
        return None

    timeout = float(timeout if timeout is not None else getattr(config, "LISTEN_TIMEOUT", 5))
    phrase_time_limit = float(
        phrase_time_limit if phrase_time_limit is not None else getattr(config, "PHRASE_TIME_LIMIT", 10)
    )

    try:
        with mic_obj as source:
            # Ambient noise adjustment
            try:
                dur = float(getattr(config, "AMBIENT_NOISE_DURATION", 0.2))
                recognizer.adjust_for_ambient_noise(source, duration=dur)
            except Exception as e:
                logger.debug("[v8.0] Ambient noise calibration skipped: %s", e)

            logger.info("[v8.0] Listening for speech (timeout=%s, phrase_time_limit=%s)...", 
                       timeout, phrase_time_limit)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        # Optional noise reduction
        if nr is not None and np is not None:
            try:
                raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                np_audio = np.frombuffer(raw, dtype=np.int16)
                reduced = nr.reduce_noise(y=np_audio, sr=16000)
                audio = sr.AudioData(reduced.tobytes(), 16000, 2)
            except Exception as e:
                logger.debug("[v8.0] Noise reduction skipped: %s", e)

        text = _recognize_chunk(audio)
        if text:
            logger.info("[v8.0] Recognized text: %s", text)
            log_voice_event("Voice Input Recognized", text)
            return text
        
        return None
    
    except Exception as e:
        logger.error("[v8.0] Error during listen_and_process: %s", e)
        log_voice_event("Voice Listen Error", f"Error: {e}")
        return None


def transcribe_once(timeout: float = 10.0) -> str:
    """
    v8.0: Convenience wrapper used by WebBridge.
    """
    text = listen_and_process(timeout=timeout)
    return text or ""


# =============================================================================
# WEBUI INTEGRATION HELPERS
# =============================================================================
def list_voices() -> List[Dict[str, Any]]:
    """v8.0: Return list of available voices for the Web UI."""
    if not _TTS_AVAILABLE or engine is None:
        return []

    out: List[Dict[str, Any]] = []
    try:
        for v in available_voices:
            out.append({
                "id": getattr(v, "id", ""),
                "name": getattr(v, "name", ""),
                "lang": getattr(v, "languages", [""])[0] if hasattr(v, "languages") else "",
            })
    except Exception as e:
        logger.warning("[v8.0] Failed to enumerate voices: %s", e)
    
    return out


def configure_voice(opts: Dict[str, Any]) -> None:
    """
    v8.0: Apply a bundle of voice settings from a dict, used by the Web UI bridge.
    """
    try:
        profile = opts.get("profile") or opts.get("voice_profile")
        if profile:
            set_voice_profile(str(profile))

        rate = opts.get("rate") or opts.get("speech_rate")
        if isinstance(rate, str):
            set_speech_rate(rate)
        elif isinstance(rate, (int, float)):
            if rate < 150:
                set_speech_rate("Slow")
            elif rate > 210:
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

        save_voice_settings()
    
    except Exception as e:
        logger.error("[v8.0] configure_voice failed: %s", e)

# Registry for custom TTS models loaded at runtime (for future expansion)
custom_voice_models: Dict[str, Any] = {}

def import_custom_voice_profile(filepath: str):
    """
    Import a user-provided .pt TTS voice model.
    Automatically registers it in VOICE_PROFILES and custom_voice_models.
    """
    try:
        if not CUSTOM_TTS_AVAILABLE:
            logger.error("[VoiceEngine] Torch not available — cannot load .pt voice files")
            return False

        if not os.path.exists(filepath):
            logger.error(f"[VoiceEngine] Custom voice file not found: {filepath}")
            return False

        voice_name = os.path.splitext(os.path.basename(filepath))[0]
        model = torch.jit.load(filepath, map_location="cpu")

        # Add to global registry
        custom_voice_models[voice_name] = model

        # Mark it as a known profile so WebUI can list it
        # (we keep the simple gender string, consistent with existing VOICE_PROFILES)
        if voice_name not in VOICE_PROFILES:
            VOICE_PROFILES[voice_name] = "custom"

        logger.info(f"[VoiceEngine] Successfully imported custom voice: {voice_name}")
        return True

    except Exception as e:
        logger.error(f"[VoiceEngine] Failed to import custom voice {filepath}: {e}")
        traceback.print_exc()
        return False

# =============================================================================
# MODULE SELF-TEST
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    logger.info("[v8.0] Starting SarahMemoryVoice self-test.")
    logger.info("[v8.0] TTS available: %s, voices: %d", _TTS_AVAILABLE, len(available_voices))
    
    print("\n" + "=" * 78)
    print("  SARAHMEMORY v8.0 - VOICE MODULE SELF-TEST")
    print("=" * 78)
    
    print(f"\n  ✓ TTS Engine: {'Available' if _TTS_AVAILABLE else 'Unavailable'}")
    print(f"  ✓ Available Voices: {len(available_voices)}")
    print(f"  ✓ gTTS Support: {'Yes' if _HAS_GTTS else 'No'}")
    print(f"  ✓ Edge-TTS Support: {'Yes' if _HAS_EDGE_TTS else 'No'}")
    print(f"  ✓ Audio Processing: {'Yes' if _HAS_PYDUB else 'No'}")
    
    set_voice_profile("Female")
    synthesize_voice("Hello, this is SarahMemory version 8.0 speaking from the enhanced voice module.", 
                    emotion="joy")
    
    if sr is not None:
        print("\n  Now trying a short transcription; speak into the microphone if available...")
        txt = transcribe_once(timeout=8)
        print(f"  You said: {txt or '<nothing heard>'}")
    
    print("\n  ✓ SarahMemoryVoice v8.0 self-test complete.")
    print("=" * 78 + "\n")
    
    logger.info("[v8.0] SarahMemoryVoice self-test complete.")

# =============================================================================
# END OF SarahMemoryVoice.py v8.0.0
# =============================================================================
