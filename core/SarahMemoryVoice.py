"""--==The SarahMemory Project==--
File: SarahMemoryVoice.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

SarahMemory v8.0 - Voice & Sound Synthesis Module

CORE GOALS
- One authoritative backend voice pipeline (no FE speech injection required).
- Avoid mid-sentence cutoffs by serializing TTS requests through a single worker.
- Remain headless-safe on cloud servers (PythonAnywhere) and Windows-safe locally.
- Preserve existing public APIs (synthesize_voice, transcribe_once, etc.) while
adding a robust queue-based speak_text().

NOTES
- WebUI speech should call backend endpoints (e.g., /api/ui/event) which in turn
call speak_text() in this file. This eliminates the need for FE_v800_app_speech.py
and any speech.js concept.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "voice_engine"
# CATEGORY = "voice_and_audio_io"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "voice"
# HARDWARE_DOMAIN = "audio_microphone_speakers"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "voice"
# FAMILY = "core_voice"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Authoritative backend voice pipeline for TTS/STT, queue-based speech output, engine fallback, voice settings, sanitization, and headless-safe audio behavior."
# --- SARAHMETA END ---

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
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

# ARILE organ sentinel helper. This file reports local variance to the central
# SarahMemoryARILE.py engine without owning ARILE authority.
try:
    from SarahMemoryARILE import ARILESentinelBase, arile_emit, arile_should_run
except Exception:  # pragma: no cover
    ARILESentinelBase = object  # type: ignore
    arile_emit = None  # type: ignore
    def arile_should_run(lane: str, source: str = "unknown", default: bool = True) -> bool:
        return bool(default)

class LocalARILESentinel(ARILESentinelBase):
    organ_name = __name__

    def report(self, failure_type: str, summary: str, severity: float = 0.50, **data) -> None:
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, organ=self.organ_name, kind="organ_variance", failure_type=failure_type, severity=severity, confidence=0.82, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
        except Exception:
            pass

_local_arile_sentinel = LocalARILESentinel()



# -----------------------------------------------------------------------------
# Output Sanitization (hide system prompts / chain-of-thought from TTS)
# -----------------------------------------------------------------------------
def _sm_sanitize_llm_text_local(text: str) -> str:
    if text is None:
        return ""

    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(?is)<think>.*?</think>", "", t)
    t = re.sub(r"(?is)<analysis>.*?</analysis>", "", t)
    t = re.sub(r"(?im)^\s*(system|user|assistant)\s*:\s*.*$", "", t)
    t = re.sub(r"(?im)^\s*\[(system|user|assistant)\]\s*.*$", "", t)
    if "Assistant:" in t:
        t = t.split("Assistant:")[-1].strip()

    # Markdown / provenance cleanup for spoken output.
    t = re.sub(r"(?is)```.*?```", lambda m: re.sub(r"^```[a-zA-Z0-9_-]*\n?|\n?```$", "", m.group(0).strip()), t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
    t = re.sub(r"(?im)^\s*\[\s*source\s*:[^\]]*\]\s*$", "", t)
    t = re.sub(r"(?im)^\s*\(\s*intent\s*:[^\)]*\)\s*$", "", t)
    t = re.sub(r"\s*\[\s*source\s*:[^\]]*\]\s*", " ", t, flags=re.I)
    t = re.sub(r"\s*\(\s*intent\s*:[^\)]*\)\s*", " ", t, flags=re.I)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", t)
    t = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", t)
    t = re.sub(r"~~([^~]+)~~", r"\1", t)
    t = t.replace("™", " ").replace("®", " ").replace("©", " ")
    t = t.replace("•", "\n").replace("▪", "\n").replace("▫", "\n").replace("◦", "\n")
    t = t.replace("&", " and ")
    t = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", " ", t)
    t = re.sub(r"[*/_~`#|^<>\[\]{}\\]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = t.replace("\n", ". ")
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?]){2,}", lambda m: m.group(0)[0], t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip(" \t\n-–—*_`#|:;,")

# =============================================================================
# GLOBAL CONFIG (safe import)
# =============================================================================
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    # Allow module import in isolated tests
    class _Cfg:  # pragma: no cover
        _here = Path(__file__).resolve().parent
        BASE_DIR = str(_here.parent if _here.name.lower() == "core" else _here)
        DATA_DIR = os.path.join(BASE_DIR, "data")
        SETTINGS_DIR = os.path.join(DATA_DIR, "settings")
        DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
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
    try:
        recognizer.pause_threshold = float(getattr(config, "VOICE_PAUSE_THRESHOLD", 0.95) or 0.95)
        recognizer.non_speaking_duration = float(getattr(config, "VOICE_NON_SPEAKING_DURATION", 0.45) or 0.45)
        recognizer.phrase_threshold = float(getattr(config, "VOICE_PHRASE_THRESHOLD", 0.25) or 0.25)
    except Exception:
        pass

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

# CosyVoice dependencies must remain optional.
try:
    import torchaudio  # type: ignore
    _HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None  # type: ignore
    _HAS_TORCHAUDIO = False

# =============================================================================
# PATH HELPERS (consistent BASE/DATA directories)
# =============================================================================
def _base_dir() -> Path:
    bd = getattr(config, "BASE_DIR", None)
    if bd:
        return Path(bd)
    try:
        here = Path(__file__).resolve().parent
        return here.parent if here.name.lower() == "core" else here
    except Exception:
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
    "tts_engine": "auto",  # pyttsx3|gtts|edge|cosyvoice|auto
    "language": "en",
}

active_voice_profile: str = "Female"

# =============================================================================
# CATEGORY-DRIVEN TTS RESOLUTION
# =============================================================================
_SUPPORTED_TTS_ENGINES: Tuple[str, ...] = ("pyttsx3", "gtts", "edge", "cosyvoice", "auto")
_TTS_MODEL_BACKEND_MAP: Dict[str, str] = {
    "FunAudioLLM/CosyVoice2-0.5B": "cosyvoice",
}

_COSYVOICE_RUNTIME = None
_COSYVOICE_RUNTIME_META: Dict[str, Any] = {
    "repo": None,
    "model_dir": None,
    "backend": None,
    "sample_rate": 22050,
}
_COSYVOICE_LOCK = threading.Lock()


def _normalize_tts_engine_name(engine_name: Optional[str]) -> str:
    val = str(engine_name or "").strip().lower()
    aliases = {
        "pytts": "pyttsx3",
        "pyttsx": "pyttsx3",
        "pyttsx3": "pyttsx3",
        "gtts": "gtts",
        "google": "gtts",
        "googletts": "gtts",
        "edge": "edge",
        "edge-tts": "edge",
        "edgetts": "edge",
        "cosyvoice": "cosyvoice",
        "cosyvoice2": "cosyvoice",
        "auto": "auto",
    }
    return aliases.get(val, val)


def _tts_repo_to_backend(repo: Optional[str]) -> str:
    repo_val = str(repo or "").strip()
    if not repo_val:
        return ""
    backend = _TTS_MODEL_BACKEND_MAP.get(repo_val, "")
    if backend:
        return backend
    if "cosyvoice" in repo_val.lower():
        return "cosyvoice"
    return ""


def _resolve_tts_model_candidates(text: str = "", lang: str = "en") -> Dict[str, Any]:
    default = {
        "selected": None,
        "fallbacks": [],
        "source": "none",
        "score": 0.0,
        "tier": "low",
        "tier_rating": "Poor",
        "third_party_autoload_allowed": False,
    }
    try:
        if hasattr(config, "resolve_model"):
            resolved = config.resolve_model(
                "tts",
                text=text or "",
                meta={"lang": str(lang or "en")},
                models_dir=getattr(config, "MODELS_DIR", None),
            )
            if isinstance(resolved, dict):
                return {**default, **resolved}
    except Exception as exc:
        logger.debug("[Voice] TTS model resolution failed: %s", exc)
    return default


def _build_engine_fallback_chain(primary_engine: str, allow_remote: bool = True) -> List[str]:
    normalized = _normalize_tts_engine_name(primary_engine)
    chain: List[str] = []

    def _add(name: Optional[str]) -> None:
        eng = _normalize_tts_engine_name(name)
        if not eng or eng == "auto":
            return
        if eng not in chain:
            chain.append(eng)

    _add(normalized or "pyttsx3")
    if allow_remote:
        _add("edge")
        _add("gtts")
    _add("pyttsx3")
    return chain or ["pyttsx3"]


def _resolve_tts_runtime_plan(text: str = "", lang: str = "en", explicit_engine: Optional[str] = None) -> Dict[str, Any]:
    requested_engine = _normalize_tts_engine_name(explicit_engine or current_settings.get("tts_engine", "pyttsx3"))
    allow_remote = not bool(getattr(config, "LOCAL_ONLY_MODE", False))
    if requested_engine and requested_engine != "auto":
        return {
            "engine": requested_engine,
            "engine_chain": _build_engine_fallback_chain(requested_engine, allow_remote=allow_remote),
            "requested_engine": requested_engine,
            "selected_repo": None,
            "fallback_repos": [],
            "backend_source": "user_engine",
            "tier_rating": "Poor",
            "third_party_autoload_allowed": False,
            "model_resolution": None,
        }

    resolved = _resolve_tts_model_candidates(text=text or "", lang=lang or "en")
    selected_repo = str((resolved or {}).get("selected") or "").strip() or None
    fallback_repos = [str(x).strip() for x in ((resolved or {}).get("fallbacks") or []) if str(x).strip()]
    engine = _tts_repo_to_backend(selected_repo)
    if not engine:
        for repo_name in fallback_repos:
            engine = _tts_repo_to_backend(repo_name)
            if engine:
                break

    if not engine:
        if _HAS_EDGE_TTS:
            engine = "edge"
        elif _HAS_GTTS and allow_remote:
            engine = "gtts"
        else:
            engine = "pyttsx3"

    return {
        "engine": engine,
        "engine_chain": _build_engine_fallback_chain(engine, allow_remote=allow_remote),
        "requested_engine": requested_engine or "auto",
        "selected_repo": selected_repo,
        "fallback_repos": fallback_repos,
        "backend_source": str((resolved or {}).get("source") or "none"),
        "tier_rating": str((resolved or {}).get("tier_rating") or "Poor"),
        "third_party_autoload_allowed": bool((resolved or {}).get("third_party_autoload_allowed", False)),
        "model_resolution": resolved,
    }


def _cosyvoice_model_dir(repo: Optional[str]) -> Optional[str]:
    repo_val = str(repo or "").strip()
    if not repo_val:
        return None

    candidates: List[str] = []
    try:
        if hasattr(config, "_repo_to_local_dir"):
            local_dir = config._repo_to_local_dir(repo_val, getattr(config, "MODELS_DIR", None) or str(_data_dir() / "models"))
            if local_dir:
                candidates.append(str(local_dir))
    except Exception:
        pass

    models_dir = getattr(config, "MODELS_DIR", None)
    if models_dir:
        safe_name = repo_val.replace("/", "_")
        candidates.append(os.path.join(str(models_dir), safe_name))
        candidates.append(os.path.join(str(models_dir), os.path.basename(repo_val)))
    candidates.append(repo_val)

    for item in candidates:
        try:
            p = Path(str(item)).expanduser()
            if p.is_dir() and any(p.iterdir()):
                return str(p)
        except Exception:
            continue
    return None


def _iter_cosyvoice_audio_segments(result: Any) -> Generator[Any, None, None]:
    if result is None:
        return
    if isinstance(result, dict):
        if "tts_speech" in result:
            yield result["tts_speech"]
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            yield from _iter_cosyvoice_audio_segments(item)
        return
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray, dict)):
        try:
            for item in result:
                yield from _iter_cosyvoice_audio_segments(item)
        except TypeError:
            pass
        return
    yield result


def _pick_cosyvoice_speaker(runtime_obj: Any, profile: str, lang: str) -> Optional[str]:
    try:
        lister = getattr(runtime_obj, "list_avaliable_spks", None) or getattr(runtime_obj, "list_available_spks", None)
        if not callable(lister):
            return None
        speakers = list(lister() or [])
        if not speakers:
            return None
        want_male = (VOICE_PROFILES.get(profile) or profile or "female").lower() == "male"
        lang_tokens: List[str] = []
        if str(lang or "").lower().startswith("en"):
            lang_tokens.extend(["en", "eng", "english"])
        elif str(lang or "").lower().startswith("zh"):
            lang_tokens.extend(["zh", "chinese", "中文"])
        for spk in speakers:
            spk_low = str(spk).lower()
            if want_male and any(tok in spk_low for tok in ("male", "man", "guy", "男")):
                return str(spk)
            if (not want_male) and any(tok in spk_low for tok in ("female", "woman", "girl", "zira", "aria", "jenny", "女")):
                return str(spk)
        for spk in speakers:
            spk_low = str(spk).lower()
            if lang_tokens and any(tok in spk_low for tok in lang_tokens):
                return str(spk)
        return str(speakers[0])
    except Exception:
        return None


def _ensure_cosyvoice_runtime(repo: Optional[str]) -> Tuple[Optional[Any], Dict[str, Any]]:
    repo_val = str(repo or "").strip()
    meta = {
        "ok": False,
        "repo": repo_val,
        "model_dir": None,
        "backend": None,
        "reason": "uninitialized",
    }
    if not repo_val:
        meta["reason"] = "missing_repo"
        return None, meta
    if _headless_safe():
        meta["reason"] = "headless_runtime"
        return None, meta
    if not _HAS_TORCHAUDIO or torchaudio is None:
        meta["reason"] = "torchaudio_unavailable"
        return None, meta

    model_dir = _cosyvoice_model_dir(repo_val)
    meta["model_dir"] = model_dir
    if not model_dir:
        meta["reason"] = "model_dir_missing"
        return None, meta

    with _COSYVOICE_LOCK:
        global _COSYVOICE_RUNTIME, _COSYVOICE_RUNTIME_META
        if _COSYVOICE_RUNTIME is not None and _COSYVOICE_RUNTIME_META.get("repo") == repo_val and _COSYVOICE_RUNTIME_META.get("model_dir") == model_dir:
            meta.update({
                "ok": True,
                "backend": _COSYVOICE_RUNTIME_META.get("backend"),
                "reason": "ready",
            })
            return _COSYVOICE_RUNTIME, meta

        runtime_obj = None
        backend_name = None
        errors: List[str] = []

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
            runtime_obj = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
            backend_name = "CosyVoice2"
        except Exception as exc:
            errors.append(f"CosyVoice2:{exc}")

        if runtime_obj is None:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
                runtime_obj = CosyVoice(model_dir, load_jit=False, load_onnx=False, load_trt=False, fp16=False)
                backend_name = "CosyVoice"
            except Exception as exc:
                errors.append(f"CosyVoice:{exc}")

        if runtime_obj is None:
            meta["reason"] = "init_failed:" + " | ".join(errors[:3])
            return None, meta

        _COSYVOICE_RUNTIME = runtime_obj
        _COSYVOICE_RUNTIME_META = {
            "repo": repo_val,
            "model_dir": model_dir,
            "backend": backend_name,
            "sample_rate": int(getattr(runtime_obj, "sample_rate", 22050) or 22050),
        }
        meta.update({
            "ok": True,
            "backend": backend_name,
            "reason": "ready",
        })
        return runtime_obj, meta


def _save_wave_tensor_to_file(audio_obj: Any, sample_rate: int, out_path: Path) -> bool:
    if not _HAS_TORCHAUDIO or torchaudio is None:
        return False
    try:
        import torch  # type: ignore
    except Exception:
        return False

    try:
        tensor = audio_obj
        if isinstance(audio_obj, (list, tuple)):
            tensor = torch.tensor(audio_obj)
        elif not hasattr(audio_obj, "shape"):
            tensor = torch.tensor(audio_obj)
        if getattr(tensor, "dim", lambda: 0)() == 1:
            tensor = tensor.unsqueeze(0)
        elif getattr(tensor, "dim", lambda: 0)() > 2:
            tensor = tensor.squeeze()
            if getattr(tensor, "dim", lambda: 0)() == 1:
                tensor = tensor.unsqueeze(0)
        tensor = tensor.detach().cpu().float()
        torchaudio.save(str(out_path), tensor, int(sample_rate or 22050))
        return True
    except Exception:
        return False


def _speak_with_cosyvoice(text: str, profile: str, emotion: str, lang: str, repo: Optional[str]) -> bool:
    runtime_obj, runtime_meta = _ensure_cosyvoice_runtime(repo)
    if runtime_obj is None:
        logger.info("[Voice] CosyVoice unavailable: %s", runtime_meta.get("reason"))
        return False
    if _TTS_STOP_FLAG.is_set():
        return False

    chunks = _split_text_for_tts(text, max_chars=220)
    if not chunks:
        return False

    speaker = _pick_cosyvoice_speaker(runtime_obj, profile, lang)
    if not speaker:
        logger.info("[Voice] CosyVoice runtime ready but no SFT speakers exposed; falling back.")
        return False

    sample_rate = int(_COSYVOICE_RUNTIME_META.get("sample_rate") or getattr(runtime_obj, "sample_rate", 22050) or 22050)
    spoke_any = False
    infer = getattr(runtime_obj, "inference_sft", None)
    if not callable(infer):
        logger.info("[Voice] CosyVoice runtime does not expose inference_sft; falling back.")
        return False

    for idx, chunk in enumerate(chunks):
        if _TTS_STOP_FLAG.is_set():
            return spoke_any
        try:
            result = infer(chunk, speaker, stream=False)
            audio_segments = list(_iter_cosyvoice_audio_segments(result))
            for seg_idx, segment in enumerate(audio_segments):
                if _TTS_STOP_FLAG.is_set():
                    return spoke_any
                out_path = _downloads_dir() / f"sm_cosyvoice_{uuid.uuid4().hex}_{idx}_{seg_idx}.wav"
                try:
                    if _save_wave_tensor_to_file(segment, sample_rate, out_path):
                        _play_audio_file(str(out_path))
                        spoke_any = True
                finally:
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("[Voice] CosyVoice inference failed: %s", exc)
            logger.debug(traceback.format_exc())
            return spoke_any
    return spoke_any

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
_TTS_WORKER_THREAD: Optional[threading.Thread] = None
_TTS_WORKER_LOCK = threading.Lock()
_TTS_SHUTDOWN_FLAG = threading.Event()
_TTS_LAST_LOCK = threading.Lock()
_TTS_LAST_TEXT_HASH = ""
_TTS_LAST_ACCEPTED_TS = 0.0

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


def _speak_with_pyttsx3(text: str, profile: str, emotion: str) -> bool:
    if not _ensure_pyttsx3_engine() or _TTS_SHUTDOWN_FLAG.is_set() or _TTS_STOP_FLAG.is_set():
        return False
    _apply_engine_tuning(profile, emotion)
    chunks = _split_text_for_tts(text)
    if not chunks:
        return False
    try:
        for chunk in chunks:
            if _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
                try:
                    _engine.stop()  # type: ignore
                except Exception:
                    pass
                return False
            _engine.say(chunk)  # type: ignore
        _engine.runAndWait()  # type: ignore
        return not _TTS_STOP_FLAG.is_set() and not _TTS_SHUTDOWN_FLAG.is_set()
    except Exception as exc:
        logger.warning("[Voice] pyttsx3 speak failed, resetting engine: %s", exc)
        try:
            if _engine is not None:
                _engine.stop()  # type: ignore
        except Exception:
            pass
        _reset_pyttsx3_engine()
        return False

def _reset_pyttsx3_engine() -> None:
    """Force a clean re-init of pyttsx3 on next use (fixes stuck/cutoff engines)."""
    global _engine_ready, _engine, _engine_voices
    _engine_ready = False
    _engine = None
    _engine_voices = []


def _speak_with_gtts(text: str, lang: str = "en") -> bool:
    if not _HAS_GTTS or gTTS is None or _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
        return False
    tmp = _downloads_dir() / f"sm_tts_{uuid.uuid4().hex}.mp3"
    try:
        tts = gTTS(text=text, lang=lang, slow=(str(current_settings.get("speech_rate")) == "Slow"))
        tts.save(str(tmp))
        if _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
            return False
        _play_audio_file(str(tmp))
        return not _TTS_STOP_FLAG.is_set() and not _TTS_SHUTDOWN_FLAG.is_set()
    except Exception as exc:
        logger.warning("[Voice] gTTS playback failed: %s", exc)
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

def _speak_with_edge_tts(text: str, profile: str) -> bool:
    if not _HAS_EDGE_TTS or edge_tts is None or _headless_safe() or _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
        return False
    voice = "en-US-AriaNeural"
    if (VOICE_PROFILES.get(profile) or "").lower() == "male":
        voice = "en-US-GuyNeural"
    rate = "+0%"
    sr_label = str(current_settings.get("speech_rate", "Normal"))
    if sr_label == "Slow":
        rate = "-25%"
    elif sr_label == "Fast":
        rate = "+25%"
    out = _downloads_dir() / f"sm_edge_{uuid.uuid4().hex}.mp3"
    try:
        asyncio.run(_edge_tts_async(text, voice, rate, out))
        if _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
            return False
        _play_audio_file(str(out))
        return not _TTS_STOP_FLAG.is_set() and not _TTS_SHUTDOWN_FLAG.is_set()
    except Exception as exc:
        logger.warning("[Voice] edge-tts playback failed: %s", exc)
        return False
    finally:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass

# =============================================================================
# TTS QUEUE / WORKER
# =============================================================================
@dataclass
class _TTSTask:
    text: str
    blocking_event: Optional[threading.Event]
    emotion: str
    engine_pref: str
    voice_profile: str
    lang: str
    selected_repo: Optional[str] = None
    fallback_repos: Optional[List[str]] = None
    backend_source: str = "none"
    task_id: str = ""
    result: Optional[Dict[str, Any]] = None


def _start_tts_worker() -> bool:
    global _TTS_WORKER_STARTED, _TTS_WORKER_THREAD
    if _TTS_SHUTDOWN_FLAG.is_set():
        return False
    with _TTS_WORKER_LOCK:
        if _TTS_WORKER_STARTED and _TTS_WORKER_THREAD is not None and _TTS_WORKER_THREAD.is_alive():
            return True
        _TTS_WORKER_STARTED = False
        t = threading.Thread(target=_tts_worker_loop, name="SarahMemoryVoiceTTS", daemon=True)
        _TTS_WORKER_THREAD = t
        t.start()
        _TTS_WORKER_STARTED = True
        return True

def _tts_worker_loop() -> None:
    global _TTS_WORKER_STARTED
    try:
        while not _TTS_SHUTDOWN_FLAG.is_set():
            task = _TTS_QUEUE.get()
            try:
                if task is None:
                    break
                if not isinstance(task, _TTSTask) or not task.text.strip():
                    if isinstance(task, _TTSTask) and task.result is not None:
                        task.result.update({"ok": False, "reason": "empty_task"})
                    continue
                config.AVATAR_IS_SPEAKING = True
                _TTS_STOP_FLAG.clear()
                requested_engine = _normalize_tts_engine_name(task.engine_pref or "pyttsx3") or "pyttsx3"
                emotion = (task.emotion or "neutral").strip().lower()
                profile = (task.voice_profile or "Female").strip()
                selected_repo = str(task.selected_repo or "").strip() or None
                fallback_repos = list(task.fallback_repos or [])
                backend_source = str(task.backend_source or "none")
                engine_chain = _build_engine_fallback_chain(requested_engine, allow_remote=not bool(getattr(config, "LOCAL_ONLY_MODE", False)))
                completed_engine = None
                completed = False
                failure_notes: List[str] = []
                for engine in engine_chain:
                    if _TTS_STOP_FLAG.is_set() or _TTS_SHUTDOWN_FLAG.is_set():
                        break
                    try:
                        if engine == "cosyvoice":
                            repo_candidates: List[str] = []
                            if selected_repo:
                                repo_candidates.append(selected_repo)
                            for repo_name in fallback_repos:
                                if repo_name and repo_name not in repo_candidates:
                                    repo_candidates.append(repo_name)
                            if not repo_candidates and hasattr(config, "get_stack_primary_repo"):
                                default_repo = config.get_stack_primary_repo("tts", task.text, {"lang": task.lang or "en"})
                                if default_repo:
                                    repo_candidates.append(str(default_repo))
                            for repo_name in repo_candidates:
                                if _speak_with_cosyvoice(task.text, profile, emotion, task.lang or "en", repo_name):
                                    completed_engine, completed = "cosyvoice", True
                                    break
                            if completed:
                                break
                            failure_notes.append("cosyvoice_unavailable")
                        elif engine == "edge":
                            completed = _speak_with_edge_tts(task.text, profile)
                            if completed:
                                completed_engine = "edge"
                                break
                            failure_notes.append("edge_unavailable_or_failed")
                        elif engine == "gtts":
                            if bool(getattr(config, "LOCAL_ONLY_MODE", False)):
                                failure_notes.append("gtts_blocked_local_only")
                            else:
                                completed = _speak_with_gtts(task.text, lang=task.lang or "en")
                                if completed:
                                    completed_engine = "gtts"
                                    break
                                failure_notes.append("gtts_unavailable_or_failed")
                        elif engine == "pyttsx3":
                            completed = _speak_with_pyttsx3(task.text, profile, emotion)
                            if completed:
                                completed_engine = "pyttsx3"
                                break
                            failure_notes.append("pyttsx3_unavailable_or_failed")
                        else:
                            failure_notes.append(f"unknown_engine:{engine}")
                    except Exception as engine_exc:
                        failure_notes.append(f"{engine}:{engine_exc}")
                        logger.warning("[Voice] TTS backend '%s' failed: %s", engine, engine_exc)
                        logger.debug(traceback.format_exc())
                if task.result is not None:
                    task.result.update({"ok": bool(completed), "engine": completed_engine or "", "requested_engine": requested_engine, "stopped": bool(_TTS_STOP_FLAG.is_set()), "shutdown": bool(_TTS_SHUTDOWN_FLAG.is_set()), "failure_notes": failure_notes[:8]})
                if not completed and failure_notes:
                    logger.info("[Voice] TTS fallback chain exhausted: %s", " | ".join(failure_notes[:5]))
                try:
                    log_voice_event("TTS", f"task_id={task.task_id} ok={completed} engine={completed_engine or 'none'} requested={requested_engine} source={backend_source} emotion={emotion} text={task.text[:120]}")
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("[Voice] TTS task failed: %s", exc)
                logger.debug(traceback.format_exc())
                if isinstance(task, _TTSTask) and task.result is not None:
                    task.result.update({"ok": False, "reason": str(exc)})
            finally:
                config.AVATAR_IS_SPEAKING = False
                if isinstance(task, _TTSTask) and task.blocking_event is not None:
                    try:
                        task.blocking_event.set()
                    except Exception:
                        pass
                try:
                    _TTS_QUEUE.task_done()
                except Exception:
                    pass
    finally:
        config.AVATAR_IS_SPEAKING = False
        with _TTS_WORKER_LOCK:
            _TTS_WORKER_STARTED = False

# =============================================================================
# PUBLIC API: SPEAK / STOP
# =============================================================================
def stop_speaking(clear_queue: bool = True) -> None:
    """Stop current speech and optionally cancel queued utterances."""
    _TTS_STOP_FLAG.set()
    try:
        if _engine is not None:
            _engine.stop()  # type: ignore
    except Exception:
        pass
    if clear_queue:
        while True:
            try:
                pending = _TTS_QUEUE.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(pending, _TTSTask):
                    if pending.result is not None:
                        pending.result.update({"ok": False, "stopped": True, "reason": "queue_cancelled"})
                    if pending.blocking_event is not None:
                        pending.blocking_event.set()
            finally:
                try:
                    _TTS_QUEUE.task_done()
                except Exception:
                    pass

def speak_text(text: str, blocking: bool = True, emotion: Optional[str] = None, engine_pref: Optional[str] = None) -> bool:
    """Queue one complete utterance and report whether playback completed."""
    global _TTS_LAST_TEXT_HASH, _TTS_LAST_ACCEPTED_TS
    if not text or not str(text).strip() or SAFE_MODE or _TTS_SHUTDOWN_FLAG.is_set():
        return False
    try:
        from SarahMemoryAPI import _sm_sanitize_llm_text as _san
        text = _san(text)
    except Exception:
        pass
    text = _sm_sanitize_llm_text_local(text)
    if not text:
        return False
    import hashlib
    now = time.time()
    text_hash = hashlib.sha256(str(text).strip().encode("utf-8", "ignore")).hexdigest()
    dedupe_window = float(getattr(config, "TTS_DEDUPE_WINDOW_SECONDS", 1.25) or 1.25)
    with _TTS_LAST_LOCK:
        if text_hash == _TTS_LAST_TEXT_HASH and (now - _TTS_LAST_ACCEPTED_TS) <= max(0.0, dedupe_window):
            return True
        _TTS_LAST_TEXT_HASH = text_hash
        _TTS_LAST_ACCEPTED_TS = now
    if not _start_tts_worker():
        return False
    lang = str(current_settings.get("language", "en") or "en")
    runtime_plan = _resolve_tts_runtime_plan(text=text, lang=lang, explicit_engine=engine_pref)
    chosen_engine = _normalize_tts_engine_name(runtime_plan.get("engine") or "pyttsx3") or "pyttsx3"
    if chosen_engine not in _SUPPORTED_TTS_ENGINES:
        chosen_engine = "pyttsx3"
    ev: Optional[threading.Event] = threading.Event() if blocking else None
    result: Dict[str, Any] = {"ok": False, "accepted": True}
    task = _TTSTask(text=str(text).strip(), blocking_event=ev, emotion=(emotion or str(current_settings.get("emotion", "neutral"))), engine_pref=chosen_engine, voice_profile=str(current_settings.get("voice_profile", active_voice_profile or "Female")), lang=lang, selected_repo=runtime_plan.get("selected_repo"), fallback_repos=list(runtime_plan.get("fallback_repos") or []), backend_source=str(runtime_plan.get("backend_source") or "none"), task_id=uuid.uuid4().hex, result=result)
    _TTS_QUEUE.put(task)
    if ev is None:
        return True
    max_wait = float(getattr(config, "TTS_BLOCKING_MAX_SECONDS", 120.0) or 120.0)
    completed = ev.wait(timeout=max(1.0, min(600.0, max_wait)))
    if not completed:
        logger.warning("[Voice] TTS blocking wait timed out task_id=%s", task.task_id)
        return False
    return bool(result.get("ok"))

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
    normalized = _normalize_tts_engine_name(engine_name)
    if normalized not in _SUPPORTED_TTS_ENGINES:
        normalized = "pyttsx3"
    current_settings["tts_engine"] = normalized


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
    # SAFE_MODE disables microphone capture. LOCAL_ONLY_MODE must not disable the
    # local microphone; STT is a local hardware/runtime function, not a cloud call.
    if SAFE_MODE:
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


def listen_and_process(timeout: Optional[float] = None, phrase_time_limit: Optional[float] = None, retry_count: Optional[int] = None) -> Optional[str]:
    if SAFE_MODE:
        return None
    mic_obj = initialize_microphone()
    if mic_obj is None or sr is None or recognizer is None:
        return None

    timeout = float(timeout if timeout is not None else getattr(config, "VOICE_STT_TIMEOUT_SECONDS", 5.0))
    phrase_time_limit = float(phrase_time_limit if phrase_time_limit is not None else getattr(config, "VOICE_STT_PHRASE_LIMIT_SECONDS", 12.0))
    retries = max(1, min(4, int(retry_count if retry_count is not None else getattr(config, "VOICE_STT_RETRY_COUNT", 2))))

    for attempt in range(retries):
        try:
            with mic_obj as source:
                try:
                    recognizer.pause_threshold = float(getattr(config, "VOICE_PAUSE_THRESHOLD", 0.95) or 0.95)
                    recognizer.non_speaking_duration = float(getattr(config, "VOICE_NON_SPEAKING_DURATION", 0.45) or 0.45)
                    recognizer.phrase_threshold = float(getattr(config, "VOICE_PHRASE_THRESHOLD", 0.25) or 0.25)
                    recognizer.adjust_for_ambient_noise(source, duration=float(getattr(config, "VOICE_AMBIENT_ADJUST_SECONDS", 0.25) or 0.25))
                except Exception:
                    pass
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            txt = _recognize_chunk(audio)
            if txt:
                log_voice_event("Voice Input Recognized", txt)
                return txt
            log_voice_event("Voice Input Empty", f"No recognized text on attempt {attempt + 1}/{retries}.")
        except Exception as exc:
            name = type(exc).__name__
            log_voice_event("Voice Input Retry", f"{name}: attempt {attempt + 1}/{retries}")
            # SpeechRecognition may raise WaitTimeoutError/UnknownValueError; both
            # should fail soft and immediately allow the UI/backend loop to re-listen.
            continue
    return None


def transcribe_once(timeout: float = 10.0, phrase_time_limit: Optional[float] = None) -> str:
    txt = listen_and_process(timeout=timeout, phrase_time_limit=phrase_time_limit)
    return txt or ""

# =============================================================================
# TTS SHUTDOWN
# =============================================================================
def shutdown_tts() -> None:
    """Cancel queued speech and stop/join the TTS worker cleanly."""
    global _TTS_WORKER_THREAD, _TTS_WORKER_STARTED
    if _TTS_SHUTDOWN_FLAG.is_set():
        return
    _TTS_SHUTDOWN_FLAG.set()
    stop_speaking(clear_queue=True)
    try:
        _TTS_QUEUE.put_nowait(None)
    except Exception:
        pass
    thread = _TTS_WORKER_THREAD
    join_seconds = float(getattr(config, "TTS_SHUTDOWN_JOIN_SECONDS", 4.0) or 4.0)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=max(0.5, min(15.0, join_seconds)))
    try:
        if _engine is not None:
            _engine.stop()  # type: ignore
    except Exception:
        pass
    _reset_pyttsx3_engine()
    with _TTS_WORKER_LOCK:
        _TTS_WORKER_STARTED = False
        _TTS_WORKER_THREAD = None

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

# ====================================================================
# END OF SarahMemoryVoice.py v9.0.0
# ====================================================================
