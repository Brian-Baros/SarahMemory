"""--==The SarahMemory Project==--
File: SarahMemoryAvatarPanel.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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

===============================================================================

AVATAR PANEL - MULTIFUNCTIONAL DISPLAY MODULE
=========================================================

This panel is located below the local cam panel in the WebUI and serves
multiple critical functions:

LOCAL MODE:
- Mirrored desktop display
- 2D avatar animation
- 3D avatar animation (when supported)
- Switch between modes seamlessly

CLOUD/WEB MODE:
- Animated avatar display (2D/3D)
- Video conference when contact is made via WebUI Contact list or keypad
- Auto-switch back to avatar when video conference ends

DESKTOP MODE:
- Mirror desktop capability
- Avatar switching (2D/3D)
- Video conference integration

MEDIA DISPLAY:
- Show AI-generated images
- Show AI-generated videos
- Preview content in the panel

PANEL CONTROLS:
- Enlarge/Maximize
- Pop out to separate window
- Resize (drag corners/edges)
- Minimize back to inline

PLATFORM SUPPORT:
- Windows (Command Prompt via python SarahMemoryMain.py)
- Linux (Desktop and Headless)
- PythonAnywhere (Headless cloud hosting)
- WebUI accessible via https://ai.sarahmemory.com
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "ui_panel"
# CATEGORY = "avatar_display"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "custom_ui"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "display_camera_media"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "avatar_panel"
# FAMILY = "avatar"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = True
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Multifunction avatar/media/conference/display panel for WebUI/Desktop with 2D/3D avatar, desktop mirror, conference mode, and media preview."
# --- SARAHMETA END ---

import sys
import os
import logging
import threading
import time
import json
import base64
import math
import copy
import uuid
import io
import re
import queue
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

MODULE_VERSION = "9.1.0a"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("SarahMemoryAvatarPanel")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ============================================================================
# GLOBAL CONFIGURATION IMPORTS
# ============================================================================
try:
    import SarahMemoryGlobals as config
    DATASETS_DIR = config.DATASETS_DIR
    AVATAR_DIR = getattr(config, 'AVATAR_DIR', os.path.join(config.BASE_DIR, 'resources', 'avatars'))
    AVATAR_3D_DIR = getattr(config, 'AVATAR_3D_DIR', os.path.join(config.BASE_DIR, 'resources', 'avatars', '3D'))
    RUN_MODE = getattr(config, 'RUN_MODE', 'local')
    DEVICE_MODE = getattr(config, 'DEVICE_MODE', 'local_agent')
    DEVICE_PROFILE = getattr(config, 'DEVICE_PROFILE', 'Standard')
    ENABLE_AVATAR_PANEL = getattr(config, 'ENABLE_AVATAR_PANEL', True)
except ImportError:
    logger.warning("SarahMemoryGlobals not found; using defaults")
    _fallback_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if os.path.basename(os.path.dirname(__file__)).lower() == "core" else os.path.dirname(__file__)
    DATASETS_DIR = os.path.join(_fallback_root, 'data', 'memory', 'datasets')
    AVATAR_DIR = os.path.join(_fallback_root, 'resources', 'avatars')
    AVATAR_3D_DIR = os.path.join(_fallback_root, 'resources', 'avatars', '3D')
    RUN_MODE = 'local'
    DEVICE_MODE = 'local_agent'
    DEVICE_PROFILE = 'Standard'
    ENABLE_AVATAR_PANEL = True
    config = None

# ============================================================================
# OPTIONAL DEPENDENCY DETECTION
# ============================================================================
# PyQt5 for GUI mode
HAVE_QT = False
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QSlider, QGroupBox, QFrame,
        QStackedWidget, QSizePolicy, QDialog, QToolBar, QAction,
        QSplitter, QMenu, QMessageBox, QFileDialog, QProgressBar
    )
    from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QUrl, QByteArray
    from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QFont, QColor
    HAVE_QT = True
except ImportError:
    HAVE_QT = False
    logger.info("PyQt5 not available; GUI features disabled")

# OpenCV for video/webcam
HAVE_CV2 = False
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    logger.info("OpenCV not available; video features limited")

# PIL for image processing
HAVE_PIL = False
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except ImportError:
    logger.info("PIL not available; image features limited")

# Screen capture for desktop mirror
HAVE_MSS = False
try:
    import mss
    HAVE_MSS = True
except ImportError:
    logger.info("mss not available; desktop mirror limited")

# Optional integrations
try:
    import SarahMemoryVoice as Voice
except ImportError:
    Voice = None

try:
    import SarahMemoryAvatar as Avatar
except ImportError:
    Avatar = None

# Enterprise presentation boundaries: never grant device, network, or hardware authority.
_MAX_IMAGE_SIDE = 2048
_MAX_RENDER_PIXELS = 2048 * 2048
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024
_MAX_MEDIA_BYTES = 512 * 1024 * 1024
_MAX_MEDIA_QUEUE = 64
_PANEL_SINGLETON_LOCK = threading.RLock()


def _panel_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False,
                'on': True, 'off': False}.get(value.strip().lower(), default)
    return default


def _panel_number(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return max(low, min(high, number))
    except (ValueError, TypeError, OverflowError):
        return default


def _panel_dimension(value: Any, *, low: int = 1, high: int = _MAX_IMAGE_SIDE) -> int:
    if isinstance(value, bool):
        raise ValueError('invalid_dimension')
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('invalid_dimension') from None
    if not low <= number <= high:
        raise ValueError('dimension_out_of_bounds')
    return number


def _panel_is_local() -> bool:
    return str(RUN_MODE).lower() != 'cloud' and str(DEVICE_MODE).lower() != 'public_web'


def _panel_approved_media_path(path: str, *, kind: str = 'image') -> Optional[str]:
    if not isinstance(path, str) or not path or len(path) > 4096 or '://' in path or '\x00' in path:
        return None
    try:
        original = Path(path).expanduser()
        resolved = original.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink() or original.is_symlink():
            return None
        roots = [Path(DATASETS_DIR), Path(AVATAR_DIR)]
        if config is not None:
            for key in ('DATA_DIR', 'CANVAS_DIR', 'RESOURCES_DIR'):
                value = getattr(config, key, None)
                if value:
                    roots.append(Path(value))
        if not any(resolved.is_relative_to(root.expanduser().resolve()) for root in roots):
            return None
        if resolved.stat().st_size > _MAX_MEDIA_BYTES:
            return None
        suffix = resolved.suffix.lower()
        if kind == 'image' and suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}:
            return None
        if kind == 'video' and suffix not in {'.mp4', '.webm', '.mov', '.mkv', '.avi'}:
            return None
        return str(resolved)
    except (OSError, ValueError, RuntimeError):
        return None


def _panel_verify_image(path: str) -> bool:
    if not HAVE_PIL:
        return False
    try:
        with Image.open(path) as img:
            if img.width <= 0 or img.height <= 0 or img.width * img.height > _MAX_RENDER_PIXELS:
                return False
            img.verify()
        return True
    except (OSError, ValueError, SyntaxError, MemoryError):
        return False


def _panel_authorize_capture(kind: str, *, local_ui_consent: bool = False) -> bool:
    # Only a local human UI gesture can supply consent in this module.
    # API-provided arbitrary flags are never treated as user approval.
    if not (_panel_is_local() and local_ui_consent):
        return False
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
        verdict = authorize_external_domain_action(
            f'Use local {kind} presentation capture',
            origin='SarahMemoryAvatarPanel.local_ui',
            proposed_action={
                'action_type': f'avatar_panel.{kind}.capture',
                'target': 'local_display' if kind == 'desktop' else 'local_camera',
                'capability_name': 'display.capture' if kind == 'desktop' else 'camera.capture',
                'risk_level': 'TIER_3_PRIVILEGED_SYSTEM',
                'required_permissions': [f'{kind}.capture'],
                'verification_checks': [{'name': 'capture_stopped_on_exit', 'required': True}],
                'rollback_plan': [{'name': 'release_capture_device', 'required': True}],
            },
            meta={'source': 'SarahMemoryAvatarPanel', 'execution_authority': False},
            user_consented=True,
        )
        return bool(isinstance(verdict, dict) and verdict.get('allow') is True
                    and verdict.get('domain_execution_authorized') is True
                    and str(verdict.get('decision', '')).upper() == 'ALLOW')
    except Exception as exc:
        logger.warning('Capture authorization unavailable (%s): %s', kind, type(exc).__name__)
        return False


def _panel_verified_glb(path: str, root: str) -> bool:
    # The enterprise AvatarBuilder is the existing geometry/asset verification owner.
    try:
        resolved_root = Path(root).resolve()
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            return False
        import SarahMemoryAvatarBuilder as builder
        if str(getattr(builder, 'MODULE_VERSION', '')) != '9.1.0a':
            return False
        inspector = getattr(builder, '_read_glb_header', None)
        if not callable(inspector):
            return False
        evidence = inspector(str(resolved))
        return bool(isinstance(evidence, dict) and evidence.get('valid_glb_structure') is True
                    and evidence.get('has_mesh') is True)
    except Exception:
        return False


# ============================================================================
# ENUMERATIONS FOR PANEL MODES
# ============================================================================
class PanelMode(Enum):
    """
    Enumeration of all supported panel display modes.
    """
    AVATAR_2D = auto()          # 2D animated avatar display
    AVATAR_3D = auto()          # 3D animated avatar display
    DESKTOP_MIRROR = auto()     # Mirrored desktop display
    VIDEO_CONFERENCE = auto()   # Video conference with remote peer
    MEDIA_IMAGE = auto()        # Display AI-generated image
    MEDIA_VIDEO = auto()        # Display AI-generated video
    IDLE = auto()               # Idle/standby state

class ConferenceState(Enum):
    """
    Video conference connection states.
    """
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()

class PanelSize(Enum):
    """
    Panel size presets for different display modes.
    """
    COMPACT = (320, 240)
    STANDARD = (480, 360)
    LARGE = (640, 480)
    FULLSCREEN = (0, 0)  # Calculated at runtime

# ============================================================================
# AVATAR PANEL STATE MANAGER
# ============================================================================
class AvatarPanelState:
    """
    Centralized state management for the Avatar Panel.
    Tracks current mode, conference state, media queue, and preferences.
    Thread-safe using locks for concurrent access.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Core state
        self._mode: PanelMode = PanelMode.AVATAR_2D
        self._previous_mode: PanelMode = PanelMode.AVATAR_2D
        self._conference_state: ConferenceState = ConferenceState.DISCONNECTED
        
        # Avatar state
        self._current_emotion: str = "neutral"
        self._avatar_type: str = "2d"  # "2d" or "3d"
        self._avatar_zoom: float = 1.0
        self._avatar_animation_active: bool = True
        self._avatar_speaking: bool = False
        self._avatar_listening: bool = False
        self._avatar_thinking: bool = False
        self._avatar_busy: bool = False
        self._avatar_diagnostics: bool = False
        self._avatar_life_state: str = "boot_greeting"
        # Avatar Organ 3D runtime state. This is visual/interface state only;
        # future physical humanoid execution must be mediated by a separate
        # governed Avatar-to-MSDC bridge, not this panel.
        self._avatar_3d_loader_state: str = "2D_MODE"
        self._avatar_3d_skin: str = "default"
        self._avatar_3d_model_file: str = "SarahMemoryAvatar_RigBootstrap.glb"
        self._avatar_3d_quality: str = "cinematic_2m"
        self._avatar_3d_fps_cap: int = 60
        self._avatar_body_profile: Dict[str, Any] = {
            "profile_name": "SarahMemory_default_humanoid",
            "height_m": 1.68,
            "weight_kg_visual_target": 58.0,
            "head_height_ratio": 0.132,
            "shoulder_width_ratio": 0.245,
            "hip_width_ratio": 0.190,
            "leg_length_ratio": 0.515,
            "arm_span_ratio": 1.0,
            "center_of_mass_y_ratio": 0.55,
            "rig_units": "meters",
            "biological_plausibility": "visual_anatomy_reference_only",
            "robot_mapping_status": "future_bridge_required_no_physical_actuation_here",
        }
        self._last_interaction_at: float = time.time()
        
        # Conference state
        self._remote_peer: Optional[str] = None
        self._local_stream_active: bool = False
        self._remote_stream_active: bool = False
        self._call_start_time: Optional[float] = None
        
        # Media state
        self._media_queue: queue.Queue = queue.Queue()
        self._current_media_path: Optional[str] = None
        self._media_playback_active: bool = False
        
        # Panel state
        self._is_popped_out: bool = False
        self._is_maximized: bool = False
        self._panel_size: Tuple[int, int] = PanelSize.STANDARD.value
        self._custom_size: Optional[Tuple[int, int]] = None
        
        # Event callbacks
        self._mode_change_callbacks: List[Callable] = []
        self._state_change_callbacks: List[Callable] = []
        
        logger.info("AvatarPanelState initialized")
    
    # -------------------------------------------------------------------------
    # Mode Management
    # -------------------------------------------------------------------------
    def get_mode(self) -> PanelMode:
        """Get current panel mode."""
        with self._lock:
            return self._mode
    
    def set_mode(self, mode: PanelMode, save_previous: bool = True) -> None:
        """Change mode once; publish callbacks outside the state lock."""
        if not isinstance(mode, PanelMode):
            raise ValueError('unsupported_panel_mode')
        with self._lock:
            old_mode = self._mode
            if old_mode == mode:
                return
            if save_previous:
                self._previous_mode = old_mode
            self._mode = mode
            callbacks = tuple(self._mode_change_callbacks[:64])
        logger.info('Panel mode changed: %s -> %s', old_mode.name, mode.name)
        for callback in callbacks:
            try:
                callback(old_mode, mode)
            except Exception as exc:
                logger.warning('Mode change callback error: %s', exc)
    
    def restore_previous_mode(self) -> None:
        with self._lock:
            previous = self._previous_mode
        self.set_mode(previous, save_previous=False)
    
    def _notify_mode_change(self, old_mode: PanelMode, new_mode: PanelMode) -> None:
        """Notify all registered callbacks of mode change."""
        for callback in self._mode_change_callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                logger.warning(f"Mode change callback error: {e}")
    
    def register_mode_callback(self, callback: Callable) -> None:
        """Register a callback for mode changes."""
        with self._lock:
            if callback not in self._mode_change_callbacks:
                self._mode_change_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # Avatar State
    # -------------------------------------------------------------------------
    def get_emotion(self) -> str:
        """Get current avatar emotion."""
        with self._lock:
            return self._current_emotion

    def set_emotion(self, emotion: str) -> None:
        with self._lock:
            clean = str(emotion or 'neutral').strip().lower()[:64]
            self._current_emotion = clean if clean in AvatarAnimationEngine.EMOTION_EXPRESSIONS else 'neutral'
    
    def get_avatar_type(self) -> str:
        """Get avatar type (2d/3d)."""
        with self._lock:
            return self._avatar_type
    
    def set_avatar_type(self, avatar_type: str) -> None:
        kind = str(avatar_type or '').strip().lower()
        if kind not in {'2d', '3d'}:
            raise ValueError('unsupported_avatar_type')
        with self._lock:
            self._avatar_type = kind
            self._avatar_3d_loader_state = '3D_LOADING' if kind == '3d' else '2D_MODE'
        self.set_mode(PanelMode.AVATAR_3D if kind == '3d' else PanelMode.AVATAR_2D)

    def get_avatar_3d_dir(self) -> str:
        """Return canonical 3D Avatar Organ asset directory."""
        try:
            return str(Path(AVATAR_3D_DIR).expanduser().resolve())
        except Exception:
            return os.path.join(AVATAR_DIR, "3D")

    def get_avatar_3d_model_path(self) -> str:
        with self._lock:
            model_file = os.path.basename(str(self._avatar_3d_model_file or 'SarahMemoryAvatar_RigBootstrap.glb'))
            skin = str(self._avatar_3d_skin or 'default').strip()
        base = self.get_avatar_3d_dir()
        options = [os.path.join(base, model_file)]
        if skin and skin.lower() != 'default' and skin not in {'.', '..'} and '/' not in skin and '\\' not in skin:
            options.append(os.path.join(base, skin, model_file))
        options += [os.path.join(base, name) for name in ('SarahMemoryAvatar_RigBootstrap.glb', 'sarahmemory_3d_avatar.glb', 'sarahmemory_happy_face_ball.glb')]
        for candidate in options:
            if _panel_verified_glb(candidate, base):
                return candidate
        return options[0]

    def get_avatar_3d_spec(self) -> Dict[str, Any]:
        """Return the frontend 3D avatar spec packet used by the V9 Avatar Panel."""
        with self._lock:
            model_path = self.get_avatar_3d_model_path()
            model_file = os.path.basename(model_path)
            exists = _panel_verified_glb(model_path, self.get_avatar_3d_dir())
            loader_state = "3D_READY_VALIDATED_MESH" if exists else "3D_FAILED_FALLBACK_GOLD_REFERENCE"
            if self._avatar_type == "3d" and self._avatar_3d_loader_state == "3D_LOADING" and exists:
                loader_state = "3D_READY_VALIDATED_MESH"
                self._avatar_3d_loader_state = loader_state
            return {
                "renderMode": "gltf_model" if exists else "gold_standard_avatar",
                "modelUrl": f"/api/avatar/3d/{model_file}" if exists else "",
                "modelFile": model_file if exists else "",
                "backgroundType": "none",
                "pose": "stand",
                "gesture": "none",
                "lookAt": {"x": 0, "y": 1.45, "z": 0},
                "expression": self._current_emotion,
                "speaking": self._avatar_speaking,
                "listening": self._avatar_listening,
                "thinking": self._avatar_thinking,
                "loaderState": loader_state,
                "quality": self._avatar_3d_quality,
                "fpsCap": self._avatar_3d_fps_cap,
                "lightingProfile": "high_end",
                "shadowQuality": "high",
                "animationAuthority": "visual_only_no_msdc_no_operator_action",
                "avatarEyeCameraAnchor": "Sarah_AvatarEye_Center",
                "stageOffsetY": -0.08,
                "avatarOffsetY": -0.08,
                "useRuntimeStage": True,
                "materialMode": "high_end",
                "runtimeVisualPriority": "gltf_model",
                "meshFallbackUrl": f"/api/avatar/3d/{model_file}" if exists else "",
                "forceMeshRuntime": bool(exists),
                "goldReferenceOnly": not exists,
                "authoringPolyTarget": 12000000,
                "runtimeLod": "configured_unmeasured",
                "productionAssetContract": "SARAHMEMORY_GOLDSTANDARD_EMBODIED_ENTITY_GLB_V1",
                "blueprintConstructReady": bool(exists),
                "visualEffectsMode": "secondary",
                "goldStandardScale": 1.0,
                "goldStandardYOffset": 0,
        "goldStandardPanelBottomPx": 58,
        "goldStandardPanelHeightPct": 92,
                "source": "resources/avatars/3D",
                "bodyProfile": dict(self._avatar_body_profile),
                "robotBridge": {
                    "enabled": False,
                    "future_owner": "AvatarToMSDCBridge",
                    "physical_actuation_allowed_here": False,
                },
            }
    
    def get_zoom(self) -> float:
        """Get avatar zoom level."""
        with self._lock:
            return self._avatar_zoom
    
    def set_zoom(self, zoom: float) -> None:
        with self._lock:
            self._avatar_zoom = _panel_number(zoom, self._avatar_zoom, 0.25, 3.0)

    def set_speaking(self, speaking: bool) -> None:
        """Set live speaking flag for WebUI/avatar API synchronization."""
        with self._lock:
            self._avatar_speaking = _panel_bool(speaking)
            if self._avatar_speaking:
                self._avatar_listening = False
                self._avatar_busy = False
                self._avatar_diagnostics = False
                self._avatar_life_state = "speaking"
                self._last_interaction_at = time.time()
            elif self._avatar_life_state == "speaking":
                self._avatar_life_state = "ready"

    def get_speaking(self) -> bool:
        """Return live speaking flag."""
        with self._lock:
            return bool(self._avatar_speaking)

    def set_listening(self, listening: bool) -> None:
        """Set live listening flag for WebUI/avatar API synchronization."""
        with self._lock:
            self._avatar_listening = _panel_bool(listening)
            if self._avatar_listening:
                self._avatar_speaking = False
                self._avatar_busy = False
                self._avatar_diagnostics = False
                self._avatar_life_state = "listening"
                self._last_interaction_at = time.time()
            elif self._avatar_life_state == "listening":
                self._avatar_life_state = "ready"

    def get_listening(self) -> bool:
        """Return live listening flag."""
        with self._lock:
            return bool(self._avatar_listening)

    def set_thinking(self, thinking: bool) -> None:
        """Set live thinking flag for diagnostics/life-cycle sync."""
        with self._lock:
            self._avatar_thinking = _panel_bool(thinking)
            if self._avatar_thinking:
                self._avatar_life_state = "thinking"
                self._last_interaction_at = time.time()

    def set_busy(self, busy: bool) -> None:
        """Set live busy flag for diagnostics/life-cycle sync."""
        with self._lock:
            self._avatar_busy = _panel_bool(busy)
            if self._avatar_busy:
                self._avatar_life_state = "busy"
                self._last_interaction_at = time.time()

    def set_diagnostics(self, diagnostics: bool) -> None:
        """Set live diagnostics flag for diagnostics/life-cycle sync."""
        with self._lock:
            self._avatar_diagnostics = _panel_bool(diagnostics)
            if self._avatar_diagnostics:
                self._avatar_life_state = "diagnostics"
                self._last_interaction_at = time.time()

    def set_life_state(self, life_state: str) -> None:
        """Set avatar life-cycle state without changing panel mode."""
        with self._lock:
            self._avatar_life_state = str(life_state or "ready").strip() or "ready"
            self._last_interaction_at = time.time()

    
    # -------------------------------------------------------------------------
    # Conference State
    # -------------------------------------------------------------------------
    def get_conference_state(self) -> ConferenceState:
        """Get current conference state."""
        with self._lock:
            return self._conference_state
    
    def set_conference_state(self, state: ConferenceState) -> None:
        """Set conference state and handle mode transitions."""
        with self._lock:
            old_state = self._conference_state
            self._conference_state = state
            
            if state == ConferenceState.CONNECTED:
                self.set_mode(PanelMode.VIDEO_CONFERENCE)
                self._call_start_time = time.time()
            elif state == ConferenceState.DISCONNECTED and old_state == ConferenceState.CONNECTED:
                # Auto-restore previous mode when conference ends
                self.restore_previous_mode()
                self._call_start_time = None
            
            logger.info(f"Conference state: {old_state.name} -> {state.name}")
    
    def get_remote_peer(self) -> Optional[str]:
        """Get remote peer identifier."""
        with self._lock:
            return self._remote_peer
    
    def set_remote_peer(self, peer: Optional[str]) -> None:
        """Set remote peer identifier."""
        with self._lock:
            self._remote_peer = peer
    
    def get_call_duration(self) -> float:
        """Get current call duration in seconds."""
        with self._lock:
            if self._call_start_time:
                return time.time() - self._call_start_time
            return 0.0
    
    # -------------------------------------------------------------------------
    # Media State
    # -------------------------------------------------------------------------
    def queue_media(self, media_path: str, media_type: str = "image") -> None:
        if media_type not in {'image', 'video'}:
            raise ValueError('unsupported_media_type')
        verified = _panel_approved_media_path(media_path, kind=media_type)
        if not verified:
            raise ValueError('unapproved_media_path')
        with self._lock:
            if self._media_queue.qsize() >= _MAX_MEDIA_QUEUE:
                raise OverflowError('media_queue_full')
            self._media_queue.put_nowait((verified, media_type))
    
    def get_next_media(self) -> Optional[Tuple[str, str]]:
        """Get next media item from queue."""
        try:
            return self._media_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_current_media(self, path: Optional[str]) -> None:
        """Set currently displayed media path."""
        with self._lock:
            self._current_media_path = path
    
    def get_current_media(self) -> Optional[str]:
        """Get currently displayed media path."""
        with self._lock:
            return self._current_media_path
    
    # -------------------------------------------------------------------------
    # Panel State
    # -------------------------------------------------------------------------
    def is_popped_out(self) -> bool:
        """Check if panel is in popped-out window."""
        with self._lock:
            return self._is_popped_out
    
    def set_popped_out(self, popped: bool) -> None:
        """Set popped-out state."""
        with self._lock:
            self._is_popped_out = popped
    
    def is_maximized(self) -> bool:
        """Check if panel is maximized."""
        with self._lock:
            return self._is_maximized
    
    def set_maximized(self, maximized: bool) -> None:
        """Set maximized state."""
        with self._lock:
            self._is_maximized = maximized
    
    def get_panel_size(self) -> Tuple[int, int]:
        """Get current panel size."""
        with self._lock:
            if self._custom_size:
                return self._custom_size
            return self._panel_size
    
    def set_panel_size(self, width: int, height: int) -> None:
        width = _panel_dimension(width, low=64, high=8192)
        height = _panel_dimension(height, low=64, high=8192)
        with self._lock:
            self._custom_size = (width, height)
    
    def reset_panel_size(self) -> None:
        """Reset to default panel size."""
        with self._lock:
            self._custom_size = None
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for WebUI/API communication."""
        with self._lock:
            return {
                "mode": self._mode.name,
                "previous_mode": self._previous_mode.name,
                "conference_state": self._conference_state.name,
                "emotion": self._current_emotion,
                "avatar_type": self._avatar_type,
                "avatar_zoom": self._avatar_zoom,
                "avatar_animation_active": self._avatar_animation_active,
                "speaking": self._avatar_speaking,
                "listening": self._avatar_listening,
                "thinking": self._avatar_thinking,
                "busy": self._avatar_busy,
                "diagnostics": self._avatar_diagnostics,
                "life_state": self._avatar_life_state,
                "avatar_3d_loader_state": self._avatar_3d_loader_state,
                "avatar_3d_skin": self._avatar_3d_skin,
                "avatar_3d_model_file": self._avatar_3d_model_file,
                "avatar_3d_quality": self._avatar_3d_quality,
                "avatar_3d_fps_cap": self._avatar_3d_fps_cap,
                "body_profile": dict(self._avatar_body_profile),
                "spec": self.get_avatar_3d_spec(),
                "last_interaction_at": self._last_interaction_at,
                "remote_peer": self._remote_peer,
                "local_stream_active": self._local_stream_active,
                "remote_stream_active": self._remote_stream_active,
                "call_duration": self.get_call_duration(),
                "current_media": self._current_media_path,
                "media_playback_active": self._media_playback_active,
                "is_popped_out": self._is_popped_out,
                "is_maximized": self._is_maximized,
                "panel_size": self.get_panel_size()
            }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Validate before applying any restored presentation state."""
        if not isinstance(data, dict):
            raise ValueError('panel_state_must_be_dict')
        with self._lock:
            if 'mode' in data:
                mode_text = str(data['mode']).upper()
                if mode_text not in PanelMode.__members__:
                    raise ValueError('unsupported_panel_mode')
            if 'avatar_type' in data and str(data['avatar_type']).lower() not in {'2d', '3d'}:
                raise ValueError('unsupported_avatar_type')
            if 'avatar_3d_model_file' in data:
                model_filename = os.path.basename(str(data['avatar_3d_model_file'] or ''))
                if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}\.glb', model_filename, flags=re.I):
                    raise ValueError('unsupported_3d_model_name')
            if 'mode' in data:
                self._mode = PanelMode[str(data['mode']).upper()]
            if 'emotion' in data:
                self.set_emotion(data['emotion'])
            if 'avatar_type' in data:
                self._avatar_type = str(data['avatar_type']).lower()
            if 'avatar_zoom' in data:
                self.set_zoom(data['avatar_zoom'])
            for field_name, attr in (('speaking','_avatar_speaking'), ('listening','_avatar_listening'),
                ('thinking','_avatar_thinking'), ('busy','_avatar_busy'), ('diagnostics','_avatar_diagnostics')):
                if field_name in data:
                    setattr(self, attr, _panel_bool(data[field_name]))
            if 'life_state' in data:
                self._avatar_life_state = str(data['life_state'] or 'ready').strip()[:64]
            if 'avatar_3d_skin' in data:
                skin = str(data['avatar_3d_skin'] or 'default').strip()[:64]
                if not skin or skin in {'.', '..'} or '/' in skin or '\\' in skin:
                    skin = 'default'
                self._avatar_3d_skin = skin
            if 'avatar_3d_model_file' in data:
                model_file = os.path.basename(str(data['avatar_3d_model_file'] or ''))
                if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}\.glb', model_file, flags=re.I):
                    raise ValueError('unsupported_3d_model_name')
                self._avatar_3d_model_file = model_file
            if 'avatar_3d_quality' in data:
                quality = str(data['avatar_3d_quality'] or 'cinematic_2m').lower()
                self._avatar_3d_quality = quality if quality in {'preview','balanced','high','ultra','cinematic_2m','goldstandard_entity'} else 'cinematic_2m'


# ============================================================================
# GLOBAL STATE INSTANCE
# ============================================================================
_panel_state: Optional[AvatarPanelState] = None

def get_panel_state() -> AvatarPanelState:
    """Get or create the global panel state instance."""
    global _panel_state
    with _PANEL_SINGLETON_LOCK:
        if _panel_state is None:
            _panel_state = AvatarPanelState()
        return _panel_state


# ============================================================================
# DESKTOP MIRROR CAPTURE THREAD
# ============================================================================
class DesktopMirrorThread(QThread if HAVE_QT else threading.Thread):
    """
    Background thread for capturing desktop screen frames.
    Used for desktop mirror mode in local operation.
    """
    
    if HAVE_QT:
        frame_ready = pyqtSignal(object)
    
    def __init__(self, fps: int = 15, monitor: int = 1):
        super().__init__()
        self._running = False
        self._fps = _panel_dimension(fps, low=1, high=30)
        self._monitor = max(1, min(8, int(monitor)))
        self._frame_interval = 1.0 / self._fps
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._capture_authorized = False
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for frame updates."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a frame callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def start_capture(self) -> None:
        if not HAVE_MSS or not _panel_is_local() or not self._capture_authorized:
            raise RuntimeError('desktop_capture_unavailable_or_not_authorized')
        if self.isRunning() if HAVE_QT else self.is_alive():
            return
        if getattr(self, '_started', None) is not None and not HAVE_QT and self._started.is_set():
            raise RuntimeError('desktop_capture_worker_must_be_recreated')
        self._running = True
        self.start()
    
    def stop_capture(self) -> None:
        """Stop desktop capture."""
        self._running = False
        if self.isRunning() if HAVE_QT else self.is_alive():
            self.wait() if HAVE_QT else self.join(timeout=2.0)
    
    def run(self) -> None:
        if not HAVE_MSS or not _panel_is_local():
            self._running = False
            return
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if len(monitors) <= 1:
                    return
                monitor = monitors[self._monitor] if self._monitor < len(monitors) else monitors[1]
                while self._running:
                    started = time.monotonic()
                    try:
                        shot = sct.grab(monitor)
                        frame = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX') if HAVE_PIL else shot
                        if HAVE_QT:
                            self.frame_ready.emit(frame)
                        else:
                            with self._lock:
                                callbacks = tuple(self._callbacks[:16])
                            for callback in callbacks:
                                try: callback(frame)
                                except Exception: logger.debug('Desktop capture callback failed')
                    except Exception as exc:
                        logger.warning('Desktop capture frame unavailable: %s', type(exc).__name__)
                        self._running = False
                        break
                    remaining = self._frame_interval - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self._running = False


# ============================================================================
# AVATAR ANIMATION ENGINE
# ============================================================================
class AvatarAnimationEngine:
    """
    Handles 2D and 3D avatar animation rendering.
    Supports emotion-based expressions, lip sync, and idle animations.
    """
    
    # Emotion to visual mapping
    EMOTION_EXPRESSIONS = {
        "joy": {"emoji": "😊", "color": "#FFD700", "animation": "bounce"},
        "trust": {"emoji": "🤝", "color": "#4169E1", "animation": "nod"},
        "surprise": {"emoji": "😲", "color": "#FF69B4", "animation": "jump"},
        "sadness": {"emoji": "😢", "color": "#4682B4", "animation": "droop"},
        "fear": {"emoji": "😨", "color": "#8B008B", "animation": "shake"},
        "anger": {"emoji": "😠", "color": "#FF4500", "animation": "pulse"},
        "neutral": {"emoji": "😐", "color": "#808080", "animation": "idle"},
        "thinking": {"emoji": "🤔", "color": "#9370DB", "animation": "think"}
    }
    
    def __init__(self):
        self._current_emotion = "neutral"
        self._animation_active = True
        self._lip_sync_active = False
        self._frame_index = 0
        self._animation_speed = 1.0
        self._loaded_sprites: Dict[str, Any] = {}
        self._animation_timer: Optional[threading.Timer] = None
        self._live_controller: Optional[Any] = None
        self._live_canvas: Optional[Any] = None
        self._live_last_health: Dict[str, Any] = {}
        self._live_init_attempted = False
        self._compute_plan: Dict[str, Any] = {}
        self._compute_plan_at = 0.0
        self._live_last_frame: Optional[Any] = None
        self._live_last_frame_at = 0.0
        self._live_last_frame_size: Tuple[int, int] = (0, 0)
        
        logger.info("AvatarAnimationEngine initialized")

    def _adaptive_render_policy(self) -> Dict[str, Any]:
        """Return a cached, bounded avatar plan without mutating hardware settings."""
        now = time.monotonic()
        if self._compute_plan and (now - self._compute_plan_at) < 5.0:
            return self._compute_plan
        fallback = {
            "ok": False,
            "mode": "SAFE_DISCOVERY",
            "machine_class": "unknown",
            "workloads": {"avatar": {"fps_cap": 12, "render_scale": 0.50, "temporal_history": False, "neon_effects": False}},
            "execution_authority": False,
        }
        try:
            import SarahMemoryAdaptive as adaptive  # type: ignore
            build = getattr(adaptive, "build_compute_passport", None)
            select = getattr(adaptive, "select_adaptive_compute_plan", None)
            if callable(build) and callable(select):
                passport = build(context={"source": "SarahMemoryAvatarPanel", "workload": "avatar"})
                candidate = select("avatar", passport=passport, context={"source": "SarahMemoryAvatarPanel"})
                if isinstance(candidate, dict):
                    fallback = candidate
        except Exception as exc:
            logger.debug(f"Adaptive avatar plan unavailable; safe discovery budget retained: {exc}")
        self._compute_plan = fallback
        self._compute_plan_at = now
        return self._compute_plan

    def _ensure_live_renderer(self) -> bool:
        """Lazily connect presentation to Avatar -> Controller -> CanvasStudio."""
        if self._live_controller is not None and self._live_canvas is not None:
            return True
        if self._live_init_attempted:
            return False
        self._live_init_attempted = True
        try:
            from UnifiedAvatarController import UnifiedAvatarController  # type: ignore
            from SarahMemoryCanvasStudio import CanvasStudio  # type: ignore
            self._live_controller = UnifiedAvatarController()
            self._live_canvas = CanvasStudio()
            return True
        except Exception as exc:
            logger.debug(f"Persistent live avatar renderer unavailable; compatibility fallback retained: {exc}")
            self._live_controller = None
            self._live_canvas = None
            return False

    def _render_persistent_live_frame(self, width: int, height: int) -> Optional[Any]:
        if not HAVE_PIL or not self._ensure_live_renderer():
            return None
        try:
            output_width = max(64, min(4096, int(width)))
            output_height = max(64, min(4096, int(height)))
            plan = self._adaptive_render_policy()
            avatar_budget = plan.get("workloads", {}).get("avatar", {}) if isinstance(plan.get("workloads"), dict) else {}
            fps_cap = max(1, min(60, int(avatar_budget.get("fps_cap") or 12)))
            render_scale = max(0.25, min(1.0, float(avatar_budget.get("render_scale") or 0.50)))
            use_temporal = bool(avatar_budget.get("temporal_history", False))
            now = time.monotonic()
            if (
                self._live_last_frame is not None
                and self._live_last_frame_size == (output_width, output_height)
                and (now - self._live_last_frame_at) < (1.0 / fps_cap)
            ):
                return self._live_last_frame.copy()

            controller = self._live_controller
            canvas = self._live_canvas
            avatar = getattr(controller, "avatar", None)
            if avatar is not None and hasattr(avatar, "update_live_avatar_state"):
                avatar.update_live_avatar_state({
                    "expression": self._current_emotion,
                    "speaking": bool(self._lip_sync_active),
                    "speaking_energy": 0.45 if self._lip_sync_active else 0.0,
                    "mouth_shape": "speech" if self._lip_sync_active else "neutral",
                })
            packet = controller.build_live_avatar_state_packet(now_monotonic=now)
            if not isinstance(packet, dict) or not packet.get("ok"):
                return None
            render_width = max(64, int(round(output_width * render_scale)))
            render_height = max(64, int(round(output_height * render_scale)))
            rendered = canvas.render_live_avatar_frame(
                packet,
                width=render_width,
                height=render_height,
                use_temporal_history=use_temporal,
            )
            if not isinstance(rendered, dict) or not rendered.get("ok"):
                return None
            rgba = rendered.get("frame_rgba")
            if rgba is None:
                return None
            frame = Image.fromarray(rgba, mode="RGBA")
            if frame.size != (output_width, output_height):
                resampling = getattr(Image, "Resampling", Image)
                frame = frame.resize((output_width, output_height), getattr(resampling, "LANCZOS"))
            self._live_last_health = {
                **dict(rendered.get("render_health") or {}),
                "adaptive_compute": {
                    "mode": plan.get("mode"),
                    "machine_class": plan.get("machine_class"),
                    "fps_cap": fps_cap,
                    "render_scale": render_scale,
                    "temporal_history": use_temporal,
                    "execution_authority": False,
                },
            }
            self._frame_index = int(rendered.get("frame_id") or (self._frame_index + 1))
            self._live_last_frame = frame.copy()
            self._live_last_frame_at = now
            self._live_last_frame_size = (output_width, output_height)
            return frame
        except Exception as exc:
            logger.debug(f"Persistent live avatar frame failed; using compatibility fallback: {exc}")
            return None
    
    def set_emotion(self, emotion: str) -> None:
        """Set avatar emotion and trigger animation change."""
        emotion = emotion.lower()
        if emotion in self.EMOTION_EXPRESSIONS:
            self._current_emotion = emotion
            self._frame_index = 0
            logger.debug(f"Avatar emotion set: {emotion}")
        else:
            logger.warning(f"Unknown emotion: {emotion}, defaulting to neutral")
            self._current_emotion = "neutral"
    
    def get_emotion(self) -> str:
        """Get current emotion."""
        return self._current_emotion
    
    def get_expression_data(self) -> Dict[str, Any]:
        """Get current expression visual data."""
        return self.EMOTION_EXPRESSIONS.get(
            self._current_emotion,
            self.EMOTION_EXPRESSIONS["neutral"]
        )
    
    def start_lip_sync(self, duration: float = 0.0) -> None:
        seconds = _panel_number(duration, 0.0, 0.0, 300.0)
        self.stop_lip_sync()
        self._lip_sync_active = True
        if seconds > 0:
            timer = threading.Timer(seconds, self.stop_lip_sync)
            timer.daemon = True
            self._animation_timer = timer
            timer.start()
    
    def stop_lip_sync(self) -> None:
        """Stop lip sync animation."""
        self._lip_sync_active = False
        if self._animation_timer:
            self._animation_timer.cancel()
            self._animation_timer = None
        logger.debug("Lip sync stopped")
    
    def is_lip_syncing(self) -> bool:
        """Check if lip sync is active."""
        return self._lip_sync_active
    
    def render_2d_frame(self, width: int = 300, height: int = 300) -> Optional[Any]:
        """
        Render a 2D avatar frame.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            PIL Image or None if PIL unavailable
        """
        if not HAVE_PIL:
            return None
        try:
            width = _panel_dimension(width, low=16)
            height = _panel_dimension(height, low=16)
            if width * height > _MAX_RENDER_PIXELS:
                return None
        except ValueError:
            return None

        # Primary path: persistent AvatarState -> Rhythm -> Controller -> CanvasStudio.
        # Legacy drawing below is compatibility-only if the live renderer cannot initialize.
        live_frame = self._render_persistent_live_frame(width, height)
        if live_frame is not None:
            return live_frame
        
        try:
            # Compatibility fallback only; not the authoritative avatar state engine.
            # Create base image
            img = Image.new('RGBA', (width, height), (20, 26, 33, 255))
            draw = ImageDraw.Draw(img)
            
            # Get expression data
            expr_data = self.get_expression_data()
            
            # Draw avatar circle background
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 3
            
            # Parse color
            color_hex = expr_data["color"]
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw outer glow
            for i in range(20, 0, -1):
                alpha = int(255 * (i / 20) * 0.3)
                glow_color = (*color_rgb, alpha)
                draw.ellipse(
                    [center_x - radius - i*2, center_y - radius - i*2,
                     center_x + radius + i*2, center_y + radius + i*2],
                    fill=None, outline=glow_color, width=2
                )
            
            # Draw main circle
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill=color_rgb, outline=(255, 255, 255, 128)
            )
            
            # Draw emoji expression
            try:
                # Try to use a font that supports emojis
                font_paths = [
                    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                    "C:\\Windows\\Fonts\\seguiemj.ttf",
                    "/System/Library/Fonts/Apple Color Emoji.ttc"
                ]
                font = None
                for fp in font_paths:
                    if os.path.exists(fp):
                        try:
                            font = ImageFont.truetype(fp, radius)
                            break
                        except Exception:
                            continue
                
                if font is None:
                    font = ImageFont.load_default()
                
                emoji = expr_data["emoji"]
                # Calculate text position (centered)
                bbox = draw.textbbox((0, 0), emoji, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2 - bbox[1]
                
                draw.text((text_x, text_y), emoji, font=font, fill=(255, 255, 255))
            except Exception as e:
                logger.debug(f"Emoji render fallback: {e}")
                # Fallback: draw simple face
                draw.text((center_x - 20, center_y - 20), expr_data["emoji"],
                         fill=(255, 255, 255))
            
            # Add lip sync indicator
            if self._lip_sync_active:
                # Animate mouth opening/closing
                mouth_phase = (self._frame_index % 10) / 10
                mouth_height = int(10 + 15 * abs(0.5 - mouth_phase) * 2)
                draw.ellipse(
                    [center_x - 20, center_y + radius//2 - mouth_height//2,
                     center_x + 20, center_y + radius//2 + mouth_height//2],
                    fill=(30, 30, 30)
                )
            
            self._frame_index += 1
            return img
            
        except Exception as e:
            logger.error(f"2D frame render error: {e}")
            return None
    
    def get_3d_render_command(self) -> Dict[str, Any]:
        """
        Get command data for external 3D rendering engine (Blender/Unity/Unreal).
        
        Returns:
            Dictionary with render parameters
        """
        return {
            "emotion": self._current_emotion,
            "expression_data": self.get_expression_data(),
            "lip_sync": self._lip_sync_active,
            "frame_index": self._frame_index,
            "animation_speed": self._animation_speed,
            "persistent_live_renderer": bool(self._live_controller is not None and self._live_canvas is not None),
            "render_health": dict(self._live_last_health),
            "execution_authority": False,
        }


# ============================================================================
# VIDEO CONFERENCE CONTROLLER
# ============================================================================
class VideoConferenceController:
    """
    Manages video conference connections and streams.
    Handles WebRTC signaling, peer connections, and stream management.
    """
    
    def __init__(self, panel_state: AvatarPanelState):
        self._state = panel_state
        self._local_video_active = False
        self._remote_video_active = False
        self._audio_active = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_remote_frame": [],
            "on_error": []
        }
        self._connection_thread: Optional[threading.Thread] = None
        self._local_capture: Optional[Any] = None
        
        logger.info("VideoConferenceController initialized")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for conference events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Conference callback error ({event}): {e}")
    
    def start_call(self, peer_id: str, video: bool = True, audio: bool = True) -> bool:
        """Reject an unconfigured conference transport; never simulate CONNECTED."""
        self._state.set_conference_state(ConferenceState.FAILED)
        self._local_video_active = False
        self._remote_video_active = False
        self._audio_active = False
        self._emit('on_error', 'conference_transport_unavailable')
        return False
    
    def end_call(self) -> None:
        """End the current call."""
        logger.info("Ending call")
        
        self._stop_local_video()
        self._local_video_active = False
        self._remote_video_active = False
        self._audio_active = False
        
        peer = self._state.get_remote_peer()
        self._state.set_remote_peer(None)
        self._state.set_conference_state(ConferenceState.DISCONNECTED)
        
        self._emit("on_disconnect", peer)
    
    def answer_call(self, peer_id: str) -> bool:
        self._state.set_conference_state(ConferenceState.FAILED)
        self._emit('on_error', 'conference_transport_unavailable')
        return False
    
    def _start_local_video(self) -> None:
        """No implicit webcam activation from a call-state transition."""
        return None
    
    def _stop_local_video(self) -> None:
        """Stop local video capture."""
        if self._local_capture:
            try:
                self._local_capture.release()
            except Exception:
                pass
            self._local_capture = None
    
    def get_local_frame(self) -> Optional[Any]:
        """Get current local video frame."""
        if self._local_capture and self._local_capture.isOpened():
            ret, frame = self._local_capture.read()
            if ret:
                return frame
        return None
    
    def toggle_video(self) -> bool:
        # No active authenticated transport; do not activate camera.
        self._local_video_active = False
        return False
    
    def toggle_audio(self) -> bool:
        # No active authenticated transport; do not claim microphone activation.
        self._audio_active = False
        return False
    
    def is_in_call(self) -> bool:
        """Check if currently in a call."""
        return self._state.get_conference_state() == ConferenceState.CONNECTED
    
    def get_call_info(self) -> Dict[str, Any]:
        """Get current call information."""
        return {
            "peer": self._state.get_remote_peer(),
            "state": self._state.get_conference_state().name,
            "duration": self._state.get_call_duration(),
            "video_active": self._local_video_active,
            "audio_active": self._audio_active,
            "remote_video": self._remote_video_active
        }


# ============================================================================
# MEDIA DISPLAY CONTROLLER
# ============================================================================
class MediaDisplayController:
    """
    Handles display of AI-generated images and videos in the panel.
    """
    
    def __init__(self, panel_state: AvatarPanelState):
        self._state = panel_state
        self._video_capture: Optional[Any] = None
        self._video_playing = False
        self._playback_thread: Optional[threading.Thread] = None
        self._frame_callbacks: List[Callable] = []
        
        logger.info("MediaDisplayController initialized")
    
    def register_frame_callback(self, callback: Callable) -> None:
        """Register callback for video frame updates."""
        if callback not in self._frame_callbacks:
            self._frame_callbacks.append(callback)
    
    def display_image(self, image_path: str) -> bool:
        verified = _panel_approved_media_path(image_path, kind='image')
        if not verified or not _panel_verify_image(verified):
            return False
        self._state.set_current_media(verified)
        self._state.set_mode(PanelMode.MEDIA_IMAGE)
        return True
    
    def display_video(self, video_path: str, loop: bool = False) -> bool:
        verified = _panel_approved_media_path(video_path, kind='video')
        if not verified or not HAVE_CV2:
            return False
        self.stop_video()
        try:
            cap = cv2.VideoCapture(verified)
            if not cap.isOpened():
                cap.release()
                return False
            self._video_capture = cap
            self._video_playing = True
            self._state.set_current_media(verified)
            self._state.set_mode(PanelMode.MEDIA_VIDEO)
            self._playback_thread = threading.Thread(target=self._video_playback_loop, args=(bool(loop),), daemon=True)
            self._playback_thread.start()
            return True
        except Exception:
            self.stop_video()
            return False
    
    def _video_playback_loop(self, loop: bool) -> None:
        cap = self._video_capture
        try:
            if cap is None:
                return
            fps = _panel_number(cap.get(cv2.CAP_PROP_FPS), 30.0, 1.0, 60.0)
            frame_interval = 1.0 / fps
            while self._video_playing and self._video_capture is cap:
                started = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    if loop and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if not cap.grab():
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                for callback in tuple(self._frame_callbacks[:16]):
                    try: callback(frame)
                    except Exception: logger.debug('Video frame callback failed')
                time.sleep(max(0.0, frame_interval - (time.monotonic() - started)))
        finally:
            if self._video_capture is cap:
                self._video_playing = False
                self._video_capture = None
                try: cap.release()
                except Exception: pass
                if self._state.get_mode() == PanelMode.MEDIA_VIDEO:
                    self._state.restore_previous_mode()
    
    def stop_video(self) -> None:
        self._video_playing = False
        worker = self._playback_thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=1.5)
        if worker is None or not worker.is_alive():
            cap = self._video_capture
            self._video_capture = None
            if cap is not None:
                try: cap.release()
                except Exception: pass
            self._playback_thread = None
    
    def pause_video(self) -> None:
        self.stop_video()
    
    def resume_video(self) -> None:
        # A stopped stream has released its file handle; explicit display_video is required.
        return None
    
    def is_playing(self) -> bool:
        """Check if video is playing."""
        return self._video_playing
    
    def get_current_media_info(self) -> Dict[str, Any]:
        """Get information about current media."""
        media_path = self._state.get_current_media()
        if not media_path:
            return {"active": False}
        
        info = {
            "active": True,
            "path": media_path,
            "type": "video" if self._video_capture else "image"
        }
        
        if self._video_capture:
            info["playing"] = self._video_playing
            info["position"] = self._video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            info["total_frames"] = self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            info["fps"] = self._video_capture.get(cv2.CAP_PROP_FPS)
        
        return info


# ============================================================================
# AVATAR WINDOW (PyQt5 GUI)
# ============================================================================
if HAVE_QT:
    class AvatarWindow(QMainWindow):
        """
        Main Avatar Panel window with full GUI implementation.
        Supports all panel modes, controls, and features.
        """
        
        # Signals for thread-safe updates
        frame_updated = pyqtSignal(object)
        mode_changed = pyqtSignal(str)
        state_changed = pyqtSignal(dict)
        
        def __init__(
            self,
            fullscreen: bool = False,
            force_fixed: bool = False,
            force_resizable: bool = False,
            hide_icon: bool = False,
            popped_out: bool = False
        ):
            super().__init__()
            
            # Initialize state and controllers
            self._state = get_panel_state()
            self._animation_engine = AvatarAnimationEngine()
            self._conference_controller = VideoConferenceController(self._state)
            self._media_controller = MediaDisplayController(self._state)
            self._media_controller.register_frame_callback(lambda frame: self.frame_updated.emit(frame))
            self._desktop_mirror: Optional[DesktopMirrorThread] = None
            self._desktop_capture_ui_intent = False
            
            # Window setup
            self._setup_window(hide_icon, popped_out)
            
            # Build UI
            self._build_ui()
            
            # Connect signals
            self._connect_signals()
            
            # Setup timers
            self._setup_timers()
            
            # Apply initial sizing
            if force_fixed:
                self.setFixedSize(480, 400)
            elif force_resizable:
                self.resize(640, 480)
            else:
                self.resize(480, 400)
            
            self._fullscreen_mode = False
            if fullscreen:
                self.toggle_fullscreen()
            
            # Register state callbacks
            self._state.register_mode_callback(self._on_mode_change)
            
            logger.info("AvatarWindow initialized")
        
        def _setup_window(self, hide_icon: bool, popped_out: bool) -> None:
            """Configure window properties."""
            title = "SarahMemory — Avatar Panel"
            if popped_out:
                title += " (Pop-out)"
            self.setWindowTitle(title)
            
            # Set window icon
            if not hide_icon:
                try:
                    icon_paths = [
                        Path(os.path.join(getattr(config, "RESOURCES_DIR", os.path.join(BASE_DIR, "resources")), "icons", "softdev0.png")),
                        Path(os.path.join(BASE_DIR, "icon.png")),
                    ]
                    for ico_path in icon_paths:
                        if ico_path.exists():
                            self.setWindowIcon(QIcon(str(ico_path)))
                            break
                except Exception:
                    pass
            
            # Window flags for pop-out mode
            if popped_out:
                self.setWindowFlags(
                    Qt.Window |
                    Qt.WindowStaysOnTopHint |
                    Qt.CustomizeWindowHint |
                    Qt.WindowTitleHint |
                    Qt.WindowCloseButtonHint |
                    Qt.WindowMinMaxButtonsHint
                )
        
        def _build_ui(self) -> None:
            """Build the complete UI layout."""
            # Central widget
            central = QWidget(self)
            self.setCentralWidget(central)
            
            main_layout = QVBoxLayout(central)
            main_layout.setContentsMargins(4, 4, 4, 4)
            main_layout.setSpacing(4)
            
            # --- Toolbar ---
            self._toolbar = self._create_toolbar()
            main_layout.addWidget(self._toolbar)
            
            # --- Main Display Stack ---
            self._display_stack = QStackedWidget()
            main_layout.addWidget(self._display_stack, 1)
            
            # Create display pages
            self._avatar_display = self._create_avatar_display()
            self._desktop_display = self._create_desktop_display()
            self._conference_display = self._create_conference_display()
            self._media_display = self._create_media_display()
            
            self._display_stack.addWidget(self._avatar_display)
            self._display_stack.addWidget(self._desktop_display)
            self._display_stack.addWidget(self._conference_display)
            self._display_stack.addWidget(self._media_display)
            
            # --- Control Panel ---
            self._control_panel = self._create_control_panel()
            main_layout.addWidget(self._control_panel)
            
            # --- Status Bar ---
            self._status_label = QLabel("Ready")
            self._status_label.setStyleSheet("color: #9aa4ad; font-size: 11px;")
            main_layout.addWidget(self._status_label)
            
            # Set initial page
            self._display_stack.setCurrentIndex(0)
        
        def _create_toolbar(self) -> QToolBar:
            """Create the toolbar with mode and control buttons."""
            toolbar = QToolBar()
            toolbar.setMovable(False)
            toolbar.setIconSize(QSize(20, 20))
            
            # Mode buttons
            self._btn_avatar = QPushButton("Avatar")
            self._btn_avatar.setCheckable(True)
            self._btn_avatar.setChecked(True)
            self._btn_avatar.clicked.connect(lambda: self._switch_mode(PanelMode.AVATAR_2D))
            toolbar.addWidget(self._btn_avatar)
            
            self._btn_desktop = QPushButton("Desktop")
            self._btn_desktop.setCheckable(True)
            self._btn_desktop.clicked.connect(lambda: self._switch_mode(PanelMode.DESKTOP_MIRROR))
            toolbar.addWidget(self._btn_desktop)
            
            self._btn_conference = QPushButton("Conference")
            self._btn_conference.setCheckable(True)
            self._btn_conference.clicked.connect(lambda: self._switch_mode(PanelMode.VIDEO_CONFERENCE))
            toolbar.addWidget(self._btn_conference)
            
            toolbar.addSeparator()
            
            # Window control buttons
            self._btn_popout = QPushButton("⬈")
            self._btn_popout.setToolTip("Pop out to separate window")
            self._btn_popout.setFixedWidth(30)
            self._btn_popout.clicked.connect(self._on_popout)
            toolbar.addWidget(self._btn_popout)
            
            self._btn_maximize = QPushButton("⬜")
            self._btn_maximize.setToolTip("Maximize/Restore")
            self._btn_maximize.setFixedWidth(30)
            self._btn_maximize.clicked.connect(self.toggle_fullscreen)
            toolbar.addWidget(self._btn_maximize)
            
            return toolbar
        
        def _create_avatar_display(self) -> QWidget:
            """Create the avatar display widget."""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Avatar display label
            self._avatar_label = QLabel()
            self._avatar_label.setAlignment(Qt.AlignCenter)
            self._avatar_label.setStyleSheet("background-color: #171a21;")
            self._avatar_label.setMinimumSize(320, 240)
            layout.addWidget(self._avatar_label)
            
            # Avatar type selector
            type_row = QHBoxLayout()
            type_row.addWidget(QLabel("Type:"))
            
            self._avatar_type_combo = QComboBox()
            self._avatar_type_combo.addItems(["2D Avatar", "3D Avatar"])
            self._avatar_type_combo.currentIndexChanged.connect(self._on_avatar_type_change)
            type_row.addWidget(self._avatar_type_combo)
            
            type_row.addStretch()
            layout.addLayout(type_row)
            
            return widget
        
        def _create_desktop_display(self) -> QWidget:
            """Create the desktop mirror display widget."""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            self._desktop_label = QLabel("Desktop Mirror")
            self._desktop_label.setAlignment(Qt.AlignCenter)
            self._desktop_label.setStyleSheet("background-color: #0b0e12;")
            self._desktop_label.setMinimumSize(320, 240)
            layout.addWidget(self._desktop_label)
            
            # Monitor selector
            monitor_row = QHBoxLayout()
            monitor_row.addWidget(QLabel("Monitor:"))
            
            self._monitor_combo = QComboBox()
            self._monitor_combo.addItems(["Primary", "Secondary", "All"])
            monitor_row.addWidget(self._monitor_combo)
            
            monitor_row.addStretch()
            layout.addLayout(monitor_row)
            
            return widget
        
        def _create_conference_display(self) -> QWidget:
            """Create the video conference display widget."""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Remote video display
            self._remote_video_label = QLabel("Remote Video")
            self._remote_video_label.setAlignment(Qt.AlignCenter)
            self._remote_video_label.setStyleSheet("background-color: #0b0e12;")
            self._remote_video_label.setMinimumSize(320, 180)
            layout.addWidget(self._remote_video_label, 3)
            
            # Local video preview (picture-in-picture)
            self._local_video_label = QLabel("Local")
            self._local_video_label.setAlignment(Qt.AlignCenter)
            self._local_video_label.setStyleSheet("background-color: #171a21; border: 1px solid #5f9ef7;")
            self._local_video_label.setFixedSize(120, 90)
            
            # Call controls
            controls_row = QHBoxLayout()
            
            self._btn_mute_audio = QPushButton("🎤")
            self._btn_mute_audio.setCheckable(True)
            self._btn_mute_audio.setToolTip("Mute/Unmute Audio")
            self._btn_mute_audio.clicked.connect(self._on_toggle_audio)
            controls_row.addWidget(self._btn_mute_audio)
            
            self._btn_mute_video = QPushButton("📹")
            self._btn_mute_video.setCheckable(True)
            self._btn_mute_video.setToolTip("Toggle Video")
            self._btn_mute_video.clicked.connect(self._on_toggle_video)
            controls_row.addWidget(self._btn_mute_video)
            
            self._btn_end_call = QPushButton("🔴 End")
            self._btn_end_call.setStyleSheet("background-color: #f26a7a;")
            self._btn_end_call.clicked.connect(self._on_end_call)
            controls_row.addWidget(self._btn_end_call)
            
            # Call duration label
            self._call_duration_label = QLabel("00:00")
            self._call_duration_label.setStyleSheet("color: #9aa4ad;")
            controls_row.addWidget(self._call_duration_label)
            
            layout.addLayout(controls_row)
            
            return widget
        
        def _create_media_display(self) -> QWidget:
            """Create the media display widget."""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            self._media_label = QLabel("Media Display")
            self._media_label.setAlignment(Qt.AlignCenter)
            self._media_label.setStyleSheet("background-color: #0b0e12;")
            self._media_label.setMinimumSize(320, 240)
            layout.addWidget(self._media_label)
            
            # Media controls
            controls_row = QHBoxLayout()
            
            self._btn_play_pause = QPushButton("▶")
            self._btn_play_pause.clicked.connect(self._on_play_pause)
            controls_row.addWidget(self._btn_play_pause)
            
            self._btn_stop = QPushButton("⬛")
            self._btn_stop.clicked.connect(self._on_stop_media)
            controls_row.addWidget(self._btn_stop)
            
            self._media_progress = QProgressBar()
            self._media_progress.setRange(0, 100)
            controls_row.addWidget(self._media_progress)
            
            layout.addLayout(controls_row)
            
            return widget
        
        def _create_control_panel(self) -> QGroupBox:
            """Create the control panel with emotion and voice controls."""
            group = QGroupBox("Controls")
            layout = QVBoxLayout(group)
            
            # Emotion selector
            emotion_row = QHBoxLayout()
            emotion_row.addWidget(QLabel("Emotion:"))
            
            self._emotion_combo = QComboBox()
            self._emotion_combo.addItems([
                "neutral", "joy", "trust", "surprise",
                "sadness", "fear", "anger", "thinking"
            ])
            self._emotion_combo.currentTextChanged.connect(self._on_emotion_change)
            emotion_row.addWidget(self._emotion_combo)
            
            layout.addLayout(emotion_row)
            
            # Voice selector
            voice_row = QHBoxLayout()
            voice_row.addWidget(QLabel("Voice:"))
            
            self._voice_combo = QComboBox()
            try:
                if Voice and hasattr(Voice, "get_voice_profiles"):
                    for name in Voice.get_voice_profiles():
                        self._voice_combo.addItem(name)
                else:
                    self._voice_combo.addItem("Default")
            except Exception:
                self._voice_combo.addItem("Default")
            self._voice_combo.currentTextChanged.connect(self._on_voice_change)
            voice_row.addWidget(self._voice_combo)
            
            layout.addLayout(voice_row)
            
            # Zoom slider
            zoom_row = QHBoxLayout()
            zoom_row.addWidget(QLabel("Zoom:"))
            
            self._zoom_slider = QSlider(Qt.Horizontal)
            self._zoom_slider.setMinimum(25)
            self._zoom_slider.setMaximum(300)
            self._zoom_slider.setValue(100)
            self._zoom_slider.valueChanged.connect(self._on_zoom_change)
            zoom_row.addWidget(self._zoom_slider)
            
            self._zoom_label = QLabel("100%")
            zoom_row.addWidget(self._zoom_label)
            
            layout.addLayout(zoom_row)
            
            # Mic/Camera toggles
            toggle_row = QHBoxLayout()
            
            self._btn_mic = QPushButton("Mic: On")
            self._btn_mic.setCheckable(True)
            self._btn_mic.setChecked(True)
            self._btn_mic.toggled.connect(self._on_mic_toggle)
            toggle_row.addWidget(self._btn_mic)
            
            self._btn_cam = QPushButton("Cam: On")
            self._btn_cam.setCheckable(True)
            self._btn_cam.setChecked(True)
            self._btn_cam.toggled.connect(self._on_cam_toggle)
            toggle_row.addWidget(self._btn_cam)
            
            self._btn_reset = QPushButton("Reset")
            self._btn_reset.clicked.connect(self._on_reset)
            toggle_row.addWidget(self._btn_reset)
            
            layout.addLayout(toggle_row)
            
            return group
        
        def _connect_signals(self) -> None:
            """Connect internal signals."""
            self.frame_updated.connect(self._update_display_frame)
            self.mode_changed.connect(self._handle_mode_change)
        
        def _setup_timers(self) -> None:
            """Setup update timers."""
            # Avatar animation timer
            self._avatar_timer = QTimer()
            self._avatar_timer.timeout.connect(self._update_avatar_frame)
            self._avatar_timer.start(50)  # 20 FPS
            
            # Call duration timer
            self._duration_timer = QTimer()
            self._duration_timer.timeout.connect(self._update_call_duration)
            self._duration_timer.start(1000)
        
        # =====================================================================
        # Event Handlers
        # =====================================================================
        
        def _switch_mode(self, mode: PanelMode) -> None:
            # This method is connected only to the local Qt controls.
            self._desktop_capture_ui_intent = (mode == PanelMode.DESKTOP_MIRROR)
            try:
                old_mode = self._state.get_mode()
                self._state.set_mode(mode)
                if old_mode == mode:
                    self._handle_mode_change(mode.name)
            finally:
                self._desktop_capture_ui_intent = False
        
        def _on_mode_change(self, old_mode: PanelMode, new_mode: PanelMode) -> None:
            """Handle mode change from state manager."""
            self.mode_changed.emit(new_mode.name)
        
        def _handle_mode_change(self, mode_name: str) -> None:
            try:
                mode = PanelMode[mode_name]
            except KeyError:
                return
            self._btn_avatar.setChecked(mode in (PanelMode.AVATAR_2D, PanelMode.AVATAR_3D))
            self._btn_desktop.setChecked(mode == PanelMode.DESKTOP_MIRROR)
            self._btn_conference.setChecked(mode == PanelMode.VIDEO_CONFERENCE)
            if mode in (PanelMode.AVATAR_2D, PanelMode.AVATAR_3D):
                self._display_stack.setCurrentIndex(0)
                self._stop_desktop_mirror()
            elif mode == PanelMode.DESKTOP_MIRROR:
                self._display_stack.setCurrentIndex(1)
                self._start_desktop_mirror()
            elif mode == PanelMode.VIDEO_CONFERENCE:
                self._display_stack.setCurrentIndex(2)
                self._stop_desktop_mirror()
            elif mode in (PanelMode.MEDIA_IMAGE, PanelMode.MEDIA_VIDEO):
                self._display_stack.setCurrentIndex(3)
                self._stop_desktop_mirror()
                if mode == PanelMode.MEDIA_IMAGE and HAVE_PIL:
                    path = self._state.get_current_media()
                    if path and _panel_verify_image(path):
                        with Image.open(path) as im:
                            rgba = im.convert('RGBA')
                            data = rgba.tobytes('raw', 'RGBA')
                            qimg = QImage(data, rgba.width, rgba.height, QImage.Format_RGBA8888).copy()
                            self._media_label.setPixmap(QPixmap.fromImage(qimg))
            self._status_set('Mode: '+mode.name)
        
        def _on_avatar_type_change(self, index: int) -> None:
            """Handle avatar type change."""
            avatar_type = "3d" if index == 1 else "2d"
            self._state.set_avatar_type(avatar_type)
            self._animation_engine.set_emotion(self._state.get_emotion())
            self._status_set(f"Avatar type: {avatar_type.upper()}")
        
        def _on_emotion_change(self, emotion: str) -> None:
            """Handle emotion selection change."""
            self._state.set_emotion(emotion)
            self._animation_engine.set_emotion(emotion)
            
            # Notify Avatar module if available
            if Avatar and hasattr(Avatar, "set_avatar_emotion"):
                try:
                    Avatar.set_avatar_emotion(emotion)
                except Exception as e:
                    logger.warning(f"Avatar emotion set failed: {e}")
            
            self._status_set(f"Emotion: {emotion}")
        
        def _on_voice_change(self, voice: str) -> None:
            """Handle voice selection change."""
            if Voice:
                try:
                    if hasattr(Voice, "set_voice_profile"):
                        Voice.set_voice_profile(voice)
                    if hasattr(Voice, "save_voice_settings"):
                        Voice.save_voice_settings(profile=voice)
                except Exception as e:
                    logger.warning(f"Voice set failed: {e}")
            self._status_set(f"Voice: {voice}")
        
        def _on_zoom_change(self, value: int) -> None:
            """Handle zoom slider change."""
            self._state.set_zoom(value / 100.0)
            self._zoom_label.setText(f"{value}%")
            
            if Avatar and hasattr(Avatar, "set_zoom"):
                try:
                    Avatar.set_zoom(value / 100.0)
                except Exception as e:
                    logger.warning(f"Zoom set failed: {e}")
        
        def _on_mic_toggle(self, checked: bool) -> None:
            """Handle microphone toggle."""
            self._btn_mic.setText(f"Mic: {'On' if checked else 'Off'}")
            if Voice and hasattr(Voice, "toggle_microphone"):
                try:
                    Voice.toggle_microphone(checked)
                except Exception as e:
                    logger.warning(f"Mic toggle failed: {e}")
        
        def _on_cam_toggle(self, checked: bool) -> None:
            """Handle camera toggle."""
            self._btn_cam.setText(f"Cam: {'On' if checked else 'Off'}")
            if Avatar and hasattr(Avatar, "toggle_camera"):
                try:
                    Avatar.toggle_camera(checked)
                except Exception as e:
                    logger.warning(f"Cam toggle failed: {e}")
        
        def _on_reset(self) -> None:
            """Reset avatar to default state."""
            self._emotion_combo.setCurrentText("neutral")
            self._zoom_slider.setValue(100)
            self._state.set_emotion("neutral")
            self._state.set_zoom(1.0)
            self._animation_engine.set_emotion("neutral")
            
            if Avatar and hasattr(Avatar, "reset_avatar"):
                try:
                    Avatar.reset_avatar()
                except Exception as e:
                    logger.warning(f"Reset failed: {e}")
            
            self._status_set("Avatar reset")
        
        def _on_popout(self) -> None:
            """Handle pop-out button click."""
            if self._state.is_popped_out():
                # Already popped out, ignore
                return
            
            # Create new popped-out window
            self._popped_window = AvatarWindow(popped_out=True)
            self._popped_window.show()
            self._state.set_popped_out(True)
            self._status_set("Popped out to separate window")
        
        def _on_toggle_audio(self) -> None:
            """Toggle conference audio."""
            active = self._conference_controller.toggle_audio()
            self._btn_mute_audio.setText("🎤" if active else "🔇")
        
        def _on_toggle_video(self) -> None:
            """Toggle conference video."""
            active = self._conference_controller.toggle_video()
            self._btn_mute_video.setText("📹" if active else "📷")
        
        def _on_end_call(self) -> None:
            """End current call."""
            self._conference_controller.end_call()
            self._status_set("Call ended")
        
        def _on_play_pause(self) -> None:
            """Toggle media playback."""
            if self._media_controller.is_playing():
                self._media_controller.pause_video()
                self._btn_play_pause.setText("▶")
            else:
                self._media_controller.resume_video()
                self._btn_play_pause.setText("⏸")
        
        def _on_stop_media(self) -> None:
            """Stop media playback."""
            self._media_controller.stop_video()
            self._state.restore_previous_mode()
            self._btn_play_pause.setText("▶")
        
        # =====================================================================
        # Display Updates
        # =====================================================================
        
        def _update_avatar_frame(self) -> None:
            """Update avatar display frame."""
            if self._state.get_mode() not in [PanelMode.AVATAR_2D, PanelMode.AVATAR_3D]:
                return
            if self._state.get_mode() == PanelMode.AVATAR_3D:
                self._status_set('Native Qt 3D rendering unavailable; use WebUI Avatar3D')
                # A 2D visual fallback is still permitted, but never reported as rendered GLB.
            try:
                # Get display size
                size = self._avatar_label.size()
                width = max(100, size.width())
                height = max(100, size.height())
                
                # Render frame
                frame = self._animation_engine.render_2d_frame(width, height)
                
                if frame and HAVE_PIL:
                    # Convert PIL Image to QPixmap
                    data = frame.tobytes("raw", "RGBA")
                    qimg = QImage(data, frame.width, frame.height, QImage.Format_RGBA8888)
                    pixmap = QPixmap.fromImage(qimg)
                    
                    # Apply zoom
                    zoom = self._state.get_zoom()
                    if zoom != 1.0:
                        new_width = int(pixmap.width() * zoom)
                        new_height = int(pixmap.height() * zoom)
                        pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    self._avatar_label.setPixmap(pixmap)
            
            except Exception as e:
                logger.debug(f"Avatar frame update error: {e}")
        
        def _update_display_frame(self, frame) -> None:
            """Update display with new frame (thread-safe)."""
            try:
                mode = self._state.get_mode()
                
                if mode == PanelMode.DESKTOP_MIRROR and HAVE_PIL:
                    # Convert PIL Image to QPixmap for desktop mirror
                    if hasattr(frame, 'tobytes'):
                        data = frame.convert('RGBA').tobytes("raw", "RGBA")
                        qimg = QImage(data, frame.width, frame.height, QImage.Format_RGBA8888)
                        pixmap = QPixmap.fromImage(qimg)
                        self._desktop_label.setPixmap(pixmap.scaled(
                            self._desktop_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        ))
                
                elif mode == PanelMode.MEDIA_VIDEO and HAVE_CV2:
                    # Convert OpenCV frame to QPixmap
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    bytes_per_line = ch * w
                    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self._media_label.setPixmap(pixmap.scaled(
                        self._media_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
            
            except Exception as e:
                logger.debug(f"Display frame update error: {e}")
        
        def _update_call_duration(self) -> None:
            """Update call duration display."""
            if self._state.get_conference_state() == ConferenceState.CONNECTED:
                duration = self._state.get_call_duration()
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self._call_duration_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        # =====================================================================
        # Desktop Mirror Control
        # =====================================================================
        
        def _start_desktop_mirror(self) -> None:
            if self._desktop_mirror is not None and (self._desktop_mirror.isRunning() if HAVE_QT else self._desktop_mirror.is_alive()):
                return
            if not HAVE_MSS or not self._desktop_capture_ui_intent or not _panel_authorize_capture('desktop', local_ui_consent=True):
                self._status_set('Desktop mirror unavailable or awaiting authorization')
                return
            self._desktop_mirror = DesktopMirrorThread()
            self._desktop_mirror._capture_authorized = True
            self._desktop_mirror.frame_ready.connect(self.frame_updated.emit)
            self._desktop_mirror.start_capture()
            self._status_set('Desktop mirror started')
        
        def _stop_desktop_mirror(self) -> None:
            worker = self._desktop_mirror
            if worker is not None:
                worker.stop_capture()
                self._desktop_mirror = None
        
        # =====================================================================
        # Window Management
        # =====================================================================
        
        def toggle_fullscreen(self) -> None:
            """Toggle fullscreen mode."""
            if self._fullscreen_mode:
                self.showNormal()
                self._btn_maximize.setText("⬜")
                self._state.set_maximized(False)
            else:
                self.showFullScreen()
                self._btn_maximize.setText("❐")
                self._state.set_maximized(True)
            self._fullscreen_mode = not self._fullscreen_mode
        
        def keyPressEvent(self, event) -> None:
            """Handle key press events."""
            if event.key() == Qt.Key_F11:
                self.toggle_fullscreen()
            elif event.key() == Qt.Key_Escape and self._fullscreen_mode:
                self.toggle_fullscreen()
            else:
                super().keyPressEvent(event)
        
        def closeEvent(self, event) -> None:
            """Handle window close."""
            self._stop_desktop_mirror()
            self._media_controller.stop_video()
            
            if self._state.is_popped_out():
                self._state.set_popped_out(False)
            
            logger.info("Avatar Panel closed")
            event.accept()
        
        def resizeEvent(self, event) -> None:
            """Handle window resize."""
            super().resizeEvent(event)
            size = event.size()
            self._state.set_panel_size(size.width(), size.height())
        
        # =====================================================================
        # Status Updates
        # =====================================================================
        
        def _status_set(self, text: str) -> None:
            """Set status bar text."""
            try:
                self._status_label.setText(text)
            except Exception:
                pass
        
        # =====================================================================
        # Public API
        # =====================================================================
        
        def start_call(self, peer_id: str) -> bool:
            """
            Start a video call to the specified peer.
            Called from WebUI contacts list or keypad.
            
            Args:
                peer_id: Identifier of the peer to call
                
            Returns:
                True if call initiated successfully
            """
            return self._conference_controller.start_call(peer_id)
        
        def answer_call(self, peer_id: str) -> bool:
            """
            Answer an incoming call.
            
            Args:
                peer_id: Identifier of the calling peer
                
            Returns:
                True if call answered successfully
            """
            return self._conference_controller.answer_call(peer_id)
        
        def end_call(self) -> None:
            """End the current call."""
            self._conference_controller.end_call()
        
        def display_generated_image(self, image_path: str) -> bool:
            """
            Display an AI-generated image in the panel.
            
            Args:
                image_path: Path to the generated image
                
            Returns:
                True if displayed successfully
            """
            return self._media_controller.display_image(image_path)
        
        def display_generated_video(self, video_path: str, loop: bool = False) -> bool:
            """
            Play an AI-generated video in the panel.
            
            Args:
                video_path: Path to the generated video
                loop: Whether to loop playback
                
            Returns:
                True if playback started
            """
            return self._media_controller.display_video(video_path, loop)
        
        def set_avatar_emotion(self, emotion: str) -> None:
            """Set avatar emotion from external caller."""
            self._emotion_combo.setCurrentText(emotion)
        
        def start_lip_sync(self, duration: float = 0.0) -> None:
            """Start lip sync animation."""
            self._animation_engine.start_lip_sync(duration)
        
        def stop_lip_sync(self) -> None:
            """Stop lip sync animation."""
            self._animation_engine.stop_lip_sync()
        
        def get_state(self) -> Dict[str, Any]:
            """Get current panel state."""
            return self._state.to_dict()


# ============================================================================
# HEADLESS/WEB API INTERFACE
# ============================================================================
class AvatarPanelAPI:
    """
    API interface for headless/web operation.
    Provides methods for controlling the Avatar Panel from WebUI/Flask.
    """
    
    def __init__(self):
        self._state = get_panel_state()
        self._animation_engine = AvatarAnimationEngine()
        self._conference_controller = VideoConferenceController(self._state)
        self._media_controller = MediaDisplayController(self._state)
        
        logger.info("AvatarPanelAPI initialized for headless operation")
    
    def start_desktop_mirror(self) -> Dict[str, Any]:
        """Headless route contract. No user-consent witness is accepted from an API payload."""
        return {'success': False, 'ok': False, 'error': 'desktop_capture_requires_authorized_local_ui',
                'execution_authority': False}

    def stop_desktop_mirror(self) -> Dict[str, Any]:
        """No implicit headless capture worker exists to keep running."""
        return {'success': True, 'ok': True, 'capturing': False, 'execution_authority': False}

    def get_state(self) -> Dict[str, Any]:
        """Get current panel state for WebUI sync."""
        return self._state.to_dict()
    
    def set_mode(self, mode: str) -> Dict[str, Any]:
        """
        Set panel mode.
        
        Args:
            mode: Mode name (AVATAR_2D, AVATAR_3D, DESKTOP_MIRROR, VIDEO_CONFERENCE, MEDIA_IMAGE, MEDIA_VIDEO)
            
        Returns:
            Updated state dictionary
        """
        try:
            mode_key = str(mode or "AVATAR_2D").strip().upper().replace("-", "_")
            alias_map = {
                "2D": "AVATAR_2D",
                "AVATAR2D": "AVATAR_2D",
                "AVATAR_2D": "AVATAR_2D",
                "3D": "AVATAR_3D",
                "AVATAR3D": "AVATAR_3D",
                "AVATAR_3D": "AVATAR_3D",
                "DESKTOP": "DESKTOP_MIRROR",
                "MIRROR": "DESKTOP_MIRROR",
                "DESKTOP_MIRROR": "DESKTOP_MIRROR",
                "MEDIA": "MEDIA_IMAGE",
                "IDLE": "IDLE",
            }
            panel_mode = PanelMode[alias_map.get(mode_key, mode_key)]
            self._state.set_mode(panel_mode)
            return {"success": True, "state": self.get_state()}
        except KeyError:
            return {"success": False, "error": f"Unknown mode: {mode}"}
    
    def set_emotion(self, emotion: str) -> Dict[str, Any]:
        """Set avatar emotion."""
        self._state.set_emotion(emotion)
        self._animation_engine.set_emotion(emotion)
        return {"success": True, "emotion": emotion}
    
    def set_avatar_type(self, avatar_type: str) -> Dict[str, Any]:
        try:
            self._state.set_avatar_type(avatar_type)
            return {'success': True, 'avatar_type': self._state.get_avatar_type()}
        except ValueError as exc:
            return {'success': False, 'error': str(exc)}
    
    def set_zoom(self, zoom: float) -> Dict[str, Any]:
        self._state.set_zoom(zoom)
        return {'success': True, 'zoom': self._state.get_zoom()}
    
    def start_call(self, peer_id: str, video: bool = True, audio: bool = True) -> Dict[str, Any]:
        """Start a call to the specified peer."""
        success = self._conference_controller.start_call(peer_id, video, audio)
        return {
            "success": success,
            "call_info": self._conference_controller.get_call_info()
        }
    
    def answer_call(self, peer_id: str) -> Dict[str, Any]:
        """Answer an incoming call."""
        success = self._conference_controller.answer_call(peer_id)
        return {
            "success": success,
            "call_info": self._conference_controller.get_call_info()
        }
    
    def end_call(self) -> Dict[str, Any]:
        """End the current call."""
        self._conference_controller.end_call()
        return {"success": True, "state": self.get_state()}
    
    def toggle_call_video(self) -> Dict[str, Any]:
        """Toggle call video."""
        active = self._conference_controller.toggle_video()
        return {"success": True, "video_active": active}
    
    def toggle_call_audio(self) -> Dict[str, Any]:
        """Toggle call audio."""
        active = self._conference_controller.toggle_audio()
        return {"success": True, "audio_active": active}
    
    def get_call_info(self) -> Dict[str, Any]:
        """Get current call information."""
        return self._conference_controller.get_call_info()
    
    def display_image(self, image_path: str) -> Dict[str, Any]:
        """Display an image in the panel."""
        success = self._media_controller.display_image(image_path)
        return {"success": success, "path": image_path}
    
    def display_video(self, video_path: str, loop: bool = False) -> Dict[str, Any]:
        """Play a video in the panel."""
        success = self._media_controller.display_video(video_path, loop)
        return {"success": success, "path": video_path}
    
    def stop_media(self) -> Dict[str, Any]:
        """Stop media playback."""
        self._media_controller.stop_video()
        self._state.restore_previous_mode()
        return {"success": True}
    
    def get_media_info(self) -> Dict[str, Any]:
        """Get current media information."""
        return self._media_controller.get_current_media_info()
    
    def queue_media(self, media_path: str, media_type: str = "image") -> Dict[str, Any]:
        try:
            self._state.queue_media(media_path, media_type)
            return {'success': True, 'queued': media_path}
        except (ValueError, OverflowError) as exc:
            return {'success': False, 'error': str(exc)}


    def display_media_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {'success': False, 'error': 'result_must_be_dict'}
        kind = str(result.get('type') or result.get('media_type') or 'unknown').strip().lower()
        media_type = 'image' if kind in {'image','png','jpg','jpeg','webp'} else 'video' if kind in {'video','mp4','webm','mov','mkv'} else ''
        if not media_type:
            return {'success': False, 'error': 'unsupported_media_type'}
        uri = result.get('path') or result.get('uri') or result.get('url')
        if not uri and media_type == 'image' and isinstance(result.get('base64'), str):
            encoded = result['base64']
            if len(encoded) > (_MAX_UPLOAD_BYTES * 4 // 3 + 16):
                return {'success': False, 'error': 'media_upload_too_large'}
            try:
                raw = base64.b64decode(encoded, validate=True)
                if not raw or len(raw) > _MAX_UPLOAD_BYTES or not HAVE_PIL:
                    return {'success': False, 'error': 'invalid_image_upload'}
                with Image.open(io.BytesIO(raw)) as img:
                    if img.width * img.height > _MAX_RENDER_PIXELS or img.width <= 0 or img.height <= 0:
                        raise ValueError('image_dimensions_out_of_bounds')
                    img.verify()
                out_dir = Path(DATASETS_DIR) / 'generated'
                out_dir.mkdir(parents=True, exist_ok=True)
                uri = str(out_dir / ('media_' + uuid.uuid4().hex + '.png'))
                # Decode and re-encode rather than trusting the submitted extension.
                with Image.open(io.BytesIO(raw)) as img:
                    img.convert('RGBA').save(uri, format='PNG')
            except (ValueError, OSError, SyntaxError, MemoryError, base64.binascii.Error):
                return {'success': False, 'error': 'invalid_image_upload'}
        if not isinstance(uri, str):
            return {'success': False, 'error': 'invalid_media_uri'}
        if _panel_bool(result.get('queue', False)):
            return self.queue_media(uri, media_type)
        if media_type == 'image':
            return self.display_image(uri)
        if _panel_bool(result.get('autoplay', True), True):
            return self.display_video(uri, loop=_panel_bool(result.get('loop', False)))
        return self.queue_media(uri, 'video')

    def set_speaking(self, speaking: bool) -> Dict[str, Any]:
        value = _panel_bool(speaking)
        self._state.set_speaking(value)
        if value:
            self._animation_engine.start_lip_sync(0.0)
        else:
            self._animation_engine.stop_lip_sync()
        return {'success': True, 'speaking': value, 'state': self.get_state()}

    def set_listening(self, listening: bool) -> Dict[str, Any]:
        value = _panel_bool(listening)
        self._state.set_listening(value)
        return {'success': True, 'listening': value, 'state': self.get_state()}

    def get_avatar_frame(self, width: int = 300, height: int = 300, format: str = "base64") -> Dict[str, Any]:
        try:
            width = _panel_dimension(width, low=16)
            height = _panel_dimension(height, low=16)
            if width * height > _MAX_RENDER_PIXELS or format not in {'base64', 'raw', 'binary'}:
                return {'success': False, 'error': 'invalid_frame_request'}
        except ValueError:
            return {'success': False, 'error': 'invalid_frame_size'}
        frame = self._animation_engine.render_2d_frame(width, height)
        if frame is None:
            return {'success': False, 'error': 'frame_render_unavailable'}
        if format == 'base64':
            buffer = io.BytesIO()
            frame.save(buffer, format='PNG')
            payload = base64.b64encode(buffer.getvalue()).decode('ascii')
        else:
            payload = frame.tobytes()
        return {'success': True, 'frame': payload, 'format': format, 'width': width, 'height': height}
    
    def start_lip_sync(self, duration: float = 0.0) -> Dict[str, Any]:
        """Start lip sync animation."""
        self._animation_engine.start_lip_sync(duration)
        return {"success": True, "lip_sync": True}
    
    def stop_lip_sync(self) -> Dict[str, Any]:
        """Stop lip sync animation."""
        self._animation_engine.stop_lip_sync()
        return {"success": True, "lip_sync": False}
    
    def set_panel_size(self, width: int, height: int) -> Dict[str, Any]:
        try:
            self._state.set_panel_size(width, height)
            return {'success': True, 'size': self._state.get_panel_size()}
        except ValueError as exc:
            return {'success': False, 'error': str(exc)}
    
    def toggle_popout(self) -> Dict[str, Any]:
        """Toggle pop-out state."""
        current = self._state.is_popped_out()
        self._state.set_popped_out(not current)
        return {"success": True, "popped_out": not current}
    
    def toggle_maximize(self) -> Dict[str, Any]:
        """Toggle maximize state."""
        current = self._state.is_maximized()
        self._state.set_maximized(not current)
        return {"success": True, "maximized": not current}


# ============================================================================
# GLOBAL API INSTANCE
# ============================================================================
_panel_api: Optional[AvatarPanelAPI] = None

def get_panel_api() -> AvatarPanelAPI:
    """Get or create the global API instance."""
    global _panel_api
    with _PANEL_SINGLETON_LOCK:
        if _panel_api is None:
            _panel_api = AvatarPanelAPI()
        return _panel_api


# ============================================================================
# LAUNCH FUNCTIONS
# ============================================================================
def launch() -> Optional['AvatarWindow']:
    """
    Launch the Avatar Panel.
    Detects environment and launches appropriate mode.
    
    Returns:
        AvatarWindow instance if GUI mode, None for headless
    """
    if not ENABLE_AVATAR_PANEL:
        logger.info("Avatar Panel disabled in configuration")
        return None
    
    # Check if we're in headless mode
    if RUN_MODE == "cloud" or DEVICE_MODE == "public_web":
        logger.info("Headless mode detected; API-only operation")
        return None
    
    # Check for GUI availability
    if not HAVE_QT:
        logger.warning("PyQt5 not installed; GUI unavailable")
        return None
    
    return main()


def main() -> Optional['AvatarWindow']:
    """
    Main entry point for GUI mode.
    
    Returns:
        AvatarWindow instance
    """
    if not HAVE_QT:
        logger.error("PyQt5 not installed; cannot show Avatar panel")
        return None
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Parse command line arguments
    args = sys.argv[1:]
    fullscreen = '--fullscreen' in args
    force_fixed = '--fixed' in args
    force_resizable = '--resizable' in args
    hide_icon = '--hide-icon' in args
    
    # Create and show window
    window = AvatarWindow(
        fullscreen=fullscreen,
        force_fixed=force_fixed,
        force_resizable=force_resizable,
        hide_icon=hide_icon
    )
    window.show()
    
    # Only exec if we created the app
    if not QApplication.instance().property("sarah_main_app"):
        app.setProperty("sarah_main_app", True)
        return window
    
    return window


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check for API test mode
    if '--api-test' in sys.argv:
        print("Testing Avatar Panel API...")
        api = get_panel_api()
        
        # Test state
        print(f"Initial state: {api.get_state()}")
        
        # Test emotion
        result = api.set_emotion("joy")
        print(f"Set emotion: {result}")
        
        # Test frame generation
        frame_result = api.get_avatar_frame(200, 200)
        print(f"Frame generated: {frame_result.get('success')}, size: {len(frame_result.get('frame', ''))}")
        
        # Test mode switching
        result = api.set_mode("AVATAR_3D")
        print(f"Mode switch: {result}")
        
        print("API test complete")
    else:
        # Standard GUI launch
        window = launch()
        if window:
            sys.exit(QApplication.instance().exec_())
        else:
            print("Avatar Panel could not be launched (headless mode or missing dependencies)")
            print("Use --api-test to test the API interface")

# ====================================================================
# END OF SarahMemoryAvatarPanel.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAvatarPanel',
    "version": "v9.1.0a-enterprise",
    "category": 'Presentation',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['avatar', 'execution', 'presentation'],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 45,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAvatarPanel.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Read-only availability signal; unknown qualities are not reported as perfect."""
    return {
        'status': 'Available' if ENABLE_AVATAR_PANEL else 'Disabled',
        'availability': 1.0 if ENABLE_AVATAR_PANEL else 0.0,
        'integrity': 0.0, 'performance': 0.0, 'reliability': 0.0, 'confidence': 0.0,
        'latency_ms': 0.0, 'stability': 0.0, 'compatibility': 0.0,
        'notes': ['Adapter loaded; quality and hardware operation unmeasured',
                  'camera/conference/desktop capture require separate verified authorization',
                  'physical actuation forbidden'],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAvatarPanel',
        "sml_adapter": True,
        "metadata": sml_get_metadata(),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAvatarPanel', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---
