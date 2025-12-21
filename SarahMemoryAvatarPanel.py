"""--==The SarahMemory Project==--
File: SarahMemoryAvatarPanel.py
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

import sys
import os
import logging
import threading
import time
import json
import base64
import queue
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

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
    RUN_MODE = getattr(config, 'RUN_MODE', 'local')
    DEVICE_MODE = getattr(config, 'DEVICE_MODE', 'local_agent')
    DEVICE_PROFILE = getattr(config, 'DEVICE_PROFILE', 'Standard')
    ENABLE_AVATAR_PANEL = getattr(config, 'ENABLE_AVATAR_PANEL', True)
except ImportError:
    logger.warning("SarahMemoryGlobals not found; using defaults")
    DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'memory', 'datasets')
    AVATAR_DIR = os.path.join(os.path.dirname(__file__), 'resources', 'avatars')
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
        """
        Set panel mode with optional save of previous mode for auto-return.
        
        Args:
            mode: New panel mode to set
            save_previous: If True, saves current mode for later restoration
        """
        with self._lock:
            if save_previous and self._mode != mode:
                self._previous_mode = self._mode
            old_mode = self._mode
            self._mode = mode
            logger.info(f"Panel mode changed: {old_mode.name} -> {mode.name}")
            self._notify_mode_change(old_mode, mode)
    
    def restore_previous_mode(self) -> None:
        """Restore the previous panel mode (used after conference ends)."""
        with self._lock:
            self.set_mode(self._previous_mode, save_previous=False)
    
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
        """Set avatar emotion."""
        with self._lock:
            self._current_emotion = emotion
            logger.debug(f"Avatar emotion set: {emotion}")
    
    def get_avatar_type(self) -> str:
        """Get avatar type (2d/3d)."""
        with self._lock:
            return self._avatar_type
    
    def set_avatar_type(self, avatar_type: str) -> None:
        """Set avatar type."""
        with self._lock:
            self._avatar_type = avatar_type.lower()
            if self._avatar_type == "3d":
                self.set_mode(PanelMode.AVATAR_3D)
            else:
                self.set_mode(PanelMode.AVATAR_2D)
    
    def get_zoom(self) -> float:
        """Get avatar zoom level."""
        with self._lock:
            return self._avatar_zoom
    
    def set_zoom(self, zoom: float) -> None:
        """Set avatar zoom level (0.25 - 3.0)."""
        with self._lock:
            self._avatar_zoom = max(0.25, min(3.0, zoom))
    
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
        """
        Queue media for display in the panel.
        
        Args:
            media_path: Path to media file
            media_type: "image" or "video"
        """
        with self._lock:
            self._media_queue.put((media_path, media_type))
            logger.info(f"Media queued: {media_path} ({media_type})")
    
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
        """Set custom panel size."""
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
        """Restore state from dictionary."""
        with self._lock:
            if "mode" in data:
                self._mode = PanelMode[data["mode"]]
            if "emotion" in data:
                self._current_emotion = data["emotion"]
            if "avatar_type" in data:
                self._avatar_type = data["avatar_type"]
            if "avatar_zoom" in data:
                self._avatar_zoom = data["avatar_zoom"]


# ============================================================================
# GLOBAL STATE INSTANCE
# ============================================================================
_panel_state: Optional[AvatarPanelState] = None

def get_panel_state() -> AvatarPanelState:
    """Get or create the global panel state instance."""
    global _panel_state
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
        self._fps = fps
        self._monitor = monitor
        self._frame_interval = 1.0 / fps
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
    
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
        """Start desktop capture."""
        self._running = True
        self.start()
    
    def stop_capture(self) -> None:
        """Stop desktop capture."""
        self._running = False
        if self.isRunning() if HAVE_QT else self.is_alive():
            self.wait() if HAVE_QT else self.join(timeout=2.0)
    
    def run(self) -> None:
        """Main capture loop."""
        if not HAVE_MSS:
            logger.warning("Desktop mirror unavailable: mss not installed")
            return
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[self._monitor] if self._monitor < len(sct.monitors) else sct.monitors[1]
                
                while self._running:
                    start_time = time.time()
                    
                    try:
                        # Capture frame
                        screenshot = sct.grab(monitor)
                        
                        # Convert to format suitable for display
                        if HAVE_PIL:
                            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                            frame_data = img
                        else:
                            frame_data = screenshot
                        
                        # Emit to callbacks
                        if HAVE_QT:
                            self.frame_ready.emit(frame_data)
                        else:
                            with self._lock:
                                for callback in self._callbacks:
                                    try:
                                        callback(frame_data)
                                    except Exception as e:
                                        logger.warning(f"Desktop mirror callback error: {e}")
                    
                    except Exception as e:
                        logger.warning(f"Desktop capture error: {e}")
                    
                    # Maintain frame rate
                    elapsed = time.time() - start_time
                    sleep_time = self._frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Desktop mirror thread error: {e}")


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
        
        logger.info("AvatarAnimationEngine initialized")
    
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
        """
        Start lip sync animation.
        
        Args:
            duration: Duration in seconds (0 = until stopped)
        """
        self._lip_sync_active = True
        logger.debug(f"Lip sync started (duration: {duration}s)")
        
        if duration > 0:
            self._animation_timer = threading.Timer(duration, self.stop_lip_sync)
            self._animation_timer.start()
    
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
            "animation_speed": self._animation_speed
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
        """
        Initiate a video/audio call to a peer.
        
        Args:
            peer_id: Identifier of the peer to call
            video: Enable video
            audio: Enable audio
            
        Returns:
            True if call initiated successfully
        """
        try:
            logger.info(f"Starting call to peer: {peer_id}")
            self._state.set_remote_peer(peer_id)
            self._state.set_conference_state(ConferenceState.CONNECTING)
            
            # Start local video capture if video enabled
            if video and HAVE_CV2:
                self._start_local_video()
            
            self._local_video_active = video
            self._audio_active = audio
            
            # In a real implementation, this would initiate WebRTC signaling
            # For now, we simulate connection
            def _simulate_connect():
                time.sleep(1.5)  # Simulate connection delay
                self._state.set_conference_state(ConferenceState.CONNECTED)
                self._remote_video_active = True
                self._emit("on_connect", peer_id)
            
            self._connection_thread = threading.Thread(target=_simulate_connect, daemon=True)
            self._connection_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start call: {e}")
            self._state.set_conference_state(ConferenceState.FAILED)
            self._emit("on_error", str(e))
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
        """
        Answer an incoming call.
        
        Args:
            peer_id: Identifier of the calling peer
            
        Returns:
            True if call answered successfully
        """
        try:
            logger.info(f"Answering call from: {peer_id}")
            self._state.set_remote_peer(peer_id)
            self._state.set_conference_state(ConferenceState.CONNECTED)
            
            if HAVE_CV2:
                self._start_local_video()
            
            self._local_video_active = True
            self._remote_video_active = True
            self._audio_active = True
            
            self._emit("on_connect", peer_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to answer call: {e}")
            self._emit("on_error", str(e))
            return False
    
    def _start_local_video(self) -> None:
        """Start local video capture."""
        if not HAVE_CV2:
            return
        
        try:
            self._local_capture = cv2.VideoCapture(0)
            if not self._local_capture.isOpened():
                logger.warning("Could not open webcam")
                self._local_capture = None
        except Exception as e:
            logger.warning(f"Local video start error: {e}")
    
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
        """Toggle local video on/off."""
        self._local_video_active = not self._local_video_active
        if self._local_video_active and HAVE_CV2:
            self._start_local_video()
        else:
            self._stop_local_video()
        return self._local_video_active
    
    def toggle_audio(self) -> bool:
        """Toggle audio on/off."""
        self._audio_active = not self._audio_active
        return self._audio_active
    
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
        """
        Display an image in the panel.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image displayed successfully
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return False
            
            self._state.set_mode(PanelMode.MEDIA_IMAGE)
            self._state.set_current_media(image_path)
            logger.info(f"Displaying image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Image display error: {e}")
            return False
    
    def display_video(self, video_path: str, loop: bool = False) -> bool:
        """
        Play a video in the panel.
        
        Args:
            video_path: Path to video file
            loop: Whether to loop playback
            
        Returns:
            True if video playback started
        """
        if not HAVE_CV2:
            logger.warning("OpenCV required for video playback")
            return False
        
        try:
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                return False
            
            self.stop_video()
            
            self._video_capture = cv2.VideoCapture(video_path)
            if not self._video_capture.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            self._state.set_mode(PanelMode.MEDIA_VIDEO)
            self._state.set_current_media(video_path)
            self._video_playing = True
            
            # Start playback thread
            self._playback_thread = threading.Thread(
                target=self._video_playback_loop,
                args=(loop,),
                daemon=True
            )
            self._playback_thread.start()
            
            logger.info(f"Playing video: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video playback error: {e}")
            return False
    
    def _video_playback_loop(self, loop: bool) -> None:
        """Video playback loop running in separate thread."""
        try:
            fps = self._video_capture.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = 1.0 / fps
            
            while self._video_playing and self._video_capture:
                start_time = time.time()
                
                ret, frame = self._video_capture.read()
                
                if not ret:
                    if loop:
                        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Emit frame to callbacks
                for callback in self._frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        logger.warning(f"Frame callback error: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self._video_playing = False
            self._state.restore_previous_mode()
            
        except Exception as e:
            logger.error(f"Video playback loop error: {e}")
            self._video_playing = False
    
    def stop_video(self) -> None:
        """Stop video playback."""
        self._video_playing = False
        if self._video_capture:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None
    
    def pause_video(self) -> None:
        """Pause video playback."""
        self._video_playing = False
    
    def resume_video(self) -> None:
        """Resume video playback."""
        if self._video_capture and not self._video_playing:
            self._video_playing = True
            self._playback_thread = threading.Thread(
                target=self._video_playback_loop,
                args=(False,),
                daemon=True
            )
            self._playback_thread.start()
    
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
            self._desktop_mirror: Optional[DesktopMirrorThread] = None
            
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
                        Path(os.path.join(os.path.dirname(__file__), "resources", "icons", "softdev0.png")),
                        Path(os.path.join(os.path.dirname(__file__), "icon.png")),
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
            """Switch to specified panel mode."""
            self._state.set_mode(mode)
            
            # Update button states
            self._btn_avatar.setChecked(mode in [PanelMode.AVATAR_2D, PanelMode.AVATAR_3D])
            self._btn_desktop.setChecked(mode == PanelMode.DESKTOP_MIRROR)
            self._btn_conference.setChecked(mode == PanelMode.VIDEO_CONFERENCE)
            
            # Switch display stack
            if mode in [PanelMode.AVATAR_2D, PanelMode.AVATAR_3D]:
                self._display_stack.setCurrentIndex(0)
                self._stop_desktop_mirror()
            elif mode == PanelMode.DESKTOP_MIRROR:
                self._display_stack.setCurrentIndex(1)
                self._start_desktop_mirror()
            elif mode == PanelMode.VIDEO_CONFERENCE:
                self._display_stack.setCurrentIndex(2)
                self._stop_desktop_mirror()
            elif mode in [PanelMode.MEDIA_IMAGE, PanelMode.MEDIA_VIDEO]:
                self._display_stack.setCurrentIndex(3)
                self._stop_desktop_mirror()
            
            self._status_set(f"Mode: {mode.name}")
        
        def _on_mode_change(self, old_mode: PanelMode, new_mode: PanelMode) -> None:
            """Handle mode change from state manager."""
            self.mode_changed.emit(new_mode.name)
        
        def _handle_mode_change(self, mode_name: str) -> None:
            """Handle mode change signal (thread-safe)."""
            try:
                mode = PanelMode[mode_name]
                self._switch_mode(mode)
            except KeyError:
                pass
        
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
            """Start desktop mirror capture."""
            if self._desktop_mirror is None:
                self._desktop_mirror = DesktopMirrorThread()
                if HAVE_QT:
                    self._desktop_mirror.frame_ready.connect(
                        lambda f: self.frame_updated.emit(f)
                    )
                else:
                    self._desktop_mirror.register_callback(
                        lambda f: self.frame_updated.emit(f)
                    )
            
            if not self._desktop_mirror.isRunning() if HAVE_QT else not self._desktop_mirror.is_alive():
                self._desktop_mirror.start_capture()
                self._status_set("Desktop mirror started")
        
        def _stop_desktop_mirror(self) -> None:
            """Stop desktop mirror capture."""
            if self._desktop_mirror:
                self._desktop_mirror.stop_capture()
                self._status_set("Desktop mirror stopped")
        
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
            panel_mode = PanelMode[mode.upper()]
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
        """Set avatar type (2d/3d)."""
        self._state.set_avatar_type(avatar_type)
        return {"success": True, "avatar_type": avatar_type}
    
    def set_zoom(self, zoom: float) -> Dict[str, Any]:
        """Set avatar zoom level."""
        self._state.set_zoom(zoom)
        return {"success": True, "zoom": self._state.get_zoom()}
    
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
        """Queue media for display."""
        self._state.queue_media(media_path, media_type)
        return {"success": True, "queued": media_path}


    def display_media_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept a normalized media result payload and display/queue it.

        Expected (best-effort) fields:
          - type: "image" | "video" | "audio" | "model3d" | "unknown"
          - uri/path/url/base64
          - autoplay, loop
        For this panel, we currently support image/video display & queue.
        """
        try:
            if not isinstance(result, dict):
                return {"success": False, "error": "result must be a dict"}

            mtype = (result.get("type") or result.get("media_type") or "unknown").lower()
            uri = result.get("path") or result.get("uri") or result.get("url")

            # Handle base64 image payloads (write to datasets temp folder)
            if not uri and result.get("base64") and mtype in ("image", "png", "jpg", "jpeg"):
                try:
                    raw = base64.b64decode(result.get("base64"))
                    out_dir = Path(DATASETS_DIR) / "generated"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = str(out_dir / f"media_{ts}.png")
                    with open(out_path, "wb") as f:
                        f.write(raw)
                    uri = out_path
                    mtype = "image"
                except Exception as e:
                    return {"success": False, "error": f"base64 decode failed: {e}"}

            if not uri:
                return {"success": False, "error": "No media uri/path provided"}

            autoplay = bool(result.get("autoplay", True))
            loop = bool(result.get("loop", False))
            queue_it = bool(result.get("queue", False))

            if mtype in ("image", "png", "jpg", "jpeg"):
                if queue_it:
                    return self.queue_media(uri, "image")
                return self.display_image(uri)

            if mtype in ("video", "mp4", "webm", "mov", "mkv"):
                if queue_it:
                    return self.queue_media(uri, "video")
                if autoplay:
                    return self.display_video(uri, loop=loop)
                # If not autoplay, just queue it so UI can decide
                return self.queue_media(uri, "video")

            # Unsupported media types (audio/3d) are acknowledged but not displayed here (yet)
            return {"success": False, "error": f"Unsupported media type for panel: {mtype}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_avatar_frame(self, width: int = 300, height: int = 300, format: str = "base64") -> Dict[str, Any]:
        """
        Get current avatar frame for WebUI rendering.
        
        Args:
            width: Frame width
            height: Frame height
            format: Output format ("base64" or "raw")
            
        Returns:
            Dictionary with frame data
        """
        frame = self._animation_engine.render_2d_frame(width, height)
        
        if frame is None:
            return {"success": False, "error": "Frame render failed"}
        
        if format == "base64":
            import io
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return {
                "success": True,
                "frame": img_base64,
                "format": "base64",
                "width": width,
                "height": height
            }
        else:
            return {
                "success": True,
                "frame": frame.tobytes(),
                "format": "raw",
                "width": width,
                "height": height
            }
    
    def start_lip_sync(self, duration: float = 0.0) -> Dict[str, Any]:
        """Start lip sync animation."""
        self._animation_engine.start_lip_sync(duration)
        return {"success": True, "lip_sync": True}
    
    def stop_lip_sync(self) -> Dict[str, Any]:
        """Stop lip sync animation."""
        self._animation_engine.stop_lip_sync()
        return {"success": True, "lip_sync": False}
    
    def set_panel_size(self, width: int, height: int) -> Dict[str, Any]:
        """Set panel size."""
        self._state.set_panel_size(width, height)
        return {"success": True, "size": self._state.get_panel_size()}
    
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
# END OF SarahMemoryAvatarPanel.py v8.0.0
# ====================================================================