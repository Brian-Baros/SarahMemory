"""--==The SarahMemory Project==--
File: SarahMemoryCanvasStudio.py
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

===============================================================================

SarahMemory Canvas Studio -Art & Graphics Editing Engine
=====================================================================

OVERVIEW:
---------
Canvas Studio is the premier creative art engine for SarahMemory, providing
professional-grade graphics editing, image generation, and rendering capabilities.
This module serves as the foundation for all visual creativity within the
SarahMemory ecosystem.

CAPABILITIES:
-------------
1. Advanced Image Creation & Editing
- Multi-layer composition with blend modes
- Professional color correction and grading
- HDR and tone mapping
- Advanced filters and effects

2. AI-Powered Art Generation
- Text-to-image synthesis
- Style transfer and artistic effects
- Intelligent upscaling and enhancement
- Content-aware editing

3. Professional Graphics Tools
- Vector graphics support
- Brush engine with custom brushes
- Selection tools and masking
- Transform operations (rotate, scale, skew, perspective)

4. Rendering Pipeline
- High-quality anti-aliasing
- Batch processing capabilities
- Export to multiple formats (PNG, JPG, WebP, TIFF, SVG)
- ICC color profile management

5. Effects & Filters
- Gaussian/Motion/Box blur
- Edge detection (Sobel, Canny, Laplacian)
- Artistic filters (oil paint, watercolor, sketch)
- Color adjustment (HSL, curves, levels)
- Noise generation and reduction

INTEGRATION POINTS:
------------------
- SarahMemoryGlobals: Configuration and paths
- SarahMemoryDatabase: Store artwork metadata and history
- SarahMemoryAiFunctions: AI-powered generation and enhancement
- SarahMemoryLLM: Natural language art direction
- UnifiedAvatarController: Generate avatar assets

FILE STRUCTURE:
--------------
{DATA_DIR}/
canvas/
projects/          # Saved project files (.scp format)
exports/           # Final rendered outputs
cache/             # Temporary processing files
templates/         # Preset templates and styles
brushes/           # Custom brush definitions

USAGE EXAMPLES:
--------------
Basic canvas creation
studio = CanvasStudio()
canvas = studio.create_canvas(1920, 1080, "My Artwork")

Add layers and effects
layer1 = canvas.add_layer("Background")
layer1.fill_color((100, 150, 200))
layer1.apply_gradient("linear", colors=[(0,0,0), (255,255,255)])

AI generation
ai_image = studio.generate_from_prompt(
"A serene landscape with mountains and lakes at sunset",
style="photorealistic",
quality="high"
)

Apply professional effects
canvas.apply_effect("gaussian_blur", radius=5)
canvas.color_correct(brightness=10, contrast=15, saturation=5)

Export final artwork
studio.export_canvas(canvas, "masterpiece.png", format="PNG", quality=95)

TECHNICAL SPECIFICATIONS:
------------------------
- Color Depth: 8-bit, 16-bit, 32-bit float per channel
- Color Spaces: RGB, RGBA, CMYK, HSL, HSV, LAB
- Max Canvas Size: 16,384 x 16,384 pixels (hardware dependent)
- Supported Formats: PNG, JPG, WebP, TIFF, BMP, TGA, SVG, PDF
- Layer Blend Modes: 20+ modes including normal, multiply, screen, overlay
- Undo History: Configurable (default 50 steps)

PERFORMANCE NOTES:
-----------------
- GPU acceleration available when supported
- Multi-threaded processing for batch operations
- Intelligent caching for faster re-rendering
- Progressive rendering for large canvases
- Memory-efficient streaming for huge images

ERROR HANDLING:
--------------
All functions implement comprehensive error handling and logging.
Failures are gracefully handled with fallbacks where appropriate.
All exceptions are logged to SarahMemory unified logging system.

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "creative_engine"
# CATEGORY = "graphics_and_art"
# USER_FACING = True
# UI_EXPOSURE = "candidate"
# DEPLOYMENT_TARGET = "addon"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "gpu_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "canvas_studio"
# FAMILY = "creative_studios"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = True
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Professional art and graphics editing engine for image creation, editing, rendering, filters, layer workflows, and AI-assisted visual generation."
# --- SARAHMETA END ---

import os
import sys

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

import json
import logging
import time
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from enum import Enum
import hashlib
import base64
from io import BytesIO

# Standard image processing
import numpy as np
import cv2

# Advanced imaging (attempt imports, fall back gracefully)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps, ImageChops
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageFilter = None
    ImageEnhance = None
    ImageOps = None
    ImageChops = None
    PIL_AVAILABLE = False
    logging.warning("[CanvasStudio] PIL/Pillow not available - some features disabled")

# Scientific computing
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("[CanvasStudio] SciPy not available - some filters disabled")

# Import SarahMemory globals
try:
    import SarahMemoryGlobals as SMG
    DEBUG_MODE = SMG.DEBUG_MODE
except ImportError:
    SMG = None  # type: ignore
    DEBUG_MODE = True
    logging.warning("[CanvasStudio] Running in standalone mode without SarahMemoryGlobals")

DATA_DIR = str(getattr(SMG, "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))) if SMG is not None else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Version information
CANVAS_STUDIO_VERSION = "2.0.0"
CANVAS_STUDIO_BUILD = "20251204"

# Directory structure
try:
    # Prefer centralized v9.0.0 paths
    CANVAS_DIR = SMG.CANVAS_DIR
    CANVAS_PROJECTS_DIR = SMG.CANVAS_PROJECTS_DIR
    CANVAS_EXPORTS_DIR = SMG.CANVAS_EXPORTS_DIR
    CANVAS_CACHE_DIR = SMG.CANVAS_CACHE_DIR
    CANVAS_BRUSHES_DIR = getattr(SMG, "CANVAS_BRUSHES_DIR", os.path.join(CANVAS_DIR, "brushes"))
    CANVAS_TEMPLATES_DIR = SMG.CANVAS_TEMPLATES_DIR
except Exception:
    CANVAS_DIR = os.path.join(DATA_DIR, "canvas")
    CANVAS_PROJECTS_DIR = os.path.join(CANVAS_DIR, "projects")
    CANVAS_EXPORTS_DIR = os.path.join(CANVAS_DIR, "exports")
    CANVAS_CACHE_DIR = os.path.join(CANVAS_DIR, "cache")
    CANVAS_BRUSHES_DIR = os.path.join(CANVAS_DIR, "brushes")
    CANVAS_TEMPLATES_DIR = os.path.join(CANVAS_DIR, "templates")

# Directories are created lazily by CanvasStudio.__init__ to keep imports side-effect free.
CANVAS_BRUSHES_DIR = os.path.join(CANVAS_DIR, "brushes")

# Canvas limitations
MAX_CANVAS_WIDTH = 16384
MAX_CANVAS_HEIGHT = 16384
MIN_CANVAS_WIDTH = 1
MIN_CANVAS_HEIGHT = 1
DEFAULT_CANVAS_WIDTH = 1920
DEFAULT_CANVAS_HEIGHT = 1080

# Color depth options
COLOR_DEPTH_8BIT = 8
COLOR_DEPTH_16BIT = 16
COLOR_DEPTH_32BIT = 32

# Supported file formats
SUPPORTED_EXPORT_FORMATS = ["PNG", "JPG", "JPEG", "WEBP", "TIFF", "BMP", "TGA", "SVG", "PDF"]
SUPPORTED_IMPORT_FORMATS = ["PNG", "JPG", "JPEG", "WEBP", "TIFF", "BMP", "TGA", "GIF"]

# Default settings
DEFAULT_UNDO_HISTORY = 50
DEFAULT_JPEG_QUALITY = 90
DEFAULT_PNG_COMPRESSION = 6


# ============================================================================
# ENUMERATIONS
# ============================================================================

class BlendMode(Enum):
    """Layer blend modes for compositing"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    LUMINOSITY = "luminosity"


class FilterType(Enum):
    """Available image filters"""
    BLUR_GAUSSIAN = "gaussian_blur"
    BLUR_BOX = "box_blur"
    BLUR_MOTION = "motion_blur"
    SHARPEN = "sharpen"
    EDGE_SOBEL = "edge_sobel"
    EDGE_CANNY = "edge_canny"
    EDGE_LAPLACIAN = "edge_laplacian"
    EMBOSS = "emboss"
    CONTOUR = "contour"
    FIND_EDGES = "find_edges"
    NOISE_GAUSSIAN = "noise_gaussian"
    NOISE_SALT_PEPPER = "noise_salt_pepper"
    DENOISE = "denoise"
    OIL_PAINT = "oil_paint"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"
    CARTOON = "cartoon"
    VIGNETTE = "vignette"
    SEPIA = "sepia"
    VINTAGE = "vintage"


class GradientType(Enum):
    """Gradient fill types"""
    LINEAR = "linear"
    RADIAL = "radial"
    ANGULAR = "angular"
    REFLECTED = "reflected"
    DIAMOND = "diamond"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_canvas_directories():
    """Create all required Canvas Studio directories"""
    directories = [
        CANVAS_DIR,
        CANVAS_PROJECTS_DIR,
        CANVAS_EXPORTS_DIR,
        CANVAS_CACHE_DIR,
        CANVAS_TEMPLATES_DIR,
        CANVAS_BRUSHES_DIR
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to create directory {directory}: {e}")


def validate_canvas_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Validate and clamp canvas dimensions to acceptable ranges"""
    width = max(MIN_CANVAS_WIDTH, min(width, MAX_CANVAS_WIDTH))
    height = max(MIN_CANVAS_HEIGHT, min(height, MAX_CANVAS_HEIGHT))
    return width, height


def generate_unique_id() -> str:
    """Generate a unique identifier for canvas objects"""
    timestamp = datetime.now().isoformat()
    random_component = os.urandom(8)
    combined = f"{timestamp}{random_component}".encode()
    return hashlib.sha256(combined).hexdigest()[:16]


def clamp_color(value: Union[int, float], depth: int = 8) -> int:
    """Clamp color values to valid range based on color depth"""
    if depth == 8:
        return max(0, min(255, int(value)))
    elif depth == 16:
        return max(0, min(65535, int(value)))
    else:
        return max(0.0, min(1.0, float(value)))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV color space"""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    s = 0 if max_c == 0 else (diff / max_c)
    v = max_c
    
    return h, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV to RGB color space"""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


# ============================================================================
# INTERNAL PIXEL / CONTRACT HELPERS
# ============================================================================

def _depth_max_value(depth: int) -> float:
    """Return the numeric full-scale value for a layer depth."""
    if int(depth) == COLOR_DEPTH_16BIT:
        return 65535.0
    if int(depth) == COLOR_DEPTH_32BIT:
        return 1.0
    return 255.0


def _depth_dtype(depth: int):
    if int(depth) == COLOR_DEPTH_16BIT:
        return np.uint16
    if int(depth) == COLOR_DEPTH_32BIT:
        return np.float32
    return np.uint8


def _coerce_rgba_color_for_depth(color: Tuple[int, int, int, int], depth: int) -> np.ndarray:
    """Coerce an RGBA color tuple into the layer's native depth.

    User-facing colors may be supplied as conventional 0-255 RGBA values. For
    16-bit layers those values are scaled up; for 32-bit float layers they are
    normalized into 0.0-1.0. Existing full-range 16-bit / normalized float values
    are also accepted.
    """
    if color is None:
        color = (255, 255, 255, 255)
    if len(color) == 3:
        color = (*color, 255)
    arr = np.asarray(color[:4], dtype=np.float32)
    depth = int(depth)
    if depth == COLOR_DEPTH_32BIT:
        if float(np.nanmax(arr)) > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    if depth == COLOR_DEPTH_16BIT:
        if float(np.nanmax(arr)) <= 255.0:
            arr = arr * 257.0
        return np.clip(arr, 0, 65535).astype(np.uint16)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _rgba_to_float01(data: np.ndarray, depth: int) -> np.ndarray:
    """Convert native RGBA data into float32 0..1 RGBA."""
    if data is None:
        return np.zeros((1, 1, 4), dtype=np.float32)
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, _depth_max_value(depth))], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), _depth_max_value(depth), dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=2)
    elif arr.ndim != 3 or arr.shape[2] < 4:
        raise ValueError("image_data_must_be_rgba_compatible")
    arr = arr[:, :, :4].astype(np.float32, copy=False)
    max_value = _depth_max_value(depth)
    if max_value <= 0:
        max_value = 255.0
    return np.clip(arr / max_value, 0.0, 1.0).astype(np.float32)


def _float01_to_rgba_depth(data: np.ndarray, depth: int) -> np.ndarray:
    """Convert float32 0..1 RGBA data into the requested native depth."""
    arr = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
    if int(depth) == COLOR_DEPTH_32BIT:
        return arr.astype(np.float32)
    if int(depth) == COLOR_DEPTH_16BIT:
        return np.round(arr * 65535.0).astype(np.uint16)
    return np.round(arr * 255.0).astype(np.uint8)


def _rgba_to_uint8(data: np.ndarray, depth: int) -> np.ndarray:
    return _float01_to_rgba_depth(_rgba_to_float01(data, depth), COLOR_DEPTH_8BIT)


def _rgb_uint8_from_layer(layer: "CanvasLayer") -> Tuple[np.ndarray, np.ndarray]:
    rgba = _rgba_to_uint8(layer.data, layer.depth)
    return rgba[:, :, :3].copy(), rgba[:, :, 3].copy()


def _write_rgb_uint8_to_layer(layer: "CanvasLayer", rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if alpha is None:
        try:
            alpha = _rgba_to_uint8(layer.data, layer.depth)[:, :, 3]
        except Exception:
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8)])
    layer.data = _float01_to_rgba_depth(rgba.astype(np.float32) / 255.0, layer.depth)


def _normalize_export_format(fmt: str) -> str:
    fmt = (fmt or "PNG").strip().upper().lstrip('.')
    if fmt == "JPG":
        return "JPEG"
    return fmt


def _project_layer_dir(filepath: str) -> str:
    root, ext = os.path.splitext(filepath)
    if ext.lower() != ".scp":
        root = filepath
    return root + "_layers"


def _parse_dt(value: Any, default: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    return default or datetime.now()


def _safe_imwrite(path: str, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ok = cv2.imwrite(path, image, params or [])
        return bool(ok) and os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception as exc:
        logging.error(f"[CanvasStudio] Image write failed for {path}: {exc}")
        return False



# ============================================================================
# LAYER CLASS
# ============================================================================

class CanvasLayer:
    """
    Represents a single layer in the canvas composition.
    
    Each layer has its own image data, opacity, blend mode, and transformations.
    Layers can be independently edited, hidden, locked, and reordered.
    """
    
    def __init__(self, name: str, width: int, height: int, depth: int = 8):
        """
        Initialize a new canvas layer
        
        Args:
            name: Layer name for identification
            width: Layer width in pixels
            height: Layer height in pixels
            depth: Color depth (8, 16, or 32 bits per channel)
        """
        self.id = generate_unique_id()
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        
        # Initialize layer data based on depth
        if depth == 32:
            self.data = np.zeros((height, width, 4), dtype=np.float32)
        elif depth == 16:
            self.data = np.zeros((height, width, 4), dtype=np.uint16)
        else:
            self.data = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Layer properties
        self.opacity = 100  # 0-100
        self.blend_mode = BlendMode.NORMAL
        self.visible = True
        self.locked = False
        
        # Transform properties
        self.position = (0, 0)  # x, y offset
        self.rotation = 0  # degrees
        self.scale = (1.0, 1.0)  # x, y scale factors
        
        # Metadata
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        
        logging.info(f"[CanvasStudio] Created layer '{name}' ({width}x{height}, {depth}-bit)")
    
    def fill_color(self, color: Tuple[int, int, int, int] = None):
        """Fill the entire layer with a solid RGBA color using depth-safe scaling."""
        self.data[:] = _coerce_rgba_color_for_depth(color, self.depth)
        self.modified_at = datetime.now()
        logging.debug(f"[CanvasStudio] Filled layer '{self.name}' with color {color}")

    def clear(self):
        """Clear the layer (make it fully transparent)"""
        self.data[:] = 0
        self.modified_at = datetime.now()
        logging.debug(f"[CanvasStudio] Cleared layer '{self.name}'")
    
    def apply_opacity(self, opacity: int):
        """Set layer opacity (0-100)"""
        self.opacity = max(0, min(100, opacity))
        self.modified_at = datetime.now()
        logging.debug(f"[CanvasStudio] Set opacity of layer '{self.name}' to {self.opacity}%")
    
    def set_blend_mode(self, mode: BlendMode):
        """Set the blend mode for this layer"""
        self.blend_mode = mode
        self.modified_at = datetime.now()
        logging.debug(f"[CanvasStudio] Set blend mode of layer '{self.name}' to {mode.value}")
    
    def apply_gradient(self, gradient_type: str, colors: List[Tuple[int, int, int]], 
                       angle: float = 0, center: Tuple[float, float] = None):
        """Apply a depth-safe gradient fill to the layer.

        Supported gradient types: linear, radial, angular, reflected, diamond.
        Unknown gradient types raise ValueError instead of reporting false success.
        """
        if not colors:
            raise ValueError("gradient_requires_at_least_one_color")
        if center is None:
            center = (0.5, 0.5)

        gradient_type = str(gradient_type or "linear").lower()
        height, width = self.data.shape[:2]
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = float(center[0]) * max(1, width - 1), float(center[1]) * max(1, height - 1)

        if gradient_type == "linear":
            angle_rad = np.radians(float(angle))
            projection = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            t = (projection - projection.min()) / max(1e-6, float(projection.max() - projection.min()))
        elif gradient_type == "radial":
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            t = dist / max(1e-6, float(dist.max()))
        elif gradient_type == "angular":
            theta = (np.arctan2(y - cy, x - cx) + np.pi) / (2.0 * np.pi)
            t = theta
        elif gradient_type == "reflected":
            angle_rad = np.radians(float(angle))
            projection = x * np.cos(angle_rad) + y * np.sin(angle_rad)
            projection = (projection - projection.min()) / max(1e-6, float(projection.max() - projection.min()))
            t = np.abs((projection * 2.0) - 1.0)
        elif gradient_type == "diamond":
            dist = np.abs(x - cx) / max(1.0, width) + np.abs(y - cy) / max(1.0, height)
            t = dist / max(1e-6, float(dist.max()))
        else:
            raise ValueError(f"unsupported_gradient_type:{gradient_type}")

        t = np.clip(t, 0.0, 1.0)
        stops = np.asarray([_coerce_rgba_color_for_depth((*c[:3], c[3] if len(c) > 3 else 255), COLOR_DEPTH_8BIT) for c in colors], dtype=np.float32) / 255.0
        if len(stops) == 1:
            rgba = np.broadcast_to(stops[0], (height, width, 4)).copy()
        else:
            scaled = t * (len(stops) - 1)
            idx = np.floor(scaled).astype(np.int32)
            idx = np.clip(idx, 0, len(stops) - 2)
            local_t = (scaled - idx)[..., None]
            rgba = stops[idx] * (1.0 - local_t) + stops[idx + 1] * local_t

        self.data = _float01_to_rgba_depth(rgba, self.depth)
        self.modified_at = datetime.now()
        logging.debug(f"[CanvasStudio] Applied {gradient_type} gradient to layer '{self.name}'")

    def _interpolate_colors(self, colors: List[Tuple[int, int, int]], t: float) -> Tuple[int, int, int]:
        """Interpolate between color stops"""
        if len(colors) < 2:
            return colors[0] if colors else (0, 0, 0)
        
        # Find the two colors to interpolate between
        segment = t * (len(colors) - 1)
        idx = int(segment)
        local_t = segment - idx
        
        if idx >= len(colors) - 1:
            return colors[-1]
        
        c1 = colors[idx]
        c2 = colors[idx + 1]
        
        r = int(c1[0] + (c2[0] - c1[0]) * local_t)
        g = int(c1[1] + (c2[1] - c1[1]) * local_t)
        b = int(c1[2] + (c2[2] - c1[2]) * local_t)
        
        return (r, g, b)
    
    def to_dict(self) -> Dict:
        """Serialize layer to dictionary for saving"""
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode.value,
            "visible": self.visible,
            "locked": self.locked,
            "position": self.position,
            "rotation": self.rotation,
            "scale": self.scale,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat()
        }


# ============================================================================
# CANVAS CLASS
# ============================================================================

class Canvas:
    """
    Main canvas object representing a complete artwork with multiple layers.
    
    The Canvas class manages the layer stack, handles composition, and provides
    high-level operations for the entire artwork.
    """
    
    def __init__(self, name: str, width: int, height: int, depth: int = 8, 
                 background_color: Tuple[int, int, int, int] = None):
        """
        Initialize a new canvas
        
        Args:
            name: Canvas/project name
            width: Canvas width in pixels
            height: Canvas height in pixels
            depth: Color depth (8, 16, or 32 bits per channel)
            background_color: Initial background color (RGBA)
        """
        self.id = generate_unique_id()
        self.name = name
        self.width, self.height = validate_canvas_dimensions(width, height)
        self.depth = depth
        
        # Initialize layers
        self.layers: List[CanvasLayer] = []
        self.active_layer_index = 0
        
        # Create background layer
        bg_layer = CanvasLayer("Background", self.width, self.height, self.depth)
        if background_color:
            bg_layer.fill_color(background_color)
        else:
            bg_layer.fill_color((255, 255, 255, 255))  # White background
        self.layers.append(bg_layer)
        
        # Metadata
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self.author = os.getenv("USER", "SarahMemory")
        
        # Undo/redo history
        self.history = []
        self.history_index = -1
        self.max_history = DEFAULT_UNDO_HISTORY
        
        logging.info(f"[CanvasStudio] Created canvas '{name}' ({width}x{height}, {depth}-bit)")
    
    def add_layer(self, name: str, position: int = None) -> CanvasLayer:
        """
        Add a new layer to the canvas
        
        Args:
            name: Layer name
            position: Insert position (None = top of stack)
        
        Returns:
            The newly created layer
        """
        self.push_history("add_layer")
        layer = CanvasLayer(name, self.width, self.height, self.depth)
        
        if position is None:
            self.layers.append(layer)
            self.active_layer_index = len(self.layers) - 1
        else:
            position = max(0, min(position, len(self.layers)))
            self.layers.insert(position, layer)
            self.active_layer_index = position
        
        self.modified_at = datetime.now()
        logging.info(f"[CanvasStudio] Added layer '{name}' to canvas '{self.name}'")
        return layer
    
    def remove_layer(self, layer_index: int) -> bool:
        """Remove a layer from the canvas."""
        if 0 <= layer_index < len(self.layers):
            if len(self.layers) > 1:
                self.push_history("remove_layer")
                removed_layer = self.layers.pop(layer_index)
                self.active_layer_index = min(self.active_layer_index, len(self.layers) - 1)
                self.modified_at = datetime.now()
                logging.info(f"[CanvasStudio] Removed layer '{removed_layer.name}' from canvas '{self.name}'")
                return True
            logging.warning(f"[CanvasStudio] Cannot remove last layer from canvas '{self.name}'")
            return False
        return False
    
    def get_active_layer(self) -> Optional[CanvasLayer]:
        """Get the currently active layer."""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
    
    def set_active_layer(self, layer_index: int) -> bool:
        """Set the active layer by index."""
        if 0 <= layer_index < len(self.layers):
            self.active_layer_index = layer_index
            logging.debug(f"[CanvasStudio] Set active layer to index {layer_index}")
            return True
        return False

    def _history_snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "active_layer_index": self.active_layer_index,
            "author": self.author,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "layers": [
                {
                    "id": layer.id,
                    "name": layer.name,
                    "width": layer.width,
                    "height": layer.height,
                    "depth": layer.depth,
                    "opacity": layer.opacity,
                    "blend_mode": layer.blend_mode,
                    "visible": layer.visible,
                    "locked": layer.locked,
                    "position": tuple(layer.position),
                    "rotation": layer.rotation,
                    "scale": tuple(layer.scale),
                    "created_at": layer.created_at,
                    "modified_at": layer.modified_at,
                    "data": layer.data.copy(),
                }
                for layer in self.layers
            ],
        }

    def _restore_history_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.id = snapshot.get("id", self.id)
        self.name = snapshot.get("name", self.name)
        self.width = int(snapshot.get("width", self.width))
        self.height = int(snapshot.get("height", self.height))
        self.depth = int(snapshot.get("depth", self.depth))
        self.active_layer_index = int(snapshot.get("active_layer_index", 0))
        self.author = snapshot.get("author", self.author)
        self.created_at = snapshot.get("created_at", self.created_at)
        self.modified_at = datetime.now()
        restored: List[CanvasLayer] = []
        for item in snapshot.get("layers", []):
            layer = CanvasLayer(item.get("name", "Layer"), int(item.get("width", self.width)), int(item.get("height", self.height)), int(item.get("depth", self.depth)))
            layer.id = item.get("id", layer.id)
            layer.opacity = int(item.get("opacity", 100))
            try:
                layer.blend_mode = item.get("blend_mode") if isinstance(item.get("blend_mode"), BlendMode) else BlendMode(item.get("blend_mode", BlendMode.NORMAL.value))
            except Exception:
                layer.blend_mode = BlendMode.NORMAL
            layer.visible = bool(item.get("visible", True))
            layer.locked = bool(item.get("locked", False))
            layer.position = tuple(item.get("position", (0, 0)))
            layer.rotation = float(item.get("rotation", 0))
            layer.scale = tuple(item.get("scale", (1.0, 1.0)))
            layer.created_at = item.get("created_at", datetime.now())
            layer.modified_at = item.get("modified_at", datetime.now())
            layer.data = np.asarray(item.get("data", layer.data)).copy()
            restored.append(layer)
        if restored:
            self.layers = restored
        self.active_layer_index = max(0, min(self.active_layer_index, len(self.layers) - 1))

    def push_history(self, label: str = "edit") -> None:
        """Capture a bounded undo snapshot."""
        try:
            if self.max_history <= 0:
                return
            # Drop redo branch.
            if self.history_index < len(self.history) - 1:
                self.history = self.history[:self.history_index + 1]
            snap = self._history_snapshot()
            snap["label"] = label
            self.history.append(snap)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.history_index = len(self.history) - 1
        except Exception as exc:
            logging.warning(f"[CanvasStudio] Failed to push history snapshot: {exc}")

    def undo(self) -> bool:
        """Restore the previous canvas state when available."""
        if self.history_index < 0 or not self.history:
            return False
        current = self._history_snapshot()
        snapshot = self.history[self.history_index]
        self._restore_history_snapshot(snapshot)
        self.history_index -= 1
        if self.history_index == len(self.history) - 2:
            self.history.append(current)
        return True

    def redo(self) -> bool:
        """Restore a redone state when available."""
        next_index = self.history_index + 1
        if not (0 <= next_index < len(self.history)):
            return False
        self._restore_history_snapshot(self.history[next_index])
        self.history_index = next_index
        return True

    def resize(self, width: int, height: int) -> bool:
        """Resize the canvas and every layer."""
        width, height = validate_canvas_dimensions(int(width), int(height))
        if width == self.width and height == self.height:
            return True
        self.push_history("resize")
        for layer in self.layers:
            interp = cv2.INTER_AREA if width < layer.width or height < layer.height else cv2.INTER_LINEAR
            layer.data = cv2.resize(layer.data, (width, height), interpolation=interp)
            layer.width = width
            layer.height = height
            layer.modified_at = datetime.now()
        self.width = width
        self.height = height
        self.modified_at = datetime.now()
        return True

    @staticmethod
    def _blend_rgb(base_rgb: np.ndarray, top_rgb: np.ndarray, mode: BlendMode) -> np.ndarray:
        """Blend two float RGB arrays before alpha compositing."""
        b = np.clip(base_rgb, 0.0, 1.0)
        t = np.clip(top_rgb, 0.0, 1.0)
        mode_value = mode.value if isinstance(mode, BlendMode) else str(mode)
        if mode_value == BlendMode.MULTIPLY.value:
            return b * t
        if mode_value == BlendMode.SCREEN.value:
            return 1.0 - (1.0 - b) * (1.0 - t)
        if mode_value == BlendMode.OVERLAY.value:
            return np.where(b <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
        if mode_value == BlendMode.HARD_LIGHT.value:
            return np.where(t <= 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
        if mode_value == BlendMode.SOFT_LIGHT.value:
            return (1.0 - 2.0 * t) * b * b + 2.0 * t * b
        if mode_value == BlendMode.DARKEN.value:
            return np.minimum(b, t)
        if mode_value == BlendMode.LIGHTEN.value:
            return np.maximum(b, t)
        if mode_value == BlendMode.COLOR_DODGE.value:
            return np.where(t >= 1.0, 1.0, np.minimum(1.0, b / np.maximum(1e-6, 1.0 - t)))
        if mode_value == BlendMode.COLOR_BURN.value:
            return np.where(t <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(1e-6, t)))
        if mode_value == BlendMode.LINEAR_DODGE.value:
            return np.minimum(1.0, b + t)
        if mode_value == BlendMode.LINEAR_BURN.value:
            return np.maximum(0.0, b + t - 1.0)
        if mode_value == BlendMode.DIFFERENCE.value:
            return np.abs(b - t)
        if mode_value == BlendMode.EXCLUSION.value:
            return b + t - 2.0 * b * t
        # Hue/saturation/color/luminosity need perceptual color-space handling; keep them safe.
        return t

    def _transformed_layer_rgba(self, layer: CanvasLayer) -> np.ndarray:
        """Return a layer as canvas-sized float RGBA, with transforms applied."""
        rgba = _rgba_to_float01(layer.data, layer.depth)
        if rgba.shape[:2] != (self.height, self.width):
            rgba = cv2.resize(rgba, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        sx, sy = layer.scale if isinstance(layer.scale, (tuple, list)) and len(layer.scale) == 2 else (1.0, 1.0)
        px, py = layer.position if isinstance(layer.position, (tuple, list)) and len(layer.position) == 2 else (0, 0)
        rot = float(layer.rotation or 0.0)
        if abs(float(sx) - 1.0) < 1e-6 and abs(float(sy) - 1.0) < 1e-6 and abs(rot) < 1e-6 and int(px) == 0 and int(py) == 0:
            return rgba
        center = (self.width / 2.0, self.height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, rot, 1.0)
        matrix[0, 0] *= float(sx)
        matrix[0, 1] *= float(sx)
        matrix[1, 0] *= float(sy)
        matrix[1, 1] *= float(sy)
        matrix[0, 2] += float(px)
        matrix[1, 2] += float(py)
        return cv2.warpAffine(rgba, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def flatten(self) -> np.ndarray:
        """Flatten visible layers into one depth-safe RGBA composite."""
        if not self.layers:
            return np.zeros((self.height, self.width, 4), dtype=_depth_dtype(self.depth))
        result = np.zeros((self.height, self.width, 4), dtype=np.float32)
        for layer in self.layers:
            if not layer.visible:
                continue
            src = self._transformed_layer_rgba(layer)
            layer_alpha = np.clip(float(layer.opacity) / 100.0, 0.0, 1.0)
            src_a = np.clip(src[:, :, 3:4] * layer_alpha, 0.0, 1.0)
            if np.max(src_a) <= 0.0:
                continue
            dst_a = result[:, :, 3:4]
            blend_rgb = self._blend_rgb(result[:, :, :3], src[:, :, :3], layer.blend_mode)
            out_a = src_a + dst_a * (1.0 - src_a)
            out_rgb_premul = blend_rgb * src_a + result[:, :, :3] * dst_a * (1.0 - src_a)
            result[:, :, :3] = np.where(out_a > 1e-6, out_rgb_premul / np.maximum(out_a, 1e-6), 0.0)
            result[:, :, 3:4] = out_a
        logging.debug(f"[CanvasStudio] Flattened {len(self.layers)} layers")
        return _float01_to_rgba_depth(result, self.depth)
    
    def merge_layers(self, layer1_index: int, layer2_index: int) -> bool:
        """Merge layer2 onto layer1 using the same compositor as flatten."""
        if not (0 <= layer1_index < len(self.layers) and 0 <= layer2_index < len(self.layers) and layer1_index != layer2_index):
            return False
        self.push_history("merge_layers")
        layer1 = self.layers[layer1_index]
        layer2 = self.layers[layer2_index]
        temp = Canvas("_merge", self.width, self.height, self.depth, (0, 0, 0, 0))
        temp.layers = []
        temp.layers.append(layer1)
        temp.layers.append(layer2)
        merged = temp.flatten()
        layer1.data = merged
        layer1.modified_at = datetime.now()
        self.layers.pop(layer2_index)
        if self.active_layer_index >= layer2_index:
            self.active_layer_index = max(0, self.active_layer_index - 1)
        self.modified_at = datetime.now()
        logging.info(f"[CanvasStudio] Merged layers in canvas '{self.name}'")
        return True
    
    def apply_effect(self, effect_type: str, **kwargs) -> bool:
        """Apply an effect to the active layer. Returns False for unsupported effects."""
        layer = self.get_active_layer()
        if not layer:
            logging.warning("[CanvasStudio] No active layer to apply effect")
            return False
        effect_type = str(effect_type or "").lower()
        rgb, alpha = _rgb_uint8_from_layer(layer)
        applied = True
        try:
            if effect_type == "gaussian_blur":
                radius = max(0.1, float(kwargs.get("radius", 5)))
                out = cv2.GaussianBlur(rgb, (0, 0), radius)
            elif effect_type == "box_blur":
                k = max(1, int(kwargs.get("kernel", kwargs.get("radius", 5))))
                k = k if k % 2 == 1 else k + 1
                out = cv2.blur(rgb, (k, k))
            elif effect_type == "motion_blur":
                k = max(3, int(kwargs.get("kernel", 9)))
                k = k if k % 2 == 1 else k + 1
                kernel = np.zeros((k, k), dtype=np.float32)
                kernel[k // 2, :] = 1.0 / k
                out = cv2.filter2D(rgb, -1, kernel)
            elif effect_type == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                out = cv2.filter2D(rgb, -1, kernel)
            elif effect_type == "edge_sobel":
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sx ** 2 + sy ** 2)
                max_edge = float(edges.max())
                edges = np.uint8(edges / max_edge * 255) if max_edge > 0 else np.zeros_like(gray, dtype=np.uint8)
                out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            elif effect_type == "edge_canny":
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, int(kwargs.get("threshold1", 100)), int(kwargs.get("threshold2", 200)))
                out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            elif effect_type == "edge_laplacian":
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                out = cv2.cvtColor(np.uint8(np.clip(np.abs(lap), 0, 255)), cv2.COLOR_GRAY2RGB)
            elif effect_type == "emboss":
                kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
                out = cv2.filter2D(rgb, -1, kernel) + 128
            elif effect_type in ("contour", "find_edges"):
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 80, 160)
                out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            elif effect_type == "noise_gaussian":
                sigma = float(kwargs.get("sigma", 12.0))
                noise = np.random.normal(0.0, sigma, rgb.shape).astype(np.float32)
                out = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            elif effect_type == "noise_salt_pepper":
                amount = float(kwargs.get("amount", 0.01))
                out = rgb.copy()
                mask = np.random.random(rgb.shape[:2])
                out[mask < amount / 2.0] = 0
                out[mask > 1.0 - amount / 2.0] = 255
            elif effect_type == "denoise":
                out = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, 7, 7, 7, 21)
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            elif effect_type == "oil_paint":
                if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "oilPainting"):
                    out = cv2.xphoto.oilPainting(rgb, 7, 1)
                else:
                    out = cv2.bilateralFilter(rgb, 9, 80, 80)
            elif effect_type == "watercolor":
                smooth = cv2.bilateralFilter(rgb, 9, 75, 75)
                out = cv2.addWeighted(smooth, 0.82, cv2.GaussianBlur(smooth, (0, 0), 1.2), 0.18, 0)
            elif effect_type == "sketch":
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blur, scale=256)
                out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            elif effect_type == "cartoon":
                smooth = cv2.bilateralFilter(rgb, 9, 90, 90)
                edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 80, 160)
                edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)
                out = cv2.bitwise_and(smooth, edges)
            elif effect_type == "vignette":
                rows, cols = rgb.shape[:2]
                kernel_x = cv2.getGaussianKernel(cols, cols / 2.5)
                kernel_y = cv2.getGaussianKernel(rows, rows / 2.5)
                mask = kernel_y @ kernel_x.T
                mask = mask / max(mask.max(), 1e-6)
                out = np.clip(rgb.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)
            elif effect_type == "sepia":
                matrix = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], dtype=np.float32)
                out = np.clip(rgb.astype(np.float32) @ matrix.T, 0, 255).astype(np.uint8)
            elif effect_type == "vintage":
                matrix = np.array([[1.08, 0.05, 0.02], [0.03, 0.95, 0.04], [0.02, 0.06, 0.82]], dtype=np.float32)
                out = np.clip(rgb.astype(np.float32) @ matrix.T + np.array([8, 4, -6]), 0, 255).astype(np.uint8)
            else:
                logging.warning(f"[CanvasStudio] Unsupported effect: {effect_type}")
                return False
            self.push_history(f"effect:{effect_type}")
            _write_rgb_uint8_to_layer(layer, out, alpha)
            layer.modified_at = datetime.now()
            self.modified_at = datetime.now()
            logging.info(f"[CanvasStudio] Applied effect '{effect_type}' to layer '{layer.name}'")
            return applied
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to apply effect '{effect_type}': {e}")
            return False
    
    def color_correct(self, brightness: int = 0, contrast: int = 0, saturation: int = 0) -> bool:
        """Apply RGB-only color correction to the active layer while preserving alpha."""
        layer = self.get_active_layer()
        if not layer:
            return False
        try:
            rgb, alpha = _rgb_uint8_from_layer(layer)
            out = rgb.astype(np.float32)
            if brightness != 0:
                out += float(brightness)
            if contrast != 0:
                c = max(-255.0, min(255.0, float(contrast)))
                factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
                out = factor * (out - 128.0) + 128.0
            out = np.clip(out, 0, 255).astype(np.uint8)
            if saturation != 0:
                hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + float(saturation) / 100.0), 0, 255)
                out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            self.push_history("color_correct")
            _write_rgb_uint8_to_layer(layer, out, alpha)
            layer.modified_at = datetime.now()
            self.modified_at = datetime.now()
            logging.info(f"[CanvasStudio] Applied color correction to layer '{layer.name}'")
            return True
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to apply color correction: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Serialize canvas to dictionary for saving."""
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "layers": [layer.to_dict() for layer in self.layers],
            "active_layer_index": self.active_layer_index,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "author": self.author,
            "version": CANVAS_STUDIO_VERSION,
        }


# ============================================================================
# CANVAS STUDIO - Main Class
# ============================================================================

class CanvasStudio:
    """
    Main Canvas Studio interface providing high-level art creation and editing capabilities.
    
    This class serves as the primary API for all Canvas Studio operations,
    managing canvases, rendering, export, and AI-powered generation.
    """
    
    def __init__(self):
        """Initialize Canvas Studio"""
        ensure_canvas_directories()
        self.canvases: Dict[str, Canvas] = {}
        self.active_canvas_id: Optional[str] = None

        # Persistent live-avatar renderer state.  This is presentation-only RAM
        # history; CanvasStudio has no cognitive or execution authority.
        self._live_avatar_lock = threading.RLock()
        self._live_avatar_history: Optional[np.ndarray] = None
        self._live_avatar_previous_landmarks: Dict[str, Tuple[float, float]] = {}
        self._live_avatar_previous_parameters: Dict[str, Any] = {}
        self._live_avatar_frame_id = 0
        self._live_avatar_last_health: Dict[str, Any] = {}
        self._live_avatar_streams: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        # Bounded stat-aware RAM cache avoids decoding the same identity artwork
        # every render frame. Cached data is presentation-only and invalidates on
        # file metadata changes; it grants no execution or memory authority.
        self._live_avatar_reference_cache: Dict[Tuple[str, int, int, int, int], np.ndarray] = {}
        self._live_avatar_reference_cache_max = 4
        
        logging.info(f"[CanvasStudio] Initialized v{CANVAS_STUDIO_VERSION} (Build {CANVAS_STUDIO_BUILD})")
    
    # ---------------------------------------------------------------------
    # Persistent Live Avatar Renderer
    # ---------------------------------------------------------------------
    @staticmethod
    def _live_avatar_landmark_atlas() -> Dict[str, Tuple[float, float]]:
        """Normalized reference topology; identity artwork remains external.

        These are topology/anchor rails rather than animation frames.  Projects
        may later replace this default atlas with calibrated landmarks derived
        from the user's existing SarahMemory avatar reference artwork.
        """
        return {
            "head_top": (0.50, 0.10),
            "temple_left": (0.31, 0.25), "temple_right": (0.69, 0.25),
            "left_brow": (0.40, 0.30), "right_brow": (0.60, 0.30),
            "left_eye": (0.40, 0.36), "right_eye": (0.60, 0.36),
            "nose_bridge": (0.50, 0.35), "nose_tip": (0.50, 0.47),
            "mouth_left": (0.43, 0.55), "upper_lip": (0.50, 0.54),
            "mouth_right": (0.57, 0.55), "lower_lip": (0.50, 0.58),
            "jaw_left": (0.36, 0.52), "chin": (0.50, 0.66), "jaw_right": (0.64, 0.52),
            "neck_left": (0.42, 0.68), "neck_right": (0.58, 0.68),
            "shoulder_left": (0.25, 0.76), "shoulder_right": (0.75, 0.76),
            "chest_left": (0.34, 0.80), "chest_center": (0.50, 0.82), "chest_right": (0.66, 0.80),
            "torso_left": (0.28, 0.96), "torso_right": (0.72, 0.96),
        }

    @staticmethod
    def _live_avatar_identity_stiffness() -> Dict[str, float]:
        return {
            "head_top": 0.92, "temple_left": 0.86, "temple_right": 0.86,
            "nose_bridge": 0.94, "nose_tip": 0.88,
            "left_eye": 0.72, "right_eye": 0.72,
            "jaw_left": 0.70, "jaw_right": 0.70,
            "neck_left": 0.55, "neck_right": 0.55,
            "mouth_left": 0.25, "mouth_right": 0.25, "upper_lip": 0.18, "lower_lip": 0.15,
            "left_brow": 0.20, "right_brow": 0.20, "chin": 0.45,
            "shoulder_left": 0.25, "shoulder_right": 0.25,
            "chest_left": 0.10, "chest_center": 0.08, "chest_right": 0.10,
            "torso_left": 0.15, "torso_right": 0.15,
        }

    @staticmethod
    def _live_avatar_triangles() -> List[Tuple[str, str, str]]:
        return [
            ("head_top", "temple_left", "left_brow"), ("head_top", "left_brow", "right_brow"),
            ("head_top", "right_brow", "temple_right"), ("temple_left", "left_brow", "left_eye"),
            ("left_brow", "nose_bridge", "left_eye"), ("right_brow", "right_eye", "nose_bridge"),
            ("temple_right", "right_eye", "right_brow"), ("left_eye", "nose_bridge", "nose_tip"),
            ("nose_bridge", "right_eye", "nose_tip"), ("temple_left", "left_eye", "jaw_left"),
            ("left_eye", "nose_tip", "mouth_left"), ("right_eye", "mouth_right", "nose_tip"),
            ("temple_right", "jaw_right", "right_eye"), ("nose_tip", "upper_lip", "mouth_left"),
            ("nose_tip", "mouth_right", "upper_lip"), ("mouth_left", "upper_lip", "lower_lip"),
            ("upper_lip", "mouth_right", "lower_lip"), ("mouth_left", "lower_lip", "jaw_left"),
            ("mouth_right", "jaw_right", "lower_lip"), ("jaw_left", "lower_lip", "chin"),
            ("lower_lip", "jaw_right", "chin"), ("jaw_left", "chin", "neck_left"),
            ("chin", "neck_right", "neck_left"), ("chin", "jaw_right", "neck_right"),
            ("neck_left", "neck_right", "chest_center"), ("neck_left", "chest_center", "chest_left"),
            ("neck_right", "chest_right", "chest_center"), ("shoulder_left", "neck_left", "chest_left"),
            ("neck_right", "shoulder_right", "chest_right"), ("shoulder_left", "chest_left", "torso_left"),
            ("chest_left", "chest_center", "torso_left"), ("chest_center", "torso_right", "torso_left"),
            ("chest_center", "chest_right", "torso_right"), ("chest_right", "shoulder_right", "torso_right"),
        ]

    @staticmethod
    def _live_avatar_reference_candidates() -> List[str]:
        values: List[str] = []
        if SMG is not None:
            for attr in ("DEFAULT_AVATAR",):
                candidate = getattr(SMG, attr, None)
                if candidate:
                    values.append(os.path.abspath(os.fspath(candidate)))
            avatar_dir = getattr(SMG, "AVATAR_DIR", None)
            if avatar_dir:
                for name in ("avatar.png", "avatar.jpg", "SarahMemory.png", "SarahMemory.jpg", "default.png", "default.jpg"):
                    values.append(os.path.abspath(os.path.join(os.fspath(avatar_dir), name)))
        return values

    @staticmethod
    def _coerce_live_avatar_rgba(reference_rgba: Any, width: int, height: int) -> np.ndarray:
        """Convert caller/reference artwork into a bounded RGBA working surface."""
        if reference_rgba is None:
            raise ValueError("reference_rgba_required")
        arr = np.asarray(reference_rgba)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("reference_must_be_rgb_or_rgba")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2RGBA)
        if arr.shape[1] != width or arr.shape[0] != height:
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(arr)

    def _load_live_avatar_reference(self, width: int, height: int, reference_path: Optional[str] = None) -> Tuple[Optional[np.ndarray], str]:
        candidates = [os.path.abspath(reference_path)] if reference_path else []
        candidates.extend(self._live_avatar_reference_candidates())
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                if not os.path.isfile(path):
                    continue
                st = os.stat(path)
                key = (
                    os.path.abspath(path), int(width), int(height),
                    int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
                    int(st.st_size),
                )
                with self._live_avatar_lock:
                    cached = self._live_avatar_reference_cache.get(key)
                    if isinstance(cached, np.ndarray):
                        return cached, path

                bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if bgr is None:
                    continue
                if bgr.ndim == 2:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGRA)
                elif bgr.shape[2] == 3:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                else:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
                rgba = self._coerce_live_avatar_rgba(bgr, width, height)
                with self._live_avatar_lock:
                    # Remove stale versions of the same path/working size.
                    stale = [k for k in self._live_avatar_reference_cache if k[0] == key[0] and k[1:3] == key[1:3] and k != key]
                    for k in stale:
                        self._live_avatar_reference_cache.pop(k, None)
                    self._live_avatar_reference_cache[key] = rgba
                    while len(self._live_avatar_reference_cache) > self._live_avatar_reference_cache_max:
                        self._live_avatar_reference_cache.pop(next(iter(self._live_avatar_reference_cache)), None)
                return rgba, path
            except Exception:
                continue
        return None, ""

    @staticmethod
    def _warp_live_avatar_triangle(source: np.ndarray, destination: np.ndarray, source_tri: List[Tuple[float, float]], target_tri: List[Tuple[float, float]]) -> None:
        """Warp one RGBA triangle using OpenCV's affine raster primitive."""
        src = np.float32(source_tri)
        dst = np.float32(target_tri)
        src_rect = cv2.boundingRect(src)
        dst_rect = cv2.boundingRect(dst)
        sx, sy, sw, sh = src_rect
        dx, dy, dw, dh = dst_rect
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            return
        src_crop = source[sy:sy + sh, sx:sx + sw]
        if src_crop.size == 0:
            return
        src_local = np.float32([(p[0] - sx, p[1] - sy) for p in source_tri])
        dst_local = np.float32([(p[0] - dx, p[1] - dy) for p in target_tri])
        matrix = cv2.getAffineTransform(src_local, dst_local)
        warped = cv2.warpAffine(src_crop, matrix, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_local), 255, lineType=cv2.LINE_AA)
        y2 = min(destination.shape[0], dy + dh)
        x2 = min(destination.shape[1], dx + dw)
        if dx < 0 or dy < 0 or x2 <= dx or y2 <= dy:
            return
        warped = warped[:y2 - dy, :x2 - dx]
        mask = mask[:y2 - dy, :x2 - dx]
        roi = destination[dy:y2, dx:x2]
        inv = cv2.bitwise_not(mask)
        for channel in range(destination.shape[2]):
            roi[:, :, channel] = cv2.bitwise_or(cv2.bitwise_and(roi[:, :, channel], inv), cv2.bitwise_and(warped[:, :, channel], mask))

    @staticmethod
    def _resolve_live_avatar_logic():
        """Return SarahMemoryLogicCalc or a bounded local math fallback for diagnostics."""
        try:
            import SarahMemoryLogicCalc as _LC  # type: ignore
            return _LC, "SarahMemoryLogicCalc"
        except Exception:
            class _FallbackLogic:
                @staticmethod
                def sml_clamp(value, low, high):
                    return max(low, min(high, float(value)))

                @staticmethod
                def sml_avatar_deformation_offsets(params):
                    clamp = _FallbackLogic.sml_clamp
                    jaw = clamp(params.get("jaw_open", 0.0), 0.0, 1.0)
                    blink_l = clamp(params.get("blink_left", 0.0), 0.0, 1.0)
                    blink_r = clamp(params.get("blink_right", 0.0), 0.0, 1.0)
                    gaze_x = clamp(params.get("gaze_x", 0.0), -1.0, 1.0)
                    gaze_y = clamp(params.get("gaze_y", 0.0), -1.0, 1.0)
                    breath = clamp(params.get("breath", 0.0), -1.0, 1.0)
                    smile = clamp(params.get("smile", 0.0), -1.0, 1.0)
                    return {
                        "lower_lip": (0.0, 0.035 * jaw),
                        "chin": (0.0, 0.018 * jaw),
                        "mouth_left": (-0.012 * smile, -0.012 * smile),
                        "mouth_right": (0.012 * smile, -0.012 * smile),
                        "left_eye": (0.010 * gaze_x, 0.006 * gaze_y + 0.012 * blink_l),
                        "right_eye": (0.010 * gaze_x, 0.006 * gaze_y + 0.012 * blink_r),
                        "left_brow": (0.0, -0.010 * (1.0 - blink_l)),
                        "right_brow": (0.0, -0.010 * (1.0 - blink_r)),
                        "chest_center": (0.0, -0.015 * breath),
                        "chest_left": (-0.006 * breath, -0.010 * breath),
                        "chest_right": (0.006 * breath, -0.010 * breath),
                    }

                @staticmethod
                def sml_apply_normalized_offsets(points, offsets, stiffness):
                    out = dict(points)
                    for name, delta in offsets.items():
                        if name in out:
                            x, y = out[name]
                            dx, dy = delta
                            mobility = 1.0 - float(stiffness.get(name, 0.5))
                            out[name] = (_FallbackLogic.sml_clamp(x + dx * mobility, 0.0, 1.0), _FallbackLogic.sml_clamp(y + dy * mobility, 0.0, 1.0))
                    return out

                @staticmethod
                def sml_scale_normalized_points(points, width, height):
                    return {k: (float(v[0]) * (int(width) - 1), float(v[1]) * (int(height) - 1)) for k, v in points.items()}

                @staticmethod
                def sml_motion_vector(previous, current):
                    return (float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1]))

                @staticmethod
                def sml_vector_magnitude(x, y):
                    return float((float(x) ** 2 + float(y) ** 2) ** 0.5)

                @staticmethod
                def sml_temporal_history_weight(motion_magnitude, color_difference):
                    motion_penalty = min(1.0, float(motion_magnitude) / 24.0)
                    color_penalty = min(1.0, float(color_difference) * 2.0)
                    return max(0.0, min(0.72, 0.55 * (1.0 - max(motion_penalty, color_penalty))))

                @staticmethod
                def sml_frame_budget_level(frame_ms, target_fps):
                    budget = 1000.0 / max(1.0, float(target_fps))
                    if frame_ms <= budget:
                        return 0
                    if frame_ms <= budget * 1.5:
                        return 1
                    if frame_ms <= budget * 2.0:
                        return 2
                    if frame_ms <= budget * 3.0:
                        return 3
                    return 4
            return _FallbackLogic, "local_fallback"

    def _apply_live_avatar_neon(self, frame: np.ndarray, landmarks_px: Dict[str, Tuple[float, float]], parameters: Dict[str, Any], quality_level: int) -> np.ndarray:
        _LC, _logic_source = self._resolve_live_avatar_logic()
        try:
            intensity = _LC.sml_clamp(parameters.get("neon_intensity", 0.35), 0.0, 1.0)
            wave = _LC.sml_clamp(parameters.get("neon_wave", 0.0), -1.0, 1.0)
        except Exception:
            return frame
        if intensity <= 0.001:
            return frame
        points = []
        for key in ("shoulder_left", "chest_left", "chest_center", "chest_right", "shoulder_right"):
            if key in landmarks_px:
                points.append(tuple(int(v) for v in landmarks_px[key]))
        if len(points) < 2:
            return frame
        emission = np.zeros(frame.shape[:2], dtype=np.uint8)
        thickness = 2 if quality_level >= 4 else 3
        cv2.polylines(emission, [np.int32(points)], False, int(128 + (127 * intensity)), thickness=thickness, lineType=cv2.LINE_AA)
        highlight_index = 0 if wave < -0.33 else (len(points) // 2 if wave < 0.33 else len(points) - 1)
        cv2.circle(emission, points[highlight_index], 5 if quality_level < 3 else 3, 255, -1, lineType=cv2.LINE_AA)
        blur_radius = 11 if quality_level == 0 else (7 if quality_level <= 2 else 3)
        glow = cv2.GaussianBlur(emission, (blur_radius | 1, blur_radius | 1), 0)
        overlay = np.zeros_like(frame)
        overlay[:, :, 0] = np.maximum(emission, glow)
        overlay[:, :, 1] = np.maximum(emission, glow)
        overlay[:, :, 2] = np.maximum(emission, glow)
        overlay[:, :, 3] = np.maximum(emission, glow)
        return cv2.addWeighted(frame, 1.0, overlay, float(intensity) * 0.32, 0.0)

    def render_live_avatar_frame(
        self,
        parameter_packet: Optional[Dict[str, Any]] = None,
        *,
        width: int = 512,
        height: int = 512,
        reference_path: Optional[str] = None,
        reference_rgba: Any = None,
        use_temporal_history: bool = True,
    ) -> Dict[str, Any]:
        """Render one persistent live-avatar RGBA frame with isolated stream history."""
        started = time.perf_counter()
        width, height = validate_canvas_dimensions(width, height)
        packet = dict(parameter_packet or {})
        params = packet.get("parameters") if isinstance(packet.get("parameters"), dict) else packet
        _LC, logic_source = self._resolve_live_avatar_logic()

        if reference_rgba is not None:
            try:
                source = self._coerce_live_avatar_rgba(reference_rgba, width, height)
                source_id = "caller_rgba"
            except Exception as exc:
                return {"ok": False, "error": f"invalid_reference_rgba:{exc}", "execution_authority": False, "pixel_authority": True}
        else:
            source, source_id = self._load_live_avatar_reference(width, height, reference_path=reference_path)
            if source is None:
                return {
                    "ok": False,
                    "error": "avatar_reference_artwork_not_found",
                    "reference_candidates": self._live_avatar_reference_candidates(),
                    "execution_authority": False,
                    "pixel_authority": True,
                    "fallback_required": True,
                    "logic_source": logic_source,
                }

        normalized = self._live_avatar_landmark_atlas()
        offsets = _LC.sml_avatar_deformation_offsets(params)
        target_norm = _LC.sml_apply_normalized_offsets(normalized, offsets, self._live_avatar_identity_stiffness())
        source_px = _LC.sml_scale_normalized_points(normalized, width, height)
        target_px = _LC.sml_scale_normalized_points(target_norm, width, height)

        current = source.copy()
        for names in self._live_avatar_triangles():
            if not all(name in source_px and name in target_px for name in names):
                continue
            self._warp_live_avatar_triangle(source, current, [source_px[n] for n in names], [target_px[n] for n in names])

        stream_key = (str(source_id), int(width), int(height))
        with self._live_avatar_lock:
            stream = self._live_avatar_streams.setdefault(stream_key, {"history": None, "landmarks": {}, "parameters": {}, "frame_id": 0, "last_health": {}})
            quality_level = int((stream.get("last_health") or {}).get("quality_level") or 0)

        current = self._apply_live_avatar_neon(current, target_px, params, quality_level)

        with self._live_avatar_lock:
            stream = self._live_avatar_streams.setdefault(stream_key, {"history": None, "landmarks": {}, "parameters": {}, "frame_id": 0, "last_health": {}})
            history_weight = 0.0
            motion_magnitude = 0.0
            color_difference = 0.0
            previous_history = stream.get("history")
            previous_landmarks = stream.get("landmarks") or {}
            previous_parameters = stream.get("parameters") or {}
            if use_temporal_history and isinstance(previous_history, np.ndarray) and previous_history.shape == current.shape:
                vectors = []
                for name, point in target_px.items():
                    prev = previous_landmarks.get(name)
                    if prev is not None:
                        vectors.append(_LC.sml_motion_vector(prev, point))
                if vectors:
                    magnitudes = [_LC.sml_vector_magnitude(v[0], v[1]) for v in vectors]
                    motion_magnitude = sum(magnitudes) / len(magnitudes)
                diff = cv2.absdiff(current, previous_history)
                color_difference = _LC.sml_clamp(float(np.mean(diff)) / 255.0, 0.0, 1.0)
                history_weight = _LC.sml_temporal_history_weight(motion_magnitude, color_difference)
                if history_weight > 0.0:
                    current = cv2.addWeighted(current, 1.0 - history_weight, previous_history, history_weight, 0.0)

            changed_parameters = [k for k, v in params.items() if previous_parameters.get(k) != v]
            stream["history"] = current.copy()
            stream["landmarks"] = dict(target_px)
            stream["parameters"] = dict(params)
            stream["frame_id"] = int(stream.get("frame_id") or 0) + 1
            frame_id = int(stream["frame_id"])
            self._live_avatar_frame_id += 1

        frame_ms = (time.perf_counter() - started) * 1000.0
        quality_level = _LC.sml_frame_budget_level(frame_ms, 30.0)
        health = {
            "schema": "SarahMemory.avatar.render_health.v1",
            "frame_id": frame_id,
            "global_frame_id": self._live_avatar_frame_id,
            "frame_ms": frame_ms,
            "target_fps": 30.0,
            "quality_level": quality_level,
            "history_weight": history_weight,
            "motion_magnitude": motion_magnitude,
            "color_difference": color_difference,
            "changed_parameters": changed_parameters[:64],
            "dirty_region_tracking": False,
            "partial_raster_update": False,
            "reference": source_id,
            "stream_key": stream_key,
            "logic_source": logic_source,
            "execution_authority": False,
        }
        with self._live_avatar_lock:
            self._live_avatar_streams[stream_key]["last_health"] = dict(health)
            self._live_avatar_last_health = dict(health)
        return {
            "ok": True,
            "schema": "SarahMemory.avatar.live_frame.v1",
            "frame_id": frame_id,
            "timestamp_monotonic": time.monotonic(),
            "frame_rgba": current,
            "landmarks": target_px,
            "render_health": health,
            "execution_authority": False,
            "pixel_authority": True,
            "owner": "SarahMemoryCanvasStudio",
        }

    def live_avatar_renderer_self_test(self) -> Dict[str, Any]:
        """In-memory renderer smoke test; touches no files, network, or devices."""
        synthetic = np.zeros((256, 256, 4), dtype=np.uint8)
        synthetic[:, :, 3] = 255
        cv2.circle(synthetic, (128, 92), 58, (85, 110, 145, 255), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(synthetic, (70, 145), (186, 255), (45, 65, 90, 255), -1)
        packet = {"parameters": {"breath": 0.5, "jaw_open": 0.25, "blink_left": 0.1, "blink_right": 0.1, "gaze_x": 0.1, "neon_intensity": 0.4, "neon_wave": 0.2}}
        out = self.render_live_avatar_frame(packet, width=256, height=256, reference_rgba=synthetic, use_temporal_history=True)
        second = self.render_live_avatar_frame(packet, width=256, height=256, reference_rgba=synthetic, use_temporal_history=True)
        frame = out.get("frame_rgba")
        checks = [
            {"name": "render_ok", "passed": bool(out.get("ok"))},
            {"name": "rgba_shape", "passed": isinstance(frame, np.ndarray) and tuple(frame.shape) == (256, 256, 4)},
            {"name": "persistent_frame_id", "passed": int(second.get("frame_id") or 0) > int(out.get("frame_id") or 0)},
            {"name": "temporal_metadata", "passed": "history_weight" in (second.get("render_health") or {})},
            {"name": "no_execution_authority", "passed": out.get("execution_authority") is False},
        ]
        return {"ok": all(c["passed"] for c in checks), "checks": checks, "render_health": second.get("render_health"), "execution_authority": False}



    def create_canvas(self, width: int, height: int, name: str = None, 
                     depth: int = 8, background_color: Tuple[int, int, int, int] = None) -> Canvas:
        """
        Create a new canvas
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            name: Canvas name (auto-generated if None)
            depth: Color depth (8, 16, or 32 bits)
            background_color: Initial background color
        
        Returns:
            The newly created Canvas object
        """
        if name is None:
            name = f"Canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        canvas = Canvas(name, width, height, depth, background_color)
        self.canvases[canvas.id] = canvas
        self.active_canvas_id = canvas.id
        
        logging.info(f"[CanvasStudio] Created canvas '{name}' ({width}x{height})")
        return canvas
    
    def get_canvas(self, canvas_id: str = None) -> Optional[Canvas]:
        """Get a canvas by ID (or active canvas if ID is None)"""
        if canvas_id is None:
            canvas_id = self.active_canvas_id
        return self.canvases.get(canvas_id)
    
    def save_canvas(self, canvas: Canvas, filepath: str = None) -> bool:
        """Save canvas project file (.scp format) with lossless layer payloads."""
        try:
            if filepath is None:
                filename = f"{canvas.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.scp"
                filepath = os.path.join(CANVAS_PROJECTS_DIR, filename)
            if not filepath.lower().endswith(".scp"):
                filepath = f"{filepath}.scp"
            os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
            project_data = {
                "canvas": canvas.to_dict(),
                "studio_version": CANVAS_STUDIO_VERSION,
                "saved_at": datetime.now().isoformat(),
                "layer_storage": "npy_lossless_rgba",
            }
            layer_dir = _project_layer_dir(filepath)
            os.makedirs(layer_dir, exist_ok=True)
            for i, layer in enumerate(canvas.layers):
                np.save(os.path.join(layer_dir, f"layer_{i:03d}.npy"), layer.data)
                preview = _rgba_to_uint8(layer.data, layer.depth)
                preview_bgra = cv2.cvtColor(preview, cv2.COLOR_RGBA2BGRA)
                if not _safe_imwrite(os.path.join(layer_dir, f"layer_{i:03d}.png"), preview_bgra):
                    logging.warning(f"[CanvasStudio] Preview PNG write failed for layer {i}; lossless NPY was written")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2)
            ok = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
            logging.info(f"[CanvasStudio] Saved canvas '{canvas.name}' to {filepath}")
            return bool(ok)
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to save canvas: {e}")
            traceback.print_exc()
            return False
    
    def load_canvas(self, filepath: str) -> Optional[Canvas]:
        """Load canvas project file (.scp format), restoring IDs, transforms, and layer state."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            canvas_data = project_data['canvas']
            canvas = Canvas(
                name=canvas_data.get('name', 'Canvas'),
                width=int(canvas_data.get('width', DEFAULT_CANVAS_WIDTH)),
                height=int(canvas_data.get('height', DEFAULT_CANVAS_HEIGHT)),
                depth=int(canvas_data.get('depth', COLOR_DEPTH_8BIT)),
                background_color=(0, 0, 0, 0),
            )
            canvas.id = canvas_data.get('id', canvas.id)
            canvas.created_at = _parse_dt(canvas_data.get('created_at'), canvas.created_at)
            canvas.modified_at = _parse_dt(canvas_data.get('modified_at'), canvas.modified_at)
            canvas.author = canvas_data.get('author', canvas.author)
            canvas.layers.clear()
            layer_dir = _project_layer_dir(filepath)
            legacy_layer_dir = filepath.replace('.scp', '_layers')
            if not os.path.isdir(layer_dir) and os.path.isdir(legacy_layer_dir):
                layer_dir = legacy_layer_dir
            for index, layer_data in enumerate(canvas_data.get('layers', [])):
                layer = CanvasLayer(
                    layer_data.get('name', f'Layer {index}'),
                    int(layer_data.get('width', canvas.width)),
                    int(layer_data.get('height', canvas.height)),
                    int(layer_data.get('depth', canvas.depth)),
                )
                layer.id = layer_data.get('id', layer.id)
                layer.opacity = int(layer_data.get('opacity', 100))
                try:
                    layer.blend_mode = BlendMode(layer_data.get('blend_mode', BlendMode.NORMAL.value))
                except Exception:
                    layer.blend_mode = BlendMode.NORMAL
                layer.visible = bool(layer_data.get('visible', True))
                layer.locked = bool(layer_data.get('locked', False))
                layer.position = tuple(layer_data.get('position', (0, 0)))
                layer.rotation = float(layer_data.get('rotation', 0))
                layer.scale = tuple(layer_data.get('scale', (1.0, 1.0)))
                layer.created_at = _parse_dt(layer_data.get('created_at'), layer.created_at)
                layer.modified_at = _parse_dt(layer_data.get('modified_at'), layer.modified_at)
                npy_file = os.path.join(layer_dir, f"layer_{index:03d}.npy")
                png_file = os.path.join(layer_dir, f"layer_{index:03d}.png")
                if os.path.exists(npy_file):
                    layer.data = np.load(npy_file, allow_pickle=False)
                elif os.path.exists(png_file):
                    raw = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        if raw.ndim == 2:
                            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGBA)
                        elif raw.shape[2] == 3:
                            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)
                        else:
                            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
                        layer.data = _float01_to_rgba_depth(raw.astype(np.float32) / 255.0, layer.depth)
                canvas.layers.append(layer)
            if not canvas.layers:
                canvas.layers.append(CanvasLayer("Background", canvas.width, canvas.height, canvas.depth))
            canvas.active_layer_index = max(0, min(int(canvas_data.get('active_layer_index', 0)), len(canvas.layers) - 1))
            self.canvases[canvas.id] = canvas
            self.active_canvas_id = canvas.id
            logging.info(f"[CanvasStudio] Loaded canvas '{canvas.name}' from {filepath}")
            return canvas
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to load canvas: {e}")
            traceback.print_exc()
            return None
    
    def _export_uint8_rgba(self, canvas: Canvas, flatten: bool = True) -> np.ndarray:
        if flatten:
            image_data = canvas.flatten()
            depth = canvas.depth
        else:
            if not (0 <= canvas.active_layer_index < len(canvas.layers)):
                raise ValueError("active_layer_index_out_of_range")
            layer = canvas.layers[canvas.active_layer_index]
            image_data = layer.data
            depth = layer.depth
        return _rgba_to_uint8(image_data, depth)

    def export_canvas(self, canvas: Canvas, filepath: str, 
                     format: str = "PNG", quality: int = 90, flatten: bool = True) -> bool:
        """Export canvas to an image/PDF/SVG file and report actual write success."""
        try:
            fmt = _normalize_export_format(format)
            if fmt not in [_normalize_export_format(x) for x in SUPPORTED_EXPORT_FORMATS]:
                logging.error(f"[CanvasStudio] Unsupported format: {fmt}")
                return False
            extension = "jpg" if fmt == "JPEG" else fmt.lower()
            if not filepath.lower().endswith(f".{extension}"):
                filepath = f"{filepath}.{extension}"
            rgba8 = self._export_uint8_rgba(canvas, flatten=flatten)
            quality = int(max(0, min(100, quality)))
            if fmt in ("PNG", "BMP", "TGA", "TIFF", "WEBP"):
                if fmt in ("PNG", "WEBP", "TIFF"):
                    out = cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGRA)
                else:
                    out = cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGR)
                params: List[int] = []
                if fmt == "WEBP":
                    params = [cv2.IMWRITE_WEBP_QUALITY, quality]
                elif fmt == "PNG":
                    params = [cv2.IMWRITE_PNG_COMPRESSION, DEFAULT_PNG_COMPRESSION]
                elif fmt == "TIFF":
                    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
                ok = _safe_imwrite(filepath, out, params)
            elif fmt == "JPEG":
                bgr = cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGR)
                ok = _safe_imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif fmt == "PDF":
                if not PIL_AVAILABLE or Image is None:
                    logging.error("[CanvasStudio] PDF export requires Pillow")
                    return False
                rgb = Image.fromarray(rgba8[:, :, :3], mode="RGB")
                os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
                rgb.save(filepath, "PDF", resolution=100.0)
                ok = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
            elif fmt == "SVG":
                import base64 as _b64
                ok_png, encoded = cv2.imencode('.png', cv2.cvtColor(rgba8, cv2.COLOR_RGBA2BGRA))
                if not ok_png:
                    return False
                payload = _b64.b64encode(encoded.tobytes()).decode('ascii')
                svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" viewBox="0 0 {canvas.width} {canvas.height}"><image width="{canvas.width}" height="{canvas.height}" href="data:image/png;base64,{payload}"/></svg>'
                os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(svg)
                ok = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
            else:
                ok = False
            if ok:
                logging.info(f"[CanvasStudio] Exported canvas '{canvas.name}' to {filepath}")
            else:
                logging.error(f"[CanvasStudio] Export failed for canvas '{canvas.name}' to {filepath}")
            return bool(ok)
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to export canvas: {e}")
            traceback.print_exc()
            return False

    def generate_from_prompt(self, prompt: str, width: int = None, height: int = None,
                             style: str = "default", quality: str = "standard") -> Optional[Canvas]:
        """Generate artwork from a prompt through governed SarahMemory routing, then offline fallback."""
        try:
            width = int(width or 1024)
            height = int(height or 1024)
            canvas_name = f"AI_Generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            canvas = self.create_canvas(width, height, canvas_name)
            prompt_text = (prompt or "").strip()
            if not prompt_text:
                return canvas
            img_bytes = None
            mime = None
            try:
                img_bytes, mime = self._try_generate_via_sarahmemory_api(prompt_text, width, height, style=style, quality=quality)
            except Exception as e:
                logging.info(f"[CanvasStudio] SarahMemoryAPI image route unavailable: {e}")
            if img_bytes is None:
                img_bytes, mime = self._generate_offline_fallback(prompt_text, width, height, style=style)
            self._apply_image_bytes_to_canvas(canvas, img_bytes, mime=mime)
            logging.info(f"[CanvasStudio] Generated artwork from prompt: '{prompt_text[:80]}...'")
            return canvas
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to generate from prompt: {e}")
            traceback.print_exc()
            return None
    
    def _try_generate_via_sarahmemory_api(self, prompt: str, width: int, height: int, *, style: str = "default", quality: str = "standard"):
        """Ask SarahMemoryAPI to generate an image. This is the preferred provider-agnostic hook."""
        try:
            import SarahMemoryAPI as _API  # type: ignore
        except Exception:
            return (None, None)
        fn = getattr(_API, "generate_image", None) or getattr(_API, "image_generate", None) or getattr(_API, "generate_media_image", None)
        if not callable(fn):
            return (None, None)
        try:
            res = fn(prompt=prompt, width=width, height=height, style=style, quality=quality)
        except TypeError:
            try:
                res = fn(prompt, width, height)
            except Exception:
                res = fn(prompt)
        if isinstance(res, (bytes, bytearray)):
            return (bytes(res), "image/png")
        if isinstance(res, dict):
            b = res.get("bytes") or res.get("image_bytes")
            mime = res.get("mime") or res.get("content_type") or "image/png"
            if isinstance(b, (bytes, bytearray)):
                return (bytes(b), mime)
            b64 = res.get("b64") or res.get("image_base64")
            if b64:
                return (base64.b64decode(b64), mime)
        return (None, None)
    
    def _try_generate_via_openai(self, prompt: str, width: int, height: int, *, style: str = "default", quality: str = "standard"):
        """Disabled direct-vendor path. Use SarahMemoryAPI/provider adapters instead."""
        logging.info("[CanvasStudio] Direct OpenAI image generation is disabled; use SarahMemoryAPI/provider routing")
        return (None, None)
    
    def _generate_offline_fallback(self, prompt: str, width: int, height: int, *, style: str = "default"):
        """Guaranteed offline generator: procedural background + prompt overlay."""
        import io, hashlib, random
        w, h = int(width), int(height)
        seed = int(hashlib.sha256((prompt + "|" + str(style)).encode("utf-8")).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        if PIL_AVAILABLE and Image is not None and ImageDraw is not None:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
            d = ImageDraw.Draw(img)
            for y in range(h):
                v = int(25 + 80 * (y / max(1, h - 1)))
                d.line([(0, y), (w, y)], fill=(v, v, min(255, v + 25), 255))
            for _ in range(140):
                x = rnd.randint(0, max(0, w - 1))
                y = rnd.randint(0, max(0, h - 1))
                r = rnd.randint(8, max(10, min(w, h)//9))
                col = (rnd.randint(60, 220), rnd.randint(60, 220), rnd.randint(60, 220), rnd.randint(70, 150))
                d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=2)
            try:
                font = ImageFont.truetype("arial.ttf", 28) if ImageFont else None
            except Exception:
                font = ImageFont.load_default() if ImageFont else None
            pad = 24
            text = prompt if len(prompt) <= 180 else (prompt[:177] + "...")
            d.rectangle((pad-12, max(0, h-170), w-pad+12, h-pad+12), fill=(0, 0, 0, 160))
            if font:
                d.text((pad, max(0, h-155)), "OFFLINE GENERATION", fill=(255, 255, 255, 230), font=font)
                d.text((pad, max(0, h-118)), text, fill=(230, 230, 230, 230), font=font)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return (bio.getvalue(), "image/png")
        # Pillow-free fallback: numpy/cv2 procedural image.
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        for y in range(h):
            v = int(25 + 80 * (y / max(1, h - 1)))
            arr[y, :, :] = (v, v, min(255, v + 25), 255)
        for _ in range(80):
            center = (rnd.randint(0, max(0, w - 1)), rnd.randint(0, max(0, h - 1)))
            radius = rnd.randint(6, max(8, min(w, h)//10))
            color = (rnd.randint(60, 220), rnd.randint(60, 220), rnd.randint(60, 220), rnd.randint(90, 180))
            cv2.circle(arr, center, radius, color, 1, lineType=cv2.LINE_AA)
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
        return (encoded.tobytes() if ok else b'', "image/png")
    
    def _apply_image_bytes_to_canvas(self, canvas: 'Canvas', img_bytes: bytes, *, mime: str | None = None):
        """Decode image bytes and push them into the active layer data as internal RGBA."""
        if not img_bytes:
            return
        if PIL_AVAILABLE and Image is not None:
            import io
            im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            if im.size != (int(canvas.width), int(canvas.height)):
                im = im.resize((int(canvas.width), int(canvas.height)))
            rgba = np.array(im)
        else:
            data = np.frombuffer(img_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                return
            if decoded.ndim == 2:
                rgba = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGBA)
            elif decoded.shape[2] == 3:
                rgba = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGBA)
            else:
                rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
            if rgba.shape[:2] != (canvas.height, canvas.width):
                rgba = cv2.resize(rgba, (canvas.width, canvas.height), interpolation=cv2.INTER_LINEAR)
        layer = canvas.get_active_layer()
        if layer is not None:
            layer.data = _float01_to_rgba_depth(rgba.astype(np.float32) / 255.0, layer.depth)
            layer.modified_at = datetime.now()
            canvas.modified_at = datetime.now()

    def batch_process(self, canvas_ids: List[str], operation: str, **kwargs) -> List[bool]:
        """Apply an operation to multiple canvases without false-success reporting."""
        results: List[bool] = []
        operation = str(operation or "").lower()
        for canvas_id in canvas_ids:
            canvas = self.get_canvas(canvas_id)
            if not canvas:
                results.append(False)
                continue
            try:
                if operation == "resize":
                    width = kwargs.get("width", kwargs.get("new_width", canvas.width))
                    height = kwargs.get("height", kwargs.get("new_height", canvas.height))
                    results.append(bool(canvas.resize(int(width), int(height))))
                elif operation == "color_correct":
                    results.append(bool(canvas.color_correct(**kwargs)))
                elif operation == "apply_effect":
                    effect = kwargs.pop("effect_type", kwargs.pop("effect", None))
                    results.append(bool(canvas.apply_effect(effect, **kwargs)))
                elif operation == "export":
                    results.append(bool(self.export_canvas(canvas, **kwargs)))
                else:
                    logging.warning(f"[CanvasStudio] Unknown batch operation: {operation}")
                    results.append(False)
            except Exception as e:
                logging.error(f"[CanvasStudio] Batch operation failed for canvas {canvas_id}: {e}")
                results.append(False)
        successful = sum(1 for item in results if item)
        logging.info(f"[CanvasStudio] Batch operation '{operation}': {successful}/{len(canvas_ids)} successful")
        return results

    def get_studio_info(self) -> Dict:
        """Get Canvas Studio system information."""
        return {
            "version": CANVAS_STUDIO_VERSION,
            "build": CANVAS_STUDIO_BUILD,
            "engine": "SarahMemoryCanvasStudio",
            "local_first": True,
            "import_side_effect_free": True,
            "execution_authority": False,
            "active_canvases": len(self.canvases),
            "pil_available": PIL_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "supported_formats": list(SUPPORTED_EXPORT_FORMATS),
            "implemented_blend_modes": [mode.value for mode in BlendMode],
            "implemented_filters": [item.value for item in FilterType],
            "implemented_gradients": [item.value for item in GradientType],
            "max_canvas_size": (MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT),
            "directories": {
                "projects": CANVAS_PROJECTS_DIR,
                "exports": CANVAS_EXPORTS_DIR,
                "cache": CANVAS_CACHE_DIR,
                "templates": CANVAS_TEMPLATES_DIR,
                "brushes": CANVAS_BRUSHES_DIR
            }
        }


    def build_output_manifest(self, canvas: Canvas, filepath: str = "") -> Dict[str, Any]:
        """Return a bounded, serializable description of a canvas/output contract."""
        layer_count = len(canvas.layers) if canvas is not None else 0
        return {
            "schema": "SARAHMEMORY_CANVAS_OUTPUT_V1",
            "studio_version": CANVAS_STUDIO_VERSION,
            "canvas_id": getattr(canvas, "id", ""),
            "name": getattr(canvas, "name", ""),
            "width": int(getattr(canvas, "width", 0) or 0),
            "height": int(getattr(canvas, "height", 0) or 0),
            "depth": int(getattr(canvas, "depth", 0) or 0),
            "layer_count": layer_count,
            "output_path": os.path.abspath(filepath) if filepath else "",
            "created_at": datetime.now().isoformat(),
            "local_first": True,
            "execution_authority": False,
        }

    def enterprise_self_test(self) -> Dict[str, Any]:
        """Run bounded in-memory Canvas Studio checks without exporting files."""
        checks: List[Dict[str, Any]] = []
        try:
            dims = validate_canvas_dimensions(320, 180)
            checks.append({"name": "dimension_validation", "passed": dims == (320, 180), "observed": dims})
            canvas = Canvas("EnterpriseSelfTest", 64, 64, 8, (0, 0, 0, 0))
            top = canvas.add_layer("Top")
            top.fill_color((255, 0, 0, 255))
            flattened = canvas.flatten()
            checks.append({"name": "alpha_composite", "passed": int(flattened[0, 0, 3]) == 255 and int(flattened[0, 0, 0]) == 255, "observed": flattened[0, 0].tolist()})
            top.apply_opacity(50)
            before_alpha = int(top.data[0, 0, 3])
            ok_cc = canvas.color_correct(brightness=15)
            after_alpha = int(canvas.get_active_layer().data[0, 0, 3])
            checks.append({"name": "color_preserves_alpha", "passed": ok_cc and before_alpha == after_alpha, "observed": [before_alpha, after_alpha]})
            checks.append({"name": "resize_operation", "passed": canvas.resize(32, 48) and canvas.width == 32 and canvas.height == 48, "observed": [canvas.width, canvas.height]})
            checks.append({"name": "unsupported_effect_rejected", "passed": canvas.apply_effect("not_a_filter") is False})
            manifest = self.build_output_manifest(canvas)
            checks.append({"name": "manifest_contract", "passed": manifest.get("schema") == "SARAHMEMORY_CANVAS_OUTPUT_V1", "observed": manifest})
            avatar = self.live_avatar_renderer_self_test()
            checks.append({"name": "live_avatar_self_test", "passed": bool(avatar.get("ok")), "observed": avatar.get("render_health")})
        except Exception as exc:
            checks.append({"name": "unexpected_exception", "passed": False, "observed": str(exc)})
        passed = sum(1 for check in checks if check.get("passed"))
        return {
            "ok": passed == len(checks),
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "file_write_performed": False,
            "network_used": False,
            "hardware_control": False,
        }



def get_canvas_studio_capabilities() -> Dict[str, Any]:
    """Read-only module capability report; does not initialize a project or write files."""
    return {
        "ok": True,
        "module": "SarahMemoryCanvasStudio",
        "version": CANVAS_STUDIO_VERSION,
        "build": CANVAS_STUDIO_BUILD,
        "pil_available": bool(PIL_AVAILABLE),
        "opencv_available": cv2 is not None,
        "scipy_available": bool(SCIPY_AVAILABLE),
        "supported_import_formats": list(SUPPORTED_IMPORT_FORMATS),
        "supported_export_formats": list(SUPPORTED_EXPORT_FORMATS),
        "implemented_filters": [item.value for item in FilterType],
        "implemented_blend_modes": [item.value for item in BlendMode],
        "implemented_gradients": [item.value for item in GradientType],
        "max_canvas_size": [MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT],
        "local_first": True,
        "network_optional": False,
        "direct_vendor_network_disabled": True,
        "execution_authority": False,
        "persistent_live_avatar_renderer": True,
        "live_avatar_schema": "SarahMemory.avatar.live_frame.v1",
        "persistent_frame_history": True,
        "temporal_reconstruction": True,
        "reference_atlas_required_for_identity_render": True,
        "import_side_effect_free": True,
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for standalone execution"""
    print("=" * 80)
    print("SarahMemory Canvas Studio - World-Class Art & Graphics Engine")
    print(f"Version {CANVAS_STUDIO_VERSION} (Build {CANVAS_STUDIO_BUILD})")
    print("=" * 80)
    print()
    
    # Initialize studio
    studio = CanvasStudio()
    
    # Display system info
    info = studio.get_studio_info()
    print(f"Active Canvases: {info['active_canvases']}")
    print(f"PIL Available: {info['pil_available']}")
    print(f"SciPy Available: {info['scipy_available']}")
    print(f"Supported Formats: {', '.join(info['supported_formats'])}")
    print(f"Max Canvas Size: {info['max_canvas_size'][0]}x{info['max_canvas_size'][1]} pixels")
    print()
    
    # Create demo canvas
    print("Creating demo canvas...")
    canvas = studio.create_canvas(1920, 1080, "Demo_Canvas")
    
    # Add layers and effects
    print("Adding layers...")
    layer1 = canvas.add_layer("Gradient Layer")
    layer1.apply_gradient("radial", [(255, 0, 0), (0, 0, 255), (0, 255, 0)])
    
    layer2 = canvas.add_layer("Effect Layer")
    layer2.fill_color((255, 255, 255, 128))
    
    # Apply effects
    print("Applying effects...")
    canvas.set_active_layer(1)
    canvas.apply_effect("gaussian_blur", radius=10)
    canvas.color_correct(brightness=20, contrast=10, saturation=15)
    
    # Export canvas
    export_path = os.path.join(CANVAS_EXPORTS_DIR, "demo_output.png")
    print(f"Exporting to {export_path}...")
    
    if studio.export_canvas(canvas, export_path, format="PNG", quality=95):
        print(f"✓ Successfully exported to: {export_path}")
    else:
        print("✗ Export failed")
    
    # Save project
    project_path = os.path.join(CANVAS_PROJECTS_DIR, "demo_project.scp")
    print(f"Saving project to {project_path}...")
    
    if studio.save_canvas(canvas, project_path):
        print(f"✓ Successfully saved project: {project_path}")
    else:
        print("✗ Save failed")
    
    print()
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

# ====================================================================
# END OF SarahMemoryCanvasStudio.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCanvasStudio',
    "version": "v9.0.0-alpha-sml-0.3",
    "category": 'CreativeRendering',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['graphics_rendering', 'image_editing', 'avatar_frame_rendering'],
    "supported_missions": ['Conversation', 'CreativeRendering', 'AvatarPresentation'],
    "supported_omega": ['Ω001', 'Ω070', 'Ω100'],
    "required_authority": ['Read', 'WriteCanvas'],
    "execution_authority": False,
    "priority": 50,
    "trust_level": "source_integrated",
    "internal_only": False,
    "metadata": {"sml_adapter": "creative_rendering_non_executing", "source_file": 'SarahMemoryCanvasStudio.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    capabilities = get_canvas_studio_capabilities()
    dependency_score = 1.0
    notes = ["SML adapter present", "execution_authority=False"]
    if not capabilities.get("pil_available"):
        dependency_score -= 0.15
        notes.append("Pillow unavailable: PDF/text overlay paths limited")
    if not capabilities.get("opencv_available"):
        dependency_score -= 0.45
        notes.append("OpenCV unavailable: raster engine unavailable")
    dependency_score = max(0.0, dependency_score)
    return {
        "status": "Healthy" if dependency_score >= 0.75 else "Degraded",
        "availability": dependency_score,
        "integrity": 1.0,
        "performance": 0.85,
        "reliability": 0.85,
        "confidence": 0.86,
        "latency_ms": 0.0,
        "stability": 0.90,
        "compatibility": 0.90,
        "execution_authority": False,
        "notes": notes,
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCanvasStudio',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
        "capabilities": get_canvas_studio_capabilities(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCanvasStudio', action=action, note=note or "creative rendering organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

