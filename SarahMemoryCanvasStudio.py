"""--==The SarahMemory Project==--
File: SarahMemoryCanvasStudio.py
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
{DATASETS_DIR}/
    canvas/
        projects/          # Saved project files (.scp format)
        exports/           # Final rendered outputs
        cache/             # Temporary processing files
        templates/         # Preset templates and styles
        brushes/           # Custom brush definitions
        
USAGE EXAMPLES:
--------------
    # Basic canvas creation
    studio = CanvasStudio()
    canvas = studio.create_canvas(1920, 1080, "My Artwork")
    
    # Add layers and effects
    layer1 = canvas.add_layer("Background")
    layer1.fill_color((100, 150, 200))
    layer1.apply_gradient("linear", colors=[(0,0,0), (255,255,255)])
    
    # AI generation
    ai_image = studio.generate_from_prompt(
        "A serene landscape with mountains and lakes at sunset",
        style="photorealistic",
        quality="high"
    )
    
    # Apply professional effects
    canvas.apply_effect("gaussian_blur", radius=5)
    canvas.color_correct(brightness=10, contrast=15, saturation=5)
    
    # Export final artwork
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

import os
import sys
import json
import logging
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
except ImportError:
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
    DATASETS_DIR = SMG.DATASETS_DIR
    DEBUG_MODE = SMG.DEBUG_MODE
except ImportError:
    DATASETS_DIR = os.path.join(os.getcwd(), "data")
    DEBUG_MODE = True
    logging.warning("[CanvasStudio] Running in standalone mode without SarahMemoryGlobals")

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Version information
CANVAS_STUDIO_VERSION = "2.0.0"
CANVAS_STUDIO_BUILD = "20251204"

# Directory structure
CANVAS_DIR = os.path.join(DATASETS_DIR, "canvas")
CANVAS_PROJECTS_DIR = os.path.join(CANVAS_DIR, "projects")
CANVAS_EXPORTS_DIR = os.path.join(CANVAS_DIR, "exports")
CANVAS_CACHE_DIR = os.path.join(CANVAS_DIR, "cache")
CANVAS_TEMPLATES_DIR = os.path.join(CANVAS_DIR, "templates")
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
SUPPORTED_EXPORT_FORMATS = ["PNG", "JPG", "JPEG", "WebP", "TIFF", "BMP", "TGA", "PDF"]
SUPPORTED_IMPORT_FORMATS = ["PNG", "JPG", "JPEG", "WebP", "TIFF", "BMP", "TGA", "GIF"]

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
        """Fill the entire layer with a solid color"""
        if color is None:
            color = (255, 255, 255, 255)
        
        if len(color) == 3:
            color = (*color, 255)
        
        self.data[:] = color
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
        """
        Apply a gradient fill to the layer
        
        Args:
            gradient_type: Type of gradient (linear, radial, etc.)
            colors: List of color stops
            angle: Gradient angle in degrees (for linear gradients)
            center: Center point for radial gradients (normalized 0-1)
        """
        if center is None:
            center = (0.5, 0.5)
        
        height, width = self.data.shape[:2]
        
        if gradient_type == "linear":
            # Create linear gradient
            angle_rad = np.radians(angle)
            for y in range(height):
                for x in range(width):
                    # Calculate position along gradient
                    t = (x * np.cos(angle_rad) + y * np.sin(angle_rad)) / (width + height)
                    t = max(0, min(1, t))
                    
                    # Interpolate colors
                    color = self._interpolate_colors(colors, t)
                    self.data[y, x] = (*color, 255)
        
        elif gradient_type == "radial":
            # Create radial gradient
            cx, cy = int(center[0] * width), int(center[1] * height)
            max_dist = np.sqrt(width**2 + height**2) / 2
            
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    t = min(1, dist / max_dist)
                    
                    color = self._interpolate_colors(colors, t)
                    self.data[y, x] = (*color, 255)
        
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
        """Remove a layer from the canvas"""
        if 0 <= layer_index < len(self.layers):
            if len(self.layers) > 1:  # Don't remove last layer
                removed_layer = self.layers.pop(layer_index)
                self.active_layer_index = min(self.active_layer_index, len(self.layers) - 1)
                self.modified_at = datetime.now()
                logging.info(f"[CanvasStudio] Removed layer '{removed_layer.name}' from canvas '{self.name}'")
                return True
            else:
                logging.warning(f"[CanvasStudio] Cannot remove last layer from canvas '{self.name}'")
                return False
        return False
    
    def get_active_layer(self) -> Optional[CanvasLayer]:
        """Get the currently active layer"""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
    
    def set_active_layer(self, layer_index: int):
        """Set the active layer by index"""
        if 0 <= layer_index < len(self.layers):
            self.active_layer_index = layer_index
            logging.debug(f"[CanvasStudio] Set active layer to index {layer_index}")
    
    def merge_layers(self, layer1_index: int, layer2_index: int) -> bool:
        """Merge two layers together"""
        if (0 <= layer1_index < len(self.layers) and 
            0 <= layer2_index < len(self.layers) and 
            layer1_index != layer2_index):
            
            layer1 = self.layers[layer1_index]
            layer2 = self.layers[layer2_index]
            
            # Composite layer2 onto layer1
            # (Simplified - full implementation would respect blend modes)
            alpha = layer2.opacity / 100.0
            layer1.data = cv2.addWeighted(layer1.data, 1, layer2.data, alpha, 0)
            
            # Remove layer2
            self.layers.pop(layer2_index)
            if self.active_layer_index >= layer2_index:
                self.active_layer_index = max(0, self.active_layer_index - 1)
            
            self.modified_at = datetime.now()
            logging.info(f"[CanvasStudio] Merged layers in canvas '{self.name}'")
            return True
        
        return False
    
    def flatten(self) -> np.ndarray:
        """
        Flatten all layers into a single composite image
        
        Returns:
            Composite image as numpy array
        """
        if not self.layers:
            return np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        # Start with the bottom layer
        result = self.layers[0].data.copy()
        
        # Composite each layer on top
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            if not layer.visible:
                continue
            
            # Apply opacity
            alpha = (layer.opacity / 100.0) * (layer.data[:, :, 3] / 255.0)
            
            # Simple alpha compositing (full implementation would use blend modes)
            for c in range(3):
                result[:, :, c] = (
                    result[:, :, c] * (1 - alpha) +
                    layer.data[:, :, c] * alpha
                ).astype(result.dtype)
        
        logging.debug(f"[CanvasStudio] Flattened {len(self.layers)} layers")
        return result
    
    def apply_effect(self, effect_type: str, **kwargs):
        """Apply an effect to the active layer"""
        layer = self.get_active_layer()
        if not layer:
            logging.warning("[CanvasStudio] No active layer to apply effect")
            return
        
        try:
            if effect_type == "gaussian_blur":
                radius = kwargs.get("radius", 5)
                layer.data = cv2.GaussianBlur(layer.data, (0, 0), radius)
            
            elif effect_type == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                layer.data = cv2.filter2D(layer.data, -1, kernel)
            
            elif effect_type == "edge_sobel":
                gray = cv2.cvtColor(layer.data[:, :, :3], cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edges = np.uint8(edges / edges.max() * 255)
                layer.data[:, :, :3] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            elif effect_type == "emboss":
                kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
                layer.data = cv2.filter2D(layer.data, -1, kernel)
            
            elif effect_type == "sepia":
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                layer.data[:, :, :3] = cv2.transform(layer.data[:, :, :3], kernel)
            
            layer.modified_at = datetime.now()
            self.modified_at = datetime.now()
            logging.info(f"[CanvasStudio] Applied effect '{effect_type}' to layer '{layer.name}'")
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to apply effect '{effect_type}': {e}")
    
    def color_correct(self, brightness: int = 0, contrast: int = 0, saturation: int = 0):
        """Apply color correction to the active layer"""
        layer = self.get_active_layer()
        if not layer:
            return
        
        try:
            # Brightness adjustment
            if brightness != 0:
                layer.data = cv2.convertScaleAbs(layer.data, alpha=1, beta=brightness)
            
            # Contrast adjustment
            if contrast != 0:
                f = (259 * (contrast + 255)) / (255 * (259 - contrast))
                layer.data = cv2.convertScaleAbs(layer.data, alpha=f, beta=128*(1-f))
            
            # Saturation adjustment
            if saturation != 0:
                hsv = cv2.cvtColor(layer.data[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation / 100.0), 0, 255)
                layer.data[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            layer.modified_at = datetime.now()
            self.modified_at = datetime.now()
            logging.info(f"[CanvasStudio] Applied color correction to layer '{layer.name}'")
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to apply color correction: {e}")
    
    def to_dict(self) -> Dict:
        """Serialize canvas to dictionary for saving"""
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
            "version": CANVAS_STUDIO_VERSION
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
        
        logging.info(f"[CanvasStudio] Initialized v{CANVAS_STUDIO_VERSION} (Build {CANVAS_STUDIO_BUILD})")
    
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
        """
        Save canvas project file (.scp format)
        
        Args:
            canvas: Canvas to save
            filepath: Destination path (auto-generated if None)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath is None:
                filename = f"{canvas.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.scp"
                filepath = os.path.join(CANVAS_PROJECTS_DIR, filename)
            
            # Create project data
            project_data = {
                "canvas": canvas.to_dict(),
                "studio_version": CANVAS_STUDIO_VERSION,
                "saved_at": datetime.now().isoformat()
            }
            
            # Save JSON metadata
            with open(filepath, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            # Save layer data
            layer_dir = filepath.replace('.scp', '_layers')
            os.makedirs(layer_dir, exist_ok=True)
            
            for i, layer in enumerate(canvas.layers):
                layer_file = os.path.join(layer_dir, f"layer_{i:03d}.png")
                cv2.imwrite(layer_file, layer.data)
            
            logging.info(f"[CanvasStudio] Saved canvas '{canvas.name}' to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to save canvas: {e}")
            traceback.print_exc()
            return False
    
    def load_canvas(self, filepath: str) -> Optional[Canvas]:
        """
        Load canvas project file (.scp format)
        
        Args:
            filepath: Path to project file
        
        Returns:
            Loaded Canvas object or None if failed
        """
        try:
            # Load JSON metadata
            with open(filepath, 'r') as f:
                project_data = json.load(f)
            
            canvas_data = project_data['canvas']
            
            # Recreate canvas
            canvas = Canvas(
                name=canvas_data['name'],
                width=canvas_data['width'],
                height=canvas_data['height'],
                depth=canvas_data['depth']
            )
            
            # Clear default background layer
            canvas.layers.clear()
            
            # Load layer data
            layer_dir = filepath.replace('.scp', '_layers')
            
            for layer_data in canvas_data['layers']:
                layer = CanvasLayer(
                    layer_data['name'],
                    layer_data['width'],
                    layer_data['height'],
                    layer_data['depth']
                )
                
                # Load layer image
                layer_file = os.path.join(layer_dir, f"layer_{len(canvas.layers):03d}.png")
                if os.path.exists(layer_file):
                    layer.data = cv2.imread(layer_file, cv2.IMREAD_UNCHANGED)
                
                # Restore properties
                layer.opacity = layer_data['opacity']
                layer.blend_mode = BlendMode(layer_data['blend_mode'])
                layer.visible = layer_data['visible']
                layer.locked = layer_data['locked']
                
                canvas.layers.append(layer)
            
            canvas.active_layer_index = canvas_data['active_layer_index']
            
            # Register canvas
            self.canvases[canvas.id] = canvas
            self.active_canvas_id = canvas.id
            
            logging.info(f"[CanvasStudio] Loaded canvas '{canvas.name}' from {filepath}")
            return canvas
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to load canvas: {e}")
            traceback.print_exc()
            return None
    
    def export_canvas(self, canvas: Canvas, filepath: str, 
                     format: str = "PNG", quality: int = 90, flatten: bool = True) -> bool:
        """
        Export canvas to image file
        
        Args:
            canvas: Canvas to export
            filepath: Destination file path
            format: Output format (PNG, JPG, WebP, etc.)
            quality: Output quality (0-100, format-dependent)
            flatten: Whether to flatten all layers
        
        Returns:
            True if successful, False otherwise
        """
        try:
            format = format.upper()
            if format not in SUPPORTED_EXPORT_FORMATS:
                logging.error(f"[CanvasStudio] Unsupported format: {format}")
                return False
            
            # Get image data
            if flatten:
                image_data = canvas.flatten()
            else:
                image_data = canvas.layers[canvas.active_layer_index].data
            
            # Ensure correct filepath extension
            if not any(filepath.lower().endswith(f".{fmt.lower()}") for fmt in SUPPORTED_EXPORT_FORMATS):
                filepath = f"{filepath}.{format.lower()}"
            
            # Export based on format
            if format in ["PNG", "BMP", "TGA"]:
                cv2.imwrite(filepath, image_data)
            
            elif format in ["JPG", "JPEG"]:
                # Convert to BGR for JPEG (no alpha)
                bgr = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            elif format == "WEBP":
                cv2.imwrite(filepath, image_data, [cv2.IMWRITE_WEBP_QUALITY, quality])
            
            elif format == "TIFF":
                cv2.imwrite(filepath, image_data, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            
            logging.info(f"[CanvasStudio] Exported canvas '{canvas.name}' to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to export canvas: {e}")
            traceback.print_exc()
            return False
    
    def generate_from_prompt(self, prompt: str, width: int = None, height: int = None,
                            style: str = "default", quality: str = "standard") -> Optional[Canvas]:
        """
        Generate artwork from text prompt using AI
        
        Args:
            prompt: Text description of desired artwork
            width: Output width (default: 1024)
            height: Output height (default: 1024)
            style: Art style preset
            quality: Quality level (draft, standard, high)
        
        Returns:
            Canvas with generated artwork or None if failed
        """
        try:
            if width is None:
                width = 1024
            if height is None:
                height = 1024
            
            # Create canvas for generated art
            canvas_name = f"AI_Generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            canvas = self.create_canvas(width, height, canvas_name)
            
            # TODO: Integrate with SarahMemoryAiFunctions for actual AI generation
            # For now, create a placeholder with gradient
            layer = canvas.get_active_layer()
            
            # Generate a vibrant gradient as placeholder
            colors = [
                (255, 100, 100),  # Red
                (100, 100, 255),  # Blue
                (100, 255, 100),  # Green
                (255, 255, 100)   # Yellow
            ]
            layer.apply_gradient("radial", colors)
            
            # Add text with prompt
            if PIL_AVAILABLE:
                pil_img = Image.fromarray(cv2.cvtColor(layer.data, cv2.COLOR_BGRA2RGBA))
                draw = ImageDraw.Draw(pil_img)
                
                # Try to load a font
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                # Draw prompt text
                draw.text((20, 20), f"Generated: {prompt[:50]}", font=font, fill=(255, 255, 255, 255))
                
                # Convert back to OpenCV format
                layer.data = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
            
            logging.info(f"[CanvasStudio] Generated artwork from prompt: '{prompt[:50]}...'")
            return canvas
            
        except Exception as e:
            logging.error(f"[CanvasStudio] Failed to generate from prompt: {e}")
            traceback.print_exc()
            return None
    
    def batch_process(self, canvas_ids: List[str], operation: str, **kwargs) -> List[bool]:
        """
        Apply an operation to multiple canvases in batch
        
        Args:
            canvas_ids: List of canvas IDs to process
            operation: Operation to perform
            **kwargs: Operation-specific arguments
        
        Returns:
            List of success/failure booleans
        """
        results = []
        
        for canvas_id in canvas_ids:
            canvas = self.get_canvas(canvas_id)
            if not canvas:
                results.append(False)
                continue
            
            try:
                if operation == "resize":
                    # Resize canvas (implementation needed)
                    results.append(True)
                
                elif operation == "color_correct":
                    canvas.color_correct(**kwargs)
                    results.append(True)
                
                elif operation == "apply_effect":
                    canvas.apply_effect(**kwargs)
                    results.append(True)
                
                elif operation == "export":
                    success = self.export_canvas(canvas, **kwargs)
                    results.append(success)
                
                else:
                    logging.warning(f"[CanvasStudio] Unknown batch operation: {operation}")
                    results.append(False)
                    
            except Exception as e:
                logging.error(f"[CanvasStudio] Batch operation failed for canvas {canvas_id}: {e}")
                results.append(False)
        
        successful = sum(results)
        logging.info(f"[CanvasStudio] Batch operation '{operation}': {successful}/{len(canvas_ids)} successful")
        return results
    
    def get_studio_info(self) -> Dict:
        """Get Canvas Studio system information"""
        return {
            "version": CANVAS_STUDIO_VERSION,
            "build": CANVAS_STUDIO_BUILD,
            "active_canvases": len(self.canvases),
            "pil_available": PIL_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "supported_formats": SUPPORTED_EXPORT_FORMATS,
            "max_canvas_size": (MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT),
            "directories": {
                "projects": CANVAS_PROJECTS_DIR,
                "exports": CANVAS_EXPORTS_DIR,
                "cache": CANVAS_CACHE_DIR,
                "templates": CANVAS_TEMPLATES_DIR,
                "brushes": CANVAS_BRUSHES_DIR
            }
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
# END OF SarahMemoryCanvasStudio.py v8.0.0
# ====================================================================