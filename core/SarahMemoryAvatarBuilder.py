"""--==The SarahMemory Project==--
File: SarahMemoryAvatarBuilder.py
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

SarahMemory-AvatarBuilder.py
Deterministic local avatar asset builder for SarahMemory AiOS.

PURPOSE:
- Build stable 2D avatar assets locally from existing artwork.
- Build deterministic TRUE_3D_GOLD_MESH_RUNTIME GLB avatar proxy assets locally.
- Eliminate suit jitter / line distortion by locking one suit base.
- Composite face/hair/expression region over a fixed suit layer.
- Export clean sprite frames, sprite sheets, and real GLB mesh bodies for frontend / backend use.

DESIGN RULES:
- Local-only processing.
- No external API calls.
- No new runtime dependency on cloud image generation.
- Uses existing SarahMemory pathing when available.
- Safe for repeated rebuilds.
- Produces deterministic outputs from the same inputs.

OUTPUTS:
- suit_base_locked.png
- face_only_01.png ... face_only_08.png
- avatar_frame_01.png ... avatar_frame_08.png
- avatar_sprite_sheet.png
- avatar_manifest.json

USAGE EXAMPLES:
python SM-AVATARBUILD.py --sheet "C:\\SarahMemory\\data\\canvas\\imports\\sprite_sheet.png" --suit-ref "C:\\SarahMemory\\data\\canvas\\imports\\suit_ref.png"
python SM-AVATARBUILD.py --sheet "./sprite_sheet.png" --suit-ref "./suit_ref.png" --rows 2 --cols 4 --outdir "./build/avatar"
python SM-AVATARBUILD.py --sheet "./sheet.png" --suit-ref "./multihair_strip.png" --suit-column 4

NOTES:
- If --suit-ref is a multi-column strip (e.g. brown/red/blonde/pink),
  use --suit-column to pick the desired column (1-based).
- This script does NOT invent new art. It stabilizes and rebuilds your
  existing avatar assets locally so the suit remains consistent.
===============================================================================
"""
from __future__ import annotations

import os
import sys
import uuid
import tempfile
import hashlib
import copy
import re
from pathlib import Path
import json
import math
import argparse
import logging
import subprocess
import struct
import shutil
import time
import threading
import queue
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

try:
    from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageStat
    _PIL_AVAILABLE = True
except ImportError:
    # Runtime GLB inspection must work on headless servers without Pillow.
    Image = ImageFilter = ImageOps = ImageChops = ImageStat = None
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# SarahMemory globals integration (best-effort, never hard-fail)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
    BASE_DIR = getattr(config, "BASE_DIR", os.getcwd())
    DATA_DIR = getattr(config, "DATA_DIR", os.path.join(BASE_DIR, "data"))
    CANVAS_DIR = getattr(config, "CANVAS_DIR", os.path.join(DATA_DIR, "canvas"))
    CANVAS_EXPORTS_DIR = getattr(config, "CANVAS_EXPORTS_DIR", os.path.join(CANVAS_DIR, "exports"))
except Exception:
    config = None
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CANVAS_DIR = os.path.join(DATA_DIR, "canvas")
    CANVAS_EXPORTS_DIR = os.path.join(CANVAS_DIR, "exports")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SM-AVATARBUILD")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [SM-AVATARBUILD] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


# v9.1.0a: bounded deterministic asset utility; no autonomous execution authority.
MODULE_VERSION = "9.1.0a"
_BUILD_STATUS_LOCK = threading.RLock()
_BLENDER_AUTHORIZATION_CTX = threading.local()
_BUILD_STATUS = {"last_result": "not_run", "last_error": "", "last_finished": "", "builds": 0, "failures": 0}


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(os.environ.get(name, str(default)))))
    except (ValueError, OverflowError, TypeError):
        return default


def _max_pixels() -> int:
    return _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_PIXELS", 16000000, 65536, 64000000)


def _require_pillow() -> None:
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for 2D avatar processing (pip install pillow). 3D inspection remains available.")


def _validate_image_size(size: Tuple[int, int]) -> None:
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0 or w * h > _max_pixels():
        raise ValueError("avatar image dimensions outside allowed pixel budget")


def _canonical_child(path: str, root: str, *, must_exist: bool = False) -> str:
    base = Path(root).expanduser().resolve()
    child = Path(path).expanduser().resolve(strict=must_exist)
    if child != base and base not in child.parents:
        raise ValueError("asset path escapes the approved output root")
    return str(child)


def _safe_asset_name(name: str) -> str:
    name = str(name or "").strip()
    if not name or len(name) > 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
        raise ValueError("invalid avatar artifact name")
    return name


def _asset_path(root: str, filename: str) -> str:
    _safe_asset_name(filename)
    candidate = os.path.join(os.path.abspath(root), filename)
    _canonical_child(candidate, root)
    return candidate


def _validate_output_root(root: str) -> str:
    if not isinstance(root, (str, os.PathLike)) or not str(root).strip():
        raise ValueError("output directory is required")
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("output directory does not exist")
    return str(resolved)


def _safe_finite(value: Any, default: float, low: float, high: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("nonfinite")
        return max(low, min(high, v))
    except (TypeError, ValueError, OverflowError):
        return default


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    ensure_dir(parent)
    _canonical_child(path, parent)
    fd, staged = tempfile.mkstemp(prefix=".smbuilder_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


def _build_result(ok: bool, error: str = "") -> None:
    with _BUILD_STATUS_LOCK:
        _BUILD_STATUS["builds"] += 1
        _BUILD_STATUS["failures"] += int(not ok)
        _BUILD_STATUS["last_result"] = "verified" if ok else "failed"
        _BUILD_STATUS["last_error"] = str(error)[:240]
        _BUILD_STATUS["last_finished"] = datetime.now().isoformat()


def _check_blender_operator_optin() -> None:
    if os.environ.get("SARAH_AVATAR_BUILDER_ALLOW_BLENDER", "").strip().lower() not in {"1", "true", "yes", "on"} or os.environ.get("SARAH_AVATAR_BUILDER_USER_CONSENTED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise PermissionError("explicit Blender build opt-in and user-consent request are required")


def _authorize_blender_build(action: str, target: str, blender_binary: str = "") -> Tuple[Dict[str, Any], Any]:
    # Configuration requests a build; only OperatorCore may authorize it.
    explicit = os.environ.get("SARAH_AVATAR_BUILDER_ALLOW_BLENDER", "").strip().lower() in {"1", "true", "yes", "on"}
    consent = os.environ.get("SARAH_AVATAR_BUILDER_USER_CONSENTED", "").strip().lower() in {"1", "true", "yes", "on"}
    if not explicit or not consent:
        raise PermissionError("Blender build requires explicit operator opt-in and consent, followed by OperatorCore authorization")
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
    except Exception as exc:
        raise PermissionError("OperatorCore authorization unavailable") from exc
    proposed = {
        "action_type": "avatar_asset_build", "capability_name": "avatar.asset.build",
        "target": target, "target_ref": target, "blender_binary": os.path.realpath(blender_binary) if blender_binary else "",
        "risk_level": "TIER_2_BOUNDED_LOCAL_OPERATION",
        "required_permissions": ["FilesystemWrite", "ProcessExecute"],
        "verification_checks": [{"name": "glb_structure_valid", "required": True}],
        "expected_result": "A newly created structurally validated GLB is promoted to the configured asset directory",
        "preconditions": ["OperatorCore authorization", "bounded subprocess", "staged output"],
        "rollback_plan": [{"name": "retain_prior_asset", "required": True}],
        "touches_filesystem": True, "touches_network": False,
        "dry_run": False, "apply": True,
    }
    try:
        auth = authorize_external_domain_action(
            "Build local visual-only avatar assets using Blender", origin="SarahMemoryAvatarBuilder",
            proposed_action=proposed, meta={"local_only": True, "execution_authority": False},
            user_consented=consent,
        )
    except Exception as exc:
        raise PermissionError("OperatorCore failed to authorize Blender build") from exc
    if not isinstance(auth, dict) or auth.get("decision") != "ALLOW" or auth.get("allow") is not True or auth.get("domain_execution_authorized") is not True:
        raise PermissionError("OperatorCore denied Blender build")
    _BLENDER_AUTHORIZATION_CTX.operator_review = auth
    return auth, proposed


def _record_blender_outcome(auth: Dict[str, Any], ok: bool, *, artifact: str = "", error: str = "") -> None:
    _BLENDER_AUTHORIZATION_CTX.operator_review = None
    try:
        from SarahMemoryOperatorCore import record_external_domain_action_result
        record_external_domain_action_result(
            auth, {"ok": ok, "artifact": os.path.basename(artifact), "error": error[:200], "execution_authority": False},
            verified=ok, origin="SarahMemoryAvatarBuilder",
        )
    except Exception:
        logger.warning("OperatorCore build result receipt unavailable")


def _promote_verified_asset(staged: str, final: str) -> str:
    # Never overwrite an active asset until the newly staged GLB is structurally verified.
    proof = _read_glb_header(staged)
    if not proof.get("valid_glb_structure") or not proof.get("has_mesh"):
        raise ValueError("staged GLB failed structural and mesh validation")
    parent = os.path.dirname(os.path.abspath(final))
    _canonical_child(final, parent)
    backup = ""
    if os.path.lexists(final):
        backups = list(Path(parent).glob(Path(final).name + ".rollback.*"))
        if len(backups) >= _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_ROLLBACKS", 3, 1, 20):
            raise RuntimeError("rollback backup budget exhausted; manually archive existing backups")
        backup = final + ".rollback." + uuid.uuid4().hex[:12]
        os.replace(final, backup)
    try:
        os.replace(staged, final)
    except BaseException:
        if backup:
            os.replace(backup, final)
        raise
    return backup


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class BuildConfig:
    sheet_path: str
    suit_ref_path: str
    outdir: str
    rows: int = 2
    cols: int = 4
    suit_column: int = 1
    neck_y_ratio: float = 0.335
    blend_top_ratio: float = 0.295
    blend_bottom_ratio: float = 0.435
    hair_face_margin_ratio: float = 0.08
    sharpen_face: bool = True
    feather_radius: int = 14
    export_face_only: bool = True
    export_sheet: bool = True
    sheet_name: str = "avatar_sprite_sheet.png"
    suit_name: str = "suit_base_locked.png"
    manifest_name: str = "avatar_manifest.json"
    transparent_background: bool = True


@dataclass
class FrameRecord:
    index: int
    name: str
    filename: str
    width: int
    height: int

@dataclass
class Avatar3DBuildConfig:
    """Configuration for the local Blender-backed 3D avatar build lane."""
    concept_image: str
    outdir: str
    avatar_name: str = "sarahmemory_3d_avatar"
    blender_path: str = ""
    gpu_backend: str = "AUTO"
    save_blend: bool = True
    render_preview: bool = False
    keep_blender_script: bool = True
    poly_target: int = 12000000
    timeout_seconds: int = 0
    vfx_enabled: bool = True
    vfx_quality: str = "goldstandard_entity"
    vfx_intensity: float = 0.92


@dataclass
class BlendExportConfig:
    """Configuration for exporting an existing Blender source file into GLB."""
    blend_path: str
    outdir: str
    avatar_name: str = "sarahmemory_3d_avatar"
    blender_path: str = ""
    gpu_backend: str = "AUTO"
    timeout_seconds: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_rgba(path: str) -> Image.Image:
    _require_pillow()
    with Image.open(path) as img:
        _validate_image_size(img.size)
        return img.convert("RGBA")


def save_png(img: Image.Image, path: str) -> None:
    _require_pillow()
    _validate_image_size(img.size)
    ensure_dir(os.path.dirname(path))
    img.save(path, format="PNG", optimize=True)


def crop_grid(sheet: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    _require_pillow()
    w, h = sheet.size
    _validate_image_size((w, h))
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0 or rows > h or cols > w:
        raise ValueError("rows/cols must fit within the source image")
    if rows * cols > _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_FRAMES", 64, 1, 256):
        raise ValueError("sprite frame budget exceeded")
    frame_w = w // cols
    frame_h = h // rows
    frames: List[Image.Image] = []

    for r in range(rows):
        for c in range(cols):
            left = c * frame_w
            top = r * frame_h
            right = left + frame_w
            bottom = top + frame_h
            frames.append(sheet.crop((left, top, right, bottom)).convert("RGBA"))
    return frames


def extract_column(img: Image.Image, column_index_1_based: int) -> Image.Image:
    col = max(1, int(column_index_1_based))
    width, height = img.size

    _validate_image_size(img.size)
    # A square/portrait image is a single suit reference, not a four-column strip.
    if width >= height * 2 and width >= 4:
        possible_cols = 4
        if col > possible_cols:
            raise ValueError("suit column exceeds available strip columns")
        col_w = width // possible_cols
        left = (col - 1) * col_w
        return img.crop((left, 0, left + col_w, height)).convert("RGBA")
    if col != 1:
        raise ValueError("requested suit column but source is not a valid multi-column strip")
    return img.convert("RGBA")


def resize_cover(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    _require_pillow()
    _validate_image_size(img.size)
    _validate_image_size(size)
    target_w, target_h = size
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(new_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(new_w / src_ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h)).convert("RGBA")


def resize_contain(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    _require_pillow()
    _validate_image_size(img.size)
    _validate_image_size(size)
    target_w, target_h = size
    src_w, src_h = img.size

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.alpha_composite(resized, (left, top))
    return canvas


def build_vertical_gradient_mask(width: int, height: int, start_ratio: float, end_ratio: float) -> Image.Image:
    start_y = int(height * start_ratio)
    end_y = int(height * end_ratio)
    if end_y <= start_y:
        end_y = start_y + 1

    _validate_image_size((width, height))
    rows = bytearray()
    for y in range(height):
        if y <= start_y:
            alpha = 255
        elif y >= end_y:
            alpha = 0
        else:
            alpha = int(255 * (1.0 - (y - start_y) / float(end_y - start_y)))
        rows.extend(bytes((alpha,)) * width)
    mask = Image.frombytes("L", (width, height), bytes(rows))
    return mask.filter(ImageFilter.GaussianBlur(radius=8))


def extract_face_hair_region(frame: Image.Image, cfg: BuildConfig) -> Tuple[Image.Image, Image.Image]:
    """Return face/hair-only overlay and its alpha mask."""
    w, h = frame.size
    overlay = frame.copy()

    # Build a soft vertical alpha that preserves head/hair and fades before torso.
    base_mask = build_vertical_gradient_mask(
        w,
        h,
        start_ratio=cfg.blend_top_ratio,
        end_ratio=cfg.blend_bottom_ratio,
    )

    # Strengthen alpha in upper region.
    alpha = base_mask.point(lambda v: max(0, min(255, int(v * 1.15))))
    # Preserve the source transparency; never make fully transparent artwork opaque.
    original_alpha = overlay.getchannel("A")
    overlay.putalpha(ImageChops.multiply(original_alpha, alpha))

    if cfg.sharpen_face:
        rgb = overlay.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))
        overlay = Image.merge("RGBA", (*rgb.split(), overlay.getchannel("A")))

    return overlay, overlay.getchannel("A")


def isolate_suit_base(suit_ref: Image.Image, target_size: Tuple[int, int], cfg: BuildConfig) -> Image.Image:
    """
    Normalize the suit reference to the target frame size.
    Keeps the suit deterministic across all output frames.
    """
    target_w, target_h = target_size
    ref = resize_cover(suit_ref, target_size)

    # Preserve full torso / suit.
    # If there is residual head/hair in source, soften upper crop.
    cleaned = ref.copy()

    # Fade top area slightly so face overlays blend cleaner.
    fade_end = max(0, min(target_h, int(target_h * _safe_finite(cfg.neck_y_ratio, 0.335, 0.0, 1.0))))
    fade_rows = bytearray()
    for y in range(target_h):
        t = y / max(1, fade_end)
        alpha = int(255 * min(1.0, max(0.0, t * 1.5))) if y < fade_end else 255
        fade_rows.extend(bytes((alpha,)) * target_w)
    fade_mask = Image.frombytes("L", (target_w, target_h), bytes(fade_rows))
    fade_mask = fade_mask.filter(ImageFilter.GaussianBlur(radius=max(0, min(64, int(cfg.feather_radius)))))
    cleaned.putalpha(ImageChops.multiply(cleaned.getchannel("A"), fade_mask))
    return cleaned


def composite_frame(suit_base: Image.Image, face_overlay: Image.Image) -> Image.Image:
    base = suit_base.copy()
    base.alpha_composite(face_overlay, (0, 0))
    return base


def make_sprite_sheet(frames: List[Image.Image], rows: int, cols: int) -> Image.Image:
    _require_pillow()
    if not frames or rows < 1 or cols < 1 or len(frames) > rows * cols:
        raise ValueError("invalid sprite sheet frame grid")
    fw, fh = frames[0].size
    _validate_image_size((fw * cols, fh * rows))
    sheet = Image.new("RGBA", (fw * cols, fh * rows), (0, 0, 0, 0))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(frames):
                break
            sheet.alpha_composite(frames[idx], (c * fw, r * fh))
            idx += 1
    return sheet


def default_outdir() -> str:
    return os.path.join(CANVAS_EXPORTS_DIR, "avatar_build")


def build_manifest(
    cfg: BuildConfig,
    suit_path: str,
    frame_records: List[FrameRecord],
    sheet_path: Optional[str],
) -> Dict[str, Any]:
    return {
        "builder": "SM-AVATARBUILD",
        "version": MODULE_VERSION,
        "built_at": datetime.now().isoformat(),
        "local_only": True,
        "sheet_source": cfg.sheet_path,
        "suit_reference_source": cfg.suit_ref_path,
        "output_dir": cfg.outdir,
        "rows": cfg.rows,
        "cols": cfg.cols,
        "frame_count": len(frame_records),
        "suit_base": os.path.basename(suit_path),
        "sprite_sheet": os.path.basename(sheet_path) if sheet_path else None,
        "frames": [asdict(r) for r in frame_records],
        "notes": [
            "Suit topology locked from reference image.",
            "Upper face/hair region composited over fixed suit base.",
            "Deterministic rebuild path for 2D avatar pipeline.",
        ],
    }


# ---------------------------------------------------------------------------
# 3D avatar / GLB builder lane
# ---------------------------------------------------------------------------
def default_3d_outdir() -> str:
    """Default runtime location for WebUI / AvatarPanel 3D GLB files."""
    return os.path.join(BASE_DIR, "resources", "avatars", "3D")


def _clamp01(v: float) -> float:
    return _safe_finite(v, 0.0, 0.0, 1.0)


def _average_region_color(
    img: Image.Image,
    box_ratio: Tuple[float, float, float, float],
    fallback: Tuple[float, float, float],
    predicate: Optional[callable] = None,
) -> Tuple[float, float, float]:
    """Sample an approximate concept-image region into Blender-friendly 0-1 RGB."""
    try:
        w, h = img.size
        x0 = max(0, min(w - 1, int(w * box_ratio[0])))
        y0 = max(0, min(h - 1, int(h * box_ratio[1])))
        x1 = max(x0 + 1, min(w, int(w * box_ratio[2])))
        y1 = max(y0 + 1, min(h, int(h * box_ratio[3])))
        region = img.crop((x0, y0, x1, y1)).convert("RGB")
        # Bound color sampling independently of source image dimensions.
        if region.width * region.height > 65536:
            region.thumbnail((256, 256))
        pixels = list(region.getdata())
        if predicate is not None:
            filtered = [p for p in pixels if predicate(p)]
            if filtered:
                pixels = filtered
        if not pixels:
            return fallback
        r = sum(p[0] for p in pixels) / (255.0 * len(pixels))
        g = sum(p[1] for p in pixels) / (255.0 * len(pixels))
        b = sum(p[2] for p in pixels) / (255.0 * len(pixels))
        return (_clamp01(r), _clamp01(g), _clamp01(b))
    except Exception:
        return fallback


def extract_concept_palette(concept_image: str) -> Dict[str, Any]:
    """
    Extract a deterministic color/material hint packet from the provided model image.

    This does not hallucinate mesh geometry from one picture. It uses the image as a
    local design reference for palette and style: pink/magenta hair, glossy black suit,
    cyan emissive tracing, skin tone, and boot/suit darkness.
    """
    defaults = {
        "hair": (0.92, 0.08, 0.42),
        "suit": (0.005, 0.008, 0.012),
        "glow": (0.00, 0.90, 1.00),
        "skin": (0.86, 0.62, 0.50),
        "boot": (0.005, 0.006, 0.008),
    }
    meta: Dict[str, Any] = {
        "source_image": os.path.abspath(concept_image) if concept_image else "",
        "source_exists": bool(concept_image and os.path.exists(concept_image)),
        "palette_source": "defaults",
    }
    if not concept_image or not os.path.exists(concept_image):
        meta.update(defaults)
        return meta

    try:
        img = Image.open(concept_image).convert("RGB")
        meta["image_size"] = list(img.size)
        meta["hair"] = _average_region_color(
            img,
            (0.40, 0.05, 0.64, 0.35),
            defaults["hair"],
            lambda p: p[0] > 110 and p[2] > 80 and p[1] < 125 and p[0] > p[1] * 1.30,
        )
        meta["suit"] = _average_region_color(
            img,
            (0.40, 0.20, 0.60, 0.72),
            defaults["suit"],
            lambda p: (p[0] + p[1] + p[2]) < 150,
        )
        meta["glow"] = _average_region_color(
            img,
            (0.35, 0.18, 0.66, 0.77),
            defaults["glow"],
            lambda p: p[2] > 120 and p[1] > 100 and p[0] < 120,
        )
        meta["skin"] = _average_region_color(
            img,
            (0.46, 0.08, 0.55, 0.20),
            defaults["skin"],
            lambda p: p[0] > 120 and p[1] > 70 and p[2] > 55 and p[0] >= p[1] >= p[2] * 0.55,
        )
        meta["boot"] = _average_region_color(
            img,
            (0.43, 0.62, 0.58, 0.84),
            defaults["boot"],
            lambda p: (p[0] + p[1] + p[2]) < 150,
        )
        meta["palette_source"] = "concept_image_sampled"
        return meta
    except Exception as e:
        meta.update(defaults)
        meta["palette_error"] = str(e)
        return meta


def _resolve_real_blender_binary(candidate: str) -> str:
    """Return blender.exe instead of blender-launcher.exe when possible."""
    candidate = os.path.abspath(os.path.expanduser(candidate))
    basename = os.path.basename(candidate).lower()
    if basename not in {"blender", "blender.exe", "blender-launcher.exe"}:
        raise ValueError("only the Blender executable is permitted")
    if basename != "blender-launcher.exe":
        if not os.path.isfile(candidate):
            raise FileNotFoundError("Blender binary is not a regular file")
        return candidate
    sibling = os.path.join(os.path.dirname(candidate), "blender.exe")
    if os.path.exists(sibling):
        return os.path.abspath(sibling)
    # Do not recurse through Program Files or drive roots during startup.
    raise RuntimeError(
        "blender-launcher.exe was supplied, but the real blender.exe was not found. "
        "Use blender.exe directly for background GLB builds."
    )


def find_blender_executable(explicit_path: str = "") -> str:
    """Find Blender without hard-failing import/runtime."""
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    for env_key in ("BLENDER_PATH", "SM_BLENDER_PATH"):
        val = os.environ.get(env_key)
        if val:
            candidates.append(val)

    try:
        if config is not None:
            val = getattr(config, "BLENDER_PATH", "")
            if val:
                candidates.append(str(val))
    except Exception:
        pass

    # Prefer explicit blender.exe paths over launcher wrappers.
    candidates.extend([
        r"C:\Blender51\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    ])

    which = shutil.which("blender")
    if which:
        candidates.append(which)

    program_files = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    versions = ("5.1", "5.0", "4.4", "4.3", "4.2", "4.1", "4.0", "3.6")
    for root in program_files:
        if not root:
            continue
        for v in versions:
            candidates.extend([
                os.path.join(root, "Blender Foundation", f"Blender {v}", "blender.exe"),
                os.path.join(root, "Blender Foundation", f"Blender {v}", "blender-launcher.exe"),
            ])

    for candidate in candidates:
        try:
            candidate = os.path.abspath(os.path.expanduser(candidate))
            if os.path.exists(candidate):
                return _resolve_real_blender_binary(candidate)
        except Exception:
            continue

    raise RuntimeError(
        "Blender executable not found. Install Blender or pass --blender \"C:\\Path\\To\\blender.exe\"."
    )

def _jsonable_palette(palette: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in palette.items():
        if isinstance(v, tuple):
            out[k] = [float(x) for x in v]
        else:
            out[k] = v
    return out


def generate_blender_avatar_script(settings: Dict[str, Any]) -> str:
    """Return a self-contained Blender Python script that creates/export a GLB."""
    settings_json = json.dumps(settings, indent=2)
    return f'''# Auto-generated by SM-AVATARBUILD.py. Local-only deterministic 3D avatar build.
import os
import math
import json
import bpy
from mathutils import Vector

CFG = json.loads({settings_json!r})
POLY_TARGET = max(90000, min(2000000, int(CFG.get('poly_target', 2000000) or 2000000)))
DETAIL_SCALE = max(1.0, min(3.2, (POLY_TARGET / 400000.0) ** 0.38))
AUTHORING_2M = POLY_TARGET >= 1500000
AUTHORING_10M = False


def detail_int(base, minimum=8, maximum=256):
    try:
        return max(int(minimum), min(int(maximum), int(round(float(base) * DETAIL_SCALE))))
    except Exception:
        return int(base)


def _color(name, default):
    val = CFG.get("palette", {{}}).get(name, default)
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return (float(val[0]), float(val[1]), float(val[2]), 1.0)
    return (float(default[0]), float(default[1]), float(default[2]), 1.0)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def set_socket(node, socket_names, value):
    if node is None:
        return
    for s in socket_names:
        if s in node.inputs:
            try:
                node.inputs[s].default_value = value
                return
            except Exception:
                pass


def make_mat(name, base=(1,1,1,1), metallic=0.0, roughness=0.45, emission=None, emission_strength=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        set_socket(bsdf, ('Base Color',), base)
        set_socket(bsdf, ('Metallic',), float(metallic))
        set_socket(bsdf, ('Roughness',), float(roughness))
        set_socket(bsdf, ('Alpha',), float(alpha))
        if emission is not None:
            set_socket(bsdf, ('Emission Color', 'Emission'), emission)
            set_socket(bsdf, ('Emission Strength',), float(emission_strength))
    mat.diffuse_color = (base[0], base[1], base[2], alpha)
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.use_screen_refraction = True
    return mat


def assign(obj, mat):
    if obj and mat:
        obj.data.materials.append(mat)
    return obj


def shade(obj):
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        obj.select_set(False)
    except Exception:
        pass
    return obj


def add_uv(name, loc, scale, mat, segments=48, rings=24):
    segments = detail_int(segments, 16, 512 if AUTHORING_10M else 288)
    rings = detail_int(rings, 8, 256 if AUTHORING_10M else 144)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign(obj, mat)
    shade(obj)
    return obj


def add_cylinder_between(name, p1, p2, radius, mat, vertices=32):
    p1 = Vector(p1)
    p2 = Vector(p2)
    delta = p2 - p1
    length = max(0.001, delta.length)
    mid = p1 + delta * 0.5
    vertices = detail_int(vertices, 12, 384 if AUTHORING_10M else 192)
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat('Z', 'Y').to_euler()
    assign(obj, mat)
    shade(obj)
    return obj


def add_box(name, loc, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    assign(obj, mat)
    shade(obj)
    return obj


def add_torus(name, loc, major_radius, minor_radius, mat, rotation=(0.0, 0.0, 0.0), major_segments=128, minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(
        major_segments=major_segments,
        minor_segments=minor_segments,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=loc,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    shade(obj)
    return obj


def add_curve(name, points, mat, bevel=0.018):
    curve = bpy.data.curves.new(name=name, type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = detail_int(24 if AUTHORING_10M else 16, 8, 128 if AUTHORING_10M else 64)
    curve.bevel_depth = bevel
    curve.bevel_resolution = detail_int(6 if AUTHORING_10M else 4, 2, 18 if AUTHORING_10M else 12)
    spline = curve.splines.new('POLY')
    spline.points.add(len(points)-1)
    for p, co in zip(spline.points, points):
        p.co = (float(co[0]), float(co[1]), float(co[2]), 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    assign(obj, mat)
    return obj


def add_hair_strand(name, offset, mat):
    x = offset
    return add_curve(
        name,
        [
            (x, 0.11, 3.62),
            (x * 1.05, 0.20, 3.34),
            (x * 1.18, 0.18, 3.06),
            (x * 1.05, 0.14, 2.76),
            (x * 0.92, 0.12, 2.52),
        ],
        mat,
        bevel=0.026,
    )


def look_at(obj, target):
    loc = Vector(obj.location)
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def add_authoring_smooth_modifier(obj, name='GoldStandardAuthoringSubd'):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return obj
    try:
        subd = obj.modifiers.new(name=name, type='SUBSURF')
        subd.levels = 3 if AUTHORING_10M else (2 if AUTHORING_2M else 1)
        subd.render_levels = min(4, subd.levels + (1 if AUTHORING_10M else 0))
    except Exception:
        pass
    try:
        weighted = obj.modifiers.new(name=name + '_WeightedNormals', type='WEIGHTED_NORMAL')
        weighted.keep_sharp = True
    except Exception:
        pass
    obj['sarahmemory_authoring_poly_target'] = POLY_TARGET
    obj['sarahmemory_goldstandard_detail_scale'] = round(DETAIL_SCALE, 4)
    return obj


def configure_gpu():
    backend = str(CFG.get('gpu_backend') or 'AUTO').upper()
    if backend in ('', 'AUTO'):
        preferred = ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL')
    else:
        preferred = (backend,)
    try:
        bpy.context.scene.render.engine = 'CYCLES'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for device_type in preferred:
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                enabled = 0
                for d in prefs.devices:
                    if getattr(d, 'type', '') != 'CPU':
                        d.use = True
                        enabled += 1
                    elif backend == 'CPU':
                        d.use = True
                if enabled or backend == 'CPU':
                    bpy.context.scene.cycles.device = 'GPU' if backend != 'CPU' else 'CPU'
                    CFG['gpu_selected'] = device_type
                    return
            except Exception:
                continue
    except Exception as exc:
        CFG['gpu_setup_error'] = str(exc)


def build_avatar():
    clear_scene()
    configure_gpu()

    skin = make_mat('SarahMemory_skin_from_concept', _color('skin', (0.86,0.62,0.50)), roughness=0.58)
    hair = make_mat('SarahMemory_magenta_hair_from_concept', _color('hair', (0.92,0.08,0.42)), roughness=0.32)
    suit = make_mat('SarahMemory_gloss_black_suit_from_concept', _color('suit', (0.005,0.008,0.012)), metallic=0.15, roughness=0.18)
    boot = make_mat('SarahMemory_gloss_black_boots', _color('boot', (0.005,0.006,0.008)), metallic=0.2, roughness=0.14)
    glow_col = _color('glow', (0.00,0.90,1.00))
    neon = make_mat('SarahMemory_cyan_emissive_lines_from_concept', glow_col, roughness=0.15, emission=glow_col, emission_strength=6.0)
    eye_mat = make_mat('SarahMemory_eye_black', (0.02,0.015,0.012,1), roughness=0.25)
    lip_mat = make_mat('SarahMemory_lip_soft', (0.75,0.18,0.28,1), roughness=0.45)
    vfx_enabled = bool(CFG.get('vfx_enabled', True))
    vfx_intensity = max(0.05, min(1.5, float(CFG.get('vfx_intensity', 0.92) or 0.92)))
    vfx_aura = make_mat('SarahMemory_VFX_Aura_Cyan_visual_only', (0.02,0.72,1.0,0.28), roughness=0.18, emission=(0.02,0.82,1.0,1), emission_strength=1.7 * vfx_intensity, alpha=0.28)
    vfx_magenta = make_mat('SarahMemory_VFX_EmotionBloom_Magenta_visual_only', (0.95,0.10,0.52,0.40), roughness=0.20, emission=(0.95,0.06,0.44,1), emission_strength=1.35 * vfx_intensity, alpha=0.38)
    vfx_shield = make_mat('SarahMemory_VFX_GovernanceShield_visual_only', (0.16,0.95,1.0,0.22), roughness=0.28, emission=(0.0,0.55,0.82,1), emission_strength=0.95 * vfx_intensity, alpha=0.22)

    # Humanoid proxy proportions based on the supplied concept sheet: T-pose, boots, long hair, neon suit.
    # TRUE_3D_GOLD_MESH_RUNTIME: these are real depth volumes, not billboard planes.
    add_uv('torso_gloss_suit', (0, -0.01, 2.18), (0.42, 0.24, 0.74), suit)
    add_uv('pelvis_gloss_suit', (0, -0.01, 1.48), (0.36, 0.23, 0.30), suit)
    add_uv('chest_suit_form_L', (-0.17, -0.18, 2.42), (0.18, 0.08, 0.18), suit, 32, 16)
    add_uv('chest_suit_form_R', (0.17, -0.18, 2.42), (0.18, 0.08, 0.18), suit, 32, 16)
    add_uv('ribcage_back_suit', (0, 0.18, 2.30), (0.46, 0.15, 0.62), suit, 48, 18)
    add_uv('spine_back_suit', (0, 0.315, 2.14), (0.075, 0.045, 0.66), suit, 32, 12)
    add_uv('lat_back_L_suit', (-0.31, 0.16, 2.23), (0.15, 0.105, 0.50), suit, 32, 12)
    add_uv('lat_back_R_suit', (0.31, 0.16, 2.23), (0.15, 0.105, 0.50), suit, 32, 12)
    add_uv('shoulder_back_mass_L', (-0.39, 0.11, 2.61), (0.15, 0.13, 0.15), suit, 32, 16)
    add_uv('shoulder_back_mass_R', (0.39, 0.11, 2.61), (0.15, 0.13, 0.15), suit, 32, 16)
    add_uv('hip_side_L_suit', (-0.31, 0.02, 1.47), (0.13, 0.17, 0.24), suit, 32, 12)
    add_uv('hip_side_R_suit', (0.31, 0.02, 1.47), (0.13, 0.17, 0.24), suit, 32, 12)
    add_uv('glute_back_L_suit', (-0.16, 0.19, 1.35), (0.17, 0.12, 0.20), suit, 32, 12)
    add_uv('glute_back_R_suit', (0.16, 0.19, 1.35), (0.17, 0.12, 0.20), suit, 32, 12)
    add_cylinder_between('neck_skin', (0, -0.01, 2.82), (0, -0.01, 3.08), 0.10, skin)
    add_uv('head_skin', (0, -0.03, 3.36), (0.25, 0.20, 0.30), skin, 48, 24)
    add_uv('head_back_volume_skin', (0, 0.105, 3.36), (0.235, 0.115, 0.285), skin, 48, 18)

    # Arms in T-pose with side/back mass so orbit view does not collapse into flat cylinders.
    for side, sx in [('L', -1), ('R', 1)]:
        add_uv(f'shoulder_{{side}}', (sx*0.43, -0.01, 2.62), (0.13,0.12,0.13), suit, 32, 16)
        add_cylinder_between(f'upper_arm_{{side}}_suit', (sx*0.44, -0.01, 2.58), (sx*0.92, -0.01, 2.55), 0.075, suit)
        add_cylinder_between(f'forearm_{{side}}_suit', (sx*0.92, -0.01, 2.55), (sx*1.40, -0.01, 2.53), 0.063, suit)
        add_uv(f'upper_arm_{{side}}_side_volume', (sx*0.69, 0.065, 2.56), (0.23,0.045,0.073), suit, 24, 10)
        add_uv(f'forearm_{{side}}_side_volume', (sx*1.16, 0.055, 2.54), (0.22,0.038,0.060), suit, 24, 10)
        add_uv(f'hand_{{side}}_skin', (sx*1.53, -0.03, 2.52), (0.105,0.045,0.055), skin, 24, 12)

    # Legs and boots.  Extra side/back volume protects the side/back silhouette.
    for side, sx in [('L', -1), ('R', 1)]:
        add_cylinder_between(f'thigh_{{side}}_suit', (sx*0.18, -0.01, 1.38), (sx*0.22, -0.01, 0.76), 0.115, suit)
        add_cylinder_between(f'calf_{{side}}_suit', (sx*0.22, -0.01, 0.76), (sx*0.20, -0.01, 0.24), 0.095, suit)
        add_uv(f'thigh_{{side}}_back_volume', (sx*0.20, 0.105, 1.08), (0.095,0.070,0.32), suit, 24, 10)
        add_uv(f'calf_{{side}}_back_volume', (sx*0.20, 0.095, 0.50), (0.078,0.058,0.25), suit, 24, 10)
        add_box(f'platform_boot_{{side}}', (sx*0.20, -0.08, 0.08), (0.13, 0.26, 0.08), boot)
        add_box(f'boot_toe_{{side}}', (sx*0.20, -0.25, 0.04), (0.135, 0.125, 0.060), boot)
        add_box(f'boot_heel_{{side}}', (sx*0.20, 0.075, 0.02), (0.120, 0.075, 0.085), boot)
        add_box(f'boot_shaft_{{side}}', (sx*0.20, -0.01, 0.33), (0.12, 0.12, 0.20), boot)
        add_cylinder_between(f'heel_{{side}}_cyan_core', (sx*0.20, 0.09, 0.06), (sx*0.20, 0.09, -0.22), 0.025, neon, vertices=16)

    # Face details.
    add_uv('eye_L', (-0.085, -0.218, 3.39), (0.025,0.013,0.018), eye_mat, 16, 8)
    add_uv('eye_R', (0.085, -0.218, 3.39), (0.025,0.013,0.018), eye_mat, 16, 8)
    add_curve('soft_smile_lip', [(-0.075,-0.232,3.28), (-0.025,-0.245,3.255), (0.035,-0.245,3.255), (0.085,-0.232,3.28)], lip_mat, bevel=0.008)

    # Hair cap and flowing strands.  The rear shell keeps side/back orbit views volumetric.
    add_uv('hair_cap_magenta', (0, 0.02, 3.49), (0.285,0.215,0.18), hair, 48, 18)
    add_uv('hair_back_mass_magenta', (0, 0.19, 3.21), (0.305,0.135,0.54), hair, 48, 18)
    add_uv('hair_side_mass_L_magenta', (-0.27, 0.055, 3.06), (0.095,0.090,0.50), hair, 32, 12)
    add_uv('hair_side_mass_R_magenta', (0.27, 0.055, 3.06), (0.095,0.090,0.50), hair, 32, 12)
    hair_count = detail_int(360 if AUTHORING_10M else (180 if AUTHORING_2M else 36), 36, 1800)
    for i in range(hair_count):
        t = i / max(1, hair_count - 1)
        x = -0.30 + (0.60 * t)
        add_hair_strand(f'magenta_hair_strand_{{i:02d}}', x, hair)

    # Cyan suit traces from the image: centerline, V chest, abdomen, limbs, boots.
    yfront = -0.285
    add_curve('neon_center_torso', [(0,yfront,2.72),(0,yfront,2.28),(0,yfront,1.78),(0,yfront,1.48)], neon, 0.012)
    add_curve('neon_chest_left_V', [(0,yfront,2.50),(-0.22,yfront,2.70),(-0.36,yfront,2.52)], neon, 0.011)
    add_curve('neon_chest_right_V', [(0,yfront,2.50),(0.22,yfront,2.70),(0.36,yfront,2.52)], neon, 0.011)
    add_curve('neon_waist_left', [(0,yfront,1.74),(-0.22,yfront,1.92),(-0.31,yfront,2.20)], neon, 0.011)
    add_curve('neon_waist_right', [(0,yfront,1.74),(0.22,yfront,1.92),(0.31,yfront,2.20)], neon, 0.011)
    add_curve('neon_pelvis_v', [(-0.23,yfront,1.62),(0,yfront,1.42),(0.23,yfront,1.62)], neon, 0.011)
    for side, sx in [('L', -1), ('R', 1)]:
        add_curve(f'neon_arm_{{side}}', [(sx*0.46,yfront,2.60),(sx*0.82,yfront,2.57),(sx*1.31,yfront,2.54)], neon, 0.010)
        add_curve(f'neon_leg_{{side}}', [(sx*0.18,yfront,1.32),(sx*0.22,yfront,0.80),(sx*0.20,yfront,0.28)], neon, 0.010)
        add_curve(f'neon_boot_{{side}}', [(sx*0.10,yfront,0.13),(sx*0.20,yfront,0.02),(sx*0.31,yfront,0.13)], neon, 0.010)

    # Rear cyan accents for back view compatibility.
    yback = 0.245
    add_curve('neon_back_spine', [(0,yback,2.68),(0,yback,2.18),(0,yback,1.55)], neon, 0.010)
    add_curve('neon_back_cross_L', [(-0.28,yback,2.22),(-0.05,yback,1.95),(-0.26,yback,1.68)], neon, 0.010)
    add_curve('neon_back_cross_R', [(0.28,yback,2.22),(0.05,yback,1.95),(0.26,yback,1.68)], neon, 0.010)

    # Construct-runtime visual environment anchor.  Avatar3D.tsx detects stage/grid/floor
    # names and separates them into the runtime stage group.  This is simulation
    # metadata/visual context only; it cannot execute hardware movement.
    stage_mat = make_mat('SarahMemory_construct_stage_mat', (0.01,0.025,0.035,1), metallic=0.05, roughness=0.30, emission=(0.0,0.22,0.28,1), emission_strength=0.18)
    add_box('construct_stage_floor', (0, 0.0, -0.04), (1.55, 1.55, 0.015), stage_mat)
    add_curve('construct_stage_orbit_ring', [(-1.05,0,-0.02),(-0.52,-0.52,-0.02),(0,-0.72,-0.02),(0.52,-0.52,-0.02),(1.05,0,-0.02),(0.52,0.52,-0.02),(0,0.72,-0.02),(-0.52,0.52,-0.02),(-1.05,0,-0.02)], neon, 0.006)

    # AvatarPanel VFX Organ Parts.  These are lightweight visual organs embedded
    # as named GLB geometry for identity/inspection; live pulse animation is owned
    # by Avatar3D.tsx and remains visual-only.
    if vfx_enabled:
        add_torus('vfx_governance_halo_ring_visual_only', (0, -0.02, 3.73), 0.46, 0.007, vfx_shield, rotation=(math.radians(90), 0, 0), major_segments=128, minor_segments=10)
        add_torus('vfx_voice_wave_chest_ring_visual_only', (0, -0.04, 2.14), 0.58, 0.0055, vfx_aura, rotation=(math.radians(90), 0, 0), major_segments=128, minor_segments=8)
        add_torus('vfx_construct_orbit_ring_outer_visual_only', (0, 0, 0.03), 1.20, 0.005, neon, rotation=(math.radians(90), 0, 0), major_segments=144, minor_segments=8)
        add_torus('vfx_construct_orbit_ring_inner_visual_only', (0, 0, 0.06), 0.82, 0.004, vfx_aura, rotation=(math.radians(90), 0, 0), major_segments=120, minor_segments=8)
        add_uv('vfx_emotion_core_aura_shell_visual_only', (0, 0.0, 2.05), (0.62, 0.42, 0.95), vfx_magenta, 48, 18)
        add_uv('vfx_governance_shield_shell_visual_only', (0, 0.0, 1.92), (0.92, 0.62, 1.55), vfx_shield, 48, 18)
        add_curve('vfx_trace_data_ribbon_A_visual_only', [(-0.74,-0.02,1.12),(-0.34,-0.25,1.82),(0.18,-0.05,2.45),(0.54,0.16,3.05)], vfx_aura, 0.005)
        add_curve('vfx_trace_data_ribbon_B_visual_only', [(0.74,0.02,1.04),(0.38,0.24,1.72),(-0.20,0.08,2.52),(-0.54,-0.16,3.10)], vfx_magenta, 0.0045)
        for i, ang in enumerate([0, 60, 120, 180, 240, 300]):
            r = math.radians(ang)
            x = math.cos(r) * 0.92
            y = math.sin(r) * 0.92
            add_uv('vfx_micro_spark_%02d_visual_only' % i, (x, y, 1.82 + (i % 3) * 0.28), (0.018,0.018,0.018), vfx_aura, 12, 6)

    # Stage/camera/lights so Blender preview opens usefully; export remains runtime-safe.
    bpy.ops.object.light_add(type='AREA', location=(0, -3.2, 4.2))
    bpy.context.object.name = 'avatar_key_light'
    bpy.context.object.data.energy = 900 if AUTHORING_10M else 650
    bpy.context.object.data.size = 4.6 if AUTHORING_10M else 4.0
    bpy.ops.object.camera_add(location=(0, -5.2, 2.45))
    cam = bpy.context.object
    look_at(cam, (0, -0.04, 1.90))
    bpy.context.scene.camera = cam

    # Metadata object marker.
    empty = bpy.data.objects.new('SarahMemory_Avatar_Metadata', None)
    empty['builder'] = 'SM-AVATARBUILD.py'
    empty['runtime_contract'] = 'TRUE_3D_GOLD_MESH_RUNTIME'
    empty['runtime_geometry'] = 'TRUE_3D_GLB_MESH'
    empty['gold_reference_role'] = 'visual_reference_only_not_geometry'
    empty['simulation_before_execution'] = True
    empty['physical_actuation_allowed_here'] = False
    empty['vfx_organs_contract'] = 'SARAHMEMORY_AVATAR_VFX_ORGANS_V1'
    empty['vfx_organs_visual_only'] = True
    empty['vfx_organs'] = 'aura,neuralPulse,voiceWave,constructGrid,governanceHalo,shieldShell,emotionBloom,dataRibbons,constructOrbit,particles'
    empty['concept_image'] = CFG.get('concept_image', '')
    empty['design_note'] = 'Concept-inspired procedural GLB mesh: volumetric body plus visual-only VFX organs, glossy black suit, cyan emissive traces, T-pose humanoid.'
    empty['authoring_poly_target'] = POLY_TARGET
    empty['goldstandard_detail_scale'] = round(DETAIL_SCALE, 4)
    empty['runtime_lod_policy'] = 'Cinematic GoldStandard authoring lane; AvatarPanel advertises cinematic_2m and can fall back to lower LODs if needed.'
    bpy.context.collection.objects.link(empty)

    for obj in list(bpy.context.scene.objects):
        if getattr(obj, 'type', '') == 'MESH' and not str(obj.name).startswith('vfx_'):
            add_authoring_smooth_modifier(obj)

    out_blend = CFG.get('blend_path')
    if out_blend:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        except Exception as exc:
            CFG['blend_save_error'] = str(exc)

    out_glb = CFG['glb_path']
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format='GLB',
        export_apply=True,
        export_yup=True,
        export_materials='EXPORT',
        export_animations=False,
        export_extras=True,
    )

    if CFG.get('render_preview'):
        try:
            bpy.context.scene.render.resolution_x = 3840 if AUTHORING_10M else 2400
            bpy.context.scene.render.resolution_y = 5120 if AUTHORING_10M else 3200
            bpy.context.scene.render.filepath = CFG.get('preview_path')
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            CFG['preview_error'] = str(exc)


build_avatar()
'''


def generate_blend_export_script(settings: Dict[str, Any]) -> str:
    settings_json = json.dumps(settings, indent=2)
    return f'''# Auto-generated by SM-AVATARBUILD.py. Existing .blend to GLB export.
import os
import json
import bpy
CFG = json.loads({settings_json!r})

try:
    backend = str(CFG.get('gpu_backend') or 'AUTO').upper()
    if backend and backend != 'AUTO':
        bpy.context.scene.render.engine = 'CYCLES'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = backend
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        bpy.context.scene.cycles.device = 'GPU' if backend != 'CPU' else 'CPU'
except Exception as exc:
    CFG['gpu_setup_error'] = str(exc)

os.makedirs(os.path.dirname(CFG['glb_path']), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=CFG['glb_path'],
    export_format='GLB',
    export_apply=True,
    export_yup=True,
    export_materials='EXPORT',
    export_animations=True,
)
'''


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_blender_python(blender_path: str, script_path: str, timeout_seconds: int, blend_path: str = "") -> subprocess.CompletedProcess:
    """Bounded local Blender subprocess. Authorization belongs to the calling build lane."""
    if not bool(getattr(_BLENDER_AUTHORIZATION_CTX, "authorized", False)):
        raise PermissionError("direct Blender subprocess execution requires an authorized builder operation")
    if not os.path.isfile(blender_path) or not os.path.isfile(script_path):
        raise FileNotFoundError("Blender executable/script missing")
    cmd = [blender_path, "--background", "--disable-autoexec"]
    if blend_path:
        if not os.path.isfile(blend_path):
            raise FileNotFoundError("source blend missing")
        cmd.append(blend_path)
    cmd.extend(["--python", script_path])
    timeout_seconds = max(30, min(_bounded_int_env("SARAH_AVATAR_BUILDER_MAX_TIMEOUT", 2700, 30, 7200), int(timeout_seconds or 2700)))
    output_max = _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_LOG_BYTES", 2 * 1024 * 1024, 65536, 16 * 1024 * 1024)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    logger.info("Starting governed local Blender build: %s", os.path.basename(blender_path))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False,
        creationflags=creationflags, start_new_session=(os.name != "nt"),
    )
    # A dedicated reader avoids deadlocks if the child floods the stdout pipe.
    lines = []
    used = 0
    q: queue.Queue = queue.Queue(maxsize=256)
    reader_drops = [0]
    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for raw in iter(lambda: proc.stdout.read(4096), b""):
                try:
                    q.put_nowait(raw)
                except queue.Full:
                    reader_drops[0] += 1
        except Exception:
            reader_drops[0] += 1
        finally:
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
    threading.Thread(target=_reader, daemon=True, name="SarahAvatarBlenderLog").start()
    started = time.monotonic()
    truncated = False
    while proc.poll() is None or not q.empty():
        try:
            chunk = q.get(timeout=0.15)
            remaining = output_max - used
            if remaining > 0:
                clipped = chunk[:remaining]
                used += len(clipped)
                lines.append(clipped.decode("utf-8", "replace"))
            if len(chunk) > remaining:
                truncated = True
        except queue.Empty:
            pass
        if proc.poll() is None and time.monotonic() - started > timeout_seconds:
            _terminate_process_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            return subprocess.CompletedProcess(cmd, 124, "".join(lines) + "\n[ERROR] Blender timeout.\n", "")
    if truncated or reader_drops[0]:
        lines.append("\n[INFO] Blender output clipped by bounded capture budget.\n")
    return subprocess.CompletedProcess(cmd, int(proc.returncode or 0), "".join(lines), "")


def _run_authorized_blender(blender_path: str, script_path: str, timeout_seconds: int, blend_path: str = "") -> subprocess.CompletedProcess:
    review = getattr(_BLENDER_AUTHORIZATION_CTX, "operator_review", None)
    if not isinstance(review, dict) or review.get("decision") != "ALLOW" or review.get("allow") is not True or review.get("domain_execution_authorized") is not True:
        raise PermissionError("missing approved OperatorCore build review")
    if bool(getattr(_BLENDER_AUTHORIZATION_CTX, "authorized", False)):
        raise PermissionError("nested Blender execution prohibited")
    _BLENDER_AUTHORIZATION_CTX.authorized = True
    try:
        return run_blender_python(blender_path, script_path, timeout_seconds, blend_path=blend_path)
    finally:
        _BLENDER_AUTHORIZATION_CTX.authorized = False
        _BLENDER_AUTHORIZATION_CTX.operator_review = None


def write_text(path: str, content: str) -> None:
    _write_atomic(path, content)


def build_3d_avatar(cfg: Avatar3DBuildConfig) -> Dict[str, Any]:
    """Build a new GLB in isolation, validate it, then promote preserving rollback."""
    outdir = os.path.abspath(cfg.outdir)
    name = _safe_asset_name(os.path.splitext(os.path.basename(cfg.avatar_name))[0])
    glb_path = _asset_path(outdir, name + ".glb")
    blend_path = _asset_path(outdir, name + ".blend") if cfg.save_blend else ""
    preview_path = _asset_path(outdir, name + "_preview.png") if cfg.render_preview else ""
    log_path = _asset_path(outdir, name + "_blender_build.log")
    manifest_path = _asset_path(outdir, name + "_manifest.json")
    script_final = _asset_path(outdir, name + "_blender_build.py")
    _check_blender_operator_optin()
    blender = find_blender_executable(cfg.blender_path)
    if not os.path.isfile(blender):
        raise FileNotFoundError("Blender executable not found")
    auth, _ = _authorize_blender_build("build_3d_avatar", glb_path, blender)
    try:
        ensure_dir(outdir)
        outdir = _validate_output_root(outdir)
        with tempfile.TemporaryDirectory(prefix=".smbuilder_3d_", dir=outdir) as staged_dir:
            staged_glb = os.path.join(staged_dir, name + ".glb")
            staged_blend = os.path.join(staged_dir, name + ".blend") if cfg.save_blend else ""
            staged_preview = os.path.join(staged_dir, name + "_preview.png") if cfg.render_preview else ""
            staged_script = os.path.join(staged_dir, name + "_blender_build.py")
            palette = extract_concept_palette(cfg.concept_image)
            settings = {
                "builder": "SM-AVATARBUILD.py", "mode": "build_3d_avatar",
                "concept_image": os.path.abspath(cfg.concept_image) if cfg.concept_image else "",
                "outdir": staged_dir, "glb_path": staged_glb, "blend_path": staged_blend,
                "preview_path": staged_preview, "render_preview": bool(cfg.render_preview),
                "gpu_backend": str(cfg.gpu_backend or "AUTO").upper(),
                "poly_target": max(90000, min(2000000, int(cfg.poly_target))),
                "vfx_enabled": bool(cfg.vfx_enabled),
                "vfx_quality": str(cfg.vfx_quality or "goldstandard_entity"),
                "vfx_intensity": _safe_finite(cfg.vfx_intensity, 0.92, 0.05, 1.5),
                "palette": _jsonable_palette(palette),
                "runtime_contract": "TRUE_3D_GOLD_MESH_RUNTIME",
                "runtime_geometry": "TRUE_3D_GLB_MESH", "simulation_before_execution": True,
            }
            write_text(staged_script, generate_blender_avatar_script(settings))
            result = _run_authorized_blender(blender, staged_script, cfg.timeout_seconds)
            write_text(log_path, (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""))
            if result.returncode != 0:
                raise RuntimeError("Blender process exited unsuccessfully")
            proof = _read_glb_header(staged_glb)
            if not proof.get("valid_glb_structure") or not proof.get("has_mesh"):
                raise RuntimeError("Blender did not produce a fresh, valid GLB mesh: " + str(proof.get("error") or "missing/invalid"))
            digest = _file_sha256(staged_glb)
            manifest = {
                "builder": "SM-AVATARBUILD", "version": MODULE_VERSION, "built_at": datetime.now().isoformat(),
                "local_only": True, "mode": "build_3d_avatar", "ok": True,
                "concept_image": settings["concept_image"], "concept_image_exists": bool(cfg.concept_image and os.path.isfile(cfg.concept_image)),
                "avatar_name": name, "output_dir": outdir, "glb": name + ".glb",
                "blend": os.path.basename(blend_path) if staged_blend and os.path.isfile(staged_blend) else None,
                "preview": os.path.basename(preview_path) if staged_preview and os.path.isfile(staged_preview) else None,
                "blender_path": blender, "gpu_backend_requested": settings["gpu_backend"],
                "poly_target": int(cfg.poly_target), "effective_detail_target": settings["poly_target"],
                "actual_polygon_count": None, "rig_status": "procedural_mesh_named_parts_armature_future",
                "vfx_enabled": bool(cfg.vfx_enabled), "vfx_organs_ready": bool(cfg.vfx_enabled),
                "vfx_quality": settings["vfx_quality"], "vfx_intensity": settings["vfx_intensity"],
                "palette": settings["palette"], "runtime_contract": "TRUE_3D_GOLD_MESH_RUNTIME",
                "runtime_geometry": "TRUE_3D_GLB_MESH", "execution_authority": False,
                "physical_actuation_allowed_here": False,
                "animation_contract": {"frontend_hooks": ["idle_breathing", "blink", "eye_follow", "mouth_lipsync", "wave", "walk_in_place"], "glb_exported_clips": False, "rig_status": "procedural_mesh_named_parts_armature_future"},
                "validation": {"level": "glb_v2_chunks_json_and_mesh", "mesh_count": proof["mesh_count"], "node_count": proof["node_count"], "glb_size_bytes": proof["size_bytes"], "sha256": digest, "browser_rendering_verified": False},
                "blender_returncode": result.returncode, "blender_log": os.path.basename(log_path),
                "blender_script": os.path.basename(script_final) if cfg.keep_blender_script else None,
                "notes": ["Procedural 3D proxy, not image-to-mesh reconstruction", "Physical actuation is never authorized here"],
            }
            # Validate everything before replacing the previously active GLB.
            backup = _promote_verified_asset(staged_glb, glb_path)
            promoted_aux = []
            try:
                if staged_blend and os.path.isfile(staged_blend) and os.path.getsize(staged_blend) > 0:
                    promoted_aux.append((blend_path, _promote_optional_asset(staged_blend, blend_path)))
                if staged_preview and os.path.isfile(staged_preview) and os.path.getsize(staged_preview) > 0:
                    promoted_aux.append((preview_path, _promote_optional_asset(staged_preview, preview_path)))
                if cfg.keep_blender_script:
                    promoted_aux.append((script_final, _promote_optional_asset(staged_script, script_final)))
                write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
            except BaseException:
                for prior_path, prior_backup in reversed(promoted_aux):
                    _restore_active_artifact(prior_path, prior_backup)
                _restore_active_artifact(glb_path, backup)
                raise
        _record_blender_outcome(auth, True, artifact=glb_path)
        _build_result(True)
        logger.info("3D avatar GLB verified and promoted: %s", glb_path)
        return manifest
    except Exception as exc:
        _record_blender_outcome(auth, False, error=str(exc))
        _build_result(False, str(exc))
        raise


def export_blend_to_glb(cfg: BlendExportConfig) -> Dict[str, Any]:
    """Export an approved local .blend into a validated, staged GLB."""
    if not os.path.isfile(cfg.blend_path):
        raise FileNotFoundError("Blend file not found: " + str(cfg.blend_path))
    outdir = os.path.abspath(cfg.outdir)
    name = _safe_asset_name(os.path.splitext(os.path.basename(cfg.avatar_name))[0])
    glb_path = _asset_path(outdir, name + ".glb")
    log_path = _asset_path(outdir, name + "_blend_export.log")
    script_final = _asset_path(outdir, name + "_blend_export.py")
    manifest_path = _asset_path(outdir, name + "_manifest.json")
    _check_blender_operator_optin()
    blender = find_blender_executable(cfg.blender_path)
    auth, _ = _authorize_blender_build("export_blend_to_glb", glb_path, blender)
    try:
        ensure_dir(outdir)
        outdir = _validate_output_root(outdir)
        with tempfile.TemporaryDirectory(prefix=".smbuilder_export_", dir=outdir) as staged_dir:
            staged_glb = os.path.join(staged_dir, name + ".glb")
            staged_script = os.path.join(staged_dir, name + "_blend_export.py")
            settings = {"builder": "SM-AVATARBUILD.py", "mode": "export_blend_to_glb", "blend_path": os.path.abspath(cfg.blend_path), "glb_path": staged_glb, "gpu_backend": str(cfg.gpu_backend or "AUTO").upper()}
            write_text(staged_script, generate_blend_export_script(settings))
            result = _run_authorized_blender(blender, staged_script, cfg.timeout_seconds, blend_path=os.path.abspath(cfg.blend_path))
            write_text(log_path, (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""))
            if result.returncode != 0:
                raise RuntimeError("Blender export exited unsuccessfully")
            proof = _read_glb_header(staged_glb)
            if not proof.get("valid_glb_structure") or not proof.get("has_mesh"):
                raise RuntimeError("Blender export did not create a valid new GLB mesh")
            manifest = {
                "builder": "SM-AVATARBUILD", "version": MODULE_VERSION, "built_at": datetime.now().isoformat(),
                "local_only": True, "mode": "export_blend_to_glb", "ok": True,
                "source_blend": os.path.abspath(cfg.blend_path), "output_dir": outdir,
                "glb": name + ".glb", "blender_path": blender, "gpu_backend_requested": settings["gpu_backend"],
                "blender_returncode": result.returncode, "blender_log": os.path.basename(log_path),
                "blender_script": os.path.basename(script_final), "execution_authority": False,
                "validation": {"level": "glb_v2_chunks_json_and_mesh", "mesh_count": proof["mesh_count"], "sha256": _file_sha256(staged_glb), "browser_rendering_verified": False},
            }
            backup = _promote_verified_asset(staged_glb, glb_path)
            promoted_aux = []
            try:
                promoted_aux.append((script_final, _promote_optional_asset(staged_script, script_final)))
                write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
            except BaseException:
                for prior_path, prior_backup in reversed(promoted_aux):
                    _restore_active_artifact(prior_path, prior_backup)
                _restore_active_artifact(glb_path, backup)
                raise
        _record_blender_outcome(auth, True, artifact=glb_path)
        _build_result(True)
        return manifest
    except Exception as exc:
        _record_blender_outcome(auth, False, error=str(exc))
        _build_result(False, str(exc))
        raise


def _restore_active_artifact(path: str, backup: str) -> None:
    if backup:
        if os.path.isfile(path):
            os.unlink(path)
        os.replace(backup, path)
    elif os.path.isfile(path):
        os.unlink(path)


def _promote_optional_asset(src: str, dst: str) -> str:
    """Promote staged auxiliary output without losing an existing target."""
    parent = os.path.dirname(os.path.abspath(dst))
    _canonical_child(dst, parent)
    backup = ""
    if os.path.lexists(dst):
        backups = list(Path(parent).glob(Path(dst).name + ".rollback.*"))
        if len(backups) >= _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_ROLLBACKS", 3, 1, 20):
            raise RuntimeError("auxiliary rollback budget exhausted")
        backup = dst + ".rollback." + uuid.uuid4().hex[:12]
        os.replace(dst, backup)
    try:
        os.replace(src, dst)
    except BaseException:
        if backup:
            os.replace(backup, dst)
        raise
    return backup


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
class AvatarBuildEngine:
    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg

    def run(self) -> Dict[str, Any]:
        _require_pillow()
        cfg = self.cfg
        ensure_dir(cfg.outdir)
        root = _validate_output_root(cfg.outdir)
        for name in (cfg.suit_name, cfg.sheet_name, cfg.manifest_name):
            _safe_asset_name(name)
        try:
            sheet = open_rgba(cfg.sheet_path)
            suit_ref = open_rgba(cfg.suit_ref_path)
            suit_ref = extract_column(suit_ref, cfg.suit_column)
            frames = crop_grid(sheet, cfg.rows, cfg.cols)
            if not frames:
                raise RuntimeError("No frames extracted")
            frame_w, frame_h = frames[0].size
            if frame_w * frame_h * len(frames) > _max_pixels():
                raise ValueError("combined sprite frame pixel budget exceeded")
            suit_base = isolate_suit_base(suit_ref, (frame_w, frame_h), cfg)
            with tempfile.TemporaryDirectory(prefix=".smbuilder_2d_", dir=root) as stage:
                save_png(suit_base, os.path.join(stage, cfg.suit_name))
                built_frames = []
                records = []
                for idx, frame in enumerate(frames, start=1):
                    overlay, _mask = extract_face_hair_region(frame, cfg)
                    if cfg.export_face_only:
                        save_png(overlay, os.path.join(stage, f"face_only_{idx:02d}.png"))
                    composed = composite_frame(suit_base, overlay)
                    name = self._default_frame_name(idx)
                    filename = _safe_asset_name(f"{name}.png")
                    save_png(composed, os.path.join(stage, filename))
                    built_frames.append(composed)
                    records.append(FrameRecord(index=idx, name=name, filename=filename, width=frame_w, height=frame_h))
                sheet_path = None
                if cfg.export_sheet:
                    compiled = make_sprite_sheet(built_frames, cfg.rows, cfg.cols)
                    sheet_path = _asset_path(root, cfg.sheet_name)
                    save_png(compiled, os.path.join(stage, cfg.sheet_name))
                manifest = build_manifest(cfg, _asset_path(root, cfg.suit_name), records, sheet_path)
                manifest["version"] = MODULE_VERSION
                manifest["deterministic_pixel_outputs"] = True
                manifest["execution_authority"] = False
                # Output bytes deterministic; built_at deliberately reports wall clock.
                manifest["timestamp_is_non_deterministic"] = True
                manifest["sha256"] = {
                    n: _file_sha256(os.path.join(stage, n))
                    for n in os.listdir(stage) if n.endswith(".png")
                }
                write_text(os.path.join(stage, cfg.manifest_name), json.dumps(manifest, indent=2, ensure_ascii=False))
                # Preflight all destination paths and rollback budget before mutation.
                artifacts = list(os.listdir(stage))
                previous = []
                for n in artifacts:
                    dst = _asset_path(root, n)
                    if os.path.lexists(dst):
                        previous.append(dst)
                        backups = list(Path(root).glob(Path(dst).name + ".rollback.*"))
                        if len(backups) >= _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_ROLLBACKS", 3, 1, 20):
                            raise RuntimeError("2D artifact rollback budget exhausted")
                promoted = []
                try:
                    for n in artifacts:
                        dst = _asset_path(root, n)
                        backup = ""
                        if os.path.lexists(dst):
                            backup = dst + ".rollback." + uuid.uuid4().hex[:12]
                            os.replace(dst, backup)
                        try:
                            os.replace(os.path.join(stage, n), dst)
                        except BaseException:
                            if backup:
                                os.replace(backup, dst)
                            raise
                        promoted.append((dst, backup))
                except BaseException:
                    for dst, backup in reversed(promoted):
                        _restore_active_artifact(dst, backup)
                    raise
            _build_result(True)
            logger.info("Avatar asset build complete: %s", root)
            return manifest
        except Exception as exc:
            _build_result(False, str(exc))
            raise

    @staticmethod
    def _default_frame_name(idx: int) -> str:
        names = {
            1: "avatar_neutral",
            2: "avatar_blink",
            3: "avatar_smile",
            4: "avatar_talk_1",
            5: "avatar_talk_2",
            6: "avatar_thinking",
            7: "avatar_grin",
            8: "avatar_talk_3",
        }
        return names.get(idx, f"avatar_frame_{idx:02d}")


# ---------------------------------------------------------------------------
# Realtime runtime asset contract / validation
# ---------------------------------------------------------------------------

def _default_runtime_avatar_dir() -> str:
    try:
        if config is not None:
            configured = str(getattr(config, "AVATAR_3D_DIR", "") or "")
            if configured:
                return os.path.abspath(configured)
            base = str(getattr(config, "BASE_DIR", "") or "")
            if base:
                return os.path.abspath(os.path.join(base, "resources", "avatars", "3D"))
    except Exception:
        pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "avatars", "3D"))


def _read_glb_header(path: str) -> Dict[str, Any]:
    """Bounded binary GLB v2 structure inspection; not a browser rendering test."""
    result: Dict[str, Any] = {
        "path": os.path.abspath(path) if path else "", "exists": bool(path and os.path.isfile(path)),
        "size_bytes": 0, "valid_glb_header": False, "valid_glb_structure": False,
        "has_mesh": False, "mesh_count": 0, "node_count": 0,
        "version": None, "declared_length": None,
    }
    if not result["exists"]:
        return result
    try:
        size = os.path.getsize(path)
        result["size_bytes"] = size
        if size > _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_GLB_BYTES", 512 * 1024 * 1024, 1024 * 1024, 2147483648):
            raise ValueError("GLB exceeds configured size budget")
        if size < 20:
            return result
        with open(path, "rb") as fh:
            header = fh.read(12)
            magic, version, declared_length = struct.unpack("<4sII", header)
            result["version"] = int(version)
            result["declared_length"] = int(declared_length)
            if magic != b"glTF" or version != 2 or declared_length != size or size % 4:
                return result
            result["valid_glb_header"] = True
            offset = 12
            json_document = None
            bin_chunk_size = 0
            chunk_index = 0
            while offset < size:
                if size - offset < 8:
                    raise ValueError("truncated GLB chunk header")
                chunk_length, chunk_type = struct.unpack("<I4s", fh.read(8))
                offset += 8
                if chunk_length % 4 or chunk_length > size - offset:
                    raise ValueError("invalid GLB chunk length")
                if chunk_index == 0:
                    if chunk_type != b"JSON":
                        raise ValueError("first GLB chunk must be JSON")
                    if chunk_length > _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_GLB_JSON", 16 * 1024 * 1024, 1024, 64 * 1024 * 1024):
                        raise ValueError("GLB JSON budget exceeded")
                    raw_json = fh.read(chunk_length)
                    json_document = json.loads(raw_json.decode("utf-8").rstrip(" \t\r\n\x00"))
                else:
                    if chunk_index != 1 or chunk_type != b"BIN\x00":
                        raise ValueError("unexpected GLB chunk type/order")
                    bin_chunk_size = chunk_length
                    fh.seek(chunk_length, os.SEEK_CUR)
                offset += chunk_length
                chunk_index += 1
            if offset != size or not isinstance(json_document, dict):
                raise ValueError("incomplete GLB JSON document")
            asset = json_document.get("asset")
            if not isinstance(asset, dict) or str(asset.get("version")) != "2.0":
                raise ValueError("invalid glTF 2.0 asset descriptor")
            meshes = json_document.get("meshes")
            nodes = json_document.get("nodes")
            result["mesh_count"] = len(meshes) if isinstance(meshes, list) else 0
            result["node_count"] = len(nodes) if isinstance(nodes, list) else 0
            accessors = json_document.get("accessors")
            views = json_document.get("bufferViews")
            buffers = json_document.get("buffers")
            scene_nodes = json_document.get("scenes")
            scene_valid = bool(isinstance(scene_nodes, list) and scene_nodes and isinstance(nodes, list) and nodes and any(
                isinstance(scene, dict) and isinstance(scene.get("nodes"), list) and any(
                    isinstance(node_id, int) and 0 <= node_id < len(nodes) for node_id in scene.get("nodes", [])
                ) for scene in scene_nodes
            ))
            buffer_valid = bool(isinstance(buffers, list) and len(buffers) == 1 and isinstance(buffers[0], dict)
                and isinstance(buffers[0].get("byteLength"), int) and 0 < buffers[0]["byteLength"] <= bin_chunk_size
                and isinstance(views, list) and views and isinstance(accessors, list) and accessors)
            def valid_accessor(accessor_id):
                if not buffer_valid or not isinstance(accessor_id, int) or not 0 <= accessor_id < len(accessors):
                    return False
                accessor = accessors[accessor_id]
                if not isinstance(accessor, dict) or not isinstance(accessor.get("bufferView"), int):
                    return False
                view_id = accessor["bufferView"]
                if not 0 <= view_id < len(views):
                    return False
                view = views[view_id]
                return bool(isinstance(view, dict) and view.get("buffer") == 0
                    and isinstance(view.get("byteLength"), int) and view["byteLength"] > 0
                    and isinstance(view.get("byteOffset", 0), int) and view.get("byteOffset", 0) >= 0
                    and view.get("byteOffset", 0) + view["byteLength"] <= buffers[0]["byteLength"]
                    and isinstance(accessor.get("count"), int) and accessor["count"] > 0)
            primitives_present = bool(isinstance(meshes, list) and any(
                isinstance(mesh, dict) and isinstance(mesh.get("primitives"), list) and any(
                    isinstance(prim, dict) and isinstance(prim.get("attributes"), dict)
                    and valid_accessor(prim["attributes"].get("POSITION"))
                    for prim in mesh.get("primitives", [])
                ) for mesh in meshes
            ))
            result["has_mesh"] = bool(result["mesh_count"] and primitives_present and scene_valid)
            result["valid_glb_structure"] = True
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
        result["error"] = str(exc)[:200]
    return result


def inspect_realtime_avatar_assets(asset_dir: str = "", avatar_name: str = "SarahMemoryAvatar_RigBootstrap") -> Dict[str, Any]:
    """Inspect the AvatarPanel runtime asset contract without launching Blender.

    A valid GLB proves that a loadable binary asset exists; it does not prove the
    visual quality, animation behavior, or WebGL runtime on the user's machine.
    """
    root = os.path.abspath(asset_dir or _default_runtime_avatar_dir())
    stem = os.path.splitext(os.path.basename(avatar_name or "SarahMemoryAvatar_RigBootstrap"))[0]
    glb_candidates = [
        os.path.join(root, f"{stem}.glb"),
        os.path.join(root, "SarahMemoryAvatar_RigBootstrap.glb"),
        os.path.join(root, "sarahmemory_3d_avatar.glb"),
    ]
    manifest_candidates = [
        os.path.join(root, f"{stem}.json"),
        os.path.join(root, f"{stem}_manifest.json"),
        os.path.join(root, "avatar_3d_manifest.json"),
    ]
    blend_candidates = [
        os.path.join(root, f"{stem}.blend"),
        os.path.join(root, "SarahMemoryAvatar_RigBootstrap.blend"),
    ]

    def approved(path: str) -> bool:
        try:
            _canonical_child(path, root)
            return os.path.isfile(path)
        except (OSError, ValueError):
            return False
    glb_path = next((path for path in glb_candidates if approved(path)), glb_candidates[0])
    manifest_path = next((path for path in manifest_candidates if approved(path)), "")
    blend_path = next((path for path in blend_candidates if approved(path)), "")
    glb = _read_glb_header(glb_path) if approved(glb_path) else {
        "path": os.path.abspath(glb_path), "exists": False, "size_bytes": 0,
        "valid_glb_header": False, "valid_glb_structure": False, "has_mesh": False,
        "mesh_count": 0, "version": None, "declared_length": None,
        "error": "asset_missing_or_outside_approved_root",
    }
    manifest: Dict[str, Any] = {}
    manifest_error = ""
    if manifest_path:
        try:
            if os.path.getsize(manifest_path) > _bounded_int_env("SARAH_AVATAR_BUILDER_MAX_MANIFEST_BYTES", 1048576, 1024, 8388608):
                raise ValueError("manifest exceeds size limit")
            with open(manifest_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            manifest = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            manifest_error = str(exc)

    shape_keys = (manifest.get("shape_keys") or [])[:256] if isinstance(manifest.get("shape_keys"), list) else []
    bones = (manifest.get("bones") or [])[:256] if isinstance(manifest.get("bones"), list) else []
    panel_contract = manifest.get("avatar_panel_contract") if isinstance(manifest.get("avatar_panel_contract"), dict) else {}
    animation_contract = manifest.get("animation_contract") if isinstance(manifest.get("animation_contract"), dict) else {}
    panel_contract = {str(k)[:80]: str(v)[:200] if isinstance(v, str) else v for k, v in list(panel_contract.items())[:64]}
    animation_contract = {str(k)[:80]: str(v)[:200] if isinstance(v, str) else v for k, v in list(animation_contract.items())[:64]}
    actions = (manifest.get("actions") or [])[:256] if isinstance(manifest.get("actions"), list) else []
    runtime_ready = bool(glb.get("valid_glb_structure") and glb.get("has_mesh") and approved(glb_path))

    return {
        "ok": True,
        "schema": "SARAHMEMORY_REALTIME_AVATAR_STATUS_V1",
        "asset_dir": root,
        "asset_dir_exists": os.path.isdir(root),
        "glb": glb,
        "blend": {
            "path": os.path.abspath(blend_path) if blend_path else "",
            "exists": bool(blend_path),
            "size_bytes": int(os.path.getsize(blend_path)) if blend_path else 0,
        },
        "manifest_path": os.path.abspath(manifest_path) if manifest_path else "",
        "manifest_loaded": bool(manifest),
        "manifest_error": manifest_error,
        "quality": manifest.get("quality") or manifest.get("vfx_quality") or "unknown",
        "bone_count": int(manifest.get("bone_count") or len(bones)),
        "shape_keys": shape_keys,
        "actions": actions,
        "animation_contract": animation_contract,
        "avatar_panel_contract": panel_contract,
        "runtime_ready": runtime_ready,
        "validation_level": "glb_container_json_and_mesh_only",
        "browser_rendering_verified": False,
        "runtime_mode": "lazy_glb_webgl",
        "realtime_expression_hooks": ["blink", "mouth_lipsync", "eye_follow", "idle_breathing", "expression", "speaking", "listening"],
        "load_policy": "load_only_when_avatar_3d_mode_selected",
        "fallback_policy": "retain_2d_avatar_if_glb_or_webgl_fails",
        "execution_authority": False,
        "hardware_control": False,
        "validation_boundary": "file/header/manifest inspection only; Blender and browser rendering were not executed",
    }


def get_avatar_builder_capabilities(blender_path: str = "") -> Dict[str, Any]:
    """Return a truthful, read-only builder/runtime capability report."""
    resolved_blender = ""
    blender_error = ""
    try:
        resolved_blender = find_blender_executable(blender_path)
    except Exception as exc:
        blender_error = str(exc)
    return {
        "ok": True,
        "module": "SarahMemoryAvatarBuilder",
        "version": MODULE_VERSION,
        "pillow_available": _PIL_AVAILABLE,
        "blender_available": bool(resolved_blender),
        "blender_path": resolved_blender,
        "blender_error": blender_error,
        "build_modes": ["2d_sprite", "procedural_3d_glb", "blend_to_glb"],
        "runtime_format": "GLB/GLTF 2.0",
        "runtime_inspection": inspect_realtime_avatar_assets(),
        "import_side_effect_free": True,
        "build_status": dict(_BUILD_STATUS),
        "execution_authority": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SarahMemory avatar assets. Default mode preserves the legacy 2D "
            "sprite/suit builder. Use --build-3d-avatar for local Blender GLB output."
        )
    )

    # 2D legacy lane. Kept optional at parser level so 3D mode does not require these.
    parser.add_argument("--sheet", help="Path to 8-frame sprite sheet image for the legacy 2D build lane")
    parser.add_argument("--suit-ref", help="Path to correct suit reference image for the legacy 2D build lane")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--rows", type=int, default=2, help="Sprite sheet rows")
    parser.add_argument("--cols", type=int, default=4, help="Sprite sheet cols")
    parser.add_argument("--suit-column", type=int, default=1, help="1-based column to crop from multi-column suit reference image")
    parser.add_argument("--neck-y-ratio", type=float, default=0.335, help="Approx collar / neck transition ratio")
    parser.add_argument("--blend-top-ratio", type=float, default=0.295, help="Face overlay full-strength start ratio")
    parser.add_argument("--blend-bottom-ratio", type=float, default=0.435, help="Face overlay fade-out end ratio")
    parser.add_argument("--feather-radius", type=int, default=14, help="Feather radius for upper/lower blend")
    parser.add_argument("--no-face-only", action="store_true", help="Disable face-only exports")
    parser.add_argument("--no-sheet", action="store_true", help="Disable sprite sheet export")

    # 3D GLB lane.
    parser.add_argument("--build-3d-avatar", action="store_true", help="Build a concept-inspired SarahMemory 3D avatar GLB using Blender")
    parser.add_argument("--build-3d-placeholder", action="store_true", help="Alias for --build-3d-avatar")
    parser.add_argument("--concept-image", default="", help="Path to the SarahMemory 3D concept/reference image")
    parser.add_argument("--avatar-name", default="sarahmemory_3d_avatar", help="Output avatar base filename without extension")
    parser.add_argument("--blender", default="", help="Path to blender.exe/blender binary; auto-detected when omitted")
    parser.add_argument("--gpu-backend", default="AUTO", choices=["AUTO", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "CPU"], help="Blender Cycles GPU backend preference")
    parser.add_argument("--poly-target", type=int, default=12000000, help="Authoring polygon target metadata and procedural detail scale. Default 12000000 is the GoldStandard embodied entity AvatarPanel validation lane.")
    parser.add_argument("--save-blend", action="store_true", help="Save the generated .blend source beside the .glb")
    parser.add_argument("--no-save-blend", action="store_true", help="Do not save the generated .blend source")
    parser.add_argument("--render-preview", action="store_true", help="Render a preview PNG after building the GLB")
    parser.add_argument("--delete-blender-script", action="store_true", help="Delete generated Blender Python script after successful run")
    parser.add_argument("--timeout-seconds", type=int, default=2700, help="Blender subprocess timeout")
    parser.add_argument("--vfx-off", action="store_true", help="Disable embedded AvatarPanel VFX organ marker geometry in the generated GLB")
    parser.add_argument("--vfx-quality", default="goldstandard_entity", choices=["preview", "balanced", "high", "ultra", "cinematic_2m", "goldstandard_entity"], help="VFX organ metadata quality hint")
    parser.add_argument("--vfx-intensity", type=float, default=0.92, help="VFX organ intensity metadata, clamped by runtime")

    # Existing .blend export lane.
    parser.add_argument("--export-blend-to-glb", default="", help="Path to an existing .blend file to export as a runtime GLB")

    return parser.parse_args()


def _resolve_2d_outdir(args: argparse.Namespace) -> str:
    return os.path.abspath(args.outdir) if args.outdir else os.path.abspath(default_outdir())


def _resolve_3d_outdir(args: argparse.Namespace) -> str:
    return os.path.abspath(args.outdir) if args.outdir else os.path.abspath(default_3d_outdir())


def _run_2d_lane(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.sheet or not args.suit_ref:
        raise ValueError(
            "2D avatar build requires --sheet and --suit-ref. "
            "For GLB output, use --build-3d-avatar --concept-image <image>."
        )
    cfg = BuildConfig(
        sheet_path=os.path.abspath(args.sheet),
        suit_ref_path=os.path.abspath(args.suit_ref),
        outdir=_resolve_2d_outdir(args),
        rows=args.rows,
        cols=args.cols,
        suit_column=args.suit_column,
        neck_y_ratio=args.neck_y_ratio,
        blend_top_ratio=args.blend_top_ratio,
        blend_bottom_ratio=args.blend_bottom_ratio,
        feather_radius=args.feather_radius,
        export_face_only=not args.no_face_only,
        export_sheet=not args.no_sheet,
    )
    ensure_dir(cfg.outdir)
    logger.info("Output directory: %s", cfg.outdir)
    return AvatarBuildEngine(cfg).run()


def _run_3d_lane(args: argparse.Namespace) -> Dict[str, Any]:
    save_blend = True
    if args.no_save_blend:
        save_blend = False
    elif args.save_blend:
        save_blend = True

    cfg = Avatar3DBuildConfig(
        concept_image=os.path.abspath(args.concept_image) if args.concept_image else "",
        outdir=_resolve_3d_outdir(args),
        avatar_name=args.avatar_name,
        blender_path=args.blender,
        gpu_backend=args.gpu_backend,
        save_blend=save_blend,
        render_preview=bool(args.render_preview),
        keep_blender_script=not bool(args.delete_blender_script),
        poly_target=args.poly_target,
        timeout_seconds=args.timeout_seconds,
        vfx_enabled=not bool(args.vfx_off),
        vfx_quality=args.vfx_quality,
        vfx_intensity=args.vfx_intensity,
    )
    ensure_dir(cfg.outdir)
    logger.info("3D output directory: %s", cfg.outdir)
    return build_3d_avatar(cfg)


def _run_blend_export_lane(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = BlendExportConfig(
        blend_path=os.path.abspath(args.export_blend_to_glb),
        outdir=_resolve_3d_outdir(args),
        avatar_name=args.avatar_name,
        blender_path=args.blender,
        gpu_backend=args.gpu_backend,
        timeout_seconds=args.timeout_seconds,
    )
    ensure_dir(cfg.outdir)
    logger.info("3D output directory: %s", cfg.outdir)
    return export_blend_to_glb(cfg)


def main() -> int:
    try:
        args = parse_args()
        if args.export_blend_to_glb:
            manifest = _run_blend_export_lane(args)
            logger.info("Exported GLB: %s", manifest.get("glb"))
            return 0
        if args.build_3d_avatar or args.build_3d_placeholder:
            manifest = _run_3d_lane(args)
            logger.info("Built 3D avatar GLB: %s", manifest.get("glb"))
            return 0

        manifest = _run_2d_lane(args)
        logger.info("Built %s frames", manifest.get("frame_count"))
        return 0
    except Exception as e:
        logger.error("Avatar build failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAvatarBuilder',
    "version": "v9.1.0a-sml-0.2",
    "category": 'Presentation',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['avatar', 'execution', 'learning', 'presentation'],
    "supported_missions": ['Conversation', 'Learning', 'Patch', 'Repair'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Learning', 'Read'],
    "priority": 45,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAvatarBuilder.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Report known capabilities without fabricating measured operational metrics."""
    with _BUILD_STATUS_LOCK:
        status = dict(_BUILD_STATUS)
    return {
        "status": "Available" if _PIL_AVAILABLE else "Degraded",
        "availability": 1.0 if _PIL_AVAILABLE else 0.5,
        "integrity": None,
        "performance": None,
        "reliability": None,
        "confidence": 0.65,
        "latency_ms": None,
        "stability": None,
        "compatibility": None,
        "notes": ["Builds require explicit OperatorCore authorization; GLB inspection operates offline", "Build status: " + status["last_result"]],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAvatarBuilder',
        "sml_adapter": True,
        "metadata": copy.deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAvatarBuilder', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

