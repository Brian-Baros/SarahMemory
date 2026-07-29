"""--==The SarahMemory Project==--
File: SarahMemoryBlenderAvatarBootstrap.py
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

=========================================================
Creates a fully 3D model representation for interactive visual use.
Blender 3D AVATAR Creator - for the AvatarPanel as a fully interactive 3D Avatar 
=========================================================

Purpose:
- Bootstrap a SarahMemory 3D avatar staging scene inside Blender from 2D reference images.
- Build a materially improved SarahMemory avatar bootstrap with:
  body shell, head, eyes, hair, armature, finger bones, facial shape keys, controls,
  materials, lights, camera, and export outputs.
- Produce a governed, repeatable, exportable rigged starter asset so AvatarPanel /
  WebUI integration can proceed without hand-building the Blender scene.

Usage examples:
    python SarahMemoryBlenderAvatarBootstrap.py
    python SarahMemoryBlenderAvatarBootstrap.py --front "C:\\SarahMemory\\smfbwalk2.png" --side "C:\\SarahMemory\\smfbturnwalk.png" --back "C:\\SarahMemory\\smfbback.png"
    python SarahMemoryBlenderAvatarBootstrap.py --blueprint "C:\\SarahMemory\\resources\\avatar\\3D\\HDR-SMAVATAR-BP.png"
    python SarahMemoryBlenderAvatarBootstrap.py --blender "C:\\Blender51\\blender.exe" --outdir "C:\\SarahMemory\\resources\\avatar\\3D"

Notes:
- This script still does NOT generate a final production sculpt from 2D art alone.
- What it does do is build a substantially better procedural bootstrap than the
  cylinder/graph proxy so a non-animator can move forward.
- v9.0.1 adds a high-end-but-lightweight Avatar Organ bootstrap path:
  visible facial details, lips/brows/lashes, layered hair strands, raised neon
  circuit geometry, cyber stage lighting, runtime quality tiers, and explicit
  GLB readiness metadata for the V9 Avatar Panel micro game-engine viewport.
- v9.0.3 enforces a clean runtime export collection and modular appearance-slot
  doctrine so reference boards, proxy helpers, guide meshes, and construction
  layers never export as a second hovering avatar.
- v9.0.8 raises the Avatar Panel 3D target toward a 780p-class preview: higher
  mesh density, richer hair/detail counts, stronger GLB-safe shader metadata,
  brighter runtime material response, and a strict under-10-second load budget.
- v9.0.6 surface-binds outfit/neon/face detail geometry onto the generated
  humanoid mesh so suit lines no longer float in front of the Avatar Panel body.
- v9.0.11 adds governed ultra/cinematic authoring quality tiers for GoldStandard
  closer-form skinning, higher surface density, richer hair/face detail, and an
  explicit 2M-polygon authoring lane while preserving runtime warnings and user
  authority over heavy builds.
- v9.0.15 adds the GoldStandard Embodied Entity lane: stronger face sculpt
  refinement, high-density hair silhouette ribbons/strands, suit/body proportion
  refinement, exported action clips, and Avatar-Eye anchors while preserving
  observe-only governance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import threading
import queue
from typing import Dict, Optional, Tuple


MODULE_NAME = "SarahMemoryBlenderAvatarBootstrap"
MODULE_VERSION = "9.0.15-goldstandard-embodied-entity"


def _normalize_base_dir(candidate: str) -> str:
    """
    Resolve SarahMemory BASE_DIR from either:
    - SarahMemoryGlobals.BASE_DIR when available, or
    - this script location when installed under BASE_DIR/core.

    Project contract:
        BASE_DIR/core/SarahMemoryBlenderAvatarBootstrap.py
        BASE_DIR/resources/avatars/3D/<exported avatar files>

    The user-facing "../" notation refers to BASE_DIR, not to the process CWD.
    """
    if not candidate:
        candidate = os.getcwd()
    candidate = os.path.abspath(candidate)
    if os.path.basename(candidate).lower() == "core":
        return os.path.abspath(os.path.join(candidate, os.pardir))
    return candidate


def _discover_base_dir() -> str:
    """
    Prefer SarahMemoryGlobals.BASE_DIR. If unavailable, infer BASE_DIR from __file__.
    This avoids accidentally treating BASE_DIR/core as the project root when the
    script is launched from the core folder.
    """
    try:
        import SarahMemoryGlobals as config  # type: ignore
        return _normalize_base_dir(str(getattr(config, "BASE_DIR", "")))
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir).lower() == "core":
            return os.path.abspath(os.path.join(script_dir, os.pardir))
        return _normalize_base_dir(os.getcwd())


def _try_import_globals() -> Dict[str, str]:
    base_dir = _discover_base_dir()
    data_dir = os.path.join(base_dir, "data")
    resources_dir = os.path.join(base_dir, "resources")
    avatar_dir = os.path.join(resources_dir, "avatars")
    avatar_3d_dir = os.path.join(avatar_dir, "3D")
    sandbox_dir = os.path.join(base_dir, "sandbox")

    try:
        import SarahMemoryGlobals as config  # type: ignore
        data_dir = str(getattr(config, "DATA_DIR", data_dir))
        resources_dir = str(getattr(config, "RESOURCES_DIR", resources_dir))
        # Explicit v9 Avatar Organ output contract. The user-level ../ path means
        # BASE_DIR, and the canonical Avatar Organ folder is plural:
        # BASE_DIR/resources/avatars/3D.  If an old global still points to the
        # singular resources/avatar/3D lane, ignore it for new exports.
        configured_avatar_3d = str(getattr(config, "AVATAR_3D_DIR", avatar_3d_dir))
        configured_norm = configured_avatar_3d.replace("\\", "/").lower()
        if "resources/avatar/3d" not in configured_norm or "resources/avatars/3d" in configured_norm:
            avatar_3d_dir = configured_avatar_3d
        sandbox_dir = str(getattr(config, "SANDBOX_DIR", sandbox_dir))
    except Exception:
        pass

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "resources_dir": resources_dir,
        "avatar_dir": avatar_dir,
        "avatar_3d_dir": avatar_3d_dir,
        "sandbox_dir": sandbox_dir,
    }


def _default_blender_path() -> str:
    candidates = [
        r"C:\Blender51\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender-launcher.exe",
        shutil.which("blender") or "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return r"C:\Blender51\blender.exe"


def _resolve_blender_executable(blender_path: str) -> Tuple[str, str]:
    """
    Resolve a real Blender binary.

    Important Windows behavior:
    - blender-launcher.exe may spawn another process and keep this bootstrap waiting.
    - background CLI builds must use blender.exe directly whenever possible.

    Returns:
        (resolved_path, warning_message)
    """
    path = os.path.abspath(os.path.expanduser(blender_path or _default_blender_path()))
    base = os.path.basename(path).lower()
    if base == "blender-launcher.exe":
        same_dir = os.path.join(os.path.dirname(path), "blender.exe")
        if os.path.exists(same_dir):
            return os.path.abspath(same_dir), (
                "blender-launcher.exe was replaced with sibling blender.exe for background CLI execution."
            )
        parent = os.path.dirname(os.path.dirname(path))
        for root, _dirs, files in os.walk(parent):
            if "blender.exe" in [f.lower() for f in files]:
                candidate = os.path.join(root, "blender.exe")
                if os.path.exists(candidate):
                    return os.path.abspath(candidate), (
                        "blender-launcher.exe was replaced with discovered blender.exe for background CLI execution."
                    )
        return path, (
            "WARNING: blender-launcher.exe was supplied, but blender.exe was not found. "
            "This may hang. Install or pass the real blender.exe path."
        )
    return path, ""


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate Blender and any child process without leaving a background build alive."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_streaming_blender(cmd, timeout_seconds: int, log_path: str) -> int:
    """
    Run Blender with live console output and a hard timeout.

    The old implementation used subprocess.run(capture_output=True), which made a
    long Blender job look frozen for hours and provided no timeout.  This version
    streams Blender output to the terminal and to a build log.
    """
    timeout_seconds = int(timeout_seconds or 0)
    started = time.monotonic()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write("SarahMemory Blender Avatar Bootstrap Build Log\n")
        log_fh.write("Command: " + " ".join(cmd) + "\n")
        log_fh.write("TimeoutSeconds: " + str(timeout_seconds) + "\n\n")
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )

        output_queue = queue.Queue()

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    output_queue.put(line)
            except Exception as exc:
                output_queue.put(f"[SarahMemoryBlenderBootstrap] output reader stopped: {exc}\n")

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        last_heartbeat = started
        while proc.poll() is None:
            try:
                line = output_queue.get(timeout=0.5)
                print(line, end="")
                log_fh.write(line)
                log_fh.flush()
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last_heartbeat >= 30:
                elapsed = int(now - started)
                msg = f"[INFO] Blender still running... elapsed={elapsed}s timeout={timeout_seconds or 'disabled'}s\n"
                print(msg, end="")
                log_fh.write(msg)
                log_fh.flush()
                last_heartbeat = now

            if timeout_seconds > 0 and (now - started) > timeout_seconds:
                msg = f"[ERROR] Blender build timeout after {timeout_seconds} seconds. Terminating process tree.\n"
                print(msg, end="")
                log_fh.write(msg)
                log_fh.flush()
                _terminate_process_tree(proc)
                return 124

        # Drain any buffered output.
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            print(line, end="")
            log_fh.write(line)

        return int(proc.returncode or 0)


def _resolve_candidate(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return ""


def _build_defaults() -> Dict[str, str]:
    gp = _try_import_globals()
    base_dir = gp["base_dir"]

    front = _resolve_candidate(
        os.path.join(base_dir, "resources", "avatars", "smfbstriaght.png"),
        os.path.join(base_dir, "resources", "avatars", "smfbwalk2.png"),
        os.path.join(base_dir, "resources", "avatars", "smfbwalk1.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "smfbstriaght.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "smfbwalk2.png"),
        os.path.join(base_dir, "smfbwalk2.png"),
        os.path.join(base_dir, "smfbwalk1.png"),
        os.path.join(base_dir, "resources", "avatars", "3D", "source", "front.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "front.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "front.png"),
        os.path.join(base_dir, "data", "avatar", "source", "front.png"),
    )
    side = _resolve_candidate(
        os.path.join(base_dir, "resources", "avatars", "smfbturn.png"),
        os.path.join(base_dir, "resources", "avatars", "smfbturnwalk.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "smfbturn.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "smfbturnwalk.png"),
        os.path.join(base_dir, "smfbturnwalk.png"),
        os.path.join(base_dir, "resources", "avatars", "3D", "source", "side.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "side.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "side.png"),
        os.path.join(base_dir, "data", "avatar", "source", "side.png"),
    )
    back = _resolve_candidate(
        os.path.join(base_dir, "resources", "avatars", "smfbback.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "smfbback.png"),
        os.path.join(base_dir, "smfbback.png"),
        os.path.join(base_dir, "resources", "avatars", "3D", "source", "back.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "back.png"),
        os.path.join(base_dir, "resources", "avatars", "source", "back.png"),
        os.path.join(base_dir, "data", "avatar", "source", "back.png"),
    )
    blueprint = _resolve_candidate(
        os.path.join(base_dir, "resources", "avatars", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "resources", "avatars", "jpg", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "resources", "avatars", "3D", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "resources", "avatars", "3D", "source", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "resources", "avatars", "models", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "resources", "avatars", "models", "HDR-SMAVATAR-BP.png"),
        os.path.join(base_dir, "data", "avatar", "models", "HDR-SMAVATAR-BP.png"),
    )

    return {
        "blender": _default_blender_path(),
        "front": front,
        "side": side,
        "back": back,
        "blueprint": blueprint,
        "outdir": gp["avatar_3d_dir"],
        "sandbox_dir": gp["sandbox_dir"],
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _blender_script_template() -> str:
    return r'''
import bpy
import bmesh
import json
import math
import os
from math import radians
from mathutils import Vector

FRONT_IMAGE = __FRONT_IMAGE__
SIDE_IMAGE = __SIDE_IMAGE__
BACK_IMAGE = __BACK_IMAGE__
BLUEPRINT_IMAGE = __BLUEPRINT_IMAGE__
OUT_BLEND = __OUT_BLEND__
OUT_GLB = __OUT_GLB__
OUT_FBX = __OUT_FBX__
MANIFEST = __MANIFEST_PATH__

CHARACTER_NAME = "SarahMemoryAvatar"
RIG_NAME = "Sarah_Rig"
HEIGHT_METERS = 1.68
QUALITY = __QUALITY__
EXPORT_FBX_ENABLED = __EXPORT_FBX_ENABLED__
PREVIEW_RENDER_ENABLED = __PREVIEW_RENDER_ENABLED__

# SarahMemory Avatar Organ runtime doctrine:
# - Blender is the asset forge, not the live runtime.
# - GLB is the Avatar Panel / WebGL runtime asset.
# - This bootstrap favors recognizable embodied identity and low runtime cost.
# - It does not download or execute third-party model content.
QUALITY_PRESETS = {
    # Preview remains available for fallback bodies and weak machines.
    'preview': {
        'mesh_segments': 36,
        'curve_resolution': 6,
        'hair_strands': 36,
        'neon_bevel': 0.00055,
        'hair_bevel': 0.0020,
        'render_resolution': 1280,
        'max_runtime_fps': 15,
        'runtime_texture_target': '1K',
        'target_triangles': '45K-70K',
        'shader_detail_level': 'medium_fast',
        'avatar_panel_target': 'fast_preview',
        'load_budget_seconds': 8,
    },
    # Balanced is now the normal SarahMemory Avatar Panel lane: detail is raised
    # but still budgeted for the FX-8350 / DDR3 / RTX 3060 development body.
    'balanced': {
        'mesh_segments': 72,
        'curve_resolution': 12,
        'hair_strands': 112,
        'neon_bevel': 0.00075,
        'hair_bevel': 0.0024,
        'render_resolution': 1560,
        'max_runtime_fps': 24,
        'runtime_texture_target': '780p-class procedural/PBR; no baked 4K atlas',
        'target_triangles': '85K-135K',
        'shader_detail_level': '780p_shader_max_balanced',
        'avatar_panel_target': '780p_readable_runtime',
        'load_budget_seconds': 10,
    },
    # High is the local showcase lane. It is still GLB-safe and should be tested
    # only after balanced remains under the 10-second Avatar Panel switch budget.
    'high': {
        'mesh_segments': 160,
        'curve_resolution': 22,
        'hair_strands': 420,
        'neon_bevel': 0.00074,
        'hair_bevel': 0.0021,
        'render_resolution': 2560,
        'max_runtime_fps': 45,
        'runtime_texture_target': '2K procedural/PBR; production panel lane',
        'target_triangles': '450K-900K production runtime target',
        'shader_detail_level': 'goldstandard_high_runtime',
        'avatar_panel_target': 'high_detail_goldstandard_runtime',
        'load_budget_seconds': 16,
        'body_subdivision_levels': 2,
        'body_subdivision_render_levels': 2,
        'head_subdivision_levels': 2,
        'hair_subdivision_levels': 2,
        'head_segments': 112,
        'head_ring_count': 56,
        'face_mass_segments': 48,
        'eye_segments': 48,
        'iris_segments': 40,
        'hair_mass_segments': 58,
        'body_corrective_iterations': 5,
    },
    # Ultra is a heavier authoring/runtime test lane for closer GoldStandard body
    # shaping.  It raises real mesh density and face/hair surface detail, but the
    # user must validate AvatarPanel load behavior before making it the default.
    'ultra': {
        'mesh_segments': 256,
        'curve_resolution': 32,
        'hair_strands': 960,
        'neon_bevel': 0.00062,
        'hair_bevel': 0.00155,
        'render_resolution': 3840,
        'max_runtime_fps': 60,
        'runtime_texture_target': '4K authoring/PBR; GoldStandard production showcase',
        'target_triangles': '1.8M-3.5M production authoring target',
        'shader_detail_level': 'goldstandard_ultra_production',
        'avatar_panel_target': 'ultra_goldstandard_runtime_showcase',
        'load_budget_seconds': 28,
        'body_subdivision_levels': 3,
        'body_subdivision_render_levels': 3,
        'head_subdivision_levels': 3,
        'hair_subdivision_levels': 3,
        'head_segments': 160,
        'head_ring_count': 80,
        'face_mass_segments': 72,
        'eye_segments': 64,
        'iris_segments': 56,
        'hair_mass_segments': 84,
        'body_corrective_iterations': 7,
    },
    # Cinematic 2M is the GoldStandard AvatarPanel authoring lane.  v9.0.14 face-hair-motion longform tuning
    # keeps the visible-detail target high while removing the previous hang risk:
    # no 10M+ raw mesh escalation, no 1800 separate hair curve objects, no default
    # FBX export.  The visual target is produced through controlled topology,
    # surface-bound emissive details, runtime-safe hair layers, and shader detail.
    'cinematic_2m': {
        'mesh_segments': 360,
        'curve_resolution': 48,
        'hair_strands': 1600,
        'hair_ribbons': 96,
        'neon_bevel': 0.00046,
        'hair_bevel': 0.00092,
        'render_resolution': 4096,
        'max_runtime_fps': 60,
        'runtime_texture_target': '4K optimized procedural/PBR GoldStandard runtime validation',
        'target_triangles': '4M-8M optimized cinematic runtime-authoring target',
        'shader_detail_level': 'cinematic_goldstandard_entity_authoring_v9_0_15',
        'avatar_panel_target': 'cinematic_goldstandard_embodied_runtime_showcase',
        'load_budget_seconds': 0,
        'body_subdivision_levels': 3,
        'body_subdivision_render_levels': 4,
        'head_subdivision_levels': 4,
        'hair_subdivision_levels': 4,
        'head_segments': 256,
        'head_ring_count': 128,
        'face_mass_segments': 144,
        'eye_segments': 144,
        'iris_segments': 112,
        'hair_mass_segments': 160,
        'body_corrective_iterations': 12,
        'face_refinement_passes': 2,
        'goldstandard_entity': True,
    },
    'goldstandard_entity': {
        'mesh_segments': 420,
        'curve_resolution': 56,
        'hair_strands': 2200,
        'hair_ribbons': 144,
        'neon_bevel': 0.00042,
        'hair_bevel': 0.00082,
        'render_resolution': 6144,
        'max_runtime_fps': 60,
        'runtime_texture_target': '4K-8K procedural/PBR GoldStandard embodied entity validation',
        'target_triangles': '8M-16M high-definition embodied entity authoring target',
        'shader_detail_level': 'goldstandard_embodied_entity_full_authoring_v9_0_15',
        'avatar_panel_target': 'goldstandard_embodied_entity_avatarpanel_showcase',
        'load_budget_seconds': 0,
        'body_subdivision_levels': 4,
        'body_subdivision_render_levels': 4,
        'head_subdivision_levels': 4,
        'hair_subdivision_levels': 4,
        'head_segments': 320,
        'head_ring_count': 160,
        'face_mass_segments': 192,
        'eye_segments': 192,
        'iris_segments': 144,
        'hair_mass_segments': 220,
        'body_corrective_iterations': 14,
        'face_refinement_passes': 3,
        'goldstandard_entity': True,
    },
}
Q = QUALITY_PRESETS.get(QUALITY, QUALITY_PRESETS['balanced'])


def q_int(name, default, min_value=None, max_value=None):
    try:
        value = int(Q.get(name, default))
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def q_float(name, default, min_value=None, max_value=None):
    try:
        value = float(Q.get(name, default))
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def q_is_heavy_authoring():
    return str(QUALITY).lower() in ('ultra', 'cinematic_2m', 'goldstandard_entity')


def q_is_cinematic_authoring():
    return str(QUALITY).lower() in ('cinematic_2m', 'goldstandard_entity')

def q_is_goldstandard_entity():
    return bool(Q.get('goldstandard_entity')) or str(QUALITY).lower() == 'goldstandard_entity'


def log(msg):
    print(f"[SarahMemoryBlenderBootstrap] {msg}")


def ensure_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        if parent is None:
            bpy.context.scene.collection.children.link(col)
        else:
            parent.children.link(col)
    return col


def move_to_collection(obj, col):
    if obj is None or col is None:
        return
    for old_col in list(obj.users_collection):
        try:
            old_col.objects.unlink(obj)
        except Exception:
            pass
    if obj.name not in col.objects:
        col.objects.link(obj)


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for datablock_list in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.actions,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.collections,
        bpy.data.metaballs,
    ):
        try:
            for block in list(datablock_list):
                if not block.users:
                    datablock_list.remove(block)
        except Exception:
            pass


def scene_setup():
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x = int(Q.get('render_resolution', 1400))
    scene.render.resolution_y = int(Q.get('render_resolution', 1400))
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 72
    scene.frame_set(1)

    engine_names = []
    try:
        engine_names = [item.identifier for item in scene.bl_rna.properties['render'].fixed_type.properties['engine'].enum_items]
    except Exception:
        engine_names = []

    # Production visual rule:
    # - preview/balanced/high prefer EEVEE for fast GLB authoring.
    # - ultra/cinematic authoring prefer Cycles where available so still previews
    #   and saved .blend files carry the GoldStandard lighting lane.
    if q_is_heavy_authoring() and 'CYCLES' in engine_names:
        scene.render.engine = 'CYCLES'
    elif 'BLENDER_EEVEE_NEXT' in engine_names:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    elif 'BLENDER_EEVEE' in engine_names:
        scene.render.engine = 'BLENDER_EEVEE'
    else:
        scene.render.engine = 'CYCLES'

    try:
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = True
        scene.eevee.taa_render_samples = 256 if q_is_heavy_authoring() else 96
    except Exception:
        pass

    try:
        scene.cycles.samples = 384 if q_is_cinematic_authoring() else (192 if q_is_heavy_authoring() else 96)
        scene.cycles.preview_samples = 96 if q_is_cinematic_authoring() else (64 if q_is_heavy_authoring() else 32)
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 10 if q_is_cinematic_authoring() else 8
        scene.cycles.transparent_max_bounces = 8 if q_is_cinematic_authoring() else 6
        try:
            scene.cycles.device = 'GPU'
        except Exception:
            pass
    except Exception:
        pass

    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for compute_backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
            try:
                prefs.compute_device_type = compute_backend
                break
            except Exception:
                continue
        try:
            prefs.get_devices()
        except Exception:
            pass
        for device in getattr(prefs, 'devices', []):
            try:
                device.use = True
            except Exception:
                pass
    except Exception:
        pass

    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new('World')
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputWorld')
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs[0].default_value = (0.012, 0.013, 0.02, 1.0)
    bg.inputs[1].default_value = 0.55
    links.new(bg.outputs[0], out.inputs[0])
    scene.render.film_transparent = True


def safe_set_input(node, names, value):
    if not isinstance(names, (list, tuple)):
        names = [names]
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def create_reference_image(name, image_path, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), offset_axis=None):
    if not image_path or not os.path.isfile(image_path):
        log(f"Reference missing: {image_path}")
        return None
    img = bpy.data.images.load(image_path, check_existing=True)
    bpy.ops.object.empty_add(type='IMAGE', location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.data = img
    obj.empty_display_type = 'IMAGE'
    obj.empty_image_depth = 'FRONT'
    obj.empty_image_side = 'DOUBLE_SIDED'
    obj.scale = scale
    try:
        obj.color = (1.0, 1.0, 1.0, 0.85)
    except Exception:
        pass
    if offset_axis is not None:
        try:
            if offset_axis == 'Y-':
                obj.location.y -= 0.001
            elif offset_axis == 'Y+':
                obj.location.y += 0.001
            elif offset_axis == 'X+':
                obj.location.x += 0.001
            elif offset_axis == 'X-':
                obj.location.x -= 0.001
        except Exception:
            pass

    # Reference boards are Blender authoring aids only. They must remain visible
    # in the .blend file for alignment, but they must never be exported to GLB
    # where they would appear as a second hovering Sarah image plane.
    obj["sarahmemory_export_role"] = "reference_only"
    obj.hide_render = True
    return obj


def add_reference_set(ref_collection):
    refs = []
    refs.append(create_reference_image(
        'SarahRef_Front',
        FRONT_IMAGE,
        location=(0.0, -1.60, 1.02),
        rotation=(radians(90), 0.0, 0.0),
        scale=(1.12, 1.12, 1.12),
        offset_axis='Y-'
    ))
    refs.append(create_reference_image(
        'SarahRef_Back',
        BACK_IMAGE,
        location=(0.0, 1.60, 1.02),
        rotation=(radians(90), 0.0, radians(180)),
        scale=(1.12, 1.12, 1.12),
        offset_axis='Y+'
    ))
    refs.append(create_reference_image(
        'SarahRef_Side',
        SIDE_IMAGE,
        location=(1.25, 0.0, 1.02),
        rotation=(radians(90), 0.0, radians(90)),
        scale=(1.12, 1.12, 1.12),
        offset_axis='X+'
    ))
    refs.append(create_reference_image(
        'SarahBlueprintBoard',
        BLUEPRINT_IMAGE,
        location=(-2.55, 0.0, 1.15),
        rotation=(radians(90), 0.0, radians(-90)),
        scale=(1.75, 1.75, 1.75),
        offset_axis='X-'
    ))
    for obj in refs:
        if obj is not None:
            move_to_collection(obj, ref_collection)
    return [obj for obj in refs if obj is not None]


def create_skin_material():
    mat = bpy.data.materials.new(name='Sarah_Skin')
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    safe_set_input(bsdf, 'Base Color', (0.91, 0.735, 0.655, 1.0))
    safe_set_input(bsdf, 'Subsurface Weight', 0.08)
    safe_set_input(bsdf, 'Subsurface', 0.08)
    safe_set_input(bsdf, 'Subsurface Radius', (1.0, 0.45, 0.35))
    safe_set_input(bsdf, 'Roughness', 0.39)
    safe_set_input(bsdf, 'Specular IOR Level', 0.54)
    safe_set_input(bsdf, 'Specular', 0.54)
    links.new(bsdf.outputs[0], out.inputs[0])
    mat['sarahmemory_material_role'] = 'skin_goldstandard_soft_sss_runtime'
    mat['sarahmemory_skin_profile'] = 'goldstandard_embodied_entity_v9_0_15'
    return mat


def create_eye_materials():
    sclera = bpy.data.materials.new(name='Sarah_Eye_Sclera')
    sclera.use_nodes = True
    nt = sclera.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    safe_set_input(bsdf, 'Base Color', (0.97, 0.98, 1.0, 1.0))
    safe_set_input(bsdf, 'Roughness', 0.18)
    safe_set_input(bsdf, 'Specular IOR Level', 0.62)
    safe_set_input(bsdf, 'Specular', 0.62)
    nt.links.new(bsdf.outputs[0], out.inputs[0])

    iris = bpy.data.materials.new(name='Sarah_Eye_Iris')
    iris.use_nodes = True
    nt2 = iris.node_tree
    nt2.nodes.clear()
    out2 = nt2.nodes.new('ShaderNodeOutputMaterial')
    bsdf2 = nt2.nodes.new('ShaderNodeBsdfPrincipled')
    safe_set_input(bsdf2, 'Base Color', (0.12, 0.19, 0.26, 1.0))
    safe_set_input(bsdf2, 'Roughness', 0.14)
    safe_set_input(bsdf2, 'Specular IOR Level', 0.72)
    safe_set_input(bsdf2, 'Specular', 0.72)
    safe_set_input(bsdf2, ['Emission Color', 'Emission'], (0.0, 0.08, 0.14, 1.0))
    safe_set_input(bsdf2, 'Emission Strength', 0.08)
    nt2.links.new(bsdf2.outputs[0], out2.inputs[0])
    sclera['sarahmemory_material_role'] = 'eye_sclera_runtime'
    iris['sarahmemory_material_role'] = 'eye_iris_runtime_lookat_enabled'
    return sclera, iris

def create_detail_materials():
    """Create lightweight visible-detail materials for the high-end Avatar Organ bootstrap."""
    def mat_principled(name, base, roughness=0.38, metallic=0.0, emission=None, emission_strength=0.0, alpha=1.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links
        nodes.clear()
        out = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        safe_set_input(bsdf, 'Base Color', (base[0], base[1], base[2], alpha))
        safe_set_input(bsdf, 'Metallic', metallic)
        safe_set_input(bsdf, 'Roughness', roughness)
        safe_set_input(bsdf, 'Alpha', alpha)
        if emission is not None:
            safe_set_input(bsdf, ['Emission Color', 'Emission'], (emission[0], emission[1], emission[2], 1.0))
            safe_set_input(bsdf, 'Emission Strength', emission_strength)
        links.new(bsdf.outputs[0], out.inputs[0])
        if alpha < 1.0:
            mat.blend_method = 'BLEND'
            mat.use_screen_refraction = True
            mat.show_transparent_back = False
        return mat

    return {
        'lips': mat_principled('Sarah_Lips_SoftRose', (0.72, 0.22, 0.28), roughness=0.28, emission=(0.10, 0.02, 0.04), emission_strength=0.05),
        'brow': mat_principled('Sarah_Brows_DarkMagenta', (0.23, 0.035, 0.09), roughness=0.48),
        'lash': mat_principled('Sarah_Lashes_Black', (0.01, 0.008, 0.012), roughness=0.22),
        'tooth': mat_principled('Sarah_Teeth_Pearl', (0.96, 0.93, 0.86), roughness=0.20),
        'neon': mat_principled('Sarah_Neon_Geometry_Cyan_780p', (0.16, 0.95, 1.0), roughness=0.065, metallic=0.02, emission=(0.02, 0.92, 1.0), emission_strength=4.2),
        'neon_soft': mat_principled('Sarah_Neon_Soft_Cyan_780p', (0.08, 0.72, 0.90), roughness=0.18, metallic=0.01, emission=(0.02, 0.78, 1.0), emission_strength=2.2),
        'hair_strand': mat_principled('Sarah_Hair_Strands_Magenta_GoldStandard', (1.00, 0.12, 0.60), roughness=0.18, metallic=0.0, emission=(0.50, 0.012, 0.22), emission_strength=0.34),
        'hair_shadow': mat_principled('Sarah_Hair_Shadow_DeepMagenta', (0.16, 0.012, 0.065), roughness=0.32, metallic=0.0, emission=(0.06, 0.002, 0.025), emission_strength=0.08),
        'eye_wetline': mat_principled('Sarah_Eye_Wetline_Gloss', (0.98, 0.74, 0.68), roughness=0.08, metallic=0.0, emission=(0.06, 0.01, 0.012), emission_strength=0.03),
        'stage_dark': mat_principled('Sarah_Stage_DarkRuntime', (0.006, 0.008, 0.014), roughness=0.42),
        'stage_grid': mat_principled('Sarah_Stage_Grid_Cyan', (0.03, 0.55, 0.70), roughness=0.24, emission=(0.02, 0.64, 1.0), emission_strength=0.90),
    }


def create_hair_material():
    mat = bpy.data.materials.new(name='Sarah_Hair_Magenta')
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    noise = nodes.new('ShaderNodeTexNoise')
    ramp = nodes.new('ShaderNodeValToRGB')
    mix = nodes.new('ShaderNodeMixRGB')
    texcoord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')

    texcoord.location = (-1000, 0)
    mapping.location = (-800, 0)
    noise.location = (-580, 0)
    ramp.location = (-350, 0)
    mix.location = (-100, 0)
    bsdf.location = (180, 0)
    out.location = (380, 0)

    mapping.inputs['Scale'].default_value = (1.8, 1.8, 3.6)
    noise.inputs['Scale'].default_value = 5.2
    noise.inputs['Detail'].default_value = 7.0
    noise.inputs['Roughness'].default_value = 0.42
    ramp.color_ramp.elements[0].position = 0.36
    ramp.color_ramp.elements[0].color = (0.48, 0.03, 0.19, 1.0)
    ramp.color_ramp.elements[1].position = 0.84
    ramp.color_ramp.elements[1].color = (0.96, 0.19, 0.58, 1.0)
    mix.blend_type = 'MULTIPLY'
    mix.inputs[1].default_value = (0.90, 0.17, 0.58, 1.0)
    safe_set_input(bsdf, 'Roughness', 0.36)
    safe_set_input(bsdf, 'Specular IOR Level', 0.50)
    safe_set_input(bsdf, 'Specular', 0.50)
    safe_set_input(bsdf, ['Emission Color', 'Emission'], (0.22, 0.03, 0.11, 1.0))
    safe_set_input(bsdf, 'Emission Strength', 0.15)

    links.new(texcoord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], mix.inputs[2])
    links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat


def create_suit_material():
    """Create a GLB-safe shader-assisted glossy black bodysuit material.

    The Avatar Panel should read the cyan outfit detail as light emitted from the
    suit surface, not as loose geometry floating in front of Sarah.  glTF cannot
    preserve arbitrary Blender procedural node graphs reliably, so this material
    stays inside the portable PBR subset and exposes explicit metadata for the
    Three.js Avatar3D runtime to enhance clearcoat/rim/emissive behavior.

    Performance contract:
    - no heavy baked texture atlas in the bootstrap by default
    - no live Blender renderer
    - no postprocessing requirement
    - target AvatarPanel GLB switch/load stays under 10 seconds on the dev PC
    """
    mat = bpy.data.materials.new(name='Sarah_Suit_ShaderBound_BlackGloss')
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (220, 0)
    out.location = (460, 0)

    # Dark synthetic suit.  Keep values GLB-safe.  Avatar3D.tsx may enhance
    # clearcoat/envMap response at runtime without adding load-heavy textures.
    safe_set_input(bsdf, 'Base Color', (0.0016, 0.0020, 0.0038, 1.0))
    safe_set_input(bsdf, 'Metallic', 0.18)
    safe_set_input(bsdf, 'Roughness', 0.072)
    safe_set_input(bsdf, 'Specular IOR Level', 1.00)
    safe_set_input(bsdf, 'Specular', 1.00)
    safe_set_input(bsdf, 'Coat Weight', 0.96)
    safe_set_input(bsdf, 'Coat Roughness', 0.026)
    safe_set_input(bsdf, ['Emission Color', 'Emission'], (0.0, 0.025, 0.040, 1.0))
    safe_set_input(bsdf, 'Emission Strength', 0.035)
    links.new(bsdf.outputs[0], out.inputs[0])

    # Runtime/export metadata. This is a material contract for the AvatarPanel,
    # not physical-world robot authority.
    mat['sarahmemory_material_role'] = 'one_piece_suit_base'
    mat['sarahmemory_glb_safe'] = True
    mat['sarahmemory_shader_profile'] = 'goldstandard_black_gloss_clearcoat_cyan_rim_shader_max'
    mat['sarahmemory_shader_budget'] = 'goldstandard_authoring_longform_runtime_lod_controlled'
    mat['sarahmemory_shader_detail_level'] = Q.get('shader_detail_level', '780p_shader_max_balanced')
    mat['sarahmemory_trim_source'] = 'shader_assisted_surface_bound_neon_geometry'
    mat['sarahmemory_load_budget_seconds'] = int(Q.get('load_budget_seconds', 10))
    mat['sarahmemory_runtime_note'] = 'Avatar3D enhances this PBR material with clearcoat/env/rim response; GoldStandard GLB is authored high-detail, but execution remains visual-only.'
    return mat

def smooth_object(obj):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        if hasattr(obj.data, 'use_auto_smooth'):
            obj.data.use_auto_smooth = True
        obj.select_set(False)
    except Exception:
        pass


def add_scaled_primitive(kind, location, scale, rotation=(0.0, 0.0, 0.0), segments=24):
    if kind == 'uvsphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=max(16, segments), ring_count=max(8, segments // 2), radius=1.0, location=location, rotation=rotation)
    elif kind == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=max(10, segments), radius=1.0, depth=2.0, location=location, rotation=rotation)
    elif kind == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location, rotation=rotation)
    elif kind == 'ico':
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0, location=location, rotation=rotation)
    else:
        raise ValueError(f"Unsupported primitive kind: {kind}")
    obj = bpy.context.active_object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def join_selected_objects(active_obj):
    bpy.context.view_layer.objects.active = active_obj
    bpy.ops.object.join()
    return bpy.context.active_object


def object_rotation_from_points(p0, p1):
    direction = Vector(p1) - Vector(p0)
    if direction.length < 1e-6:
        return (0.0, 0.0, 0.0), 0.0
    center = (Vector(p0) + Vector(p1)) * 0.5
    quat = direction.to_track_quat('Z', 'Y')
    return center, quat.to_euler(), direction.length


def add_segment_capsule(parts, p0, p1, radius_x, radius_y=None, segments=18):
    radius_y = radius_y if radius_y is not None else radius_x
    center, rotation, length = object_rotation_from_points(p0, p1)
    cyl = add_scaled_primitive('cylinder', center, (radius_x, radius_y, max(0.02, length * 0.5)), rotation=rotation, segments=segments)
    parts.append(cyl)

    a = add_scaled_primitive('uvsphere', p0, (radius_x * 1.02, radius_y * 1.02, radius_x * 1.02), segments=segments)
    b = add_scaled_primitive('uvsphere', p1, (radius_x * 1.02, radius_y * 1.02, radius_x * 1.02), segments=segments)
    parts.extend([a, b])


def clean_mesh_geometry(obj):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0008)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
    except Exception as exc:
        log(f"Mesh cleanup warning on {obj.name}: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass


def voxel_finish(obj, voxel_size=0.035, adapt=0.0, smooth_factor=0.48, smooth_iter=6, subd_levels=1):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return obj

    try:
        remesh = obj.modifiers.new(name='VoxelRemesh', type='REMESH')
        remesh.mode = 'VOXEL'
        remesh.voxel_size = voxel_size
        remesh.adaptivity = adapt
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=remesh.name)
        obj.select_set(False)
    except Exception as exc:
        log(f"Voxel remesh warning on {obj.name}: {exc}")

    clean_mesh_geometry(obj)

    try:
        corr = obj.modifiers.new(name='CorrectiveSmooth', type='CORRECTIVE_SMOOTH')
        corr.factor = smooth_factor
        corr.iterations = smooth_iter
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=corr.name)
        obj.select_set(False)
    except Exception as exc:
        log(f"Corrective smooth warning on {obj.name}: {exc}")

    try:
        subd = obj.modifiers.new(name='Subd', type='SUBSURF')
        subd.levels = subd_levels
        subd.render_levels = q_int('subdivision_render_levels', max(2, subd_levels + 1), 0, 4)
    except Exception:
        pass

    smooth_object(obj)
    return obj


def _create_ring_mesh_object(name, rings, axis='Z', segments=40, material=None, cap_start=True, cap_end=True):
    """Create a smooth ring-based organic mesh component.

    This replaces the old capsule/sphere/cylinder stack that made Sarah read as
    a toy/LEGO proxy in the Avatar Panel.  Each component is generated from
    anatomical cross-section rings, then joined into one modular body object.
    """
    verts = []
    faces = []
    segments = max(12, int(segments or 40))

    def ring_point(center, a, b, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        x, y, z = center
        if axis == 'Z':
            return (x + a * c, y + b * s, z)
        if axis == 'X':
            return (x, y + a * c, z + b * s)
        if axis == 'Y':
            return (x + a * c, y, z + b * s)
        return (x + a * c, y + b * s, z)

    for center, a, b in rings:
        for i in range(segments):
            theta = (i / segments) * math.tau
            verts.append(ring_point(center, a, b, theta))

    ring_count = len(rings)
    for r in range(ring_count - 1):
        base = r * segments
        nxt = (r + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((base + i, base + j, nxt + j, nxt + i))

    if cap_start and ring_count > 0:
        cidx = len(verts)
        verts.append(tuple(rings[0][0]))
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((cidx, j, i))

    if cap_end and ring_count > 1:
        cidx = len(verts)
        verts.append(tuple(rings[-1][0]))
        base = (ring_count - 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((cidx, base + i, base + j))

    mesh = bpy.data.meshes.new(f'{name}Mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    smooth_object(obj)
    try:
        weighted = obj.modifiers.new(name='WeightedNormals', type='WEIGHTED_NORMAL')
        weighted.keep_sharp = True
    except Exception:
        pass
    return obj


def _join_objects_to_named_mesh(objects, name):
    objects = [o for o in objects if o is not None]
    if not objects:
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    out = bpy.context.active_object
    out.name = name
    out.data.name = f'{name}Mesh'
    clean_mesh_geometry(out)
    smooth_object(out)
    try:
        corr = out.modifiers.new(name='HumanFormCorrectiveSmooth', type='CORRECTIVE_SMOOTH')
        corr.factor = 0.22
        corr.iterations = q_int('body_corrective_iterations', 3, 1, 12)
    except Exception:
        pass
    try:
        subd = out.modifiers.new(name='HumanFormSubd', type='SUBSURF')
        subd.levels = q_int('body_subdivision_levels', 1, 0, 4)
        subd.render_levels = q_int('body_subdivision_render_levels', subd.levels, 0, 4)
    except Exception:
        pass
    try:
        weighted = out.modifiers.new(name='HumanFormWeightedNormals', type='WEIGHTED_NORMAL')
        weighted.keep_sharp = True
    except Exception:
        pass
    return out



def _apply_goldstandard_body_refinement(body_obj):
    """Refine the generated humanoid toward the GoldStandard body silhouette.

    This pass edits the procedural ring mesh in place.  It does not create any
    action authority; it only improves the visual AvatarPanel body.  The target
    is a more game-engine readable adult humanoid silhouette: narrower waist,
    cleaner shoulder/hip taper, longer leg read, more realistic hands, and less
    toy-like limb thickness.
    """
    if body_obj is None or getattr(body_obj, 'type', '') != 'MESH':
        return body_obj
    try:
        bpy.context.view_layer.objects.active = body_obj
        body_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(body_obj.data)
        for v in bm.verts:
            x, y, z = v.co.x, v.co.y, v.co.z
            ax = abs(x)
            side = 1.0 if x >= 0.0 else -1.0

            # Waist pinch and rib/hip shaping for the black suit silhouette.
            if 1.05 < z < 1.25 and ax < 0.26:
                v.co.x *= 0.86
                v.co.y *= 0.90
            if 0.88 < z < 1.04 and ax < 0.28:
                v.co.x *= 1.06
                v.co.y *= 1.02
            if 1.34 < z < 1.56 and ax < 0.30:
                v.co.x *= 1.025
                v.co.y *= 0.96
            if 1.50 < z < 1.62 and ax > 0.12:
                v.co.z += 0.010

            # Legs: improve long athletic read and reduce cylinder/mannequin look.
            if 0.18 < z < 0.55 and 0.05 < ax < 0.22:
                v.co.x *= 0.94
                v.co.y *= 0.88
            if 0.58 < z < 0.95 and 0.05 < ax < 0.22:
                v.co.x *= 1.04
                v.co.y *= 0.96
            if z < 0.12 and ax > 0.05:
                v.co.y -= 0.010

            # Arms/hands: slimmer forearm silhouette while keeping visible palms/fingers.
            if ax > 0.38 and z > 1.25:
                v.co.y *= 0.86
                v.co.z = z + (0.006 * math.sin(ax * 12.0))
            if ax > 0.90 and 1.34 < z < 1.48:
                v.co.x += side * 0.010
                v.co.y *= 0.90

        bmesh.update_edit_mesh(body_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        body_obj.select_set(False)
        try:
            body_obj['sarahmemory_body_refinement'] = 'goldstandard_entity_suit_body_proportion_v9_0_15'
        except Exception:
            pass
    except Exception as exc:
        log(f"GoldStandard body refinement warning: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return body_obj

def create_body_mesh(name='Sarah_Body'):
    """Create Sarah's v9.0.5 ring-based humanoid base mesh.

    Earlier versions used joined spheres/capsules.  That was stable but read as
    a LEGO-like mannequin.  This version builds an anatomical software body from
    cross-section rings: feet/legs, pelvis, waist, ribcage, shoulders, arms,
    palms, and tapered fingers.  It remains a bootstrap asset, but it gives the
    Avatar Panel a much more human silhouette while preserving modular slots.
    """
    seg = int(Q.get('mesh_segments', 40))
    parts = []

    torso_rings = [
        ((0.000,  0.010, 0.900), 0.150, 0.095),
        ((0.000,  0.015, 0.965), 0.188, 0.122),
        ((0.000,  0.018, 1.030), 0.205, 0.136),
        ((0.000,  0.005, 1.100), 0.180, 0.118),
        ((0.000, -0.010, 1.180), 0.150, 0.098),
        ((0.000, -0.020, 1.255), 0.132, 0.088),
        ((0.000, -0.028, 1.325), 0.150, 0.100),
        ((0.000, -0.036, 1.390), 0.185, 0.116),
        ((0.000, -0.045, 1.455), 0.220, 0.134),
        ((0.000, -0.036, 1.520), 0.235, 0.124),
        ((0.000, -0.018, 1.585), 0.160, 0.082),
        ((0.000, -0.006, 1.650), 0.064, 0.055),
    ]
    torso = _create_ring_mesh_object('Sarah_Torso_Surface', torso_rings, axis='Z', segments=seg)
    parts.append(torso)

    # Anatomical but non-explicit bust shaping, covered by the bodysuit material.
    for sign in (-1.0, 1.0):
        contour = [
            ((0.060 * sign, -0.124, 1.345), 0.035, 0.030),
            ((0.082 * sign, -0.137, 1.390), 0.057, 0.050),
            ((0.086 * sign, -0.141, 1.435), 0.070, 0.058),
            ((0.076 * sign, -0.132, 1.480), 0.050, 0.042),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Chest_Contour_{"L" if sign < 0 else "R"}', contour, axis='Z', segments=max(24, seg // 2)))

    # Legs as smooth continuous tapered surfaces.
    for side, sign in (('L', -1.0), ('R', 1.0)):
        leg_rings = [
            ((0.112 * sign,  0.000, 0.070), 0.042, 0.038),
            ((0.105 * sign, -0.005, 0.150), 0.050, 0.043),
            ((0.103 * sign, -0.005, 0.275), 0.058, 0.050),
            ((0.110 * sign, -0.004, 0.410), 0.066, 0.056),
            ((0.118 * sign, -0.002, 0.535), 0.050, 0.046),
            ((0.120 * sign,  0.002, 0.665), 0.066, 0.058),
            ((0.116 * sign,  0.006, 0.800), 0.080, 0.067),
            ((0.108 * sign,  0.010, 0.925), 0.086, 0.071),
            ((0.102 * sign,  0.010, 1.000), 0.074, 0.064),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Leg_Surface_{side}', leg_rings, axis='Z', segments=seg))

    # Feet base, kept compact so boots can sit over them.
    for side, sign in (('L', -1.0), ('R', 1.0)):
        foot_rings = [
            ((0.105 * sign, -0.020, 0.070), 0.040, 0.030),
            ((0.105 * sign, -0.080, 0.056), 0.045, 0.027),
            ((0.105 * sign, -0.155, 0.047), 0.047, 0.024),
            ((0.105 * sign, -0.232, 0.044), 0.035, 0.019),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Foot_Surface_{side}', foot_rings, axis='Y', segments=max(20, seg // 2)))

    # Arms, palms, and tapered fingers.
    for side, sign in (('L', -1.0), ('R', 1.0)):
        arm_rings = [
            ((0.170 * sign, -0.005, 1.505), 0.060, 0.060),
            ((0.290 * sign, -0.010, 1.500), 0.058, 0.056),
            ((0.430 * sign, -0.013, 1.485), 0.054, 0.050),
            ((0.560 * sign, -0.014, 1.470), 0.046, 0.044),
            ((0.700 * sign, -0.014, 1.455), 0.041, 0.039),
            ((0.880 * sign, -0.014, 1.438), 0.034, 0.032),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Arm_Surface_{side}', arm_rings, axis='X', segments=max(24, seg // 2)))

        palm_rings = [
            ((0.905 * sign, -0.014, 1.438), 0.032, 0.035),
            ((0.980 * sign, -0.017, 1.430), 0.040, 0.046),
            ((1.055 * sign, -0.020, 1.422), 0.030, 0.038),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Palm_Surface_{side}', palm_rings, axis='X', segments=18))

        finger_specs = [
            ('index', -0.038, 1.448, 0.014, 0.012, 0.150),
            ('middle', -0.016, 1.432, 0.015, 0.013, 0.170),
            ('ring', 0.006, 1.412, 0.014, 0.012, 0.152),
            ('pinky', 0.026, 1.392, 0.011, 0.010, 0.120),
        ]
        for fname, yoff, zbase, ry, rz, length in finger_specs:
            x0 = 1.050 * sign
            x1 = (1.050 + length * 0.33) * sign
            x2 = (1.050 + length * 0.68) * sign
            x3 = (1.050 + length) * sign
            rings = [
                ((x0, yoff, zbase), ry, rz),
                ((x1, yoff, zbase + 0.004), ry * 0.88, rz * 0.88),
                ((x2, yoff, zbase + 0.004), ry * 0.74, rz * 0.74),
                ((x3, yoff, zbase + 0.002), ry * 0.46, rz * 0.46),
            ]
            parts.append(_create_ring_mesh_object(f'Sarah_Finger_{fname}_{side}', rings, axis='X', segments=12))

        thumb_rings = [
            ((1.000 * sign, 0.020, 1.402), 0.015, 0.014),
            ((1.070 * sign, 0.052, 1.374), 0.014, 0.012),
            ((1.142 * sign, 0.082, 1.348), 0.012, 0.010),
        ]
        parts.append(_create_ring_mesh_object(f'Sarah_Thumb_Surface_{side}', thumb_rings, axis='X', segments=12))

    body = _join_objects_to_named_mesh(parts, name)
    if body is not None:
        _apply_goldstandard_body_refinement(body)
        body['sarahmemory_mesh_generation'] = 'v9.0.15_goldstandard_refined_ring_humanoid'
        body['sarahmemory_avatar_role'] = 'embodied_entity_visible_body'
    return body

def create_boot_pair(name='Sarah_Boots'):
    """Create more plausible platform/heel boots as modular appearance parts."""
    seg = max(24, int(Q.get('mesh_segments', 40)) // 2)
    boots = []
    for side, sign in (('L', -1.0), ('R', 1.0)):
        shaft_rings = [
            ((0.105 * sign, -0.025, 0.055), 0.060, 0.034),
            ((0.105 * sign, -0.025, 0.120), 0.064, 0.044),
            ((0.105 * sign, -0.018, 0.210), 0.060, 0.050),
            ((0.105 * sign, -0.010, 0.295), 0.052, 0.046),
        ]
        boots.append(_create_ring_mesh_object(f'Sarah_Boot_Shaft_{side}', shaft_rings, axis='Z', segments=seg))

        foot_rings = [
            ((0.105 * sign, -0.030, 0.055), 0.055, 0.030),
            ((0.105 * sign, -0.095, 0.045), 0.065, 0.030),
            ((0.105 * sign, -0.175, 0.035), 0.070, 0.027),
            ((0.105 * sign, -0.255, 0.028), 0.050, 0.020),
        ]
        boots.append(_create_ring_mesh_object(f'Sarah_Boot_Foot_{side}', foot_rings, axis='Y', segments=seg))

        platform_rings = [
            ((0.105 * sign, -0.050, 0.005), 0.067, 0.012),
            ((0.105 * sign, -0.145, 0.000), 0.074, 0.014),
            ((0.105 * sign, -0.245, 0.002), 0.058, 0.010),
        ]
        boots.append(_create_ring_mesh_object(f'Sarah_Boot_Platform_{side}', platform_rings, axis='Y', segments=seg))

        heel_rings = [
            ((0.105 * sign, 0.020, 0.020), 0.020, 0.018),
            ((0.105 * sign, 0.024, -0.065), 0.015, 0.013),
        ]
        boots.append(_create_ring_mesh_object(f'Sarah_Boot_Heel_{side}', heel_rings, axis='Z', segments=14))

    boot_obj = _join_objects_to_named_mesh(boots, name)
    if boot_obj is not None:
        boot_obj['sarahmemory_avatar_slot'] = 'boots'
    return boot_obj

def mark_body_material_regions(body_obj, suit_mat, skin_mat):
    if body_obj is None or getattr(body_obj, 'type', '') != 'MESH':
        return
    body_obj.data.materials.clear()
    body_obj.data.materials.append(suit_mat)
    body_obj.data.materials.append(skin_mat)

    try:
        bpy.context.view_layer.objects.active = body_obj
        body_obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(body_obj.data)
        for f in bm.faces:
            center = f.calc_center_median()
            x = abs(center.x)
            y = center.y
            z = center.z

            v_cut = max(0.015, (1.64 - z) * 0.34 - 0.03)
            is_cleavage = (1.26 < z < 1.60) and (y < -0.055) and (x < v_cut)
            is_neck = (1.58 < z < 1.70) and (y < -0.035) and (x < 0.07)
            if is_cleavage or is_neck:
                f.material_index = 1
            else:
                f.material_index = 0
        bmesh.update_edit_mesh(body_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        body_obj.select_set(False)
    except Exception as exc:
        log(f"Body material region warning: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass



def _apply_goldstandard_face_refinement(head_obj):
    """Refine the generated face so it stops reading as a cartoon proxy.

    This is a deterministic sculpt pass over the procedural head.  It strengthens
    eye sockets, nose bridge/tip, lips, cheek plane, jaw taper, and chin while
    staying inside the same generated GLB lane.
    """
    if head_obj is None or getattr(head_obj, 'type', '') != 'MESH':
        return head_obj
    passes = q_int('face_refinement_passes', 1, 1, 4)
    try:
        for _pass in range(passes):
            bpy.context.view_layer.objects.active = head_obj
            head_obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(head_obj.data)
            for v in bm.verts:
                x, y, z = v.co.x, v.co.y, v.co.z
                ax = abs(x)

                # Overall adult face: reduce round cartoon ball profile.
                if z > 0.090:
                    v.co.x *= 0.970
                    v.co.y *= 0.978
                if -0.125 < z < -0.055:
                    v.co.x *= 0.910
                    v.co.y *= 0.950
                if z < -0.118:
                    v.co.x *= 0.820
                    v.co.y *= 0.920

                # Eye sockets / brow ridge / almond eye area.
                if y < -0.050 and 0.018 < z < 0.095 and 0.026 < ax < 0.102:
                    v.co.y += 0.014
                    v.co.z -= 0.004
                if y < -0.045 and 0.088 < z < 0.135 and 0.025 < ax < 0.112:
                    v.co.y -= 0.006
                    v.co.z += 0.003

                # Cheekbone / cheek plane: stronger human facial planes.
                if y < -0.055 and -0.020 < z < 0.055 and 0.045 < ax < 0.118:
                    v.co.y -= 0.012
                    v.co.z += 0.002
                if y < -0.030 and -0.055 < z < 0.005 and 0.030 < ax < 0.095:
                    v.co.x *= 1.018

                # Nose bridge/tip and nostril shadow zone.
                if ax < 0.030 and y < -0.050 and 0.000 < z < 0.105:
                    strength = 1.0 - min(1.0, ax / 0.030)
                    v.co.y -= 0.018 * strength
                if ax < 0.045 and y < -0.050 and -0.040 < z < 0.018:
                    strength = 1.0 - min(1.0, ax / 0.045)
                    v.co.y -= 0.010 * strength
                if 0.014 < ax < 0.038 and y < -0.055 and -0.030 < z < 0.010:
                    v.co.y += 0.004

                # Lips/mouth region. Keep subtle geometry; curve detail handles color.
                if y < -0.050 and -0.083 < z < -0.030 and ax < 0.072:
                    v.co.y -= 0.012 * (1.0 - min(1.0, ax / 0.072))
                    v.co.z += 0.003
                if y < -0.050 and -0.112 < z < -0.075 and ax < 0.060:
                    v.co.y -= 0.006
                    v.co.z -= 0.002

                # Jawline and chin: less sphere, more face.
                if -0.145 < z < -0.070 and 0.055 < ax < 0.135:
                    v.co.y -= 0.006
                    v.co.x *= 0.94
                if z < -0.125 and ax < 0.064:
                    v.co.y -= 0.010
                    v.co.z -= 0.004

            bmesh.update_edit_mesh(head_obj.data)
            bpy.ops.object.mode_set(mode='OBJECT')
            head_obj.select_set(False)
            try:
                bpy.ops.object.shade_smooth()
            except Exception:
                pass
        head_obj['sarahmemory_face_refinement'] = 'goldstandard_likeness_sculpt_v9_0_15'
    except Exception as exc:
        log(f"GoldStandard face refinement warning: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return head_obj

def create_head_mesh(name='Sarah_Head'):
    """Create a more anatomical head/face mesh.

    The v9.0.5 head is still procedurally generated, but it is no longer a plain
    sphere with face curves pasted onto it.  It includes cranium taper, cheek
    plane, jaw/chin, subtle nose volume, and ear forms before corrective smooth.
    """
    pieces = []

    head_segments = q_int('head_segments', 72, 32, 384)
    head_ring_count = q_int('head_ring_count', max(16, head_segments // 2), 16, 192)
    face_mass_segments = q_int('face_mass_segments', 24, 16, 224)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=head_segments, ring_count=head_ring_count, radius=0.155, location=(0.0, -0.006, 1.790))
    head = bpy.context.active_object
    head.name = f'{name}_Base'
    head.scale = (0.86, 0.96, 1.12)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pieces.append(head)

    # Nose/chin/cheeks as small mesh masses merged into the head surface.
    nose = add_scaled_primitive('uvsphere', (0.0, -0.147, 1.790), (0.028, 0.030, 0.038), segments=face_mass_segments)
    nose.name = 'Sarah_Head_NoseMass'
    pieces.append(nose)
    chin = add_scaled_primitive('uvsphere', (0.0, -0.084, 1.692), (0.055, 0.040, 0.035), segments=face_mass_segments)
    chin.name = 'Sarah_Head_ChinMass'
    pieces.append(chin)
    for side, sign in (('L', -1.0), ('R', 1.0)):
        cheek = add_scaled_primitive('uvsphere', (0.070 * sign, -0.095, 1.770), (0.046, 0.026, 0.050), segments=face_mass_segments)
        cheek.name = f'Sarah_Head_Cheek_{side}'
        pieces.append(cheek)
        ear = add_scaled_primitive('uvsphere', (0.132 * sign, -0.002, 1.790), (0.026, 0.014, 0.058), rotation=(0.0, radians(0.0), radians(4.0 * sign)), segments=max(20, face_mass_segments // 2))
        ear.name = f'Sarah_Head_Ear_{side}'
        pieces.append(ear)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in pieces:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    head = bpy.context.active_object
    head.name = name
    head.data.name = f'{name}Mesh'
    voxel_finish(head, voxel_size=0.009 if q_is_heavy_authoring() else 0.011, adapt=0.012 if q_is_heavy_authoring() else 0.02, smooth_factor=0.16, smooth_iter=q_int('body_corrective_iterations', 3, 1, 12), subd_levels=q_int('head_subdivision_levels', 1, 0, 4))

    try:
        bpy.context.view_layer.objects.active = head
        head.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(head.data)
        for v in bm.verts:
            x, y, z = v.co.x, v.co.y, v.co.z

            # Flatten back of skull slightly and taper crown/jaw.
            if y > 0.060:
                v.co.y *= 0.90
            if z > 0.110:
                v.co.x *= 0.93
                v.co.y *= 0.94
            if z < -0.075:
                v.co.x *= 0.76
                v.co.y *= 0.88

            # Eye socket relief and cheek plane.
            if y < -0.060 and 0.010 < z < 0.095 and 0.025 < abs(x) < 0.095:
                v.co.y += 0.010
                v.co.z -= 0.003
            if y < -0.055 and -0.040 < z < 0.045 and 0.045 < abs(x) < 0.105:
                v.co.y -= 0.008

            # Nose bridge/tip and philtrum zone.
            if abs(x) < 0.028 and y < -0.050 and -0.010 < z < 0.080:
                v.co.y -= 0.014 * (1.0 - min(1.0, abs(x) / 0.028))
            if abs(x) < 0.020 and y < -0.050 and -0.070 < z < -0.025:
                v.co.y += 0.008

            # Soft jaw/chin definition.
            if -0.130 < z < -0.065 and y < -0.035:
                v.co.y -= 0.006
            if z < -0.100:
                v.co.x *= 0.88

        bmesh.update_edit_mesh(head.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        head.select_set(False)
    except Exception as exc:
        log(f"Human-form head sculpt warning: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    _apply_goldstandard_face_refinement(head)

    try:
        weighted = head.modifiers.new(name='HeadWeightedNormals', type='WEIGHTED_NORMAL')
        weighted.keep_sharp = True
    except Exception:
        pass
    return head

def create_eye(side='L'):
    sx = -0.055 if side == 'L' else 0.055
    bpy.ops.mesh.primitive_uv_sphere_add(segments=q_int('eye_segments', 32, 16, 224), ring_count=max(8, q_int('eye_segments', 32, 16, 224) // 2), radius=0.026, location=(sx, -0.118, 1.80))
    sclera = bpy.context.active_object
    sclera.name = f'Sarah_Eye_{side}'
    sclera.scale = (1.0, 1.18, 1.0)
    smooth_object(sclera)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=q_int('iris_segments', 24, 12, 180), ring_count=max(6, q_int('iris_segments', 24, 12, 180) // 2), radius=0.0125, location=(sx, -0.142, 1.80))
    iris = bpy.context.active_object
    iris.name = f'Sarah_Iris_{side}'
    iris.scale = (1.0, 0.24, 1.0)
    smooth_object(iris)
    iris.parent = sclera
    return sclera, iris


def create_hair_mesh(name='Sarah_Hair'):
    pieces = []

    hair_mass_segments = q_int('hair_mass_segments', 58, 18, 256)
    shell = add_scaled_primitive('uvsphere', (0.0, 0.02, 1.84), (0.20, 0.17, 0.22), segments=hair_mass_segments)
    pieces.append(shell)
    crown = add_scaled_primitive('uvsphere', (0.0, 0.00, 1.90), (0.17, 0.14, 0.12), segments=max(24, hair_mass_segments - 4))
    pieces.append(crown)
    back_mass = add_scaled_primitive('uvsphere', (0.0, 0.14, 1.52), (0.19, 0.12, 0.46 if q_is_goldstandard_entity() else 0.42), segments=max(24, hair_mass_segments - 4))
    pieces.append(back_mass)
    if q_is_goldstandard_entity():
        # Fuller GoldStandard silhouette: crown volume, side waves, and shoulder-length fall.
        pieces.append(add_scaled_primitive('uvsphere', (-0.075, -0.075, 1.835), (0.082, 0.035, 0.145), rotation=(radians(4.0), radians(0.0), radians(-12.0)), segments=max(32, hair_mass_segments // 2)))
        pieces.append(add_scaled_primitive('uvsphere', (0.075, -0.075, 1.835), (0.082, 0.035, 0.145), rotation=(radians(4.0), radians(0.0), radians(12.0)), segments=max(32, hair_mass_segments // 2)))
        pieces.append(add_scaled_primitive('uvsphere', (-0.205, 0.055, 1.405), (0.060, 0.055, 0.355), rotation=(radians(-2.0), radians(8.0), radians(-14.0)), segments=max(32, hair_mass_segments // 2)))
        pieces.append(add_scaled_primitive('uvsphere', (0.205, 0.055, 1.405), (0.060, 0.055, 0.355), rotation=(radians(-2.0), radians(-8.0), radians(14.0)), segments=max(32, hair_mass_segments // 2)))

    for side, sign in (('L', -1.0), ('R', 1.0)):
        front_lock = add_scaled_primitive('uvsphere', (0.10 * sign, -0.09, 1.77), (0.07, 0.04, 0.16), rotation=(radians(8.0), radians(-4.0 * sign), radians(8.0 * sign)), segments=max(18, hair_mass_segments // 2))
        side_fall = add_scaled_primitive('uvsphere', (0.16 * sign, -0.01, 1.60), (0.05, 0.04, 0.26), rotation=(radians(6.0), radians(-8.0 * sign), radians(14.0 * sign)), segments=max(18, hair_mass_segments // 2))
        shoulder_fall = add_scaled_primitive('uvsphere', (0.18 * sign, 0.05, 1.36), (0.04, 0.04, 0.26), rotation=(radians(0.0), radians(-12.0 * sign), radians(18.0 * sign)), segments=max(18, hair_mass_segments // 2))
        pieces.extend([front_lock, side_fall, shoulder_fall])

    bpy.ops.object.select_all(action='DESELECT')
    for obj in pieces:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = shell
    bpy.ops.object.join()
    hair = bpy.context.active_object
    hair.name = name
    hair.data.name = f'{name}_Mesh'
    voxel_finish(hair, voxel_size=0.0065 if q_is_cinematic_authoring() else (0.008 if q_is_heavy_authoring() else 0.012), adapt=0.006 if q_is_cinematic_authoring() else (0.012 if q_is_heavy_authoring() else 0.026), smooth_factor=0.14, smooth_iter=q_int('body_corrective_iterations', 3, 1, 12), subd_levels=q_int('hair_subdivision_levels', 1, 0, 4))
    try:
        hair['sarahmemory_hair_silhouette'] = 'goldstandard_layered_magenta_flow_v9_0_15'
    except Exception:
        pass
    return hair


def assign_material(obj, mat):
    if obj is None or mat is None or getattr(obj, 'type', '') != 'MESH':
        return
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat



def make_poly_curve(name, points, mat, bevel_depth=0.004, bevel_resolution=2, resolution_u=4):
    """Create a lightweight 3D curve for visible neon/hair/face detail.

    Curves are intentionally used here because they are far cheaper than sculpted
    mesh detail and make the runtime avatar immediately more recognizable in the
    Avatar Panel. They can be converted to mesh during export if required by a
    target runtime, but GLB export can preserve them as renderable geometry in
    current Blender pipelines.
    """
    curve = bpy.data.curves.new(name=f'{name}_Curve', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = int(resolution_u)
    curve.bevel_depth = float(bevel_depth)
    curve.bevel_resolution = int(bevel_resolution)
    curve.fill_mode = 'FULL'
    spline = curve.splines.new('POLY')
    spline.points.add(max(0, len(points) - 1))
    for idx, point in enumerate(points):
        spline.points[idx].co = (float(point[0]), float(point[1]), float(point[2]), 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def make_disc(name, location, scale, mat, segments=32, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=max(16, segments), ring_count=max(8, segments // 2), radius=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    smooth_object(obj)
    assign_material(obj, mat)
    return obj


def parent_keep_transform(obj, parent):
    if obj is None or parent is None:
        return
    try:
        matrix_world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = matrix_world
    except Exception:
        pass


def create_high_end_face_details(rig_obj, mats):
    """Add visible face details so the bootstrap no longer reads as a stick proxy."""
    objs = []
    lips = mats.get('lips')
    brow = mats.get('brow')
    lash = mats.get('lash')
    tooth = mats.get('tooth')

    # Lips and mouth line, placed on the front of the head.
    objs.append(make_poly_curve('Sarah_Lip_Upper', [(-0.042, -0.158, 1.765), (-0.020, -0.166, 1.774), (0.000, -0.170, 1.776), (0.020, -0.166, 1.774), (0.042, -0.158, 1.765)], lips, bevel_depth=0.0035, bevel_resolution=2, resolution_u=Q['curve_resolution']))
    objs.append(make_poly_curve('Sarah_Lip_Lower', [(-0.039, -0.159, 1.753), (-0.018, -0.166, 1.746), (0.000, -0.169, 1.744), (0.018, -0.166, 1.746), (0.039, -0.159, 1.753)], lips, bevel_depth=0.0038, bevel_resolution=2, resolution_u=Q['curve_resolution']))
    objs.append(make_poly_curve('Sarah_Mouth_Shadow', [(-0.035, -0.171, 1.758), (0.000, -0.177, 1.756), (0.035, -0.171, 1.758)], lash, bevel_depth=0.0018, bevel_resolution=1, resolution_u=Q['curve_resolution']))

    # Brows and lashes make the face readable at panel scale.
    for side, sign in (('L', -1.0), ('R', 1.0)):
        objs.append(make_poly_curve(f'Sarah_Brow_{side}', [(0.030 * sign, -0.151, 1.836), (0.058 * sign, -0.158, 1.846), (0.094 * sign, -0.153, 1.840)], brow, bevel_depth=0.0035, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Lash_Top_{side}', [(0.026 * sign, -0.155, 1.805), (0.055 * sign, -0.164, 1.812), (0.087 * sign, -0.157, 1.807)], lash, bevel_depth=0.0022, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Lash_Wing_{side}', [(0.082 * sign, -0.157, 1.807), (0.104 * sign, -0.151, 1.812)], lash, bevel_depth=0.0020, bevel_resolution=1, resolution_u=Q['curve_resolution']))

    # Nose bridge/highlight and subtle facial shaping markers.
    objs.append(make_poly_curve('Sarah_Nose_Bridge', [(0.000, -0.162, 1.815), (0.000, -0.170, 1.790), (0.000, -0.168, 1.775)], mats.get('neon_soft'), bevel_depth=0.0014, bevel_resolution=1, resolution_u=Q['curve_resolution']))
    objs.append(make_disc('Sarah_Tooth_Glimmer', (0.000, -0.174, 1.761), (0.028, 0.003, 0.006), tooth, segments=18))

    for obj in objs:
        parent_keep_transform(obj, rig_obj)
    return objs



def create_goldstandard_face_micro_details(rig_obj, mats):
    """Additional readable face detail for GoldStandard AvatarPanel scale."""
    if not q_is_goldstandard_entity():
        return []
    objs = []
    lips = mats.get('lips')
    lash = mats.get('lash')
    brow = mats.get('brow')
    wet = mats.get('eye_wetline') or mats.get('tooth')
    soft = mats.get('neon_soft')
    try:
        # Refined lip volume / cupid bow / lower lip highlight.
        objs.append(make_poly_curve('Sarah_Lip_CupidBow_Gold', [(-0.025, -0.179, 1.771), (-0.010, -0.184, 1.780), (0.000, -0.187, 1.775), (0.010, -0.184, 1.780), (0.025, -0.179, 1.771)], lips, bevel_depth=0.0024, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_Lip_LowerHighlight_Gold', [(-0.030, -0.178, 1.748), (0.000, -0.185, 1.742), (0.030, -0.178, 1.748)], lips, bevel_depth=0.0022, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        for side, sign in (('L', -1.0), ('R', 1.0)):
            objs.append(make_poly_curve(f'Sarah_Eyelid_Upper_Gold_{side}', [(0.026 * sign, -0.168, 1.809), (0.055 * sign, -0.178, 1.818), (0.090 * sign, -0.167, 1.811)], lash, bevel_depth=0.0016, bevel_resolution=1, resolution_u=Q['curve_resolution']))
            objs.append(make_poly_curve(f'Sarah_Eyelid_Lower_Gold_{side}', [(0.030 * sign, -0.166, 1.794), (0.058 * sign, -0.174, 1.790), (0.086 * sign, -0.164, 1.796)], wet, bevel_depth=0.0012, bevel_resolution=1, resolution_u=Q['curve_resolution']))
            objs.append(make_poly_curve(f'Sarah_Brow_Secondary_Gold_{side}', [(0.028 * sign, -0.153, 1.852), (0.060 * sign, -0.163, 1.861), (0.102 * sign, -0.154, 1.852)], brow, bevel_depth=0.0017, bevel_resolution=1, resolution_u=Q['curve_resolution']))
            objs.append(make_poly_curve(f'Sarah_Cheek_SoftPlane_Gold_{side}', [(0.050 * sign, -0.161, 1.760), (0.088 * sign, -0.154, 1.744), (0.111 * sign, -0.136, 1.724)], soft, bevel_depth=0.00065, bevel_resolution=1, resolution_u=Q['curve_resolution']))
            objs.append(make_poly_curve(f'Sarah_Iris_Gloss_Catchlight_{side}', [(0.050 * sign, -0.155, 1.807), (0.056 * sign, -0.157, 1.813)], wet, bevel_depth=0.0015, bevel_resolution=1, resolution_u=Q['curve_resolution']))
    except Exception as exc:
        log(f"GoldStandard face micro details skipped: {exc}")
    for obj in objs:
        parent_keep_transform(obj, rig_obj)
    return objs


def create_goldstandard_hair_silhouette_layers(rig_obj, mats):
    """Add larger wavy ribbons so the hair reads as flowing hair, not a helmet."""
    if not q_is_goldstandard_entity():
        return []
    objs = []
    mat = mats.get('hair_strand')
    shadow = mats.get('hair_shadow') or mat
    ribbon_count = q_int('hair_ribbons', 64, 16, 180)
    try:
        for idx in range(ribbon_count):
            t = idx / max(1, ribbon_count - 1)
            side = -1.0 if t < 0.5 else 1.0
            lane = (t - 0.5) * 0.40
            lateral = abs(t - 0.5) * 2.0
            phase = t * math.pi * 6.0
            x0 = lane * 0.72
            y0 = -0.092 + 0.020 * math.sin(phase)
            z0 = 1.935 - 0.020 * math.cos(phase)
            x1 = x0 + side * (0.025 + 0.035 * lateral)
            y1 = -0.010 + 0.075 * math.sin(phase + 0.7)
            z1 = 1.720 - 0.060 * math.sin(phase * 0.6)
            x2 = x0 + side * (0.045 + 0.055 * lateral)
            y2 = 0.055 + 0.080 * math.sin(phase + 1.2)
            z2 = 1.480 - 0.105 * math.sin(phase * 0.5)
            x3 = x0 + side * (0.055 + 0.075 * lateral)
            y3 = 0.085 + 0.070 * math.cos(phase + 0.4)
            z3 = 1.205 - 0.140 * math.sin(phase * 0.42)
            x4 = x0 + side * (0.045 + 0.060 * lateral)
            y4 = 0.115 + 0.050 * math.sin(phase + 1.7)
            z4 = 1.000 - 0.070 * math.cos(phase * 0.31)
            use_mat = mat if idx % 5 else shadow
            obj = make_poly_curve(f'Sarah_Hair_Ribbon_Gold_{idx:03d}', [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)], use_mat, bevel_depth=float(Q.get('hair_bevel', 0.0010)) * (2.3 if idx % 7 == 0 else 1.45), bevel_resolution=3, resolution_u=Q['curve_resolution'])
            parent_keep_transform(obj, rig_obj)
            objs.append(obj)
    except Exception as exc:
        log(f"GoldStandard hair silhouette layers skipped: {exc}")
    return objs

def create_high_end_hair_strands(rig_obj, mats):
    """Layer stylized magenta strand curves over the hair mass.

    This gives the avatar a recognizable SarahMemory silhouette while keeping the
    runtime cheaper than simulated hair particles.
    """
    objs = []
    count = int(Q.get('hair_strands', 24))
    bevel = float(Q.get('hair_bevel', 0.0032))
    mat = mats.get('hair_strand')
    for idx in range(count):
        t = idx / max(1, count - 1)
        side = -1.0 if idx % 2 == 0 else 1.0
        lane = (t - 0.5) * 0.34
        wave = math.sin(t * math.pi * 5.0) * 0.025
        x0 = lane
        y0 = -0.035 + 0.040 * math.sin(t * math.pi)
        z0 = 1.935 - 0.045 * math.cos(t * math.pi * 2.0)
        x1 = lane + side * (0.020 + 0.020 * math.sin(t * math.pi * 3.0))
        y1 = 0.050 + 0.070 * math.sin(t * math.pi * 2.0)
        z1 = 1.660 - 0.080 * math.sin(t * math.pi)
        x2 = lane + side * (0.035 + wave)
        y2 = 0.090 + 0.080 * math.sin(t * math.pi * 1.7)
        z2 = 1.350 - 0.170 * math.sin(t * math.pi * 0.8)
        x3 = lane + side * (0.050 + wave * 0.6)
        y3 = 0.115 + 0.060 * math.cos(t * math.pi * 2.0)
        z3 = 1.100 - 0.150 * math.sin(t * math.pi)
        obj = make_poly_curve(f'Sarah_Hair_Strands_{idx:02d}', [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)], mat, bevel_depth=bevel, bevel_resolution=2, resolution_u=Q['curve_resolution'])
        parent_keep_transform(obj, rig_obj)
        objs.append(obj)
    return objs


def create_high_end_neon_circuitry(rig_obj, mats):
    """Build raised cyan circuit geometry matching the provided reference art."""
    neon = mats.get('neon')
    soft = mats.get('neon_soft')
    bevel = float(Q.get('neon_bevel', 0.0055))
    objs = []

    # Front torso core lines.
    front_paths = [
        ('Sarah_Neon_Torso_Center', [(0.000, -0.171, 1.015), (0.000, -0.176, 1.180), (0.000, -0.180, 1.390), (0.000, -0.168, 1.560)]),
        ('Sarah_Neon_Torso_Left_V', [(0.000, -0.176, 1.230), (-0.105, -0.174, 1.300), (-0.155, -0.166, 1.445), (-0.052, -0.162, 1.570)]),
        ('Sarah_Neon_Torso_Right_V', [(0.000, -0.176, 1.230), (0.105, -0.174, 1.300), (0.155, -0.166, 1.445), (0.052, -0.162, 1.570)]),
        ('Sarah_Neon_Waist_Left', [(0.000, -0.171, 1.040), (-0.115, -0.158, 1.085), (-0.205, -0.126, 1.180)]),
        ('Sarah_Neon_Waist_Right', [(0.000, -0.171, 1.040), (0.115, -0.158, 1.085), (0.205, -0.126, 1.180)]),
        ('Sarah_Neon_Collar_Left', [(-0.010, -0.160, 1.585), (-0.075, -0.168, 1.625), (-0.145, -0.135, 1.615)]),
        ('Sarah_Neon_Collar_Right', [(0.010, -0.160, 1.585), (0.075, -0.168, 1.625), (0.145, -0.135, 1.615)]),
    ]
    for name, points in front_paths:
        objs.append(make_poly_curve(name, points, neon, bevel_depth=bevel, bevel_resolution=2, resolution_u=Q['curve_resolution']))

    # Legs, boots, arms.
    for side, sign in (('L', -1.0), ('R', 1.0)):
        objs.append(make_poly_curve(f'Sarah_Neon_Leg_Front_{side}', [(0.088 * sign, -0.105, 0.120), (0.105 * sign, -0.122, 0.410), (0.118 * sign, -0.130, 0.720), (0.118 * sign, -0.126, 0.960)], neon, bevel_depth=bevel, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Neon_Leg_Side_{side}', [(0.155 * sign, -0.010, 0.160), (0.172 * sign, -0.010, 0.470), (0.168 * sign, -0.005, 0.760), (0.145 * sign, 0.010, 0.980)], soft, bevel_depth=bevel * 0.78, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Neon_Boot_{side}', [(0.060 * sign, -0.205, 0.020), (0.100 * sign, -0.240, 0.040), (0.155 * sign, -0.176, 0.035), (0.150 * sign, -0.080, 0.060)], neon, bevel_depth=bevel, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Neon_Arm_{side}', [(0.220 * sign, -0.045, 1.500), (0.420 * sign, -0.062, 1.485), (0.650 * sign, -0.060, 1.455), (0.895 * sign, -0.060, 1.430)], neon, bevel_depth=bevel * 0.72, bevel_resolution=2, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve(f'Sarah_Neon_Shoulder_{side}', [(0.145 * sign, -0.105, 1.555), (0.230 * sign, -0.040, 1.535), (0.300 * sign, 0.020, 1.510)], soft, bevel_depth=bevel * 0.60, bevel_resolution=2, resolution_u=Q['curve_resolution']))

    # Back silhouette lines visible during turn/walk animations.
    back_paths = [
        ('Sarah_Neon_Back_Spine', [(0.000, 0.155, 1.610), (0.000, 0.180, 1.360), (0.000, 0.175, 1.120), (0.000, 0.145, 0.940)]),
        ('Sarah_Neon_Back_Left_X', [(-0.155, 0.120, 1.260), (-0.065, 0.165, 1.175), (-0.155, 0.125, 1.050)]),
        ('Sarah_Neon_Back_Right_X', [(0.155, 0.120, 1.260), (0.065, 0.165, 1.175), (0.155, 0.125, 1.050)]),
    ]
    for name, points in back_paths:
        objs.append(make_poly_curve(name, points, soft, bevel_depth=bevel * 0.82, bevel_resolution=2, resolution_u=Q['curve_resolution']))

    for obj in objs:
        parent_keep_transform(obj, rig_obj)
    return objs


def create_avatar_micro_stage(mats):
    """Create a tiny Avatar Panel stage so the exported preview does not look dead."""
    objs = []
    dark = mats.get('stage_dark')
    grid = mats.get('stage_grid')
    try:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, -0.045))
        floor = bpy.context.active_object
        floor.name = 'Sarah_AvatarPanel_MicroStage_Floor'
        floor.scale = (1.20, 1.20, 0.010)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        assign_material(floor, dark)
        objs.append(floor)
    except Exception:
        pass

    # A few cheap grid strokes. No particles, no live simulation.
    for idx, x in enumerate([-0.80, -0.40, 0.0, 0.40, 0.80]):
        objs.append(make_poly_curve(f'Sarah_Stage_Grid_X_{idx}', [(x, -0.95, -0.020), (x, 0.95, -0.020)], grid, bevel_depth=0.0015, bevel_resolution=1, resolution_u=1))
    for idx, y in enumerate([-0.80, -0.40, 0.0, 0.40, 0.80]):
        objs.append(make_poly_curve(f'Sarah_Stage_Grid_Y_{idx}', [(-0.95, y, -0.020), (0.95, y, -0.020)], grid, bevel_depth=0.0015, bevel_resolution=1, resolution_u=1))
    return objs



def _convert_curve_to_mesh_for_runtime(obj):
    """Convert a runtime curve detail into mesh geometry before surface binding.

    Curves are convenient for authoring, but for a game-engine style AvatarPanel
    runtime they should become real mesh strips before shrinkwrap.  This avoids
    straight guide-like lines floating in front of the body when viewed from the
    side.
    """
    if obj is None:
        return None
    try:
        if getattr(obj, 'type', '') != 'CURVE':
            return obj
        original_name = obj.name
        original_parent = obj.parent
        original_matrix = obj.matrix_world.copy()
        original_props = {}
        try:
            for k in obj.keys():
                original_props[k] = obj[k]
        except Exception:
            pass
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')
        converted = bpy.context.active_object
        converted.name = original_name
        converted.matrix_world = original_matrix
        converted.parent = original_parent
        for k, v in original_props.items():
            try:
                converted[k] = v
            except Exception:
                pass
        try:
            smooth_object(converted)
        except Exception:
            pass
        return converted
    except Exception as exc:
        log(f"Curve-to-mesh runtime conversion warning for {getattr(obj, 'name', 'unknown')}: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        return obj


def _apply_surface_shrinkwrap(obj, target, *, offset=0.0010, role='surface_bound_detail'):
    """Bind modular visual detail to the nearest surface of a target mesh."""
    if obj is None or target is None:
        return obj
    if getattr(obj, 'type', '') != 'MESH' or getattr(target, 'type', '') != 'MESH':
        return obj
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new(name='SarahSurfaceBind', type='SHRINKWRAP')
        mod.target = target
        try:
            mod.wrap_method = 'NEAREST_SURFACEPOINT'
        except Exception:
            pass
        try:
            mod.offset = float(offset)
        except Exception:
            pass
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception:
            # Keep modifier if Blender refuses to apply it; glTF export_apply can
            # still evaluate it in many builds.
            pass
        obj['sarahmemory_surface_bound'] = True
        obj['sarahmemory_surface_target'] = target.name
        obj['sarahmemory_avatar_slot'] = role
        obj['sarahmemory_shader_assisted'] = True
        obj['sarahmemory_surface_offset_m'] = float(offset)
        # Keep detail readable but thin. Thick raised tubes are what read as
        # floating rails in the Avatar Panel.  The shader now carries the glow.
        try:
            obj.show_transparent = False
        except Exception:
            pass
        obj.select_set(False)
    except Exception as exc:
        log(f"Surface-bind warning for {getattr(obj, 'name', 'unknown')}: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return obj


def bind_runtime_detail_surfaces(detail_objects, body_obj, head_obj, boots_obj):
    """Convert and surface-bind outfit/face details so they ride on Sarah.

    The Avatar Organ uses modular game-character parts.  However, the cyan suit
    lines must behave like outfit trim sitting on the suit, not like loose rails
    floating in front of the model.  This function converts authored curves into
    mesh strips and shrinkwraps them to the generated humanoid surface.
    """
    bound = []
    for obj in list(detail_objects or []):
        if obj is None:
            continue
        name = str(getattr(obj, 'name', ''))
        target = None
        offset = 0.0035
        role = 'runtime_detail'

        if name.startswith('Sarah_Neon_'):
            target = boots_obj if 'Boot' in name else body_obj
            offset = 0.00022
            role = 'emissive_suit_trim_780p_shader_max_surface_bound'
        elif name.startswith(('Sarah_Lip_', 'Sarah_Mouth_', 'Sarah_Brow_', 'Sarah_Lash_', 'Sarah_Nose_')):
            target = head_obj
            offset = 0.00075
            role = 'facial_detail_780p_shader_max_surface_bound'
        else:
            bound.append(obj)
            continue

        converted = _convert_curve_to_mesh_for_runtime(obj)
        converted = _apply_surface_shrinkwrap(converted, target, offset=offset, role=role)
        bound.append(converted if converted is not None else obj)

    return bound


def _add_armature_modifier(obj, rig_obj):
    """Bind a generated detail mesh to Sarah_Rig without using automatic weights.

    Automatic weights are expensive and unreliable for thin neon/hair/facial
    curves.  Explicit vertex groups keep the suit trim, lips, lashes, and hair
    strands moving with the same runtime bones as the body instead of floating
    as a second unbound avatar when the AvatarPanel animates the skeleton.
    """
    if obj is None or rig_obj is None or getattr(obj, 'type', '') != 'MESH':
        return obj
    try:
        mod = obj.modifiers.get('SarahRuntimeArmature') or obj.modifiers.new(name='SarahRuntimeArmature', type='ARMATURE')
        mod.object = rig_obj
        try:
            mod.use_vertex_groups = True
        except Exception:
            pass
        obj['sarahmemory_runtime_armature_bound'] = True
        obj['sarahmemory_runtime_armature'] = rig_obj.name
    except Exception as exc:
        log(f"Runtime detail armature modifier warning for {getattr(obj, 'name', 'unknown')}: {exc}")
    return obj


def _assign_single_group(obj, group_name, weight=1.0):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return
    try:
        group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
        ids = [v.index for v in obj.data.vertices]
        if ids:
            group.add(ids, float(weight), 'REPLACE')
    except Exception:
        pass


def _assign_weighted_groups(obj, weighted_names):
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return
    for group_name, weight in weighted_names:
        try:
            group = obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)
            ids = [v.index for v in obj.data.vertices]
            if ids:
                group.add(ids, float(weight), 'ADD')
        except Exception:
            pass


def _assign_surface_detail_vertex_groups(obj):
    """Heuristic bone binding for generated runtime details.

    The generated mesh is procedural, so there are no hand-painted weights.  The
    binding below is intentionally deterministic and local-first: each detail
    object is routed to the nearest visual control bones by name and coordinate.
    """
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return
    name = str(getattr(obj, 'name', ''))
    lower = name.lower()

    if lower.startswith(('sarah_lip_', 'sarah_mouth_')):
        _assign_weighted_groups(obj, [('head', 0.35), ('jaw', 0.85)])
        return
    if lower.startswith(('sarah_brow_', 'sarah_lash_', 'sarah_nose_', 'sarah_tooth_')):
        _assign_single_group(obj, 'head', 1.0)
        return
    if lower.startswith('sarah_hair_strands_'):
        set_hair_vertex_groups(obj)
        return
    if 'governancehalo' in lower or 'voicewave' in lower:
        _assign_single_group(obj, 'spine.03', 1.0)
        return
    if 'constructorbit' in lower or 'microspark' in lower or 'dataribbon' in lower:
        _assign_single_group(obj, 'root', 1.0)
        return

    # Suit/boot trim.  Assign by object name first, then by X/Z region.
    try:
        verts = list(obj.data.vertices)
    except Exception:
        verts = []
    if not verts:
        _assign_single_group(obj, 'spine.02', 1.0)
        return

    groups = {name: obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name) for name in (
        'root', 'pelvis', 'spine.01', 'spine.02', 'spine.03',
        'upper_arm.L', 'forearm.L', 'hand.L', 'upper_arm.R', 'forearm.R', 'hand.R',
        'thigh.L', 'shin.L', 'foot.L', 'toe.L', 'thigh.R', 'shin.R', 'foot.R', 'toe.R'
    )}

    def choose_bone(co):
        x, y, z = co.x, co.y, co.z
        side = 'L' if x < 0 else 'R'
        ax = abs(x)
        if 'boot' in lower:
            return f'foot.{side}' if z > 0.045 else f'toe.{side}'
        if 'arm' in lower or 'shoulder' in lower or ax > 0.24 and z > 1.20:
            if ax > 0.82:
                return f'hand.{side}'
            if ax > 0.52:
                return f'forearm.{side}'
            return f'upper_arm.{side}'
        if 'leg' in lower or (ax > 0.055 and z < 1.05):
            if z < 0.18:
                return f'foot.{side}'
            if z < 0.58:
                return f'shin.{side}'
            return f'thigh.{side}'
        if z < 1.03:
            return 'pelvis'
        if z < 1.25:
            return 'spine.01'
        if z < 1.48:
            return 'spine.02'
        return 'spine.03'

    for v in verts:
        bone_name = choose_bone(v.co)
        try:
            groups[bone_name].add([v.index], 1.0, 'REPLACE')
        except Exception:
            pass


def bind_runtime_detail_armature(detail_objects, rig_obj):
    """Make all generated detail layers deformation-ready for tonight's runtime.

    This specifically fixes the earlier AvatarPanel safety stop where skeleton
    animation was disabled because neon/hair/face details were not all bound to
    the same Sarah_Rig.  The visual layers remain governed interface graphics;
    this does not create physical robot authority.
    """
    rebound = []
    for obj in list(detail_objects or []):
        if obj is None:
            continue
        converted = _convert_curve_to_mesh_for_runtime(obj)
        if converted is not None:
            _assign_surface_detail_vertex_groups(converted)
            _add_armature_modifier(converted, rig_obj)
            rebound.append(converted)
        else:
            rebound.append(obj)
    return rebound


def create_high_end_runtime_detail_layers(rig_obj, mats):
    """Create all high-end visual detail layers for the Avatar Organ bootstrap."""
    detail_objects = []
    detail_objects.extend(create_high_end_face_details(rig_obj, mats))
    detail_objects.extend(create_goldstandard_face_micro_details(rig_obj, mats))
    detail_objects.extend(create_high_end_hair_strands(rig_obj, mats))
    detail_objects.extend(create_goldstandard_hair_silhouette_layers(rig_obj, mats))
    detail_objects.extend(create_high_end_neon_circuitry(rig_obj, mats))
    detail_objects.extend(create_avatar_vfx_organs(rig_obj, mats))

    # V9.0.9: the Avatar Panel owns the runtime stage/grid.
    # Exporting a Blender stage inside the GLB caused the grid plane to appear
    # through Sarah's hips/torso after frontend fit/offset correction. Keep the
    # authoring stage in the .blend for reference only, but exclude it from the
    # AvatarPanel GLB runtime export.
    for stage_obj in create_avatar_micro_stage(mats):
        try:
            stage_obj["sarahmemory_export_role"] = "reference_only"
            stage_obj.hide_render = True
        except Exception:
            pass
    return detail_objects



def create_avatar_vfx_organs(rig_obj, mats):
    """Create lightweight visual-only VFX organ parts for the GLB runtime.

    These are not simulated particles and not execution systems. They are named
    visual organs that help AvatarPanel identify aura, voice, governance, shield,
    data-ribbon, and construct-orbit layers.
    """
    objs = []
    neon = mats.get('neon')
    soft = mats.get('neon_soft') or neon
    bloom = mats.get('hair_strand') or neon

    def ring_points(radius, z, y=0.0, steps=80):
        pts = []
        for idx in range(steps + 1):
            a = (math.pi * 2.0) * (idx / float(steps))
            pts.append((math.cos(a) * radius, y + math.sin(a) * radius, z))
        return pts

    try:
        objs.append(make_poly_curve('Sarah_VFX_GovernanceHalo_visual_only', ring_points(0.285, 1.990, y=-0.008, steps=96), soft, bevel_depth=0.0020, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_VFX_VoiceWaveChest_visual_only', ring_points(0.345, 1.310, y=-0.030, steps=96), neon, bevel_depth=0.0018, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_VFX_ConstructOrbitOuter_visual_only', ring_points(0.780, 0.012, y=0.0, steps=120), neon, bevel_depth=0.0016, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_VFX_ConstructOrbitInner_visual_only', ring_points(0.530, 0.020, y=0.0, steps=96), soft, bevel_depth=0.0012, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_VFX_DataRibbon_A_visual_only', [(-0.46, -0.020, 0.72), (-0.22, -0.140, 1.12), (0.06, -0.040, 1.50), (0.34, 0.080, 1.82)], neon, bevel_depth=0.0017, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        objs.append(make_poly_curve('Sarah_VFX_DataRibbon_B_visual_only', [(0.46, 0.020, 0.68), (0.22, 0.135, 1.10), (-0.08, 0.040, 1.52), (-0.34, -0.080, 1.84)], bloom, bevel_depth=0.0016, bevel_resolution=1, resolution_u=Q['curve_resolution']))
        for idx, a_deg in enumerate((0, 60, 120, 180, 240, 300)):
            a = math.radians(a_deg)
            objs.append(make_disc('Sarah_VFX_MicroSpark_%02d_visual_only' % idx, (math.cos(a) * 0.62, math.sin(a) * 0.62, 0.82 + (idx % 3) * 0.20), (0.012, 0.012, 0.012), soft, segments=10))
    except Exception as exc:
        log(f"VFX organ creation skipped: {exc}")

    for obj in objs:
        try:
            obj["sarahmemory_vfx_organ"] = True
            obj["sarahmemory_export_role"] = "runtime_vfx_visual_only"
            obj["execution_authority"] = False
            parent_keep_transform(obj, rig_obj)
        except Exception:
            pass
    return objs


def collect_runtime_asset_stats(objects):
    stats = {
        'mesh_objects': 0,
        'curve_objects': 0,
        'triangle_estimate': 0,
        'vertices': 0,
    }
    for obj in objects or []:
        if obj is None:
            continue
        if getattr(obj, 'type', '') == 'MESH':
            stats['mesh_objects'] += 1
            try:
                stats['vertices'] += len(obj.data.vertices)
                for poly in obj.data.polygons:
                    stats['triangle_estimate'] += max(1, len(poly.vertices) - 2)
            except Exception:
                pass
        elif getattr(obj, 'type', '') == 'CURVE':
            stats['curve_objects'] += 1
    stats['quality'] = QUALITY
    stats['runtime_texture_target'] = Q.get('runtime_texture_target', '1K-2K')
    stats['target_triangles'] = Q.get('target_triangles', '30K-60K')
    stats['max_runtime_fps'] = Q.get('max_runtime_fps', 24)
    return stats


def create_armature():
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    arm = bpy.context.active_object
    arm.name = RIG_NAME
    arm.data.name = f'{RIG_NAME}_Data'
    arm.show_in_front = True
    arm.data.display_type = 'STICK'

    eb = arm.data.edit_bones

    for b in list(eb):
        eb.remove(b)

    def add_bone(name, head, tail, parent=None, use_connect=False, roll=0.0, deform=True):
        bone = eb.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        bone.roll = roll
        bone.use_deform = deform
        if parent is not None:
            bone.parent = parent
            bone.use_connect = use_connect
        return bone

    root = add_bone('root', (0.0, 0.00, 0.00), (0.0, 0.00, 0.14), deform=False)
    pelvis = add_bone('pelvis', (0.0, 0.00, 0.92), (0.0, 0.00, 1.07), parent=root, deform=True)
    spine_01 = add_bone('spine.01', (0.0, 0.00, 1.07), (0.0, 0.00, 1.24), parent=pelvis, use_connect=True)
    spine_02 = add_bone('spine.02', (0.0, 0.00, 1.24), (0.0, 0.00, 1.41), parent=spine_01, use_connect=True)
    spine_03 = add_bone('spine.03', (0.0, 0.00, 1.41), (0.0, 0.00, 1.57), parent=spine_02, use_connect=True)
    neck = add_bone('neck', (0.0, 0.00, 1.57), (0.0, 0.00, 1.70), parent=spine_03, use_connect=True)
    head = add_bone('head', (0.0, 0.00, 1.70), (0.0, -0.02, 1.93), parent=neck, use_connect=True)
    jaw = add_bone('jaw', (0.0, -0.04, 1.76), (0.0, -0.12, 1.70), parent=head, deform=True)
    eye_l = add_bone('eye.L', (-0.055, -0.085, 1.80), (-0.055, -0.17, 1.80), parent=head, deform=True)
    eye_r = add_bone('eye.R', (0.055, -0.085, 1.80), (0.055, -0.17, 1.80), parent=head, deform=True)

    hair_root = add_bone('hair_root', (0.0, 0.04, 1.82), (0.0, 0.10, 1.69), parent=head, deform=True)
    hair_01 = add_bone('hair.01', (0.0, 0.10, 1.69), (0.0, 0.13, 1.42), parent=hair_root, use_connect=True, deform=True)
    hair_02 = add_bone('hair.02', (0.0, 0.13, 1.42), (0.0, 0.16, 1.12), parent=hair_01, use_connect=True, deform=True)
    hair_03 = add_bone('hair.03', (0.0, 0.16, 1.12), (0.0, 0.18, 0.84), parent=hair_02, use_connect=True, deform=True)

    for side, sign in (('L', -1.0), ('R', 1.0)):
        clav = add_bone(f'clavicle.{side}', (0.03 * sign, 0.00, 1.53), (0.17 * sign, 0.00, 1.51), parent=spine_03)
        upper = add_bone(f'upper_arm.{side}', (0.17 * sign, 0.00, 1.51), (0.55 * sign, 0.00, 1.48), parent=clav, use_connect=True)
        fore = add_bone(f'forearm.{side}', (0.55 * sign, 0.00, 1.48), (0.90 * sign, 0.00, 1.44), parent=upper, use_connect=True)
        hand = add_bone(f'hand.{side}', (0.90 * sign, 0.00, 1.44), (1.08 * sign, 0.00, 1.43), parent=fore, use_connect=True)

        thumb_01 = add_bone(f'thumb.01.{side}', (1.02 * sign, 0.02, 1.41), (1.10 * sign, 0.05, 1.38), parent=hand)
        thumb_02 = add_bone(f'thumb.02.{side}', (1.10 * sign, 0.05, 1.38), (1.17 * sign, 0.08, 1.35), parent=thumb_01, use_connect=True)
        thumb_03 = add_bone(f'thumb.03.{side}', (1.17 * sign, 0.08, 1.35), (1.24 * sign, 0.11, 1.33), parent=thumb_02, use_connect=True)

        finger_rows = [
            ('index', -0.018, 1.44, 1.17, 1.29, 1.40),
            ('middle', -0.006, 1.425, 1.18, 1.31, 1.44),
            ('ring', 0.006, 1.405, 1.17, 1.29, 1.40),
            ('pinky', 0.018, 1.385, 1.15, 1.24, 1.31),
        ]
        for fname, yoff, zbase, p0, p1, p2 in finger_rows:
            b1 = add_bone(f'{fname}.01.{side}', (1.07 * sign, yoff, zbase), (p0 * sign, yoff, zbase), parent=hand)
            b2 = add_bone(f'{fname}.02.{side}', (p0 * sign, yoff, zbase), (p1 * sign, yoff, zbase), parent=b1, use_connect=True)
            b3 = add_bone(f'{fname}.03.{side}', (p1 * sign, yoff, zbase), (p2 * sign, yoff, zbase), parent=b2, use_connect=True)

        thigh = add_bone(f'thigh.{side}', (0.11 * sign, 0.00, 0.97), (0.12 * sign, 0.00, 0.55), parent=pelvis)
        shin = add_bone(f'shin.{side}', (0.12 * sign, 0.00, 0.55), (0.10 * sign, 0.00, 0.14), parent=thigh, use_connect=True)
        foot = add_bone(f'foot.{side}', (0.10 * sign, 0.00, 0.14), (0.10 * sign, -0.12, 0.05), parent=shin, use_connect=True)
        toe = add_bone(f'toe.{side}', (0.10 * sign, -0.12, 0.05), (0.10 * sign, -0.24, 0.05), parent=foot, use_connect=True)

        hand_ik = add_bone(f'hand_ik.{side}', (1.10 * sign, 0.00, 1.43), (1.10 * sign, -0.16, 1.43), parent=root, deform=False)
        elbow_pole = add_bone(f'elbow_pole.{side}', (0.60 * sign, -0.45, 1.43), (0.60 * sign, -0.60, 1.43), parent=root, deform=False)
        foot_ik = add_bone(f'foot_ik.{side}', (0.10 * sign, -0.02, 0.05), (0.10 * sign, -0.18, 0.05), parent=root, deform=False)
        knee_pole = add_bone(f'knee_pole.{side}', (0.12 * sign, -0.42, 0.55), (0.12 * sign, -0.57, 0.55), parent=root, deform=False)

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def configure_rig_controls(arm):
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    arm['CTRL_Blink_L'] = 0.0
    arm['CTRL_Blink_R'] = 0.0
    arm['CTRL_JawOpen'] = 0.0
    arm['CTRL_Smile'] = 0.0
    arm['CTRL_Frown'] = 0.0
    arm['CTRL_BrowsUp'] = 0.0

    for prop_name, description in (
        ('CTRL_Blink_L', 'Left eyelid blink'),
        ('CTRL_Blink_R', 'Right eyelid blink'),
        ('CTRL_JawOpen', 'Jaw open driver'),
        ('CTRL_Smile', 'Smile driver'),
        ('CTRL_Frown', 'Frown driver'),
        ('CTRL_BrowsUp', 'Brows up driver'),
    ):
        try:
            ui = arm.id_properties_ui(prop_name)
            ui.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, description=description)
        except Exception:
            pass

    for side in ('L', 'R'):
        fore = arm.pose.bones.get(f'forearm.{side}')
        hand_ik = arm.pose.bones.get(f'hand_ik.{side}')
        elbow_pole = arm.pose.bones.get(f'elbow_pole.{side}')
        if fore and hand_ik:
            con = fore.constraints.new(type='IK')
            con.name = f'IK_Arm_{side}'
            con.target = arm
            con.subtarget = hand_ik.name
            con.chain_count = 2
            if elbow_pole:
                con.pole_target = arm
                con.pole_subtarget = elbow_pole.name
                con.pole_angle = radians(180.0 if side == 'L' else 0.0)

        shin = arm.pose.bones.get(f'shin.{side}')
        foot_ik = arm.pose.bones.get(f'foot_ik.{side}')
        knee_pole = arm.pose.bones.get(f'knee_pole.{side}')
        if shin and foot_ik:
            con = shin.constraints.new(type='IK')
            con.name = f'IK_Leg_{side}'
            con.target = arm
            con.subtarget = foot_ik.name
            con.chain_count = 2
            if knee_pole:
                con.pole_target = arm
                con.pole_subtarget = knee_pole.name
                con.pole_angle = radians(-90.0 if side == 'L' else 90.0)

    for bname in ('hand_ik.L', 'hand_ik.R', 'foot_ik.L', 'foot_ik.R', 'elbow_pole.L', 'elbow_pole.R', 'knee_pole.L', 'knee_pole.R'):
        pb = arm.pose.bones.get(bname)
        if pb:
            pb.custom_shape_scale_xyz = (1.18, 1.18, 1.18)
            pb.bone.show_wire = True
    bpy.ops.object.mode_set(mode='OBJECT')


def parent_mesh_to_armature(mesh_obj, armature_obj):
    if mesh_obj is None or armature_obj is None:
        return False
    try:
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        return True
    except Exception as exc:
        log(f"Parent auto weights warning for {mesh_obj.name}: {exc}")
        try:
            bpy.ops.object.parent_set(type='ARMATURE')
            mod = mesh_obj.modifiers.get('Armature')
            if mod is None:
                mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
                mod.object = armature_obj
            return False
        except Exception as exc2:
            log(f"Parent fallback failed for {mesh_obj.name}: {exc2}")
            return False


def set_head_vertex_groups(head_obj):
    if head_obj is None or getattr(head_obj, 'type', '') != 'MESH':
        return

    vg_head = head_obj.vertex_groups.get('head') or head_obj.vertex_groups.new(name='head')
    vg_jaw = head_obj.vertex_groups.get('jaw') or head_obj.vertex_groups.new(name='jaw')
    vg_neck = head_obj.vertex_groups.get('neck') or head_obj.vertex_groups.new(name='neck')

    for v in head_obj.data.vertices:
        co = v.co
        head_w = 1.0
        jaw_w = 0.0
        neck_w = 0.0
        if co.z < -0.08:
            neck_w = 0.85
            head_w = 0.35
        elif co.z < -0.02:
            neck_w = 0.25
            head_w = 0.85
        if co.z < -0.015 and co.y < 0.03:
            jaw_w = 0.72
        vg_head.add([v.index], head_w, 'REPLACE')
        vg_jaw.add([v.index], jaw_w, 'REPLACE')
        vg_neck.add([v.index], neck_w, 'REPLACE')


def set_hair_vertex_groups(hair_obj):
    if hair_obj is None or getattr(hair_obj, 'type', '') != 'MESH':
        return

    names = ['head', 'hair_root', 'hair.01', 'hair.02', 'hair.03']
    groups = {name: hair_obj.vertex_groups.get(name) or hair_obj.vertex_groups.new(name=name) for name in names}

    for v in hair_obj.data.vertices:
        z = v.co.z
        y = v.co.y
        groups['head'].add([v.index], 1.0 if z > 0.10 else 0.35, 'REPLACE')
        groups['hair_root'].add([v.index], 1.0 if z > -0.02 else 0.15, 'REPLACE')
        groups['hair.01'].add([v.index], max(0.0, min(1.0, 0.85 - max(0.0, (z - 0.05)) * 0.8 + max(0.0, y) * 0.5)), 'REPLACE')
        groups['hair.02'].add([v.index], max(0.0, min(1.0, 0.85 - max(0.0, z + 0.20) * 1.0 + max(0.0, y - 0.04) * 0.7)), 'REPLACE')
        groups['hair.03'].add([v.index], max(0.0, min(1.0, 1.15 - max(0.0, z + 0.48) * 1.4 + max(0.0, y - 0.08) * 0.9)), 'REPLACE')


def add_head_shape_keys(head_obj, armature_obj):
    if head_obj is None or getattr(head_obj, 'type', '') != 'MESH':
        return []

    bpy.context.view_layer.objects.active = head_obj
    bpy.ops.object.shape_key_add(from_mix=False)
    head_obj.active_shape_key.name = 'Basis'

    key_names = ['Blink_L', 'Blink_R', 'BlinkBoth', 'JawOpen', 'Smile', 'Frown', 'BrowsUp', 'Squint_L', 'Squint_R', 'MouthWide', 'MouthNarrow']
    for key_name in key_names:
        bpy.ops.object.shape_key_add(from_mix=False)
        head_obj.active_shape_key.name = key_name

    keys = head_obj.data.shape_keys.key_blocks
    basis = keys['Basis']

    for idx, vert in enumerate(head_obj.data.vertices):
        base = basis.data[idx].co.copy()
        x = base.x
        y = base.y
        z = base.z

        blink_l = base.copy()
        blink_r = base.copy()
        jaw_open = base.copy()
        smile = base.copy()
        frown = base.copy()
        brows_up = base.copy()

        left_eye = x < -0.015 and y < -0.050 and 0.005 < z < 0.120
        right_eye = x > 0.015 and y < -0.050 and 0.005 < z < 0.120
        if left_eye:
            blink_l.z *= 0.55
            blink_l.y += 0.010
        if right_eye:
            blink_r.z *= 0.55
            blink_r.y += 0.010

        mouth_zone = y < -0.040 and -0.070 < z < 0.010
        lower_face = z < -0.010 and y < 0.03
        if lower_face:
            jaw_open.z -= 0.030 * (1.0 - min(1.0, abs(x) * 5.0))
            jaw_open.y -= 0.020
        if mouth_zone:
            smile.z += 0.012 * (0.40 + abs(x) * 10.0)
            smile.x += 0.008 * (1.0 if x > 0.0 else -1.0)
            smile.y -= 0.006
            frown.z -= 0.012 * (0.40 + abs(x) * 10.0)
            frown.x -= 0.004 * (1.0 if x > 0.0 else -1.0)
            frown.y += 0.003

        brow_zone = y < -0.030 and 0.065 < z < 0.145 and abs(x) < 0.11
        if brow_zone:
            brows_up.z += 0.018 * (1.0 - min(1.0, abs(x) * 4.0))
            brows_up.y -= 0.003

        keys['Blink_L'].data[idx].co = blink_l
        keys['Blink_R'].data[idx].co = blink_r
        if 'BlinkBoth' in keys:
            both = blink_l.copy() if x < 0.0 else blink_r.copy()
            keys['BlinkBoth'].data[idx].co = both
        keys['JawOpen'].data[idx].co = jaw_open
        keys['Smile'].data[idx].co = smile
        keys['Frown'].data[idx].co = frown
        keys['BrowsUp'].data[idx].co = brows_up
        if 'Squint_L' in keys:
            sq = base.copy()
            if left_eye:
                sq.z *= 0.82
                sq.y += 0.004
            keys['Squint_L'].data[idx].co = sq
        if 'Squint_R' in keys:
            sq = base.copy()
            if right_eye:
                sq.z *= 0.82
                sq.y += 0.004
            keys['Squint_R'].data[idx].co = sq
        if 'MouthWide' in keys:
            mw = base.copy()
            if mouth_zone:
                mw.x += 0.018 * (1.0 if x > 0.0 else -1.0)
                mw.y -= 0.004
            keys['MouthWide'].data[idx].co = mw
        if 'MouthNarrow' in keys:
            mn = base.copy()
            if mouth_zone:
                mn.x *= 0.88
                mn.y += 0.004
            keys['MouthNarrow'].data[idx].co = mn

    prop_map = {
        'Blink_L': 'CTRL_Blink_L',
        'Blink_R': 'CTRL_Blink_R',
        'JawOpen': 'CTRL_JawOpen',
        'Smile': 'CTRL_Smile',
        'Frown': 'CTRL_Frown',
        'BrowsUp': 'CTRL_BrowsUp',
        'BlinkBoth': 'CTRL_BlinkBoth',
        'Squint_L': 'CTRL_Squint_L',
        'Squint_R': 'CTRL_Squint_R',
        'MouthWide': 'CTRL_MouthWide',
        'MouthNarrow': 'CTRL_MouthNarrow',
    }

    for key_name, prop_name in prop_map.items():
        try:
            sk = keys[key_name]
            drv = sk.driver_add('value').driver
            drv.type = 'AVERAGE'
            var = drv.variables.new()
            var.name = 'ctrl'
            var.type = 'SINGLE_PROP'
            target = var.targets[0]
            target.id_type = 'OBJECT'
            target.id = armature_obj
            target.data_path = f'["{prop_name}"]'
        except Exception as exc:
            log(f"Driver warning for {key_name}: {exc}")
    return key_names


def create_demo_action(arm):
    if arm is None:
        return None
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    action = bpy.data.actions.new(name='Sarah_Avatar_Demo')
    arm.animation_data_create()
    arm.animation_data.action = action

    p = arm.pose.bones
    hand_ik_r = p.get('hand_ik.R')
    hand_ik_l = p.get('hand_ik.L')
    head = p.get('head')
    hair1 = p.get('hair.01')
    hair2 = p.get('hair.02')
    hair3 = p.get('hair.03')

    def kf(obj, path, frame):
        try:
            obj.keyframe_insert(data_path=path, frame=frame)
        except Exception:
            pass

    if hand_ik_r:
        hand_ik_r.location = Vector((0.0, 0.0, 0.0))
        kf(hand_ik_r, 'location', 1)
        hand_ik_r.location = Vector((0.0, -0.20, 0.18))
        kf(hand_ik_r, 'location', 24)
        hand_ik_r.location = Vector((0.0, -0.08, 0.10))
        kf(hand_ik_r, 'location', 48)
        hand_ik_r.location = Vector((0.0, 0.0, 0.0))
        kf(hand_ik_r, 'location', 72)

    if hand_ik_l:
        hand_ik_l.location = Vector((0.0, 0.0, 0.0))
        kf(hand_ik_l, 'location', 1)
        hand_ik_l.location = Vector((0.0, 0.05, -0.02))
        kf(hand_ik_l, 'location', 36)
        hand_ik_l.location = Vector((0.0, 0.0, 0.0))
        kf(hand_ik_l, 'location', 72)

    if head:
        head.rotation_mode = 'XYZ'
        head.rotation_euler = (0.0, 0.0, 0.0)
        kf(head, 'rotation_euler', 1)
        head.rotation_euler = (radians(2.0), radians(-4.0), radians(0.0))
        kf(head, 'rotation_euler', 24)
        head.rotation_euler = (radians(-1.0), radians(5.0), radians(0.0))
        kf(head, 'rotation_euler', 48)
        head.rotation_euler = (0.0, 0.0, 0.0)
        kf(head, 'rotation_euler', 72)

    for bone, values in (
        (hair1, ((0.0, 0.0, 0.0), (radians(2), radians(0), radians(1)), (radians(-2), radians(0), radians(-1)), (0.0, 0.0, 0.0))),
        (hair2, ((0.0, 0.0, 0.0), (radians(4), radians(0), radians(1)), (radians(-3), radians(0), radians(-1)), (0.0, 0.0, 0.0))),
        (hair3, ((0.0, 0.0, 0.0), (radians(5), radians(0), radians(2)), (radians(-4), radians(0), radians(-2)), (0.0, 0.0, 0.0))),
    ):
        if bone:
            bone.rotation_mode = 'XYZ'
            for frame, rot in zip((1, 24, 48, 72), values):
                bone.rotation_euler = rot
                kf(bone, 'rotation_euler', frame)

    for frame, vals in (
        (1,  (0.0, 0.0, 0.0)),
        (12, (1.0, 0.0, 0.25)),
        (24, (0.0, 0.0, 0.85)),
        (36, (0.0, 1.0, 0.0)),
        (48, (0.0, 0.0, 0.0)),
        (60, (0.0, 0.0, 0.0)),
        (72, (0.0, 0.0, 0.0)),
    ):
        arm['CTRL_Blink_L'], arm['CTRL_Blink_R'], arm['CTRL_Smile'] = vals
        arm['CTRL_JawOpen'] = 0.35 if frame in (24, 36) else 0.0
        arm['CTRL_Frown'] = 0.20 if frame == 60 else 0.0
        arm['CTRL_BrowsUp'] = 0.20 if frame in (24, 36) else 0.0
        for prop in ('["CTRL_Blink_L"]', '["CTRL_Blink_R"]', '["CTRL_Smile"]', '["CTRL_JawOpen"]', '["CTRL_Frown"]', '["CTRL_BrowsUp"]'):
            try:
                arm.keyframe_insert(data_path=prop, frame=frame)
            except Exception:
                pass

    bpy.ops.object.mode_set(mode='OBJECT')
    return action



def create_avatar_eye_camera_anchors(rig_obj):
    """Create named empty anchors for Avatar-Eye VR / first-person runtime."""
    anchors = []
    try:
        anchor_specs = [
            ('Sarah_AvatarEye_Center', (0.0, -0.155, 1.805)),
            ('Sarah_AvatarEye_L', (-0.055, -0.155, 1.805)),
            ('Sarah_AvatarEye_R', (0.055, -0.155, 1.805)),
            ('Sarah_Expression_Controller', (0.0, -0.190, 1.835)),
        ]
        for name, loc in anchor_specs:
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.035
            empty.location = loc
            empty['sarahmemory_scene_graph_role'] = 'avatar_eye_camera_anchor' if 'Eye' in name else 'expression_controller'
            empty['execution_authority'] = False
            bpy.context.scene.collection.objects.link(empty)
            parent_keep_transform(empty, rig_obj)
            anchors.append(empty)
    except Exception as exc:
        log(f"Avatar-eye anchor warning: {exc}")
    return anchors


def create_goldstandard_entity_actions(arm):
    """Export true AvatarPanel animation clips as GLB actions.

    Actions are visual-only embodied expression clips. They do not authorize
    robotics, MSDC, files, networking, or external execution.
    """
    if arm is None:
        return []
    created = []
    try:
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode='POSE')
        p = arm.pose.bones
        head = p.get('head')
        neck = p.get('neck')
        spine01 = p.get('spine.01')
        spine02 = p.get('spine.02')
        spine03 = p.get('spine.03')
        hair1 = p.get('hair.01')
        hair2 = p.get('hair.02')
        hair3 = p.get('hair.03')
        arm_r = p.get('upper_arm.R')
        fore_r = p.get('forearm.R')
        arm_l = p.get('upper_arm.L')
        fore_l = p.get('forearm.L')

        def new_action(name):
            act = bpy.data.actions.new(name=name)
            arm.animation_data_create()
            arm.animation_data.action = act
            created.append(act)
            return act

        def kf(obj, path, frame):
            try:
                obj.keyframe_insert(data_path=path, frame=frame)
            except Exception:
                pass

        def kprop(prop, frame, value):
            try:
                arm[prop] = float(value)
                arm.keyframe_insert(data_path=f'["{prop}"]', frame=frame)
            except Exception:
                pass

        # Idle breathing / head / torso / hair coupling.
        new_action('Sarah_Embodied_Idle_96f')
        for frame, phase in ((1, 0.0), (24, 0.45), (48, 1.0), (72, 0.45), (96, 0.0)):
            if spine01:
                spine01.rotation_mode = 'XYZ'; spine01.rotation_euler = (radians(phase * 1.2), 0.0, radians(math.sin(frame * 0.05) * 0.5)); kf(spine01, 'rotation_euler', frame)
            if spine02:
                spine02.rotation_mode = 'XYZ'; spine02.rotation_euler = (radians(phase * 0.8), 0.0, radians(math.sin(frame * 0.05 + 0.5) * 0.35)); kf(spine02, 'rotation_euler', frame)
            if spine03:
                spine03.rotation_mode = 'XYZ'; spine03.rotation_euler = (radians(phase * 0.55), radians(math.sin(frame * 0.04) * 0.45), 0.0); kf(spine03, 'rotation_euler', frame)
            if neck:
                neck.rotation_mode = 'XYZ'; neck.rotation_euler = (radians(math.sin(frame * 0.06) * 0.45), radians(math.sin(frame * 0.04) * 0.75), 0.0); kf(neck, 'rotation_euler', frame)
            if head:
                head.rotation_mode = 'XYZ'; head.rotation_euler = (radians(math.sin(frame * 0.07) * 0.9), radians(math.sin(frame * 0.05) * 1.4), radians(math.sin(frame * 0.035) * 0.35)); kf(head, 'rotation_euler', frame)
            for bone, amp in ((hair1, 1.8), (hair2, 3.2), (hair3, 4.6)):
                if bone:
                    bone.rotation_mode = 'XYZ'; bone.rotation_euler = (radians(math.sin(frame * 0.07) * amp * 0.35), radians(math.sin(frame * 0.03) * amp * 0.22), radians(math.sin(frame * 0.06) * amp)); kf(bone, 'rotation_euler', frame)
            kprop('CTRL_BlinkBoth', frame, 0.0)
            kprop('CTRL_Smile', frame, 0.08)

        # Blink loop clip.
        new_action('Sarah_Natural_Blink_48f')
        for frame, val in ((1, 0.0), (18, 0.0), (21, 1.0), (24, 0.0), (42, 0.0), (45, 0.85), (48, 0.0)):
            kprop('CTRL_Blink_L', frame, val * 0.96)
            kprop('CTRL_Blink_R', frame, val)
            kprop('CTRL_BlinkBoth', frame, val * 0.5)

        # Hair flow loop clip.
        new_action('Sarah_Hair_Flow_120f')
        for frame, sign in ((1, 0.0), (30, 1.0), (60, -0.7), (90, 0.8), (120, 0.0)):
            for bone, amp in ((hair1, 3.0), (hair2, 5.0), (hair3, 7.0)):
                if bone:
                    bone.rotation_mode = 'XYZ'; bone.rotation_euler = (radians(sign * amp * 0.28), radians(sign * amp * 0.12), radians(sign * amp)); kf(bone, 'rotation_euler', frame)

        # Friendly wave / greeting clip.
        new_action('Sarah_Friendly_Wave_96f')
        for frame, angle in ((1, 0.0), (18, -28.0), (36, -52.0), (54, -35.0), (72, -55.0), (96, 0.0)):
            if arm_r:
                arm_r.rotation_mode = 'XYZ'; arm_r.rotation_euler = (radians(-8.0), radians(0.0), radians(angle)); kf(arm_r, 'rotation_euler', frame)
            if fore_r:
                fore_r.rotation_mode = 'XYZ'; fore_r.rotation_euler = (radians(0.0), radians(0.0), radians(-18.0 + math.sin(frame * 0.3) * 10.0)); kf(fore_r, 'rotation_euler', frame)
            if arm_l:
                arm_l.rotation_mode = 'XYZ'; arm_l.rotation_euler = (radians(0.0), radians(0.0), radians(3.0)); kf(arm_l, 'rotation_euler', frame)
            if fore_l:
                fore_l.rotation_mode = 'XYZ'; fore_l.rotation_euler = (0.0, 0.0, 0.0); kf(fore_l, 'rotation_euler', frame)
            kprop('CTRL_Smile', frame, 0.55 if 18 <= frame <= 72 else 0.08)

        bpy.ops.object.mode_set(mode='OBJECT')
        arm['sarahmemory_animation_clips'] = [a.name for a in created]
        arm['sarahmemory_animation_authority'] = 'visual_only_no_msdc_no_operator_action'
    except Exception as exc:
        log(f"GoldStandard entity action warning: {exc}")
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return created

def create_lights_and_camera(rig_obj):
    bpy.ops.object.light_add(type='AREA', location=(2.45, -2.85, 2.60))
    key = bpy.context.active_object
    key.name = 'Sarah_KeyLight'
    key.data.energy = 4300
    key.data.shape = 'RECTANGLE'
    key.data.size = 3.3
    key.data.size_y = 2.4

    bpy.ops.object.light_add(type='AREA', location=(-2.55, -1.25, 1.95))
    fill = bpy.context.active_object
    fill.name = 'Sarah_FillLight'
    fill.data.energy = 2100
    fill.data.shape = 'RECTANGLE'
    fill.data.size = 2.2
    fill.data.size_y = 1.6
    fill.data.color = (0.58, 0.70, 1.00)

    bpy.ops.object.light_add(type='POINT', location=(0.0, 2.20, 2.25))
    rim = bpy.context.active_object
    rim.name = 'Sarah_RimLight'
    rim.data.energy = 1500
    rim.data.color = (1.0, 0.28, 0.62)

    bpy.ops.object.camera_add(location=(0.0, -4.15, 1.54), rotation=(radians(83.2), 0.0, 0.0))
    cam = bpy.context.active_object
    cam.name = 'Sarah_Camera'
    cam.data.lens = 58
    cam.data.clip_start = 0.01
    cam.data.clip_end = 250.0
    bpy.context.scene.camera = cam

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, -0.02, 1.46))
    tgt = bpy.context.active_object
    tgt.name = 'Sarah_CameraTarget'
    if rig_obj:
        tgt.parent = rig_obj
    con = cam.constraints.new(type='TRACK_TO')
    con.target = tgt
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    return cam



def _runtime_object_list(objects):
    """Return de-duplicated runtime objects eligible for GLB/FBX export."""
    out = []
    seen = set()
    for obj in objects or []:
        if obj is None:
            continue
        try:
            if obj.name in seen:
                continue
            role = str(obj.get("sarahmemory_export_role", "runtime")) if hasattr(obj, "get") else "runtime"
            if role in {"reference_only", "guide_only", "construction_only"}:
                continue
            # Image empties/reference planes are authoring guides, not AvatarPanel runtime meshes.
            if getattr(obj, "type", "") == "EMPTY" and str(getattr(obj, "empty_display_type", "")).upper() == "IMAGE":
                continue
            out.append(obj)
            seen.add(obj.name)
        except Exception:
            pass
    return out


def _tag_runtime_parts(objects):
    """Attach stable appearance-slot metadata to modular runtime objects."""
    for obj in objects or []:
        if obj is None:
            continue
        name = str(getattr(obj, "name", ""))
        slot = "runtime_detail"
        if name == "Sarah_Rig":
            slot = "armature"
        elif "Body" in name:
            slot = "base_body_suit_surface"
        elif "Boot" in name:
            slot = "boots"
        elif "Hair_Strands" in name or "Hair" in name:
            slot = "hair"
        elif "Eye" in name or "Iris" in name:
            slot = "eyes"
        elif "Lash" in name:
            slot = "eyelashes"
        elif "Brow" in name:
            slot = "eyebrows"
        elif "Lip" in name or "Mouth" in name or "Tooth" in name or "Nose" in name:
            slot = "facial_details"
        elif "Neon" in name:
            slot = "emissive_suit_trim"
        elif "Stage" in name:
            slot = "avatar_panel_stage"
        try:
            obj["sarahmemory_export_role"] = "runtime"
            obj["sarahmemory_avatar_slot"] = slot
        except Exception:
            pass


def _prepare_clean_runtime_export(objects):
    """Select only AvatarPanel runtime objects for export.

    The .blend keeps references and guides for later editing, but the GLB must
    contain exactly one Sarah runtime rig with modular appearance parts.  This
    prevents source_normalized/front-side-back image planes or earlier helper
    bodies from appearing as a second ghost/overlay avatar in the WebUI.
    """
    runtime_objects = _runtime_object_list(objects)
    _tag_runtime_parts(runtime_objects)

    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass

    for obj in bpy.context.scene.objects:
        try:
            role = str(obj.get("sarahmemory_export_role", "runtime")) if hasattr(obj, "get") else "runtime"
            if role in {"reference_only", "guide_only", "construction_only"}:
                obj.select_set(False)
                obj.hide_render = True
        except Exception:
            pass

    selected = []
    for obj in runtime_objects:
        try:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.select_set(True)
            selected.append(obj.name)
        except Exception:
            pass
    if runtime_objects:
        try:
            bpy.context.view_layer.objects.active = runtime_objects[0]
        except Exception:
            pass
    return runtime_objects, selected


def _appearance_slot_contract():
    """Game/RPG-style modular appearance doctrine for future customization."""
    return {
        'architecture': 'single_armature_multiple_interchangeable_mesh_parts',
        'default_runtime_rule': 'load one clean GLB into AvatarPanel; no reference planes or helper bodies',
        'current_slots': {
            'base_body_suit_surface': {'required': True, 'bound_to': 'Sarah_Rig', 'swap_ready': False},
            'hair': {'required': True, 'bound_to': 'head/hair bones', 'swap_ready': True, 'future_controls': ['style', 'length', 'color', 'physics_lod']},
            'eyes': {'required': True, 'bound_to': 'eye bones', 'swap_ready': True, 'future_controls': ['iris_color', 'blink', 'look_at']},
            'eyelashes': {'required': False, 'bound_to': 'head', 'swap_ready': True},
            'facial_details': {'required': True, 'bound_to': 'head/jaw/morph_targets', 'swap_ready': True},
            'boots': {'required': True, 'bound_to': 'foot/toe bones', 'swap_ready': True},
            'emissive_suit_trim': {'required': True, 'bound_to': 'body surface paths', 'swap_ready': True, 'future_controls': ['color', 'intensity', 'pattern']},
            'vfx_organs': {'required': False, 'bound_to': 'AvatarPanel visual runtime', 'swap_ready': True, 'future_controls': ['quality', 'intensity', 'emotion_palette', 'voice_wave']},
            'gloves': {'required': False, 'bound_to': 'hand/finger bones', 'swap_ready': 'future'},
            'shirt': {'required': False, 'bound_to': 'torso/arm bones', 'swap_ready': 'future'},
            'pants': {'required': False, 'bound_to': 'pelvis/leg bones', 'swap_ready': 'future'},
            'one_piece_suit': {'required': True, 'bound_to': 'body mesh/material regions', 'swap_ready': True},
        },
        'future_avatar_layout': {
            'base': 'resources/avatars/3D',
            'variants': 'resources/avatars/3D/<name>',
            'parts_library': 'resources/avatars/3D/parts',
        },
        'physical_robot_note': 'future AvatarToMSDCBridge required; this bootstrap is visual software embodiment only',
    }

def save_and_export(all_objects, rig_obj, shape_keys):
    out_dir = os.path.dirname(OUT_BLEND)
    os.makedirs(out_dir, exist_ok=True)

    runtime_objects, selected_runtime_names = _prepare_clean_runtime_export(all_objects)

    # Save full authoring scene with references retained but marked non-export.
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

    glb_ok = False
    fbx_ok = False

    try:
        bpy.ops.export_scene.gltf(
            filepath=OUT_GLB,
            export_format='GLB',
            export_apply=True,
            export_texcoords=True,
            export_normals=True,
            export_tangents=True,
            export_materials='EXPORT',
            export_yup=True,
            export_animations=True,
            export_animation_mode='ACTIONS',
            export_skins=True,
            export_morph=True,
            use_selection=True,
        )
        glb_ok = True
    except Exception as exc:
        log(f"GLB export failed: {exc}")

    if EXPORT_FBX_ENABLED:
        try:
            bpy.ops.export_scene.fbx(
                filepath=OUT_FBX,
                use_selection=True,
                apply_unit_scale=True,
                bake_space_transform=False,
                object_types={'ARMATURE', 'MESH', 'EMPTY', 'LIGHT', 'CAMERA'},
                use_armature_deform_only=False,
                add_leaf_bones=False,
                path_mode='AUTO',
                bake_anim=True,
            )
            fbx_ok = True
        except Exception as exc:
            log(f"FBX export failed: {exc}")
    else:
        log('FBX export skipped by request; GLB remains the primary Avatar Panel runtime asset.')

    bone_names = []
    try:
        bone_names = [b.name for b in rig_obj.data.bones]
    except Exception:
        bone_names = []

    runtime_stats = collect_runtime_asset_stats(runtime_objects)
    action_names = []
    try:
        action_names = sorted(action.name for action in bpy.data.actions)
    except Exception:
        action_names = []

    manifest = {
        'ok': True,
        'module': 'SarahMemoryBlenderAvatarBootstrap',
        'version': '9.0.15-goldstandard-embodied-entity',
        'quality': QUALITY,
        'quality_preset': Q,
        'authoring_triangle_target': Q.get('target_triangles', ''),
        'runtime_lod_policy': 'goldstandard_entity/cinematic_2m are high-definition embodied entity authoring lanes; high/balanced remain fallback LODs for weak machines',
        'character_name': CHARACTER_NAME,
        'body_profile': {
            'profile_name': 'SarahMemory_default_humanoid',
            'height_m': 1.68,
            'weight_kg_visual_target': 58.0,
            'head_height_ratio': 0.132,
            'shoulder_width_ratio': 0.245,
            'hip_width_ratio': 0.190,
            'leg_length_ratio': 0.515,
            'arm_span_ratio': 1.0,
            'center_of_mass_y_ratio': 0.55,
            'rig_units': 'meters',
            'biological_plausibility': 'visual_anatomy_reference_only',
            'future_robot_mapping': 'requires AvatarToMSDCBridge; no physical actuation in this file',
        },
        'output_contract': 'BASE_DIR/resources/avatars/3D',
        'runtime_export_mode': 'selected_runtime_objects_only',
        'surface_binding_mode': 'shader_assisted_trim_shrinkwrapped_to_runtime_mesh',
        'stage_export_mode': 'avatar_panel_owns_stage_grid_not_glb',
        'vfx_organ_contract': {
            'contract': 'SARAHMEMORY_AVATAR_VFX_ORGANS_V1',
            'enabled': True,
            'quality': QUALITY,
            'visual_only': True,
            'execution_authority': False,
            'organs': ['aura', 'neuralPulse', 'voiceWave', 'constructGrid', 'governanceHalo', 'shieldShell', 'emotionBloom', 'dataRibbons', 'constructOrbit', 'particles'],
        },
        'shader_profile': {
            'profile_name': 'SarahMemory_shader_bound_black_suit_v1',
            'load_budget_seconds': int(Q.get('load_budget_seconds', 10)),
            'texture_strategy': 'PBR_GLTF_safe_materials plus cinematic material metadata; runtime remains WebGL/GLB first',
            'suit_base': 'black_gloss_clearcoat',
            'trim_strategy': 'thin surface-bound emissive geometry plus Avatar3D material enhancement',
            'future_strategy': 'baked UV emissive masks once stable humanoid UVs exist',
        },
        'runtime_export_selected_objects': selected_runtime_names,
        'appearance_slot_contract': _appearance_slot_contract(),
        'blend': OUT_BLEND,
        'glb': OUT_GLB if glb_ok else '',
        'fbx': OUT_FBX if fbx_ok else '',
        'front_reference': FRONT_IMAGE if os.path.isfile(FRONT_IMAGE) else '',
        'side_reference': SIDE_IMAGE if os.path.isfile(SIDE_IMAGE) else '',
        'back_reference': BACK_IMAGE if os.path.isfile(BACK_IMAGE) else '',
        'blueprint_reference': BLUEPRINT_IMAGE if os.path.isfile(BLUEPRINT_IMAGE) else '',
        'engine': bpy.context.scene.render.engine,
        'frame_range': [bpy.context.scene.frame_start, bpy.context.scene.frame_end],
        'armature_name': rig_obj.name if rig_obj else '',
        'bone_count': len(bone_names),
        'bones': bone_names,
        'shape_keys': list(shape_keys or []),
        'actions': action_names,
        'objects': [obj.name for obj in runtime_objects],
        'excluded_from_runtime_export': ['SarahMemory_References', 'source_normalized image planes', 'guide_only', 'construction_only', 'Blender micro-stage/grid'],
        'runtime_asset_stats': runtime_stats,
        'avatar_panel_contract': {
            'primary_runtime_asset': OUT_GLB if glb_ok else '',
            'fallback_required': True,
            'recommended_loader_state': '3D_LOADING -> 3D_READY -> 3D_ACTIVE',
            'recommended_fps_cap': Q.get('max_runtime_fps', 24),
            'vfx_organs_ready': True,
            'recommended_texture_target': Q.get('runtime_texture_target', '780p-class procedural/PBR'),
            'shader_detail_level': Q.get('shader_detail_level', '780p_shader_max_balanced'),
            'avatar_panel_load_budget_seconds': Q.get('load_budget_seconds', 0),
            'goldstandard_embodied_entity': bool(Q.get('goldstandard_entity')),
            'authoring_triangle_target': Q.get('target_triangles'),
            'notes': [
                'Visual-only VFX organs are exported as named lightweight runtime parts; AvatarPanel owns live pulse animation.',
                'Humanoid proportions are visual/rigging targets for the software Avatar Organ, not physical robot authority.',
                'GLB export uses selected runtime objects only so reference/source images cannot appear as ghost overlay bodies.',
                'Cyan suit/outfit detail is now 780p shader-max assisted, shrinkwrapped at sub-millimeter offsets so it reads as embedded suit glow, not floating rails.',
                'The live AvatarPanel stage/grid is now rendered by the frontend, not exported inside the GLB, so the grid cannot pass through the body after fit/offset transforms.',
                'Appearance parts are modular mesh/material slots bound to one Sarah_Rig armature; neon, face, and hair detail layers receive explicit runtime armature weights.',
                'v9.0.5 uses a ring-based humanoid body mesh instead of primitive capsule stacking for less toy-like proportions.',
                'Use GLB in the V9 Avatar Panel micro game-engine viewport.',
                'v9.0.15 exports Avatar-Eye camera anchors, expression controller metadata, and visual-only action clips for embodied runtime.',
                'Do not run Blender as the live renderer.',
                'Load this asset lazily only when Avatar 3D mode is selected.',
                'Keep 2D avatar fallback active if GLB/WebGL initialization fails.',
            ],
        },
    }

    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)


def main():
    log(f"Starting SarahMemory avatar bootstrap quality={QUALITY}")
    clean_scene()
    scene_setup()
    log("Scene setup complete")

    root_collection = ensure_collection('SarahMemory_Avatar')
    runtime_collection = ensure_collection('SarahMemory_RuntimeExport')
    ref_collection = ensure_collection('SarahMemory_References')
    rig_collection = ensure_collection('SarahMemory_Rig')

    add_reference_set(ref_collection)
    log("Reference set staged")

    suit_mat = create_suit_material()
    skin_mat = create_skin_material()
    hair_mat = create_hair_material()
    eye_sclera_mat, eye_iris_mat = create_eye_materials()
    detail_mats = create_detail_materials()

    log("Creating body mesh")
    body = create_body_mesh()
    log("Creating boots")
    boots = create_boot_pair()
    log("Creating head mesh")
    head = create_head_mesh()
    log("Creating hair mesh")
    hair = create_hair_mesh()
    eye_l, iris_l = create_eye('L')
    eye_r, iris_r = create_eye('R')

    mark_body_material_regions(body, suit_mat, skin_mat)
    assign_material(boots, suit_mat)
    assign_material(head, skin_mat)
    assign_material(hair, hair_mat)
    assign_material(eye_l, eye_sclera_mat)
    assign_material(eye_r, eye_sclera_mat)
    assign_material(iris_l, eye_iris_mat)
    assign_material(iris_r, eye_iris_mat)

    log("Creating rig and controls")
    rig = create_armature()
    configure_rig_controls(rig)
    log("Creating runtime detail layers: neon / face / hair / visual-only VFX")
    detail_objects = create_high_end_runtime_detail_layers(rig, detail_mats)
    detail_objects = bind_runtime_detail_surfaces(detail_objects, body, head, boots)
    detail_objects = bind_runtime_detail_armature(detail_objects, rig)

    for obj in (body, boots, head, hair, eye_l, eye_r, iris_l, iris_r):
        move_to_collection(obj, runtime_collection)
    for obj in detail_objects:
        move_to_collection(obj, runtime_collection)
    move_to_collection(rig, runtime_collection)

    parent_mesh_to_armature(body, rig)
    parent_mesh_to_armature(boots, rig)
    parent_mesh_to_armature(head, rig)
    parent_mesh_to_armature(hair, rig)
    parent_mesh_to_armature(eye_l, rig)
    parent_mesh_to_armature(eye_r, rig)
    parent_mesh_to_armature(iris_l, rig)
    parent_mesh_to_armature(iris_r, rig)

    set_head_vertex_groups(head)
    set_hair_vertex_groups(hair)
    shape_keys = add_head_shape_keys(head, rig)
    create_demo_action(rig)
    create_goldstandard_entity_actions(rig)

    cam = create_lights_and_camera(rig)
    eye_anchors = create_avatar_eye_camera_anchors(rig)
    all_objects = [body, boots, head, hair, eye_l, eye_r, iris_l, iris_r, rig, cam] + list(detail_objects) + list(eye_anchors)
    log("Saving .blend and exporting GLB")
    save_and_export(all_objects, rig, shape_keys)
    log('Rigged SarahMemory avatar bootstrap complete.')


if __name__ == '__main__':
    main()
'''



def _image_content_score(img) -> float:
    """Return rough non-background information density for a candidate crop."""
    try:
        from PIL import ImageStat
        rgb = img.convert("RGB").resize((96, 96))
        stat = ImageStat.Stat(rgb)
        mean = sum(stat.mean) / 3.0
        var = sum(stat.var) / 3.0
        pixels = list(rgb.getdata())
        # reward colored/contrasty foreground pixels and avoid pure white/black background dominance
        fg = 0
        for r, g, b in pixels:
            lum = (r + g + b) / 3.0
            sat = max(r, g, b) - min(r, g, b)
            if 18 < lum < 238 and sat > 12:
                fg += 1
        return float(fg) + float(var) * 0.05 + abs(mean - 128.0) * 0.02
    except Exception:
        return 0.0


def _normalize_reference_image(src: str, dst: str, role: str) -> str:
    """Create a single-avatar reference panel for Blender.

    Several SarahMemory reference PNGs contain a black-background and white-background
    copy in one image.  Blender reference planes need one centered figure, so this
    normalizer splits obvious duplicate panels and keeps the strongest single panel.
    The original file is never modified.
    """
    if not src or not os.path.exists(src):
        return ""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return os.path.abspath(src)

    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(src).convert("RGBA")
        w, h = img.size
        candidates = [("full", img)]
        # Wide comparison sheets commonly contain two duplicate avatars side by side.
        if w >= int(h * 1.15):
            mid = w // 2
            candidates.append(("left", img.crop((0, 0, mid, h))))
            candidates.append(("right", img.crop((mid, 0, w, h))))
        # Tall sheets can contain stacked variants; keep the strongest panel.
        if h >= int(w * 1.80):
            mid = h // 2
            candidates.append(("top", img.crop((0, 0, w, mid))))
            candidates.append(("bottom", img.crop((0, mid, w, h))))

        best_name, best_img = max(candidates, key=lambda item: _image_content_score(item[1]))
        # Normalize orientation as contained RGBA on transparent canvas.  Preserve detail.
        max_side = 2048
        bw, bh = best_img.size
        scale = min(1.0, max_side / max(1, max(bw, bh)))
        if scale < 1.0:
            best_img = best_img.resize((max(1, int(bw * scale)), max(1, int(bh * scale))), Image.LANCZOS)
        best_img = ImageOps.contain(best_img, (2048, 2048), Image.LANCZOS)
        best_img.save(dst, format="PNG", optimize=True)
        print(f"[SarahMemoryBlenderBootstrap] normalized {role} reference: {os.path.basename(src)} -> {os.path.basename(dst)} ({best_name})")
        return os.path.abspath(dst)
    except Exception as exc:
        print(f"[SarahMemoryBlenderBootstrap] reference normalization failed for {role}: {exc}")
        return os.path.abspath(src)


def _prepare_reference_set(outdir: str, front: str, side: str, back: str, blueprint: str) -> Dict[str, str]:
    normalized_dir = os.path.join(outdir, "source_normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    return {
        "front": _normalize_reference_image(front, os.path.join(normalized_dir, "front.png"), "front"),
        "side": _normalize_reference_image(side, os.path.join(normalized_dir, "side.png"), "side"),
        "back": _normalize_reference_image(back, os.path.join(normalized_dir, "back.png"), "back"),
        "blueprint": _normalize_reference_image(blueprint, os.path.join(normalized_dir, "blueprint.png"), "blueprint"),
    }

def _write_blender_script(
    script_path: str,
    front: str,
    side: str,
    back: str,
    blueprint: str,
    out_blend: str,
    out_glb: str,
    out_fbx: str,
    manifest_path: str,
    quality: str = "high",
    export_fbx: bool = True,
    preview_render: bool = False,
) -> None:
    script = _blender_script_template()
    replacements = {
        "__FRONT_IMAGE__": repr(front),
        "__SIDE_IMAGE__": repr(side),
        "__BACK_IMAGE__": repr(back),
        "__BLUEPRINT_IMAGE__": repr(blueprint),
        "__OUT_BLEND__": repr(out_blend),
        "__OUT_GLB__": repr(out_glb),
        "__OUT_FBX__": repr(out_fbx),
        "__MANIFEST_PATH__": repr(manifest_path),
        "__QUALITY__": repr(quality),
        "__EXPORT_FBX_ENABLED__": "True" if export_fbx else "False",
        "__PREVIEW_RENDER_ENABLED__": "True" if preview_render else "False",
    }
    for key, value in replacements.items():
        script = script.replace(key, value)

    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(script).strip() + "\n")



def blender_bootstrap_preflight(
    blender_path: str = "",
    *,
    front: str = "",
    side: str = "",
    back: str = "",
    blueprint: str = "",
    outdir: str = "",
    quality: str = "goldstandard_entity",
    allow_heavy_build: bool = False,
) -> Dict[str, object]:
    """Return a read-only governed readiness report without creating files."""
    defaults = _build_defaults()
    resolved, warning = _resolve_blender_executable(blender_path or defaults.get("blender", ""))
    quality_name = str(quality or "goldstandard_entity").strip().lower()
    heavy = quality_name in {"ultra", "cinematic_2m", "goldstandard_entity"}
    references = {
        "front": os.path.abspath(front or defaults.get("front", "")) if (front or defaults.get("front")) else "",
        "side": os.path.abspath(side or defaults.get("side", "")) if (side or defaults.get("side")) else "",
        "back": os.path.abspath(back or defaults.get("back", "")) if (back or defaults.get("back")) else "",
        "blueprint": os.path.abspath(blueprint or defaults.get("blueprint", "")) if (blueprint or defaults.get("blueprint")) else "",
    }
    reference_status = {
        role: {"path": path, "exists": bool(path and os.path.isfile(path)), "size_bytes": int(os.path.getsize(path)) if path and os.path.isfile(path) else 0}
        for role, path in references.items()
    }
    target_dir = os.path.abspath(outdir or defaults.get("outdir", ""))
    errors = []
    if not os.path.isfile(resolved):
        errors.append(f"Blender executable not found: {resolved}")
    if os.path.basename(resolved).lower() == "blender-launcher.exe":
        errors.append("blender-launcher.exe is not permitted for background builds; use blender.exe")
    if heavy and not allow_heavy_build:
        errors.append("Heavy authoring tier requires explicit --allow-heavy-build approval")
    if not any(item["exists"] for item in reference_status.values()):
        errors.append("No avatar reference image was found")
    return {
        "ok": not errors,
        "schema": "SARAHMEMORY_BLENDER_AVATAR_PREFLIGHT_V1",
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "blender_path": resolved,
        "blender_exists": os.path.isfile(resolved),
        "warning": warning,
        "quality": quality_name,
        "heavy_authoring_tier": heavy,
        "heavy_build_approved": bool(allow_heavy_build),
        "references": reference_status,
        "output_dir": target_dir,
        "output_dir_exists": os.path.isdir(target_dir),
        "errors": errors,
        "file_write_performed": False,
        "blender_launched": False,
        "execution_authority": False,
    }


def inspect_bootstrap_outputs(outdir: str = "") -> Dict[str, object]:
    """Inspect existing Blender/GLB/manifest outputs without launching Blender."""
    defaults = _build_defaults()
    root = os.path.abspath(outdir or defaults.get("outdir", ""))
    paths = {
        "blend": os.path.join(root, "SarahMemoryAvatar_RigBootstrap.blend"),
        "glb": os.path.join(root, "SarahMemoryAvatar_RigBootstrap.glb"),
        "fbx": os.path.join(root, "SarahMemoryAvatar_RigBootstrap.fbx"),
        "manifest": os.path.join(root, "SarahMemoryAvatar_RigBootstrap.json"),
        "build_log": os.path.join(root, "SarahMemoryAvatar_RigBootstrap_build.log"),
    }
    files = {
        name: {"path": path, "exists": os.path.isfile(path), "size_bytes": int(os.path.getsize(path)) if os.path.isfile(path) else 0}
        for name, path in paths.items()
    }
    manifest_data: Dict[str, object] = {}
    manifest_error = ""
    if files["manifest"]["exists"]:
        try:
            with open(paths["manifest"], "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            manifest_data = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            manifest_error = str(exc)
    glb_valid = False
    if files["glb"]["exists"] and files["glb"]["size_bytes"] >= 20:
        try:
            with open(paths["glb"], "rb") as fh:
                glb_valid = fh.read(4) == b"glTF"
        except Exception:
            glb_valid = False
    return {
        "ok": True,
        "schema": "SARAHMEMORY_BLENDER_AVATAR_OUTPUT_STATUS_V1",
        "output_dir": root,
        "files": files,
        "manifest_loaded": bool(manifest_data),
        "manifest_error": manifest_error,
        "manifest": manifest_data,
        "glb_header_valid": glb_valid,
        "runtime_ready": bool(glb_valid and files["manifest"]["exists"]),
        "execution_authority": False,
        "validation_boundary": "file/header/manifest inspection only; Blender and WebGL were not executed",
    }



def _print_run_summary(blender_exe: str, outdir: str, manifest_path: str) -> None:
    summary: Dict[str, object] = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "blender": blender_exe,
        "outdir": outdir,
        "manifest": manifest_path,
    }
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                summary["manifest_data"] = json.load(fh)
        except Exception as exc:
            summary["manifest_read_error"] = str(exc)
    print(json.dumps(summary, indent=2))


def main() -> int:
    defaults = _build_defaults()

    parser = argparse.ArgumentParser(description="Bootstrap a rigged SarahMemory avatar scene in Blender.")
    parser.add_argument("--blender", default=defaults["blender"], help="Full path to Blender executable.")
    parser.add_argument("--front", default=defaults["front"], help="Front reference image.")
    parser.add_argument("--side", default=defaults["side"], help="Side reference image.")
    parser.add_argument("--back", default=defaults["back"], help="Back reference image.")
    parser.add_argument("--blueprint", default=defaults["blueprint"], help="Optional blueprint / board image.")
    parser.add_argument("--outdir", default=defaults["outdir"], help="Output directory for .blend/.glb/.fbx. Default: BASE_DIR/resources/avatars/3D.")
    parser.add_argument(
        "--quality",
        default="goldstandard_entity",
        choices=("preview", "balanced", "high", "ultra", "cinematic_2m", "goldstandard_entity"),
        help="Avatar bootstrap quality tier. Default goldstandard_entity is the high-definition embodied AvatarPanel authoring lane. Use cinematic_2m/high/balanced only when a lower hardware budget is required.",
    )
    parser.add_argument("--skip-fbx", action="store_true", help="Compatibility flag. FBX export is skipped by default for tonight-build speed.")
    parser.add_argument("--export-fbx", action="store_true", help="Also export FBX. Disabled by default because GLB is the AvatarPanel runtime asset and FBX can add long export time.")
    parser.add_argument("--preview-render", action="store_true", help="Reserved flag for future still-preview rendering after asset export.")
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("SARAH_BLENDER_BUILD_TIMEOUT_SECONDS", "7200") or 7200), help="Hard timeout for Blender background build. Default 7200 seconds; use --no-timeout only with explicit operator intent.")
    parser.add_argument("--no-timeout", action="store_true", help="Disable the Blender build timeout only with explicit operator intent; the governed default remains a bounded timeout.")
    parser.add_argument("--keep-script", action="store_true", help="Keep generated temporary Blender script.")
    parser.add_argument("--allow-heavy-build", action="store_true", help="Explicitly approve ultra/cinematic/goldstandard authoring tiers.")
    parser.add_argument("--preflight-only", action="store_true", help="Print governed readiness and exit without launching Blender.")
    parser.add_argument("--inspect-runtime", action="store_true", help="Inspect existing runtime outputs and exit without launching Blender.")
    args = parser.parse_args()

    if args.inspect_runtime:
        print(json.dumps(inspect_bootstrap_outputs(args.outdir), indent=2))
        return 0

    preflight = blender_bootstrap_preflight(
        args.blender,
        front=args.front, side=args.side, back=args.back, blueprint=args.blueprint,
        outdir=args.outdir, quality=args.quality, allow_heavy_build=args.allow_heavy_build,
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0 if preflight.get("ok") else 2
    if not preflight.get("ok"):
        print(json.dumps(preflight, indent=2))
        return 2

    blender_exe, blender_warning = _resolve_blender_executable(args.blender)
    outdir = os.path.abspath(args.outdir)
    _ensure_dir(outdir)

    if blender_warning:
        print(f"[WARN] {blender_warning}")

    if not os.path.exists(blender_exe):
        print(f"[ERROR] Blender executable not found: {blender_exe}")
        return 1

    if os.path.basename(blender_exe).lower() == "blender-launcher.exe":
        print("[ERROR] Refusing to run blender-launcher.exe for a background asset build because it can hang for hours.")
        print(r"[ERROR] Pass the real blender.exe path, usually C:\Blender51\blender.exe or the Blender install folder blender.exe.")
        return 2

    out_blend = os.path.join(outdir, "SarahMemoryAvatar_RigBootstrap.blend")
    out_glb = os.path.join(outdir, "SarahMemoryAvatar_RigBootstrap.glb")
    out_fbx = os.path.join(outdir, "SarahMemoryAvatar_RigBootstrap.fbx")
    manifest = os.path.join(outdir, "SarahMemoryAvatar_RigBootstrap.json")

    sandbox_dir = defaults["sandbox_dir"]
    _ensure_dir(sandbox_dir)
    temp_script = os.path.join(sandbox_dir, "SarahMemoryAvatarBootstrap_blender.py")

    normalized_refs = _prepare_reference_set(
        outdir,
        os.path.abspath(args.front) if args.front else "",
        os.path.abspath(args.side) if args.side else "",
        os.path.abspath(args.back) if args.back else "",
        os.path.abspath(args.blueprint) if args.blueprint else "",
    )

    _write_blender_script(
        script_path=temp_script,
        front=normalized_refs.get("front", ""),
        side=normalized_refs.get("side", ""),
        back=normalized_refs.get("back", ""),
        blueprint=normalized_refs.get("blueprint", ""),
        out_blend=out_blend,
        out_glb=out_glb,
        out_fbx=out_fbx,
        manifest_path=manifest,
        quality=args.quality,
        export_fbx=bool(args.export_fbx and not args.skip_fbx),
        preview_render=args.preview_render,
    )

    cmd = [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        temp_script,
    ]

    build_log = os.path.join(outdir, "SarahMemoryAvatar_RigBootstrap_build.log")
    timeout_seconds = 0 if args.no_timeout else int(args.timeout_seconds or 0)

    print("[INFO] Launching Blender rig bootstrap...")
    print("[INFO] Blender:", blender_exe)
    print("[INFO] Output :", outdir)
    print("[INFO] Quality:", args.quality)
    print("[INFO] FBX    :", "enabled" if args.export_fbx and not args.skip_fbx else "disabled")
    print("[INFO] Timeout:", "disabled" if timeout_seconds <= 0 else f"{timeout_seconds}s")
    print("[INFO] Log    :", build_log)
    print("[INFO] Temp   :", temp_script)

    returncode = _run_streaming_blender(cmd, timeout_seconds=timeout_seconds, log_path=build_log)

    if returncode != 0:
        print("[ERROR] Blender rig bootstrap failed.")
        print("[ERROR] Review build log:", build_log)
        _print_run_summary(blender_exe=blender_exe, outdir=outdir, manifest_path=manifest)
        return returncode

    if not args.keep_script:
        try:
            os.remove(temp_script)
        except OSError:
            pass

    print("[OK] SarahMemory rig bootstrap completed.")
    _print_run_summary(blender_exe=blender_exe, outdir=outdir, manifest_path=manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryBlenderAvatarBootstrap.py v9.0.15
# ====================================================================
