"""--==The SarahMemory Project==--
File: SarahMemoryAvatar.py
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
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "avatar_engine"
# CATEGORY = "avatar_system"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "display_audio"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "avatar_control"
# FAMILY = "avatar"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Avatar state/render/control engine for emotion sync, 2D rendering, animation hooks, lip sync helpers, and avatar state persistence."
# --- SARAHMETA END ---

import logging
import random
import time
import os
import sqlite3
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageTk  # For GUI display
from SarahMemoryAdaptive import load_emotional_state
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR  # for consistent pathing
import subprocess

DB_FILENAME = "avatar.db"


def _sm_avatar_base_dir() -> str:
    """Resolve SarahMemory install root for avatar resources without cwd dependency."""
    try:
        base = getattr(config, "BASE_DIR", None)
        if base:
            return os.path.abspath(os.path.expanduser(str(base)))
    except Exception:
        pass
    try:
        here = os.path.abspath(__file__)
        parent = os.path.dirname(here)
        if os.path.basename(parent).lower() == "core":
            return os.path.abspath(os.path.dirname(parent))
    except Exception:
        pass
    return os.path.abspath(os.getcwd())


def _sm_avatar_resources_dir() -> str:
    try:
        path = getattr(config, "RESOURCES_DIR", None)
        if path:
            return os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        pass
    return os.path.join(_sm_avatar_base_dir(), "resources")


# Setup logger (must exist before any init functions use it)
logger = logging.getLogger("SarahMemoryAvatar")
logger.setLevel(logging.DEBUG)
_handler = logging.NullHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(_handler)

def _ensure_avatar_state_table():
    """Ensure avatar_state table and default row exist (v8.0 safe init)."""
    try:
        db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS avatar_state (
                    id INTEGER PRIMARY KEY,
                    emotion TEXT,
                    expression TEXT,
                    state TEXT
                )
            """)
            cur.execute("SELECT COUNT(*) FROM avatar_state WHERE id = 1")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO avatar_state (id, emotion, expression, state) VALUES (1, ?, ?, ?)",
                    ("neutral", "neutral", "neutral")
                )
            conn.commit()
    except Exception as e:
        logger.error(f"avatar_state table init failed: {e}")

# initialize on import
_ensure_avatar_state_table()




# ─────────────────────────────────────────────
# 3D AVATAR ORGAN SOFTWARE BODY PROFILE
# ─────────────────────────────────────────────
# This profile is for visual embodiment and future mapping only.  It does not
# authorize physical robot actuation.  A later Avatar-to-MSDC bridge must route
# through OperatorCore / MSDC / Safety / Assurance / Compare / user approval.
AVATAR_3D_DIR = getattr(config, "AVATAR_3D_DIR", os.path.join(_sm_avatar_resources_dir(), "avatars", "3D"))
AVATAR_3D_DEFAULT_MODEL = "SarahMemoryAvatar_RigBootstrap.glb"
AVATAR_BODY_PROFILE = {
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


def get_avatar_3d_body_profile():
    """Return SarahMemory's visual humanoid body profile for the Avatar Organ."""
    return dict(AVATAR_BODY_PROFILE)


def get_avatar_3d_model_path():
    """Return the active local 3D Avatar Organ GLB path."""
    base = os.path.abspath(AVATAR_3D_DIR)
    for rel in (
        AVATAR_3D_DEFAULT_MODEL,
        os.path.join("default", AVATAR_3D_DEFAULT_MODEL),
        "sarahmemory_3d_avatar.glb",
        "sarahmemory_happy_face_ball.glb",
    ):
        candidate = os.path.abspath(os.path.join(base, rel))
        try:
            if os.path.commonpath([base, candidate]) != base:
                continue
        except Exception:
            continue
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(base, AVATAR_3D_DEFAULT_MODEL)


def get_avatar_3d_runtime_spec():
    """Return AvatarPanel-compatible 3D runtime spec."""
    model_path = get_avatar_3d_model_path()
    exists = os.path.isfile(model_path)
    model_file = os.path.basename(model_path)
    return {
        "renderMode": "gltf_model" if exists else "gold_standard_avatar",
        "modelUrl": f"/api/avatar/3d/{model_file}" if exists else "",
        "modelFile": model_file if exists else "",
        "loaderState": "3D_READY_GOLDSTANDARD_EMBODIED_ENTITY" if exists else "3D_FAILED_FALLBACK_GOLD_REFERENCE",
        "fallbackReason": "" if exists else "avatar_3d_model_missing_or_unavailable",
        "governance": {
            "visual_only": True,
            "physical_actuation_allowed": False,
            "msdc_dispatch_required_for_robot_body": True,
            "user_authority_required_for_embodied_action": True,
        },
        "pose": "stand",
        "gesture": "none",
        "lookAt": {"x": 0, "y": 1.45, "z": 0},
        "expression": get_avatar_emotion() if "get_avatar_emotion" in globals() else "neutral",
        "speaking": False,
        "listening": False,
        "quality": "goldstandard_entity",
        "fpsCap": 60,
        "lightingProfile": "high_end",
        "shadowQuality": "high",
                "animationAuthority": "visual_only_no_msdc_no_operator_action",
                "avatarEyeCameraAnchor": "Sarah_AvatarEye_Center",
        "stageOffsetY": -1.18,
        "avatarOffsetY": -0.52,
        "useRuntimeStage": True,
        "materialMode": "high_end",
        "runtimeVisualPriority": "gltf_model",
        "meshFallbackUrl": f"/api/avatar/3d/{model_file}" if exists else "",
        "forceMeshRuntime": bool(exists),
        "goldReferenceOnly": True,
        "authoringPolyTarget": 12000000,
        "runtimeLod": "goldstandard_entity",
        "productionAssetContract": "SARAHMEMORY_GOLDSTANDARD_EMBODIED_ENTITY_GLB_V1",
        "blueprintConstructReady": True,
        "visualEffectsMode": "secondary",
        "goldStandardScale": 1.0,
        "goldStandardYOffset": 0,
        "goldStandardPanelBottomPx": 58,
        "goldStandardPanelHeightPct": 92,
        "bodyProfile": get_avatar_3d_body_profile(),
        "source": "resources/avatars/3D",
        "robotBridge": {
            "enabled": False,
            "future_owner": "AvatarToMSDCBridge",
            "physical_actuation_allowed_here": False,
        },
    }


# ─────────────────────────────────────────────
# 2D AVATAR CONTROL
# ─────────────────────────────────────────────
EMOTION_EXPRESSIONS = {
    "joy": "😊",
    "anger": "😠",
    "fear": "😨",
    "trust": "🤝",
    "surprise": "😲",
    "neutral": "😐",
    "thinking": "🤔"
}
MOOD_MAP = {
    'joy': '😊',
    'fear': '😨',
    'trust': '😌',
    'anger': '😠',
    'surprise': '😲',
    'neutral': '😐',
    "thinking": "🤔"
    # Add more mappings as needed
}


def _hardware_tier_rating() -> str:
    """Return current tier_rating from SarahMemoryGlobals.hardware_score() (best-effort)."""
    try:
        hs_fn = getattr(config, "hardware_score", None)
        if callable(hs_fn):
            hs = hs_fn() or {}
            tr = str(hs.get("tier_rating") or "").strip()
            return tr if tr else "Unknown"
    except Exception:
        pass
    return "Unknown"

def log_avatar_event(event, details):
    """
    Logs an avatar-related event to the avatar.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, DB_FILENAME))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS avatar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO avatar_events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged avatar event to avatar.db successfully.")
    except Exception as e:
        logger.error(f"Error logging avatar event: {e}")

def get_dominant_emotion():
    emotions = load_emotional_state()
    if not emotions:
        logger.warning("No emotional state data available. Defaulting to neutral.")
        log_avatar_event("Get Dominant Emotion", "No emotion data; defaulting to neutral.")
        return "neutral"
    dominant = max(emotions, key=emotions.get)
    logger.info(f"Dominant emotion: {dominant} ({emotions[dominant]})")
    log_avatar_event("Get Dominant Emotion", f"Dominant: {dominant} with value {emotions[dominant]}")
    return dominant

def draw_2d_avatar(expression, extra_overlay=None):
    """
    Draws a 2D avatar with PIL representing Sarah's mood.
    ENHANCED (v6.4): Supports extra overlay for blended expressions.
    """
    face = EMOTION_EXPRESSIONS.get(expression, EMOTION_EXPRESSIONS["neutral"])
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font_path = os.path.join("C:\\Windows\\Fonts\\arial.ttf")
        font = ImageFont.truetype(font_path, 100)
    except Exception as e:
        logger.warning(f"Fallback to default font: {e}")
        font = ImageFont.load_default()
    draw.text((90, 100), face, font=font, fill='black')
    if extra_overlay:
        draw.text((10, 10), extra_overlay, font=font, fill='red')  # MOD: Extra overlay text
    img.show()
    return img
#---------------Tkinter GUI Integration (if needed)------------------
def load_sprite_frames(sprite_sheet_path, frame_width, frame_height, tk_root=None):
    """Load sprite frames from a sheet.
    NOTE: tk_root is optional; pass your Tk() root to avoid PhotoImage GC issues.
    """
    sprite_sheet = Image.open(sprite_sheet_path)
    frames = []
    for y in range(0, sprite_sheet.height, frame_height):
        for x in range(0, sprite_sheet.width, frame_width):
            frame = sprite_sheet.crop((x, y, x + frame_width, y + frame_height))
            try:
                frames.append(ImageTk.PhotoImage(frame, master=tk_root) if tk_root else ImageTk.PhotoImage(frame))
            except Exception:
                # If Tk context is not available, return raw PIL frames
                frames.append(frame.copy())
    return frames

def animate_walk(self):
    self.current_frame = (self.current_frame + 1) % len(self.walk_frames)
    self.avatar_label.configure(image=self.walk_frames[self.current_frame])
    self.root.after(100, self.animate_walk)  # update every 100ms
def animate_flames(self):
    try:
        self.flame_frames = [ImageTk.PhotoImage(frame.copy(), master=self.root)
                              for frame in ImageSequence.Iterator(Image.open("flames.gif"))]
    except Exception as e:
        logger.error(f"Failed to load flame animation: {e}")
    self.current_flame_frame = 0
    self.update_flames()

def update_flames(self):
    self.background_label.configure(image=self.flame_frames[self.current_flame_frame])
    self.current_flame_frame = (self.current_flame_frame + 1) % len(self.flame_frames)
    self.root.after(100, self.update_flames)
def animate_lip_sync(self, duration):
    # Suppose you have a list self.lip_sync_frames
    start_time = time.time()
    while time.time() - start_time < duration:
        for frame in self.lip_sync_frames:
            self.avatar_label.configure(image=frame)
            self.avatar_label.image = frame  # keep reference
            time.sleep(0.1)  # adjust based on frame rate
def animate_body(self):
    self.current_body_frame = (self.current_body_frame + 1) % len(self.body_frames)
    self.body_label.config(image=self.body_frames[self.current_body_frame])
    self.root.after(100, self.animate_body)  # Adjust frame interval as needed
def update_head_movement(self):
    # AI-based decision: could be based on live audio or a sentiment analysis module.
    # For a basic example, choose a frame from a pre-rendered list of head poses.
    chosen_frame = self.head_frames[self.ai_decide_head_pose()]
    self.head_label.config(image=chosen_frame)
    self.root.after(50, self.update_head_movement)  # Fast updates for fluid movement

def ai_decide_head_pose(self):
    # Placeholder: replace with a dynamic selection based on your AI module's output.
    return random.randint(0, len(self.head_frames) - 1)
# ─────────────────────────────────────────────

def update_gaze_direction(audio_input, current_gaze):
    # Extract features from the audio input using a pre-trained NLP/audio model.
    features = extract_features(audio_input)

    # Predict the target gaze direction (e.g., as angles in degrees or as screen coordinates)
    target_gaze = gaze_model.predict(features)

    # Smoothly interpolate between the current gaze direction and the target gaze direction.
    new_gaze = interpolate(current_gaze, target_gaze, smoothing_factor=0.1)
    return new_gaze
def sprite_main_loop():
    current_gaze = initial_gaze
    while app_running:
        audio_input = get_audio_input()  # Could be speech or a specific command
        if audio_input:
            current_gaze = update_gaze_direction(audio_input, current_gaze)
            # Update the avatar's head and eye layers based on `current_gaze`
            update_avatar_head(current_gaze)
        time.sleep(0.05)  # Adjust to match the desired frame rate
def update_avatar_head(gaze_direction):
    # Assuming gaze_direction is a tuple (x, y) representing the gaze coordinates
    # Update the head and eye layers accordingly
    head_x, head_y = gaze_direction
    head_label.place(x=head_x, y=head_y)  # Adjust position based on gaze direction
    eye_label.place(x=head_x + 10, y=head_y + 10)  # Offset for eyes
    logger.info(f"Updated avatar head position to {head_x}, {head_y} based on gaze direction.")
def extract_features(audio_input):
    # Placeholder function: replace with actual feature extraction logic.
    # For example, you might use a pre-trained model to extract MFCCs or other audio features.
    return [0.5, 0.2, 0.3]  # Dummy features
def interpolate(current, target, factor):
    # Simple linear interpolation between current and target values.
    return current + (target - current) * factor
def get_audio_input():
    # Placeholder function: replace with actual audio input retrieval.
    # For example, you might use a microphone or an audio file.
    pass
def set_avatar_expression(expression):
    """
    Sets the avatar expression in the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE avatar_state SET expression = ? WHERE id = 1", (expression,))
    conn.commit()
    conn.close()
    return True
def get_avatar_state():
    """
    Retrieves the avatar state from the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT state FROM avatar_state WHERE id = 1")
    state = cursor.fetchone()
    conn.close()
    return state[0] if state else "neutral"
def set_avatar_state(state):
    """
    Sets the avatar state in the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE avatar_state SET state = ? WHERE id = 1", (state,))
    conn.commit()
    conn.close()
def get_avatar_emotion():
    """
    Retrieves the current avatar emotion from the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT emotion FROM avatar_state WHERE id = 1")
    emotion = cursor.fetchone()
    conn.close()
    return emotion[0] if emotion else "neutral"
def set_avatar_emotion(emotion):
    """
    Sets the avatar emotion in the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE avatar_state SET emotion = ? WHERE id = 1", (emotion,))
    conn.commit()
    conn.close()
def get_avatar_expression():
    """
    Retrieves the current avatar expression from the database.
    """
    db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT expression FROM avatar_state WHERE id = 1")
    expression = cursor.fetchone()
    conn.close()
    return expression[0] if expression else "neutral"
def update_avatar_expression(expression=None):
    """
    Master controller: updates the avatar based on the current emotion.
    ENHANCED (v6.4): Integrates vector-based blending simulation with a random overlay hint.
    """
    if not expression:
        expression = get_dominant_emotion()
    logger.info(f"Avatar expression = {expression}")
    log_avatar_event("Update Avatar Expression", f"Expression set to {expression}")
    extra_overlay = f"Blend-{random.choice(['A', 'B', 'C'])}" if random.random() > 0.5 else None
    draw_2d_avatar(expression, extra_overlay)
    interact_with_gui(expression, EMOTION_EXPRESSIONS.get(expression, ''))
    trigger_3d_animation(expression)
    emotions = load_emotional_state()
    if not emotions:
        logger.warning("No emotions available; defaulting to neutral.")
        log_avatar_event("Avatar Expression Warning", "No emotion data; defaulting to neutral.")
        return "neutral"
    top_mood = max(emotions, key=emotions.get)
    expression_final = MOOD_MAP.get(top_mood, '😐')
    logger.info(f"Avatar Mood Sync: {top_mood.upper()} mapped to {expression_final}")
    log_avatar_event("Avatar Mood Sync", f"Top mood: {top_mood.upper()} mapped to {expression_final}")
    return expression_final

def simulate_lip_sync_async(duration=2.0):
    """
    Asynchronous wrapper to simulate lip-sync animation.
    NEW (v6.4): Uses threading to run without blocking the main GUI.
    """
    import threading
    threading.Thread(target=simulate_lip_sync, args=(duration,), daemon=True).start()

def simulate_lip_sync(duration=2.0):
    logger.info(f"Avatar Lip Sync Activated for {duration} sec")
    log_avatar_event("Simulate Lip Sync", f"Lip sync active for {duration} sec")
    start_time = time.time()
    while time.time() - start_time < duration:
        logger.debug("Simulated lip movement: mouth opens")
        time.sleep(0.2)
    logger.debug("Lip sync complete")
    log_avatar_event("Simulate Lip Sync", "Lip sync cycle completed")

def interact_with_gui(mood, face_repr):
    logger.info(f"[GUI] Mood: {mood}, Face: {face_repr}")
    log_avatar_event("GUI Interaction", f"Mood: {mood} | Face: {face_repr}")

def trigger_3d_animation(emotion):
    tier_rating = _hardware_tier_rating().lower()
    if tier_rating == "poor":
        logger.info("[3D] Skipped 3D animation (tier_rating=Poor).")
        log_avatar_event("Trigger 3D Animation Skipped", "tier_rating=Poor")
        return None
    logger.info(f"[3D] Triggering 3D animation for emotion: {emotion}")
    log_avatar_event("Trigger 3D Animation", f"3D animation for emotion: {emotion}")
#--------------------------------Tinkering with subprocess to run 3D animation

    try:
        subprocess.run(["python", "3DAnimationEngine.py", emotion], check=True)
        logger.info("3D animation triggered successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error triggering 3D animation: {e}")
        log_avatar_event("Trigger 3D Animation Error", f"Error triggering 3D animation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        log_avatar_event("Trigger 3D Animation Error", f"Unexpected error: {e}")
    return None
    #---------------------------------- End of subprocess tinkering
 # SarahMemoryAvatar.py (add this below trigger_3d_animation)

def display_interactive_3d_avatar(filepath, engine="Blender"):
    try:
        tier_rating = _hardware_tier_rating().lower()
        if tier_rating == "poor":
            logger.info("[3D] Skipped interactive 3D avatar (tier_rating=Poor).")
            return False

        logger.info(f"Launching 3D avatar from: {filepath} with engine: {engine}")
        if engine.lower() == "blender":
            subprocess.run([
                "blender", filepath,
                "--background",
                "--python", os.path.join(config.TOOLS_DIR, "render_avatar.py")
            ], check=True)
            logger.info("Blender avatar render launched successfully.")
        elif engine.lower() == "unreal":
            # Optional: Unreal launch logic (requires setup)
            subprocess.run(["C:\\Path\\To\\UnrealEditor.exe", "YourProject.uproject"], check=True)
    except Exception as e:
        logger.error(f"Error launching 3D avatar: {e}")

if __name__ == '__main__':

    import sys
    logger.info("Testing Unified Avatar Engine...")
    args = sys.argv[1:]

    if '--test2d' in args:
        dominant = get_dominant_emotion()
        update_avatar_expression(dominant)

    elif '--test3d' in args:
        dominant = get_dominant_emotion()
        trigger_3d_animation(dominant)

    elif '--lipsync' in args:
        simulate_lip_sync_async(3)

    else:
        print("Usage: python SarahMemoryAvatar.py [--test2d | --test3d | --lipsync]")

# [PATCH v7.7.2] Avatar render instruction hook
def apply_render_instructions(instr: dict):
    """
    Map high-level avatar state (mood/personality) to internal state.
    instr example: {"mood":"happy","energy":0.7,"style":"neon"}
    """
    try:
        mood = (instr or {}).get("mood") or "neutral"
        set_avatar_state(mood)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# SARAH_AVATAR_REM_STATE_V1
# Backend REM Sleep state persistence helpers. Additive and legacy-safe.
# -----------------------------------------------------------------------------
def _ensure_avatar_rem_table():
    try:
        db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS avatar_rem_state (
                    id INTEGER PRIMARY KEY,
                    ts TEXT,
                    phase TEXT,
                    expression TEXT,
                    reason TEXT,
                    metadata_json TEXT
                )
            """)
            cur.execute("SELECT COUNT(*) FROM avatar_rem_state WHERE id = 1")
            if cur.fetchone()[0] == 0:
                cur.execute("INSERT INTO avatar_rem_state (id, ts, phase, expression, reason, metadata_json) VALUES (1, ?, ?, ?, ?, ?)", (datetime.now().isoformat(), "awake", "ready", "initialized", "{}"))
            conn.commit()
    except Exception as e:
        logger.error(f"avatar_rem_state table init failed: {e}")

def set_avatar_rem_state(phase: str, expression: str = "sleepy", reason: str = "", metadata: dict | None = None) -> bool:
    """Persist REM Sleep visual state for backend/UI synchronization."""
    try:
        _ensure_avatar_rem_table()
        db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
        import json as _json
        clean_phase = str(phase or "awake").strip()
        clean_expression = str(expression or "ready").strip()
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE avatar_rem_state SET ts=?, phase=?, expression=?, reason=?, metadata_json=? WHERE id=1", (datetime.now().isoformat(), clean_phase, clean_expression, str(reason or ""), _json.dumps(metadata or {}, ensure_ascii=False)))
            conn.commit()
        try:
            set_avatar_state(clean_phase); set_avatar_expression(clean_expression); set_avatar_emotion(clean_expression)
        except Exception:
            pass
        log_avatar_event("REM State", f"phase={clean_phase}; expression={clean_expression}; reason={reason}")
        return True
    except Exception as e:
        logger.error(f"set_avatar_rem_state failed: {e}")
        return False

def get_avatar_rem_state() -> dict:
    """Return current REM Sleep avatar state."""
    try:
        _ensure_avatar_rem_table()
        db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor(); row = cur.execute("SELECT ts, phase, expression, reason, metadata_json FROM avatar_rem_state WHERE id=1").fetchone()
        if not row:
            return {"phase": "awake", "expression": "ready", "reason": "missing_row"}
        import json as _json
        try: meta = _json.loads(row[4] or "{}")
        except Exception: meta = {}
        return {"ts": row[0], "phase": row[1], "expression": row[2], "reason": row[3], "metadata": meta}
    except Exception as e:
        logger.error(f"get_avatar_rem_state failed: {e}")
        return {"phase": "awake", "expression": "ready", "reason": str(e)}

_ensure_avatar_rem_table()

# ====================================================================
# END OF SarahMemoryAvatar.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAvatar',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Execution',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['avatar', 'execution'],
    "supported_missions": ['Conversation', 'Execution'],
    "supported_omega": ['Ω001', 'Ω070', 'Ω100'],
    "required_authority": ['Execute', 'Read'],
    "priority": 50,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAvatar.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAvatar',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAvatar', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

