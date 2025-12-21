"""--==The SarahMemory Project==--
File: SarahMemoryAvatar.py
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
"""

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
import sqlite3

DB_FILENAME = "avatar.db"

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


# Setup logger
logger = logging.getLogger("SarahMemoryAvatar")
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

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

def log_avatar_event(event, details):
    """
    Logs an avatar-related event to the avatar.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, DB_FILENAME))
        conn = sqlite3.connect(db_path)
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "avatar.db"))
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
def load_sprite_frames(sprite_sheet_path, frame_width, frame_height):
    sprite_sheet = Image.open(sprite_sheet_path)
    frames = []
    for y in range(0, sprite_sheet.height, frame_height):
        for x in range(0, sprite_sheet.width, frame_width):
            frame = sprite_sheet.crop((x, y, x+frame_width, y+frame_height))
            frames.append(ImageTk.PhotoImage(frame, master=self.root))
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

# ====================================================================
# END OF SarahMemoryAvatar.py v8.0.0
# ====================================================================