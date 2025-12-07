"""--==The SarahMemory Project==--
File: SarahMemoryFacialRecognition.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
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

import cv2
import logging
import os
import sys
import sqlite3
import time
import numpy as np
from datetime import datetime
try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False

import SarahMemoryGlobals as config
from SarahMemoryGlobals import run_async  # Used in VSM module
from SarahMemoryGlobals import DATASETS_DIR, OBJECT_MODEL_CONFIG, FACIAL_RECOGNITION_LEARNING, MODEL_PATHS

VSM_DB = os.path.join(DATASETS_DIR, "system_logs.db")
VECTOR_MEMORY = []
VECTOR_DIM = 128
# Inject YOLO object model support
YOLO = None
try:
    from ultralytics import YOLO
except ImportError:
    pass
USE_DL_MODELS = any(model_cfg.get("enabled") for model_cfg in OBJECT_MODEL_CONFIG.values())
# ------------------------- Facial Recognition Module -------------------------
# Setup logging for the facial recognition module
logger_fr = logging.getLogger('SarahMemoryFacialRecognition')
logger_fr.setLevel(logging.DEBUG)
handler_fr = logging.NullHandler()
formatter_fr = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler_fr.setFormatter(formatter_fr)
if not logger_fr.hasHandlers():
    logger_fr.addHandler(handler_fr)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Define cascade path
def add_vector(vector):
    global VECTOR_MEMORY
    timestamp = datetime.datetime.now().isoformat()
    try:
        if faiss_available:
            index_file = os.path.join(config.DATASETS_DIR, "vector_index.faiss")
            if os.path.exists(index_file):
                index = faiss.read_index(index_file)
            else:
                index = faiss.IndexFlatL2(VECTOR_DIM)
            index.add(np.array([vector]).astype(np.float32))
            faiss.write_index(index, index_file)
            log_event("VSM", f"Vector added via FAISS at {timestamp}")
        else:
            VECTOR_MEMORY.append(vector)
            log_event("VSM", f"Vector added via fallback engine at {timestamp}")
    except Exception as e:
        log_event("VSM", f"Error adding vector: {e}")


def find_similar_vectors(query_vector, top_k=5):
    if faiss_available:
        try:
            index_file = os.path.join(config.DATASETS_DIR, "vector_index.faiss")
            if not os.path.exists(index_file):
                return []
            index = faiss.read_index(index_file)
            D, I = index.search(np.array([query_vector]).astype(np.float32), top_k)
            return list(zip(I[0], D[0]))
        except Exception as e:
            log_event("VSM", f"Search failed: {e}")
            return []
    else:
        scores = [(i, cosine_similarity(query_vector, vec)) for i, vec in enumerate(VECTOR_MEMORY)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]


def log_event(source, message):
    try:
        conn = sqlite3.connect(VSM_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                source TEXT,
                message TEXT
            )
        """)
        conn.execute("INSERT INTO logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)",
                     (datetime.datetime.now().isoformat(), "INFO", source, message))
        conn.commit()
        conn.close()
    except:
        pass

def log_facial_event(event, details):
    """
    Logs a facial recognition event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
        # db_path = os.path.abspath(config.DATASETS_DIR "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facial_recognition_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO facial_recognition_events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))
        conn.commit()
        conn.close()
        logger_fr.info("Logged facial event to system_logs.db successfully.")
    except Exception as e:
        logger_fr.error(f"Error logging facial event: {e}")

def load_face_cascade(cascade_path=None):
    """
    Load the Haar cascade classifier for face detection.
    """
    if not cascade_path:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            error_msg = "Failed to load cascade classifier. Check the cascade path."
            logger_fr.error(error_msg)
            log_facial_event("Load Cascade Error", error_msg)
            return None
        logger_fr.info("Cascade classifier loaded successfully.")
        log_facial_event("Load Cascade Success", "Cascade classifier loaded successfully.")
        return face_cascade
    except Exception as e:
        logger_fr.error(f"Error loading cascade classifier: {e}")
        log_facial_event("Load Cascade Exception", f"Exception: {e}")
        return None


def detect_faces(frame, face_cascade):
    """
    Detect faces in a given image frame using the Haar cascade classifier.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        logger_fr.info(f"Detected {len(faces)} face(s) using Haar cascade.")
        log_facial_event("Detect Faces", f"Detected {len(faces)} face(s) in frame.")
        return faces
    except Exception as e:
        logger_fr.error(f"Error detecting faces: {e}")
        log_facial_event("Detect Faces Error", f"Exception: {e}")
        return []


def detect_faces_dnn(frame):
    """
    Apply enabled object detection models to frame.
    If one model fails, fallback to the next; fallback to Haar Cascade as last resort.
    Enhanced: Now supports YOLO, TorchScript, and ONNX formats based on filename.
    Adds confidence filtering and label targeting for 'face'/'person' classes.
    """
    detections = []
    CONFIDENCE_THRESHOLD = 0.4
    TARGET_LABELS = ["person", "face"]

    if USE_DL_MODELS:
        for model_name, model_info in OBJECT_MODEL_CONFIG.items():
            if model_info.get("enabled"):
                try:
                    model_path = MODEL_PATHS.get(model_name)
                    if not model_path or not os.path.exists(model_path):
                        logger_fr.warning(f"[MODEL_PATH_MISSING] Model {model_name} skipped. Path invalid or not found: {model_path}")
                        continue

                    if model_path.endswith(".pt") and YOLO is not None:
                        model = YOLO(model_path)
                        results = model(frame)
                        for result in results:
                            for box in result.boxes:
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                label = model.names[cls]
                                if conf >= CONFIDENCE_THRESHOLD and label.lower() in TARGET_LABELS:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    detections.append(frame[y1:y2, x1:x2])
                        logger_fr.info(f"{model_name} (YOLO) detected {len(detections)} object(s).")
                        if detections:
                            return detections

                    elif model_path.endswith(".onnx"):
                        import onnxruntime as ort
                        session = ort.InferenceSession(model_path)
                        img = cv2.resize(frame, (640, 640))
                        img = img.transpose(2, 0, 1).astype(np.float32)
                        img = np.expand_dims(img, axis=0)
                        inputs = {session.get_inputs()[0].name: img}
                        outputs = session.run(None, inputs)
                        output = outputs[0][0]  # Assume shape: [num_detections, 6] (x1, y1, x2, y2, conf, class_id)
                        for det in output:
                            x1, y1, x2, y2, conf, cls = det[:6]
                            if conf >= CONFIDENCE_THRESHOLD:
                                label = str(int(cls))  # Placeholder label map
                                if label in TARGET_LABELS or label == "0":  # crude match to "person"
                                    detections.append(frame[int(y1):int(y2), int(x1):int(x2)])
                        logger_fr.info(f"{model_name} (ONNX) detected {len(detections)} object(s).")
                        if detections:
                            return detections

                    elif model_path.endswith(".ts") or model_path.endswith(".torchscript"):
                        import torch
                        model = torch.jit.load(model_path)
                        model.eval()
                        img = cv2.resize(frame, (640, 640))
                        input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
                        with torch.no_grad():
                            outputs = model(input_tensor)[0]  # Assume [N, 6]: x1, y1, x2, y2, conf, cls
                        for det in outputs:
                            x1, y1, x2, y2, conf, cls = det.tolist()
                            if conf >= CONFIDENCE_THRESHOLD:
                                label = str(int(cls))
                                if label in TARGET_LABELS or label == "0":
                                    detections.append(frame[int(y1):int(y2), int(x1):int(x2)])
                        logger_fr.info(f"{model_name} (TorchScript) detected {len(detections)} object(s).")
                        if detections:
                            return detections

                except Exception as e:
                    logger_fr.warning(f"Model {model_name} failed: {e}")

    logger_fr.info("Falling back to Haar Cascade face detection.")
    face_cascade = load_face_cascade()
    if face_cascade is not None:
        detections = detect_faces(frame, face_cascade)
        if isinstance(detections, (list, tuple, np.ndarray)) and len(detections) > 0:
            return detections

    logger_fr.warning("All detection models failed or returned nothing. Returning empty.")
    return []


# ------------------------- New Hooks for Identity Tagging -------------------------

# Simulated profile map (extend this with real DB later)
IDENTITY_PROFILES = {
    0: {"name": "Brian", "emotion": "joy"},
    1: {"name": "Briana", "emotion": "curiosity"},
    2: {"name": "Victoria", "emotion": "neutral"}
}

def recognize_identity_from_vector(query_vector):
    """
    Recognizes user by comparing input face vector to known stored ones.
    Returns name and emotion.
    """
    distances, indices = query_similar(query_vector, k=1)
    if distances is not None and indices is not None:
        closest_id = int(indices[0][0])
        profile = IDENTITY_PROFILES.get(closest_id, {"name": "Unknown", "emotion": "neutral"})
        log_facial_event("Recognized Identity", f"Matched to: {profile['name']} (Emotion: {profile['emotion']})")
        return profile
    return {"name": "Unknown", "emotion": "neutral"}

# ------------------------- Personality Greeting Export -------------------------

def personalized_greeting_from_camera():
    """
    Uses the webcam frame to recognize identity and return a greeting string.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Hi there."
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Hello!"
    face_cascade = load_face_cascade()
    faces = detect_faces_dnn(frame)
    if len(faces) == 0:
        return "Hello!"
    # Simulate vector from face
    fake_vector = np.random.rand(VECTOR_DIM)
    identity = recognize_identity_from_vector(fake_vector)
    name = identity.get("name", "User")
    emotion = identity.get("emotion", "neutral")
    return f"Welcome back, {name}. You seem {emotion} today."

# ------------------------- Idle Hook: Inject Facial Data into Optimization -------------------------

def contribute_facial_data_to_learning():
    """
    When triggered from SarahMemoryOptimization.py, simulate learning faces and saving vector states.
    """
    try:
        fake_vector = np.random.rand(VECTOR_DIM)
        async_add_vector(fake_vector)
        log_facial_event("Idle Facial Data Injected", f"Background vector trained.")
    except Exception as e:
        log_facial_event("Idle Training Failed", str(e))

def draw_faces(frame, faces):
    """
    Draw rectangles around detected faces in the image frame.
    """
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        log_facial_event("Draw Faces", f"Drew rectangles around {len(faces)} face(s).")
        return frame
    except Exception as e:
        logger_fr.error(f"Error drawing faces: {e}")
        log_facial_event("Draw Faces Error", f"Exception: {e}")
        return frame

def start_facial_recognition_monitor(interval=0.1):
    """
    Start a background loop that captures and processes webcam frames for facial recognition.
    NEW (v6.4): Runs detection asynchronously without blocking the UI.
    """
    import threading
    def monitor():
        face_cascade = load_face_cascade()
        if face_cascade is None:
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger_fr.error("Unable to access the webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                logger_fr.warning("Failed to capture frame from webcam.")
                break
            faces = detect_faces_dnn(frame)
            draw_faces(frame, faces)
            time.sleep(interval)
        cap.release()
    threading.Thread(target=monitor, daemon=True).start()

# Additional integration for storing recognized face vectors per user
USER_DB = os.path.join(DATASETS_DIR, "user_profiles.db")

def store_face_profile(user_id, vector, label, emotion):
    """
    Store labeled face vector into user_profiles.db under specific user ID.
    """
    try:
        conn = sqlite3.connect(USER_DB)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                label TEXT,
                emotion TEXT,
                vector BLOB
            )
        """)
        timestamp = datetime.now().isoformat()
        vector_blob = sqlite3.Binary(np.array(vector, dtype=np.float32).tobytes())
        cursor.execute("""
            INSERT INTO face_profiles (user_id, timestamp, label, emotion, vector)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, timestamp, label, emotion, vector_blob))
        conn.commit()
        conn.close()
        logger_fr.info(f"Stored vector for {label} in user_profiles.db.")
    except Exception as e:
        logger_fr.error(f"Error storing face profile: {e}")


# ------------------------- Vector Similarity Memory (VSM) Module -------------------------
# Setup logging for the VSM module
logger_vsm = logging.getLogger('SarahMemoryVSM')
logger_vsm.setLevel(logging.DEBUG)
handler_vsm = logging.NullHandler()
formatter_vsm = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler_vsm.setFormatter(formatter_vsm)
if not logger_vsm.hasHandlers():
    logger_vsm.addHandler(handler_vsm)

VECTOR_DIM = 128  # Example dimension
index = None  # Global index object

def log_vsm_event(event, details):
    """
    Logs a VSM-related event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vsm_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO vsm_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger_vsm.info("Logged VSM event to system_logs.db successfully.")
    except Exception as e:
        logger_vsm.error(f"Error logging VSM event: {e}")

def initialize_index(dim=VECTOR_DIM):
    """
    Initialize a FAISS index for vector similarity search using L2 distance.
    ENHANCED (v6.4): Now includes vector normalization simulation.
    """
    try:
        import faiss  # Local import to ensure availability
        idx = faiss.IndexFlatL2(dim)
        logger_vsm.info(f"FAISS index initialized with dimension {dim}.")
        log_vsm_event("Initialize Index", f"Index initialized with dimension {dim}.")
        return idx
    except Exception as e:
        logger_vsm.error(f"Error initializing FAISS index: {e}")
        log_vsm_event("Initialize Index Error", f"Exception: {e}")
        return None

def add_vector(vector):
    """
    Add a vector to the FAISS index.
    ENHANCED (v6.4): Validates vector normalization and caches index state.
    """
    global index
    try:
        if index is None:
            index = initialize_index()
            if index is None:
                logger_vsm.error("Index is not initialized.")
                log_vsm_event("Add Vector Error", "Index initialization failed.")
                return False
        vector = np.asarray(vector, dtype='float32').reshape(1, -1)
        if vector.shape[1] != VECTOR_DIM:
            error_msg = f"Vector dimension mismatch: Expected {VECTOR_DIM}, got {vector.shape[1]}"
            logger_vsm.error(error_msg)
            log_vsm_event("Add Vector Error", error_msg)
            return False
        # NEW: Normalize vector (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        index.add(vector)
        logger_vsm.info("Vector added to index successfully.")
        log_vsm_event("Add Vector", "Vector added to index successfully.")
        return True
    except Exception as e:
        logger_vsm.error(f"Error adding vector: {e}")
        log_vsm_event("Add Vector Exception", f"Exception: {e}")
        return False

def async_add_vector(vector):
    """
    Run add_vector asynchronously.
    NEW (v6.4): Uses run_async to add vector without blocking.
    """
    run_async(add_vector, vector)

def query_similar(vector, k=5):
    """
    Query the FAISS index for the k most similar vectors to the provided vector.
    ENHANCED (v6.4): Normalizes query vector and provides detailed error logs.
    """
    global index
    try:
        if index is None or getattr(index, 'ntotal', 0) == 0:
            logger_vsm.warning("Index is empty or uninitialized.")
            log_vsm_event("Query Similar", "Index is empty or uninitialized.")
            return None, None
        vector = np.asarray(vector, dtype='float32').reshape(1, -1)
        if vector.shape[1] != VECTOR_DIM:
            error_msg = f"Query vector dimension mismatch: Expected {VECTOR_DIM}, got {vector.shape[1]}"
            logger_vsm.error(error_msg)
            log_vsm_event("Query Similar Error", error_msg)
            return None, None
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        distances, indices = index.search(vector, k)
        logger_vsm.info(f"Query completed. Distances: {distances}, Indices: {indices}")
        log_vsm_event("Query Similar", f"Query returned distances {distances} and indices {indices}.")
        return distances, indices
    except Exception as e:
        logger_vsm.error(f"Error querying index: {e}")
        log_vsm_event("Query Similar Exception", f"Exception: {e}")
        return None, None

# ------------------------- Main Block -------------------------
if __name__ == '__main__':
    # Use command-line argument to choose the functionality.
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'vsm':
        # Run VSM module test
        logger_vsm.info("Starting SarahMemoryVSM module test.")
        index = initialize_index(VECTOR_DIM)
        if index is None:
            logger_vsm.error("Index initialization failed. Exiting test.")
            exit(1)
        sample_vector = np.random.rand(VECTOR_DIM)
        if add_vector(sample_vector):
            logger_vsm.info("Sample vector added successfully.")
        else:
            logger_vsm.error("Failed to add sample vector.")
        # Add additional sample vectors
        for i in range(10):
            vec = np.random.rand(VECTOR_DIM)
            add_vector(vec)
        distances, indices = query_similar(sample_vector, k=5)
        if distances is not None and indices is not None:
            logger_vsm.info(f"Query results - Distances: {distances}, Indices: {indices}")
        else:
            logger_vsm.error("Query failed.")
        logger_vsm.info("SarahMemoryVSM module testing complete.")
    else:
        # Run Facial Recognition module test
        logger_fr.info("Starting Enhanced SarahMemoryFacialRecognition module test v6.4.")
        face_cascade = load_face_cascade()
        if face_cascade is None:
            logger_fr.error("Face cascade not loaded. Exiting module test.")
            sys.exit(1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error_msg = "Unable to access the webcam."
            logger_fr.error(error_msg)
            log_facial_event("Webcam Access Error", error_msg)
            sys.exit(1)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    warning_msg = "Failed to capture frame from webcam."
                    logger_fr.warning(warning_msg)
                    log_facial_event("Frame Capture Warning", warning_msg)
                    break
                # Run both Haar and simulated DNN detection (using same functions)
                faces_haar = detect_faces(frame, face_cascade)
                faces_dnn = detect_faces_dnn(frame)
                # For this test, we prioritize the DNN detection result
                faces = faces_dnn
                frame_with_faces = draw_faces(frame, faces)
                cv2.imshow("Enhanced Facial Recognition", frame_with_faces)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_facial_event("Test Exit", "User exited facial recognition test.")
                    break
        except Exception as e:
            logger_fr.error(f"Error during webcam processing: {e}")
            log_facial_event("Webcam Processing Error", f"Exception: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger_fr.info("Enhanced SarahMemoryFacialRecognition module testing complete.")
            log_facial_event("Test Complete", "Facial recognition module testing complete.")
def get_user_fer_state(frame=None):
    """
    Return a normalized FER dict. If webcam/FER is unavailable, use last known state.
    Keys: joy, anger, fear, sadness, curiosity, trust, valence (-1..1), arousal (0..1), primary
    """
    try:
        faces = []
        if 'detect_faces_dnn' in globals():
            faces = detect_faces_dnn(frame) if frame is not None else []
            engaged = min(1.0, float(len(faces)) / 3.0)
        else:
            engaged = 0.2
    except Exception:
        engaged = 0.2
        faces = []
    try:
        if not faces:
            from SarahMemoryDatabase import load_last_emotion_state
            state = load_last_emotion_state()
        else:
            state = {"joy":0.5,"anger":0.1,"fear":0.1,"sadness":0.2,"curiosity":0.4,"trust":0.5,
                     "valence":0.1,"arousal":0.3,"primary":"neutral"}
        state["arousal"] = max(state.get("arousal",0.3), engaged*0.6)
        return state
    except Exception:
        return {"joy":0.5,"anger":0.1,"fear":0.1,"sadness":0.1,"curiosity":0.4,"trust":0.4,
                "valence":0.0,"arousal":0.2,"primary":"neutral"}


# === [APPEND] Eyeglasses cascade accessor (used by SOBJE) ===
def get_eyeglasses_cascade_path():
    try:
        import cv2
        return cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    except Exception:
        return None
