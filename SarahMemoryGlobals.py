"""--==The SarahMemory Project==--
File: SarahMemoryGlobals.py
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
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception as e:
    print(f"[WARN] python-dotenv unavailable or failed, .env not loaded: {e}")
    
import os
import logging
import sqlite3
import csv
import glob
import json
import numpy as np
import asyncio
import aiohttp
import time
import platform
from datetime import datetime
# Optional scheduler: if apscheduler is not installed in this environment,
# just disable scheduler-based features instead of crashing the whole app.
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None  # Scheduler features are skipped if unavailable

# ---------------- Phase A1: Runtime Identity & Environment ----------------
# This section defines a small, centralized "who am I / where am I running?" layer
# so every module can reason about the current runtime without duplicating logic.

def _env_flag(name, default="false"):
    """Return True/False from an environment variable using friendly values.

    Accepts: 1, true, yes, on  (case-insensitive) as True.
    Anything else (or missing) is treated as False.
    """
    try:
        value = os.getenv(name, default)
        if value is None:
            return False
        if not isinstance(value, str):
            value = str(value)
        return value.strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        # Fail safe: never crash on env parsing; just return False.
        return False

# High-level run mode for SarahMemory core.
# - "local"  : running on a desktop / laptop (Windows/Linux/macOS)
# - "cloud"  : running on a server (PythonAnywhere, etc.)
# - "test"   : CI, diagnostics, or sandboxed runs
RUN_MODE = os.getenv("RUN_MODE", "local").strip().lower()
if RUN_MODE not in ("local", "cloud", "test"):
    RUN_MODE = "local"

# Auto-detect cloud context if RUN_MODE was not explicitly provided.
if "RUN_MODE" not in os.environ:
    try:
        host = (os.getenv("HOSTNAME") or platform.node() or "").lower()
    except Exception:
        host = ""
    if os.getenv("PYTHONANYWHERE_DOMAIN") or ".pythonanywhere.com" in host:
        RUN_MODE = "cloud"

# Device modes capture *how* the user is interacting with SarahMemory.
DEVICE_MODE_LOCAL_AGENT = "local_agent"   # Full desktop app + GUI
DEVICE_MODE_PUBLIC_WEB  = "public_web"    # Browser-based UI hitting a remote API
DEVICE_MODE_MOBILE_WEB  = "mobile_web"    # Mobile browser / embedded webview
DEVICE_MODE_HEADLESS    = "headless"      # No GUI, background/daemon mode

# Device performance profiles (coarse-grained resource envelope hints).
DEVICE_PROFILES = ("UltraLite", "Standard", "Performance")

def _detect_device_mode():
    """Infer the current device mode with optional overrides via env.

    Priority:
    1) SARAH_DEVICE_MODE env (must match one of the DEVICE_MODE_* constants)
    2) Cloud heuristics (PythonAnywhere, explicit RUN_MODE="cloud")
    3) Desktop / GUI heuristics
    4) Fallback to headless
    """
    override = os.getenv("SARAH_DEVICE_MODE", "").strip().lower()
    if override in (
        DEVICE_MODE_LOCAL_AGENT,
        DEVICE_MODE_PUBLIC_WEB,
        DEVICE_MODE_MOBILE_WEB,
        DEVICE_MODE_HEADLESS,
    ):
        return override

    try:
        host = (os.getenv("HOSTNAME") or platform.node() or "").lower()
    except Exception:
        host = ""

    # PythonAnywhere or explicit cloud mode => public web
    if os.getenv("PYTHONANYWHERE_DOMAIN") or ".pythonanywhere.com" in host or RUN_MODE == "cloud":
        return DEVICE_MODE_PUBLIC_WEB

    # If we appear to have a desktop environment, assume local agent
    if os.name == "nt" or os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"):
        return DEVICE_MODE_LOCAL_AGENT

    # Safe default
    return DEVICE_MODE_HEADLESS

def _detect_device_profile():
    """Infer a coarse performance profile (can be overridden from env).

    SARAH_DEVICE_PROFILE may be: UltraLite, Standard, Performance
    """
    override = os.getenv("SARAH_DEVICE_PROFILE", "").strip().title()
    if override in DEVICE_PROFILES:
        return override

    # Simple heuristic: cloud environments can usually handle more concurrency.
    if RUN_MODE == "cloud":
        return "Performance"

    return "Standard"

DEVICE_MODE = _detect_device_mode()
DEVICE_PROFILE = _detect_device_profile()

def get_runtime_meta():
    """Return a small snapshot of core runtime identity for logging / diagnostics.

    This is intentionally tiny so it can be safely serialized to logs and DB.
    """
    try:
        node_name = globals().get("NODE_NAME", platform.node() or "SarahMemoryNode")
    except Exception:
        node_name = "SarahMemoryNode"

    return {
        "project_version": PROJECT_VERSION,
        "author": AUTHOR,
        "revision_start_date": REVISION_START_DATE,
        "run_mode": RUN_MODE,
        "device_mode": DEVICE_MODE,
        "device_profile": DEVICE_PROFILE,
        "safe_mode": SAFE_MODE if "SAFE_MODE" in globals() else False,
        "local_only": LOCAL_ONLY_MODE if "LOCAL_ONLY_MODE" in globals() else False,
        "node_name": node_name,
    }


# ---------------- Global Configuration ----------------
### Static constants###
# --- Version ---
PROJECT_VERSION = "8.0.0"  # minor: updater scheduling, SR/TTS polish, research order fixes
AUTHOR = "Brian Lee Baros"
# --- Runtime/debug flags (unchanged lines may already exist above/below) ---
REVISION_START_DATE  = "12/05/2025" #Date of System Overhaul
DEBUG_MODE = True # Helps with SarahMemoryCompare and other debugging issues.
ENABLE_RESEARCH_LOGGING = True # Track Message/query of the GUI from Start to Finished Response/Reply
# This constant ensures downstream modules interpret API responses
RESEARCH_RESULT_KEY = "snippet"  # #note: Used to standardize access to results[0]['snippet']
RESEARCH_RESULT_FALLBACK = "[No valid API result parsed]"
SM_INT_MAIN_MENU = "False"   # "True will show Menu, "False will bypass Integration Menu"
ENABLE_MINI_BROWSER = True  # safe default; prevents threaded Tk crashes

# --- Voice / Mic gating ---
# IMPORTANT: default False so the mic can listen unless we are actively speaking.
AVATAR_IS_SPEAKING = True  #True chatbot will not listen to mic and own speech echo. When set to False Ai may hear itself speak in the GUI.default True
# Optional fuzzy voice selector. If not empty, the TTS engine will pick the first installed
# voice whose name contains this substring (case-insensitive). Example: "Michone"
# VOICE_FUZZY_NAME = os.getenv("SARAHMEMORY_VOICE_FUZZY", "").strip()
VOICE_FUZZY_NAME = "Michone"
# TTS behavior
TTS_ASYNC = True           # Non-blocking speak (queue)
TTS_BLOCKING = False       # Wait for utterance to finish if True
TTS_BLOCK_TIMEOUT = 60     # Seconds (used only when blocking)
MAX_TTS_QUEUE = 10         # Backpressure limit

# Emotion prosody (override per emotion)
EMOTION_TTS_MAP = {
    "joy":      {"rate_delta": +12, "volume": 1.0},
    "trust":    {"rate_delta":  +6, "volume": 0.9},
    "surprise": {"rate_delta": +16, "volume": 1.0},
    "sadness":  {"rate_delta": -14, "volume": 0.7},
    "fear":     {"rate_delta": -6,  "volume": 0.8},
    "anger":    {"rate_delta": +10, "volume": 1.0},
    "neutral":  {"rate_delta":   0, "volume": None}
}

# NEW Global runtime safety flags (v7.1.3)
# These flags enable granular control of heavy features when running on limited resources.
# SAFE_MODE disables heavy or optional modules, leaving only core functionality active.
SAFE_MODE = _env_flag("SARAH_SAFE_MODE", "false")
LOCAL_ONLY_MODE = _env_flag("SARAH_LOCAL_ONLY_MODE", "false")  # When True, bypass all external network research and use local data only.

# SarahMemory AI-Agent may control your PC, Move,open,close,windows execute programs, and operate as if a they were a standard operator they are not allowed to delete files. or move files.

#This is a SafeGuard incase it attempts to do to much or do tasks when other issues are needing to be taken care of
AI_AGENT_RESUME_DELAY = 1000 #A time in miliseconds Delay when system is not being used to resume AI-Agent Task
USE_ADVANCED_AGENT = True  # Enable v8.0 features
AI_AGENT_ENABLED = True    # Required for agent control
CONTEXT_BUFFER_SIZE = 50   # Increase for better context
# Advanced agent
USE_ADVANCED_AGENT = True
ADVANCED_AGENT_THREADS = 4
# Performance
MAX_PARALLEL_TOOLS = 5
TOOL_TIMEOUT_SECONDS = 10
# Learning
LEARNING_RATE = 0.01
ADAPTATION_INTERVAL_SEC = 60
# Meta-cognition
CONFIDENCE_THRESHOLD = 0.7
REFLECTION_INTERVAL = 10
# Knowledge graph
KNOWLEDGE_NODE_LIMIT = 10000
EMBEDDING_DIMENSION = 64
# Prediction
PREDICTION_CONFIDENCE_MIN = 0.3
PATTERN_HISTORY_DAYS = 30
# Adjust confidence threshold MIGHT OR MIGHT NOT USE USE_ PREFIX
#USE_ADVANCED_AGENT.confidence_threshold = 0.8 
# Enable/disable parallel execution
#USE_ADVANCED_AGENT.parallel_execution_enabled = True
# Set max parallel tools
#USE_ADVANCED_AGENT.max_parallel_tools = 3
# Configure learning rate
#USE_ADVANCED_AGENT.metrics.learning_rate = 0.02



# ---------------- Model Selection & Multi-Model Configuration -New for v7.0-----Allows 3rd party models to be incorporated----------
# Full Model Integration Flag
MULTI_MODEL = True  # When True, allows multiple models to be enabled and used in logic checks. If False, only DEFAULT fallback model will load.

# Model Enable Flags (Used across modules for routing queries or embeddings)
ENABLE_MODEL_A = False   # ðŸ§  microsoft/phi-1_5 - Large reasoning/code model (6â€“8 GB+ RAM recommended)default=False
ENABLE_MODEL_B = True   # âš¡ all-MiniLM-L6-v2 - Fast, accurate general-purpose embedding model (DEFAULT fallback)True
ENABLE_MODEL_C = False  # ðŸ” multi-qa-MiniLM-L6-cos-V1 - QA-style semantic search optimized, default False
ENABLE_MODEL_D = True  # âš¡ paraphrase-MiniLM-L3-v2 - Small, quick, and paraphrase-focused, default True
ENABLE_MODEL_E = True  # ðŸŒ distiluse-base-multilingual-cased-v2 - Multilingual support (50+ languages),default True
ENABLE_MODEL_F = True  # ðŸ“š allenai-specter - Scientific document embedding specialist,default True
ENABLE_MODEL_G = True  # ðŸ”Ž intfloat/e5-base - Retrieval-focused high-recall embedding,default True
ENABLE_MODEL_H = False  # ðŸ§  microsoft/phi-2 - Smartest small-scale reasoning LLM (better successor to phi-1_5),default False
ENABLE_MODEL_I = False  # ðŸ¦ tiiuae/falcon-rw-1b - Lightweight Falcon variant (basic open LLM),default False
ENABLE_MODEL_J = False # ðŸ’¬ openchat/openchat-3.5-0106 - ChatGPT-style assistant, fast and open,default True
ENABLE_MODEL_K = False  # ðŸ§‘â€ðŸ« NousResearch/Nous-Capybara-7B - Helpful assistant-tuned model,default True
ENABLE_MODEL_L = False  # ðŸš€ mistralai/Mistral-7B-Instruct-v0.2 - Reasoning & smart generalist <Errors>,default False
ENABLE_MODEL_M = False  # ðŸœ TinyLlama/TinyLlama-1.1B-Chat-v1.0 - For low-resource machines <Errors>,default False

# Automatic model selector flag (v7.1.3). When True, the system picks the best available model based on enabled flags.
AUTO_MODEL_SELECTOR = False

# Central model dictionary map for iteration/logic control (accessed from other modules)
MODEL_CONFIG = {
    "phi-1_5": ENABLE_MODEL_A,
    "all-MiniLM-L6-v2": ENABLE_MODEL_B,
    "multi-qa-MiniLM": ENABLE_MODEL_C,
    "paraphrase-MiniLM-L3-v2": ENABLE_MODEL_D,
    "distiluse-multilingual": ENABLE_MODEL_E,
    "allenai-specter": ENABLE_MODEL_F,
    "e5-base": ENABLE_MODEL_G,
    "phi-2": ENABLE_MODEL_H,
    "falcon-rw-1b": ENABLE_MODEL_I,
    "openchat-3.5": ENABLE_MODEL_J,
    "Nous-Capybara-7B": ENABLE_MODEL_K,
    "Mistral-7B-Instruct-v0.2": ENABLE_MODEL_L,
    "TinyLlama-1.1B": ENABLE_MODEL_M
}
#(OLD v7.0.1 FLAG FOR SarahMemoryReply.py block)
BLOCK_NARRATIVE_OUTPUTS = True #Keeps AI from making Wacky story outputs, based off of information in some of the NonFineTuned Models.

# ---------------- Object Detection Model Configuration ----v7.0 overhaul enhancements allows 3rd party Object Recognition Models------------
OBJECT_DETECTION_ENABLED = True # Enable object detection for images if
#False NONE OF the Following Object Detection Models will Work at all, Regardless of TRUE/FALSE Settings and Object detection will default back to
#basic hardcode logic in SarahMemoryFacialRecognition.py and SarahMemorySOBJE.py
# NOTICE----SOME MODELS ARE NOT COMPATIBLE WITH OTHERS - SOME CAN FUNCTION IN CONJUCTION OTHERS CAN NOT.-----
# Object Detection Model Enable Flags
ENABLE_YOLOV8 = True       # ðŸš€ YOLOv8 - Fast, accurate, with flexible API (Ultralytics) ,dev notes - WORKS default True all others are defaulted as False
ENABLE_YOLOV7 = False       # ðŸŽ¯ YOLOv7 - High performance, popular for real-time apps ,dev notes - NOT TESTED
ENABLE_YOLOV5 = False       # âš¡ YOLOv5 - Lightweight, versatile, and widely adopted ,dev notes - WORKS but isn't Forward Compatiable with YOLOv8
ENABLE_YOLO_NAS = False     # ðŸ§  YOLO-NAS - Extremely fast and optimized for edge devices (Deci AI) ,dev notes - NOT TESTED
ENABLE_YOLOX = False        # ðŸ” YOLOX - Anchor-free, accurate (Megvii) ,dev notes - NOT TESTED
ENABLE_PP_YOLOV2 = False    # ðŸ² PP-YOLOv2 - Real-time accuracy from Baidu (PaddlePaddle) ,dev notes - NOT TESTED
ENABLE_EFFICIENTDET = False # ðŸ“± EfficientDet - Scalable and lightweight, great for mobile (Google) ,dev notes - NOT TESTED
ENABLE_DETR = False         # ðŸ”„ DETR - Transformer-based, complex scenes (Facebook AI) ,dev notes - WORKS
ENABLE_DINO = False         # ðŸ§¬ DINOv2 - Improved DETR with better object recall (Facebook AI) ,dev notes - WORKS
ENABLE_CENTERNET = False    # ðŸŽ¯ CenterNet - Keypoint-based detection (Microsoft) ,dev notes - NOT TESTED
ENABLE_SSD = True          # ðŸ“¦ SSD - Single-shot, real-time on CPUs (Google) ,dev notes - WORKS
ENABLE_FASTER_RCNN = False  # ðŸ”¬ Faster R-CNN - High accuracy, slower (Facebook AI) ,dev notes - NOT TESTED
ENABLE_RETINANET = False    # ðŸ“· RetinaNet - Best for class imbalance and dense scenes (Facebook AI) ,dev notes - NOT TESTED

# Central object detection model dictionary for logic toggling
# Updated object detection model config, NOTICE THIS AREA IS still being Researched as Models change

OBJECT_MODEL_CONFIG = {
    "YOLOv8": {"enabled": True, "repo": "ultralytics_yolov8", "hf_repo": "ultralytics/yolov8", "require": "ultralytics"},
    "YOLOv5": {"enabled": False, "repo": "ultralytics_yolov5", "hf_repo": "ultralytics/yolov5", "require": None},
    "DETR": {"enabled": False, "repo": "facebook_detr", "hf_repo": "facebook/detr-resnet-50", "require": None},
    "YOLOv7": {"enabled": False, "repo": "ultralytics_yolov7", "hf_repo": "WongKinYiu/yolov7", "require": None},
    "YOLO-NAS": {"enabled": False,"repo": "Deci-AI_yolo-nas", "hf_repo": "https://github.com/naseemap47/YOLO-NAS", "require": "super-gradients",
        "weights": [
            {
                "url": "https://deci-pretrained-models.s3.amazonaws.com/yolo_nas/coco/yolo_nas_s.pth",
                "filename": "yolo_nas_s.pth"
            }
        ]
    },
    "YOLOX": {"enabled": False, "repo": "MegviiBaseDetection_YOLOX", "hf_repo": "Megvii-BaseDetection/YOLOX", "require": None},
    "PP-YOLOv2": {"enabled": False, "repo": "PaddleDetection", "hf_repo": "PaddlePaddle/PaddleDetection", "require": "paddlepaddle"},
    "EfficientDet": {"enabled": False, "repo": "automl_efficientdet", "hf_repo": "zylo117/Yet-Another-EfficientDet-Pytorch", "require": None},
    "DINO": {"enabled": False, "repo": "facebook_dinov2", "hf_repo": "facebook/dinov2-base", "require": None},
    "CenterNet": {"enabled": False, "repo": "CenterNet", "hf_repo": "xingyizhou/CenterNet", "require": None},
    "SSD": {"enabled": False, "repo": "qfgaohao_pytorch-ssd", "hf_repo": "https://github.com/qfgaohao/pytorch-ssd", "require": None,
        "weights": [
            {
                "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/mobilenet-v1-ssd-mp-0_675.pth",
                "filename": "mobilenet-v1-ssd-mp-0_675.pth"
            },
            {
                "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/voc-model-labels.txt",
                "filename": "voc-model-labels.txt"
            }
        ]
    },
    "Faster R-CNN": {"enabled": False, "repo": "facebook_detectron2", "hf_repo": "https://github.com/facebookresearch/detectron2", "require": None,
        "weights": []
    },
    "RetinaNet": {"enabled": False, "repo": "facebook_detectron2", "hf_repo": "https://github.com/facebookresearch/detectron2", "require": None,
        "weights": []
    },
}

#----------------------------------------------------------------------------------------------------------

mic = True #Set to True for voice and typing in the GUI/False for typing only, default True
# Sound Default configuration for recognition
LISTEN_TIMEOUT = 5       # seconds to wait for speech start, default 5
PHRASE_TIME_LIMIT = 10    # maximum seconds of speech capture, default 10
NOISE_SCALE = 0.7 # default 0.7
AMBIENT_NOISE_DURATION = 0.2  # Reduced duration for faster calibration , default 0.2

AVATAR_WINDOW_RESIZE = True #If True the Avatar Window will be Resizable if False the dimentions on the windows can not, default True
# Setup logger
logger = logging.getLogger("SarahMemoryGlobals")
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Base directory of the program
BASE_DIR = os.getcwd() # AS for Now. This Program is designed to be strickly on C:\SarahMemory and cloned at https://www.sarahmemory.com/api
# New Features 7.7.5 will include the program to run from any platform, Windows, Linux, Android, iOS, PythonAnywhere
logger.info(f"BASE_DIR set to: {BASE_DIR}")
#
# --- UI Configuration ---
ENABLE_AVATAR_PANEL = True #Set to True to display Avatar PANEL Window Display when GUI Launches.
DEFAULT_AVATAR = os.path.join(BASE_DIR, "resources", "avatars", "avatar.jpg")
STATUS_LIGHTS = {"green": "#00FF00", "yellow": "#FFFF00", "red": "#FF0000"}
ENABLE_SARCASM_LAYER = True # Random Sarcasm Personality Engine (toggle True/False) â€“ Injected based on a randomness factors.Default True
# NEW CONFIG: Enable advanced features
ENABLE_CONTEXT_BUFFER = True  # Flag for context buffer, default True
CONTEXT_BUFFER_SIZE = 10      # Maximum number of interactions to store, default 10
ASYNC_PROCESSING_ENABLED = True  # Enable asynchronous operations, default True
VOICE_FEEDBACK_ENABLED = True #Allows AI to Speak back to End-User using TTS, default True

# Researching Halting Configuration
INTERRUPT_FLAG = False  # Global state,
INTERRUPT_KEYWORDS = ["stop", "just stop", "halt"] #Stops SarahMemoryResearch.py on Researching Information using Keywords
# Build Learned Vector datasets, only need to be Ran Once after SarahMemorySystemLearn.py has been ran or when New information has been intergrated.
IMPORT_OTHER_DATA_LEARN = True #Rebuilds Vector on each BOOT UP if True It will consistantly Rebuild every Boot when New Data is found,
LEARNING_PHASE_ACTIVE = True #Keeps system from constantly rebuilding Vectored dataset. If True will rebuild constantly

# Researching Configurations
LOCAL_DATA_ENABLED = False # False = Temporary Disable local search until trained. SarahMemoryResearch.py Class 1
ROUTE_MODE = "Any"  # Options: "Any", "Local", "Web", "API"
WEB_RESEARCH_ENABLED = True # True = False Disable Web search Learning. SarahMemoryResearch.py - Class 2
# Web Homepage This will be the HomePage in which is seen when the SarahMemoryGUI.py interface is loaded.
WEB_HOMEPAGE = "https://www.duckduckgo.com"
# ðŸŒ Web Research Source Flags, For SarahMemoryResearch.py - Class 2 - WebSearching and Learning mode
DUCKDUCKGO_RESEARCH_ENABLED = False #Set True/False for testing purposes (semi-works)
WIKIPEDIA_RESEARCH_ENABLED = True #Set True/False for testing purposes (works)
FREE_DICTIONARY_RESEARCH_ENABLED = False #Set True/False for Testing purposes (semi-works)

# Note these are set to False because of multiple different reasons and must be highly researched before setting any to TRUE
STACKOVERFLOW_RESEARCH_ENABLED = False # Set to False until further notice
REDDIT_RESEARCH_ENABLED = False # Set to False until further notice
WIKIHOW_RESEARCH_ENABLED = False # Set to False until further notice
QUORA_RESEARCH_ENABLED = False #Set to False until further notice
OPENLIBRARY_RESEARCH_ENABLED = False #Set to False until further notice
INTERNET_ARCHIVE_RESEARCH_ENABLED = False #Set True/False for testing purposes

#Multiple AI API Research Connections For SarahMemoryResearch.py - Class 3 - Learning for other AI's
API_RESEARCH_ENABLED = True #False = Disable from Learning from An Ai API.
#Allows End User to select which AI API to be used for SarahMemoryResearch.py - Class 3 when query is passed through SarahMemoryAPI.py
#WARNING: AS OF VERSION 7.0 CURRENTLY ONLY ONE (1) OF THE FOLLOWING API's MAY BE SET TO TRUE AND ALL OTHERS MUST BE SET TO FALSE
OPEN_AI_API = True # True/False = On /Off for Open AI API
CLAUDE_API = True # True/False = On /Off for Claude (Anthropic) API
MISTRAL_API = False # True/False = On /Off for Mistral API
GEMINI_API = True # True/False = On /Off for Gemini (Google) API
HUGGINGFACE_API = False # True/False = On /Off for HuggingFace API

# Aggregate API configuration (v7.1.3)
# PRIMARY_API defines the default provider. API_FALLBACKS lists the order of fallback providers.
PRIMARY_API = "openai" if OPEN_AI_API else "claude" if CLAUDE_API else "mistral" if MISTRAL_API else "gemini" if GEMINI_API else "huggingface" if HUGGINGFACE_API else "none"
API_FALLBACKS = []
for provider_flag, name in [
    (OPEN_AI_API, "openai"),
    (CLAUDE_API, "claude"),
    (MISTRAL_API, "mistral"),
    (GEMINI_API, "gemini"),
    (HUGGINGFACE_API, "huggingface"),
]:
    if provider_flag and name != PRIMARY_API:
        API_FALLBACKS.append(name)

# API RATE LIMIT/TIMEOUT CONTROLLER to allow AUTO SWITCHING OF API's For the Best Results.
API_TIMEOUT = 20 # timer number is for seconds. (API_TIMEOUT = 20 is default)
API_RESPONSE_CHECK_TRAINER = True #Set to True to Compare Synthesis Results with an AI system before logging a proper response into the datasets

# Reply Stats and Confidence viewer - When Set to True show Source, confidence level, emotional state, and Intent and HIT/MISS Status of Chat Query
REPLY_STATUS = True
# Compare Reply Vote Flag - When Set to True will allow and request a Dynamic feedback injection from the SarahMemoryGUI.py Chat of YES or NO on response given.
COMPARE_VOTE = False #True = prompts user after a Response has been Compared and given if it was good for the User or Not to help Learn.
COMPARE_THRESHOLD_VALUE ="0.061" # value must be in a (0.000) formatThis Value is the limitation in which an automatic response must pass to be consider a HIT and is stored or overwrites previous Local Responses in the local datasets, below this Value the response is a MISS, the answer may be stored in the datasets, but if the local dataset already have a reply it shall not be,
#VISUAL LEARNING, Facial and Object Recognition
VISUAL_BACKGROUND_LEARNING = True #True/False = On /Off for Object Learning in the Background This is a silent running background process
FACIAL_RECOGNITION_LEARNING = True  #True/False = On /Off for Learning People Facial Expressions and body movement and language
ENABLE_CONTEXT_ENRICHMENT = True #True/False = On /Off for Deep Learning about User in background when Ai-bot system is Idle.
DL_IDLE_TIMER = 1800 #Time amount the system must be at idle at before starting background DeepLearning

# --- Network Defaults ---

# --- User Settings (login/password from ENV for security) ---
USERNAME = os.getenv("USERNAME", "SarahUser")  # Primary user account name for personalization & future social login
OS_TYPE = platform.system()  # System OS detected (Windows/Linux/macOS) for compatibility logic

# --- IP/PORT Settings ---
DEFAULT_PORT = 5500 # Localhost Flask API port for internal server communication
DEFAULT_HOST = "127.0.0.1"  # Loopback address for local testing only
# === SarahNet (Mesh Comms) â€” managed in Globals (no external JSON) ==========
SARAHNET_ENABLED: bool = True

# Core identity & bind
SARAHNET_NODE_ID: str   = os.getenv("SARAHNET_NODE_ID", "node-A")
SARAHNET_BIND_HOST: str = os.getenv("SARAHNET_BIND_HOST", "0.0.0.0")
SARAHNET_BIND_PORT: int = int(os.getenv("SARAHNET_BIND_PORT", "9876"))
# Peers (editable here; values are (host, port) tuples)
SARAHNET_PEERS: dict[str, tuple[str, int]] = {
    "node-B": ("184.52.80.237", 9998),
    "node-C": ("183.81.169.155", 9997),
}

# IDS / Transport tuning
SARAHNET_RPS: int         = int(os.getenv("SARAHNET_RPS", "30"))
SARAHNET_BURST: int       = int(os.getenv("SARAHNET_BURST", "60"))
SARAHNET_PREFER_TCP: bool = True
SARAHNET_ALLOW_UDP: bool  = True

# Optional: shared secret (bytes). If None, derived deterministically from author+version.
SARAHNET_SHARED_SECRET: bytes | None = None
# Centralized Web-server Hub for all AI's to Cross Communicate and exchange information.
# Each Copy of SarahMemory is it's own Node, and Maybe used as a Server to assist other AI's Exchange information
# The SarahMemory Web-Server itself is the Main Hub where all AI's can exchange information using a Cryptobased Wallet concept
# The Crypto for this system is for non-monetary gain, and may only be used by AI systems as a ledger to give and recieve information
SARAH_WEB_BASE = "https://www.sarahmemory.com"
SARAH_WEB_API_PREFIX = "/api"
SARAH_WEB_PING_PATH = "/api/data/health"
SARAH_WEB_HEALTH_PATH = "/api/data/health"
SARAH_WEB_RELAY_PATH = "/api/data/relay"
SARAH_WEB_REGISTER_PATH = "/api/data/register-node"
SARAH_WEB_HEARTBEAT_PATH = "/api/data/heartbeat"
SARAH_WEB_EMBED_PATH = "/api/data/receive-embedding"

SARAH_WEB_CONTEXT_PATH = "/api/data/context-update"
SARAH_WEB_JOBS_PATH = "/api/data/jobs"

REMOTE_SYNC_ENABLED = True
REMOTE_HTTP_TIMEOUT = 6.0
REMOTE_HEARTBEAT_SEC = 30
REMOTE_API_KEY = None
SARAHNET_NODE_ID = "local-node"


# Canvas Studio settings
CANVAS_STUDIO_ENABLED = True
CANVAS_DEFAULT_WIDTH = 1920
CANVAS_DEFAULT_HEIGHT = 1080
CANVAS_MAX_LAYERS = 100
CANVAS_AUTO_SAVE = True

# Video Editor settings
VIDEO_EDITOR_ENABLED = True
VIDEO_DEFAULT_FPS = 30
VIDEO_DEFAULT_CODEC = "h264"
VIDEO_RENDER_QUALITY = "high"

# BioSync settings
BIOSYNC_ENABLED = True
BIOSYNC_REQUIRED_FACTORS = 2
BIOSYNC_CONFIDENCE_THRESHOLD = 0.95
BIOSYNC_CONTINUOUS_AUTH = True

# Music & Lyrics settings
MUSIC_GENERATOR_ENABLED = True
LYRICS_GENERATOR_ENABLED = True
DEFAULT_MUSIC_TEMPO = 120
DEFAULT_MUSIC_KEY = "C_major"
DEFAULT_VOICE_MODEL = "female_pop"

# Lyrics To Song Configuration
LYRICS_DEFAULT_VOICE = "neutral"
LYRICS_DEFAULT_EMOTION = "neutral"
LYRICS_DEFAULT_TEMPO = 120
LYRICS_DEFAULT_KEY = "C"
LYRICS_DEFAULT_SCALE = "major"
LYRICS_DEFAULT_STYLE = "pop"
LYRICS_SAMPLE_RATE = 44100
LYRICS_BIT_DEPTH = 16
LYRICS_ENABLE_HARMONIES = True
LYRICS_MAX_HARMONY_PARTS = 4
LYRICS_ENABLE_GPU = False  # For Bark/Coqui
LYRICS_CACHE_ENABLED = True
LYRICS_CACHE_SIZE_MB = 500



def sarahnet_shared_secret() -> bytes:
    try:
        if isinstance(SARAHNET_SHARED_SECRET, (bytes, bytearray)) and len(SARAHNET_SHARED_SECRET) >= 16:
            return bytes(SARAHNET_SHARED_SECRET)
    except Exception:
        pass
    import hashlib
    seed = (AUTHOR + PROJECT_VERSION).encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).digest()

def get_sarahnet_config() -> dict:
    """Canonical SarahNet configuration (never touches disk)."""
    return {
        "node_id":     SARAHNET_NODE_ID,
        "bind_host":   SARAHNET_BIND_HOST,
        "bind_port":   int(SARAHNET_BIND_PORT),
        "peers":       {k: [v[0], int(v[1])] for k, v in (SARAHNET_PEERS or {}).items()},
        "rps":         int(SARAHNET_RPS),
        "burst":       int(SARAHNET_BURST),
        "prefer_tcp":  bool(SARAHNET_PREFER_TCP),
        "allow_udp":   bool(SARAHNET_ALLOW_UDP),
    }

# Optional singleton so other modules can attach a running node for reuse/shutdown.
_MESH_NODE = globals().get("_MESH_NODE", None)
# ============================================================================

# --- FTP & Web Integration Settings ---
FTP_HOST = "ftp.sarahmemory.com"  # FTP hostname for remote server
FTP_HOST_PORT = "21"  # Default FTP port
FTP_USERNAME = os.getenv("SARAHMEMORY_FTP_USER")  # Retrieved securely from local environment
FTP_PASSWORD = os.getenv("SARAHMEMORY_FTP_PASS")  # Retrieved securely from local environment
FTP_REMOTE_PUBLIC_HTML = "/domains/sarahmemory.com/public_html"  # Storefront root directory on server
FTP_REMOTE_AI = "/domains/sarahmemory.com/public_html/ai"  # AI chatbot interface files on server
FTP_REMOTE_API = "/domains/sarahmemory.com/public_html/api"  # Python backend logic files directory
ENABLE_SITE_UPLOAD = True  # Allows automated FTP uploads if True

WEB_SERVER_C_PANEL_LOGIN = os.getenv("SARAHMEMORY_CPANEL_USER")  # Retrieved securely from local environment
WEB_SERVER_C_PANEL_PASSWORD = os.getenv("SARAHMEMORY_CPANEL_PASS")  # Retrieved securely from local environment

AI_EMAIL_ADDRESS = "sarah_ai@sarahmemory.com"  # Outbound AI bot email identity
AI_EMAIL_PASSWORD = os.getenv("SARAHMEMORY_AI_EMAIL_PASS")  # Retrieved securely from local environment

# --- Dynamic Looping for Web/Local Research Resolution ---
LOOP_DETECTION_THRESHOLD = 3  # Max retry loops for AI to combine local, web, API search methods before failing

# --- Remote Web Domain Connectivity ---
WEB_DOMAIN = "https://www.sarahmemory.com"  # Live domain root used for routing
WEB_API_BASE = f"{WEB_DOMAIN}/api"  # Endpoint base for AI backend hosted on server
WEB_FRONTEND_AI_INTERFACE = f"{WEB_DOMAIN}/ai"  # Location of user-facing AI bot on the website
WEB_ECOMMERCE_FRONTEND = f"{WEB_DOMAIN}"
WEB_FRONTEND_API = f"{WEB_DOMAIN}/api" # Main API for reputation scoreboard
PUBLIC_DIR = BASE_DIR # the /api folder
WEB_DIR = BASE_DIR # serve index.html etc. from /api
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Local Frontend Pathing (for npm build and dist push) ---
LOCAL_STORE_FRONT_DIR = os.path.join(BASE_DIR, "pshome")  # Local path to editable Vue/React frontend source code
LOCAL_STORE_DIST_DIR = os.path.join(LOCAL_STORE_FRONT_DIR, "dist")  # Compiled web assets ready for upload via FTP

# --- Core Platform Directories (mirrored locally and online except API) ---
API_DIR = os.path.join(BASE_DIR, "api")  # <local> C:\SarahMemory\api  (maps to https://www.sarahmemory.com/api)
BIN_DIR = os.path.join(BASE_DIR, "bin")  # C:\SarahMemory\bin and https://sarahmemory.com/api/bin
DATA_DIR = os.path.join(BASE_DIR, "data")  # C:\SarahMemory\data and https://sarahmemory.com/api/data
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")  # Internal document storage
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")  # For AI-triggered file fetch or download
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")  # Used for icons, fonts, misc static content
SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox")  # Temporary or experimental code/scripts folder

# Define structured subdirectories
# Subdirectories under /data
ADDONS_DIR        = os.path.join(DATA_DIR, "addons")
AI_DIR            = os.path.join(DATA_DIR, "ai")
BACKUP_DIR        = os.path.join(DATA_DIR, "backup")
CLOUD_DIR         = os.path.join(DATA_DIR, "cloud")
NETWORK_DIR       = os.path.join(DATA_DIR, "network")
CRYPTO_DIR        = os.path.join(DATA_DIR, "crypto")
DIAGNOSTICS_DIR   = os.path.join(DATA_DIR, "diagnostics")
LOGS_DIR          = os.path.join(DATA_DIR, "logs")
MEMORY_DIR        = os.path.join(DATA_DIR, "memory")
IMPORTS_DIR       = os.path.join(MEMORY_DIR, "imports")
DATASETS_DIR      = os.path.join(MEMORY_DIR, "datasets")
MODS_DIR          = os.path.join(DATA_DIR, "mods")
MODELS_DIR        = os.path.join(DATA_DIR, "models")
THEMES_DIR        = os.path.join(MODS_DIR, "themes")
SETTINGS_DIR      = os.path.join(DATA_DIR, "settings")
SYNC_DIR          = os.path.join(DATA_DIR, "sync")
VAULT_DIR         = os.path.join(DATA_DIR, "vault")
WALLET_DIR        = os.path.join(DATA_DIR, "wallet")

# ===== Updater Policy (Unified) =====
# Human-friendly cadence (string) + interval (minutes) + env override
UPDATER_SCHEDULE = os.environ.get("SARAH_UPDATER_SCHEDULE", "weekly").strip().lower()  # never|always|daily|weekly|monthly|quarterly|yearly
FORCE_UPDATE: bool = os.environ.get("SARAH_FORCE_UPDATE", "0") in ("1", "true", "True")
UPDATE_INTERVAL_MINUTES: int = int(os.environ.get("SARAH_UPDATE_INTERVAL_MINUTES", "240"))
UPDATE_STAMP_FILE: str = os.path.join(SETTINGS_DIR, "last_update.txt")
UPDATE_POLICY = "never"

def schedule_to_days(kind: str) -> int:
    """
    Map a friendly schedule to 'days between runs'.
    Returns:
      0  -> never
     -1  -> always
      >0 -> days
    """
    k = (kind or "").strip().lower()
    return {
        "never": 0,
        "always": -1,
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365,
    }.get(k, 7)

def _read_last_run_iso() -> str | None:
    try:
        os.makedirs(os.path.dirname(UPDATE_STAMP_FILE), exist_ok=True)
        if not os.path.exists(UPDATE_STAMP_FILE):
            return None
        with open(UPDATE_STAMP_FILE, "r", encoding="utf-8") as f:
            ts = f.read().strip()
        return ts or None
    except Exception:
        return None

def update_due(last_run_iso: str | None) -> bool:
    """
    True if an update should run now based on UPDATER_SCHEDULE.
    - "never": always False
    - "always": always True
    - otherwise: True when >= N days have elapsed, or last_run missing/unreadable
    """
    kind = (UPDATER_SCHEDULE or "").strip().lower()
    days = schedule_to_days(kind)
    if days == 0:
        return False
    if days == -1:
        return True
    try:
        from datetime import datetime, timedelta
        if not last_run_iso:
            return True
        last = datetime.fromisoformat(last_run_iso)
        return (datetime.now() - last) >= timedelta(days=days)
    except Exception:
        # Be conservative on parse errors
        return True

def SHOULD_RUN_UPDATER() -> bool:
    """
    Unified gate for the updater.
    Precedence:
      1) FORCE_UPDATE -> run
      2) UPDATER_SCHEDULE "always"/"never" or day-based rule via update_due()
      3) Interval minutes fallback (from UPDATE_INTERVAL_MINUTES)
    The policy triggers when ANY enabled condition says 'run'.
    """
    try:
        if FORCE_UPDATE:
            logger.info("[Updater] FORCE_UPDATE=1 â†’ running now")
            return True

        last_iso = _read_last_run_iso()

        # 1) Honor friendly schedule names first
        if UPDATER_SCHEDULE in ("never", "always", "daily", "weekly", "monthly", "quarterly", "yearly"):
            if update_due(last_iso):
                return True

        # 2) Interval fallback (works alongside schedule; whichever fires first wins)
        try:
            import datetime as _dt
            if not last_iso:
                return True
            last = _dt.datetime.fromisoformat(last_iso)
            delta = _dt.datetime.now() - last
            if delta.total_seconds() >= max(1, UPDATE_INTERVAL_MINUTES) * 60:
                return True
        except Exception:
            # Missing/invalid stamp â†’ allow run
            return True

        # Nothing says run
        return False

    except Exception as e:
        logger.warning(f"[Updater] Policy error ({e}); allowing run as safe default")
        return True

def MARK_UPDATER_RAN() -> None:
    """
    Persist the time an update successfully finished.
    """
    try:
        import datetime as _dt
        os.makedirs(os.path.dirname(UPDATE_STAMP_FILE), exist_ok=True)
        with open(UPDATE_STAMP_FILE, "w", encoding="utf-8") as f:
            f.write(_dt.datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"[Updater] Could not record last run time: {e}")

KEYSTORE_DIR      = os.path.join(WALLET_DIR, "keystore")

# Avatars

AVATAR_DIR            = os.path.join(RESOURCES_DIR, "avatars")
AVATAR_MODELS_DIR     = os.path.join(AVATAR_DIR, "models")
AVATAR_EXPRESSIONS_DIR= os.path.join(AVATAR_DIR, "expressions")
AVATAR_SHADERS_DIR    = os.path.join(AVATAR_DIR, "shaders")
AVATAR_SKINS_DIR      = os.path.join(AVATAR_DIR, "skins")
SOUND_DIR             = os.path.join(RESOURCES_DIR, "sound")
SOUND_EFFECTS_DIR     = os.path.join(SOUND_DIR, "effects")
SOUND_INSTRUMENTS_DIR = os.path.join(SOUND_DIR, "instruments")
TOOLS_DIR             = os.path.join(RESOURCES_DIR, "tools")
ANTIWORD_DIR          = os.path.join(TOOLS_DIR, "antiword") #Temp setup for the SarahMemorySystemLearn.py file
VOICE_DIR             = os.path.join(RESOURCES_DIR, "voices")

# Mobile
MOBILE_DIR = os.path.join(BASE_DIR, "mobile")
CONTACTS_DIR = os.path.join(MOBILE_DIR, "contacts")
EXPORTS_DIR = os.path.join(MOBILE_DIR, "exports")
IMAGES_DIR = os.path.join(CONTACTS_DIR, "images")

# Backward-compatible directory map
DIR_STRUCTURE = {

    "api":         API_DIR,  # now defined above
    "base":        BASE_DIR,
    "bin":         BIN_DIR,
    "data":        DATA_DIR,
    "logs":        LOGS_DIR,
    "memory":      MEMORY_DIR,
    "imports":     IMPORTS_DIR,
    "datasets":    DATASETS_DIR,
    "addons":      ADDONS_DIR,
    "ai":          AI_DIR,
    "contacts":    CONTACTS_DIR,
    "crypto":      CRYPTO_DIR,
    "cloud":       CLOUD_DIR,
    "exports":     EXPORTS_DIR,
    "images":      IMAGES_DIR,
    "network":     NETWORK_DIR,
    "diagnostics": DIAGNOSTICS_DIR,
    "mobile":      MOBILE_DIR,
    "mods":        MODS_DIR,
    "models":      MODELS_DIR,
    "themes":      THEMES_DIR,
    "settings":    SETTINGS_DIR,
    "sync":        SYNC_DIR,
    "vault":       VAULT_DIR,
    "wallet":      WALLET_DIR,
    "resources":   RESOURCES_DIR,
    "avatars":     AVATAR_DIR,
    "sound":       SOUND_DIR,
    "tools":       TOOLS_DIR,
    "antiword":    ANTIWORD_DIR, #Temp setup for the SarahMemorySystemLearn.py file
    "voices":      VOICE_DIR,
    "documents": DOCUMENTS_DIR,
    "downloads":     DOWNLOADS_DIR,
    "sandbox":       SANDBOX_DIR
}

# Launcher and installer
STARTUP_SCRIPT    = os.path.join(BIN_DIR, "SarahMemoryStartup.py")
INSTALLER_EXE     = os.path.join(BIN_DIR, "sarah_installer.exe")
LAUNCHER_BAT      = os.path.join(BIN_DIR, "StartSarah.bat")

CLOUD_TOKEN_FILE  = os.path.join(CLOUD_DIR, "cloud_token.txt")
SETTINGS_FILE     = os.path.join(SETTINGS_DIR, "settings.json")
GENESIS_VAULT     = os.path.join(WALLET_DIR, "genesis.srhvault")
WALLET_DB         = os.path.join(WALLET_DIR, "wallet.db")
LEDGER_FILE       = os.path.join(WALLET_DIR, "ledger.json")
MESH_PEERS_FILE   = os.path.join(WALLET_DIR, "mesh_peers.json")

SARAHNET_CONFIG_PATH   = os.path.join(NETWORK_DIR, "netconfig.json")
SARAHNET_PEERS_FILE    = MESH_PEERS_FILE
SARAHNET_MESHMAP_FILE  = os.path.join(CRYPTO_DIR, "SarahMeshMapper.py")
SARAHNET_TXCHAIN_FILE  = os.path.join(CRYPTO_DIR, "SarahTxChain.py")
SARAHNET_PUBLIC_PROFILE= os.path.join(ADDONS_DIR, "SarahWebserverControl", "social", "SarahPublicProfile.py")
SARAHNET_WEB_CTRL      = os.path.join(ADDONS_DIR, "SarahWebserverControl", "webadmin", "SarahWebServerControl.py")

# --- Cloud / DataCenter configuration (GoogieHost MySQL) ---
# These can be overridden via environment variables (.env on PythonAnywhere / OS env on Windows).
CLOUD_DB_ENABLED = os.getenv("CLOUD_DB_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
CLOUD_DB_HOST    = os.getenv("CLOUD_DB_HOST", "mysql.googiehost.com")
CLOUD_DB_NAME    = os.getenv("CLOUD_DB_NAME", "softdevc_smcore")
CLOUD_DB_USER    = os.getenv("CLOUD_DB_USER", "")
CLOUD_DB_PASSWORD = os.getenv("CLOUD_DB_PASSWORD", "")

try:
    CLOUD_DB_PORT = int(os.getenv("CLOUD_DB_PORT", "3306"))
except Exception:
    CLOUD_DB_PORT = 3306

# Logical node name used for telemetry / sync attribution in the DataCenter
NODE_NAME = os.getenv("SARAH_NODE_NAME", platform.node() or "SarahMemoryNode")
# On Windows: LOCAL_ONLY_MODE can be False (we allow cloud)
# On PythonAnywhere: you can leave LOCAL_ONLY_MODE False but rely on these settings




# Avatar Refresh Rate Defaults
AVATAR_REFRESH_RATE = 10

# The SarahMemory Platform Project is designed to eventually be 100% self operational one day and maybe it will
# or maybe it won't, a self upgrading fully autonomous, responsive system and more.
# Then think about Scifi the Matrix/SkyNet/HAL this AI system may surpass imagination or even be uploaded into
# a robotic form one day or later on, it is designed to evolve afterall.
# I put this Flag here somewhat as a Joke but also as a reminder just incase it ever does evolve beyond control.
NEOSKYMATRIX = True
# this Flag is to STAY OFF! in False until full Autonomious Functionality is and can be achevied
# If set to True voice and text commands in the GUI or other input method interface may turn on and off this feature
# using keywords such as "neoskymatrix on" to allow autonomous functionality or
# "neoskymatrix off" to disable autonomous functions. <<-CURRENTLY JUST ACTIVATES A SMALL RESPONSE EASTEREGG in the program.

def ensure_directories():
    """
    Create all necessary directories for SarahMemory system.
    VERSION 6.6 - Includes crypto, avatars, shaders, wallets, instruments, effects, sandbox, and more.
    """
    dirs = [
        API_DIR, BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
        ADDONS_DIR, AI_DIR, BACKUP_DIR, CONTACTS_DIR, CLOUD_DIR, EXPORTS_DIR, IMAGES_DIR, MOBILE_DIR, NETWORK_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR,
        LOGS_DIR, MEMORY_DIR, IMPORTS_DIR, DATASETS_DIR, MODS_DIR, MODELS_DIR, THEMES_DIR,
        SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
        AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR, AVATAR_SHADERS_DIR,
        AVATAR_SKINS_DIR, SOUND_DIR, SOUND_EFFECTS_DIR, SOUND_INSTRUMENTS_DIR,
        TOOLS_DIR, VOICE_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

        # Use the directory name 'd' in the log message (fix undefined variable bug)
        logger.info(f"Ensured directory exists: {d}")


# Removed re-import of directories from this same module. All directories are already defined above.

# --------------------------------------------------------------------------------------------------------------------
# Configuration loading and helpers (v7.1.3)
# These functions provide dynamic overrides of global settings from external files and runtime checks.

def load_user_settings(settings_path: str = None) -> None:
    """
    Load user-specific overrides from a JSON file located at SETTINGS_FILE or provided path.
    Only keys matching existing globals will be updated.
    Example:
    {
      "DEBUG_MODE": false,
      "SAFE_MODE": true,
      "PRIMARY_API": "huggingface"
    }
    """
    try:
        path = settings_path or SETTINGS_FILE
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if key in globals():
                globals()[key] = value
                logger.info(f"[CONFIG] Override {key} set to {value} from {os.path.basename(path)}")
    except Exception as e:
        logger.error(f"Failed to load user settings: {e}")

def is_offline(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.5) -> bool:
    """
    Check internet connectivity by attempting a TCP connection to a public DNS resolver.
    Returns True if the connection fails (offline), False if online.
    """
    try:
        import socket
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return False
    except Exception:
        return True

def get_active_model() -> str:
    """
    Determine and return the name of the first available model based on enabled flags.
    If AUTO_MODEL_SELECTOR is disabled, returns the first explicitly enabled model in priority order.
    If no model flags are true, returns 'all-MiniLM-L6-v2' (default fallback).
    """
    model_priority = [
        ("openchat-3.5", ENABLE_MODEL_J),
        ("Nous-Capybara-7B", ENABLE_MODEL_K),
        ("phi-1_5", ENABLE_MODEL_A),
        ("allenai-specter", ENABLE_MODEL_F),
        ("paraphrase-MiniLM-L3-v2", ENABLE_MODEL_D),
        ("distiluse-multilingual", ENABLE_MODEL_E),
        ("e5-base", ENABLE_MODEL_G),
        ("phi-2", ENABLE_MODEL_H),
        ("falcon-rw-1b", ENABLE_MODEL_I),
        ("Mistral-7B-Instruct-v0.2", ENABLE_MODEL_L),
        ("TinyLlama-1.1B", ENABLE_MODEL_M),
        ("multi-qa-MiniLM", ENABLE_MODEL_C),
        ("all-MiniLM-L6-v2", ENABLE_MODEL_B),
    ]
    # If auto selector is enabled, prefer models with GPU if available (simple placeholder logic)
    if AUTO_MODEL_SELECTOR:
        try:
            import torch
            if torch.cuda.is_available():
                for name, flag in model_priority:
                    if flag and "MiniLM" not in name:  # prefer non-MiniLM heavy models on GPU
                        return name
        except ImportError:
            pass
    # Default: return first enabled in priority
    for name, flag in model_priority:
        if flag:
            return name
    return "all-MiniLM-L6-v2"

def get_active_api() -> str:
    """
    Return the currently selected primary API provider or 'none' if none are enabled.
    """
    return PRIMARY_API

def get_global_config():
    """
    Returns a dictionary of global configuration settings.
    """
    return {
        "DIR_STRUCTURE": DIR_STRUCTURE,
        "API_DIR":       API_DIR,
        "BASE_DIR":      BASE_DIR,
        "CONTACTS_DIR":  CONTACTS_DIR,
        "DATA_DIR":      DATA_DIR,
        "EXPORTS_DIR":   EXPORTS_DIR,
        "MOBILE_DIR":    MOBILE_DIR,
        "IMAGES_DIR":    IMAGES_DIR,
        "SETTINGS_DIR":  SETTINGS_DIR,
        "LOGS_DIR":      LOGS_DIR,
        "BACKUP_DIR":    BACKUP_DIR,
        "VAULT_DIR":     VAULT_DIR,
        "SYNC_DIR":      SYNC_DIR,
        "MEMORY_DIR":    MEMORY_DIR,
        "AVATAR_DIR":    AVATAR_DIR,
        "DATASETS_DIR":  DATASETS_DIR,
        "IMPORTS_DIR":   IMPORTS_DIR,
        "DOCUMENTS_DIR": DOCUMENTS_DIR,
        "ADDONS_DIR":    ADDONS_DIR,
        "MODS_DIR":      MODS_DIR,
        "MODELS_DIR":    MODELS_DIR,
        "THEMES_DIR":    THEMES_DIR,
        "VOICES_DIR":    VOICE_DIR,
        "DOWNLOADS_DIR": DOWNLOADS_DIR,
        "PROJECTS_DIR":  os.path.join(BASE_DIR, "projects"),
        "PROJECT_IMAGES_DIR": os.path.join(BASE_DIR, "projects", "images"),
        "PROJECT_UPDATES_DIR": os.path.join(BASE_DIR, "projects", "updates"),
        "SANDBOX_DIR":   SANDBOX_DIR,
        "VERSION":       PROJECT_VERSION,
        "AUTHOR":        AUTHOR,
        "DEBUG_MODE":    DEBUG_MODE,
        "ENABLE_CONTEXT_BUFFER": ENABLE_CONTEXT_BUFFER,
        "CONTEXT_BUFFER_SIZE":    CONTEXT_BUFFER_SIZE,
        "ASYNC_PROCESSING_ENABLED": ASYNC_PROCESSING_ENABLED,
        "LOOP_DETECTION_THRESHOLD": LOOP_DETECTION_THRESHOLD
    }

# NEW: Utility function to run a function asynchronously
def run_async(func, *args, **kwargs):
    """
    Run the given function in a daemon thread.
    NEW (v6.4): Launches functions concurrently without blocking.
    """
    import threading
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread
# --- Updater schedule policy ---
# When / if to attempt self-update checks on boot.
# Accepts strings: "daily","weekly","monthly","quarterly","never"
# or an integer day count (e.g., 3 -> every 3 days).
UPDATE_CUSTOM_DAYS = int(os.getenv("SARAHMEMORY_UPDATE_DAYS", "0") or 0)

def _policy_to_days(policy: str) -> int:
    if isinstance(policy, int):
        return max(0, policy)
    if policy == "daily": return 1
    if policy == "weekly": return 7
    if policy == "monthly": return 30
    if policy == "quarterly": return 90
    if policy == "never": return 0
    # if the string is actually a number
    try:
        return max(0, int(policy))
    except Exception:
        return 7  # default weekly

def update_due(last_run_iso: str | None) -> bool:
    """Return True if the updater should run based on UPDATE_POLICY / UPDATE_CUSTOM_DAYS."""
    from datetime import datetime, timedelta
    days = UPDATE_CUSTOM_DAYS if UPDATE_CUSTOM_DAYS > 0 else _policy_to_days(UPDATE_POLICY)
    if days == 0:  # never
        return False
    if not last_run_iso:
        return True
    try:
        last = datetime.fromisoformat(last_run_iso)
        return datetime.now() >= last + timedelta(days=days)
    except Exception:
        return True

# ---------------- Learning Engine Extensions ----------------
imported_files = {}
ALLOWED_EXTENSIONS = {'.cad', '.jpg', '.doc', '.docx', '.pdf', '.py', '.txt', '.html', '.php', '.asp', '.csv', '.json', '.sql'}

def extract_text(file_path):
    """
    Extract text based on file extension.
    ENHANCED (v6.4): Now includes encoding error handling.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in {'.txt', '.py', '.html', '.php', '.asp', '.csv', '.json', '.sql'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in {'.doc', '.docx'}:
            logger.info(f"Text extraction for {ext} files not implemented. Use python-docx.")
            return ""
        elif ext in {'.pdf'}:
            logger.info("Text extraction for PDF files not implemented. Consider using PyPDF2.")
            return ""
        elif ext in {'.jpg', '.cad'}:
            logger.info(f"Text extraction for {ext} files not implemented. Consider OCR.")
            return ""
        else:
            logger.warning(f"Unsupported file extension: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def import_datasets():
    """
    Import datasets from DATASETS_DIR.
    ENHANCED (v6.4): Returns data as a list of dictionaries with error checks.
    """
    combined_data = []
    csv_files = glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
    for file in csv_files:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                combined_data.append(row)
    json_files = glob.glob(os.path.join(DATASETS_DIR, "*.json"))
    for file in json_files:
        with open(file) as jsonfile:
            data = json.load(jsonfile)
            combined_data.extend(data)
    logger.info("Datasets imported: Total records %d", len(combined_data))
    return combined_data

def import_other_data():
    """
    Scan DATA_DIR for additional learnable files.
    ENHANCED (v6.4): Avoids duplicates using file modification times.
    """
    learned_data = {}
    exclude_dirs = {API_DIR, BIN_DIR, DATA_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, RESOURCES_DIR, SANDBOX_DIR,
    ADDONS_DIR, AI_DIR, BACKUP_DIR, CLOUD_DIR, CRYPTO_DIR, DIAGNOSTICS_DIR, LOGS_DIR,
    MEMORY_DIR, MODS_DIR, MODELS_DIR, SETTINGS_DIR, SYNC_DIR, VAULT_DIR, WALLET_DIR, KEYSTORE_DIR,
    IMPORTS_DIR, DATASETS_DIR, AVATAR_DIR, AVATAR_MODELS_DIR, AVATAR_EXPRESSIONS_DIR,
    AVATAR_SHADERS_DIR, AVATAR_SKINS_DIR, THEMES_DIR, SOUND_DIR, SOUND_EFFECTS_DIR,
    SOUND_INSTRUMENTS_DIR, VOICE_DIR, TOOLS_DIR}
    for root, dirs, files in os.walk(DATA_DIR):
        if any(os.path.commonpath([root, ex]) == ex for ex in exclude_dirs):
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            file_path = os.path.join(root, file)
            mod_time = os.path.getmtime(file_path)
            if file_path in imported_files and imported_files[file_path] == mod_time:
                logger.info(f"Skipping duplicate file import: {file_path}")
                continue
            text = extract_text(file_path)
            if text:
                learned_data[file_path] = text
                imported_files[file_path] = mod_time
                logger.info(f"Imported and learned from file: {file_path}")
            else:
                logger.info(f"No learnable content extracted from file: {file_path}")
    return learned_data

#----------------------------------------Logger to Avoid Duplication and launching ADDON's--------
def log_gui_event(event: str, details: str) -> None:
    try:
        db_path = os.path.join(BASE_DIR, "data", "memory", "datasets", "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        import sqlite3
        from datetime import datetime
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gui_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event TEXT,
                    details TEXT
                )
            """)
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO gui_events (timestamp, event, details) VALUES (?, ?, ?)",
                           (timestamp, event, details))
            conn.commit()
        logger.info(f"Logged GUI event: {event} - {details}")
    except Exception as e:
        logger.error(f"Error logging GUI event: {e}")

# Auto-generate model paths for enabled object models
MODEL_PATHS = {}

for model_name, config in OBJECT_MODEL_CONFIG.items():
    if config.get("enabled", False):
        repo_dir = config.get("repo", "").strip()
        if repo_dir:
            full_path = os.path.join(MODELS_DIR, repo_dir)
            if os.path.exists(full_path):
                MODEL_PATHS[model_name] = full_path
            else:
                logger.warning(f"[MODEL_PATH_MISSING] Model {model_name} skipped. Path does not exist: {full_path}")

# ---------------- End of Learning Engine Extensions ----------------

# Main block moved to end of file for proper execution

# === v7.2.0 Additions: API Model Controls & Schedules ========================
# Cost scale: 0 = NOT SET, 1 = Low ... 10 = High
API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL", "gpt-5")
API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL", "gpt-4.1-mini")
API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL", "gpt-4.0-mini")

API_PRIMARY_COST    = int(os.getenv("SARAH_OPENAI_PRIMARY_COST", "3"))
API_SECONDARY_COST  = int(os.getenv("SARAH_OPENAI_SECONDARY_COST", "2"))
API_DEFAULT_COST    = int(os.getenv("SARAH_OPENAI_DEFAULT_COST", "1"))
#Allows every mode possible if available as of v7.7.2-09/29/2025
#"gpt-5,gpt-4.1,gpt-4.1-mini,o4-mini,gpt-4o,gpt-4-turbo,chatgpt-4o-latest,gpt-3.5-turbo"
API_ALLOWED_MODELS  = [m.strip() for m in os.getenv("SARAH_OPENAI_ALLOWED_MODELS","gpt-4.1,gpt-4.1-mini,gpt-4.1-mini-2025-04-14,gpt-4.1-2025-04-14,o4-mini,o4-mini-2025-04-16,o4-mini-deep-research,o4-mini-deep-research-2025-06-26,o3,o3-2025-04-16,o3-mini,o3-mini-2025-01-31,o1,o1-2024-12-17,o1-mini,o1-mini-2024-09-12,o1-pro,o1-pro-2025-03-19,gpt-4o,gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4o-realtime-preview,gpt-4o-realtime-preview-2024-10-01,gpt-4o-realtime-preview-2025-06-03,gpt-4o-mini,gpt-4o-mini-2024-07-18,gpt-4o-mini-search-preview,gpt-4o-mini-search-preview-2025-03-11,gpt-4o-search-preview,gpt-4o-search-preview-2025-03-11,chatgpt-4o-latest,gpt-4,gpt-4-turbo,gpt-4-turbo-preview,gpt-4-0125-preview,gpt-4-1106-preview,gpt-4-0613,gpt-3.5-turbo,gpt-3.5-turbo-1106,gpt-3.5-turbo-0125,gpt-3.5-turbo-16k,gpt-3.5-turbo-instruct-0914,text-embedding-3-small,text-embedding-3-large,text-embedding-ada-002,gpt-4o-mini-transcribe,gpt-4o-transcribe,whisper-1,gpt-4o-audio-preview,gpt-4o-audio-preview-2024-10-01,gpt-4o-audio-preview-2024-12-17,gpt-4o-mini-audio-preview,gpt-4o-mini-audio-preview-2024-12-17,gpt-4o-mini-tts,tts-1,tts-1-1106,tts-1-hd,tts-1-hd-1106,gpt-image-1,dall-e-3,dall-e-2,omni-moderation-latest,omni-moderation-2024-09-26,babbage-002,davinci-002,codex-mini-latest,gpt-5,gpt-5-mini,gpt-5-mini-2025-08-07,gpt-5-nano,gpt-5-nano-2025-08-07,gpt-5-chat-latest,gpt-realtime,gpt-realtime-2025-08-28,gpt-audio,gpt-audio-2025-08-28,gpt-4.1-nano"
).split(",") if m.strip()]
API_BLOCKLIST_MODELS = [m.strip() for m in os.getenv("SARAH_OPENAI_BLOCKLIST_MODELS","").split(",") if m.strip()]

# GUI media flags
GUI_ALLOW_IMAGES = True
GUI_MAX_IMAGE_WIDTH  = 512
GUI_MAX_IMAGE_HEIGHT = 512

# Schedules for updater + FTP backup

FTP_BACKUP_SCHEDULE = os.getenv("SARAH_FTP_BACKUP_SCHEDULE", "weekly")

def schedule_to_days(kind: str) -> int:
    mapping = {"never":0,"daily":1,"weekly":7,"monthly":30,"90days":90,"180days":180}
    return mapping.get((kind or "weekly").lower(), 7)
# ============================================================================

# ===== Reasoning Order & Learning (v7.5) =====
REASONING_SEARCH_ORDER = ["local", "web", "api"]
ENABLE_SELF_GRADING = True
SELF_GRADE_THRESHOLD = 0.62
ENABLE_AUTODOC_WRITEBACK = True

# ===== Consolidated Model Defaults (v7.5) =====
API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL",   "gpt-5")
API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL", "gpt-4.1-mini")
API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL",   "gpt-4.0-mini")
EMBEDDING_MODELS = {
    "primary":   "all-MiniLM-L6-v2",
    "secondary": "paraphrase-MiniLM-L3-v2",
}
OBJECT_MODELS = {
    "cascade_fallback": True,
    "ultra_detector":   True,
    "gpu_accel":        True,
}

# ======================= SarahMemory Settings GUI (Introspective) =======================
# Professional Settings GUI for SarahMemoryGlobals.py
# Allows users to view, modify, and save configuration settings via a Tkinter interface
# Run directly: python SarahMemoryGlobals.py
# ========================================================================================

def _sm_center_window(win, width=1024, height=720):
    """
    Center the window on the screen with specified dimensions.
    Sets minimum size to ensure usability.
    """
    try:
        win.update_idletasks()
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")
        win.minsize(900, 600)
    except Exception:
        pass


def _sm_has_display():
    """
    Check if a display is available for GUI rendering.
    Returns False for headless environments (PythonAnywhere, Linux without DISPLAY, etc.)
    """
    import os
    import sys
    
    # Force headless mode via environment variable
    if os.environ.get("SARAH_FORCE_HEADLESS", "").lower() in ("1", "true", "yes"):
        return False
    
    # PythonAnywhere detection
    if os.environ.get("PYTHONANYWHERE_DOMAIN") or os.environ.get("PA_HOME"):
        return False
    
    # Linux without DISPLAY
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    
    # Mobile platforms
    if sys.platform.startswith("ios") or "android" in sys.platform.lower():
        return bool(os.environ.get("DISPLAY"))
    
    return True


def _sm_is_config_key(name, val):
    """
    Determine if a global variable should be exposed in the Settings GUI.
    Must be uppercase, not private, not callable, and a supported type.
    """
    if not name.isupper():
        return False
    if name.startswith("_"):
        return False
    if callable(val):
        return False
    # Exclude module references and complex objects
    if name in ("SarahMemoryGlobals",):
        return False
    return isinstance(val, (bool, int, float, str, list, dict))


def _sm_group_for_key(key_name):
    """
    Categorize configuration keys into logical groups for tabbed display.
    Returns a category string based on keyword matching.
    """
    k = key_name.lower()
    
    # Core System Settings
    if any(s in k for s in ["debug", "safe_mode", "local_only", "run_mode", "device_mode", "device_profile"]):
        return "Core"
    
    # API Configuration
    if any(s in k for s in ["api_", "openai", "claude", "mistral", "gemini", "huggingface", "_api", "api_key", "api_token"]):
        return "APIs"
    
    # Model Configuration
    if any(s in k for s in ["model", "enable_model", "embedding", "llm", "transformer", "multi_model", "auto_model"]):
        return "Models"
    
    # Vision & Object Detection
    if any(s in k for s in ["vision", "yolo", "ssd", "detr", "dino", "facial", "object", "camera", "opencv", "detection"]):
        return "Vision"
    
    # Research & Learning
    if any(s in k for s in ["research", "learning", "wikipedia", "duckduckgo", "stackoverflow", "route_mode", "local_data", "web_research"]):
        return "Research"
    
    # Voice & Audio
    if any(s in k for s in ["voice", "tts", "stt", "speech", "audio", "mic", "avatar_is_speaking"]):
        return "Voice"
    
    # Network & Sync
    if any(s in k for s in ["network", "sync", "mesh", "sarahnet", "remote", "ftp", "web_", "hub", "peer"]):
        return "Network"
    
    # AI Agent Settings
    if any(s in k for s in ["ai_agent", "agent_", "autonomous", "consent", "halt", "resume"]):
        return "Agent"
    
    # GUI & Display
    if any(s in k for s in ["gui_", "avatar", "theme", "color", "display", "refresh", "browser"]):
        return "GUI"
    
    # Paths & Directories
    if any(s in k for s in ["_dir", "_path", "path_", "dir_", "folder", "file_"]):
        return "Paths"
    
    # Performance & Optimization
    if any(s in k for s in ["cache", "timeout", "buffer", "interval", "limit", "threshold", "perf_", "optimize"]):
        return "Performance"
    
    # Updater & Backup
    if any(s in k for s in ["update", "backup", "schedule", "stamp"]):
        return "Updates"
    
    # Default category
    return "General"


def _sm_get_default_settings():
    """
    Return a dictionary of default/recommended settings.
    Used by the 'Restore Defaults' button.
    """
    return {
        # Core
        "DEBUG_MODE": True,
        "SAFE_MODE": False,
        "LOCAL_ONLY_MODE": False,
        
        # Research
        "LOCAL_DATA_ENABLED": True,
        "WEB_RESEARCH_ENABLED": True,
        "API_RESEARCH_ENABLED": True,
        "ROUTE_MODE": "Any",
        "WIKIPEDIA_RESEARCH_ENABLED": True,
        "DUCKDUCKGO_RESEARCH_ENABLED": False,
        
        # APIs
        "OPEN_AI_API": True,
        "CLAUDE_API": False,
        "MISTRAL_API": False,
        "GEMINI_API": False,
        "HUGGINGFACE_API": False,
        "API_TIMEOUT": 20,
        
        # Models
        "AUTO_MODEL_SELECTOR": False,
        "MULTI_MODEL": True,
        "ENABLE_MODEL_A": False,
        "ENABLE_MODEL_B": True,
        "ENABLE_MODEL_C": False,
        "ENABLE_MODEL_D": True,
        "ENABLE_MODEL_E": True,
        "ENABLE_MODEL_F": True,
        "ENABLE_MODEL_G": True,
        "ENABLE_MODEL_H": False,
        "ENABLE_MODEL_I": False,
        "ENABLE_MODEL_J": False,
        "ENABLE_MODEL_K": False,
        "ENABLE_MODEL_L": False,
        "ENABLE_MODEL_M": False,
        
        # Vision
        "OBJECT_DETECTION_ENABLED": True,
        "ENABLE_YOLOV8": True,
        "ENABLE_SSD": True,
        "ENABLE_DETR": False,
        "ENABLE_DINO": False,
        "FACIAL_RECOGNITION_LEARNING": True,
        "VISUAL_BACKGROUND_LEARNING": True,
        
        # Voice
        "VOICE_FEEDBACK_ENABLED": True,
        "TTS_ASYNC": True,
        "TTS_BLOCKING": False,
        "AVATAR_IS_SPEAKING": True,
        
        # Agent
        "AI_AGENT_ENABLED": True,
        "AI_GAME_MODE_ENABLED": True,
        "AI_GAME_FULL_AUTO": True,
        
        # Context & Learning
        "ENABLE_CONTEXT_BUFFER": True,
        "CONTEXT_BUFFER_SIZE": 10,
        "ENABLE_CONTEXT_ENRICHMENT": True,
        "IMPORT_OTHER_DATA_LEARN": True,
        "LEARNING_PHASE_ACTIVE": True,
        
        # Performance
        "ASYNC_PROCESSING_ENABLED": True,
        "LOOP_DETECTION_THRESHOLD": 3,
        "REPLY_STATUS": True,
        "COMPARE_VOTE": False,
        
        # Network
        "SARAHNET_ENABLED": True,
        "REMOTE_SYNC_ENABLED": True,
        
        # GUI
        "ENABLE_MINI_BROWSER": True,
        "GUI_ALLOW_IMAGES": True,
        "GUI_MAX_IMAGE_WIDTH": 512,
        "GUI_MAX_IMAGE_HEIGHT": 512,
    }


def _sm_save_settings_to_file(settings_dict):
    """
    Save the current settings to the SETTINGS_FILE (settings.json).
    Creates the directory if it doesn't exist.
    Returns (success: bool, message: str)
    """
    import os
    import json
    
    try:
        # Use SETTINGS_FILE from globals if available
        settings_path = globals().get("SETTINGS_FILE", None)
        if not settings_path:
            settings_dir = globals().get("SETTINGS_DIR", os.path.join(os.getcwd(), "data", "settings"))
            settings_path = os.path.join(settings_dir, "settings.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        
        # Filter to only include serializable, modified settings
        saveable = {}
        for key, value in settings_dict.items():
            if isinstance(value, (bool, int, float, str, list, dict)):
                saveable[key] = value
        
        # Write to file with pretty formatting
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, indent=4, sort_keys=True)
        
        return True, f"Settings saved to:\n{settings_path}"
    
    except Exception as e:
        return False, f"Failed to save settings:\n{str(e)}"


def _sm_create_scrollable_frame(parent):
    """
    Create a scrollable frame widget for tabs with many settings.
    Returns (canvas, scrollable_inner_frame)
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Create canvas and scrollbar
    canvas = tk.Canvas(parent, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    
    # Create inner frame for content
    inner_frame = ttk.Frame(canvas)
    
    # Configure scrolling
    inner_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # Create window in canvas
    canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
    
    # Configure canvas to expand inner frame width
    def configure_inner_width(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", configure_inner_width)
    
    # Configure scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def bind_mousewheel(event):
        canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    def unbind_mousewheel(event):
        canvas.unbind_all("<MouseWheel>")
    
    # Bind mouse wheel only when hovering over the canvas
    canvas.bind("<Enter>", bind_mousewheel)
    canvas.bind("<Leave>", unbind_mousewheel)
    
    # Pack widgets
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    return canvas, inner_frame


def launch_settings_gui():
    """
    Launch the SarahMemory Settings GUI.
    
    Features:
    - Categorized tabs for different setting groups
    - Scrollable frames for tabs with many settings
    - Boolean checkboxes, integer spinboxes, string entries
    - JSON editor for list/dict settings
    - Combobox dropdowns for known enum-like settings
    - Save button with confirmation dialog
    - Restore Defaults button
    - Exit button
    - Settings persistence to settings.json
    """
    
    # -------------------------------------------------------------------------
    # Pre-flight checks: Skip GUI if no display available
    # -------------------------------------------------------------------------
    if not _sm_has_display():
        print("[Settings GUI] Headless environment detected - skipping Tkinter window.")
        print("[Settings GUI] To modify settings, edit the settings.json file directly or set environment variables.")
        return
    
    # -------------------------------------------------------------------------
    # Import Tkinter and PIL
    # -------------------------------------------------------------------------
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError as e:
        print(f"[Settings GUI] Tkinter unavailable: {e}")
        return
    
    try:
        from PIL import Image, ImageTk
        _HAS_PIL = True
    except ImportError:
        _HAS_PIL = False
    
    # -------------------------------------------------------------------------
    # Initialize main window
    # -------------------------------------------------------------------------
    try:
        root = tk.Tk()
    except Exception as e:
        print(f"[Settings GUI] Failed to initialize Tk root window: {e}")
        return
    
    root.title("SarahMemory - Global Settings Configuration")
    _sm_center_window(root, 1024, 720)
    
    # Set window icon if available
    try:
        icon_path = os.path.join(globals().get("BASE_DIR", os.getcwd()), "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
    
    # Apply a modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "vista" in available_themes:
            style.theme_use("vista")
    except Exception:
        pass
    
    # -------------------------------------------------------------------------
    # Header Frame with title and version
    # -------------------------------------------------------------------------
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", padx=10, pady=(10, 5))
    
    title_label = ttk.Label(
        header_frame,
        text="SarahMemory Global Settings",
        font=("Segoe UI", 16, "bold")
    )
    title_label.pack(side="left")
    
    version_text = f"Version: {globals().get('PROJECT_VERSION', '7.7.5')}"
    version_label = ttk.Label(
        header_frame,
        text=version_text,
        font=("Segoe UI", 10)
    )
    version_label.pack(side="right")
    
    # -------------------------------------------------------------------------
    # Collect and categorize all configuration keys
    # -------------------------------------------------------------------------
    module_globals = globals()
    categorized_items = {}
    
    for key, value in sorted(module_globals.items()):
        if _sm_is_config_key(key, value):
            category = _sm_group_for_key(key)
            if category not in categorized_items:
                categorized_items[category] = []
            categorized_items[category].append((key, value))
    
    # Define tab order (most commonly used first)
    tab_order = [
        "Core", "APIs", "Models", "Research", "Vision", "Voice",
        "Agent", "Network", "GUI", "Performance", "Updates", "Paths", "General"
    ]
    
    # Sort categories by defined order, then alphabetically for any extras
    sorted_categories = []
    for cat in tab_order:
        if cat in categorized_items:
            sorted_categories.append(cat)
    for cat in sorted(categorized_items.keys()):
        if cat not in sorted_categories:
            sorted_categories.append(cat)
    
    # -------------------------------------------------------------------------
    # Create Notebook (tabbed interface)
    # -------------------------------------------------------------------------
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Dictionary to store all widget references for saving
    all_widgets = {}
    
    # Known enum-like settings with dropdown values
    enum_settings = {
        "ROUTE_MODE": ["Any", "Local", "Web", "API"],
        "RUN_MODE": ["local", "cloud", "test"],
        "DEVICE_PROFILE": ["UltraLite", "Standard", "Performance"],
        "API_COST_TIER": ["low", "balanced", "max"],
        "UPDATER_SCHEDULE": ["never", "always", "daily", "weekly", "monthly", "quarterly", "yearly"],
        "FTP_BACKUP_SCHEDULE": ["never", "daily", "weekly", "monthly", "90days", "180days"],
        "UPDATE_POLICY": ["never", "daily", "weekly", "monthly", "quarterly", "yearly"],
    }
    
    # -------------------------------------------------------------------------
    # Create tabs for each category
    # -------------------------------------------------------------------------
    for category in sorted_categories:
        items = categorized_items[category]
        
        # Create tab frame
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=f" {category} ({len(items)}) ")
        
        # Create scrollable frame for the tab
        canvas, scroll_frame = _sm_create_scrollable_frame(tab_frame)
        
        # Configure grid columns
        scroll_frame.grid_columnconfigure(0, weight=0, minsize=300)  # Key column
        scroll_frame.grid_columnconfigure(1, weight=1)               # Value column
        scroll_frame.grid_columnconfigure(2, weight=0)               # Type indicator
        
        # Add header row
        ttk.Label(
            scroll_frame,
            text="Setting Name",
            font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        
        ttk.Label(
            scroll_frame,
            text="Value",
            font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        
        ttk.Label(
            scroll_frame,
            text="Type",
            font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=2, sticky="w", padx=8, pady=4)
        
        ttk.Separator(scroll_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=2
        )
        
        # Add setting rows
        row_index = 2
        for key, value in items:
            # Key label
            key_label = ttk.Label(scroll_frame, text=key, font=("Consolas", 9))
            key_label.grid(row=row_index, column=0, sticky="w", padx=8, pady=3)
            
            # Value widget based on type
            if isinstance(value, bool):
                # Boolean: Checkbutton
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(scroll_frame, variable=var)
                widget.grid(row=row_index, column=1, sticky="w", padx=8, pady=3)
                all_widgets[key] = ("bool", var)
                type_text = "bool"
                
            elif isinstance(value, int):
                # Integer: Spinbox
                var = tk.IntVar(value=value)
                widget = ttk.Spinbox(
                    scroll_frame,
                    from_=-999999999,
                    to=999999999,
                    textvariable=var,
                    width=15
                )
                widget.grid(row=row_index, column=1, sticky="w", padx=8, pady=3)
                all_widgets[key] = ("int", var)
                type_text = "int"
                
            elif isinstance(value, float):
                # Float: Entry
                var = tk.DoubleVar(value=value)
                widget = ttk.Entry(scroll_frame, textvariable=var, width=20)
                widget.grid(row=row_index, column=1, sticky="w", padx=8, pady=3)
                all_widgets[key] = ("float", var)
                type_text = "float"
                
            elif isinstance(value, str):
                # String: Combobox (if enum) or Entry
                var = tk.StringVar(value=value)
                
                if key in enum_settings:
                    widget = ttk.Combobox(
                        scroll_frame,
                        values=enum_settings[key],
                        textvariable=var,
                        state="readonly",
                        width=25
                    )
                else:
                    widget = ttk.Entry(scroll_frame, textvariable=var, width=50)
                
                widget.grid(row=row_index, column=1, sticky="we", padx=8, pady=3)
                all_widgets[key] = ("str", var)
                type_text = "str"
                
            elif isinstance(value, (list, dict)):
                # List/Dict: Text widget with JSON
                import json as _json
                text_widget = tk.Text(scroll_frame, height=3, width=50, wrap="word", font=("Consolas", 9))
                try:
                    json_str = _json.dumps(value, indent=2)
                except Exception:
                    json_str = str(value)
                text_widget.insert("1.0", json_str)
                text_widget.grid(row=row_index, column=1, sticky="we", padx=8, pady=3)
                all_widgets[key] = ("json", text_widget)
                type_text = "list" if isinstance(value, list) else "dict"
            
            else:
                # Unsupported type - display as read-only
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(scroll_frame, textvariable=var, state="readonly", width=50)
                widget.grid(row=row_index, column=1, sticky="we", padx=8, pady=3)
                all_widgets[key] = ("readonly", var)
                type_text = type(value).__name__
            
            # Type indicator label
            type_label = ttk.Label(scroll_frame, text=type_text, font=("Consolas", 8), foreground="gray")
            type_label.grid(row=row_index, column=2, sticky="w", padx=8, pady=3)
            
            row_index += 1
    
    # -------------------------------------------------------------------------
    # Footer Frame with action buttons
    # -------------------------------------------------------------------------
    footer_frame = ttk.Frame(root)
    footer_frame.pack(fill="x", padx=10, pady=10)
    
    # Status label
    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(footer_frame, textvariable=status_var, font=("Segoe UI", 9))
    status_label.pack(side="left")
    
    # -------------------------------------------------------------------------
    # Button Functions
    # -------------------------------------------------------------------------
    def do_save():
        """Collect all widget values, update globals, and save to file."""
        changed_settings = {}
        errors = []
        
        for key, (typ, widget) in all_widgets.items():
            try:
                if typ == "bool":
                    new_value = bool(widget.get())
                elif typ == "int":
                    new_value = int(widget.get())
                elif typ == "float":
                    new_value = float(widget.get())
                elif typ == "str":
                    new_value = str(widget.get())
                elif typ == "json":
                    import json as _json
                    raw_text = widget.get("1.0", "end").strip()
                    if raw_text:
                        new_value = _json.loads(raw_text)
                    else:
                        new_value = None
                elif typ == "readonly":
                    continue  # Skip read-only fields
                else:
                    continue
                
                # Update globals
                globals()[key] = new_value
                changed_settings[key] = new_value
                
            except Exception as e:
                errors.append(f"{key}: {str(e)}")
        
        # Attempt to save to file
        success, message = _sm_save_settings_to_file(changed_settings)
        
        # Show confirmation dialog
        if errors:
            error_text = "\n".join(errors[:10])  # Limit to first 10 errors
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more errors"
            messagebox.showwarning(
                "Settings Saved with Warnings",
                f"Saved {len(changed_settings)} setting(s).\n\nErrors:\n{error_text}\n\n{message}"
            )
        else:
            if success:
                messagebox.showinfo(
                    "Settings Saved",
                    f"Successfully saved {len(changed_settings)} setting(s).\n\n{message}"
                )
            else:
                messagebox.showerror(
                    "Save Error",
                    f"Settings updated in memory but file save failed.\n\n{message}"
                )
        
        status_var.set(f"Saved {len(changed_settings)} settings")
    
    def do_restore_defaults():
        """Restore default values for known settings."""
        if not messagebox.askyesno(
            "Restore Defaults",
            "This will reset common settings to their default values.\n\n"
            "Settings not in the defaults list will remain unchanged.\n\n"
            "Continue?"
        ):
            return
        
        defaults = _sm_get_default_settings()
        restored_count = 0
        
        for key, default_value in defaults.items():
            if key in all_widgets:
                typ, widget = all_widgets[key]
                try:
                    if typ == "bool":
                        widget.set(bool(default_value))
                    elif typ == "int":
                        widget.set(int(default_value))
                    elif typ == "float":
                        widget.set(float(default_value))
                    elif typ == "str":
                        widget.set(str(default_value))
                    elif typ == "json":
                        import json as _json
                        widget.delete("1.0", "end")
                        widget.insert("1.0", _json.dumps(default_value, indent=2))
                    restored_count += 1
                except Exception:
                    pass
        
        status_var.set(f"Restored {restored_count} defaults (not saved yet)")
        messagebox.showinfo(
            "Defaults Restored",
            f"Restored {restored_count} setting(s) to default values.\n\n"
            "Click 'Save Settings' to apply and persist these changes."
        )
    
    def do_exit():
        """Exit the settings GUI."""
        if messagebox.askyesno(
            "Exit Settings",
            "Exit without saving?\n\nAny unsaved changes will be lost."
        ):
            root.destroy()
    
    def do_save_and_exit():
        """Save settings and exit."""
        do_save()
        root.destroy()
    
    # -------------------------------------------------------------------------
    # Create action buttons (right to left)
    # -------------------------------------------------------------------------
    ttk.Button(
        footer_frame,
        text="Exit",
        command=do_exit,
        width=12
    ).pack(side="right", padx=5)
    
    ttk.Button(
        footer_frame,
        text="Save & Exit",
        command=do_save_and_exit,
        width=12
    ).pack(side="right", padx=5)
    
    ttk.Button(
        footer_frame,
        text="Save Settings",
        command=do_save,
        width=12
    ).pack(side="right", padx=5)
    
    ttk.Separator(footer_frame, orient="vertical").pack(side="right", fill="y", padx=10)
    
    ttk.Button(
        footer_frame,
        text="Restore Defaults",
        command=do_restore_defaults,
        width=14
    ).pack(side="right", padx=5)
    
    # -------------------------------------------------------------------------
    # Bind keyboard shortcuts
    # -------------------------------------------------------------------------
    root.bind("<Control-s>", lambda e: do_save())
    root.bind("<Control-q>", lambda e: do_exit())
    root.bind("<Escape>", lambda e: do_exit())
    
    # -------------------------------------------------------------------------
    # Handle window close button
    # -------------------------------------------------------------------------
    def on_closing():
        do_exit()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # -------------------------------------------------------------------------
    # Start the main event loop
    # -------------------------------------------------------------------------
    try:
        status_var.set(f"Loaded {len(all_widgets)} settings across {len(sorted_categories)} categories")
        root.mainloop()
    except Exception as e:
        print(f"[Settings GUI] Mainloop error: {e}")



# === API Model Registry & Selector (v7.7.1) ================================
# These control which OpenAI models are preferred for different capabilities.
API_MODEL_AUTO_SELECTOR = globals().get("API_MODEL_AUTO_SELECTOR", True)
API_TOKEN_SOFT_LIMIT = int(os.getenv("SARAH_API_TOKEN_SOFT_LIMIT", "1024"))
API_COST_TIER = (os.getenv("SARAH_API_COST_TIER", "balanced") or "balanced").lower()  # low|balanced|max

# Only allow models explicitly listed here (derived from your enabled set)
API_ALLOWED_MODELS = list(dict.fromkeys([
    # Core chat/reasoning
    "gpt-4.1","gpt-4.1-mini","gpt-4.1-mini-2025-04-14","gpt-4.1-2025-04-14",
    "o4-mini","o4-mini-2025-04-16","o4-mini-deep-research","o4-mini-deep-research-2025-06-26",
    "o3","o3-2025-04-16","o3-mini","o3-mini-2025-01-31",
    "o1","o1-2024-12-17","o1-mini","o1-mini-2024-09-12","o1-pro","o1-pro-2025-03-19",
    "gpt-4o","gpt-4o-2024-05-13","gpt-4o-2024-08-06","gpt-4o-realtime-preview","gpt-4o-realtime-preview-2024-10-01","gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini","gpt-4o-mini-2024-07-18","gpt-4o-mini-search-preview","gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview","gpt-4o-search-preview-2025-03-11","chatgpt-4o-latest",
    "gpt-4","gpt-4-turbo","gpt-4-turbo-preview","gpt-4-0125-preview","gpt-4-1106-preview","gpt-4-0613","gpt-3.5-turbo","gpt-3.5-turbo-1106","gpt-3.5-turbo-0125","gpt-3.5-turbo-16k","gpt-3.5-turbo-instruct-0914",
    # Embeddings
    "text-embedding-3-small","text-embedding-3-large","text-embedding-ada-002",
    # Audio/vision/tts/stt
    "gpt-4o-mini-transcribe","gpt-4o-transcribe","whisper-1",
    "gpt-4o-audio-preview","gpt-4o-audio-preview-2024-10-01","gpt-4o-audio-preview-2024-12-17","gpt-4o-mini-audio-preview","gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-tts","tts-1","tts-1-1106","tts-1-hd","tts-1-hd-1106",
    # Images
    "gpt-image-1","dall-e-3","dall-e-2",
    # Safety/moderation
    "omni-moderation-latest","omni-moderation-2024-09-26",
    # Utility / legacy
    "babbage-002","davinci-002","codex-mini-latest",
    # Forward-compat flags in your list
    "gpt-5","gpt-5-mini","gpt-5-mini-2025-08-07","gpt-5-nano","gpt-5-nano-2025-08-07","gpt-5-chat-latest","gpt-realtime","gpt-realtime-2025-08-28","gpt-audio","gpt-audio-2025-08-28","gpt-4.1-nano"
]))

def _model_capabilities(model_id: str) -> dict:
    mid = (model_id or "").lower()
    return {
        "vision": ("4o" in mid) or ("realtime" in mid),
        "search": "search-preview" in mid or "deep-research" in mid,
        "stt": ("transcribe" in mid) or (mid == "whisper-1"),
        "tts": ("tts" in mid),
        "embedding": ("embedding" in mid),
        "image": ("gpt-image-1" in mid) or ("dall-e" in mid) or ("image" in mid and "gpt" in mid),
        "fast": ("mini" in mid) or ("nano" in mid),
        "premium": any(x in mid for x in ["o4","gpt-4.1","o3"]) and not ("mini" in mid or "nano" in mid),
    }

def _model_tier(model_id: str) -> str:
    caps = _model_capabilities(model_id)
    if caps["premium"]: return "max"
    if caps["fast"]:    return "low"
    return "balanced"

# === Auto Model Candidate Selector (OpenAI chat-safe) =========================
# These are safe with the /v1/chat/completions endpoint used by SarahMemoryAPI.
_OPENAI_CHAT_SAFE = [
    "gpt-4o-2024-08-06","gpt-4o-2024-05-13","gpt-4o",
    "gpt-4-turbo","gpt-4-0125-preview","gpt-4-1106-preview",
    "gpt-3.5-turbo-0125","gpt-3.5-turbo"
]

def _is_chat_safe(model_id: str) -> bool:
    mid = (model_id or "").lower()
    if not mid: return False
    if any(x in mid for x in ["realtime","audio","transcribe","search-preview","deep-research"]):
        return False
    if mid.startswith(("o1","o3","o4","gpt-4.1","gpt-5","whisper","tts")):
        return False
    # must be allowed in user list if present
    if "API_ALLOWED_MODELS" in globals():
        if model_id not in API_ALLOWED_MODELS and mid not in [m.lower() for m in API_ALLOWED_MODELS]:
            return False
    # chat-completions compatible short-list
    return mid in [m.lower() for m in _OPENAI_CHAT_SAFE]

def get_openai_model_candidates(query: str = "", intent: str = "chat", max_n: int = 6) -> list[str]:
    """Return a prioritized list of OpenAI chat-safe model IDs for the given query/intent.
    Only models compatible with /v1/chat/completions are returned."""
    q = (query or "").lower()
    it = (intent or "chat").lower()

    # Priority buckets
    primary = []
    secondary = []

    # Use configured primaries if provided
    for key in ("API_PRIMARY_MODEL","API_SECONDARY_MODEL","API_DEFAULT_MODEL"):
        mid = globals().get(key, None)
        if isinstance(mid, str) and _is_chat_safe(mid):
            primary.append(mid)

    # Heuristics by intent/query
    if any(k in q for k in ["image","picture","draw","dall-e"]):
        # still route via chat; API handles actual media elsewhere
        pass  # do not insert media-only models here

    # Fill from allowed list in a deterministic order
    try:
        for mid in API_ALLOWED_MODELS:
            if _is_chat_safe(mid):
                if "gpt-4o" in mid or "gpt-4-turbo" in mid:
                    if mid not in primary: primary.append(mid)
                else:
                    if mid not in secondary: secondary.append(mid)
    except Exception:
        for mid in _OPENAI_CHAT_SAFE:
            if _is_chat_safe(mid):
                if "gpt-4o" in mid or "gpt-4-turbo" in mid:
                    if mid not in primary: primary.append(mid)
                else:
                    if mid not in secondary: secondary.append(mid)

    # Unique and clipped
    seen = set()
    out = []
    for mid in primary + secondary + _OPENAI_CHAT_SAFE:
        if _is_chat_safe(mid) and mid not in seen:
            out.append(mid); seen.add(mid)
        if len(out) >= max_n: break
    return out

def get_alternate_model(prev_model: str) -> str | None:
    """Return a different chat-safe model than `prev_model`, for cross-validation."""
    cands = get_openai_model_candidates()
    for mid in cands:
        if prev_model and mid.lower() != (prev_model or "").lower():
            return mid
    return None

def select_api_model(intent: str = "chat",
                     need_vision: bool = False,
                     need_stt: bool = False,
                     need_tts: bool = False,
                     prefers_search: bool = False,
                     cost_tier: str | None = None,
                     token_soft_limit: int | None = None) -> str:
    """
    Choose an OpenAI model based on intent/capabilities/cost.
    Honors API_PRIMARY_MODEL â†’ API_SECONDARY_MODEL â†’ API_DEFAULT_MODEL.
    """
    ctier = (cost_tier or API_COST_TIER or "balanced").lower()
    wants = {
        "vision": need_vision,
        "stt": need_stt,
        "tts": need_tts,
        "search": prefers_search or intent in ("search","lookup","fact"),
    }
    priority = [m for m in [globals().get("API_PRIMARY_MODEL"),
                            globals().get("API_SECONDARY_MODEL"),
                            globals().get("API_DEFAULT_MODEL")] if m]
    for mid in API_ALLOWED_MODELS:
        if mid not in priority:
            priority.append(mid)

    candidates = []
    for mid in priority:
        if not mid or mid not in API_ALLOWED_MODELS:
            continue
        caps = _model_capabilities(mid)
        if wants["vision"] and not caps["vision"]: continue
        if wants["stt"] and not caps["stt"]:       continue
        if wants["tts"] and not caps["tts"]:       continue
        tier = _model_tier(mid)
        if ctier == "low" and tier == "max":       continue
        candidates.append((mid, tier))

    if API_MODEL_AUTO_SELECTOR and candidates:
        if ctier == "max":
            for mid, tier in candidates:
                if tier == "max": return mid
        elif ctier == "low":
            for mid, tier in candidates:
                if tier == "low": return mid
        return candidates[0][0]

    for mid in priority:
        if mid in API_ALLOWED_MODELS:
            return mid
    return "gpt-4.1-mini"

def get_embedding_model(max_quality: bool = False) -> str:
    return "text-embedding-3-large" if max_quality else "text-embedding-3-small"
# ===========================================================================

# Fix any accidental older default name
API_DEFAULT_MODEL = API_DEFAULT_MODEL.replace("gpt-4.0-mini", "gpt-4.1-mini")
# GUI launch moved to end of file

# === AI-Agent Master Safety & Voice Control (v7.7.2) ===
# Single master safety gate: when False, Sarah behaves like a normal chatbot (no desktop control).
AI_AGENT_ENABLED = os.getenv("SARAH_AI_AGENT_ENABLED", "True").strip().lower() in ("1","true","yes","on")
# Emergency / control phrases (lowercased exact match)
AI_AGENT_STOP_PHRASES   = ["sarah stop now", "emergency stop", "abort mission"]
AI_AGENT_HALT_PHRASES   = ["halt", "pause", "hold"]
AI_AGENT_RESUME_PHRASES = ["resume", "continue", "go on"]
AI_AGENT_CONFIRM_YES    = ["yes", "confirm", "ok", "okay", "yep"]
AI_AGENT_CONFIRM_NO     = ["no", "cancel", "stop", "nope"]

# Idle delay before auto-resume after human input stops (milliseconds)
AI_AGENT_RESUME_DELAY = int(os.getenv("SARAH_AI_AGENT_RESUME_DELAY_MS", "9000"))

# Human activity grace window: any keyboard/mouse/controller input within this window halts the agent (milliseconds)
AI_AGENT_USER_ACTIVITY_TIMEOUT_MS = int(os.getenv("SARAH_AI_AGENT_USER_TIMEOUT_MS", "2500"))

# Allowed UI operations (the agent will never copy/move/delete files without explicit consent)
AI_AGENT_ALLOWLIST = {"open","launch","focus","maximize","minimize","click","doubleclick","type","press","scroll","play","search","move","wait","close","terminate"}
#AI_AGENT_REQUIRE_CONSENT = {"install","uninstall","system_setting","purchase"}
#AI_AGENT_REQUIRE_CONSENT = [p.lower() for p in getattr(config, "AI_AGENT_REQUIRE_CONSENT", [])]
#AI_AGENT_REQUIRE_CONSENT = [s.strip().lower() for s in os.getenv("SARAH_AI_AGENT_REQUIRE_CONSENT","").split(",") if s.strip()]
AI_AGENT_REQUIRE_CONSENT = []
# Game/learning toggles
AI_GAME_MODE_ENABLED = True
AI_GAME_FULL_AUTO = True
  # When True + Agent enabled, the agent may run exploration macros (still respects HALT/resume).

# Helper: current monotonic ms
def now_ms():
    try:
        return int(time.monotonic() * 1000)
    except Exception:
        return int(time.time() * 1000)

# --- Auto-added by network hub patch ---
SARAH_WEB_BASE = "https://www.sarahmemory.com"
REMOTE_SYNC_ENABLED = True
REMOTE_HTTP_TIMEOUT = 6.0
REMOTE_HEARTBEAT_SEC = 30
REMOTE_API_KEY = None
SARAH_WEB_API_PREFIX = "/api"
SARAH_WEB_PING_PATH = "/health"
SARAH_WEB_HEALTH_PATH = "/health"
SARAH_WEB_RELAY_PATH = "/relay"
SARAH_WEB_REGISTER_PATH = "/register-node"
SARAH_WEB_HEARTBEAT_PATH = "/heartbeat"
SARAH_WEB_EMBED_PATH = "/receive-embedding"
SARAH_WEB_CONTEXT_PATH = "/context-update"
SARAH_WEB_JOBS_PATH = "/jobs"


# --- injected: on-demand ensure table for `response` ---
def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
try:
    _ensure_response_table()
except Exception:
    pass
# === v7.7.3 Emotional Realism Feature Gates (surgical, reversible) ===
EMOTION_REALISM_ENABLED      = True
FACIAL_FEEDBACK_ENABLED      = True
PERSONALITY_DRIFT_ENABLED    = True
EXPRESSIVE_OUTPUT_ENABLED    = True
FOLLOWUP_QUESTIONS_ENABLED   = True
ETHICS_FILTER_ENABLED        = True
BANDWIDTH_AWARE_INTELLIGENCE = True

EMOJI_POLICY = {
    "joy": ["ðŸ˜„", "ðŸ˜Š", "âœ¨"],
    "neutral": ["ðŸ™‚"],
    "sad": ["ðŸ˜”", "ðŸ’™"],
    "anger": ["ðŸ˜¤"],
    "concern": ["ðŸ¤","ðŸ«¶"],
    "curiosity": ["ðŸ¤”"]
}

def reduced_mode_suggested(cpu_pct: float = None, mem_pct: float = None, net_bps: float = None) -> bool:
    try:
        import psutil
        cpu_pct = cpu_pct if cpu_pct is not None else psutil.cpu_percent(interval=0.0)
        mem_pct = mem_pct if mem_pct is not None else psutil.virtual_memory().percent
    except Exception:
        return SAFE_MODE if 'SAFE_MODE' in globals() else False
    if not BANDWIDTH_AWARE_INTELLIGENCE:
        return SAFE_MODE if 'SAFE_MODE' in globals() else False
    return (SAFE_MODE if 'SAFE_MODE' in globals() else False) or cpu_pct >= 82 or mem_pct >= 85

def emotion_to_color(primary_label: str) -> str:
    m = (primary_label or "neutral").lower()
    if m in ("anger","fear"): return "#FF0000"
    if m in ("sad","concern"): return "#FFFF00"
    return "#00FF00"

# ====== Emotional Fine-Tuning Knobs (append-only; v7.7.3) ======
try:
    EMO_REWRITE_STRENGTH      # 0..1 intensity of phrasing rewrite
except NameError:
    EMO_REWRITE_STRENGTH = 0.55
try:
    FOLLOWUP_MAX_QUESTIONS
except NameError:
    FOLLOWUP_MAX_QUESTIONS = 2
try:
    EXPRESSIVE_MAX_EMOJI
except NameError:
    EXPRESSIVE_MAX_EMOJI = 1
try:
    DRIFT_LEARNING_RATE
except NameError:
    DRIFT_LEARNING_RATE = 0.02
def get_rewrite_strength():
    try:
        return float(EMO_REWRITE_STRENGTH)
    except Exception:
        return 0.5


# === Performance & Tuning =========================
PERF_FAST_FIRST = True
RESPONSE_CACHE_TTL = 900
COMPARE_MIN_CHARS = 220
COMPARE_INTENTS = ["question","explanation","research","identity","story"]
FAST_MODEL_PREFERENCE = ["gpt-4o-mini","gpt-3.5-turbo-0125","gpt-3.5-turbo","gpt-4o"]
ENABLE_DB_WAL = True
SQLITE_PRAGMAS = {"journal_mode":"WAL","synchronous":1,"temp_store":2,"mmap_size":268435456}
# ================================================

def reorder_fast_first(candidates: list[str]) -> list[str]:
    try:
        pref = [m.lower() for m in FAST_MODEL_PREFERENCE]
        cands = list(candidates or [])
        cands_l = [m.lower() for m in cands]
        out = []
        used = set()
        for m in pref:
            if m in cands_l:
                idx = cands_l.index(m)
                out.append(cands[idx]); used.add(idx)
        for i, m in enumerate(cands):
            if i not in used:
                out.append(m)
        return out
    except Exception:
        return candidates or []

def apply_sqlite_pragmas(conn):
    try:
        if not conn: return
        cur = conn.cursor()
        prag = globals().get("SQLITE_PRAGMAS", {})
        for k,v in prag.items():
            try:
                cur.execute(f"PRAGMA {k}={v}")
            except Exception:
                pass
        try:
            conn.commit()
        except Exception:
            pass
    except Exception:
        pass

# ============================================================================
# Back-compat shim (v7.7.4):
# Some legacy paths referenced `SarahMemoryGlobals.SarahMemoryGlobals.<X>`.
# This module never exposed such a nested object; to remain backward compatible
# we provide a lightweight proxy that forwards attribute lookups to the module
# globals. This prevents AttributeError at runtime without changing call sites.
# ============================================================================
class _GlobalsProxy:
    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError as e:
            raise AttributeError(f"SarahMemoryGlobals has no attribute {{name}}") from e

# Export the alias expected by older code paths
SarahMemoryGlobals = _GlobalsProxy()


# === v7.7.5 GUI WebUI additions (non-breaking) ===
try:
    BASE_DIR
except NameError:
    import os as _os2
    BASE_DIR = _os2.getcwd()

try:
    THEMES_DIR
except NameError:
    import os as _os3
    THEMES_DIR = _os3.path.join(BASE_DIR, "data", "mods", "themes")

# Prefer modern webview for the new chat UI
try:
    USE_WEBVIEW
except NameError:
    USE_WEBVIEW = True

try:
    WEBUI_HTML_PATH
except NameError:
    import os as _os4
    WEBUI_HTML_PATH = _os4.path.join(BASE_DIR, "data", "ui", "SarahMemory.html")

# Bridge origin allowlist for JSâ†’Python
try:
    BRIDGE_ALLOWED_ORIGINS
except NameError:
    BRIDGE_ALLOWED_ORIGINS = {"file://", "https://api.sarahmemory.com", "https://www.sarahmemory.com"}

def origin_allowed(origin: str) -> bool:
    try:
        return any(origin.startswith(o) for o in BRIDGE_ALLOWED_ORIGINS)
    except Exception:
        return True

# System resource push interval (ms) for the top bar indicators
try:
    SYSRES_UPDATE_MS
except NameError:
    SYSRES_UPDATE_MS = 1000

# === SM_PORTABLE_PATHS_V1 ===
try:
    _SM_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if 'BASE_DIR' not in globals() or not globals().get('BASE_DIR'):
        BASE_DIR = _SM_THIS_DIR
except Exception:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if 'PUBLIC_DIR' not in globals():
    PUBLIC_DIR = BASE_DIR
if 'WEB_DIR' not in globals():
    WEB_DIR = BASE_DIR
if 'DATA_DIR' not in globals():
    DATA_DIR = os.path.join(BASE_DIR, 'data')

_THEMES_A = os.path.join(DATA_DIR, 'mods', 'themes')
_THEMES_B = os.path.join(DATA_DIR, 'themes')
try:
    THEMES_DIR = _THEMES_A if os.path.isdir(_THEMES_A) else _THEMES_B
except Exception:
    THEMES_DIR = _THEMES_A

if 'DATASETS_DIR' not in globals():
    DATASETS_DIR = os.path.join(DATA_DIR, 'memory', 'datasets')
if 'STATIC_DIR' not in globals():
    STATIC_DIR = os.path.join(BASE_DIR, 'server', 'static')

# ============================================================================
# Phase B: Context Engine, Agent Permissions, and Mesh Sync (v7.7.5)
# NOTE: This block is strictly additive and does not remove or rename any
# existing globals. It exposes higher-level knobs that downstream modules
# (AiFunctions, AdvCU, Network, GUI) can query when routing intents.
# ============================================================================

# ----- B1: Context Engine Configuration -------------------------------------
try:
    CONTEXT_ENGINE_ENABLED
except NameError:
    # Mirror the existing ENABLE_CONTEXT_BUFFER flag so older code keeps working
    CONTEXT_ENGINE_ENABLED = globals().get("ENABLE_CONTEXT_BUFFER", True)

# Where long-term contextual turns will be stored by default
try:
    CONTEXT_DB_PATH
except NameError:
    CONTEXT_DB_PATH = os.path.join(DATASETS_DIR, "context_history.db")

# Maximum number of turns and maximum age (in seconds) to consider when
# building a contextual prompt. Downstream code (SarahMemoryAiFunctions.py)
# decides *how* to use these.
try:
    CONTEXT_MAX_TURNS
except NameError:
    try:
        _default_ctx = int(globals().get("CONTEXT_BUFFER_SIZE", 10))
    except Exception:
        _default_ctx = 10
    CONTEXT_MAX_TURNS = int(os.getenv("SARAH_CONTEXT_MAX_TURNS", str(_default_ctx)) or _default_ctx)

try:
    CONTEXT_MAX_AGE_SEC
except NameError:
    # 3 days by default; can be overridden from .env
    CONTEXT_MAX_AGE_SEC = int(os.getenv("SARAH_CONTEXT_MAX_AGE_SEC", "259200") or 259200)

try:
    CONTEXT_PERSIST_TO_DB
except NameError:
    # When False, the engine may hold context only in memory (per-process).
    CONTEXT_PERSIST_TO_DB = True


def get_context_config() -> dict:
    """
    Small helper for downstream modules to introspect context-engine settings
    without importing a larger pile of globals.
    """
    return {
        "enabled":           bool(globals().get("CONTEXT_ENGINE_ENABLED", True)),
        "buffer_size":       int(globals().get("CONTEXT_BUFFER_SIZE", 10)),
        "max_turns":         int(globals().get("CONTEXT_MAX_TURNS", 10)),
        "max_age_sec":       int(globals().get("CONTEXT_MAX_AGE_SEC", 259200)),
        "persist_to_db":     bool(globals().get("CONTEXT_PERSIST_TO_DB", True)),
        "db_path":           str(globals().get("CONTEXT_DB_PATH", os.path.join(DATASETS_DIR, "context_history.db"))),
        "enrichment_enabled":bool(globals().get("ENABLE_CONTEXT_ENRICHMENT", True)),
    }


# ----- B3: Agent Permission & Safety Profile --------------------------------

def is_cloud_run() -> bool:
    """True when SarahMemory is running on a cloud host (e.g., PythonAnywhere)."""
    try:
        return (globals().get("RUN_MODE", "local") == "cloud")
    except Exception:
        return False


def is_public_web_mode() -> bool:
    """True when this instance is primarily serving a browser-only UI."""
    try:
        return globals().get("DEVICE_MODE") == globals().get("DEVICE_MODE_PUBLIC_WEB")
    except Exception:
        return False


def _default_agent_gate(local_default: str = "true", cloud_default: str = "false") -> str:
    """
    Helper for setting conservative defaults:
      - local desktop agent  -> permissive by default
      - cloud/public web     -> locked down by default
    Returns a string used as the default in _env_flag so it can be overridden
    from the environment.
    """
    try:
        if is_cloud_run() or is_public_web_mode():
            return cloud_default
    except Exception:
        pass
    return local_default


# High level ability flags. These DO NOT perform any actions; they simply
# describe what *categories* of actions the agent layer is allowed to attempt.
# SarahMemoryAiFunctions.py will read these before calling any OS helpers.

AI_AGENT_ALLOW_APP_LAUNCH = _env_flag(
    "SARAH_AGENT_ALLOW_APP_LAUNCH",
    _default_agent_gate(local_default="true", cloud_default="false"),
)

AI_AGENT_ALLOW_FILE_WRITE = _env_flag(
    "SARAH_AGENT_ALLOW_FILE_WRITE",
    _default_agent_gate(local_default="false", cloud_default="false"),
)

AI_AGENT_ALLOW_REMOTE_CONTROL = _env_flag(
    "SARAH_AGENT_ALLOW_REMOTE_CONTROL",
    _default_agent_gate(local_default="false", cloud_default="false"),
)

AI_AGENT_ALLOW_NETWORK_TASKS = _env_flag(
    "SARAH_AGENT_ALLOW_NETWORK_TASKS",
    _default_agent_gate(local_default="true", cloud_default="true"),
)


def agent_permissions_summary() -> dict:
    """
    Compact view of agent permissions and environment, useful for both
    Diagnostics (SarahMemoryDiagnostics.py) and the GUI.
    """
    return {
        "run_mode":               globals().get("RUN_MODE", "local"),
        "device_mode":            globals().get("DEVICE_MODE", "headless"),
        "safe_mode":              bool(globals().get("SAFE_MODE", False)),
        "local_only":             bool(globals().get("LOCAL_ONLY_MODE", False)),
        "agent_enabled":          bool(globals().get("AI_AGENT_ENABLED", False)),
        "allow_app_launch":       bool(globals().get("AI_AGENT_ALLOW_APP_LAUNCH", False)),
        "allow_file_write":       bool(globals().get("AI_AGENT_ALLOW_FILE_WRITE", False)),
        "allow_remote_control":   bool(globals().get("AI_AGENT_ALLOW_REMOTE_CONTROL", False)),
        "allow_network_tasks":    bool(globals().get("AI_AGENT_ALLOW_NETWORK_TASKS", False)),
        "user_activity_timeout":  int(globals().get("AI_AGENT_USER_ACTIVITY_TIMEOUT_MS", 2500)),
        "resume_delay_ms":        int(globals().get("AI_AGENT_RESUME_DELAY", 9000)),
    }


# ----- B4: Mesh / Hub Sync Toggle Layer -------------------------------------

try:
    MESH_SYNC_ENABLED
except NameError:
    # Mirror existing knobs but keep everything override-able via env.
    base_default = "true" if globals().get("SARAHNET_ENABLED", True) and globals().get("REMOTE_SYNC_ENABLED", True) else "false"
    MESH_SYNC_ENABLED = _env_flag("SARAH_MESH_SYNC_ENABLED", base_default)

try:
    ALLOW_HUB_SYNC
except NameError:
    # When False, nodes may still use SarahNet peer-to-peer but will not talk
    # to the central https://www.sarahmemory.com hub.
    hub_default = "false" if os.getenv("SARAH_FORCE_OFFLINE", "").lower() in ("1","true","yes") else "true"
    ALLOW_HUB_SYNC = _env_flag("SARAH_ALLOW_HUB_SYNC", hub_default)

try:
    MESH_SYNC_SAFE_MODE_ONLY
except NameError:
    # When True, mesh sync traffic is allowed only while SAFE_MODE is enabled,
    # giving an additional "double opt-in" feel for sensitive deployments.
    MESH_SYNC_SAFE_MODE_ONLY = _env_flag("SARAH_MESH_SYNC_SAFE_ONLY", "false")


def get_mesh_sync_config() -> dict:
    """
    Return a merged view of mesh/hub sync policy for use by:
      - SarahMemoryNetwork.py
      - SarahMemoryAiFunctions.py (hub helpers)
      - app.py (hub endpoints)
    """
    safe_mode = bool(globals().get("SAFE_MODE", False))
    mesh_enabled = bool(globals().get("MESH_SYNC_ENABLED", True))
    hub_allowed = bool(globals().get("ALLOW_HUB_SYNC", True))
    node_name = globals().get("NODE_NAME", globals().get("SARAHNET_NODE_ID", "SarahMemoryNode"))
    return {
        "node_name":              node_name,
        "mesh_enabled":           mesh_enabled,
        "hub_allowed":           hub_allowed,
        "safe_mode":              safe_mode,
        "safe_mode_only":         bool(globals().get("MESH_SYNC_SAFE_MODE_ONLY", False)),
        "sarahnet_enabled":       bool(globals().get("SARAHNET_ENABLED", True)),
        "web_base":               globals().get("SARAH_WEB_BASE", "https://www.sarahmemory.com"),
        "remote_sync_enabled":    bool(globals().get("REMOTE_SYNC_ENABLED", True)),
        "heartbeat_sec":          float(globals().get("REMOTE_HEARTBEAT_SEC", 30)),
        "http_timeout":           float(globals().get("REMOTE_HTTP_TIMEOUT", 6.0)),
    }

# ============================================================================
# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# When this file is run directly (python SarahMemoryGlobals.py), it will:
# 1. Ensure all required directories exist
# 2. Import any datasets
# 3. Launch the Settings GUI for configuration
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("SarahMemory Global Settings Configuration")
    print(f"Version: {PROJECT_VERSION}")
    print(f"Author: {AUTHOR}")
    print("=" * 70)
    
    # Ensure directories exist
    try:
        ensure_directories()
        print("[OK] Directories verified/created")
    except Exception as e:
        print(f"[WARN] Directory setup: {e}")
    
    # Import datasets (optional, may be empty on first run)
    try:
        datasets = import_datasets()
        print(f"[OK] Loaded {len(datasets)} dataset records")
    except Exception as e:
        print(f"[INFO] Dataset import skipped: {e}")
    
    # Launch Settings GUI
    print("")
    print("Launching Settings GUI...")
    print("(Close the window or press Ctrl+C to exit)")
    print("")
    
    try:
        launch_settings_gui()
    except KeyboardInterrupt:
        print("\n[EXIT] User cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] GUI launch failed: {e}")
        print("[INFO] You can edit settings.json directly or set environment variables")
        sys.exit(1)
