"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveServices.py
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
import requests
import json
import os
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config

# Setup logging for Cognitive Services integration module
logger = logging.getLogger('SarahMemoryCognitiveServices')
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# MOD: Ensure configuration flag is set for online cognitive services (default false for safety)
if not hasattr(config, 'COGNITIVE_ONLINE_ENABLED'):
    config.COGNITIVE_ONLINE_ENABLED = False  # MOD: Default to offline

# MOD: Define local cognitive fallback data path
LOCAL_COGNITIVE_DATA_PATH = os.path.join(config.DATA_DIR, "local_cognitive.json")

# Cognitive Services configuration from environment variables
TEXT_ANALYSIS_ENDPOINT = os.environ.get('COG_TEXT_ANALYSIS_ENDPOINT', 'https://api.cognitive.microsoft.com/text/analytics/v3.0/sentiment')
TEXT_ANALYSIS_KEY = os.environ.get('COG_TEXT_ANALYSIS_KEY', 'YOUR_TEXT_ANALYSIS_KEY')
IMAGE_ANALYSIS_ENDPOINT = os.environ.get('COG_IMAGE_ANALYSIS_ENDPOINT', 'https://api.cognitive.microsoft.com/vision/v3.2/analyze')
IMAGE_ANALYSIS_KEY = os.environ.get('COG_IMAGE_ANALYSIS_KEY', 'YOUR_IMAGE_ANALYSIS_KEY')

def log_cognitive_event(event, details):
    """
    Logs a cognitive services integration event to the system_logs.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_integration_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO cognitive_integration_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged cognitive event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging cognitive event to system_logs.db: {e}")

def analyze_text(text):
    """
    Analyze text sentiment using Microsoft Cognitive Services.
    ENHANCED (v6.4): Adaptive routing based on configuration and improved error handling.
    """
    try:
        headers = {
            "Ocp-Apim-Subscription-Key": TEXT_ANALYSIS_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        }
        response = requests.post(TEXT_ANALYSIS_ENDPOINT, headers=headers, json=payload, timeout=10)  # MOD: Added timeout
        response.raise_for_status()
        result = response.json()
        logger.info(f"Text analysis successful: {json.dumps(result)}")
        log_cognitive_event("Text Analysis Success", f"Text: '{text}' | Result: {json.dumps(result)}")
        return result
    except Exception as e:
        error_msg = f"Error analyzing text: {e}"
        logger.error(error_msg)
        log_cognitive_event("Text Analysis Error", error_msg)
        # MOD: Fallback to local data if available
        local_data = load_local_cognitive_data()
        fallback = local_data.get("default_text_analysis", {"error": "Local fallback: analysis unavailable"})
        return fallback

def analyze_image(image_path):
    """
    Analyze an image using Microsoft Cognitive Services Computer Vision API.
    ENHANCED (v6.4): Adaptive routing with improved error handling.
    """
    try:
        if not os.path.exists(image_path):
            error_msg = "Image file does not exist."
            logger.error(error_msg)
            log_cognitive_event("Image Analysis Error", error_msg)
            return {"error": "Image file not found."}
        headers = {
            "Ocp-Apim-Subscription-Key": IMAGE_ANALYSIS_KEY,
            "Content-Type": "application/octet-stream"
        }
        params = {
            "visualFeatures": "Categories,Description,Color"
        }
        with open(image_path, 'rb') as image_file:
            data = image_file.read()
        response = requests.post(IMAGE_ANALYSIS_ENDPOINT, headers=headers, params=params, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Image analysis successful: {json.dumps(result)}")
        log_cognitive_event("Image Analysis Success", f"Image: '{image_path}' | Result: {json.dumps(result)}")
        return result
    except Exception as e:
        error_msg = f"Error analyzing image: {e}"
        logger.error(error_msg)
        log_cognitive_event("Image Analysis Error", error_msg)
        return {"error": str(e)}

def load_local_cognitive_data():
    """
    Loads local cognitive data for fallback processing.
    ENHANCED (v6.4): Reads from a JSON file; returns empty dict on failure.
    """
    try:
        with open(LOCAL_COGNITIVE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Local cognitive data loaded successfully.")
        return data
    except Exception as e:
        logger.error("Failed to load local cognitive data: %s", e)
        return {}

def process_local_cognitive_request(request_text):
    """
    Processes the cognitive request using local data.
    ENHANCED (v6.4): Now performs simple keyword matching using lowercased keys.
    """
    data = load_local_cognitive_data()
    for key, response in data.items():
        if key.lower() in request_text.lower():
            logger.info("Local cognitive match found for request: %s", request_text)
            return response
    logger.info("No local cognitive match found for request: %s", request_text)
    return None

def process_online_cognitive_request(request_text):
    """
    Processes the cognitive request using an online API.
    ENHANCED (v6.4): Online call only attempted if online mode is enabled.
    """
    logger.info("Processing online cognitive request for: %s", request_text)
    try:
        api_url = "https://api.example.com/cognitive"  # MOD: Placeholder API URL
        payload = {"text": request_text}
        response = requests.post(api_url, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("Online cognitive response received.")
            return data.get("result", "No result found online.")
        else:
            logger.error("Online cognitive service failed with status code: %s", response.status_code)
            return None
    except Exception as e:
        logger.error("Exception during online cognitive service call: %s", e)
        return None

def process_cognitive_request(request_text):
    """
    Adaptive processing of cognitive requests.
    ENHANCED (v6.4): Attempts local processing first, then online if enabled.
    """
    local_result = process_local_cognitive_request(request_text)
    if local_result:
        return local_result

    if config.COGNITIVE_ONLINE_ENABLED:
        online_result = process_online_cognitive_request(request_text)
        if online_result:
            return online_result

    return "I'm sorry, I couldn't process that request at this time."

if __name__ == '__main__':
    logger.info("Starting SarahMemoryCognitiveServices module test.")
    sample_text = "I am thrilled about the new features of this platform!"
    text_result = analyze_text(sample_text)
    logger.info(f"Text Analysis Result: {text_result}")

    sample_image_path = "sample_image.jpg"  # Replace with a valid image path if available
    image_result = analyze_image(sample_image_path)
    logger.info(f"Image Analysis Result: {image_result}")

    logger.info("SarahMemoryCognitiveServices module testing complete.")

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
# ====================================================================
# END OF SarahMemoryCognitiveServices.py v8.0.0
# ====================================================================