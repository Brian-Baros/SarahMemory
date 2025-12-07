"""--==The SarahMemory Project==--
File: SarahMemory-local_api_server.py
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


from flask import Flask, request, jsonify
from flask_cors import CORS

# === SarahMemory Core Imports ===
import SarahMemoryGlobals as config
from SarahMemoryInitialization import run_initial_checks
from SarahMemoryDiagnostics import run_personality_core_diagnostics
from SarahMemoryReply import generate_reply
from SarahMemoryAPI import send_to_api_async
from SarahMemoryDL import deep_learn_user_context
from SarahMemoryResearch import get_research_data
from SarahMemoryDatabase import search_answers, log_ai_functions_event
from SarahMemoryReminder import store_contact, store_password, store_webpage
from SarahMemoryVoice import synthesize_voice, set_voice_profile
from SarahMemoryGUI import voice, avatar
from SarahMemoryAdaptive import update_personality
from SarahMemoryPersonality import get_identity_response, get_generic_fallback_response
from SarahMemoryAdvCU import classify_intent, parse_command
from SarahMemoryAiFunctions import (
    generate_embedding, retrieve_similar_context, add_to_context, get_context
)
from SarahMemoryCompare import compare_reply
from SarahMemoryWebSYM import WebSemanticSynthesizer
from SarahMemoryIntegration import integration_menu
from UnifiedAvatarController import UnifiedAvatarController

# === Optional Deep Utility/Ext Module Imports ===
from SarahMemoryReply import generate_reply
from SarahMemoryResearch import LocalResearch

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    # Instantiate a dummy GUI class if needed, otherwise skip
    # Replace with actual GUI hook if needed in integration
    class DummyGUI:
        def __init__(self):
            self.status_bar = type('obj', (object,), {'set_intent_light': lambda x: None})()
            self.trigger_sound = lambda x: None
            self.set_font_style = lambda x: None
        def display_response(self, response, source):
            pass

    dummy_self = type('obj', (object,), {'gui': DummyGUI(), 'display_response': lambda self, res, src: None})()
    
    # Generate response using full AI logic pipeline
    generate_reply(dummy_self, message)

    # If your GUI or memory engine modifies shared context, retrieve final response:
    response = "Processed."  # Or hook the actual `synthesized_response` if modified internally

    return jsonify({'response': response})

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'connected', 'online': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
