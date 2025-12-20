#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGI WSGI adapter for Flask to run on shared hosting without Passenger/mod_wsgi.
Location:/domain/sarahmemory.com/public_html/api/server/cgi/run.py

This bridges Apache CGI -> Flask WSGI app:
  UI requests go to /api/* which rewrite here with PATH_INFO preserved.
"""
import os, sys, traceback

# Compute paths
SERVER_DIR = os.path.dirname(os.path.dirname(__file__))           # /public_html/api/server
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, ".."))    # /public_html/api

# Let the app resolve its relative imports by seeing server dir first
sys.path.insert(0, SERVER_DIR)

# Optional environment tweaks similar to local
os.environ.setdefault("SARAH_BASE_DIR", PROJECT_ROOT)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Try to activate venv if available (best effort)
VENV_ACTIVATE = "/public_html/api/venv/bin/activate_this.py"
if os.path.isfile(VENV_ACTIVATE):
    try:
        with open(VENV_ACTIVATE) as f:
            code = compile(f.read(), VENV_ACTIVATE, 'exec')
            exec(code, dict(__file__=VENV_ACTIVATE))
    except Exception:
        # Non-fatal; continue without venv
        pass

# Import the Flask app from server/app.py
try:
    from app import app
except Exception:
    print("Status: 500 Internal Server Error")
    print("Content-Type: text/plain; charset=utf-8")
    print()
    traceback.print_exc()
    sys.exit(0)

# Run the Flask WSGI app under CGI
from wsgiref.handlers import CGIHandler
CGIHandler().run(app)
