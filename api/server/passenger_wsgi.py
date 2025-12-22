# Optional Passenger entrypoint (only if Passenger is available)
import os
import sys

# ------------------------------------------------------------
# SarahMemory / PythonAnywhere WSGI bootstrap (api.sarahmemory.com)
# ------------------------------------------------------------

PROJECT_ROOT = "/home/Softdev0/SarahMemory"
API_SERVER_DIR = "/home/Softdev0/SarahMemory/api/server"

# 1) Predictable working directory (some libs use relative paths)
os.chdir(PROJECT_ROOT)

# 2) Ensure imports resolve correctly (API server first, then project root)
if API_SERVER_DIR not in sys.path:
    sys.path.insert(0, API_SERVER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3) Safe defaults for a headless/cloud environment
os.environ.setdefault("RUN_MODE", "cloud")
os.environ.setdefault("SARAH_DEVICE_MODE", "public_web")
os.environ.setdefault("FLASK_ENV", "production")
os.environ.setdefault("FLASK_DEBUG", "0")

# 4) Load .env if available (non-fatal)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
except Exception:
    pass

# 5) Basic logging so import errors show up clearly in PythonAnywhere error log
try:
    import logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
except Exception:
    pass

# 6) Import the Flask app
try:
    from app import app as application  # app.py lives in API_SERVER_DIR
except Exception:
    import traceback
    _err = traceback.format_exc()

    def application(environ, start_response):
        start_response(
            "500 Internal Server Error",
            [("Content-Type", "text/plain; charset=utf-8")],
        )
        return [("WSGI import failed:\n\n" + _err).encode("utf-8")]
