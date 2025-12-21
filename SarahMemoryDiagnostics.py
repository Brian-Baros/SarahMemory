"""--==The SarahMemory Project==--
File: SarahMemoryDiagnostics.py
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

import os
import logging
import platform
from datetime import datetime
import json
import subprocess
import SarahMemoryGlobals as config
import sqlite3
import sys
import socket
import ssl
import time
import stat

# Optional: access DB paths + cloud connector without duplicating logic
try:
    import SarahMemoryDatabase as SMDB
except Exception:
    SMDB = None

# Setup logger
logger = logging.getLogger("SarahMemoryDiagnostics")
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# Core functional files that must be present in the root directory
REQUIRED_FILES = [
    os.path.join(config.API_DIR, "server", "app.py"), #API LOCAL and CLOUD SERVER
    os.path.join(config.BASE_DIR, "SarahMemoryAdaptive.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAdvCU.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAI.py") if hasattr(config, "HAS_AI_CORE") else os.path.join(config.BASE_DIR, "SarahMemoryAI.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAPI.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryAvatar.py") if hasattr(config, "HAS_AVATAR") else os.path.join(config.BASE_DIR, "SarahMemoryAvatar.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryBrowser.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryCanvasStudio.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryCompare.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDatabase.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDiagnostics.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryDL.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryEncryption.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryExpressOut.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryFacialRecognition.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryFilesystem.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryGlobals.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryGUI.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryGUI2.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryHi.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryInitialization.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryIntegration.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryLedger.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryLyricsToSong.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryMain.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryMigrations.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryMusicGenerator.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryNetwork.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryOptimization.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryPersonality.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryReminder.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryReply.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryResearch.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySOJBE.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySi.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySynapes.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySync.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySystemIndexer.py") if hasattr(config, "HAS_INDEXER") else os.path.join(config.BASE_DIR, "SarahMemorySystemIndexer.py"),
    os.path.join(config.BASE_DIR, "SarahMemorySystemLearn.py") if hasattr(config, "HAS_SYSTEM_LEARN") else os.path.join(config.BASE_DIR, "SarahMemorySystemLearn.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryVideoEditorCore.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryVoice.py"),
    os.path.join(config.BASE_DIR, "SarahMemoryWebSYM.py"),
    os.path.join(config.BASE_DIR, "UnifiedAvatarController.py")
    # "SarahMemoryCleanup.py" Stand-Alone Tool, that is in DEVELOPMENT to CLEAN THE DATABASES
    # "SarahMemoryCleanupDaily.py" Stand-Alone Tool, Daily Database Cleaning tool
    # "SarahMemoryDBCreate.py" Stand-Alone Main File that Creates the initial Databases
    # "SarahMemoryLLM.py" Stand-Alone LLM and Object Downloader file. for easy installation.
    # "SarahMemorySystemIndexer.py" Stand-Alone Tool with built in GUI File that Indexes Entire Systems
    # "SarahMemorySystemLearn.py" Stand-Alone Tool that populates all created Databases with all information indexed.
    # "SarahMemoryStartup.py" Stand-Alone File in the C:\SarahMe...directory that allows for auto-run on systemboot up via Registry
]
# ============================
# PHASE A: Identity & Device Awareness (v7.7.5–8)
# ============================
def get_db_health_summary():
    """
    Phase B: Database / Mesh health summary helper.

    Returns a lightweight summary dictionary that callers can use to
    decide whether local SQLite DBs and (optionally) the cloud MySQL
    are healthy enough for normal operation.

    NOTE:
      * This MUST remain fast and side‑effect free (no writes).
      * It is safe to call at boot time.
    """
    summary = {
        "local_sqlite_size_mb": 0.0,
        "cloud_db_size_mb": None,
        "status": "unknown",
        "local_dbs": {},
        "cloud": {},
    }

    # --- Local SQLite sizes --------------------------------------------------
    try:
        datasets_dir = getattr(config, "DATASETS_DIR", None)
        # Prefer canonical paths from SarahMemoryDatabase if available
        ai_path = getattr(SMDB, "DB_PATH", None) if SMDB is not None else None
        user_path = getattr(SMDB, "USER_DB_PATH", None) if SMDB is not None else None
        chat_path = getattr(SMDB, "CHAT_DB", None) if SMDB is not None else None

        if datasets_dir:
            if not ai_path:
                ai_path = os.path.join(datasets_dir, "ai_learning.db")
            if not user_path:
                user_path = os.path.join(datasets_dir, "user_profile.db")
            if not chat_path:
                chat_path = os.path.join(datasets_dir, "context_history.db")
            logs_path = os.path.join(datasets_dir, "system_logs.db")
        else:
            logs_path = None

        local_paths = {
            "ai_learning.db": ai_path,
            "user_profile.db": user_path,
            "context_history.db": chat_path,
            "system_logs.db": logs_path,
        }

        total_bytes = 0
        for label, p in local_paths.items():
            if not p:
                continue
            try:
                if os.path.exists(p):
                    size_b = os.path.getsize(p)
                    total_bytes += size_b
                    summary["local_dbs"][label] = {
                        "path": p,
                        "size_mb": round(size_b / (1024 * 1024), 3),
                        "exists": True,
                    }
                else:
                    summary["local_dbs"][label] = {
                        "path": p,
                        "size_mb": 0.0,
                        "exists": False,
                    }
            except Exception as e:
                summary["local_dbs"][label] = {
                    "path": p,
                    "size_mb": None,
                    "exists": False,
                    "error": str(e),
                }
        summary["local_sqlite_size_mb"] = round(total_bytes / (1024 * 1024), 3)
    except Exception as e:
        summary.setdefault("errors", []).append(f"local_db_scan: {e}")

    # --- Cloud MySQL (size / basic health) -----------------------------------
    cloud_ok = False
    try:
        if SMDB is not None:
            get_conn = getattr(SMDB, "_get_cloud_conn", None)
        else:
            get_conn = None

        if callable(get_conn):
            conn = get_conn()
        else:
            conn = None

        if conn is not None:
            try:
                cur = conn.cursor()
                # Attempt a lightweight information_schema query if available
                cloud_db_name = getattr(config, "CLOUD_DB_NAME", None)
                cloud_size_mb = None
                if cloud_db_name:
                    try:
                        cur.execute(
                            """
                            SELECT SUM(data_length + index_length) / (1024*1024) AS size_mb
                            FROM information_schema.tables
                            WHERE table_schema = %s
                            """,
                            (cloud_db_name,),
                        )
                        row = cur.fetchone()
                        if row and row[0] is not None:
                            cloud_size_mb = float(row[0])
                    except Exception:
                        cloud_size_mb = None  # information_schema may not be available

                # Fallback: count tables
                tables = []
                try:
                    cur.execute("SHOW TABLES")
                    rows = cur.fetchall()
                    tables = [r[0] for r in rows] if rows else []
                except Exception:
                    tables = []

                summary["cloud"] = {
                    "db_host": getattr(config, "CLOUD_DB_HOST", None),
                    "db_name": getattr(config, "CLOUD_DB_NAME", None),
                    "tables_count": len(tables),
                }
                if cloud_size_mb is not None:
                    summary["cloud_db_size_mb"] = round(cloud_size_mb, 3)

                cloud_ok = True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as e:
        summary.setdefault("errors", []).append(f"cloud_db_scan: {e}")

    # --- Overall status ------------------------------------------------------
    if cloud_ok:
        summary["status"] = "ok"
    elif summary["local_sqlite_size_mb"] > 0:
        summary["status"] = "local_only"
    else:
        summary["status"] = "degraded"

    return summary
# =====================================
#insert all of PHASE A: ABOVE THIS LINE
# =====================================


def log_diagnostics_event(event, details):
    """
    Logs a diagnostics event to the system_logs.db database.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnostics_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event TEXT,
            details TEXT
        )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO diagnostics_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged diagnostics event to system_logs.db successfully.")
    except Exception as e:
        logger.error(f"Error logging diagnostics event to system_logs.db: {e}")

def append_diag_log(section: str, message: str) -> None:
    """
    Append a single-line human-readable diagnostics summary
    to LOGS_DIR/diag_report.log.
    """
    try:
        log_dir = getattr(config, "LOGS_DIR", None)
        if not log_dir:
            # Fallback: put logs under BASE_DIR/logs if LOGS_DIR is missing
            base_dir = getattr(config, "BASE_DIR", os.getcwd())
            log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "diag_report.log")
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] [{section}] {message}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        logger.info(f"Diagnostics summary appended to {log_path}")
    except Exception as e:
        logger.error(f"Error writing diagnostics log file: {e}")

def run_self_check():
    """
    Perform system diagnostics to validate dependencies, configurations, and external connectivity.
    ENHANCED (v6.4): Now checks Python version, required packages, environment variables, and simulates connectivity.
    NEW: Generates a JSON summary diagnostic report.
    """
    logger.info("Running system diagnostics...")
    log_diagnostics_event("Self Check Start", "Beginning system diagnostics.")
    diagnostics_report = {}

    missing_files = []
    for file_path in REQUIRED_FILES:
        if not os.path.exists(file_path):
            warning_msg = f"Missing required file: {file_path}"
            logger.warning(warning_msg)
            log_diagnostics_event("Missing File", warning_msg)
            missing_files.append(file_path)
        else:
            info_msg = f"Verified file: {os.path.basename(file_path)}"
            logger.info(info_msg)
            log_diagnostics_event("Verified File", info_msg)
    diagnostics_report["missing_files"] = missing_files

    # Check Python version
    py_version = platform.python_version()
    diagnostics_report["python_version"] = py_version
    logger.info(f"Python version: {py_version}")
    log_diagnostics_event("Python Version", f"Python version: {py_version}")

    # Check required environment variables
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "Not Set"),
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY", "Not Set")
    }
    diagnostics_report["environment_variables"] = env_vars
    for var, val in env_vars.items():
        if val == "Not Set":
            logger.warning(f"{var} not set in environment.")
            log_diagnostics_event("Env Var Warning", f"{var} not set.")
        else:
            logger.info(f"{var} is set.")
            log_diagnostics_event("Env Var Check", f"{var} is set.")

    # Check connectivity to external services (e.g., SarahMemory API)
    try:
        import requests
        api_url = getattr(config, "DEFAULT_API_URL", "https://api.sarahmemory.com/api/health")
        logger.info(f"Checking connectivity to {api_url}...")
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            diagnostics_report["api_connectivity"] = "Online"
            logger.info("API connectivity: Online")
            log_diagnostics_event("API Connectivity", "Successfully connected to SarahMemory API.")
        else:
            diagnostics_report["api_connectivity"] = f"Unstable (Status: {response.status_code})"
            logger.warning(f"API connectivity: Unstable (Status: {response.status_code})")
            log_diagnostics_event("API Connectivity", f"Unstable (Status: {response.status_code})")
    except Exception as e:
        diagnostics_report["api_connectivity"] = f"Offline ({e})"
        logger.warning(f"API connectivity: Offline ({e})")
        log_diagnostics_event("API Connectivity", f"Offline ({e})")

    # Check network connectivity (basic ping)
    try:
        if platform.system().lower() == "windows":
            ping_cmd = ["ping", "-n", "1", "8.8.8.8"]
        else:
            ping_cmd = ["ping", "-c", "1", "8.8.8.8"]
        subprocess.run(ping_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        diagnostics_report["network_status"] = "Online"
        logger.info("Network connectivity: Online")
        log_diagnostics_event("Network Check", "Network connectivity verified.")
    except Exception:
        diagnostics_report["network_status"] = "Offline"
        logger.warning("Network connectivity: Offline")
        log_diagnostics_event("Network Check", "Network connectivity failed.")

    # NEW: Check system resource summary
    diagnostics_report["system"] = {
        "platform": platform.system(),
        "release": platform.release(),
        "processor": platform.processor()
    }
    logger.info(f"System info: {diagnostics_report['system']}")
    log_diagnostics_event("System Info", json.dumps(diagnostics_report["system"]))

    logger.info("Diagnostics complete.")
    log_diagnostics_event("Self Check Complete", "System diagnostics finished.")
    # NEW: Output JSON diagnostic report
    report_json = json.dumps(diagnostics_report, indent=2)
    logger.debug("Diagnostic Report:\n%s", report_json)
    return diagnostics_report

# NEW: Asynchronous wrapper to run diagnostics in background
def run_diagnostics_async():
    """
    Run the self-check diagnostics in a background thread.
    NEW: Uses run_async for non-blocking execution.
    """
    from SarahMemoryGlobals import run_async
    run_async(run_self_check)

def run_personality_core_diagnostics():
    """
    Validate all Core-Brain modules: Personality, Adaptive Memory, Emotion, DL, and Intent.
    ENHANCED (v6.4): Adds simulated deep learning module checks and detailed health metrics.
    ENHANCED (v7.1.1): Prevents Dataset bloating upon system startup.
    """
    logger.info("=== Running Personality Core-Brain Diagnostics ===")
    log_diagnostics_event("Personality Diagnostics Start", "Starting diagnostics for Core-Brain modules.")

    # Skip personality diagnostics in SAFE_MODE to avoid heavy DB checks and DL calls
    try:
        from SarahMemoryGlobals import SAFE_MODE  # noqa: F401
        if SAFE_MODE:
            logger.info("SAFE_MODE active; skipping Personality Core-Brain Diagnostics.")
            log_diagnostics_event("Personality Diagnostics Skipped", "SAFE_MODE active.")
            return
    except Exception:
        pass

    # Personality module check
    try:
        from SarahMemoryPersonality import integrate_with_personality
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM responses
            WHERE intent = ?
        """, ("greeting",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            result = integrate_with_personality("Hello there!")
            assert isinstance(result, str)
            msg = "Personality module responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Personality Module", msg)
        else:
            logger.info("🟡 Personality test already exists in memory. Skipping reinsert.")
        conn.close()
    except Exception as e:
        error_msg = f"Personality module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Personality Module Error", error_msg)

    # Adaptive memory module check
    try:
        from SarahMemoryAdaptive import update_personality
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM interactions
            WHERE intent = ?
        """, ("test_adaptive",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            metrics = update_personality("I love working with you", "Thanks, I appreciate that!")
            assert isinstance(metrics, dict) and "engagement" in metrics
            msg = "Adaptive memory module responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Adaptive Memory Module", msg)
        else:
            logger.info("🟡 Adaptive personality test already exists in memory. Skipping reinsert.")
        conn.close()
    except Exception as e:
        error_msg = f"Adaptive memory module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Adaptive Memory Module Error", error_msg)

    # Deep Learning evaluator check (optional, lightweight)
    try:
        from SarahMemoryDL import evaluate_conversation_patterns
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM dl_cache
            WHERE pattern_type = ?
        """, ("diagnostic",))
        exists = cursor.fetchone()[0]
        if exists == 0:
            patterns = evaluate_conversation_patterns()
            assert isinstance(patterns, dict)
            msg = "Deep learning analyzer responded: OK"
            logger.info("✅ " + msg)
            log_diagnostics_event("Deep Learning Analyzer", msg)
        else:
            logger.info("🟡 DL diagnostics pattern already recorded. Skipping.")
        conn.close()
    except Exception as e:
        error_msg = f"Deep learning analyzer module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Deep Learning Analyzer Error", error_msg)

    # Intent classifier check
    try:
        from SarahMemoryAdvCU import classify_intent
        sample_intent = classify_intent("Hello, how are you?")
        msg = f"Intent classifier responded with: {sample_intent}"
        logger.info("✅ " + msg)
        log_diagnostics_event("Intent Classifier", msg)
    except Exception as e:
        error_msg = f"Intent classifier module failed: {e}"
        logger.error("❌ " + error_msg)
        log_diagnostics_event("Intent Classifier Error", error_msg)

    logger.info("=== Personality Core-Brain Diagnostics Complete ===")
    log_diagnostics_event("Personality Diagnostics Complete", "Core-Brain diagnostics finished.")

def run_webui_bridge_diagnostics(app_path=None, js_path=None):
    """
    SuperTask (v7.7.5): Web UI bridge diagnostics for app.py + app.js.

    This is a static, non-destructive checker that:
      * Verifies that app.py and app.js exist in expected locations under BASE_DIR.
      * Inspects app.py for Flask routes /api/health and /api/chat.
      * Inspects app.js for SM_pollHealth() and SM_getApi('/chat') usage.

    It is designed to be:
      * Runnable from console:
            python SarahMemoryDiagnostics.py webui
      * Callable from other Python modules:
            from SarahMemoryDiagnostics import run_webui_bridge_diagnostics
            report = run_webui_bridge_diagnostics()

    It never runs automatically at normal SarahMemory boot; it is only triggered
    when explicitly called.
    """
    report = {"ok": False, "checks": []}

    def _add_check(file_name, target, kind, line, passed, detail):
        status = "PASS" if passed else "FAIL"
        entry = {
            "file": file_name,
            "target": target,
            "kind": kind,
            "line": int(line) if isinstance(line, int) else None,
            "status": status,
            "detail": detail,
        }
        report["checks"].append(entry)
        try:
            msg = f"{file_name} :: {target} ({kind}) @ line {entry['line'] or '?'} => {status} — {detail}"
            log_diagnostics_event("WebUI SuperTask", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            # Logging should never break diagnostics
            pass

    def _first_existing(paths):
        for p in paths:
            if p and os.path.exists(p):
                return p
        return None

    base_dir = getattr(config, "BASE_DIR", os.getcwd())

    # --- app.py checks -------------------------------------------------------
    if app_path is None:
        candidate_app_paths = [
            os.path.join(base_dir, "app.py"),
            os.path.join(base_dir, "server", "app.py"),
            os.path.join(base_dir, "api", "server", "app.py"),
            os.path.join(base_dir, "api", "app.py"),
        ]
        app_path = _first_existing(candidate_app_paths)

    if not app_path or not os.path.exists(app_path):
        _add_check("app.py", "file", "presence", None, False,
                   f"app.py not found under BASE_DIR={base_dir}")
    else:
        _add_check(os.path.basename(app_path), "file", "presence", 1, True,
                   f"Found app.py at {app_path}")
        try:
            with open(app_path, "r", encoding="utf-8") as f:
                app_src = f.read()
        except Exception as e:
            _add_check(os.path.basename(app_path), "file", "read", None, False,
                       f"Could not read app.py: {e}")
            app_src = ""

        if app_src:
            # Helper to map index → line number
            def _idx_to_line(txt, idx):
                return txt.count("\n", 0, idx) + 1

            import re as _re

            for route in ("/api/health", "/api/chat"):
                m = _re.search(r"@app\.route\(['\"]" + _re.escape(route) + r"['\"][^)]*\)", app_src)
                if m:
                    line = _idx_to_line(app_src, m.start())
                    _add_check(os.path.basename(app_path), route, "route", line,
                               True, f"Route {route} is registered in app.py.")
                else:
                    _add_check(os.path.basename(app_path), route, "route", None,
                               False, f"Route {route} not found in app.py.")

    # --- app.js checks -------------------------------------------------------
    if js_path is None:
        candidate_js_paths = [
            os.path.join(base_dir, "data", "ui", "app.js"),
            os.path.join(base_dir, "api", "data", "ui", "app.js"),
            os.path.join(base_dir, "server", "static", "app.js"),
            os.path.join(base_dir, "app.js"),
        ]
        js_path = _first_existing(candidate_js_paths)

    if not js_path or not os.path.exists(js_path):
        _add_check("app.js", "file", "presence", None, False,
                   f"app.js not found under BASE_DIR={base_dir}")
    else:
        _add_check(os.path.basename(js_path), "file", "presence", 1, True,
                   f"Found app.js at {js_path}")
        try:
            with open(js_path, "r", encoding="utf-8") as f:
                js_src = f.read()
        except Exception as e:
            _add_check(os.path.basename(js_path), "file", "read", None, False,
                       f"Could not read app.js: {e}")
            js_src = ""

        if js_src:
            def _idx_to_line_js(txt, idx):
                return txt.count("\n", 0, idx) + 1

            import re as _re

            # SM_pollHealth presence
            m_ph = _re.search(r"function\s+SM_pollHealth\s*\(", js_src)
            if m_ph:
                line = _idx_to_line_js(js_src, m_ph.start())
                _add_check(os.path.basename(js_path), "SM_pollHealth", "function", line,
                           True, "SM_pollHealth() helper is defined.")
            else:
                _add_check(os.path.basename(js_path), "SM_pollHealth", "function", None,
                           False, "SM_pollHealth() helper not found in app.js.")

            # Chat bridge via SM_getApi('/chat')
            m_chat = _re.search(r"SM_getApi\(['\"]/chat['\"]\)", js_src)
            if m_chat:
                line = _idx_to_line_js(js_src, m_chat.start())
                _add_check(os.path.basename(js_path), "/chat", "bridge", line,
                           True, "Chat submit uses SM_getApi('/chat') for API base resolution.")
            else:
                _add_check(os.path.basename(js_path), "/chat", "bridge", None,
                           False, "SM_getApi('/chat') not found; chat bridge wiring may be incorrect.")

    # Overall status
    if report["checks"]:
        report["ok"] = all(c.get("status") == "PASS" for c in report["checks"])
    else:
        report["ok"] = False

    return report


def _print_webui_bridge_report(report):
    """
    Pretty-print the Web UI bridge diagnostics report as a compact PASS/FAIL table.
    """
    checks = report.get("checks") or []
    if not checks:
        print("No checks were run.")
        return

    # Determine column widths
    file_w = max(len("FILE"), max(len(c["file"]) for c in checks))
    target_w = max(len("TARGET"), max(len(str(c["target"])) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    line_w = max(len("LINE"), 4)
    status_w = max(len("STATUS"), 4)

    header = f"{'FILE'.ljust(file_w)}  {'TARGET'.ljust(target_w)}  {'KIND'.ljust(kind_w)}  {'LINE'.ljust(line_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for c in checks:
        line_s = str(c["line"]) if c["line"] is not None else "-"
        row = (
            f"{c['file'].ljust(file_w)}  "
            f"{str(c['target']).ljust(target_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{line_s.ljust(line_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# [PATCH v7.7.5 Phase 2] WebUI HTTPS / CORS / Network diagnostics
def run_webui_network_diagnostics(ui_url=None, api_base=None):
    """
    SuperTask Phase 2 (v7.7.5): Web UI network, HTTPS, and CORS diagnostics.

    This is a non-destructive external connectivity checker that:
      * Verifies the public UI (e.g., https://ai.sarahmemory.com) is reachable and using HTTPS.
      * Verifies the public API health endpoint is reachable and using HTTPS.
      * Performs a CORS preflight OPTIONS request against /api/chat using the UI as Origin.

    It is designed to be:
      * Runnable from console:
            python SarahMemoryDiagnostics.py webui2
      * Callable from other Python modules:
            from SarahMemoryDiagnostics import run_webui_network_diagnostics
            report = run_webui_network_diagnostics()

    It never runs automatically at normal SarahMemory boot; it is only triggered
    when explicitly called.
    """
    report = {"ok": False, "checks": []}

    def _add(component, target, kind, passed, detail):
        status = "PASS" if passed else "FAIL"
        entry = {
            "component": component,
            "target": target,
            "kind": kind,
            "status": status,
            "detail": detail,
        }
        report["checks"].append(entry)
        try:
            msg = f"{component} :: {target} ({kind}) => {status} — {detail}"
            log_diagnostics_event("WebUI SuperTask Phase 2", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            # Logging should never break diagnostics
            pass

    # Ensure requests is available
    try:
        import requests  # type: ignore
        from urllib.parse import urlparse, urlunparse
    except Exception as e:
        _add("network", "requests", "import", False, f"'requests' not available: {e}")
        report["ok"] = False
        return report

    # Determine public UI URL
    ui_origin = (
        ui_url
        or getattr(config, "PUBLIC_UI_URL", None)
        or getattr(config, "UI_BASE_URL", None)
        or "https://ai.sarahmemory.com"
    )

    # Determine API base and health/chat endpoints
    api_hint = (
        api_base
        or getattr(config, "PUBLIC_API_BASE", None)
        or getattr(config, "PUBLIC_API_URL", None)
        or getattr(config, "DEFAULT_API_URL", None)
        or "https://api.sarahmemory.com/api/health"
    )

    parsed = urlparse(api_hint)
    path = parsed.path or ""

    # Derive API health and chat URLs in a robust way
    if path.endswith("/api/health"):
        api_health = urlunparse(parsed._replace(query="", fragment=""))
        api_base_derived = urlunparse(parsed._replace(path="/", query="", fragment="")).rstrip("/")
        api_chat = api_health.rsplit("/health", 1)[0] + "chat"
    elif path.endswith("/api") or path.endswith("/api/"):
        api_base_derived = urlunparse(parsed._replace(query="", fragment="")).rstrip("/")
        api_health = api_base_derived.rstrip("/") + "/health"
        api_chat = api_base_derived.rstrip("/") + "/chat"
    else:
        api_base_derived = urlunparse(parsed._replace(query="", fragment="")).rstrip("/")
        api_health = api_base_derived.rstrip("/") + "/api/health"
        api_chat = api_base_derived.rstrip("/") + "/api/chat"

    # -------------------------------------------------------------------------
    # UI reachability + HTTPS scheme + SSL
    # -------------------------------------------------------------------------
    try:
        r = requests.get(ui_origin, timeout=8)
        _add("ui", ui_origin, "reachability", r.ok, f"HTTP {r.status_code}")
    except requests.exceptions.SSLError as e:
        _add("ui", ui_origin, "ssl", False, f"SSL error when connecting to UI: {e}")
    except Exception as e:
        _add("ui", ui_origin, "reachability", False, f"UI not reachable: {e}")
    else:
        if ui_origin.lower().startswith("https://"):
            _add("ui", ui_origin, "scheme", True, "UI is using HTTPS.")
        else:
            _add(
                "ui",
                ui_origin,
                "scheme",
                False,
                "UI is not using HTTPS; some browsers may block or warn about mixed content.",
            )

    # -------------------------------------------------------------------------
    # API health reachability + HTTPS scheme + SSL
    # -------------------------------------------------------------------------
    try:
        r = requests.get(api_health, timeout=8)
        _add("api", api_health, "health", r.ok, f"HTTP {r.status_code}")
    except requests.exceptions.SSLError as e:
        _add("api", api_health, "ssl", False, f"SSL error when connecting to API: {e}")
    except Exception as e:
        _add("api", api_health, "health", False, f"API health not reachable: {e}")
    else:
        if api_health.lower().startswith("https://"):
            _add("api", api_health, "scheme", True, "API health endpoint is using HTTPS.")
        else:
            _add(
                "api",
                api_health,
                "scheme",
                False,
                "API health endpoint is not using HTTPS; modern browsers may block cross-origin calls.",
            )

    # -------------------------------------------------------------------------
    # CORS preflight for /api/chat using the UI as Origin
    # -------------------------------------------------------------------------
    cors_headers = {
        "Origin": ui_origin.rstrip("/"),
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    try:
        r = requests.options(api_chat, headers=cors_headers, timeout=8)
        ok_status = 200 <= r.status_code < 400
        _add("api", api_chat, "cors_preflight", ok_status, f"OPTIONS HTTP {r.status_code}")
        acao = r.headers.get("Access-Control-Allow-Origin", "")
        acam = r.headers.get("Access-Control-Allow-Methods", "")
        if acao in ("*", ui_origin.rstrip("/")):
            _add("api", api_chat, "cors_origin", True, f"Access-Control-Allow-Origin={acao!r}")
        else:
            _add(
                "api",
                api_chat,
                "cors_origin",
                False,
                f"Access-Control-Allow-Origin={acao!r} (expected '*' or {ui_origin!r})",
            )
        if "POST" in acam.upper():
            _add("api", api_chat, "cors_methods", True, f"Access-Control-Allow-Methods={acam!r}")
        else:
            _add(
                "api",
                api_chat,
                "cors_methods",
                False,
                f"Access-Control-Allow-Methods={acam!r} (POST not listed)",
            )
    except requests.exceptions.SSLError as e:
        _add("api", api_chat, "cors_ssl", False, f"SSL error during CORS preflight: {e}")
    except Exception as e:
        _add("api", api_chat, "cors_preflight", False, f"CORS preflight failed: {e}")

    # Overall status: only consider non-import checks
    checks = report["checks"]
    if checks:
        report["ok"] = all(c["status"] == "PASS" for c in checks if c["kind"] not in ("import",))
    else:
        report["ok"] = False

    return report


def _print_webui_network_report(report):
    """
    Pretty-print the Web UI network diagnostics report as a compact PASS/FAIL table.
    """
    checks = report.get("checks") or []
    if not checks:
        print("No network checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    target_w = max(len("TARGET"), max(len(str(c["target"])) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'TARGET'.ljust(target_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['target']).ljust(target_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


class DatabaseDiagnosticsSuperTask:
    """
    SuperTask Phase 3 (v7.7.5): Database connectivity + health diagnostics.

    This is a non-destructive checker that:

      * Verifies local SQLite databases exist and are readable in read-only mode.
        - ai_learning.db
        - user_profile.db
        - system_logs.db

      * Verifies the configured cloud MySQL (CLOUD_DB_*) can be connected to and
        that it has tables and (optionally) data rows.

    It is designed to be:

      * Runnable from console:

            python SarahMemoryDiagnostics.py database

      * Callable from other Python modules:

            from SarahMemoryDiagnostics import DatabaseDiagnosticsSuperTask
            report = DatabaseDiagnosticsSuperTask().run()

    It NEVER writes or inserts anything, and is only triggered when explicitly
    requested (it does not run during normal SarahMemory boot).
    """

    def __init__(self):
        self.report = {"ok": False, "checks": []}

    def _add(self, backend, target, kind, passed, detail):
        status = "PASS" if passed else "FAIL"
        entry = {
            "backend": backend,
            "target": target,
            "kind": kind,
            "status": status,
            "detail": detail,
        }
        self.report["checks"].append(entry)
        try:
            msg = f"{backend} :: {target} ({kind}) => {status} — {detail}"
            log_diagnostics_event("Database SuperTask", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            # Logging must never break diagnostics
            pass

    # --- Local SQLite checks -------------------------------------------------

    def _check_sqlite_db(self, label, path):
        """
        Read-only check for a local SQLite DB:
          * file presence
          * readable in mode=ro
          * has at least one table
          * (optional) first table has >=0 rows (verifies SELECT works)
        """
        target = path or "<unknown>"
        if not path or not os.path.exists(path):
            self._add(label, target, "file", False, "Database file not found.")
            return

        # Try read-only connection via URI to avoid accidental writes
        uri = f"file:{path}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            self._add(label, target, "connect", True, "Opened SQLite DB in read-only mode.")
        except Exception as e:
            self._add(label, target, "connect", False, f"Failed to open SQLite DB (read-only): {e}")
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            row = cursor.fetchone()
            if not row:
                self._add(label, target, "tables", False, "No tables found in sqlite_master.")
                return
            table_name = row[0]
            self._add(label, target, "tables", True, f"Found at least one table: '{table_name}'.")

            # Light-weight row-count check for that table (read-only)
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                if count > 0:
                    self._add(
                        label,
                        target,
                        "data",
                        True,
                        f"Table '{table_name}' has {count} rows (data exists).",
                    )
                else:
                    self._add(
                        label,
                        target,
                        "data",
                        False,
                        f"Table '{table_name}' has 0 rows; DB appears empty.",
                    )
            except Exception as e:
                self._add(
                    label,
                    target,
                    "data",
                    False,
                    f"Row-count check on table '{table_name}' failed (SELECT only): {e}",
                )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # --- Cloud MySQL (PythonAnywhere / GoogieHost) ---------------------------

    def _check_cloud_mysql(self):
        """
        Check the configured cloud MySQL database (CLOUD_DB_* in SarahMemoryGlobals).

        This uses the existing SarahMemoryDatabase._get_cloud_conn() helper if available,
        so we don't duplicate connection logic.

        It reports PASS/FAIL for:
          * connection
          * tables presence
          * data presence in at least one table (COUNT(*) only)
        """
        if SMDB is None:
            self._add(
                "cloud-mysql",
                "CLOUD_DB_*",
                "import",
                False,
                "SarahMemoryDatabase module is not available; cannot test cloud DB.",
            )
            return

        # Determine environment labeling based on host/name
        host = getattr(config, "CLOUD_DB_HOST", None)
        name = getattr(config, "CLOUD_DB_NAME", None)

        label_env = "cloud-mysql"
        host_l = (host or "").lower()
        name_l = (name or "").lower()

        if "pythonanywhere" in host_l:
            label_env = "pythonanywhere-mysql"
        elif "googiehost" in host_l or "softdevc" in name_l:
            label_env = "googiehost-mysql"

        target = f"{label_env}@{host or 'unknown_host'}/{name or '?'}"

        # Use the shared cloud-connection helper
        get_conn = getattr(SMDB, "_get_cloud_conn", None)
        if not callable(get_conn):
            self._add(
                "cloud-mysql",
                target,
                "config",
                False,
                "SarahMemoryDatabase._get_cloud_conn() not found; cannot open cloud DB.",
            )
            return

        conn = None
        try:
            conn = get_conn()
            if conn is None:
                self._add(
                    "cloud-mysql",
                    target,
                    "connect",
                    False,
                    "Cloud DB disabled or connection failed (CLOUD_DB_ENABLED or credentials).",
                )
                return

            self._add(
                "cloud-mysql",
                target,
                "connect",
                True,
                "Cloud MySQL connection established successfully.",
            )

            cursor = conn.cursor()

            # SHOW TABLES (non-destructive)
            try:
                cursor.execute("SHOW TABLES")
                rows = cursor.fetchall()
                tables = [r[0] for r in rows] if rows else []
                if not tables:
                    self._add(
                        "cloud-mysql",
                        target,
                        "tables",
                        False,
                        "SHOW TABLES returned 0 tables; schema may not be initialized.",
                    )
                    return
                preview = ", ".join(str(t) for t in tables[:3])
                self._add(
                    "cloud-mysql",
                    target,
                    "tables",
                    True,
                    f"{len(tables)} tables present (e.g., {preview}).",
                )

                # Simple COUNT(*) on the first table to verify data reads
                first_table = tables[0]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{first_table}`")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        self._add(
                            "cloud-mysql",
                            target,
                            "data",
                            True,
                            f"Table '{first_table}' has {count} rows (data present).",
                        )
                    else:
                        self._add(
                            "cloud-mysql",
                            target,
                            "data",
                            False,
                            f"Table '{first_table}' has 0 rows; data may not be written yet.",
                        )
                except Exception as e:
                    self._add(
                        "cloud-mysql",
                        target,
                        "data",
                        False,
                        f"Row-count check on '{first_table}' failed (SELECT only): {e}",
                    )
            except Exception as e:
                self._add(
                    "cloud-mysql",
                    target,
                    "tables",
                    False,
                    f"SHOW TABLES failed: {e}",
                )
        except Exception as e:
            self._add(
                "cloud-mysql",
                target,
                "connect",
                False,
                f"Cloud MySQL connection error: {e}",
            )
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # --- Orchestrator --------------------------------------------------------

    def run(self):
        """
        Run all database diagnostics and return a report dict:

            {
              "ok": bool,
              "checks": [
                {"backend": ..., "target": ..., "kind": ..., "status": "PASS"/"FAIL", "detail": ...},
                ...
              ]
            }
        """
        # Local SQLite DBs
        ai_path = None
        user_path = None

        if SMDB is not None:
            ai_path = getattr(SMDB, "DB_PATH", None)
            user_path = getattr(SMDB, "USER_DB_PATH", None)

        if not ai_path:
            ai_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        if not user_path:
            user_path = os.path.join(config.DATASETS_DIR, "user_profile.db")

        system_logs_path = os.path.join(config.DATASETS_DIR, "system_logs.db")

        self._check_sqlite_db("sqlite-ai_learning", ai_path)
        self._check_sqlite_db("sqlite-user_profile", user_path)
        self._check_sqlite_db("sqlite-system_logs", system_logs_path)

        # Cloud MySQL (PythonAnywhere / GoogieHost depending on CLOUD_DB_HOST)
        self._check_cloud_mysql()

        checks = self.report["checks"]
        if checks:
            self.report["ok"] = all(c["status"] == "PASS" for c in checks)
        else:
            self.report["ok"] = False

        return self.report


def _print_database_report(report):
    """
    Pretty-print the Database SuperTask report as a compact PASS/FAIL table.
    """
    checks = report.get("checks") or []
    if not checks:
        print("No database checks were run.")
        return

    backend_w = max(len("BACKEND"), max(len(c["backend"]) for c in checks))
    target_w = max(len("TARGET"), max(len(str(c["target"])) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'BACKEND'.ljust(backend_w)}  {'TARGET'.ljust(target_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for c in checks:
        row = (
            f"{c['backend'].ljust(backend_w)}  "
            f"{str(c['target']).ljust(target_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)



# ---------------------------------------------------------------------------
# Phase B — Context / Mesh / Agent Diagnostics
# ---------------------------------------------------------------------------

def run_context_mesh_agent_diagnostics() -> dict:
    """
    Phase B SuperTask: Context / Mesh / Agent health diagnostics.

    Console usage:
        python SarahMemoryDiagnostics.py phase-b

    This is a read-only, non-destructive check that verifies:

      * Context DB schema is present and can be read.
      * Mesh / hub / sync flags are coherent.
      * Agent permissions are wired correctly.
      * Intent + planner pipeline can be invoked without executing tools.
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("PhaseB Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    # --- Phase B helpers from Globals ---------------------------------------
    ctx_cfg = None
    mesh_cfg = None
    agent_cfg = None

    try:
        from SarahMemoryGlobals import get_context_config, get_mesh_sync_config, agent_permissions_summary
        try:
            ctx_cfg = get_context_config()
            _add("context", "config", True, f"Context config loaded: {ctx_cfg}")
        except Exception as e:
            _add("context", "config", False, f"get_context_config() failed: {e}")

        try:
            mesh_cfg = get_mesh_sync_config()
            _add("mesh", "config", True, f"Mesh config loaded: {mesh_cfg}")
        except Exception as e:
            _add("mesh", "config", False, f"get_mesh_sync_config() failed: {e}")

        try:
            agent_cfg = agent_permissions_summary()
            _add("agent", "permissions", True, f"Agent permissions: {agent_cfg}")
        except Exception as e:
            _add("agent", "permissions", False, f"agent_permissions_summary() failed: {e}")
    except Exception as e:
        _add("globals", "phase_b_helpers", False, f"Phase B helpers import failed: {e}")

    # --- Context DB schema + load check -------------------------------------
    if SMDB is not None:
        try:
            ensure_schema = getattr(SMDB, "ensure_context_turn_schema", None)
            load_recent = getattr(SMDB, "load_recent_context_turns", None)
            if callable(ensure_schema):
                ensure_schema()
                _add("context_db", "schema", True, "context_turns table ensured.")
            else:
                _add("context_db", "schema", False, "ensure_context_turn_schema() not available.")

            if callable(load_recent):
                try:
                    rows = load_recent(max_turns=3)
                    n = len(rows or [])
                    _add("context_db", "load_recent", True, f"Loaded {n} recent context turns.")
                except Exception as e:
                    _add("context_db", "load_recent", False, f"load_recent_context_turns() failed: {e}")
            else:
                _add("context_db", "load_recent", False, "load_recent_context_turns() not available.")

            if not callable(ensure_schema) and not callable(load_recent):
                _add("context_db", "availability", False, "No Phase B context DB helpers available.")
        except Exception as e:
            _add("context_db", "schema", False, f"Context DB diagnostics failed: {e}")
    else:
        _add("context_db", "availability", False, "SarahMemoryDatabase module not available.")

    # --- Intent + planner pipeline smoke test (no tool execution) -----------
    try:
        from SarahMemoryAdvCU import classify_intent
        from SarahMemoryAiFunctions import plan_actions
        sample_texts = [
            "What time is it in New York?",
            "Open calculator and add 2 + 2",
            "Search the web for SarahMemory project",
        ]
        for txt in sample_texts:
            try:
                intent = classify_intent(txt)
                actions = plan_actions(intent, entities=None, user_context={"raw_text": txt}, device_context=None)
                detail = f"intent={intent!r}, planned_actions={actions!r}"
                _add("intent_planner", "pipeline", True, f"{txt!r} -> {detail}")
            except Exception as e:
                _add("intent_planner", "pipeline", False, f"Pipeline failed for {txt!r}: {e}")
    except Exception as e:
        _add("intent_planner", "import", False, f"Import failed for intent/planner modules: {e}")

    # --- Mesh sync + QA cache sync smoke (optional) -------------------------
    try:
        if SMDB is not None:
            sync_qa = getattr(SMDB, "sync_qa_cache_from_cloud", None)
            if callable(sync_qa):
                # Do a very small pull to verify call-path; this is still non-destructive
                pulled = sync_qa(limit=5)
                _add("mesh_sync", "qa_cache", True, f"sync_qa_cache_from_cloud(limit=5) pulled {pulled} entries.")
            else:
                _add("mesh_sync", "qa_cache", False, "sync_qa_cache_from_cloud() not available.")
        else:
            _add("mesh_sync", "qa_cache", False, "SarahMemoryDatabase not available for QA sync.")
    except Exception as e:
        _add("mesh_sync", "qa_cache", False, f"QA sync smoke test failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_context_mesh_agent_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No Phase B checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 4 — API Diagnostics
# ---------------------------------------------------------------------------

def run_api_diagnostics(api_base: str | None = None) -> dict:
    """
    SuperTask Phase 4: API Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py api
    """
    report: dict = {"ok": False, "checks": []}

    def _add(target: str, kind: str, passed: bool, detail: str, latency_ms: float | None = None):
        status = "PASS" if passed else "FAIL"
        entry = {"target": target, "kind": kind, "status": status, "detail": detail}
        if latency_ms is not None:
            entry["latency_ms"] = round(latency_ms, 1)
        report["checks"].append(entry)
        msg = f"{target} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("API Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    try:
        import requests  # type: ignore
        from urllib.parse import urlparse, urlunparse
    except Exception as e:
        _add("requests", "import", False, f"'requests' not available: {e}")
        report["ok"] = False
        return report

    api_hint = (
        api_base
        or getattr(config, "PUBLIC_API_BASE", None)
        or getattr(config, "PUBLIC_API_URL", None)
        or getattr(config, "DEFAULT_API_URL", None)
        or "https://api.sarahmemory.com/api/health"
    )

    parsed = urlparse(api_hint)
    path = parsed.path or ""

    if path.endswith("/api/health"):
        api_health = urlunparse(parsed._replace(query="", fragment=""))
        api_root = api_health.rsplit("/api/health", 1)[0] + "/api"
    elif path.endswith("/api") or path.endswith("/api/"):
        api_root = urlunparse(parsed._replace(query="", fragment="")).rstrip("/")
        api_health = api_root.rstrip("/") + "/health"
    else:
        api_root = urlunparse(parsed._replace(path="/api", query="", fragment="")).rstrip("/")
        api_health = api_root.rstrip("/") + "/health"

    api_chat = api_root.rstrip("/") + "/chat"

    # /api/health
    try:
        t0 = time.perf_counter()
        r = requests.get(api_health, timeout=15)
        lat = (time.perf_counter() - t0) * 1000.0
        ok = r.status_code == 200
        _add(api_health, "health_http", ok, f"HTTP {r.status_code}", lat)
        if ok:
            try:
                payload = r.json()
                keys = list(payload.keys())
                _add(api_health, "health_json", True, f"JSON keys: {keys}")
                meta_val = payload.get("model") or payload.get("models") or payload.get("provider")
                if meta_val:
                    _add(api_health, "routing_meta", True, f"Model/provider info: {meta_val!r}")
                else:
                    _add(api_health, "routing_meta", False, "No explicit model/provider info.")
            except Exception as e:
                _add(api_health, "health_json", False, f"JSON parse failed: {e}")
    except Exception as e:
        _add(api_health, "health_http", False, f"Request failed: {e}")

    # /api/chat
    try:
        payload = {
            "message": "Diagnostics test: What is your name?",
            "meta": {"source": "SarahMemoryDiagnostics", "mode": "diagnostics"},
        }
        t0 = time.perf_counter()
        r = requests.post(api_chat, json=payload, timeout=30)
        lat = (time.perf_counter() - t0) * 1000.0
        ok = r.status_code == 200
        _add(api_chat, "chat_http", ok, f"HTTP {r.status_code}", lat)
        if ok:
            try:
                data = r.json()
                reply = data.get("reply") or data.get("text") or data.get("message")
                if isinstance(reply, str) and reply.strip():
                    _add(api_chat, "chat_json", True, "Reply field present and non-empty.")
                else:
                    _add(api_chat, "chat_json", False, "JSON parsed but reply field missing/empty.")
                used = data.get("model") or data.get("provider") or data.get("backend")
                if used:
                    _add(api_chat, "chat_routing", True, f"Served by: {used!r}")
                else:
                    _add(api_chat, "chat_routing", False, "No explicit model/provider field.")
            except Exception as e:
                _add(api_chat, "chat_json", False, f"Chat JSON parse failed: {e}")
    except Exception as e:
        _add(api_chat, "chat_http", False, f"Chat request failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_api_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No API checks were run.")
        return

    target_w = max(len("TARGET"), max(len(str(c["target"])) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)
    lat_w = len("LAT(ms)")

    header = f"{'TARGET'.ljust(target_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  {'LAT(ms)'.ljust(lat_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        lat = c.get("latency_ms")
        lat_s = f"{lat:.1f}" if isinstance(lat, (int, float)) else "-"
        row = (
            f"{str(c['target']).ljust(target_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{lat_s.ljust(lat_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 5 — Hardware Diagnostics
# ---------------------------------------------------------------------------

def run_hardware_diagnostics() -> dict:
    """
    SuperTask Phase 5: Hardware Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py hardware
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("Hardware Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    # Audio devices via sounddevice / PyAudio
    try:
        try:
            import sounddevice as sd  # type: ignore
            devices = sd.query_devices()
            inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
            outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
            _add("audio", "microphones", bool(inputs), f"Input devices: {len(inputs)}")
            _add("audio", "speakers", bool(outputs), f"Output devices: {len(outputs)}")
        except Exception as e:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            ins = outs = 0
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    ins += 1
                if info.get("maxOutputChannels", 0) > 0:
                    outs += 1
            pa.terminate()
            _add("audio", "microphones", ins > 0, f"Input devices: {ins}")
            _add("audio", "speakers", outs > 0, f"Output devices: {outs}")
    except Exception as e:
        _add("audio", "devices", False, f"Audio enumeration failed: {e}")

    # Webcam via OpenCV
    try:
        import cv2  # type: ignore
        _add("opencv", "import", True, f"OpenCV {cv2.__version__}")
        try:
            cap = cv2.VideoCapture(0)
            opened = bool(cap and cap.isOpened())
            _add("webcam", "default", opened, "Opened default camera (0)." if opened else "Failed to open camera 0.")
            if cap:
                cap.release()
        except Exception as e:
            _add("webcam", "default", False, f"Camera error: {e}")
    except Exception as e:
        _add("opencv", "import", False, f"OpenCV not available: {e}")

    # USB enumeration (best-effort)
    try:
        sysname = platform.system().lower()
        if sysname == "windows":
            cmd = ["wmic", "path", "Win32_PnPEntity", "where", "PNPClass='USB'", "get", "Name"]
        elif sysname in ("linux", "darwin"):
            cmd = ["lsusb"]
        else:
            cmd = None
        if cmd:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
            lines = [ln for ln in out.splitlines() if ln.strip()]
            _add("usb", "enumeration", bool(lines), f"USB devices lines: {len(lines)}")
        else:
            _add("usb", "enumeration", False, f"Unsupported platform: {sysname}")
    except Exception as e:
        _add("usb", "enumeration", False, f"USB enumeration failed: {e}")

    # GPU (Torch / GPUtil if available)
    gpu_ok = False
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _add("gpu", "cuda", True, f"CUDA GPU: {name} ({total:.1f} GB VRAM)")
            gpu_ok = True
        else:
            _add("gpu", "cuda", False, "CUDA not available.")
    except Exception as e:
        _add("gpu", "cuda", False, f"Torch not available or CUDA check failed: {e}")

    if not gpu_ok:
        try:
            import GPUtil  # type: ignore
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                _add("gpu", "gputil", True, f"GPU via GPUtil: {g.name} ({g.memoryTotal} MB)")
            else:
                _add("gpu", "gputil", False, "No GPUs detected via GPUtil.")
        except Exception as e:
            _add("gpu", "gputil", False, f"GPUtil not available or failed: {e}")

    # CPU info
    try:
        # Best-effort, cross-platform CPU details (never break diagnostics)
        cpu_name = platform.processor() or ""
        arch = platform.machine() or ""

        physical = None
        logical = None
        freq_cur = None
        freq_max = None
        cache_desc = None
        extra = {}

        # psutil provides the most portable core/frequency data
        try:
            import psutil  # type: ignore
            try:
                physical = psutil.cpu_count(logical=False)
                logical = psutil.cpu_count(logical=True)
            except Exception:
                pass
            try:
                f = psutil.cpu_freq()
                if f:
                    freq_cur = getattr(f, "current", None)
                    freq_max = getattr(f, "max", None)
            except Exception:
                pass
        except Exception:
            pass

        # cpuinfo (if installed) gives the best brand/model string
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info() or {}
            brand = info.get("brand_raw") or info.get("brand")
            if brand:
                cpu_name = str(brand)
            hz_advertised = info.get("hz_advertised_friendly")
            if hz_advertised:
                extra["hz_advertised"] = hz_advertised
            # Some platforms expose cache sizes
            cache_l3 = info.get("l3_cache_size")
            cache_l2 = info.get("l2_cache_size")
            cache_l1d = info.get("l1_data_cache_size")
            cache_l1i = info.get("l1_instruction_cache_size")
            parts = []
            if cache_l1d:
                parts.append(f"L1d={cache_l1d}")
            if cache_l1i:
                parts.append(f"L1i={cache_l1i}")
            if cache_l2:
                parts.append(f"L2={cache_l2}")
            if cache_l3:
                parts.append(f"L3={cache_l3}")
            if parts:
                cache_desc = ", ".join(parts)
        except Exception:
            pass

        # Windows: get detailed CPU name / cache / sockets via wmic if available
        try:
            if (platform.system() or "").lower() == "windows":
                try:
                    out = subprocess.check_output(
                        ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize,MaxClockSpeed", "/format:list"],
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=8,
                    )
                    kv = {}
                    for ln in out.splitlines():
                        ln = ln.strip()
                        if not ln or "=" not in ln:
                            continue
                        k, v = ln.split("=", 1)
                        kv[k.strip()] = v.strip()
                    if kv.get("Name"):
                        cpu_name = kv.get("Name")
                    if kv.get("NumberOfCores") and physical is None:
                        try:
                            physical = int(kv.get("NumberOfCores"))
                        except Exception:
                            pass
                    if kv.get("NumberOfLogicalProcessors") and logical is None:
                        try:
                            logical = int(kv.get("NumberOfLogicalProcessors"))
                        except Exception:
                            pass
                    l2 = kv.get("L2CacheSize")
                    l3 = kv.get("L3CacheSize")
                    if (l2 or l3) and not cache_desc:
                        parts = []
                        if l2:
                            parts.append(f"L2={l2}KB")
                        if l3:
                            parts.append(f"L3={l3}KB")
                        cache_desc = ", ".join(parts)
                    max_mhz = kv.get("MaxClockSpeed")
                    if max_mhz and freq_max is None:
                        try:
                            freq_max = float(max_mhz)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        name_out = cpu_name if cpu_name else "<unknown>"
        detail_parts = [f"{name_out}", f"arch={arch or '<unknown>'}"]
        if physical is not None:
            detail_parts.append(f"cores={physical}")
        if logical is not None:
            detail_parts.append(f"threads={logical}")
        if freq_cur is not None:
            detail_parts.append(f"freq_cur={freq_cur:.0f}MHz")
        if freq_max is not None:
            detail_parts.append(f"freq_max={freq_max:.0f}MHz")
        if cache_desc:
            detail_parts.append(f"cache={cache_desc}")
        if extra:
            detail_parts.append(f"extra={extra}")

        _add("cpu", "info", True, "; ".join(detail_parts))
    except Exception as e:
        _add("cpu", "info", False, f"CPU info failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_hardware_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No hardware checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 6 — OS / System Diagnostics
# ---------------------------------------------------------------------------

def run_system_diagnostics() -> dict:
    """
    SuperTask Phase 6: OS / System Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py system
    """
    base_report = run_self_check() or {}
    report: dict = {"ok": False, "checks": [], "self_check": base_report}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("System Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    try:
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        }
        py_info = {
            "python_version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        }
        _add("os", "version", True, f"{os_info}")
        _add("python", "version", True, f"{py_info}")
    except Exception as e:
        _add("system", "basic", False, f"OS/Python info failed: {e}")

    # Directories from config
    for attr in ("BASE_DIR", "DATA_DIR", "DATASETS_DIR", "MODELS_DIR", "WEB_DIR"):
        if hasattr(config, attr):
            path = getattr(config, attr)
            try:
                if not path:
                    _add("dir", attr, False, "Path is empty or None.")
                elif not os.path.isdir(path):
                    _add("dir", attr, False, f"{path!r} is not a directory.")
                else:
                    test_file = os.path.join(path, ".diag_write_test.tmp")
                    try:
                        with open(test_file, "w", encoding="utf-8") as f:
                            f.write("ok")
                        os.remove(test_file)
                        _add("dir", attr, True, f"{path!r} exists and is writable.")
                    except Exception as e:
                        _add("dir", attr, False, f"{path!r} exists but not writable: {e}")
            except Exception as e:
                _add("dir", attr, False, f"Directory check failed: {e}")

    # RAM / CPU load (psutil)
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)

        # Best-effort RAM module details (type/speed) without hard dependencies.
        ram_extra = {}
        try:
            sysname = (platform.system() or "").lower()
            if sysname == "windows":
                # WMIC is available on many Windows systems (legacy but useful). Keep it best-effort.
                try:
                    out = subprocess.check_output(
                        ["wmic", "memorychip", "get", "SMBIOSMemoryType,Speed,Manufacturer,PartNumber", "/format:csv"],
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=8,
                    )
                    # Parse CSV lines, collect unique values.
                    types = set()
                    speeds = set()
                    mans = set()
                    parts = set()
                    for ln in out.splitlines():
                        ln = ln.strip()
                        if not ln or ln.lower().startswith("node,"):
                            continue
                        cols = [c.strip() for c in ln.split(",")]
                        if len(cols) < 5:
                            continue
                        # Node,Manufacturer,PartNumber,SMBIOSMemoryType,Speed (ordering can vary across OS)
                        # Try to locate by header heuristics
                        # Many exports are: Node,Manufacturer,PartNumber,SMBIOSMemoryType,Speed
                        manufacturer = cols[1] if len(cols) > 1 else ""
                        part = cols[2] if len(cols) > 2 else ""
                        mem_type = cols[3] if len(cols) > 3 else ""
                        speed = cols[4] if len(cols) > 4 else ""
                        if manufacturer:
                            mans.add(manufacturer)
                        if part:
                            parts.add(part)
                        if mem_type:
                            types.add(mem_type)
                        if speed:
                            speeds.add(speed)
                    if types:
                        ram_extra["smbios_type"] = sorted(types)
                    if speeds:
                        ram_extra["speed_mhz"] = sorted(speeds)
                    if mans:
                        ram_extra["manufacturer"] = sorted(mans)[:3]
                    if parts:
                        ram_extra["part_numbers"] = sorted(parts)[:3]
                except Exception:
                    pass
            elif sysname in ("linux", "darwin"):
                # Try dmidecode on Linux (often requires sudo; best-effort only)
                if sysname == "linux":
                    try:
                        out = subprocess.check_output(["dmidecode", "-t", "memory"], stderr=subprocess.STDOUT, text=True, timeout=8)
                        # Pull a few useful fields if present
                        speed = None
                        mtype = None
                        for ln in out.splitlines():
                            s = ln.strip()
                            if s.lower().startswith("type:") and not mtype:
                                mtype = s.split(":", 1)[1].strip()
                            if s.lower().startswith("speed:") and not speed:
                                speed = s.split(":", 1)[1].strip()
                        if mtype:
                            ram_extra["type"] = mtype
                        if speed:
                            ram_extra["speed"] = speed
                    except Exception:
                        pass
        except Exception:
            pass

        ram_detail = f"Total: {vm.total / (1024**3):.1f} GB, Available: {vm.available / (1024**3):.1f} GB, Used: {vm.percent}%"
        if ram_extra:
            ram_detail += f", Details: {ram_extra}"

        _add("resources", "ram", True, ram_detail)
        _add("resources", "cpu", True, f"CPU load: {cpu}%")
    except Exception as e:
        _add("resources", "psutil", False, f"psutil not available or failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_system_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No system checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 7 — Network Environment Diagnostics
# ---------------------------------------------------------------------------

def run_network_diagnostics() -> dict:
    """
    SuperTask Phase 7: Network Environment Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py network
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("Network Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    # DNS resolution
    for host in ("ai.sarahmemory.com", "api.sarahmemory.com"):
        try:
            ip = socket.gethostbyname(host)
            _add("dns", host, True, f"Resolved to {ip}")
        except Exception as e:
            _add("dns", host, False, f"DNS failed: {e}")

    # Internet connectivity + latency
    try:
        import requests  # type: ignore
        t0 = time.perf_counter()
        r = requests.get("https://www.google.com/generate_204", timeout=10)
        lat = (time.perf_counter() - t0) * 1000.0
        ok = r.status_code in (204, 200)
        _add("internet", "https", ok, f"Status {r.status_code}, latency ~{lat:.1f} ms")
    except Exception as e:
        _add("internet", "https", False, f"Connectivity check failed: {e}")

    # SSL cert info for api.sarahmemory.com
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(("api.sarahmemory.com", 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname="api.sarahmemory.com") as ssock:
                cert = ssock.getpeercert()
        nb = cert.get("notBefore")
        na = cert.get("notAfter")
        _add("ssl", "api.sarahmemory.com", True, f"notBefore={nb}, notAfter={na}")
    except Exception as e:
        _add("ssl", "api.sarahmemory.com", False, f"SSL cert retrieval failed: {e}")

    # Ports 80/443
    for port in (80, 443):
        try:
            with socket.create_connection(("api.sarahmemory.com", port), timeout=5):
                _add("tcp", f"api.sarahmemory.com:{port}", True, "TCP connect OK")
        except Exception as e:
            _add("tcp", f"api.sarahmemory.com:{port}", False, f"TCP connect failed: {e}")

    # Cloud API reachability
    try:
        import requests  # type: ignore
        api_url = getattr(config, "DEFAULT_API_URL", "https://api.sarahmemory.com/api/health")
        r = requests.get(api_url, timeout=15)
        ok = r.status_code == 200
        _add("cloud_api", api_url, ok, f"HTTP {r.status_code}")
    except Exception as e:
        _add("cloud_api", "DEFAULT_API_URL", False, f"Health check failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_network_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No network checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 8 — Cloud Sync Diagnostics
# ---------------------------------------------------------------------------

def run_sync_diagnostics() -> dict:
    """
    SuperTask Phase 8: Cloud Sync Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py sync
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("Sync Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    # MySQL reachability via DatabaseDiagnosticsSuperTask, if available
    try:
        diag = DatabaseDiagnosticsSuperTask()
        db_report = diag.run()
        ok = bool(db_report.get("mysql"))
        _add("mysql", "connect", ok, "Cloud MySQL connectivity tested via DatabaseDiagnosticsSuperTask.")
    except Exception as e:
        _add("mysql", "connect", False, f"MySQL diagnostics failed: {e}")

    # SFTP credentials presence (non-destructive)
    host = os.environ.get("UPLOAD_SFTP_HOST") or getattr(config, "UPLOAD_SFTP_HOST", None)
    user = os.environ.get("UPLOAD_SFTP_USER") or getattr(config, "UPLOAD_SFTP_USER", None)
    base_dir = os.environ.get("UPLOAD_BASE_DIR") or getattr(config, "UPLOAD_BASE_DIR", None)
    if host and user and base_dir:
        _add("sftp", "config", True, f"Host={host}, User={user}, BaseDir={base_dir}")
    else:
        _add("sftp", "config", False, "Missing SFTP env/config values (UPLOAD_SFTP_HOST/USER/BASE_DIR).")

    # CDN / static asset fetch
    try:
        import requests  # type: ignore
        cdn_url = getattr(config, "PUBLIC_UI_URL", "https://ai.sarahmemory.com/")
        r = requests.get(cdn_url, timeout=15)
        ok = r.status_code == 200
        _add("cdn", cdn_url, ok, f"HTTP {r.status_code}")
    except Exception as e:
        _add("cdn", "PUBLIC_UI_URL", False, f"CDN fetch failed: {e}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_sync_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No sync checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 9 — Security Diagnostics
# ---------------------------------------------------------------------------

def run_security_diagnostics() -> dict:
    """
    SuperTask Phase 9: Security Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py security
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("Security Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    # API keys present
    key_names = [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
    ]
    for name in key_names:
        val = os.environ.get(name) or getattr(config, name, None)
        _add("apikey", name, bool(val), "Present." if val else "MISSING.")

    # .env permissions (POSIX only)
    env_path = getattr(config, "BASE_DIR", ".")
    env_file = os.path.join(env_path, ".env")
    if os.path.exists(env_file):
        try:
            st = os.stat(env_file)
            if hasattr(st, "st_mode"):
                mode = st.st_mode & 0o777
                world = mode & 0o007
                group = mode & 0o070
                if world or group:
                    _add("env", "permissions", False, f".env mode is {oct(mode)} (should be 0o600 or stricter).")
                else:
                    _add("env", "permissions", True, f".env mode is {oct(mode)}")
            else:
                _add("env", "permissions", False, "No st_mode on stat result.")
        except Exception as e:
            _add("env", "permissions", False, f"Stat failed: {e}")
    else:
        _add("env", "permissions", False, ".env file not found.")

    # Unsafe localhost exposure (simple heuristic)
    host = getattr(config, "DEFAULT_FLASK_HOST", None) or getattr(config, "API_HOST", None)
    if host in ("0.0.0.0", None, ""):
        _add("network", "bind_host", False, f"Flask API binding host is {host!r}; recommended to use 127.0.0.1 behind a reverse proxy.")
    else:
        _add("network", "bind_host", True, f"Binding host is {host!r}")

    # Encryption key integrity (length / default check only)
    for name in ("SARAH_VAULT_PASSWORD", "ENCRYPTION_KEY", "FERNET_SECRET", "DB_ENCRYPTION_KEY"):
        val = os.environ.get(name) or getattr(config, name, None)
        if not val:
            _add("crypto", name, False, "Missing.")
        elif isinstance(val, str) and (len(val) < 16 or val.lower() in ("changeme", "password", "secret")):
            _add("crypto", name, False, "Too short or using an insecure default value.")
        else:
            _add("crypto", name, True, "Present and looks non-trivial.")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    # Security ok if no FAILED entries
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_security_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No security checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Phase 10 — UI / JS Diagnostics
# ---------------------------------------------------------------------------

def run_ui_diagnostics() -> dict:
    """
    SuperTask Phase 10: UI / JS Diagnostics

    Console usage:
        python SarahMemoryDiagnostics.py ui
    """
    report: dict = {"ok": False, "checks": []}

    def _add(component: str, kind: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        entry = {"component": component, "kind": kind, "status": status, "detail": detail}
        report["checks"].append(entry)
        msg = f"{component} ({kind}) => {status} — {detail}"
        try:
            log_diagnostics_event("UI Diagnostics", msg)
            if passed:
                logger.info("✅ " + msg)
            else:
                logger.error("❌ " + msg)
        except Exception:
            pass

    ui_dir = getattr(config, "WEB_DIR", None) or os.path.join(getattr(config, "BASE_DIR", "."), "data", "ui")
    index_path = os.path.join(ui_dir, "index.html")
    js_path = os.path.join(ui_dir, "app.js")
    css_path = os.path.join(ui_dir, "styles.css")

    # index.html
    if os.path.isfile(index_path):
        _add("ui", "index.html", True, f"Found at {index_path}")
    else:
        _add("ui", "index.html", False, f"index.html missing at {index_path}")

    # app.js
    if os.path.isfile(js_path):
        _add("ui", "app.js", True, f"Found at {js_path}")
        try:
            js = open(js_path, "r", encoding="utf-8", errors="ignore").read()
            if "SM_getApi" in js:
                _add("ui", "SM_getApi", True, "SM_getApi() definition found.")
            else:
                _add("ui", "SM_getApi", False, "SM_getApi() not found.")
            if "SM_pollHealth" in js:
                _add("ui", "SM_pollHealth", True, "SM_pollHealth() definition found.")
            else:
                _add("ui", "SM_pollHealth", False, "SM_pollHealth() not found.")
            if "SM_BRIDGE_BASE_AUTODETECT_V1" in js:
                _add("ui", "bridge_autodetect", True, "SM_BRIDGE_BASE_AUTODETECT_V1 marker found.")
            else:
                _add("ui", "bridge_autodetect", False, "SM_BRIDGE_BASE_AUTODETECT_V1 marker missing.")
            if "navigator.userAgent" in js or "window.innerWidth" in js:
                _add("ui", "mobile_detection", True, "Mobile / viewport detection logic present.")
            else:
                _add("ui", "mobile_detection", False, "No obvious mobile detection logic found.")
        except Exception as e:
            _add("ui", "app.js_read", False, f"Failed to read app.js: {e}")
    else:
        _add("ui", "app.js", False, f"app.js missing at {js_path}")

    # styles.css
    if os.path.isfile(css_path):
        _add("ui", "styles.css", True, f"Found at {css_path}")
        try:
            css = open(css_path, "r", encoding="utf-8", errors="ignore").read()
            if "@media" in css:
                _add("ui", "responsive_css", True, "Found @media queries (responsive CSS).")
            else:
                _add("ui", "responsive_css", False, "No @media queries detected in styles.css.")
        except Exception as e:
            _add("ui", "styles.css_read", False, f"Failed to read styles.css: {e}")
    else:
        _add("ui", "styles.css", False, f"styles.css missing at {css_path}")

    checks = [c for c in report["checks"] if c.get("kind") != "import"]
    report["ok"] = bool(checks) and all(c["status"] == "PASS" for c in checks)
    return report


def _print_ui_report(report: dict) -> None:
    checks = report.get("checks") or []
    if not checks:
        print("No UI checks were run.")
        return

    comp_w = max(len("COMP"), max(len(c["component"]) for c in checks))
    kind_w = max(len("KIND"), max(len(str(c["kind"])) for c in checks))
    status_w = max(len("STATUS"), 4)

    header = f"{'COMP'.ljust(comp_w)}  {'KIND'.ljust(kind_w)}  {'STATUS'.ljust(status_w)}  DETAIL"
    print(header)
    print("-" * len(header))
    for c in checks:
        row = (
            f"{c['component'].ljust(comp_w)}  "
            f"{str(c['kind']).ljust(kind_w)}  "
            f"{c['status'].ljust(status_w)}  "
            f"{c['detail']}"
        )
        print(row)


# ---------------------------------------------------------------------------
# Console Menu (Bonus)
# ---------------------------------------------------------------------------

def _log_full_diagnostics_summary(reports: dict) -> None:
    """
    Aggregate a full diagnostics run into a single summary line
    for diag_report.log and system_logs.db.
    """
    try:
        total_checks = 0
        total_pass = 0
        total_fail = 0
        total_other = 0
        section_bits: list[str] = []

        for name, rep in (reports or {}).items():
            checks = rep.get("checks") or []
            total_checks += len(checks)
            s_pass = sum(1 for c in checks if c.get("status") == "PASS")
            s_fail = sum(1 for c in checks if c.get("status") == "FAIL")
            s_other = len(checks) - s_pass - s_fail
            total_pass += s_pass
            total_fail += s_fail
            total_other += s_other
            section_bits.append(f"{name}: {s_pass} pass, {s_fail} fail, {s_other} other")

        overall_ok = bool(total_checks) and total_fail == 0
        summary = (
            f"Full diagnostics run — overall_ok={overall_ok} ; "
            f"checks={total_checks} ; pass={total_pass} ; fail={total_fail} ; other={total_other} ; "
            + " | ".join(section_bits)
        )
        append_diag_log("FULL", summary)
        log_diagnostics_event("Full Diagnostics Summary", summary)
    except Exception as e:
        logger.error(f"Failed to summarize full diagnostics: {e}")


def run_full_diagnostics_suite(write_aggregate_log: bool = False) -> dict:
    """
    Run the FULL diagnostics suite (Phases 1–10).
    When write_aggregate_log is True, also write an aggregate
    summary line to diag_report.log and system_logs.db.
    """
    logger.info("Running FULL diagnostics suite (Phases 1–10)...")
    print("=== FULL SarahMemory Diagnostics (Phases 1–10) ===")

    reports: dict = {}

    # Phase 1 + 2 + 3: WebUI bridge + WebUI network + base self-check + DB
    w_report = run_webui_bridge_diagnostics()
    _print_webui_bridge_report(w_report)
    reports["webui_bridge"] = w_report

    wn_report = run_webui_network_diagnostics()
    _print_webui_network_report(wn_report)
    reports["webui_network"] = wn_report

    base = run_self_check()
    reports["self_check"] = base

    diag = DatabaseDiagnosticsSuperTask()
    db_report = diag.run()
    _print_database_report(db_report)
    reports["database"] = db_report

    # Phase 4: API
    api_report = run_api_diagnostics()
    _print_api_report(api_report)
    reports["api"] = api_report

    # Phase 5: Hardware
    hw_report = run_hardware_diagnostics()
    _print_hardware_report(hw_report)
    reports["hardware"] = hw_report

    # Phase 6: System/OS
    sys_report = run_system_diagnostics()
    _print_system_report(sys_report)
    reports["system"] = sys_report

    # Phase 7: Network
    net_report = run_network_diagnostics()
    _print_network_report(net_report)
    reports["network"] = net_report

    # Phase 8: Cloud Sync
    sync_report = run_sync_diagnostics()
    _print_sync_report(sync_report)
    reports["sync"] = sync_report

    # Phase 9: Security
    sec_report = run_security_diagnostics()
    _print_security_report(sec_report)
    reports["security"] = sec_report

    # Phase 10: UI
    ui_report = run_ui_diagnostics()
    _print_ui_report(ui_report)
    reports["ui"] = ui_report

    logger.info("Full diagnostics suite completed.")

    if write_aggregate_log:
        _log_full_diagnostics_summary(reports)

    return reports

def diagnostics_menu() -> None:
    """
    Text-based diagnostics UI.

    Invoked when running:
        python SarahMemoryDiagnostics.py
    with no arguments.
    """
    while True:
        print("\n===========================")
        print(" SarahMemory Diagnostics")
        print("===========================")
        print("1) Run Full System Diagnostics")
        print("2) Database Diagnostics")
        print("3) UI Diagnostics")
        print("4) API Diagnostics")
        print("5) Network Diagnostics")
        print("6) Hardware Diagnostics")
        print("7) Cloud Sync Diagnostics")
        print("8) Security Diagnostics")
        print("9) Run & Log FULL Diagnostics")
        print("0) Exit")
        choice = input("Select option: ").strip()
        if choice == "0":
            print("Exiting diagnostics.")
            break
        elif choice == "1":
            print("\n[Full] Running core self-check + system + network + API...")
            base = run_self_check()
            _ = base  # unused variable in this summary
            sys_report = run_system_diagnostics()
            net_report = run_network_diagnostics()
            api_report = run_api_diagnostics()
            _print_system_report(sys_report)
            _print_network_report(net_report)
            _print_api_report(api_report)
        elif choice == "2":
            print("\n[DB] Running database diagnostics...")
            diag = DatabaseDiagnosticsSuperTask()
            db_report = diag.run()
            _print_database_report(db_report)
        elif choice == "3":
            print("\n[UI] Running WebUI bridge + UI/JS diagnostics...")
            w_report = run_webui_bridge_diagnostics()
            _print_webui_bridge_report(w_report)
            ui_report = run_ui_diagnostics()
            _print_ui_report(ui_report)
        elif choice == "4":
            print("\n[API] Running API diagnostics...")
            api_report = run_api_diagnostics()
            _print_api_report(api_report)
        elif choice == "5":
            print("\n[NET] Running network diagnostics...")
            net_report = run_network_diagnostics()
            _print_network_report(net_report)
        elif choice == "6":
            print("\n[HW] Running hardware diagnostics...")
            hw_report = run_hardware_diagnostics()
            _print_hardware_report(hw_report)
        elif choice == "7":
            print("\n[SYNC] Running cloud sync diagnostics...")
            sync_report = run_sync_diagnostics()
            _print_sync_report(sync_report)
        elif choice == "8":
            print("\n[SEC] Running security diagnostics...")
            sec_report = run_security_diagnostics()
            _print_security_report(sec_report)
        elif choice == "9":
            print("\n[FULL+LOG] Running full diagnostics and logging summary...")
            _ = run_full_diagnostics_suite(write_aggregate_log=True)
        else:
            print("Invalid choice, please select 0-9.")

# [PATCH v7.7.2] Embedding pipeline diagnostic
def run_embedding_pipeline_check(sample_text: str = "What is your name?") -> dict:
    report = {"ok": False, "steps": []}
    try:
        from SarahMemoryDatabase import embed_and_store_dataset_sentences, vector_search
        report["steps"].append("Loaded DB embedding/search functions.")
        try:
            embed_and_store_dataset_sentences()
            report["steps"].append("Called embed_and_store_dataset_sentences()")
        except Exception as e:
            report["steps"].append(f"Embedding call failed or skipped: {e}")
        try:
            results = vector_search(sample_text, top_n=3) or []
            report["steps"].append(f"vector_search returned {len(results)} candidates.")
            report["ok"] = bool(results)
        except Exception as e:
            report["steps"].append(f"vector_search failed: {e}")
    except Exception as e:
        report["steps"].append(f"DB import failed: {e}")
    logger.info("[EMBED DIAG] " + " | ".join(map(str, report["steps"])))
    return report



if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        cmd = (args[0] or "").lower()

        if cmd in ("webui", "webui-diag", "webui-diagnostics"):
            logger.info("Running WebUI bridge diagnostics (SuperTask Phase 1)...")
            print("=== Web UI Bridge Diagnostics (app.py + app.js) ===")
            report = run_webui_bridge_diagnostics()
            _print_webui_bridge_report(report)
            if report.get("ok"):
                logger.info("WebUI bridge diagnostics completed: OK")
            else:
                logger.warning("WebUI bridge diagnostics completed with issues.")

        elif cmd in ("webui2", "webui-net", "webui-network"):
            logger.info("Running WebUI network/HTTPS diagnostics (SuperTask Phase 2)...")
            print("=== Web UI Network Diagnostics (browser → GoogieHost → PythonAnywhere) ===")
            report = run_webui_network_diagnostics()
            _print_webui_network_report(report)
            if report.get("ok"):
                logger.info("WebUI network diagnostics completed: OK")
            else:
                logger.warning("WebUI network diagnostics completed with issues.")

        elif cmd in ("database", "db", "db-diag", "db-diagnostics"):
            logger.info("Running Database connectivity diagnostics (SuperTask Phase 3)...")
            print("=== Database Connectivity Diagnostics (SQLite + Cloud MySQL) ===")
            diag = DatabaseDiagnosticsSuperTask()
            report = diag.run()
            _print_database_report(report)
            if report.get("ok"):
                logger.info("Database diagnostics completed: OK")
            else:
                logger.warning("Database diagnostics completed with issues.")

        elif cmd in ("api", "api-diag", "api-diagnostics"):
            logger.info("Running API diagnostics (SuperTask Phase 4)...")
            print("=== API Diagnostics (/api/health + /api/chat) ===")
            report = run_api_diagnostics()
            _print_api_report(report)
            if report.get("ok"):
                logger.info("API diagnostics completed: OK")
            else:
                logger.warning("API diagnostics completed with issues.")

        elif cmd in ("hardware", "hw", "hardware-diag", "hardware-diagnostics"):
            logger.info("Running hardware diagnostics (SuperTask Phase 5)...")
            print("=== Hardware Diagnostics (audio, webcam, GPU, USB) ===")
            report = run_hardware_diagnostics()
            _print_hardware_report(report)
            if report.get("ok"):
                logger.info("Hardware diagnostics completed: OK")
            else:
                logger.warning("Hardware diagnostics completed with issues.")

        elif cmd in ("system", "sys", "os", "system-diag", "system-diagnostics"):
            logger.info("Running system diagnostics (SuperTask Phase 6)...")
            print("=== System Diagnostics (OS/Python/resources) ===")
            report = run_system_diagnostics()
            _print_system_report(report)
            if report.get("ok"):
                logger.info("System diagnostics completed: OK")
            else:
                logger.warning("System diagnostics completed with issues.")

        elif cmd in ("network", "net", "network-diag", "network-diagnostics"):
            logger.info("Running network environment diagnostics (SuperTask Phase 7)...")
            print("=== Network Diagnostics (DNS, connectivity, SSL, ports) ===")
            report = run_network_diagnostics()
            _print_network_report(report)
            if report.get("ok"):
                logger.info("Network diagnostics completed: OK")
            else:
                logger.warning("Network diagnostics completed with issues.")

        elif cmd in ("sync", "cloud", "cloud-sync", "sync-diag", "sync-diagnostics"):
            logger.info("Running cloud sync diagnostics (SuperTask Phase 8)...")
            print("=== Cloud Sync Diagnostics (MySQL, SFTP config, CDN) ===")
            report = run_sync_diagnostics()
            _print_sync_report(report)
            if report.get("ok"):
                logger.info("Sync diagnostics completed: OK")
            else:
                logger.warning("Sync diagnostics completed with issues.")

        elif cmd in ("security", "sec", "security-diag", "security-diagnostics"):
            logger.info("Running security diagnostics (SuperTask Phase 9)...")
            print("=== Security Diagnostics (API keys, .env, bind host, crypto keys) ===")
            report = run_security_diagnostics()
            _print_security_report(report)
            if report.get("ok"):
                logger.info("Security diagnostics completed: OK")
            else:
                logger.warning("Security diagnostics completed with issues.")

        elif cmd in ("ui", "ui-diag", "ui-diagnostics"):
            logger.info("Running UI/JS diagnostics (SuperTask Phase 10)...")
            print("=== UI / JS Diagnostics (index.html, app.js, CSS) ===")
            w_report = run_webui_bridge_diagnostics()
            _print_webui_bridge_report(w_report)
            ui_report = run_ui_diagnostics()
            _print_ui_report(ui_report)
            if w_report.get("ok") and ui_report.get("ok"):
                logger.info("UI diagnostics completed: OK")
            else:
                logger.warning("UI diagnostics completed with issues.")

        elif cmd in ("full", "all", "full-diag", "full-diagnostics"):
            _ = run_full_diagnostics_suite(write_aggregate_log=True)

        elif cmd in ("phase-b", "phaseb", "context-mesh", "core-b"):
            logger.info("Running Phase B context/mesh/agent diagnostics...")
            print("=== Phase B Diagnostics (Context / Mesh / Agent) ===")
            report = run_context_mesh_agent_diagnostics()
            _print_context_mesh_agent_report(report)
            if report.get("ok"):
                logger.info("Phase B diagnostics completed: OK")
            else:
                logger.warning("Phase B diagnostics completed with issues.")

        else:
            print(f"Unknown diagnostics command: {cmd!r}")
            print("Run without arguments to open the interactive diagnostics menu.")
    else:
        # No arguments => show interactive diagnostics menu
        diagnostics_menu()
# ====================================================================
# END OF SarahMemoryDiagnostics.py v8.0.0
# ====================================================================