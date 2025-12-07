"""--==The SarahMemory Project==--
File: SarahMemoryIntegration.py
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
SarahMemory v8.0 - Integration & Main Menu System
Integration with Enhanced Features
===============================================================================
"""

import logging
import os
import sys
import time
import threading
import asyncio
import hashlib
from ftplib import FTP, error_temp, error_perm, error_proto, all_errors
from tqdm import tqdm

from SarahMemoryGUI import run_gui
from SarahMemoryVoice import synthesize_voice, shutdown_tts
from SarahMemoryDiagnostics import run_self_check
import SarahMemoryGlobals as config

# =============================================================================
# CONTEXT BUFFER INITIALIZATION
# =============================================================================
if config.ENABLE_CONTEXT_BUFFER:
    import SarahMemoryAiFunctions as context
    try:
        context.init_context_history()
    except Exception:
        pass

# =============================================================================
# LOGGER SETUP - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger("SarahMemoryIntegration")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - v8.0 - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# =============================================================================
# GLOBAL STATE
# =============================================================================
terminate_flag = threading.Event()


# =============================================================================
# v8.0 BOOTSTRAP STARTUP SEQUENCE
# =============================================================================
def bootstrap_startup():
    """
    v8.0 Enhanced: Run updater, dataset sync, and kick off vector warmup before menu.
    Includes better error handling and progress reporting.
    """
    logger.info("[v8.0][BOOTSTRAP] Starting bootstrap sequence...")
    
    try:
        # =====================================================================
        # REMOTE DATASET SYNC
        # =====================================================================
        try:
            host = getattr(config, "FTP_HOST", "")
            user = getattr(config, "FTP_USER", "")
            pw = getattr(config, "FTP_PASS", "")
            remote = getattr(config, "FTP_REMOTE_DATASETS_DIR", 
                           "/public_html/api/data/memory/datasets")
            local = getattr(config, "DATASETS_DIR", 
                          os.path.join(os.getcwd(), "data", "memory", "datasets"))
            
            if host and user and pw:
                try:
                    logger.info("[v8.0][BOOTSTRAP] Attempting remote dataset sync...")
                    sync_dataset_bidirectional(local, host, user, pw, remote)
                    logger.info("[v8.0][BOOTSTRAP] Remote dataset sync completed")
                except Exception as se:
                    logger.warning(f"[v8.0][BOOTSTRAP] Dataset sync skipped: {se}")
        except Exception as e:
            logger.debug(f"[v8.0][BOOTSTRAP] No FTP sync: {e}")
        
        # =====================================================================
        # VECTOR WARMUP
        # =====================================================================
        try:
            import SarahMemoryResearch as research
            if getattr(config, "LOCAL_DATA_ENABLED", True):
                logger.info("[v8.0][BOOTSTRAP] Warming up vector search...")
                _ = research.get_research_data("warmup")
                logger.info("[v8.0][BOOTSTRAP] Vector warmup completed")
        except Exception as ve:
            logger.debug(f"[v8.0][BOOTSTRAP] Vector warmup skipped: {ve}")
    
    except Exception as e:
        logger.warning(f"[v8.0][BOOTSTRAP] bootstrap_startup error: {e}")


# =============================================================================
# FILE HASH UTILITIES
# =============================================================================
def hash_file(filepath):
    """
    v8.0: Compute SHA256 hash of a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        str: Hexadecimal hash digest
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)
    return sha256.hexdigest()


def upload_with_progress(ftp, filepath, filename):
    """
    v8.0: Upload file to FTP server with progress bar.
    
    Args:
        ftp: FTP connection object
        filepath: Local file path
        filename: Remote filename
    """
    filesize = os.path.getsize(filepath)
    bar = tqdm(total=filesize, unit='B', unit_scale=True, 
               desc=f"[v8.0] Uploading {filename}", ncols=80)
    start = time.time()
    
    try:
        with open(filepath, 'rb') as f:
            def callback(chunk):
                bar.update(len(chunk))
            ftp.storbinary(f"STOR {filename}", f, 1024, callback=callback)
    except Exception as e:
        bar.close()
        logger.error(f"[v8.0][UPLOAD ERROR] {filename} failed: {type(e).__name__}: {e}")
        return
    
    bar.close()
    elapsed = time.time() - start
    logger.info(f"[v8.0][UPLOAD COMPLETE] {filename} ({filesize} bytes) in {elapsed:.2f} sec")


# =============================================================================
# BIDIRECTIONAL DATASET SYNCHRONIZATION
# =============================================================================
def sync_dataset_bidirectional():
    """
    v8.0 Enhanced: Bidirectional dataset sync with FTP server.
    
    Features:
    - Respects SAFE_MODE / LOCAL_ONLY and 'is_offline()' if present
    - Uses FTP settings from SarahMemoryGlobals.py
    - Skips sync when FTP_BACKUP_SCHEDULE is "never" or not due yet
    - Compares hashes and uploads/downloads changed files only
    - Writes a .last_ftp_backup.txt stamp on success
    """
    logger.info("[v8.0][SYNC] Starting bidirectional dataset synchronization...")
    
    # =========================================================================
    # SAFETY CHECKS
    # =========================================================================
    offline = False
    safe_mode = False
    
    try:
        if hasattr(config, "is_offline"):
            offline = config.is_offline()
    except Exception:
        offline = False
    
    try:
        safe_mode = getattr(config, "SAFE_MODE", False) or getattr(config, "LOCAL_ONLY_MODE", False)
    except Exception:
        safe_mode = False
    
    if safe_mode or offline:
        logger.info("[v8.0][SYNC] Skipping dataset sync due to safe mode or offline status.")
        return

    # =========================================================================
    # PATH AND CONFIG SETUP
    # =========================================================================
    data_dir = getattr(config, "DATA_DIR", os.path.join(os.getcwd(), "data"))
    local_dir = getattr(config, "DATASETS_DIR", 
                       os.path.join(os.getcwd(), "data", "memory", "datasets"))
    os.makedirs(local_dir, exist_ok=True)

    ftp_host = getattr(config, "FTP_HOST", "ftp.sarahmemory.com")
    ftp_port = int(getattr(config, "FTP_PORT", 21))
    ftp_user = getattr(config, "FTP_USER", "anonymous")
    ftp_pass = getattr(config, "FTP_PASS", "")
    remote_dir = (getattr(config, "FTP_REMOTE_DIR", None) or 
                 getattr(config, "FTP_REMOTE_DATASETS_DIR", 
                        "/public_html/api/data/memory/datasets"))

    # =========================================================================
    # SCHEDULE GATE
    # =========================================================================
    schedule_kind = getattr(config, "FTP_BACKUP_SCHEDULE", "weekly")
    
    try:
        from datetime import datetime, timedelta
        days = getattr(config, "schedule_to_days", lambda k: 7)(schedule_kind)
        stamp_file = os.path.join(local_dir, ".last_ftp_backup.txt")
        
        if days == 0:  # "never"
            logger.info(f"[v8.0][SYNC] FTP backup skipped (policy: {schedule_kind}).")
            return
        
        should_run = True
        try:
            with open(stamp_file, "r", encoding="utf-8") as sf:
                last = datetime.fromisoformat(sf.read().strip())
            should_run = (datetime.now() - last).days >= days
        except Exception:
            should_run = True
        
        if not should_run:
            logger.info(f"[v8.0][SYNC] FTP backup skipped (next run not due; policy: {schedule_kind}).")
            return
    
    except Exception as _sched_e:
        logger.warning(f"[v8.0][SYNC] Schedule gate error (continuing): {_sched_e}")

    # =========================================================================
    # FTP SYNC
    # =========================================================================
    try:
        logger.info(f"[v8.0][SYNC] Connecting to FTP server at {ftp_host}:{ftp_port}...")
        ftp = FTP(timeout=45)
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_user, ftp_pass)
        
        # Change to remote directory (create if needed)
        try:
            ftp.cwd(remote_dir)
        except error_perm:
            # Try to create the remote path if missing
            parts = [p for p in remote_dir.split("/") if p]
            cur = ""
            for p in parts:
                cur += "/" + p
                try:
                    ftp.cwd(cur)
                except error_perm:
                    try:
                        ftp.mkd(cur)
                        ftp.cwd(cur)
                    except error_perm:
                        pass
            ftp.cwd(remote_dir)

        # Hash helper
        def _hash_file(fp):
            import hashlib
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for b in iter(lambda: f.read(65536), b""):
                    h.update(b)
            return h.hexdigest()

        # Collect file lists
        try:
            remote_list = ftp.nlst()
        except Exception:
            remote_list = []
        
        local_list = [f for f in os.listdir(local_dir) 
                     if os.path.isfile(os.path.join(local_dir, f))]
        combined = sorted(set(remote_list + local_list))

        tmp_dl = os.path.join(local_dir, ".tmp_download")
        if os.path.exists(tmp_dl):
            try:
                os.remove(tmp_dl)
            except Exception:
                pass

        # Sync each file
        for fname in combined:
            if fname in [".", "..", ".ftpquota", ".htaccess", ".last_ftp_backup.txt"]:
                continue
            
            lp = os.path.join(local_dir, fname)
            local_exists = os.path.isfile(lp)
            remote_exists = fname in remote_list

            local_hash = _hash_file(lp) if local_exists else None
            remote_hash = None
            
            if remote_exists:
                try:
                    with open(tmp_dl, "wb") as f:
                        ftp.retrbinary(f"RETR {fname}", f.write)
                    remote_hash = _hash_file(tmp_dl)
                except error_perm as ep:
                    logger.warning(f"[v8.0][FTP RETR SKIP] {fname}: {ep}")
                    remote_hash = None

            # Compare and sync
            if local_hash != remote_hash:
                if not local_exists and remote_exists:
                    logger.info(f"[v8.0][DOWNLOAD] {fname} missing locally. Downloading...")
                    try:
                        with open(lp, "wb") as wf, open(tmp_dl, "rb") as rf:
                            wf.write(rf.read())
                    except Exception as e:
                        logger.error(f"[v8.0][DOWNLOAD ERROR] {fname}: {e}")
                
                elif local_exists:
                    logger.info(f"[v8.0][UPLOAD] Updating remote: {fname}")
                    size = os.path.getsize(lp)
                    bar = tqdm(total=size, unit='B', unit_scale=True, 
                              desc=f"[v8.0] Uploading {fname}", ncols=80)
                    try:
                        with open(lp, "rb") as rf:
                            def _cb(chunk):
                                bar.update(len(chunk))
                            ftp.storbinary(f"STOR {fname}", rf, 1024, callback=_cb)
                    except Exception as e:
                        logger.error(f"[v8.0][UPLOAD ERROR] {fname}: {e}")
                    finally:
                        bar.close()
                else:
                    logger.info(f"[v8.0][SYNC] Skipping {fname} (no local or remote?)")
            else:
                logger.info(f"[v8.0][MATCH] {fname} already synced.")

            # Cleanup temp
            try:
                if os.path.exists(tmp_dl):
                    os.remove(tmp_dl)
            except Exception:
                pass

        ftp.quit()
        
        # Write success stamp
        try:
            from datetime import datetime
            with open(os.path.join(local_dir, ".last_ftp_backup.txt"), "w", encoding="utf-8") as sf:
                sf.write(datetime.now().isoformat())
        except Exception as _e:
            logger.warning(f"[v8.0][SYNC] Could not write last backup stamp: {_e}")
        
        logger.info("[v8.0][SYNC COMPLETE] Bi-directional dataset sync finished.")

    except all_errors as e:
        logger.warning(f"[v8.0][SYNC ERROR] Dataset sync failed: {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"[v8.0][SYNC ERROR] {type(e).__name__}: {e}")


# =============================================================================
# LOOP DETECTION
# =============================================================================
def detect_loop(response):
    """
    v8.0: Detect if the AI is repeating the same response (loop detection).
    
    Args:
        response: The response to check
    
    Returns:
        bool: True if loop detected, False otherwise
    """
    if not config.ENABLE_CONTEXT_BUFFER:
        return False
    
    recent_responses = [entry.get('final_response', '') for entry in context.get_context()]
    count = recent_responses.count(response)
    threshold = config.LOOP_DETECTION_THRESHOLD + (len(recent_responses) // 10)
    
    return count >= threshold


# =============================================================================
# VOICE CHAT THREAD
# =============================================================================
def run_voice_chat():
    """
    v8.0 Enhanced: Voice chat loop with ambient noise calibration and context awareness.
    """
    try:
        logger.info("[v8.0] Starting voice chat thread with ambient noise calibration...")
        time.sleep(1.5)
        
        while not terminate_flag.is_set():
            logger.info("[v8.0] Listening for voice input...")
            result = context.get_voice_input()
            
            if result is None or result == "":
                logger.warning("[v8.0] No speech detected or not understood. Retrying...")
                continue
            
            logger.info(f"[v8.0] Voice input recognized: {result}")
            
            # Classify intent
            intent = context.classify_intent(result)
            logger.info(f"[v8.0] Intent classified as: {intent}")
            
            # Get personality response
            personality_response = context.integrate_with_personality(result)
            logger.info(f"[v8.0] Personality response: {personality_response}")
            
            final_response = personality_response
            
            # Loop detection
            if detect_loop(final_response):
                logger.warning("[v8.0] Loop detected. Modifying response.")
                final_response += " (Additional details available on request.)"
            
            # Add to context
            if config.ENABLE_CONTEXT_BUFFER:
                context.add_to_context({
                    "user_input": result,
                    "intent": intent,
                    "final_response": final_response,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            
            # Speak response
            synthesize_voice(final_response)
    
    except Exception as e:
        logger.error(f"[v8.0] Voice chat error: {e}")
    finally:
        logger.info("[v8.0] Voice chat loop terminated.")


# =============================================================================
# GUI LAUNCHER
# =============================================================================
def launch_gui():
    """
    v8.0 Enhanced: Launch the main GUI with voice chat integration.
    """
    try:
        logger.info("[v8.0] Launching main GUI...")
        synthesize_voice("Loading Main GUI interface, Please Wait.")
        
        voice_thread = threading.Thread(target=run_voice_chat)
        voice_thread.start()
        
        run_gui()
        
        terminate_flag.set()
        voice_thread.join()
        
        logger.info("[v8.0] GUI closed; returning to integration menu.")
    
    except Exception as e:
        logger.error(f"[v8.0] GUI Launch Error: {e}")


# =============================================================================
# SHUTDOWN SEQUENCE
# =============================================================================
def shutdown_sequence():
    """
    v8.0 Enhanced: Clean shutdown with voice confirmation.
    """
    synthesize_voice("Shutting down. Have a great day!")
    logger.info("[v8.0] Initiating safe shutdown procedures.")
    
    print("\n" + "═" * 78)
    print("  SARAHMEMORY v8.0 - SHUTTING DOWN")
    print("═" * 78)
    
    terminate_flag.set()
    
    logger.info("[v8.0] Safe shutdown completed successfully.")
    shutdown_tts()
    
    print("\n  ✓ Shutdown complete. Thank you for using SarahMemory!")
    print("  ✓ Visit https://www.sarahmemory.com for updates\n")
    
    sys.exit(0)


# =============================================================================
# MAIN MENU - v8.0 World-Class
# =============================================================================
def main_menu():
    """
    v8.0 Enhanced: Integration main menu with optional bypass.
    
    Respects config.SM_INT_MAIN_MENU:
      - True  => show menu as normal
      - False => bypass menu, announce via TTS, auto-launch GUI, then shut down on close
    
    The two synthesized voice lines should run in both modes.
    """
    # Normalize flag to boolean
    try:
        flag_raw = getattr(config, "SM_INT_MAIN_MENU", "True")
        flag_str = str(flag_raw).strip().lower()
        show_menu = flag_str in ("true", "1", "yes", "y", "on")
    except Exception:
        show_menu = True

    while not terminate_flag.is_set():
        if show_menu:
            # =================================================================
            # STANDARD INTERACTIVE MENU PATH
            # =================================================================
            print("\n" + "═" * 78)
            print("  SARAHMEMORY v8.0 - INTEGRATION MENU")
            print("═" * 78)
            
            try:
                synthesize_voice("...,Main Menu,....")
            except Exception:
                logger.debug("[v8.0][TTS] Main Menu prompt failed silently.")
            
            print("\n  1. Launch Main AI-Bot Text/Voice GUI")
            print("  2. Safe Shutdown and Exit")
            print("\n" + "═" * 78)

            choice = input("\n[v8.0] Enter your choice (1-2): ").strip()
            
            if choice == "1":
                try:
                    synthesize_voice("Now Loading GUI interface, Please Wait")
                except Exception:
                    logger.debug("[v8.0][TTS] Loading GUI prompt failed silently.")
                
                print("\n[v8.0] Launching Chat GUI...")
                
                try:
                    import SarahMemoryGUI as gui
                    gui.run_gui()
                except Exception as e:
                    logger.error(f"[v8.0] GUI exited with error: {e}")
                finally:
                    logger.info("[v8.0] Returning to integration menu.")
            
            elif choice == "2":
                logger.info("[v8.0] Initiating safe shutdown and exit.")
                shutdown_sequence()
            
            else:
                try:
                    synthesize_voice("Invalid Choice., try again")
                except Exception:
                    logger.debug("[v8.0][TTS] Invalid choice prompt failed silently.")
                
                print("\n[v8.0] ✗ Invalid choice. Please select a valid option (1-2).")
        
        else:
            # =================================================================
            # BYPASS MENU PATH
            # =================================================================
            try:
                synthesize_voice("Now Loading GUI interface, Please Wait")
            except Exception:
                logger.debug("[v8.0][TTS] Loading GUI prompt (bypass mode) failed silently.")
            
            print("\n[v8.0] Launching Chat GUI (bypass mode)...")
            
            try:
                import SarahMemoryGUI as gui
                gui.run_gui()
            except Exception as e:
                logger.error(f"[v8.0] GUI exited with error: {e}")
            finally:
                logger.info("[v8.0] GUI closed (bypass mode). Proceeding to shutdown.")
            
            # In bypass mode, shut down immediately after GUI closes
            logger.info("[v8.0] Initiating safe shutdown (bypass mode).")
            shutdown_sequence()
            break


# =============================================================================
# BACKWARD-COMPATIBILITY WRAPPER
# =============================================================================
def integration_menu():
    """
    v8.0: Backward-compatible wrapper. Delegates to main_menu().
    """
    return main_menu()


# =============================================================================
# ASYNC SELF-CHECK
# =============================================================================
async def run_self_check_async():
    """
    v8.0: Asynchronous self-check runner.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_self_check)


# =============================================================================
# API WAIT HELPER
# =============================================================================
def _sm_wait_for_api_ready(url="http://127.0.0.1:8765/health", retries=30, delay=0.5):
    """
    v8.0: Wait for the API server to be ready.
    
    Args:
        url: Health check endpoint
        retries: Maximum number of retries
        delay: Delay between retries in seconds
    
    Returns:
        bool: True if API is ready, False otherwise
    """
    try:
        import time
        import requests
        
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=0.5)
                if r.ok:
                    try:
                        logger_inst = globals().get("logger", None)
                        if logger_inst:
                            logger_inst.info("[v8.0][BOOT] Local API server is ready.")
                    except Exception:
                        pass
                    return True
            except Exception:
                time.sleep(delay)
        
        try:
            logger_inst = globals().get("logger", None)
            if logger_inst:
                logger_inst.warning("[v8.0][TIMEOUT] Local API server did not respond in time.")
        except Exception:
            pass
    
    except Exception:
        pass
    
    return False


# =============================================================================
# DATABASE SCHEMA VALIDATION
# =============================================================================
def _ensure_response_table(db_path=None):
    """
    v8.0: Ensure the response table exists in the database.
    """
    try:
        import sqlite3
        import logging
        
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config:
                pass
        
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            datasets_dir = getattr(config, "DATASETS_DIR", 
                                 os.path.join(base, "data", "memory", "datasets"))
            db_path = os.path.join(datasets_dir, "system_logs.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        con.commit()
        con.close()
        
        logging.debug("[v8.0][DB] Ensured table `response` in %s", db_path)
    
    except Exception as e:
        try:
            import logging
            logging.warning("[v8.0][DB] Ensure `response` failed: %s", e)
        except Exception:
            pass


# Initialize response table
try:
    _ensure_response_table()
except Exception:
    pass


# =============================================================================
# MAIN EXECUTION (when run directly)
# =============================================================================
if __name__ == "__main__":
    logger.info("[v8.0] Starting SarahMemory AI Bot.")
    run_self_check()
    sync_dataset_bidirectional()
    main_menu()

# =============================================================================
# END OF SarahMemoryIntegration.py v8.0.0
# =============================================================================
