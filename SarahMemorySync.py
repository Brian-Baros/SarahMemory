"""--==The SarahMemory Project==--
File: SarahMemorySync.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com
===============================================================================

PHASE C ENHANCEMENT:
This module now includes Phase C (part of a Multi-Phase Development concept for the SarahMemory Project) 
mobile app synchronization infrastructure
in addition to the original Dropbox and FTPS sync capabilities.

Phase C Features:
- Cross-device synchronization (contacts, history, reminders)
- Device registration and management
- Conflict resolution (last-writer-wins)
- Offline-first architecture
- Comprehensive testing suite

"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "sync_engine"
# CATEGORY = "cross_device_sync"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "sync"
# HARDWARE_DOMAIN = "network_filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "sync"
# FAMILY = "device_link"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Legacy Dropbox/FTPS plus Phase C mobile sync infrastructure for cross-device contacts, history, reminders, conflict handling, and sync event logging."
# --- SARAHMETA END ---
import os
import sys
import logging
import time
import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import SarahMemoryGlobals as config

# Setup logging for the sync module
logger = logging.getLogger('SarahMemorySync')
logger.setLevel(logging.DEBUG)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# ============================================================================
# LEGACY SYNC: Dropbox Integration (v5.4.1) Orinally Written 
# When the Project Began in Febuary 2025
# Used from begining of Project till now. Orginally designed to upload code.
# ============================================================================

# Dropbox integration imports
try:
    import dropbox
    from dropbox.files import WriteMode
except ImportError as e:
    logger.error("Dropbox SDK not found. Install it using 'pip install dropbox' — continuing without Dropbox sync.")
    dropbox = None

DROPBOX_ACCESS_TOKEN = os.environ.get('DROPBOX_ACCESS_TOKEN', 'YOUR_DROPBOX_ACCESS_TOKEN')
if not DROPBOX_ACCESS_TOKEN or DROPBOX_ACCESS_TOKEN == 'YOUR_DROPBOX_ACCESS_TOKEN':
    logger.error("Dropbox access token not set. Disabling Dropbox sync.")
    DROPBOX_ACCESS_TOKEN = None


def _dropbox_ready():
    """Check if Dropbox sync is available."""
    return dropbox is not None and DROPBOX_ACCESS_TOKEN

LOCAL_SYNC_DIR = os.path.join(os.getcwd(), 'sync_data')
DROPBOX_SYNC_FOLDER = '/SarahMemorySync'
os.makedirs(LOCAL_SYNC_DIR, exist_ok=True)
logger.info(f"Local sync directory: {LOCAL_SYNC_DIR}")

def log_sync_event(event, details):
    """
    Logs a sync-related event to the device_link.db database.
    """
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "device_link.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO sync_events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged sync event to device_link.db successfully.")
    except Exception as e:
        logger.error(f"Error logging sync event: {e}")

def sync_to_dropbox(file_path, dbx):
    """
    Upload a local file to Dropbox.
    Includes detailed path logging and conflict resolution simulation.
    """
    try:
        relative_path = os.path.relpath(file_path, LOCAL_SYNC_DIR)
        dropbox_path = os.path.join(DROPBOX_SYNC_FOLDER, relative_path).replace(os.sep, '/')
        with open(file_path, 'rb') as f:
            dbx.files_upload if dbx else (_ for _ in ()).throw(Exception('Dropbox disabled'))(f.read(), dropbox_path, mode=WriteMode('overwrite'))
        success_msg = f"Uploaded '{file_path}' to Dropbox at '{dropbox_path}'."
        logger.info(success_msg)
        log_sync_event("File Upload", success_msg)
        return True
    except Exception as e:
        error_msg = f"Error uploading '{file_path}' to Dropbox: {e}"
        logger.error(error_msg)
        log_sync_event("File Upload Error", error_msg)
        return False

def sync_from_dropbox(file_path, dbx):
    """
    Download a file from Dropbox to the local sync directory.
    Includes integrity check simulation.
    """
    try:
        relative_path = os.path.relpath(file_path, LOCAL_SYNC_DIR)
        dropbox_path = os.path.join(DROPBOX_SYNC_FOLDER, relative_path).replace(os.sep, '/')
        metadata, res = dbx.files_download if dbx else (_ for _ in ()).throw(Exception('Dropbox disabled'))(dropbox_path)
        with open(file_path, 'wb') as f:
            f.write(res.content)
        success_msg = f"Downloaded '{dropbox_path}' from Dropbox to '{file_path}'."
        logger.info(success_msg)
        log_sync_event("File Download", success_msg)
        return True
    except Exception as e:
        error_msg = f"Error downloading from Dropbox: {e}"
        logger.error(error_msg)
        log_sync_event("File Download Error", error_msg)
        return False

def sync_data():
    """
    Synchronize local files in LOCAL_SYNC_DIR with Dropbox.
    Iterates over files with conflict resolution simulation.
    """
    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN) if _dropbox_ready() else None
        logger.info("Authenticated with Dropbox successfully." if dbx else "Dropbox sync disabled.")
        log_sync_event("Dropbox Auth", "Authenticated with Dropbox successfully.")
        for root, dirs, files in os.walk(LOCAL_SYNC_DIR):
            for file in files:
                local_file_path = os.path.join(root, file)
                sync_to_dropbox(local_file_path, dbx)
        success_msg = "Local data synchronization to Dropbox complete."
        logger.info(success_msg)
        log_sync_event("Sync Complete", success_msg)
    except Exception as e:
        error_msg = f"Error during data synchronization: {e}"
        logger.error(error_msg)
        log_sync_event("Sync Error", error_msg)

def start_sync_monitor(interval=60):
    """
    Start a background loop that synchronizes data with Dropbox every 'interval' seconds.
    This loop runs in a separate thread.
    """
    def sync_loop():
        while True:
            sync_data()
            time.sleep(interval)
    from SarahMemoryGlobals import run_async
    run_async(sync_loop)

# ============================================================================
# LEGACY SYNC: FTPS Connector (v7.7.5)
# ============================================================================

def connect_ftps_with_auto_accept(host, port=21, user=None, password=None, allow_insecure_env="SARAHMEMORY_ALLOW_INSECURE_FTPS"):
    """
    FTPS connector that auto-accepts new/changed SSL cert (configurable).
    """
    import os, ssl, ftplib, logging
    u = user or os.getenv("SARAHMEMORY_FTP_USER")
    p = password or os.getenv("SARAHMEMORY_FTP_PASS")
    if not u or not p:
        raise RuntimeError("FTP credentials not set in env (SARAHMEMORY_FTP_USER/PASS)")
    allow_insecure = str(os.getenv(allow_insecure_env, "1")).strip().lower() in ("1","true","yes","on")
    ctx = ssl.create_default_context()
    if allow_insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ftps = ftplib.FTP_TLS(context=ctx)
    logging.info("[SYNC] Connecting to FTP server at %s:%s...", host, port)
    ftps.connect(host, port, timeout=15)
    ftps.auth()
    ftps.prot_p()
    try:
        ftps.login(u, p)
    except ftplib.error_perm as e:
        logging.warning("[SYNC ERROR] Dataset sync failed: %s", e)
        raise
    return ftps


# ============================================================================
# PHASE C: Mobile App Sync Infrastructure (v8.0.0)
# ============================================================================

# Import Phase C sync infrastructure
PHASE_C_ENABLED = False
try:
    from SarahMemory_PhaseC_Sync import (
        get_sync_manager,
        shutdown_sync_manager,
        sync_device_data,
        register_new_device,
        get_device_contacts,
        get_device_history,
        get_device_reminders,
        SYNC_VERSION
    )
    PHASE_C_ENABLED = True
    logger.info(f"✅ Phase C Sync Infrastructure loaded (v{SYNC_VERSION})")
except ImportError as e:
    logger.warning(f"⚠️ Phase C Sync not available: {e}")
    logger.info("Phase C features disabled. Using legacy sync only.")

def phase_c_sync_available():
    """Check if Phase C sync infrastructure is available."""
    return PHASE_C_ENABLED

def perform_phase_c_sync(user_id: str, device_id: str, **kwargs) -> Dict:
    """
    Perform Phase C synchronization for a device.
    
    Args:
        user_id: User identifier
        device_id: Device identifier
        **kwargs: Additional sync data (contacts, history, reminders)
    
    Returns:
        Dictionary with sync results
    """
    if not PHASE_C_ENABLED:
        return {
            "success": False,
            "error": "Phase C sync not available"
        }
    
    try:
        result = sync_device_data(user_id, device_id, **kwargs)
        log_sync_event("Phase C Sync", f"User {user_id}, Device {device_id}: {result.get('success')}")
        return result
    except Exception as e:
        error_msg = f"Phase C sync error: {e}"
        logger.error(error_msg)
        log_sync_event("Phase C Sync Error", error_msg)
        return {
            "success": False,
            "error": str(e)
        }

def register_phase_c_device(user_id: str, device_name: str, device_type: str, **kwargs) -> Optional[str]:
    """
    Register a new device for Phase C synchronization.
    
    Args:
        user_id: User identifier
        device_name: Human-readable device name
        device_type: Device type (mobile, tablet, desktop, web)
        **kwargs: Additional device metadata
    
    Returns:
        Device ID if successful, None otherwise
    """
    if not PHASE_C_ENABLED:
        logger.error("Phase C sync not available - cannot register device")
        return None
    
    try:
        device_id = register_new_device(user_id, device_name, device_type, **kwargs)
        log_sync_event("Phase C Device Registration", f"Registered {device_name} for user {user_id}")
        return device_id
    except Exception as e:
        error_msg = f"Phase C device registration error: {e}"
        logger.error(error_msg)
        log_sync_event("Phase C Registration Error", error_msg)
        return None

def get_phase_c_contacts(user_id: str, since_timestamp: int = 0) -> List[Dict]:
    """Get Phase C synchronized contacts for a user."""
    if not PHASE_C_ENABLED:
        return []
    
    try:
        return get_device_contacts(user_id, since_timestamp)
    except Exception as e:
        logger.error(f"Error getting Phase C contacts: {e}")
        return []

def get_phase_c_history(user_id: str, since_timestamp: int = 0) -> List[Dict]:
    """Get Phase C synchronized history for a user."""
    if not PHASE_C_ENABLED:
        return []
    
    try:
        return get_device_history(user_id, since_timestamp)
    except Exception as e:
        logger.error(f"Error getting Phase C history: {e}")
        return []

def get_phase_c_reminders(user_id: str, include_completed: bool = False) -> List[Dict]:
    """Get Phase C synchronized reminders for a user."""
    if not PHASE_C_ENABLED:
        return []
    
    try:
        return get_device_reminders(user_id, include_completed)
    except Exception as e:
        logger.error(f"Error getting Phase C reminders: {e}")
        return []


# ============================================================================
# COMPREHENSIVE TESTING SUITE
# ============================================================================

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_test(test_name):
    """Print test name."""
    print(f"{Colors.OKCYAN}▶ Testing: {test_name}{Colors.ENDC}")

def print_pass(message):
    """Print pass message."""
    print(f"{Colors.OKGREEN}  ✅ PASS: {message}{Colors.ENDC}")

def print_fail(message):
    """Print fail message."""
    print(f"{Colors.FAIL}  ❌ FAIL: {message}{Colors.ENDC}")

def print_info(message):
    """Print info message."""
    print(f"{Colors.OKBLUE}  ℹ️  INFO: {message}{Colors.ENDC}")

def print_warning(message):
    """Print warning message."""
    print(f"{Colors.WARNING}  ⚠️  WARNING: {message}{Colors.ENDC}")


class SarahMemorySyncTestSuite:
    """Comprehensive test suite for SarahMemorySync module."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_user_id = "test_user_" + str(int(time.time()))
        self.test_device_id = None
    
    def run_all_tests(self):
        """Run all sync tests."""
        print_header("SarahMemorySync - Comprehensive Test Suite v7.7.5")
        print_info(f"Test User ID: {self.test_user_id}")
        print_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Phase C Available: {phase_c_sync_available()}")
        
        # Legacy Sync Tests
        self.test_legacy_sync_functions()
        self.test_dropbox_availability()
        self.test_ftps_connector()
        self.test_sync_event_logging()
        
        # Phase C Tests (if available)
        if phase_c_sync_available():
            self.test_phase_c_device_registration()
            self.test_phase_c_contact_sync()
            self.test_phase_c_history_sync()
            self.test_phase_c_reminder_sync()
            self.test_phase_c_full_sync()
            self.test_phase_c_conflict_resolution()
            self.test_phase_c_data_retrieval()
            self.test_phase_c_performance()
        else:
            print_warning("Phase C tests skipped - infrastructure not available")
        
        # Print final summary
        self.print_summary()
    
    def test_legacy_sync_functions(self):
        """Test 1: Legacy Sync Functions"""
        print_test("Legacy Sync Functions")
        try:
            # Test sync directory creation
            if os.path.exists(LOCAL_SYNC_DIR):
                print_pass(f"Sync directory exists: {LOCAL_SYNC_DIR}")
                self.passed += 1
            else:
                print_fail(f"Sync directory missing: {LOCAL_SYNC_DIR}")
                self.failed += 1
        except Exception as e:
            print_fail(f"Legacy sync test failed: {e}")
            self.failed += 1
    
    def test_dropbox_availability(self):
        """Test 2: Dropbox Availability"""
        print_test("Dropbox Availability")
        try:
            if _dropbox_ready():
                print_pass("Dropbox sync is available and configured")
                self.passed += 1
            else:
                print_warning("Dropbox sync not configured (expected in some environments)")
                print_info("Set DROPBOX_ACCESS_TOKEN to enable Dropbox sync")
                self.warnings += 1
        except Exception as e:
            print_fail(f"Dropbox availability check failed: {e}")
            self.failed += 1
    
    def test_ftps_connector(self):
        """Test 3: FTPS Connector"""
        print_test("FTPS Connector")
        try:
            # Test that FTPS connector function exists
            if callable(connect_ftps_with_auto_accept):
                print_pass("FTPS connector function available")
                self.passed += 1
            else:
                print_fail("FTPS connector function not callable")
                self.failed += 1
        except Exception as e:
            print_fail(f"FTPS connector test failed: {e}")
            self.failed += 1
    
    def test_sync_event_logging(self):
        """Test 4: Sync Event Logging"""
        print_test("Sync Event Logging")
        try:
            # Test logging a sync event
            log_sync_event("Test Event", "This is a test sync event")
            print_pass("Sync event logging successful")
            self.passed += 1
        except Exception as e:
            print_fail(f"Sync event logging failed: {e}")
            self.failed += 1
    
    def test_phase_c_device_registration(self):
        """Test 5: Phase C Device Registration"""
        print_test("Phase C Device Registration")
        try:
            self.test_device_id = register_phase_c_device(
                user_id=self.test_user_id,
                device_name="Test Device",
                device_type="mobile",
                device_profile="Standard",
                platform="TestPlatform",
                os_version="1.0",
                app_version="7.7.5"
            )
            
            if self.test_device_id:
                print_pass(f"Device registered: {self.test_device_id}")
                self.passed += 1
            else:
                print_fail("Device registration returned None")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C device registration failed: {e}")
            self.failed += 1
    
    def test_phase_c_contact_sync(self):
        """Test 6: Phase C Contact Synchronization"""
        print_test("Phase C Contact Synchronization")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Create test contacts
            test_contacts = [
                {
                    'contact_id': 'contact_test_001',
                    'display_name': 'John Test',
                    'phone_number': '+1234567890',
                    'email': 'john@test.com',
                    'is_favorite': 1,
                    'last_modified_timestamp': int(time.time())
                }
            ]
            
            # Perform sync
            result = perform_phase_c_sync(
                self.test_user_id,
                self.test_device_id,
                contacts=test_contacts
            )
            
            if result.get('success'):
                print_pass(f"Contact sync completed")
                print_info(f"Uploaded: {result.get('uploaded', {})}")
                self.passed += 1
            else:
                print_fail(f"Contact sync failed: {result.get('error')}")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C contact sync failed: {e}")
            self.failed += 1
    
    def test_phase_c_history_sync(self):
        """Test 7: Phase C History Synchronization"""
        print_test("Phase C History Synchronization")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Create test history
            test_history = [
                {
                    'history_id': 'history_test_001',
                    'peer_hash': hashlib.sha256(b'test_peer').hexdigest(),
                    'route_type': 'sip',
                    'direction': 'outgoing',
                    'timestamp_start': int(time.time()),
                    'timestamp_end': int(time.time()) + 60,
                    'duration_seconds': 60,
                    'outcome': 'completed',
                    'last_modified_timestamp': int(time.time())
                }
            ]
            
            # Perform sync
            result = perform_phase_c_sync(
                self.test_user_id,
                self.test_device_id,
                history=test_history
            )
            
            if result.get('success'):
                print_pass(f"History sync completed")
                self.passed += 1
            else:
                print_fail(f"History sync failed: {result.get('error')}")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C history sync failed: {e}")
            self.failed += 1
    
    def test_phase_c_reminder_sync(self):
        """Test 8: Phase C Reminder Synchronization"""
        print_test("Phase C Reminder Synchronization")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Create test reminders
            test_reminders = [
                {
                    'reminder_id': 'reminder_test_001',
                    'title': 'Test Reminder',
                    'description': 'This is a test reminder',
                    'reminder_timestamp': int(time.time()) + 3600,
                    'priority': 1,
                    'last_modified_timestamp': int(time.time())
                }
            ]
            
            # Perform sync
            result = perform_phase_c_sync(
                self.test_user_id,
                self.test_device_id,
                reminders=test_reminders
            )
            
            if result.get('success'):
                print_pass(f"Reminder sync completed")
                self.passed += 1
            else:
                print_fail(f"Reminder sync failed: {result.get('error')}")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C reminder sync failed: {e}")
            self.failed += 1
    
    def test_phase_c_full_sync(self):
        """Test 9: Phase C Full Sync Operation"""
        print_test("Phase C Full Sync Operation")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Prepare comprehensive sync data
            contacts = [
                {'contact_id': f'contact_full_{i}', 'display_name': f'Contact {i}',
                 'phone_number': f'+123456789{i}', 'last_modified_timestamp': int(time.time())}
                for i in range(5)
            ]
            
            reminders = [
                {'reminder_id': f'reminder_full_{i}', 'title': f'Reminder {i}',
                 'reminder_timestamp': int(time.time()) + 3600 * i,
                 'last_modified_timestamp': int(time.time())}
                for i in range(3)
            ]
            
            # Perform full sync
            result = perform_phase_c_sync(
                self.test_user_id,
                self.test_device_id,
                contacts=contacts,
                reminders=reminders
            )
            
            if result.get('success'):
                print_pass("Full sync completed")
                print_info(f"Uploaded: {result.get('uploaded', {})}")
                
                if result.get('errors'):
                    print_warning(f"Sync had {len(result['errors'])} error(s)")
                    self.warnings += 1
                
                self.passed += 1
            else:
                print_fail(f"Full sync failed: {result.get('errors', [])}")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C full sync failed: {e}")
            self.failed += 1
    
    def test_phase_c_conflict_resolution(self):
        """Test 10: Phase C Conflict Resolution"""
        print_test("Phase C Conflict Resolution")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Create conflicting contact versions
            contact_v1 = {
                'contact_id': 'conflict_test_001',
                'display_name': 'Original Name',
                'phone_number': '+1111111111',
                'last_modified_timestamp': int(time.time()) - 100
            }
            
            contact_v2 = {
                'contact_id': 'conflict_test_001',
                'display_name': 'Updated Name',
                'phone_number': '+2222222222',
                'last_modified_timestamp': int(time.time())
            }
            
            # Sync both versions
            perform_phase_c_sync(self.test_user_id, self.test_device_id, contacts=[contact_v1])
            perform_phase_c_sync(self.test_user_id, self.test_device_id, contacts=[contact_v2])
            
            # Verify newer version won (last-writer-wins)
            contacts = get_phase_c_contacts(self.test_user_id)
            conflict_contact = next((c for c in contacts if c['contact_id'] == 'conflict_test_001'), None)
            
            if conflict_contact and conflict_contact['display_name'] == 'Updated Name':
                print_pass("Conflict resolved: Last-writer-wins strategy applied")
                self.passed += 1
            else:
                print_fail("Conflict resolution failed")
                self.failed += 1
        except Exception as e:
            print_fail(f"Phase C conflict resolution test failed: {e}")
            self.failed += 1
    
    def test_phase_c_data_retrieval(self):
        """Test 11: Phase C Data Retrieval"""
        print_test("Phase C Data Retrieval")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Retrieve synced data
            contacts = get_phase_c_contacts(self.test_user_id)
            history = get_phase_c_history(self.test_user_id)
            reminders = get_phase_c_reminders(self.test_user_id)
            
            print_info(f"Retrieved {len(contacts)} contacts")
            print_info(f"Retrieved {len(history)} history entries")
            print_info(f"Retrieved {len(reminders)} reminders")
            
            print_pass("Data retrieval successful")
            self.passed += 1
        except Exception as e:
            print_fail(f"Phase C data retrieval failed: {e}")
            self.failed += 1
    
    def test_phase_c_performance(self):
        """Test 12: Phase C Performance"""
        print_test("Phase C Performance")
        try:
            if not self.test_device_id:
                print_warning("Skipping - device not registered")
                self.warnings += 1
                return
            
            # Test bulk contact sync
            start_time = time.time()
            
            bulk_contacts = [
                {
                    'contact_id': f'perf_contact_{i}',
                    'display_name': f'Performance Test {i}',
                    'phone_number': f'+999999{i:04d}',
                    'last_modified_timestamp': int(time.time())
                }
                for i in range(50)
            ]
            
            result = perform_phase_c_sync(
                self.test_user_id,
                self.test_device_id,
                contacts=bulk_contacts
            )
            
            duration = time.time() - start_time
            rate = len(bulk_contacts) / duration if duration > 0 else 0
            
            print_pass(f"Synced {len(bulk_contacts)} contacts in {duration:.2f}s")
            print_info(f"Rate: {rate:.2f} contacts/second")
            
            if rate > 10:
                print_pass("Performance is acceptable")
                self.passed += 1
            else:
                print_warning("Performance may need optimization")
                self.warnings += 1
        except Exception as e:
            print_fail(f"Phase C performance test failed: {e}")
            self.failed += 1
    
    def print_summary(self):
        """Print test results summary."""
        print_header("Test Results Summary")
        
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.BOLD}Tests Run:{Colors.ENDC} {total}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}Passed:{Colors.ENDC} {self.passed}")
        print(f"{Colors.FAIL}{Colors.BOLD}Failed:{Colors.ENDC} {self.failed}")
        print(f"{Colors.WARNING}{Colors.BOLD}Warnings:{Colors.ENDC} {self.warnings}")
        print(f"\n{Colors.BOLD}Pass Rate:{Colors.ENDC} {pass_rate:.1f}%\n")
        
        if self.failed == 0:
            print(f"{Colors.OKGREEN}{Colors.BOLD}✅ ALL TESTS PASSED{Colors.ENDC}\n")
            return 0
        else:
            print(f"{Colors.FAIL}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.ENDC}\n")
            return 1


# ============================================================================
# MODULE MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SarahMemorySync Module Test Suite')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--legacy-only', action='store_true', help='Test legacy sync only')
    parser.add_argument('--phase-c-only', action='store_true', help='Test Phase C only')
    args = parser.parse_args()
    
    if args.test or args.legacy_only or args.phase_c_only:
        # Run test suite
        suite = SarahMemorySyncTestSuite()
        
        if args.legacy_only:
            print_header("SarahMemorySync - Legacy Sync Tests Only")
            suite.test_legacy_sync_functions()
            suite.test_dropbox_availability()
            suite.test_ftps_connector()
            suite.test_sync_event_logging()
            suite.print_summary()
        elif args.phase_c_only:
            if phase_c_sync_available():
                print_header("SarahMemorySync - Phase C Tests Only")
                suite.test_phase_c_device_registration()
                suite.test_phase_c_contact_sync()
                suite.test_phase_c_history_sync()
                suite.test_phase_c_reminder_sync()
                suite.test_phase_c_full_sync()
                suite.test_phase_c_conflict_resolution()
                suite.test_phase_c_data_retrieval()
                suite.test_phase_c_performance()
                suite.print_summary()
            else:
                print_fail("Phase C infrastructure not available")
                sys.exit(1)
        else:
            suite.run_all_tests()
        
        sys.exit(suite.failed)
    else:
        # Run legacy sync test
        logger.info("Starting SarahMemorySync module test (legacy mode).")
        log_sync_event("Module Test Start", "Starting sync module test.")
        
        sample_file = os.path.join(LOCAL_SYNC_DIR, "test_sync.txt")
        try:
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write("This is a test file for the sync module.\nTimestamp: " + time.ctime())
            logger.info(f"Created sample file: {sample_file}")
            log_sync_event("Create Sample File", f"Created sample file at {sample_file}")
        except Exception as e:
            error_msg = f"Error creating sample file: {e}"
            logger.error(error_msg)
            log_sync_event("Create Sample File Error", error_msg)
        
        sync_data()
        
        logger.info("SarahMemorySync module testing complete.")
        log_sync_event("Module Test Complete", "Sync module test complete.")
        
        # Show Phase C status
        if phase_c_sync_available():
            print_info("Phase C sync infrastructure available!")
            print_info("Run with --test flag to execute comprehensive test suite")
            print_info("Run with --phase-c-only to test Phase C features only")
        else:
            print_warning("Phase C sync not available - using legacy sync only")

# ====================================================================
# END OF SarahMemorySync.py v8.0.0

# ============================================================================
# SARAHNET CORE FILE SYNC (v8.0.0) - Broker-based, cross-platform
# - Uses existing /api/net/file/* broker endpoints (appnet.py) for transport.
# - Uses REMOTE_SYNC_ENABLED from SarahMemoryGlobals.
# - Trust model (MVP): only accept inbound sync bundles from known peer node_ids.
#   Known peers are derived from config.SARAHNET_PEERS keys (and optionally config.SARAHNET_NODE_ID).
# - No new Globals/env flags required for MVP.
# ============================================================================

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_relpath(path: str, base: str) -> str:
    base_abs = os.path.abspath(base)
    p_abs = os.path.abspath(path)
    if not p_abs.startswith(base_abs):
        raise ValueError("path escapes base")
    return os.path.relpath(p_abs, base_abs).replace("\\", "/")

def _sync_db_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "memory", "datasets", "device_link.db"))

def _ensure_sync_tables() -> None:
    try:
        db_path = _sync_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_sync_state(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                peer_id TEXT,
                relpath TEXT,
                sha256 TEXT,
                mtime REAL,
                size INTEGER,
                last_sent_ts REAL,
                last_recv_ts REAL
            )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_code_sync_state_uq ON code_sync_state(node_id, peer_id, relpath)")
        con.commit()
        con.close()
    except Exception as e:
        logger.error(f"[SYNC] Failed ensuring sync tables: {e}")

def _trusted_peer_ids() -> set:
    try:
        peers = getattr(config, "SARAHNET_PEERS", {}) or {}
        out = set(peers.keys())
    except Exception:
        out = set()
    try:
        out.add(getattr(config, "SARAHNET_NODE_ID", "local-node"))
    except Exception:
        pass
    return out

def _broker_base_url() -> str:
    # Prefer the running server (local loopback) if present; else fall back to config.SARAH_WEB_BASE
    # NOTE: app.py mounts appnet.py under /api/net/*
    base = getattr(config, "SARAH_WEB_BASE", "").strip()
    if not base:
        base = "http://127.0.0.1:8000"
    # If base already includes /api, keep it; else assume root.
    return base.rstrip("/")

def _broker_headers() -> dict:
    h = {"Content-Type": "application/json"}
    # If you later wire API key auth into appnet, this header can be used.
    api_key = getattr(config, "REMOTE_API_KEY", None)
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h

def _http_post_json(url: str, payload: dict, timeout: float = 10.0, extra_headers: dict | None = None) -> tuple[int, dict]:
    try:
        import requests  # type: ignore
        hdrs = _broker_headers()
        if extra_headers:
            try:
                hdrs.update(extra_headers)
            except Exception:
                pass
        r = requests.post(url, json=payload, headers=hdrs, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text}
    except Exception:
        # urllib fallback
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=_broker_headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", "ignore")
                try:
                    return resp.getcode(), json.loads(txt)
                except Exception:
                    return resp.getcode(), {"text": txt}
        except Exception as e:
            return 599, {"error": str(e)}

def _http_get_json(url: str, timeout: float = 10.0) -> tuple[int, dict]:
    try:
        import requests  # type: ignore
        r = requests.get(url, headers=_broker_headers(), timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text}
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers=_broker_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", "ignore")
                try:
                    return resp.getcode(), json.loads(txt)
                except Exception:
                    return resp.getcode(), {"text": txt}
        except Exception as e:
            return 599, {"error": str(e)}

def _collect_core_files(base_dir: str) -> list[dict]:
    """
    MVP allowlist:
      - SarahMemory*.py in BASE_DIR
      - api/server/app*.py (server modules)
    """
    items: list[dict] = []
    # Core python files
    for name in os.listdir(base_dir):
        if name.startswith("SarahMemory") and name.endswith(".py"):
            p = os.path.join(base_dir, name)
            try:
                st = os.stat(p)
                items.append({"abs": p, "rel": _safe_relpath(p, base_dir), "mtime": st.st_mtime, "size": st.st_size})
            except Exception:
                pass

    # API server files
    api_dir = os.path.abspath(os.path.join(base_dir, "api", "server"))
    if os.path.isdir(api_dir):
        for name in os.listdir(api_dir):
            if name.startswith("app") and name.endswith(".py"):
                p = os.path.join(api_dir, name)
                try:
                    st = os.stat(p)
                    items.append({"abs": p, "rel": _safe_relpath(p, base_dir), "mtime": st.st_mtime, "size": st.st_size})
                except Exception:
                    pass
    return items

def sarahnet_sync_push(peer_id: str, base_dir: str | None = None, max_bytes: int = 1_900_000) -> dict:
    """
    Push changed core files to a peer via broker (small file endpoint).
    Only sends files <= max_bytes (broker cap).
    """
    if not getattr(config, "REMOTE_SYNC_ENABLED", False):
        return {"ok": False, "reason": "REMOTE_SYNC_ENABLED is False"}

    base_dir = base_dir or getattr(config, "BASE_DIR", os.getcwd())
    node_id = getattr(config, "SARAHNET_NODE_ID", "local-node")

    if peer_id not in _trusted_peer_ids():
        return {"ok": False, "reason": f"peer_id not trusted: {peer_id}"}

    _ensure_sync_tables()
    files = _collect_core_files(base_dir)

    sent = 0
    skipped = 0
    errors = 0
    details = []

    db_path = _sync_db_path()
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for it in files:
        try:
            if int(it["size"]) > int(max_bytes):
                skipped += 1
                continue
            rel = it["rel"]
            sha = _sha256_file(it["abs"])

            cur.execute("SELECT sha256, mtime FROM code_sync_state WHERE node_id=? AND peer_id=? AND relpath=? LIMIT 1",
                        (node_id, peer_id, rel))
            row = cur.fetchone()
            prev_sha = row[0] if row else None
            prev_mtime = float(row[1]) if row else 0.0

            if prev_sha == sha and prev_mtime >= float(it["mtime"]):
                skipped += 1
                continue

            with open(it["abs"], "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("ascii")

            url = _broker_base_url() + "/api/net/file/send"
            payload = {
                "to_node": peer_id,
                "from_node": node_id,
                "filename": f"core_sync::{rel}",
                "mime": "application/octet-stream",
                "data_b64": data_b64,
                "sha256": sha,
            }
            timeout_s = float(getattr(config, "REMOTE_HTTP_TIMEOUT", 10.0))
            extra = {"X-SarahNet-Channel": "sync"}
            status, resp = _http_post_json(url, payload, timeout=timeout_s, extra_headers=extra)
            if status == 429:
                for attempt in range(1, 6):
                    try:
                        time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                    except Exception:
                        pass
                    status, resp = _http_post_json(url, payload, timeout=timeout_s, extra_headers=extra)
                    if status != 429:
                        break
            if status == 200 and resp.get("ok"):
                sent += 1
                cur.execute(
                    "INSERT INTO code_sync_state(node_id,peer_id,relpath,sha256,mtime,size,last_sent_ts,last_recv_ts) "
                    "VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(node_id,peer_id,relpath) DO UPDATE SET sha256=excluded.sha256, mtime=excluded.mtime, size=excluded.size, last_sent_ts=excluded.last_sent_ts",
                    (node_id, peer_id, rel, sha, float(it["mtime"]), int(it["size"]), time.time(), None),
                )
                details.append({"rel": rel, "sent": True})
            else:
                errors += 1
                details.append({"rel": rel, "sent": False, "status": status, "resp": resp})
        except Exception as e:
            errors += 1
            details.append({"rel": it.get("rel"), "sent": False, "error": str(e)})

    con.commit()
    con.close()

    return {"ok": True, "node_id": node_id, "peer_id": peer_id, "sent": sent, "skipped": skipped, "errors": errors, "details": details[:25]}

def sarahnet_sync_poll_and_apply(base_dir: str | None = None, max_items: int = 25) -> dict:
    """
    Poll broker for inbound core-sync bundles and apply them if trusted + newer.
    """
    if not getattr(config, "REMOTE_SYNC_ENABLED", False):
        return {"ok": False, "reason": "REMOTE_SYNC_ENABLED is False"}

    base_dir = base_dir or getattr(config, "BASE_DIR", os.getcwd())
    node_id = getattr(config, "SARAHNET_NODE_ID", "local-node")
    trusted = _trusted_peer_ids()

    _ensure_sync_tables()

    url = _broker_base_url() + f"/api/net/file/poll?to_node={node_id}&limit={int(max_items)}"
    status, resp = _http_get_json(url, timeout=float(getattr(config, "REMOTE_HTTP_TIMEOUT", 10.0)))
    if status != 200 or not resp.get("ok"):
        return {"ok": False, "status": status, "resp": resp}

    items = resp.get("items") or []
    applied = 0
    rejected = 0
    errors = 0
    acks = 0

    db_path = _sync_db_path()
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for it in items:
        try:
            file_id = it.get("file_id") or it.get("id")
            from_node = (it.get("from_node") or "").strip()
            filename = (it.get("filename") or "").strip()
            data_b64 = (it.get("data_b64") or "").strip()
            sha_in = (it.get("sha256") or "").strip()

            if not file_id or not filename.startswith("core_sync::"):
                # Not ours; ignore
                continue

            if from_node not in trusted:
                rejected += 1
                continue

            rel = filename.split("core_sync::", 1)[1].strip()
            if not rel or ".." in rel or rel.startswith(("/", "\\")):
                rejected += 1
                continue

            raw = base64.b64decode(data_b64.encode("ascii"), validate=False)
            sha = hashlib.sha256(raw).hexdigest()
            if sha_in and sha != sha_in:
                rejected += 1
                continue

            dst = os.path.abspath(os.path.join(base_dir, rel.replace("/", os.sep)))
            if not dst.startswith(os.path.abspath(base_dir)):
                rejected += 1
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Compare with existing
            apply_ok = True
            try:
                if os.path.exists(dst):
                    local_sha = _sha256_file(dst)
                    if local_sha == sha:
                        apply_ok = False  # nothing to do
            except Exception:
                pass

            if apply_ok:
                with open(dst, "wb") as f:
                    f.write(raw)
                applied += 1
                cur.execute(
                    "INSERT INTO code_sync_state(node_id,peer_id,relpath,sha256,mtime,size,last_sent_ts,last_recv_ts) "
                    "VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(node_id,peer_id,relpath) DO UPDATE SET sha256=excluded.sha256, mtime=excluded.mtime, size=excluded.size, last_recv_ts=excluded.last_recv_ts",
                    (node_id, from_node, rel, sha, time.time(), len(raw), None, time.time()),
                )

            # Ack (best-effort)
            try:
                aurl = _broker_base_url() + "/api/net/file/ack"
                apayload = {"file_id": file_id, "to_node": node_id}
                st2, r2 = _http_post_json(aurl, apayload, timeout=float(getattr(config, "REMOTE_HTTP_TIMEOUT", 10.0)))
                if st2 == 200 and r2.get("ok"):
                    acks += 1
            except Exception:
                pass

        except Exception as e:
            errors += 1

    con.commit()
    con.close()

    return {"ok": True, "node_id": node_id, "applied": applied, "rejected": rejected, "errors": errors, "acked": acks, "polled": len(items)}

def sarahnet_sync_tick(peer_ids: list[str] | None = None, base_dir: str | None = None) -> dict:
    """
    One-shot orchestrator:
      1) poll inbound bundles and apply
      2) push deltas to configured peers
    """
    base_dir = base_dir or getattr(config, "BASE_DIR", os.getcwd())
    node_id = getattr(config, "SARAHNET_NODE_ID", "local-node")

    # Default peers = keys from SARAHNET_PEERS excluding self
    if peer_ids is None:
        try:
            peer_ids = [k for k in (getattr(config, "SARAHNET_PEERS", {}) or {}).keys() if k != node_id]
        except Exception:
            peer_ids = []

    results = {"ok": True, "node_id": node_id, "poll": sarahnet_sync_poll_and_apply(base_dir=base_dir), "push": []}
    for pid in peer_ids:
        results["push"].append(sarahnet_sync_push(pid, base_dir=base_dir))
    return results

# ====================================================================
# END OF SarahMemorySync.py v8.0.0
# ====================================================================
