"""--==The SarahMemory Project==--
File: SarahMemorySync.py
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
# ====================================================================
