"""--==The SarahMemory Project==--
File: SarahMemoryFilesystem.py
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

SarahMemoryFilesystem.py - Backup System & File Manager
=====================================================================

This module provides:
- Comprehensive backup and restore capabilities (full, incremental, differential)
- Advanced file monitoring and inspection
- Built-in antivirus and malware protection
- File operations (move, copy, rename, delete, set attributes)
- Real-time filesystem event tracking
- Checksum verification and integrity checking
- Multi-threaded compression and backup operations
- Smart backup rotation and retention policies
- Quarantine system for suspicious files
- File analysis and reporting

"""

import os
import sys
import zipfile
import hashlib
import logging
import shutil
import sqlite3
import time
import argparse
import json
import re
import stat
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Import SarahMemoryGlobals for centralized configuration
try:
    from SarahMemoryGlobals import (
        BASE_DIR, BACKUP_DIR, SETTINGS_DIR, MEMORY_DIR, LOGS_DIR,
        DATASETS_DIR, VAULT_DIR, DIAGNOSTICS_DIR, DATA_DIR
    )
    GLOBALS_IMPORTED = True
except ImportError:
    # Fallback to default paths if Globals not available
    BASE_DIR = os.getcwd()
    BACKUP_DIR = os.path.join(BASE_DIR, "data", "backup")
    SETTINGS_DIR = os.path.join(BASE_DIR, "data", "settings")
    MEMORY_DIR = os.path.join(BASE_DIR, "data", "memory")
    LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")
    DATASETS_DIR = os.path.join(MEMORY_DIR, "datasets")
    VAULT_DIR = os.path.join(BASE_DIR, "data", "vault")
    DIAGNOSTICS_DIR = os.path.join(BASE_DIR, "data", "diagnostics")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    GLOBALS_IMPORTED = False

# Setup logging
logger = logging.getLogger("SarahMemoryFilesystem")
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Quarantine directory for suspicious files
QUARANTINE_DIR = os.path.join(DATA_DIR, "quarantine")

# Backup configuration
BACKUP_RETENTION_DAYS = 30  # Keep backups for 30 days
MAX_BACKUP_COUNT = 50  # Maximum number of backups to keep
BACKUP_COMPRESSION_LEVEL = 6  # ZIP compression level (0-9)


# ============================================================================
# MALWARE SIGNATURE DATABASE
# ============================================================================

class MalwareSignatures:
    """
    Malware and virus signature database for detection.
    This class contains patterns, hashes, and behaviors associated with malware.
    """
    
    # Known malicious file hashes (SHA-256)
    MALICIOUS_HASHES = {
        # Add known malware hashes here - these are examples only
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Empty file hash - example
    }
    
    # Suspicious file patterns (regex)
    SUSPICIOUS_PATTERNS = [
        r"eval\s*\(",  # Eval commands (often used in malicious scripts)
        r"exec\s*\(",  # Exec commands
        r"__import__\s*\(",  # Dynamic imports
        r"base64\.b64decode",  # Base64 decoding (common in obfuscated malware)
        r"subprocess\.call",  # System command execution
        r"os\.system",  # OS command execution
        r"powershell\.exe",  # PowerShell execution
        r"cmd\.exe\s*/c",  # Command prompt execution
        r"wget\s+http",  # File download
        r"curl\s+http",  # File download
        r"rm\s+-rf\s+/",  # Dangerous delete commands
        r"del\s+/f\s+/s\s+/q",  # Windows delete commands
        r"format\s+c:",  # Format commands
    ]
    
    # Suspicious file extensions
    SUSPICIOUS_EXTENSIONS = {
        ".exe", ".dll", ".bat", ".cmd", ".vbs", ".ps1", ".sh",
        ".scr", ".pif", ".msi", ".com", ".jar", ".app"
    }
    
    # Ransomware indicators
    RANSOMWARE_INDICATORS = [
        r"README.*\.txt",  # Common ransomware readme pattern
        r"DECRYPT.*\.txt",
        r"HOW_TO_DECRYPT",
        r"\.encrypted$",
        r"\.locked$",
        r"\.crypto$",
    ]
    
    @classmethod
    def is_suspicious_extension(cls, filename: str) -> bool:
        """Check if file extension is suspicious."""
        ext = os.path.splitext(filename.lower())[1]
        return ext in cls.SUSPICIOUS_EXTENSIONS
    
    @classmethod
    def scan_content(cls, content: bytes, filename: str = "") -> List[str]:
        """
        Scan file content for malicious patterns.
        Returns list of detected threats.
        """
        threats = []
        
        try:
            # Try to decode as text for pattern matching
            text_content = content.decode('utf-8', errors='ignore')
            
            # Check for suspicious patterns
            for pattern in cls.SUSPICIOUS_PATTERNS:
                if re.search(pattern, text_content, re.IGNORECASE):
                    threats.append(f"Suspicious pattern detected: {pattern}")
            
            # Check for ransomware indicators
            for pattern in cls.RANSOMWARE_INDICATORS:
                if re.search(pattern, filename, re.IGNORECASE):
                    threats.append(f"Ransomware indicator in filename: {pattern}")
                if re.search(pattern, text_content, re.IGNORECASE):
                    threats.append(f"Ransomware indicator in content: {pattern}")
                    
        except Exception as e:
            logger.warning(f"Error scanning content: {e}")
        
        return threats
    
    @classmethod
    def check_hash(cls, file_hash: str) -> bool:
        """Check if file hash matches known malware."""
        return file_hash.lower() in {h.lower() for h in cls.MALICIOUS_HASHES}


# ============================================================================
# DATABASE LOGGING SYSTEM
# ============================================================================

def get_db_connection() -> sqlite3.Connection:
    """Get database connection for filesystem logging."""
    db_path = os.path.join(DATASETS_DIR, "filesystem_logs.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_filesystem_database():
    """Initialize filesystem event logging database with comprehensive tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Filesystem events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filesystem_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                path TEXT,
                details TEXT,
                user TEXT,
                success INTEGER DEFAULT 1,
                error_message TEXT
            )
        """)
        
        # Backup history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                backup_type TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                source_path TEXT NOT NULL,
                size_bytes INTEGER,
                file_count INTEGER,
                checksum TEXT,
                compression_ratio REAL,
                duration_seconds REAL,
                status TEXT DEFAULT 'completed',
                error_message TEXT
            )
        """)
        
        # File integrity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_integrity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                last_checked TEXT NOT NULL,
                checksum TEXT NOT NULL,
                size_bytes INTEGER,
                modified_time TEXT,
                status TEXT DEFAULT 'verified'
            )
        """)
        
        # Malware scan results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS malware_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                scan_type TEXT,
                threats_found INTEGER DEFAULT 0,
                threat_details TEXT,
                action_taken TEXT,
                file_hash TEXT
            )
        """)
        
        # File operations audit table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                source_path TEXT,
                destination_path TEXT,
                user TEXT,
                size_bytes INTEGER,
                success INTEGER DEFAULT 1,
                error_message TEXT
            )
        """)
        
        # Quarantine log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quarantine_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                original_path TEXT NOT NULL,
                quarantine_path TEXT NOT NULL,
                reason TEXT,
                threat_level TEXT,
                restored INTEGER DEFAULT 0,
                restore_timestamp TEXT
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON filesystem_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backups_timestamp ON backup_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_integrity_path ON file_integrity(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON malware_scans(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON file_operations(timestamp)")
        
        conn.commit()
        conn.close()
        logger.info("Filesystem database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize filesystem database: {e}")


def log_filesystem_event(event_type: str, path: str = "", details: str = "", 
                         success: bool = True, error_message: str = ""):
    """
    Log filesystem event to database.
    
    Args:
        event_type: Type of event (backup, scan, move, copy, etc.)
        path: File or directory path involved
        details: Additional details about the event
        success: Whether operation was successful
        error_message: Error message if operation failed
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO filesystem_events 
            (timestamp, event_type, path, details, user, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event_type,
            path,
            details,
            os.getenv("USERNAME", "system"),
            1 if success else 0,
            error_message
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed logging filesystem event: {e}")


def log_backup_operation(backup_type: str, backup_path: str, source_path: str,
                         size_bytes: int, file_count: int, checksum: str,
                         compression_ratio: float, duration: float,
                         status: str = "completed", error_message: str = ""):
    """Log backup operation details to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO backup_history 
            (timestamp, backup_type, backup_path, source_path, size_bytes, 
             file_count, checksum, compression_ratio, duration_seconds, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            backup_type,
            backup_path,
            source_path,
            size_bytes,
            file_count,
            checksum,
            compression_ratio,
            duration,
            status,
            error_message
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed logging backup operation: {e}")


def log_malware_scan(file_path: str, scan_type: str, threats_found: int,
                     threat_details: str, action_taken: str, file_hash: str):
    """Log malware scan results to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO malware_scans 
            (timestamp, file_path, scan_type, threats_found, threat_details, 
             action_taken, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            file_path,
            scan_type,
            threats_found,
            threat_details,
            action_taken,
            file_hash
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed logging malware scan: {e}")


# ============================================================================
# FILE INTEGRITY AND CHECKSUM VERIFICATION
# ============================================================================

def calculate_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate checksum of a file using specified algorithm.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)
    
    Returns:
        Hexadecimal checksum string
    """
    try:
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        else:
            hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        logger.error(f"Error calculating checksum for {file_path}: {e}")
        return ""


def verify_file_integrity(file_path: str, expected_checksum: str, 
                          algorithm: str = "sha256") -> bool:
    """
    Verify file integrity by comparing checksums.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm used
    
    Returns:
        True if checksums match, False otherwise
    """
    try:
        actual_checksum = calculate_checksum(file_path, algorithm)
        return actual_checksum.lower() == expected_checksum.lower()
    except Exception as e:
        logger.error(f"Error verifying file integrity: {e}")
        return False


def update_file_integrity_record(file_path: str):
    """Update or create file integrity record in database."""
    try:
        checksum = calculate_checksum(file_path)
        if not checksum:
            return
        
        size = os.path.getsize(file_path)
        modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("SELECT id FROM file_integrity WHERE file_path = ?", (file_path,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute("""
                UPDATE file_integrity 
                SET last_checked = ?, checksum = ?, size_bytes = ?, 
                    modified_time = ?, status = 'verified'
                WHERE file_path = ?
            """, (datetime.now().isoformat(), checksum, size, modified, file_path))
        else:
            cursor.execute("""
                INSERT INTO file_integrity 
                (file_path, last_checked, checksum, size_bytes, modified_time, status)
                VALUES (?, ?, ?, ?, ?, 'verified')
            """, (file_path, datetime.now().isoformat(), checksum, size, modified))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating file integrity record: {e}")


# ============================================================================
# ANTIVIRUS AND MALWARE SCANNING
# ============================================================================

class FileScanner:
    """
    File scanner for malware and virus detection.
    Provides real-time and on-demand scanning capabilities.
    """
    
    def __init__(self):
        self.scan_queue = queue.Queue()
        self.scanning_active = False
        self.scan_thread = None
        self.statistics = {
            "files_scanned": 0,
            "threats_found": 0,
            "files_quarantined": 0,
            "scan_start_time": None
        }
    
    def scan_file(self, file_path: str, quarantine_on_threat: bool = True) -> Dict:
        """
        Scan a single file for malware and suspicious content.
        
        Args:
            file_path: Path to file to scan
            quarantine_on_threat: Whether to quarantine file if threat detected
        
        Returns:
            Dictionary with scan results
        """
        result = {
            "file_path": file_path,
            "scan_time": datetime.now().isoformat(),
            "threats": [],
            "threat_level": "clean",
            "action_taken": "none",
            "file_hash": ""
        }
        
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                result["threats"].append("File not found or not a regular file")
                result["threat_level"] = "error"
                return result
            
            # Calculate file hash
            file_hash = calculate_checksum(file_path)
            result["file_hash"] = file_hash
            
            # Check against known malware hashes
            if MalwareSignatures.check_hash(file_hash):
                result["threats"].append("Known malware hash detected")
                result["threat_level"] = "critical"
            
            # Check file extension
            if MalwareSignatures.is_suspicious_extension(file_path):
                result["threats"].append("Suspicious file extension")
                if not result["threat_level"] == "critical":
                    result["threat_level"] = "medium"
            
            # Scan file content
            try:
                with open(file_path, 'rb') as f:
                    content = f.read(1024 * 1024)  # Read first 1MB
                
                content_threats = MalwareSignatures.scan_content(
                    content, 
                    os.path.basename(file_path)
                )
                
                if content_threats:
                    result["threats"].extend(content_threats)
                    if result["threat_level"] == "clean":
                        result["threat_level"] = "high"
            
            except Exception as e:
                logger.warning(f"Could not read file content for scanning: {e}")
            
            # Take action if threats found
            if result["threats"] and quarantine_on_threat:
                if result["threat_level"] in ["critical", "high"]:
                    quarantined = self.quarantine_file(file_path, result["threat_level"])
                    if quarantined:
                        result["action_taken"] = "quarantined"
                        self.statistics["files_quarantined"] += 1
            
            # Log scan results
            log_malware_scan(
                file_path,
                "file_scan",
                len(result["threats"]),
                "; ".join(result["threats"]),
                result["action_taken"],
                file_hash
            )
            
            self.statistics["files_scanned"] += 1
            if result["threats"]:
                self.statistics["threats_found"] += 1
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            result["threats"].append(f"Scan error: {str(e)}")
            result["threat_level"] = "error"
        
        return result
    
    def scan_directory(self, directory: str, recursive: bool = True,
                       quarantine_on_threat: bool = True) -> List[Dict]:
        """
        Scan entire directory for malware.
        
        Args:
            directory: Path to directory to scan
            recursive: Whether to scan subdirectories
            quarantine_on_threat: Whether to quarantine threats
        
        Returns:
            List of scan results for each file
        """
        results = []
        
        try:
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return results
            
            self.statistics["scan_start_time"] = datetime.now()
            logger.info(f"Starting directory scan: {directory}")
            
            # Collect files to scan
            files_to_scan = []
            
            if recursive:
                for root, dirs, files in os.walk(directory):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        files_to_scan.append(file_path)
            else:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        files_to_scan.append(file_path)
            
            logger.info(f"Scanning {len(files_to_scan)} files...")
            
            # Scan files with thread pool for better performance
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(self.scan_file, fp, quarantine_on_threat): fp 
                    for fp in files_to_scan
                }
                
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["threats"]:
                            logger.warning(
                                f"Threats found in {result['file_path']}: "
                                f"{', '.join(result['threats'])}"
                            )
                    
                    except Exception as e:
                        logger.error(f"Error processing scan result: {e}")
            
            scan_duration = (datetime.now() - self.statistics["scan_start_time"]).total_seconds()
            logger.info(
                f"Directory scan completed. Files: {self.statistics['files_scanned']}, "
                f"Threats: {self.statistics['threats_found']}, "
                f"Duration: {scan_duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return results
    
    def quarantine_file(self, file_path: str, threat_level: str = "medium") -> bool:
        """
        Move suspicious file to quarantine directory.
        
        Args:
            file_path: Path to file to quarantine
            threat_level: Severity level of threat
        
        Returns:
            True if file was quarantined successfully
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Create quarantine directory if it doesn't exist
            os.makedirs(QUARANTINE_DIR, exist_ok=True)
            
            # Generate quarantine filename with timestamp
            basename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{timestamp}_{basename}.quarantined"
            quarantine_path = os.path.join(QUARANTINE_DIR, quarantine_name)
            
            # Move file to quarantine
            shutil.move(file_path, quarantine_path)
            
            # Log quarantine action
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quarantine_log 
                (timestamp, original_path, quarantine_path, reason, threat_level)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                file_path,
                quarantine_path,
                "Malware/suspicious content detected",
                threat_level
            ))
            conn.commit()
            conn.close()
            
            logger.warning(f"File quarantined: {file_path} -> {quarantine_path}")
            log_filesystem_event(
                "quarantine",
                file_path,
                f"Moved to quarantine: {quarantine_path} (threat level: {threat_level})"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error quarantining file {file_path}: {e}")
            return False
    
    def restore_from_quarantine(self, quarantine_path: str, 
                                restore_path: str = None) -> bool:
        """
        Restore file from quarantine.
        
        Args:
            quarantine_path: Path to quarantined file
            restore_path: Optional path to restore to (uses original if None)
        
        Returns:
            True if restored successfully
        """
        try:
            if not os.path.exists(quarantine_path):
                logger.error(f"Quarantined file not found: {quarantine_path}")
                return False
            
            # Get original path from database if restore_path not provided
            if restore_path is None:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT original_path FROM quarantine_log WHERE quarantine_path = ?",
                    (quarantine_path,)
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    restore_path = row[0]
                else:
                    logger.error("Could not find original path for quarantined file")
                    return False
            
            # Restore file
            os.makedirs(os.path.dirname(restore_path), exist_ok=True)
            shutil.move(quarantine_path, restore_path)
            
            # Update quarantine log
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE quarantine_log 
                SET restored = 1, restore_timestamp = ?
                WHERE quarantine_path = ?
            """, (datetime.now().isoformat(), quarantine_path))
            conn.commit()
            conn.close()
            
            logger.info(f"File restored from quarantine: {restore_path}")
            log_filesystem_event(
                "restore",
                restore_path,
                f"Restored from quarantine: {quarantine_path}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error restoring file from quarantine: {e}")
            return False


# ============================================================================
# ADVANCED BACKUP SYSTEM
# ============================================================================

class BackupManager:
    """
    Comprehensive backup manager with support for:
    - Full backups
    - Incremental backups
    - Differential backups
    - Backup rotation and retention
    - Compression and checksumming
    - Multi-threaded operations
    """
    
    def __init__(self):
        self.backup_history = []
        self.compression_level = BACKUP_COMPRESSION_LEVEL
    
    def generate_backup_filename(self, backup_type: str, prefix: str = "") -> str:
        """Generate unique backup filename with timestamp."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            return f"SarahMemory_{prefix}_{backup_type}_backup_{date_str}.zip"
        else:
            return f"SarahMemory_{backup_type}_backup_{date_str}.zip"
    
    def create_full_backup(self, source_dir: str = None, 
                           destination: str = None) -> Optional[str]:
        """
        Create full backup of specified directory.
        
        Args:
            source_dir: Directory to backup (defaults to BASE_DIR)
            destination: Backup destination path
        
        Returns:
            Path to created backup file, or None on failure
        """
        if source_dir is None:
            source_dir = BASE_DIR
        
        if destination is None:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = self.generate_backup_filename("full", "F")
            destination = os.path.join(BACKUP_DIR, backup_filename)
        
        logger.info(f"Starting full backup: {source_dir} -> {destination}")
        start_time = time.time()
        
        try:
            checksum_map = {}
            file_count = 0
            total_size = 0
            original_size = 0
            
            with zipfile.ZipFile(destination, 'w', 
                                zipfile.ZIP_DEFLATED,
                                compresslevel=self.compression_level) as backup_zip:
                
                # Collect all files
                files_to_backup = []
                for foldername, subdirs, filenames in os.walk(source_dir):
                    # Skip backup directory itself
                    if BACKUP_DIR in foldername:
                        continue
                    
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        files_to_backup.append(file_path)
                
                logger.info(f"Backing up {len(files_to_backup)} files...")
                
                # Backup files with thread pool
                def backup_file(file_path):
                    try:
                        arcname = os.path.relpath(file_path, start=source_dir)
                        backup_zip.write(file_path, arcname)
                        checksum = calculate_checksum(file_path)
                        size = os.path.getsize(file_path)
                        return arcname, checksum, size
                    except Exception as e:
                        logger.error(f"Error backing up {file_path}: {e}")
                        return None, None, 0
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = executor.map(backup_file, files_to_backup)
                    
                    for arcname, checksum, size in results:
                        if arcname:
                            checksum_map[arcname] = checksum
                            original_size += size
                            file_count += 1
                
                # Write checksum manifest
                checksum_data = "\n".join([f"{k}: {v}" for k, v in checksum_map.items()])
                backup_zip.writestr("CHECKSUM_MANIFEST.txt", checksum_data)
                
                # Write backup metadata
                metadata = {
                    "backup_type": "full",
                    "timestamp": datetime.now().isoformat(),
                    "source_directory": source_dir,
                    "file_count": file_count,
                    "original_size_bytes": original_size,
                    "sarah_memory_version": "7.7.5"
                }
                backup_zip.writestr("BACKUP_METADATA.json", json.dumps(metadata, indent=2))
            
            # Get final backup size
            total_size = os.path.getsize(destination)
            compression_ratio = (1 - (total_size / original_size)) * 100 if original_size > 0 else 0
            duration = time.time() - start_time
            
            # Calculate overall backup checksum
            backup_checksum = calculate_checksum(destination)
            
            # Log backup operation
            log_backup_operation(
                "full",
                destination,
                source_dir,
                total_size,
                file_count,
                backup_checksum,
                compression_ratio,
                duration
            )
            
            logger.info(
                f"Full backup completed: {destination}\n"
                f"Files: {file_count}, Size: {total_size / (1024*1024):.2f} MB, "
                f"Compression: {compression_ratio:.1f}%, Duration: {duration:.2f}s"
            )
            
            log_filesystem_event(
                "backup_full",
                destination,
                f"Full backup created with {file_count} files"
            )
            
            return destination
        
        except Exception as e:
            logger.error(f"Error creating full backup: {e}")
            log_backup_operation(
                "full",
                destination,
                source_dir,
                0, 0, "", 0, 0,
                "failed",
                str(e)
            )
            return None
    
    def create_incremental_backup(self, source_dir: str = None,
                                   base_backup: str = None) -> Optional[str]:
        """
        Create incremental backup (only changed files since last backup).
        
        Args:
            source_dir: Directory to backup
            base_backup: Path to last backup (for comparison)
        
        Returns:
            Path to created backup file
        """
        if source_dir is None:
            source_dir = BASE_DIR
        
        logger.info(f"Starting incremental backup: {source_dir}")
        start_time = time.time()
        
        try:
            # Get list of files that changed since last backup
            changed_files = self._find_changed_files(source_dir, base_backup)
            
            if not changed_files:
                logger.info("No files changed since last backup")
                return None
            
            logger.info(f"Found {len(changed_files)} changed files")
            
            # Create incremental backup
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = self.generate_backup_filename("incremental", "I")
            destination = os.path.join(BACKUP_DIR, backup_filename)
            
            checksum_map = {}
            file_count = 0
            total_size = 0
            original_size = 0
            
            with zipfile.ZipFile(destination, 'w',
                                zipfile.ZIP_DEFLATED,
                                compresslevel=self.compression_level) as backup_zip:
                
                for file_path in changed_files:
                    try:
                        arcname = os.path.relpath(file_path, start=source_dir)
                        backup_zip.write(file_path, arcname)
                        checksum = calculate_checksum(file_path)
                        checksum_map[arcname] = checksum
                        size = os.path.getsize(file_path)
                        original_size += size
                        file_count += 1
                    except Exception as e:
                        logger.error(f"Error backing up {file_path}: {e}")
                
                # Write manifests
                checksum_data = "\n".join([f"{k}: {v}" for k, v in checksum_map.items()])
                backup_zip.writestr("CHECKSUM_MANIFEST.txt", checksum_data)
                
                metadata = {
                    "backup_type": "incremental",
                    "timestamp": datetime.now().isoformat(),
                    "source_directory": source_dir,
                    "base_backup": base_backup,
                    "file_count": file_count,
                    "original_size_bytes": original_size
                }
                backup_zip.writestr("BACKUP_METADATA.json", json.dumps(metadata, indent=2))
            
            total_size = os.path.getsize(destination)
            compression_ratio = (1 - (total_size / original_size)) * 100 if original_size > 0 else 0
            duration = time.time() - start_time
            backup_checksum = calculate_checksum(destination)
            
            log_backup_operation(
                "incremental",
                destination,
                source_dir,
                total_size,
                file_count,
                backup_checksum,
                compression_ratio,
                duration
            )
            
            logger.info(
                f"Incremental backup completed: {destination}\n"
                f"Files: {file_count}, Size: {total_size / (1024*1024):.2f} MB, "
                f"Duration: {duration:.2f}s"
            )
            
            return destination
        
        except Exception as e:
            logger.error(f"Error creating incremental backup: {e}")
            return None
    
    def _find_changed_files(self, source_dir: str, 
                            base_backup: str = None) -> List[str]:
        """Find files that have changed since last backup."""
        changed_files = []
        
        try:
            # Get last backup checksums
            last_checksums = {}
            
            if base_backup and os.path.exists(base_backup):
                with zipfile.ZipFile(base_backup, 'r') as z:
                    if "CHECKSUM_MANIFEST.txt" in z.namelist():
                        manifest = z.read("CHECKSUM_MANIFEST.txt").decode('utf-8')
                        for line in manifest.split('\n'):
                            if ': ' in line:
                                path, checksum = line.split(': ', 1)
                                last_checksums[path] = checksum
            
            # Compare current files with last backup
            for root, dirs, files in os.walk(source_dir):
                if BACKUP_DIR in root:
                    continue
                
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, start=source_dir)
                    
                    # Check if file is new or modified
                    current_checksum = calculate_checksum(file_path)
                    
                    if arcname not in last_checksums or \
                       last_checksums[arcname] != current_checksum:
                        changed_files.append(file_path)
        
        except Exception as e:
            logger.error(f"Error finding changed files: {e}")
        
        return changed_files
    
    def restore_backup(self, backup_path: str, 
                       destination: str = None,
                       verify_checksum: bool = True) -> bool:
        """
        Restore backup to specified location.
        
        Args:
            backup_path: Path to backup file
            destination: Where to restore (defaults to original location)
            verify_checksum: Whether to verify file checksums after restore
        
        Returns:
            True if restore successful
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        if not zipfile.is_zipfile(backup_path):
            logger.error(f"Not a valid backup file: {backup_path}")
            return False
        
        logger.info(f"Restoring backup: {backup_path}")
        start_time = time.time()
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as z:
                # Read metadata
                metadata = {}
                if "BACKUP_METADATA.json" in z.namelist():
                    metadata = json.loads(z.read("BACKUP_METADATA.json").decode('utf-8'))
                
                # Read checksums
                checksums = {}
                if "CHECKSUM_MANIFEST.txt" in z.namelist():
                    manifest = z.read("CHECKSUM_MANIFEST.txt").decode('utf-8')
                    for line in manifest.split('\n'):
                        if ': ' in line:
                            path, checksum = line.split(': ', 1)
                            checksums[path] = checksum
                
                # Determine destination
                if destination is None:
                    destination = metadata.get("source_directory", BASE_DIR)
                
                logger.info(f"Restoring to: {destination}")
                
                # Extract all files
                z.extractall(destination)
                
                # Verify checksums if requested
                if verify_checksum and checksums:
                    logger.info("Verifying restored files...")
                    verification_failed = []
                    
                    for arcname, expected_checksum in checksums.items():
                        file_path = os.path.join(destination, arcname)
                        if os.path.exists(file_path):
                            actual_checksum = calculate_checksum(file_path)
                            if actual_checksum != expected_checksum:
                                verification_failed.append(arcname)
                    
                    if verification_failed:
                        logger.error(
                            f"Checksum verification failed for {len(verification_failed)} files"
                        )
                        for path in verification_failed[:10]:  # Show first 10
                            logger.error(f"  - {path}")
                        return False
                    else:
                        logger.info("All files verified successfully")
            
            duration = time.time() - start_time
            logger.info(f"Restore completed in {duration:.2f}s")
            
            log_filesystem_event(
                "backup_restore",
                backup_path,
                f"Backup restored to: {destination}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            log_filesystem_event(
                "backup_restore",
                backup_path,
                f"Restore failed: {str(e)}",
                success=False,
                error_message=str(e)
            )
            return False
    
    def rotate_old_backups(self, max_count: int = MAX_BACKUP_COUNT,
                           max_age_days: int = BACKUP_RETENTION_DAYS):
        """
        Delete old backups based on retention policy.
        
        Args:
            max_count: Maximum number of backups to keep
            max_age_days: Maximum age of backups in days
        """
        try:
            if not os.path.exists(BACKUP_DIR):
                return
            
            # Get all backup files
            backups = []
            for filename in os.listdir(BACKUP_DIR):
                if filename.endswith('.zip'):
                    file_path = os.path.join(BACKUP_DIR, filename)
                    mtime = os.path.getmtime(file_path)
                    backups.append((file_path, mtime))
            
            # Sort by modification time (oldest first)
            backups.sort(key=lambda x: x[1])
            
            # Delete backups exceeding count limit
            if len(backups) > max_count:
                to_delete = backups[:len(backups) - max_count]
                for file_path, _ in to_delete:
                    logger.info(f"Deleting old backup (count limit): {file_path}")
                    os.remove(file_path)
                    log_filesystem_event("backup_delete", file_path, "Deleted by rotation policy")
            
            # Delete backups exceeding age limit
            cutoff_time = time.time() - (max_age_days * 86400)
            for file_path, mtime in backups:
                if mtime < cutoff_time and os.path.exists(file_path):
                    logger.info(f"Deleting old backup (age limit): {file_path}")
                    os.remove(file_path)
                    log_filesystem_event("backup_delete", file_path, "Deleted by age policy")
        
        except Exception as e:
            logger.error(f"Error rotating backups: {e}")
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups with metadata.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        try:
            if not os.path.exists(BACKUP_DIR):
                return backups
            
            for filename in os.listdir(BACKUP_DIR):
                if filename.endswith('.zip'):
                    file_path = os.path.join(BACKUP_DIR, filename)
                    
                    info = {
                        "filename": filename,
                        "path": file_path,
                        "size_bytes": os.path.getsize(file_path),
                        "created": datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat(),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    }
                    
                    # Try to read metadata from backup
                    try:
                        with zipfile.ZipFile(file_path, 'r') as z:
                            if "BACKUP_METADATA.json" in z.namelist():
                                metadata = json.loads(
                                    z.read("BACKUP_METADATA.json").decode('utf-8')
                                )
                                info.update(metadata)
                    except:
                        pass
                    
                    backups.append(info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)
        
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
        
        return backups


# ============================================================================
# FILE OPERATIONS (Move, Copy, Rename, Delete, Attributes)
# ============================================================================

class FileOperations:
    """
    Advanced file operations with logging, verification, and safety checks.
    """
    
    @staticmethod
    def safe_delete(file_path: str, secure: bool = False) -> bool:
        """
        Safely delete file with optional secure deletion.
        
        Args:
            file_path: Path to file to delete
            secure: If True, overwrite file before deletion
        
        Returns:
            True if deletion successful
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            size = os.path.getsize(file_path)
            
            # Secure deletion: overwrite file before deleting
            if secure:
                logger.info(f"Securely deleting file: {file_path}")
                try:
                    with open(file_path, 'wb') as f:
                        f.write(os.urandom(size))
                        f.flush()
                        os.fsync(f.fileno())
                except Exception as e:
                    logger.warning(f"Could not securely overwrite file: {e}")
            
            # Delete file
            os.remove(file_path)
            
            logger.info(f"File deleted: {file_path}")
            
            # Log operation
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_operations 
                (timestamp, operation, source_path, user, size_bytes, success)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "delete",
                file_path,
                os.getenv("USERNAME", "system"),
                size,
                1
            ))
            conn.commit()
            conn.close()
            
            log_filesystem_event("file_delete", file_path, 
                               "Secure deletion" if secure else "Normal deletion")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            log_filesystem_event("file_delete", file_path, 
                               f"Deletion failed: {str(e)}", 
                               success=False, error_message=str(e))
            return False
    
    @staticmethod
    def safe_move(source: str, destination: str, 
                  overwrite: bool = False) -> bool:
        """
        Safely move file with verification.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Whether to overwrite existing file
        
        Returns:
            True if move successful
        """
        try:
            if not os.path.exists(source):
                logger.error(f"Source file not found: {source}")
                return False
            
            if os.path.exists(destination) and not overwrite:
                logger.error(f"Destination already exists: {destination}")
                return False
            
            # Calculate source checksum before move
            source_checksum = calculate_checksum(source)
            source_size = os.path.getsize(source)
            
            # Create destination directory if needed
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Move file
            shutil.move(source, destination)
            
            # Verify destination checksum
            dest_checksum = calculate_checksum(destination)
            
            if source_checksum != dest_checksum:
                logger.error("Checksum mismatch after move operation")
                return False
            
            logger.info(f"File moved: {source} -> {destination}")
            
            # Log operation
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_operations 
                (timestamp, operation, source_path, destination_path, 
                 user, size_bytes, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "move",
                source,
                destination,
                os.getenv("USERNAME", "system"),
                source_size,
                1
            ))
            conn.commit()
            conn.close()
            
            log_filesystem_event("file_move", source, 
                               f"Moved to: {destination}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            log_filesystem_event("file_move", source, 
                               f"Move failed: {str(e)}", 
                               success=False, error_message=str(e))
            return False
    
    @staticmethod
    def safe_copy(source: str, destination: str, 
                  verify: bool = True) -> bool:
        """
        Safely copy file with optional verification.
        
        Args:
            source: Source file path
            destination: Destination file path
            verify: Whether to verify checksums after copy
        
        Returns:
            True if copy successful
        """
        try:
            if not os.path.exists(source):
                logger.error(f"Source file not found: {source}")
                return False
            
            # Calculate source checksum if verification requested
            source_checksum = None
            if verify:
                source_checksum = calculate_checksum(source)
            
            source_size = os.path.getsize(source)
            
            # Create destination directory if needed
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            # Verify if requested
            if verify:
                dest_checksum = calculate_checksum(destination)
                if source_checksum != dest_checksum:
                    logger.error("Checksum mismatch after copy operation")
                    os.remove(destination)
                    return False
            
            logger.info(f"File copied: {source} -> {destination}")
            
            # Log operation
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_operations 
                (timestamp, operation, source_path, destination_path, 
                 user, size_bytes, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "copy",
                source,
                destination,
                os.getenv("USERNAME", "system"),
                source_size,
                1
            ))
            conn.commit()
            conn.close()
            
            log_filesystem_event("file_copy", source, 
                               f"Copied to: {destination}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            log_filesystem_event("file_copy", source, 
                               f"Copy failed: {str(e)}", 
                               success=False, error_message=str(e))
            return False
    
    @staticmethod
    def safe_rename(old_path: str, new_path: str) -> bool:
        """
        Safely rename file.
        
        Args:
            old_path: Current file path
            new_path: New file path
        
        Returns:
            True if rename successful
        """
        try:
            if not os.path.exists(old_path):
                logger.error(f"File not found: {old_path}")
                return False
            
            if os.path.exists(new_path):
                logger.error(f"Destination already exists: {new_path}")
                return False
            
            size = os.path.getsize(old_path)
            
            # Rename file
            os.rename(old_path, new_path)
            
            logger.info(f"File renamed: {old_path} -> {new_path}")
            
            # Log operation
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_operations 
                (timestamp, operation, source_path, destination_path, 
                 user, size_bytes, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "rename",
                old_path,
                new_path,
                os.getenv("USERNAME", "system"),
                size,
                1
            ))
            conn.commit()
            conn.close()
            
            log_filesystem_event("file_rename", old_path, 
                               f"Renamed to: {new_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error renaming file: {e}")
            log_filesystem_event("file_rename", old_path, 
                               f"Rename failed: {str(e)}", 
                               success=False, error_message=str(e))
            return False
    
    @staticmethod
    def set_file_attributes(file_path: str, readonly: bool = None,
                           hidden: bool = None, system: bool = None) -> bool:
        """
        Set file attributes (Windows and Unix compatible).
        
        Args:
            file_path: Path to file
            readonly: Set read-only attribute
            hidden: Set hidden attribute (Windows only)
            system: Set system attribute (Windows only)
        
        Returns:
            True if attributes set successfully
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            current_mode = os.stat(file_path).st_mode
            new_mode = current_mode
            
            # Handle read-only attribute
            if readonly is not None:
                if readonly:
                    # Remove write permissions
                    new_mode = current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                else:
                    # Add write permissions
                    new_mode = current_mode | stat.S_IWUSR
            
            # Set permissions
            if new_mode != current_mode:
                os.chmod(file_path, new_mode)
            
            # Windows-specific attributes
            if sys.platform == 'win32' and (hidden is not None or system is not None):
                try:
                    import ctypes
                    
                    FILE_ATTRIBUTE_HIDDEN = 0x02
                    FILE_ATTRIBUTE_SYSTEM = 0x04
                    
                    attrs = ctypes.windll.kernel32.GetFileAttributesW(file_path)
                    
                    if hidden is not None:
                        if hidden:
                            attrs |= FILE_ATTRIBUTE_HIDDEN
                        else:
                            attrs &= ~FILE_ATTRIBUTE_HIDDEN
                    
                    if system is not None:
                        if system:
                            attrs |= FILE_ATTRIBUTE_SYSTEM
                        else:
                            attrs &= ~FILE_ATTRIBUTE_SYSTEM
                    
                    ctypes.windll.kernel32.SetFileAttributesW(file_path, attrs)
                
                except Exception as e:
                    logger.warning(f"Could not set Windows-specific attributes: {e}")
            
            logger.info(f"File attributes updated: {file_path}")
            log_filesystem_event("file_attributes", file_path, 
                               f"Attributes updated")
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting file attributes: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
        
        Returns:
            Dictionary with file information
        """
        info = {
            "path": file_path,
            "exists": False,
            "size_bytes": 0,
            "created": None,
            "modified": None,
            "accessed": None,
            "is_file": False,
            "is_directory": False,
            "is_readonly": False,
            "extension": "",
            "checksum": ""
        }
        
        try:
            if not os.path.exists(file_path):
                return info
            
            stat_info = os.stat(file_path)
            
            info["exists"] = True
            info["size_bytes"] = stat_info.st_size
            info["created"] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            info["modified"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            info["accessed"] = datetime.fromtimestamp(stat_info.st_atime).isoformat()
            info["is_file"] = os.path.isfile(file_path)
            info["is_directory"] = os.path.isdir(file_path)
            info["is_readonly"] = not (stat_info.st_mode & stat.S_IWUSR)
            
            if os.path.isfile(file_path):
                info["extension"] = os.path.splitext(file_path)[1]
                info["checksum"] = calculate_checksum(file_path)
        
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
        
        return info


# ============================================================================
# MONITORING AND INSPECTION
# ============================================================================

class FileSystemMonitor:
    """
    Real-time filesystem monitoring and inspection.
    Tracks changes, detects suspicious activity, and maintains audit trails.
    """
    
    def __init__(self):
        self.monitored_directories = []
        self.file_cache = {}
        self.monitoring_active = False
        self.monitor_thread = None
    
    def add_monitored_directory(self, directory: str):
        """Add directory to monitoring list."""
        if os.path.exists(directory) and directory not in self.monitored_directories:
            self.monitored_directories.append(directory)
            logger.info(f"Added directory to monitoring: {directory}")
    
    def scan_directory_tree(self, directory: str) -> Dict:
        """
        Scan directory tree and return comprehensive statistics.
        
        Args:
            directory: Root directory to scan
        
        Returns:
            Dictionary with directory statistics
        """
        stats = {
            "total_files": 0,
            "total_directories": 0,
            "total_size_bytes": 0,
            "file_types": defaultdict(int),
            "largest_files": [],
            "oldest_files": [],
            "newest_files": [],
            "suspicious_files": []
        }
        
        try:
            all_files = []
            
            for root, dirs, files in os.walk(directory):
                stats["total_directories"] += len(dirs)
                
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    try:
                        stat_info = os.stat(file_path)
                        size = stat_info.st_size
                        mtime = stat_info.st_mtime
                        ext = os.path.splitext(filename)[1].lower()
                        
                        stats["total_files"] += 1
                        stats["total_size_bytes"] += size
                        stats["file_types"][ext if ext else "no_extension"] += 1
                        
                        all_files.append({
                            "path": file_path,
                            "size": size,
                            "mtime": mtime,
                            "ext": ext
                        })
                        
                        # Check for suspicious files
                        if MalwareSignatures.is_suspicious_extension(filename):
                            stats["suspicious_files"].append(file_path)
                    
                    except Exception as e:
                        logger.warning(f"Could not stat file {file_path}: {e}")
            
            # Find largest files
            all_files.sort(key=lambda x: x["size"], reverse=True)
            stats["largest_files"] = [f["path"] for f in all_files[:10]]
            
            # Find oldest files
            all_files.sort(key=lambda x: x["mtime"])
            stats["oldest_files"] = [f["path"] for f in all_files[:10]]
            
            # Find newest files
            all_files.sort(key=lambda x: x["mtime"], reverse=True)
            stats["newest_files"] = [f["path"] for f in all_files[:10]]
        
        except Exception as e:
            logger.error(f"Error scanning directory tree: {e}")
        
        return stats
    
    def find_duplicate_files(self, directory: str) -> Dict[str, List[str]]:
        """
        Find duplicate files by comparing checksums.
        
        Args:
            directory: Directory to scan for duplicates
        
        Returns:
            Dictionary mapping checksums to lists of duplicate file paths
        """
        duplicates = defaultdict(list)
        checksums = {}
        
        try:
            logger.info(f"Scanning for duplicate files in: {directory}")
            
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    try:
                        checksum = calculate_checksum(file_path)
                        
                        if checksum in checksums:
                            # Found duplicate
                            if checksum not in duplicates:
                                duplicates[checksum].append(checksums[checksum])
                            duplicates[checksum].append(file_path)
                        else:
                            checksums[checksum] = file_path
                    
                    except Exception as e:
                        logger.warning(f"Could not check file {file_path}: {e}")
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} sets of duplicate files")
            else:
                logger.info("No duplicate files found")
        
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
        
        return dict(duplicates)


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def save_code_to_addons(filename, code):
    """
    Legacy function: Save code files to addons directory.
    Maintained for backward compatibility.
    """
    try:
        from SarahMemoryGlobals import ADDONS_DIR
        addons_dir = ADDONS_DIR
    except:
        addons_dir = os.path.join(BASE_DIR, "data", "addons")
    
    os.makedirs(addons_dir, exist_ok=True)
    file_path = os.path.join(addons_dir, filename)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Code saved to addons: {file_path}")
        log_filesystem_event("save_addon", file_path, "Code saved to addons directory")
    except Exception as e:
        logger.error(f"Error saving code to addons: {e}")


def create_settings_backup():
    """Legacy function: Create settings backup."""
    backup_mgr = BackupManager()
    return backup_mgr.create_full_backup(SETTINGS_DIR)


def create_memory_backup():
    """Legacy function: Create memory backup."""
    backup_mgr = BackupManager()
    return backup_mgr.create_full_backup(MEMORY_DIR)


def create_full_backup():
    """Legacy function: Create full system backup."""
    backup_mgr = BackupManager()
    return backup_mgr.create_full_backup()


def restore_backup(zip_path):
    """Legacy function: Restore backup from ZIP file."""
    backup_mgr = BackupManager()
    return backup_mgr.restore_backup(zip_path)


def start_backup_monitor(interval=3600):
    """
    Legacy function: Start automatic backup monitoring.
    Maintained for backward compatibility.
    """
    try:
        from SarahMemoryGlobals import run_async
        
        def backup_loop():
            while True:
                try:
                    create_full_backup()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in backup monitor loop: {e}")
                    time.sleep(60)  # Wait before retrying
        
        run_async(backup_loop)
        logger.info(f"Backup monitor started (interval: {interval}s)")
    
    except Exception as e:
        logger.error(f"Could not start backup monitor: {e}")


def create_weekly_backup():
    """
    Legacy function: Create weekly backup if needed.
    Maintained for backward compatibility.
    """
    now = datetime.now()
    cutoff = now.timestamp() - (7 * 86400)
    
    try:
        # Check for recent backups
        backup_mgr = BackupManager()
        backups = backup_mgr.list_backups()
        
        recent = False
        for backup in backups:
            try:
                backup_time = datetime.fromisoformat(backup["created"]).timestamp()
                if backup_time >= cutoff:
                    logger.info(f"Recent backup exists: {backup['filename']}")
                    recent = True
                    break
            except:
                pass
        
        if not recent:
            logger.info("No recent backup found, creating weekly backup...")
            backup_mgr.create_full_backup()
    
    except Exception as e:
        logger.error(f"Error in weekly backup check: {e}")


# ============================================================================
# INITIALIZATION AND MAIN
# ============================================================================

def initialize():
    """Initialize filesystem management system."""
    try:
        # Create required directories
        for directory in [BACKUP_DIR, QUARANTINE_DIR, LOGS_DIR, DATASETS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize database
        initialize_filesystem_database()
        
        logger.info("SarahMemoryFilesystem initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing filesystem: {e}")


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SarahMemory Filesystem Manager - World-Class Backup and File Management"
    )
    
    parser.add_argument("--init", action="store_true", 
                       help="Initialize filesystem database")
    parser.add_argument("--backup-full", action="store_true",
                       help="Create full backup")
    parser.add_argument("--backup-incremental", action="store_true",
                       help="Create incremental backup")
    parser.add_argument("--restore", type=str, metavar="BACKUP_FILE",
                       help="Restore from backup file")
    parser.add_argument("--scan", type=str, metavar="DIRECTORY",
                       help="Scan directory for malware")
    parser.add_argument("--scan-file", type=str, metavar="FILE",
                       help="Scan single file for malware")
    parser.add_argument("--list-backups", action="store_true",
                       help="List all available backups")
    parser.add_argument("--rotate-backups", action="store_true",
                       help="Rotate old backups based on retention policy")
    parser.add_argument("--find-duplicates", type=str, metavar="DIRECTORY",
                       help="Find duplicate files in directory")
    parser.add_argument("--directory-stats", type=str, metavar="DIRECTORY",
                       help="Get directory statistics")
    
    args = parser.parse_args()
    
    # Initialize system
    initialize()
    
    # Execute requested operations
    if args.init:
        print("Filesystem database initialized")
    
    elif args.backup_full:
        backup_mgr = BackupManager()
        result = backup_mgr.create_full_backup()
        if result:
            print(f"Full backup created: {result}")
        else:
            print("Backup failed")
    
    elif args.backup_incremental:
        backup_mgr = BackupManager()
        result = backup_mgr.create_incremental_backup()
        if result:
            print(f"Incremental backup created: {result}")
        else:
            print("No changes detected or backup failed")
    
    elif args.restore:
        backup_mgr = BackupManager()
        if backup_mgr.restore_backup(args.restore):
            print("Restore completed successfully")
        else:
            print("Restore failed")
    
    elif args.scan:
        scanner = FileScanner()
        results = scanner.scan_directory(args.scan, recursive=True)
        
        threats_found = sum(1 for r in results if r["threats"])
        print(f"\nScan completed:")
        print(f"  Files scanned: {len(results)}")
        print(f"  Threats found: {threats_found}")
        
        if threats_found > 0:
            print("\nThreat details:")
            for result in results:
                if result["threats"]:
                    print(f"\n  {result['file_path']}")
                    print(f"    Threat level: {result['threat_level']}")
                    print(f"    Threats: {', '.join(result['threats'])}")
                    print(f"    Action: {result['action_taken']}")
    
    elif args.scan_file:
        scanner = FileScanner()
        result = scanner.scan_file(args.scan_file)
        
        print(f"\nScan results for: {args.scan_file}")
        print(f"  Threat level: {result['threat_level']}")
        
        if result["threats"]:
            print(f"  Threats found:")
            for threat in result["threats"]:
                print(f"    - {threat}")
            print(f"  Action taken: {result['action_taken']}")
        else:
            print("  No threats detected")
    
    elif args.list_backups:
        backup_mgr = BackupManager()
        backups = backup_mgr.list_backups()
        
        print(f"\nAvailable backups ({len(backups)}):\n")
        for backup in backups:
            size_mb = backup["size_bytes"] / (1024 * 1024)
            print(f"  {backup['filename']}")
            print(f"    Size: {size_mb:.2f} MB")
            print(f"    Created: {backup['created']}")
            if "backup_type" in backup:
                print(f"    Type: {backup['backup_type']}")
            if "file_count" in backup:
                print(f"    Files: {backup['file_count']}")
            print()
    
    elif args.rotate_backups:
        backup_mgr = BackupManager()
        backup_mgr.rotate_old_backups()
        print("Backup rotation completed")
    
    elif args.find_duplicates:
        monitor = FileSystemMonitor()
        duplicates = monitor.find_duplicate_files(args.find_duplicates)
        
        if duplicates:
            print(f"\nFound {len(duplicates)} sets of duplicate files:\n")
            for checksum, paths in duplicates.items():
                print(f"Checksum: {checksum}")
                for path in paths:
                    size = os.path.getsize(path) / (1024 * 1024)
                    print(f"  - {path} ({size:.2f} MB)")
                print()
        else:
            print("No duplicate files found")
    
    elif args.directory_stats:
        monitor = FileSystemMonitor()
        stats = monitor.scan_directory_tree(args.directory_stats)
        
        print(f"\nDirectory statistics for: {args.directory_stats}\n")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total directories: {stats['total_directories']}")
        print(f"  Total size: {stats['total_size_bytes'] / (1024**3):.2f} GB")
        
        print(f"\n  File types:")
        for ext, count in sorted(stats['file_types'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {ext}: {count}")
        
        if stats['suspicious_files']:
            print(f"\n  Suspicious files: {len(stats['suspicious_files'])}")
            for path in stats['suspicious_files'][:10]:
                print(f"    - {path}")
    
    else:
        parser.print_help()
