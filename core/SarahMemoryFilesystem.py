"""--==The SarahMemory Project==--
File: SarahMemoryFilesystem.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com
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

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "filesystem_engine"
# CATEGORY = "filesystem_and_backup"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "filesystem"
# FAMILY = "core_operations"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Core filesystem action layer for backup/restore, file operations, integrity checks, malware scanning, quarantine, audit logging, and retention management."
# --- SARAHMETA END ---

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
        DATASETS_DIR, VAULT_DIR, DIAGNOSTICS_DIR, DATA_DIR,
        DOCUMENTS_DIR, DOWNLOADS_DIR, SANDBOX_DIR
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
    DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
    DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
    SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox")
    GLOBALS_IMPORTED = False
# REM filesystem lane path fail-safes. These preserve SarahMemoryGlobals.py as read-only
# while preventing idle REM scans from failing if a legacy runtime import omits a path.
DOCUMENTS_DIR = globals().get("DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))
DOWNLOADS_DIR = globals().get("DOWNLOADS_DIR", os.path.join(BASE_DIR, "downloads"))
SANDBOX_DIR = globals().get("SANDBOX_DIR", os.path.join(BASE_DIR, "sandbox"))

# ARILE filesystem interlock.  FileSystem remains the enforcement gate; ARILE
# receives compact security/variance packets and protects core immune-system files.
try:
    from SarahMemoryARILE import arile_emit, arile_assert_not_protected_mutation, arile_is_protected_core_file, arile_protected_files
except Exception:  # pragma: no cover - filesystem must survive without ARILE
    arile_emit = None  # type: ignore
    arile_assert_not_protected_mutation = None  # type: ignore
    def arile_is_protected_core_file(path_value):
        try:
            return Path(path_value).name.lower() in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}
        except Exception:
            return str(path_value or "").lower().endswith(("sarahmemoryglobals.py", "sarahmemoryarile.py"))
    def arile_protected_files():
        return ["SarahMemoryGlobals.py", "SarahMemoryARILE.py"]

def _sm_arile_guard_file_mutation(path_value: str, operation: str) -> None:
    if arile_is_protected_core_file(path_value):
        if callable(arile_assert_not_protected_mutation):
            arile_assert_not_protected_mutation(path_value, operation=operation, source="SarahMemoryFilesystem")
        raise PermissionError(f"Protected core file mutation blocked: {path_value}")

# Setup logging
logger = logging.getLogger("SarahMemoryFilesystem")
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Quarantine directory for suspicious files
QUARANTINE_DIR = os.path.join(DATA_DIR, "quarantine")

# Backup configuration
BACKUP_RETENTION_DAYS = 30  # Keep backups for 30 days
MAX_BACKUP_COUNT = 50  # Maximum number of backups to keep
BACKUP_COMPRESSION_LEVEL = int(os.getenv("SARAH_BACKUP_COMPRESSION_LEVEL", "3") or 3)  # Lower CPU/I/O pressure default

# Runtime anti-thrash backup policy.  Backups must not crawl dependency trees,
# active caches, model stores, logs, backup output folders, or volatile runtime
# directories.  These folders are either reproducible, large, constantly
# changing, or unsafe to snapshot while SarahMemory is running from C:\ NVMe.
_BACKUP_EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env", "virtualenv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".vite", ".next", "dist", "build",
    "cache", "caches", ".cache", "runtime", "tmp", "temp",
    "logs", "log", "backup", "backups", "archive", "archives",
    "models", "model_cache", "downloads",
}

_BACKUP_EXCLUDED_FILE_SUFFIXES = {
    ".tmp", ".temp", ".bak", ".lock", ".pid",
    ".log", ".zip", ".7z", ".rar",
    "-wal", "-shm",
}

_BACKUP_MAX_FILE_BYTES = max(1, int(os.getenv("SARAH_BACKUP_MAX_FILE_MB", "256") or 256)) * 1024 * 1024


def _norm_path(path: str) -> str:
    try:
        return os.path.abspath(os.fspath(path))
    except Exception:
        return str(path or "")


def _is_path_under(child: str, parent: str) -> bool:
    try:
        child_abs = os.path.realpath(_norm_path(child))
        parent_abs = os.path.realpath(_norm_path(parent))
        return os.path.commonpath([child_abs, parent_abs]) == parent_abs
    except Exception:
        return False


def _is_backup_excluded_dir(path: str) -> bool:
    name = os.path.basename(_norm_path(path)).lower()
    if name in _BACKUP_EXCLUDED_DIR_NAMES:
        return True
    # BACKUP_DIR may not literally be named "backup" in future layouts.
    if _is_path_under(path, BACKUP_DIR):
        return True
    return False


def _is_backup_excluded_file(file_path: str, destination: str = "") -> Tuple[bool, str]:
    try:
        p = _norm_path(file_path)
        if destination and p == _norm_path(destination):
            return True, "destination_zip"
        base = os.path.basename(p).lower()
        if base in {".env", ".env.local", ".env.production", "secrets.json", "credentials.json", "encryption.key"}:
            return True, "sensitive_secret_file"
        if base.startswith("~$") or base.endswith(".swp") or base.endswith(".part"):
            return True, "temporary_file"
        for suffix in _BACKUP_EXCLUDED_FILE_SUFFIXES:
            if base.endswith(suffix):
                return True, f"excluded_suffix:{suffix}"
        try:
            size = os.path.getsize(p)
            if size > _BACKUP_MAX_FILE_BYTES:
                return True, f"over_size_limit:{size}"
        except Exception:
            return True, "stat_failed"
    except Exception as exc:
        return True, f"exclude_check_failed:{exc}"
    return False, ""


def _collect_backup_files(source_dir: str, destination: str = "") -> Tuple[List[str], Dict[str, int]]:
    files_to_backup: List[str] = []
    skipped: Dict[str, int] = defaultdict(int)
    max_files = 1000
    max_total_bytes = 512 * 1024 * 1024
    start = time.monotonic()
    total_source_bytes = 0
    source_abs = _norm_path(source_dir)

    for foldername, subdirs, filenames in os.walk(source_abs):
        if time.monotonic() - start > 10.0:
            skipped["file_budget_exceeded"] += 1
            break
        # Prune excluded directories in-place so os.walk never descends into them.
        kept_subdirs = []
        for subdir in list(subdirs):
            subdir_path = os.path.join(foldername, subdir)
            if os.path.islink(subdir_path) or not _is_path_under(subdir_path, source_abs):
                skipped["directory_symlink_or_escape"] += 1
                continue
            if _is_backup_excluded_dir(subdir_path):
                skipped[f"dir:{subdir.lower()}"] += 1
            else:
                kept_subdirs.append(subdir)
        subdirs[:] = kept_subdirs

        if _is_backup_excluded_dir(foldername):
            skipped[f"dir:{os.path.basename(foldername).lower()}"] += 1
            continue

        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if os.path.islink(file_path) or not _is_path_under(file_path, source_abs):
                skipped["symlink_or_escape"] += 1
                continue
            if len(files_to_backup) >= max_files:
                skipped["file_budget_exceeded"] += 1
                continue
            excluded, reason = _is_backup_excluded_file(file_path, destination)
            if excluded:
                skipped[reason] += 1
                continue
            try:
                nbytes = os.path.getsize(file_path)
            except OSError:
                skipped["stat_failed"] += 1
                continue
            if total_source_bytes + nbytes > max_total_bytes:
                skipped["file_budget_exceeded"] += 1
                continue
            total_source_bytes += nbytes
            files_to_backup.append(file_path)

    return files_to_backup, skipped


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
         # No example hashes: only externally validated indicators may be added by operator
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
            
            # Heuristic matches are observations, not permission to move user files.
            if result["threats"] and quarantine_on_threat:
                result["action_taken"] = "review_required"
            
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
                    dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        if not os.path.islink(file_path):
                            files_to_scan.append(file_path)
                        if len(files_to_scan) >= 200:
                            break
                    if len(files_to_scan) >= 200:
                        logger.warning("Filesystem scan budget reached (200 files)")
                        break
            else:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        files_to_scan.append(file_path)
            
            files_to_scan = files_to_scan[:200]
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
        """Biometric/malware quarantine changes require authenticated authorization.

            Heuristics may suggest review but must never move or restore files themselves.
            """
        logger.warning("FileScanner.quarantine_file requires OperatorCore action authorization")
        return False
    
    def restore_from_quarantine(self, quarantine_path: str, 
                                restore_path: str = None) -> bool:
        """Biometric/malware quarantine changes require authenticated authorization.

            Heuristics may suggest review but must never move or restore files themselves.
            """
        logger.warning("FileScanner.restore_from_quarantine requires OperatorCore action authorization")
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
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if prefix:
            return f"SarahMemory_{prefix}_{backup_type}_backup_{date_str}.zip"
        else:
            return f"SarahMemory_{backup_type}_backup_{date_str}.zip"
    
    def create_full_backup(self, source_dir: str = None, 
                           destination: str = None) -> Optional[str]:
        """
        Create full backup of specified directory.

        Anti-thrash policy:
        - single-writer ZIP creation only; zipfile is not thread-safe
        - excludes node_modules/.venv/caches/logs/models/backups/build output
        - skips oversized/volatile files by default
        - writes a compact skipped-summary into backup metadata
        """
        if source_dir is None:
            source_dir = BASE_DIR

        source_dir = _norm_path(source_dir)
        if not _is_path_under(source_dir, BASE_DIR) or not os.path.isdir(source_dir):
            logger.error("Backup source outside SarahMemory project root")
            return None

        if destination is None:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = self.generate_backup_filename("full", "F")
            destination = os.path.join(BACKUP_DIR, backup_filename)
        destination = _norm_path(destination)
        if not _is_path_under(destination, BACKUP_DIR) or os.path.islink(destination):
            logger.error("Backup output must remain inside the approved backup directory")
            return None

        logger.info(f"Starting full backup: {source_dir} -> {destination}")
        start_time = time.time()

        try:
            checksum_map: Dict[str, str] = {}
            file_count = 0
            total_size = 0
            original_size = 0
            failed_count = 0

            files_to_backup, skipped = _collect_backup_files(source_dir, destination)
            if skipped.get("file_budget_exceeded"):
                logger.error("Backup budget exceeded; refusing incomplete snapshot")
                return None
            logger.info(
                "Backing up %s files after exclusions; skipped=%s",
                len(files_to_backup),
                dict(list(skipped.items())[:20]),
            )

            os.makedirs(os.path.dirname(destination), exist_ok=True)

            # IMPORTANT: zipfile.ZipFile is not safe for concurrent writes.  The
            # previous threaded writer caused repeated "open writing handle" errors
            # and hammered node_modules.  Keep the ZIP writer single-threaded.
            with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as backup_zip:
                for file_path in files_to_backup:
                    try:
                        arcname = os.path.relpath(file_path, start=source_dir)
                        # SQLite online backup captures committed WAL transactions.
                        if file_path.lower().endswith((".db", ".sqlite", ".sqlite3")):
                            import contextlib
                            import tempfile
                            fd, temporary_db = tempfile.mkstemp(suffix=".sqlite")
                            os.close(fd)
                            try:
                                with contextlib.closing(sqlite3.connect(f"file:{Path(file_path).as_posix()}?mode=ro", uri=True, timeout=5)) as source_db, contextlib.closing(sqlite3.connect(temporary_db)) as staged_db:
                                    source_db.backup(staged_db)
                                backup_zip.write(temporary_db, arcname)
                            finally:
                                os.unlink(temporary_db)
                        else:
                            backup_zip.write(file_path, arcname)
                        import hashlib
                        checksum = hashlib.sha256(backup_zip.read(arcname)).hexdigest()
                        size = backup_zip.getinfo(arcname).file_size
                        checksum_map[arcname] = checksum
                        original_size += size
                        file_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Backup skipped file after write/check failure {file_path}: {e}")

                checksum_data = "\n".join([f"{k}: {v}" for k, v in checksum_map.items()])
                backup_zip.writestr("CHECKSUM_MANIFEST.txt", checksum_data)

                metadata = {
                    "backup_type": "full",
                    "timestamp": datetime.now().isoformat(),
                    "source_directory": source_dir,
                    "file_count": file_count,
                    "failed_count": failed_count,
                    "original_size_bytes": original_size,
                    "excluded_policy": "anti_thrash_runtime_safe",
                    "excluded_summary": dict(skipped),
                    "max_file_bytes": _BACKUP_MAX_FILE_BYTES,
                    "sarah_memory_version": "9.0.0",
                }
                backup_zip.writestr("BACKUP_METADATA.json", json.dumps(metadata, indent=2))

            if failed_count or not file_count or skipped.get("file_budget_exceeded", 0):
                logger.error("Incomplete backup refused: %s failed files", failed_count)
                try:
                    os.remove(destination)
                except OSError:
                    pass
                return None
            total_size = os.path.getsize(destination)
            compression_ratio = (1 - (total_size / original_size)) * 100 if original_size > 0 else 0
            duration = time.time() - start_time
            backup_checksum = calculate_checksum(destination)

            log_backup_operation(
                "full",
                destination,
                source_dir,
                total_size,
                file_count,
                backup_checksum,
                compression_ratio,
                duration,
            )

            logger.info(
                f"Full backup completed: {destination}\n"
                f"Files: {file_count}, Failed: {failed_count}, Size: {total_size / (1024*1024):.2f} MB, "
                f"Compression: {compression_ratio:.1f}%, Duration: {duration:.2f}s"
            )

            log_filesystem_event(
                "backup_full",
                destination,
                f"Full backup created with {file_count} files; skipped={sum(skipped.values())}; failed={failed_count}"
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

        Uses the same anti-thrash exclusions as full backup and writes the ZIP
        sequentially to avoid zipfile concurrent write-handle failures.
        """
        if source_dir is None:
            source_dir = BASE_DIR
        source_dir = _norm_path(source_dir)
        if not _is_path_under(source_dir, BASE_DIR) or not os.path.isdir(source_dir):
            logger.error("Incremental source outside SarahMemory project root")
            return None

        logger.info(f"Starting incremental backup: {source_dir}")
        start_time = time.time()

        try:
            changed_files = self._find_changed_files(source_dir, base_backup)
            # WAL commits can change without modifying the main SQLite file;
            # include databases as fresh online snapshots on every incremental run.
            all_source_files, _limits = _collect_backup_files(source_dir)
            if _limits.get("file_budget_exceeded"):
                logger.error("Source file inventory exceeded safe budget")
                return None
            changed_files = list(dict.fromkeys(changed_files + [f for f in all_source_files if f.lower().endswith((".db", ".sqlite", ".sqlite3"))]))

            if not changed_files:
                logger.info("No files changed since last backup")
                return None

            logger.info(f"Found {len(changed_files)} changed files")

            os.makedirs(BACKUP_DIR, exist_ok=True)
            backup_filename = self.generate_backup_filename("incremental", "I")
            destination = _norm_path(os.path.join(BACKUP_DIR, backup_filename))

            checksum_map: Dict[str, str] = {}
            file_count = 0
            total_size = 0
            original_size = 0
            failed_count = 0
            skipped: Dict[str, int] = defaultdict(int)

            with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as backup_zip:
                for file_path in changed_files:
                    excluded, reason = _is_backup_excluded_file(file_path, destination)
                    if excluded:
                        skipped[reason] += 1
                        continue
                    try:
                        arcname = os.path.relpath(file_path, start=source_dir)
                        if file_path.lower().endswith((".db", ".sqlite", ".sqlite3")):
                            import contextlib
                            import tempfile
                            fd, stage = tempfile.mkstemp(suffix=".sqlite")
                            os.close(fd)
                            try:
                                with contextlib.closing(sqlite3.connect(f"file:{Path(file_path).as_posix()}?mode=ro", uri=True, timeout=5)) as source_db, contextlib.closing(sqlite3.connect(stage)) as staged_db:
                                    source_db.backup(staged_db)
                                backup_zip.write(stage, arcname)
                            finally:
                                os.unlink(stage)
                        else:
                            backup_zip.write(file_path, arcname)
                        checksum = hashlib.sha256(backup_zip.read(arcname)).hexdigest()
                        checksum_map[arcname] = checksum
                        size = backup_zip.getinfo(arcname).file_size
                        original_size += size
                        file_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Backup skipped changed file after write/check failure {file_path}: {e}")

                checksum_data = "\n".join([f"{k}: {v}" for k, v in checksum_map.items()])
                backup_zip.writestr("CHECKSUM_MANIFEST.txt", checksum_data)

                metadata = {
                    "backup_type": "incremental",
                    "timestamp": datetime.now().isoformat(),
                    "source_directory": source_dir,
                    "base_backup": base_backup,
                    "file_count": file_count,
                    "failed_count": failed_count,
                    "original_size_bytes": original_size,
                    "excluded_policy": "anti_thrash_runtime_safe",
                    "excluded_summary": dict(skipped),
                    "max_file_bytes": _BACKUP_MAX_FILE_BYTES,
                }
                backup_zip.writestr("BACKUP_METADATA.json", json.dumps(metadata, indent=2))

            if failed_count or not file_count or skipped.get("file_budget_exceeded"):
                logger.error("Incomplete incremental backup refused")
                try:
                    os.unlink(destination)
                except OSError:
                    pass
                return None
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
                duration,
            )

            logger.info(
                f"Incremental backup completed: {destination}\n"
                f"Files: {file_count}, Failed: {failed_count}, Size: {total_size / (1024*1024):.2f} MB, "
                f"Duration: {duration:.2f}s"
            )

            return destination

        except Exception as e:
            logger.error(f"Error creating incremental backup: {e}")
            return None

    def _find_changed_files(self, source_dir: str,
                            base_backup: str = None) -> List[str]:
        """Find files that have changed since last backup, excluding volatile trees."""
        changed_files: List[str] = []

        try:
            last_checksums: Dict[str, str] = {}

            if base_backup and os.path.exists(base_backup):
                with zipfile.ZipFile(base_backup, 'r') as z:
                    if "CHECKSUM_MANIFEST.txt" in z.namelist():
                        manifest = z.read("CHECKSUM_MANIFEST.txt").decode('utf-8')
                        for line in manifest.split('\n'):
                            if ': ' in line:
                                path, checksum = line.split(': ', 1)
                                last_checksums[path] = checksum

            files_to_check, skipped = _collect_backup_files(source_dir)
            if skipped:
                logger.debug("Incremental backup skipped volatile paths: %s", dict(list(skipped.items())[:20]))

            for file_path in files_to_check:
                try:
                    arcname = os.path.relpath(file_path, start=source_dir)
                    current_checksum = calculate_checksum(file_path)
                    if arcname not in last_checksums or last_checksums[arcname] != current_checksum:
                        changed_files.append(file_path)
                except Exception as exc:
                    logger.warning(f"Incremental backup skipped unreadable file {file_path}: {exc}")

        except Exception as e:
            logger.error(f"Error finding changed files: {e}")

        return changed_files


    def restore_backup(self, backup_path: str, destination: str = None, verify_checksum: bool = True) -> bool:
        """Restore a local backup only after full ZIP verification and staging.

        This does not certify the ZIP's publisher: untrusted external archives must
        not be placed in the operator-controlled BACKUP_DIR. Existing destinations
        require separate OperatorCore authorization and are not overwritten here.
        """
        import tempfile
        import posixpath
        import uuid
        if not isinstance(backup_path, str) or not isinstance(destination, str) or not destination:
            logger.error("Restore requires an explicit sandbox destination and a verified local archive")
            return False
        source = os.path.realpath(os.path.abspath(backup_path))
        backup_root = os.path.realpath(os.path.abspath(BACKUP_DIR))
        dest = os.path.realpath(os.path.abspath(destination))
        if (not _is_path_under(source, backup_root) or os.path.islink(backup_path)
                or not os.path.isfile(source) or not zipfile.is_zipfile(source)
                or os.path.islink(destination)):
            return False
        # This legacy restore interface cannot prove OperatorCore approval for
        # replacing active data. Stage new files only inside governed SANDBOX_DIR.
        if not _is_path_under(dest, SANDBOX_DIR) or not os.path.isdir(dest):
            return False
        temp_root = None
        committed = []
        try:
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                if len(names) > 5002 or len(names) != len(set(names)):
                    return False
                if "CHECKSUM_MANIFEST.txt" not in names or "BACKUP_METADATA.json" not in names:
                    return False
                if archive.getinfo("CHECKSUM_MANIFEST.txt").file_size > 1024 * 1024:
                    return False
                manifest_bytes = archive.read("CHECKSUM_MANIFEST.txt")
                if len(manifest_bytes) > 1024 * 1024:
                    return False
                checksums = {}
                for line in manifest_bytes.decode("utf-8").splitlines():
                    if ": " not in line:
                        return False
                    key, sha = line.rsplit(": ", 1)
                    if key in checksums or not re.fullmatch(r"[0-9a-f]{64}", sha):
                        return False
                    checksums[key] = sha
                entries = [entry for entry in archive.infolist() if entry.filename not in ("CHECKSUM_MANIFEST.txt", "BACKUP_METADATA.json")]
                if len(checksums) != len(entries) or sum(x.file_size for x in entries) > 2 * 1024**3:
                    return False
                for member in entries:
                    key = member.filename
                    parts = key.replace("\\", "/").split("/")
                    if (member.is_dir() or not key or key.startswith(("/", "\\"))
                            or any(part in ("", ".", "..") for part in parts)
                            or ":" in parts[0] or key not in checksums
                            or (member.external_attr >> 16) & 0o170000 == 0o120000
                            or member.file_size > _BACKUP_MAX_FILE_BYTES):
                        return False
                    target = os.path.join(dest, *parts)
                    if not _is_path_under(target, dest) or os.path.lexists(target):
                        # Legacy signature provides no permission to replace an existing file.
                        return False
                    if any(os.path.islink(os.path.join(dest, *parts[:i])) for i in range(1, len(parts))):
                        return False
                temp_root = tempfile.mkdtemp(prefix=".sm_restore_", dir=dest)
                # Stage and hash all bytes before promoting even one destination.
                for member in entries:
                    import hashlib
                    out = os.path.join(temp_root, *member.filename.replace("\\", "/").split("/"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(member) as reader, open(out, "xb") as writer:
                        while True:
                            chunk = reader.read(1024 * 1024)
                            if not chunk:
                                break
                            digest.update(chunk)
                            writer.write(chunk)
                    if digest.hexdigest() != checksums[member.filename]:
                        return False
                for member in entries:
                    target = os.path.join(dest, *member.filename.replace("\\", "/").split("/"))
                    if os.path.lexists(target):
                        raise FileExistsError(target)
                    staged = os.path.join(temp_root, *member.filename.replace("\\", "/").split("/"))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    # Exclusive creation prevents accidental overwrite, including races.
                    with open(staged, "rb") as reader, open(target, "xb") as writer:
                        shutil.copyfileobj(reader, writer, 1024 * 1024)
                        writer.flush()
                        os.fsync(writer.fileno())
                    committed.append(target)
            log_filesystem_event("backup_restore", source, "verified new-file restore completed")
            return True
        except Exception as exc:
            logger.error("Restore failed, reverting new files: %s", exc)
            for target in reversed(committed):
                try:
                    os.remove(target)
                except OSError:
                    logger.critical("Restore rollback incomplete for %s", target)
            return False
        finally:
            if temp_root:
                shutil.rmtree(temp_root, ignore_errors=True)
    

    def rotate_old_backups(self, max_count: int = MAX_BACKUP_COUNT, max_age_days: int = BACKUP_RETENTION_DAYS):
        """Read-only retention inventory; automatic deletion requires OperatorCore."""
        logger.info("Backup rotation is review-only; no authenticated deletion receipt provided")
        return None
    
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
        """Governed state-changing operation unavailable without an authenticated receipt.

            The original signature has no trustworthy approval parameter; a caller
            boolean, environment flag or model text must not grant mutation rights.
            """
        logger.warning("FileOperations.safe_delete held for OperatorCore execution approval")
        return False
    
    @staticmethod
    def safe_move(source: str, destination: str, 
                  overwrite: bool = False) -> bool:
        """Governed state-changing operation unavailable without an authenticated receipt.

            The original signature has no trustworthy approval parameter; a caller
            boolean, environment flag or model text must not grant mutation rights.
            """
        logger.warning("FileOperations.safe_move held for OperatorCore execution approval")
        return False
    
    @staticmethod
    def safe_copy(source: str, destination: str, 
                  verify: bool = True) -> bool:
        """Governed state-changing operation unavailable without an authenticated receipt.

            The original signature has no trustworthy approval parameter; a caller
            boolean, environment flag or model text must not grant mutation rights.
            """
        logger.warning("FileOperations.safe_copy held for OperatorCore execution approval")
        return False
    
    @staticmethod
    def safe_rename(old_path: str, new_path: str) -> bool:
        """Governed state-changing operation unavailable without an authenticated receipt.

            The original signature has no trustworthy approval parameter; a caller
            boolean, environment flag or model text must not grant mutation rights.
            """
        logger.warning("FileOperations.safe_rename held for OperatorCore execution approval")
        return False
    
    @staticmethod
    def set_file_attributes(file_path: str, readonly: bool = None,
                           hidden: bool = None, system: bool = None) -> bool:
        """Governed state-changing operation unavailable without an authenticated receipt.

            The original signature has no trustworthy approval parameter; a caller
            boolean, environment flag or model text must not grant mutation rights.
            """
        logger.warning("FileOperations.set_file_attributes held for OperatorCore execution approval")
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

    Runtime optimization policy: disabled unless explicitly enabled with
    SARAH_BACKUP_MONITOR_ENABLED=1. Backups are heavy disk writers and should
    not run silently on an NVMe-only host.
    """
    if str(os.getenv("SARAH_BACKUP_MONITOR_ENABLED", "0")).strip().lower() not in ("1", "true", "yes", "on"):
        logger.info("Backup monitor skipped by optimized runtime policy (SARAH_BACKUP_MONITOR_ENABLED not enabled).")
        return None
    try:
        from SarahMemoryGlobals import run_async

        def backup_loop():
            while True:
                try:
                    create_full_backup()
                    time.sleep(max(3600, int(interval or 3600)))
                except Exception as e:
                    logger.error(f"Error in backup monitor loop: {e}")
                    time.sleep(300)

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

# =============================================================================
# SARAH_REM_FILESYSTEM_LANE_V1
# Read-only REM filesystem study. No deletes, moves, writes, uploads, executions,
# or SarahMemoryGlobals.py patch/stage operations.
# =============================================================================

def _rem_is_protected_path(path: str) -> bool:
    try:
        name = os.path.basename(path or "").lower()
        if name in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}:
            return True
        parts = {p.lower() for p in Path(path).parts}
        blocked_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "backup", "backups", "quarantine"}
        return bool(parts & blocked_dirs)
    except Exception:
        return True


def rem_filesystem_study_tick(root_paths: Optional[List[str]] = None, max_files: int = 300, max_read_chars: int = 4096) -> Dict[str, Any]:
    """Study local files in read-only mode for REM awareness."""
    started = time.time()
    roots = list(root_paths or [])
    if not roots:
        env_roots = os.getenv("SARAH_REM_USER_SCAN_ROOTS", "").strip()
        if env_roots:
            roots = [x.strip() for x in env_roots.split(os.pathsep) if x.strip()]
    if not roots:
        roots = [DOCUMENTS_DIR, os.path.join(DATA_DIR, "documents"), os.path.join(DATA_DIR, "projects")]

    text_exts = {".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".css", ".js", ".ts", ".tsx"}
    doc_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
    summary: Dict[str, Any] = {
        "ok": True,
        "lane": "filesystem",
        "read_only": True,
        "writes": False,
        "uploads": False,
        "executions": False,
        "protected_files": arile_protected_files(),
        "roots": roots,
        "counts": {"seen": 0, "text": 0, "documents": 0, "images": 0, "other": 0, "skipped": 0},
        "samples": [],
        "questions": [],
        "duration_ms": 0,
    }
    try:
        from PIL import Image as _PILImage  # type: ignore
    except Exception:
        _PILImage = None

    for root in roots:
        if summary["counts"]["seen"] >= max_files:
            break
        try:
            root_path = Path(root).expanduser().resolve()
        except Exception:
            summary["counts"]["skipped"] += 1
            continue
        if not root_path.exists() or _rem_is_protected_path(str(root_path)):
            summary["counts"]["skipped"] += 1
            continue
        iterator = root_path.rglob("*") if root_path.is_dir() else [root_path]
        for p in iterator:
            if summary["counts"]["seen"] >= max_files:
                break
            try:
                if not p.is_file() or _rem_is_protected_path(str(p)):
                    summary["counts"]["skipped"] += 1
                    continue
                ext = p.suffix.lower()
                rec: Dict[str, Any] = {"path": str(p), "name": p.name, "ext": ext, "size_bytes": int(p.stat().st_size)}
                if ext in text_exts:
                    summary["counts"]["text"] += 1
                    try:
                        rec["text_preview"] = p.read_text(encoding="utf-8", errors="ignore")[:max_read_chars]
                    except Exception:
                        rec["text_preview"] = ""
                elif ext in doc_exts:
                    summary["counts"]["documents"] += 1
                    rec["document_policy"] = "metadata_only_until_document_parser_lane"
                elif ext in image_exts:
                    summary["counts"]["images"] += 1
                    rec["image_policy"] = "local_metadata_only_no_upload"
                    if _PILImage is not None:
                        try:
                            with _PILImage.open(str(p)) as img:
                                rec["image_size"] = list(img.size)
                                rec["image_mode"] = img.mode
                        except Exception:
                            pass
                else:
                    summary["counts"]["other"] += 1
                summary["counts"]["seen"] += 1
                if len(summary["samples"]) < 30:
                    summary["samples"].append(rec)
            except Exception:
                summary["counts"]["skipped"] += 1
                continue
    if summary["counts"]["images"]:
        summary["questions"].append("I found local images. User confirmation is required before labeling people or identities.")
    if summary["counts"]["documents"]:
        summary["questions"].append("I found document files. Deeper parsing should remain read-only and bounded.")
    try:
        log_filesystem_event("rem_filesystem_study", details=json.dumps(summary.get("counts", {}), ensure_ascii=False), success=True)
    except Exception:
        pass
    summary["duration_ms"] = int((time.time() - started) * 1000)
    return summary

# ====================================================================
# END OF SarahMemoryFilesystem.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryFilesystem',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Input',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['filesystem', 'input'],
    "supported_missions": ['Conversation', 'Filesystem'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω004'],
    "required_authority": ['Filesystem', 'Read'],
    "priority": 60,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryFilesystem.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return __import__("copy").deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Report only observed SML registration state; no fabricated performance scores."""
    return {
        "status": "unverified_runtime", "availability": None,
        "integrity": None, "performance": None, "reliability": None,
        "confidence": 0.0, "latency_ms": None, "stability": None,
        "compatibility": None, "notes": ["SML registration present; runtime health not measured"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryFilesystem',
        "sml_adapter": True,
        "metadata": __import__("copy").deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryFilesystem', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

