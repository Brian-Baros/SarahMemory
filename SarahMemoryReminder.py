"""--==The SarahMemory Project==--
File: SarahMemoryReminder.py
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

CALENDAR, DATE/TIME PLANNER & SCHEDULER MODULE
============================================================
This module provides comprehensive calendar and scheduling functionality for
the SarahMemory AiOS platform, supporting:

- Full Calendar Management (day, week, month, year views)
- Event/Appointment Scheduling with recurrence patterns
- Task Management with priorities and due dates
- Reminder System with multiple notification types
- Time Zone Support for global operations
- Natural Language Date/Time Parsing
- Calendar Sync capabilities (iCal, Google Calendar format)
- Smart Scheduling Suggestions
- Conflict Detection and Resolution
- Holiday/Special Date Awareness
- Contact Integration for event attendees
- WebUI API Integration
- Cross-platform compatibility (Windows, Linux, Headless)

===============================================================================
"""

import logging
import datetime
import time
import os
import sys
import sqlite3
import json
import uuid
import hashlib
import threading
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import calendar as cal_module

# Import SarahMemory configuration
try:
    import SarahMemoryGlobals as config
    DATASETS_DIR = config.DATASETS_DIR
    BASE_DIR = config.BASE_DIR
    SETTINGS_DIR = config.SETTINGS_DIR
    PROJECT_VERSION = config.PROJECT_VERSION
except ImportError:
    # Fallback for standalone testing
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR = os.path.join(BASE_DIR, "data", "memory", "datasets")
    SETTINGS_DIR = os.path.join(BASE_DIR, "data", "settings")
    PROJECT_VERSION = "7.7.5"
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(SETTINGS_DIR, exist_ok=True)

# Optional APScheduler for background scheduling
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None

# Optional encryption module
try:
    from SarahMemoryEncryption import encrypt_data, decrypt_data
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    def encrypt_data(data): return data
    def decrypt_data(data): return data

# Optional dateutil for advanced date parsing
try:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta
    from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, YEARLY
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    date_parser = None
    relativedelta = None

# Optional pytz for timezone support
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    pytz = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger('SarahMemoryReminder')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

# Database path
REMINDER_DB = os.path.join(DATASETS_DIR, 'reminders.db')
CALENDAR_DB = os.path.join(DATASETS_DIR, 'calendar.db')

# Default timezone (America/Phoenix for Brian's location)
DEFAULT_TIMEZONE = os.getenv("SARAH_TIMEZONE", "America/Phoenix")


class EventType(Enum):
    """Types of calendar events"""
    REMINDER = "reminder"
    APPOINTMENT = "appointment"
    MEETING = "meeting"
    TASK = "task"
    DEADLINE = "deadline"
    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    HOLIDAY = "holiday"
    RECURRING = "recurring"
    ALL_DAY = "all_day"
    CUSTOM = "custom"


class Priority(IntEnum):
    """Priority levels for tasks and events"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    NONE = 5


class RecurrencePattern(Enum):
    """Recurrence patterns for recurring events"""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    WEEKDAYS = "weekdays"
    WEEKENDS = "weekends"
    CUSTOM = "custom"


class ReminderType(Enum):
    """Types of reminder notifications"""
    POPUP = "popup"
    SOUND = "sound"
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    PUSH = "push"
    SILENT = "silent"


class EventStatus(Enum):
    """Status of events and tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SNOOZED = "snoozed"
    MISSED = "missed"
    DELEGATED = "delegated"


class CalendarView(Enum):
    """Calendar view modes"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    AGENDA = "agenda"
    LIST = "list"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalendarEvent:
    """Comprehensive calendar event data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    location: str = ""
    event_type: EventType = EventType.REMINDER
    
    # Date/Time fields
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    all_day: bool = False
    timezone: str = DEFAULT_TIMEZONE
    
    # Recurrence
    recurrence: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end: Optional[datetime.datetime] = None
    recurrence_count: int = 0
    recurrence_interval: int = 1
    recurrence_days: List[int] = field(default_factory=list)
    parent_event_id: Optional[str] = None
    
    # Priority and status
    priority: Priority = Priority.MEDIUM
    status: EventStatus = EventStatus.PENDING
    
    # Reminders
    reminder_minutes: List[int] = field(default_factory=lambda: [15])
    reminder_type: ReminderType = ReminderType.POPUP
    reminder_sent: bool = False
    
    # Contacts/Attendees
    attendees: List[str] = field(default_factory=list)
    organizer: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    color: str = "#4285F4"
    notes: str = ""
    attachments: List[str] = field(default_factory=list)
    url: str = ""
    
    # System fields
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    completed_at: Optional[datetime.datetime] = None
    user_id: str = "default"
    synced: bool = False
    external_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['recurrence'] = self.recurrence.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['reminder_type'] = self.reminder_type.value
        for key in ['start_time', 'end_time', 'recurrence_end', 'created_at', 'updated_at', 'completed_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime.datetime) else data[key]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalendarEvent':
        """Create event from dictionary"""
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = EventType(data['event_type'])
        if 'recurrence' in data and isinstance(data['recurrence'], str):
            data['recurrence'] = RecurrencePattern(data['recurrence'])
        if 'priority' in data and isinstance(data['priority'], (str, int)):
            data['priority'] = Priority(int(data['priority'])) if isinstance(data['priority'], str) else Priority(data['priority'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = EventStatus(data['status'])
        if 'reminder_type' in data and isinstance(data['reminder_type'], str):
            data['reminder_type'] = ReminderType(data['reminder_type'])
        for key in ['start_time', 'end_time', 'recurrence_end', 'created_at', 'updated_at', 'completed_at']:
            if key in data and data[key] is not None and isinstance(data[key], str):
                try:
                    data[key] = datetime.datetime.fromisoformat(data[key])
                except (ValueError, TypeError):
                    data[key] = None
        return cls(**data)


@dataclass
class Task:
    """Task data structure with due dates and completion tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    due_date: Optional[datetime.datetime] = None
    priority: Priority = Priority.MEDIUM
    status: EventStatus = EventStatus.PENDING
    progress: int = 0
    estimated_minutes: int = 0
    actual_minutes: int = 0
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    assigned_to: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    completed_at: Optional[datetime.datetime] = None
    user_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        for key in ['due_date', 'created_at', 'updated_at', 'completed_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime.datetime) else data[key]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        if 'priority' in data and isinstance(data['priority'], (str, int)):
            data['priority'] = Priority(int(data['priority'])) if isinstance(data['priority'], str) else Priority(data['priority'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = EventStatus(data['status'])
        for key in ['due_date', 'created_at', 'updated_at', 'completed_at']:
            if key in data and data[key] is not None and isinstance(data[key], str):
                try:
                    data[key] = datetime.datetime.fromisoformat(data[key])
                except (ValueError, TypeError):
                    data[key] = None
        return cls(**data)


# =============================================================================
# HOLIDAY DATA
# =============================================================================

US_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (7, 4): "Independence Day",
    (11, 11): "Veterans Day",
    (12, 25): "Christmas Day",
}


def get_us_holidays(year: int) -> Dict[datetime.date, str]:
    """Get all US holidays for a given year"""
    holidays = {}
    for (month, day), name in US_HOLIDAYS.items():
        holidays[datetime.date(year, month, day)] = name
    holidays[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    holidays[_nth_weekday(year, 2, 0, 3)] = "Presidents' Day"
    holidays[_last_weekday(year, 5, 0)] = "Memorial Day"
    holidays[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    holidays[_nth_weekday(year, 10, 0, 2)] = "Columbus Day"
    holidays[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Get the nth occurrence of a weekday in a month"""
    first = datetime.date(year, month, 1)
    first_weekday = first.weekday()
    days_until = (weekday - first_weekday) % 7
    return first + datetime.timedelta(days=days_until + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
    """Get the last occurrence of a weekday in a month"""
    last_day = datetime.date(year, month, cal_module.monthrange(year, month)[1])
    days_since = (last_day.weekday() - weekday) % 7
    return last_day - datetime.timedelta(days=days_since)


# =============================================================================
# DATABASE MANAGER CLASS
# =============================================================================

class CalendarDatabase:
    """Database manager for calendar and reminder data."""
    
    def __init__(self, db_path: str = CALENDAR_DB):
        """Initialize the calendar database"""
        self.db_path = db_path
        self._ensure_directory()
        self._init_db()
        self._lock = threading.RLock()
        logger.info(f"[CALENDAR DB] Initialized at: {self.db_path}")
    
    def _ensure_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize all database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                location TEXT DEFAULT '',
                event_type TEXT DEFAULT 'reminder',
                start_time TEXT,
                end_time TEXT,
                all_day INTEGER DEFAULT 0,
                timezone TEXT DEFAULT 'America/Phoenix',
                recurrence TEXT DEFAULT 'none',
                recurrence_end TEXT,
                recurrence_count INTEGER DEFAULT 0,
                recurrence_interval INTEGER DEFAULT 1,
                recurrence_days TEXT DEFAULT '[]',
                parent_event_id TEXT,
                priority INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                reminder_minutes TEXT DEFAULT '[15]',
                reminder_type TEXT DEFAULT 'popup',
                reminder_sent INTEGER DEFAULT 0,
                attendees TEXT DEFAULT '[]',
                organizer TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                color TEXT DEFAULT '#4285F4',
                notes TEXT DEFAULT '',
                attachments TEXT DEFAULT '[]',
                url TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT,
                completed_at TEXT,
                user_id TEXT DEFAULT 'default',
                synced INTEGER DEFAULT 0,
                external_id TEXT DEFAULT ''
            )
        """)
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                due_date TEXT,
                priority INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                estimated_minutes INTEGER DEFAULT 0,
                actual_minutes INTEGER DEFAULT 0,
                parent_task_id TEXT,
                subtasks TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                assigned_to TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT,
                completed_at TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Reminders table (legacy support)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                title TEXT,
                message TEXT,
                remind_time TEXT,
                note TEXT DEFAULT '',
                created_at TEXT,
                completed INTEGER DEFAULT 0,
                snoozed_until TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                address TEXT DEFAULT '',
                birthday TEXT,
                notes TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                created_at TEXT,
                updated_at TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Passwords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS passwords (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                encrypted_value TEXT,
                url TEXT DEFAULT '',
                username TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Webpages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS webpages (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                favicon TEXT DEFAULT '',
                saved_on TEXT,
                last_visited TEXT,
                visit_count INTEGER DEFAULT 0,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Event logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                event_id TEXT,
                action TEXT,
                timestamp TEXT,
                details TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Calendar settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calendar_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                user_id TEXT DEFAULT 'default'
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders(remind_time)")
        
        conn.commit()
        conn.close()
        logger.debug("[CALENDAR DB] Tables initialized successfully")
    
    def log_event(self, event_type: str, event_id: str, action: str, details: str = "", user_id: str = "default"):
        """Log an event action for audit trail"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO event_logs (event_type, event_id, action, timestamp, details, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type, event_id, action, datetime.datetime.now().isoformat(), details, user_id))
            conn.commit()
            conn.close()


# =============================================================================
# CALENDAR MANAGER CLASS
# =============================================================================

class CalendarManager:
    """Main calendar and scheduling manager for SarahMemory."""
    
    def __init__(self, db_path: str = CALENDAR_DB):
        """Initialize the calendar manager"""
        self.db = CalendarDatabase(db_path)
        self.scheduler = None
        self._reminder_callbacks = []
        self._initialized = False
        self._scheduler_lock = threading.Lock()
        
        if APSCHEDULER_AVAILABLE:
            self._init_scheduler()
        
        logger.info("[CALENDAR] Manager initialized")
    
    def _init_scheduler(self):
        """Initialize the background scheduler"""
        if not APSCHEDULER_AVAILABLE:
            logger.warning("[CALENDAR] APScheduler not available")
            return
        
        with self._scheduler_lock:
            if self.scheduler is None:
                try:
                    self.scheduler = BackgroundScheduler()
                    self.scheduler.start()
                    self._initialized = True
                    logger.info("[CALENDAR] Background scheduler started")
                except Exception as e:
                    logger.error(f"[CALENDAR] Failed to start scheduler: {e}")
    
    # -------------------------------------------------------------------------
    # EVENT MANAGEMENT
    # -------------------------------------------------------------------------
    
    def create_event(self, event: CalendarEvent) -> str:
        """Create a new calendar event"""
        event.updated_at = datetime.datetime.now()
        if event.created_at is None:
            event.created_at = datetime.datetime.now()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO events (
                id, title, description, location, event_type,
                start_time, end_time, all_day, timezone,
                recurrence, recurrence_end, recurrence_count, recurrence_interval, recurrence_days,
                parent_event_id, priority, status,
                reminder_minutes, reminder_type, reminder_sent,
                attendees, organizer, tags, color, notes, attachments, url,
                created_at, updated_at, completed_at,
                user_id, synced, external_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id, event.title, event.description, event.location, event.event_type.value,
            event.start_time.isoformat() if event.start_time else None,
            event.end_time.isoformat() if event.end_time else None,
            1 if event.all_day else 0, event.timezone,
            event.recurrence.value,
            event.recurrence_end.isoformat() if event.recurrence_end else None,
            event.recurrence_count, event.recurrence_interval,
            json.dumps(event.recurrence_days),
            event.parent_event_id, event.priority.value, event.status.value,
            json.dumps(event.reminder_minutes), event.reminder_type.value, 1 if event.reminder_sent else 0,
            json.dumps(event.attendees), event.organizer,
            json.dumps(event.tags), event.color, event.notes,
            json.dumps(event.attachments), event.url,
            event.created_at.isoformat() if event.created_at else None,
            event.updated_at.isoformat() if event.updated_at else None,
            event.completed_at.isoformat() if event.completed_at else None,
            event.user_id, 1 if event.synced else 0, event.external_id
        ))
        
        conn.commit()
        conn.close()
        
        if event.start_time and event.reminder_minutes:
            self._schedule_event_reminders(event)
        
        self.db.log_event("event", event.id, "created", f"Created event: {event.title}")
        logger.info(f"[CALENDAR] Created event: {event.id} - {event.title}")
        
        return event.id
    
    def get_event(self, event_id: str) -> Optional[CalendarEvent]:
        """Get a single event by ID"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_event(dict(row))
        return None
    
    def update_event(self, event: CalendarEvent) -> bool:
        """Update an existing event"""
        event.updated_at = datetime.datetime.now()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE events SET
                title = ?, description = ?, location = ?, event_type = ?,
                start_time = ?, end_time = ?, all_day = ?, timezone = ?,
                recurrence = ?, recurrence_end = ?, recurrence_count = ?, 
                recurrence_interval = ?, recurrence_days = ?,
                parent_event_id = ?, priority = ?, status = ?,
                reminder_minutes = ?, reminder_type = ?, reminder_sent = ?,
                attendees = ?, organizer = ?, tags = ?, color = ?, 
                notes = ?, attachments = ?, url = ?,
                updated_at = ?, completed_at = ?,
                user_id = ?, synced = ?, external_id = ?
            WHERE id = ?
        """, (
            event.title, event.description, event.location, event.event_type.value,
            event.start_time.isoformat() if event.start_time else None,
            event.end_time.isoformat() if event.end_time else None,
            1 if event.all_day else 0, event.timezone,
            event.recurrence.value,
            event.recurrence_end.isoformat() if event.recurrence_end else None,
            event.recurrence_count, event.recurrence_interval,
            json.dumps(event.recurrence_days),
            event.parent_event_id, event.priority.value, event.status.value,
            json.dumps(event.reminder_minutes), event.reminder_type.value, 1 if event.reminder_sent else 0,
            json.dumps(event.attendees), event.organizer,
            json.dumps(event.tags), event.color, event.notes,
            json.dumps(event.attachments), event.url,
            event.updated_at.isoformat(),
            event.completed_at.isoformat() if event.completed_at else None,
            event.user_id, 1 if event.synced else 0, event.external_id,
            event.id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self.db.log_event("event", event.id, "updated", f"Updated event: {event.title}")
        
        return success
    
    def delete_event(self, event_id: str, delete_recurring: bool = False) -> bool:
        """Delete an event"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        if delete_recurring:
            cursor.execute("DELETE FROM events WHERE id = ? OR parent_event_id = ?", (event_id, event_id))
        else:
            cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self.db.log_event("event", event_id, "deleted", "Event deleted")
        
        return success
    
    def get_events_by_date_range(self, start_date: datetime.datetime, end_date: datetime.datetime,
                                  user_id: str = "default", include_all_day: bool = True) -> List[CalendarEvent]:
        """Get all events within a date range"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM events WHERE user_id = ?
            AND ((start_time >= ? AND start_time <= ?)
                OR (end_time >= ? AND end_time <= ?)
                OR (start_time <= ? AND end_time >= ?)
        """
        params = [user_id, start_date.isoformat(), end_date.isoformat(),
                  start_date.isoformat(), end_date.isoformat(),
                  start_date.isoformat(), end_date.isoformat()]
        
        if include_all_day:
            query += " OR all_day = 1"
        
        query += ") ORDER BY start_time ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_event(dict(row)) for row in rows]
    
    def get_events_for_day(self, date: datetime.date, user_id: str = "default") -> List[CalendarEvent]:
        """Get all events for a specific day"""
        start = datetime.datetime.combine(date, datetime.time.min)
        end = datetime.datetime.combine(date, datetime.time.max)
        return self.get_events_by_date_range(start, end, user_id)
    
    def get_events_for_week(self, date: datetime.date, user_id: str = "default") -> List[CalendarEvent]:
        """Get all events for the week containing the given date"""
        start_of_week = date - datetime.timedelta(days=date.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        start = datetime.datetime.combine(start_of_week, datetime.time.min)
        end = datetime.datetime.combine(end_of_week, datetime.time.max)
        return self.get_events_by_date_range(start, end, user_id)
    
    def get_events_for_month(self, year: int, month: int, user_id: str = "default") -> List[CalendarEvent]:
        """Get all events for a specific month"""
        start = datetime.datetime(year, month, 1)
        last_day = cal_module.monthrange(year, month)[1]
        end = datetime.datetime(year, month, last_day, 23, 59, 59)
        return self.get_events_by_date_range(start, end, user_id)
    
    def search_events(self, query: str, user_id: str = "default", limit: int = 50) -> List[CalendarEvent]:
        """Search events by title, description, or tags"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM events WHERE user_id = ?
            AND (title LIKE ? OR description LIKE ? OR tags LIKE ? OR notes LIKE ?)
            ORDER BY start_time DESC LIMIT ?
        """, (user_id, search_term, search_term, search_term, search_term, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_event(dict(row)) for row in rows]
    
    def get_upcoming_events(self, days: int = 7, user_id: str = "default", limit: int = 50) -> List[CalendarEvent]:
        """Get upcoming events for the next N days"""
        now = datetime.datetime.now()
        end = now + datetime.timedelta(days=days)
        events = self.get_events_by_date_range(now, end, user_id)
        return events[:limit]
    
    # -------------------------------------------------------------------------
    # TASK MANAGEMENT
    # -------------------------------------------------------------------------
    
    def create_task(self, task: Task) -> str:
        """Create a new task"""
        task.updated_at = datetime.datetime.now()
        if task.created_at is None:
            task.created_at = datetime.datetime.now()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tasks (
                id, title, description, due_date, priority, status,
                progress, estimated_minutes, actual_minutes,
                parent_task_id, subtasks, tags, assigned_to,
                created_at, updated_at, completed_at, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id, task.title, task.description,
            task.due_date.isoformat() if task.due_date else None,
            task.priority.value, task.status.value,
            task.progress, task.estimated_minutes, task.actual_minutes,
            task.parent_task_id, json.dumps(task.subtasks),
            json.dumps(task.tags), task.assigned_to,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None,
            task.user_id
        ))
        
        conn.commit()
        conn.close()
        
        self.db.log_event("task", task.id, "created", f"Created task: {task.title}")
        logger.info(f"[CALENDAR] Created task: {task.id} - {task.title}")
        
        return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_task(dict(row))
        return None
    
    def update_task(self, task: Task) -> bool:
        """Update an existing task"""
        task.updated_at = datetime.datetime.now()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks SET
                title = ?, description = ?, due_date = ?, priority = ?, status = ?,
                progress = ?, estimated_minutes = ?, actual_minutes = ?,
                parent_task_id = ?, subtasks = ?, tags = ?, assigned_to = ?,
                updated_at = ?, completed_at = ?, user_id = ?
            WHERE id = ?
        """, (
            task.title, task.description,
            task.due_date.isoformat() if task.due_date else None,
            task.priority.value, task.status.value,
            task.progress, task.estimated_minutes, task.actual_minutes,
            task.parent_task_id, json.dumps(task.subtasks),
            json.dumps(task.tags), task.assigned_to,
            task.updated_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None,
            task.user_id, task.id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed"""
        task = self.get_task(task_id)
        if task:
            task.status = EventStatus.COMPLETED
            task.completed_at = datetime.datetime.now()
            task.progress = 100
            return self.update_task(task)
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def get_tasks_by_due_date(self, start_date: Optional[datetime.datetime] = None,
                               end_date: Optional[datetime.datetime] = None,
                               user_id: str = "default", include_completed: bool = False) -> List[Task]:
        """Get tasks by due date range"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM tasks WHERE user_id = ?"
        params = [user_id]
        
        if not include_completed:
            query += " AND status != 'completed'"
        
        if start_date:
            query += " AND due_date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND due_date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY due_date ASC, priority ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_task(dict(row)) for row in rows]
    
    def get_overdue_tasks(self, user_id: str = "default") -> List[Task]:
        """Get all overdue tasks"""
        now = datetime.datetime.now()
        return [t for t in self.get_tasks_by_due_date(end_date=now, user_id=user_id)
                if t.status not in [EventStatus.COMPLETED, EventStatus.CANCELLED]]
    
    # -------------------------------------------------------------------------
    # SIMPLE REMINDER MANAGEMENT
    # -------------------------------------------------------------------------
    
    def add_reminder(self, reminder_id: str, message: str, remind_time: datetime.datetime,
                     title: str = "", note: str = "", user_id: str = "default") -> bool:
        """Add a simple reminder"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO reminders 
                (id, title, message, remind_time, note, created_at, completed, user_id)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (reminder_id, title or message[:50], message, remind_time.isoformat(),
                  note, datetime.datetime.now().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            
            if self.scheduler and APSCHEDULER_AVAILABLE:
                try:
                    trigger = DateTrigger(run_date=remind_time)
                    self.scheduler.add_job(self._trigger_reminder, trigger=trigger,
                                          args=[reminder_id, message],
                                          id=f"reminder_{reminder_id}",
                                          replace_existing=True)
                except Exception as e:
                    logger.warning(f"[REMINDER] Could not schedule: {e}")
            
            self.db.log_event("reminder", reminder_id, "created", f"Reminder: {message[:50]}")
            logger.info(f"[REMINDER] Added reminder '{reminder_id}' for {remind_time.isoformat()}")
            
            return True
            
        except Exception as e:
            logger.error(f"[REMINDER] Error adding reminder: {e}")
            return False
    
    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Get a reminder by ID"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_all_reminders(self, user_id: str = "default", include_completed: bool = False) -> List[Dict[str, Any]]:
        """Get all reminders for a user"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM reminders WHERE user_id = ?"
        params = [user_id]
        
        if not include_completed:
            query += " AND completed = 0"
        
        query += " ORDER BY remind_time ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def snooze_reminder(self, reminder_id: str, snooze_minutes: int = 15) -> bool:
        """Snooze a reminder"""
        new_time = datetime.datetime.now() + datetime.timedelta(minutes=snooze_minutes)
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE reminders SET snoozed_until = ?, remind_time = ? WHERE id = ?",
                       (new_time.isoformat(), new_time.isoformat(), reminder_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if self.scheduler:
            try:
                self.scheduler.remove_job(f"reminder_{reminder_id}")
            except:
                pass
        
        return success
    
    def _trigger_reminder(self, reminder_id: str, message: str):
        """Callback when a reminder triggers"""
        logger.info(f"[REMINDER TRIGGERED] {reminder_id}: {message}")
        print(f"\n🔔 REMINDER: {message}\n")
        
        for callback in self._reminder_callbacks:
            try:
                callback(reminder_id, message)
            except Exception as e:
                logger.error(f"[REMINDER] Callback error: {e}")
        
        self.db.log_event("reminder", reminder_id, "triggered", message)
    
    def register_reminder_callback(self, callback):
        """Register a callback for reminder triggers"""
        if callable(callback):
            self._reminder_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # CONTACT MANAGEMENT
    # -------------------------------------------------------------------------
    
    def store_contact(self, name: str, email: str = "", phone: str = "", address: str = "",
                      birthday: Optional[datetime.date] = None, notes: str = "",
                      tags: List[str] = None, user_id: str = "default") -> str:
        """Store a contact"""
        contact_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO contacts 
            (id, name, email, phone, address, birthday, notes, tags, created_at, updated_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contact_id, name, email, phone, address,
              birthday.isoformat() if birthday else None,
              notes, json.dumps(tags or []), now, now, user_id))
        
        conn.commit()
        conn.close()
        
        self.db.log_event("contact", contact_id, "created", f"Contact: {name}")
        return contact_id
    
    def get_contacts(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Get all contacts"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM contacts WHERE user_id = ? ORDER BY name COLLATE NOCASE", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        contacts = []
        for row in rows:
            contact = dict(row)
            if contact.get('tags'):
                try:
                    contact['tags'] = json.loads(contact['tags'])
                except:
                    contact['tags'] = []
            contacts.append(contact)
        
        return contacts
    
    def search_contacts(self, query: str, user_id: str = "default") -> List[Dict[str, Any]]:
        """Search contacts"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM contacts WHERE user_id = ? 
            AND (name LIKE ? OR email LIKE ? OR phone LIKE ?)
            ORDER BY name COLLATE NOCASE
        """, (user_id, search_term, search_term, search_term))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    # -------------------------------------------------------------------------
    # PASSWORD MANAGEMENT
    # -------------------------------------------------------------------------
    
    def store_password(self, label: str, plaintext_password: str, url: str = "",
                       username: str = "", notes: str = "", user_id: str = "default") -> str:
        """Store an encrypted password"""
        password_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        encrypted = encrypt_data(plaintext_password)
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO passwords 
            (id, label, encrypted_value, url, username, notes, created_at, updated_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (password_id, label, encrypted, url, username, notes, now, now, user_id))
        
        conn.commit()
        conn.close()
        
        self.db.log_event("password", password_id, "created", f"Password stored for: {label}")
        return password_id
    
    def get_password(self, label: str, user_id: str = "default") -> Optional[str]:
        """Retrieve and decrypt a password"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT encrypted_value FROM passwords WHERE label = ? AND user_id = ?",
                       (label, user_id))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return decrypt_data(row['encrypted_value'])
        return None
    
    def delete_password(self, label: str, user_id: str = "default") -> bool:
        """Delete a password entry"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM passwords WHERE label = ? AND user_id = ?", (label, user_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    # -------------------------------------------------------------------------
    # WEBPAGE/BOOKMARK MANAGEMENT
    # -------------------------------------------------------------------------
    
    def store_webpage(self, label: str, url: str, description: str = "",
                      tags: List[str] = None, favicon: str = "", user_id: str = "default") -> str:
        """Store a webpage bookmark"""
        webpage_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO webpages 
            (id, label, url, description, tags, favicon, saved_on, last_visited, visit_count, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (webpage_id, label, url, description, json.dumps(tags or []), favicon, now, now, user_id))
        
        conn.commit()
        conn.close()
        
        self.db.log_event("webpage", webpage_id, "created", f"Bookmark: {label}")
        return webpage_id
    
    def get_webpages(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Get all bookmarks"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM webpages WHERE user_id = ? ORDER BY saved_on DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        webpages = []
        for row in rows:
            wp = dict(row)
            if wp.get('tags'):
                try:
                    wp['tags'] = json.loads(wp['tags'])
                except:
                    wp['tags'] = []
            webpages.append(wp)
        
        return webpages
    
    def delete_webpage(self, webpage_id: str) -> bool:
        """Delete a bookmark"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM webpages WHERE id = ?", (webpage_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    # -------------------------------------------------------------------------
    # NATURAL LANGUAGE DATE PARSING
    # -------------------------------------------------------------------------
    
    def parse_natural_date(self, text: str) -> Optional[datetime.datetime]:
        """Parse natural language date/time expressions"""
        text = text.lower().strip()
        now = datetime.datetime.now()
        
        if DATEUTIL_AVAILABLE and date_parser:
            try:
                return date_parser.parse(text, fuzzy=True)
            except:
                pass
        
        # Manual parsing patterns
        if 'tomorrow' in text:
            result = now + datetime.timedelta(days=1)
        elif 'today' in text:
            result = now
        elif 'yesterday' in text:
            result = now - datetime.timedelta(days=1)
        elif 'next week' in text:
            result = now + datetime.timedelta(weeks=1)
        else:
            # Try to extract time patterns
            time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', text)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                period = time_match.group(3)
                if period == 'pm' and hour < 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                # Try "in X hours/minutes/days"
                in_match = re.search(r'in\s+(\d+)\s+(hour|minute|day|week)s?', text)
                if in_match:
                    amount = int(in_match.group(1))
                    unit = in_match.group(2)
                    if unit == 'minute':
                        result = now + datetime.timedelta(minutes=amount)
                    elif unit == 'hour':
                        result = now + datetime.timedelta(hours=amount)
                    elif unit == 'day':
                        result = now + datetime.timedelta(days=amount)
                    elif unit == 'week':
                        result = now + datetime.timedelta(weeks=amount)
                    else:
                        return None
                else:
                    # Try ISO format
                    try:
                        return datetime.datetime.fromisoformat(text)
                    except:
                        return None
        
        # Handle time if present
        time_match = re.search(r'at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?', text)
        if time_match and 'result' in locals():
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            if period == 'pm' and hour < 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0
            result = result.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return result if 'result' in locals() else None
    
    # -------------------------------------------------------------------------
    # CALENDAR VIEW HELPERS
    # -------------------------------------------------------------------------
    
    def get_month_calendar_grid(self, year: int, month: int, user_id: str = "default") -> List[List[Dict[str, Any]]]:
        """Get a calendar grid for month view"""
        events = self.get_events_for_month(year, month, user_id)
        holidays = get_us_holidays(year)
        
        events_by_date = defaultdict(list)
        for event in events:
            if event.start_time:
                date_key = event.start_time.date()
                events_by_date[date_key].append(event.to_dict())
        
        cal = cal_module.Calendar(firstweekday=6)
        weeks = []
        
        for week in cal.monthdayscalendar(year, month):
            week_data = []
            for day_num in week:
                if day_num == 0:
                    week_data.append({"day": 0, "events": [], "holiday": None})
                else:
                    date = datetime.date(year, month, day_num)
                    day_info = {
                        "day": day_num,
                        "date": date.isoformat(),
                        "events": events_by_date.get(date, []),
                        "holiday": holidays.get(date),
                        "is_today": date == datetime.date.today(),
                        "is_weekend": date.weekday() >= 5
                    }
                    week_data.append(day_info)
            weeks.append(week_data)
        
        return weeks
    
    # -------------------------------------------------------------------------
    # CONFLICT DETECTION
    # -------------------------------------------------------------------------
    
    def check_conflicts(self, start_time: datetime.datetime, end_time: datetime.datetime,
                        exclude_event_id: str = None, user_id: str = "default") -> List[CalendarEvent]:
        """Check for scheduling conflicts"""
        events = self.get_events_by_date_range(
            start_time - datetime.timedelta(hours=24),
            end_time + datetime.timedelta(hours=24),
            user_id
        )
        
        conflicts = []
        for event in events:
            if event.id == exclude_event_id:
                continue
            if event.all_day or not event.start_time or not event.end_time:
                continue
            if start_time < event.end_time and end_time > event.start_time:
                conflicts.append(event)
        
        return conflicts
    
    def suggest_available_slots(self, duration_minutes: int, date: datetime.date,
                                 working_hours: Tuple[int, int] = (9, 17),
                                 user_id: str = "default") -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """Suggest available time slots"""
        events = self.get_events_for_day(date, user_id)
        busy_periods = []
        
        for event in events:
            if event.start_time and event.end_time and not event.all_day:
                busy_periods.append((event.start_time, event.end_time))
        
        busy_periods.sort(key=lambda x: x[0])
        
        available = []
        day_start = datetime.datetime.combine(date, datetime.time(working_hours[0], 0))
        day_end = datetime.datetime.combine(date, datetime.time(working_hours[1], 0))
        
        current = day_start
        duration = datetime.timedelta(minutes=duration_minutes)
        
        for busy_start, busy_end in busy_periods:
            if busy_start - current >= duration:
                slot_end = min(busy_start, current + duration)
                available.append((current, slot_end))
            current = max(current, busy_end)
        
        if day_end - current >= duration:
            available.append((current, current + duration))
        
        return available
    
    # -------------------------------------------------------------------------
    # SCHEDULER HELPERS
    # -------------------------------------------------------------------------
    
    def _schedule_event_reminders(self, event: CalendarEvent):
        """Schedule reminders for an event"""
        if not self.scheduler or not event.start_time:
            return
        
        for minutes_before in event.reminder_minutes:
            remind_time = event.start_time - datetime.timedelta(minutes=minutes_before)
            
            if remind_time > datetime.datetime.now():
                try:
                    trigger = DateTrigger(run_date=remind_time)
                    self.scheduler.add_job(self._event_reminder_callback, trigger=trigger,
                                          args=[event.id, event.title, minutes_before],
                                          id=f"event_reminder_{event.id}_{minutes_before}",
                                          replace_existing=True)
                except Exception as e:
                    logger.warning(f"[CALENDAR] Could not schedule reminder: {e}")
    
    def _event_reminder_callback(self, event_id: str, title: str, minutes_before: int):
        """Callback when an event reminder triggers"""
        logger.info(f"[EVENT REMINDER] {title} - in {minutes_before} minutes")
        print(f"\n🗓️ EVENT REMINDER: {title} starts in {minutes_before} minutes\n")
        
        for callback in self._reminder_callbacks:
            try:
                callback(event_id, f"Event '{title}' starts in {minutes_before} minutes")
            except Exception as e:
                logger.error(f"[CALENDAR] Callback error: {e}")
    
    def load_and_schedule_pending(self):
        """Load and schedule all pending reminders"""
        now = datetime.datetime.now()
        
        reminders = self.get_all_reminders()
        for rem in reminders:
            try:
                remind_time = datetime.datetime.fromisoformat(rem['remind_time'])
                if remind_time > now:
                    self.add_reminder(rem['id'], rem['message'], remind_time,
                                     rem.get('title', ''), rem.get('note', ''),
                                     rem.get('user_id', 'default'))
            except Exception as e:
                logger.warning(f"[CALENDAR] Could not reschedule reminder: {e}")
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM events WHERE start_time > ? AND status = 'pending' AND reminder_sent = 0
        """, (now.isoformat(),))
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            try:
                event = self._row_to_event(dict(row))
                self._schedule_event_reminders(event)
            except Exception as e:
                logger.warning(f"[CALENDAR] Could not schedule event reminder: {e}")
        
        logger.info(f"[CALENDAR] Loaded and scheduled {len(reminders)} reminders and {len(rows)} events")
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def _row_to_event(self, row: Dict[str, Any]) -> CalendarEvent:
        """Convert database row to CalendarEvent"""
        for field in ['reminder_minutes', 'attendees', 'tags', 'attachments', 'recurrence_days']:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except:
                    row[field] = []
        
        row['all_day'] = bool(row.get('all_day', 0))
        row['reminder_sent'] = bool(row.get('reminder_sent', 0))
        row['synced'] = bool(row.get('synced', 0))
        
        return CalendarEvent.from_dict(row)
    
    def _row_to_task(self, row: Dict[str, Any]) -> Task:
        """Convert database row to Task"""
        for field in ['subtasks', 'tags']:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except:
                    row[field] = []
        
        return Task.from_dict(row)
    
    def get_statistics(self, user_id: str = "default") -> Dict[str, Any]:
        """Get calendar and task statistics"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (user_id,))
        total_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE user_id = ? AND start_time >= ?",
                       (user_id, datetime.datetime.now().isoformat()))
        upcoming_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ?", (user_id,))
        total_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status = 'completed'", (user_id,))
        completed_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status != 'completed' AND due_date < ?",
                       (user_id, datetime.datetime.now().isoformat()))
        overdue_tasks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM reminders WHERE user_id = ? AND completed = 0", (user_id,))
        pending_reminders = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_events": total_events,
            "upcoming_events": upcoming_events,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "overdue_tasks": overdue_tasks,
            "pending_reminders": pending_reminders,
            "task_completion_rate": round(completed_tasks / max(total_tasks, 1) * 100, 1)
        }
    
    def export_to_ical(self, events: List[CalendarEvent] = None, user_id: str = "default") -> str:
        """Export events to iCal format"""
        if events is None:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            conn.close()
            events = [self._row_to_event(dict(row)) for row in rows]
        
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//SarahMemory//Calendar//EN",
                 "CALSCALE:GREGORIAN", "METHOD:PUBLISH"]
        
        for event in events:
            lines.append("BEGIN:VEVENT")
            lines.append(f"UID:{event.id}")
            lines.append(f"SUMMARY:{event.title}")
            if event.description:
                lines.append(f"DESCRIPTION:{event.description}")
            if event.location:
                lines.append(f"LOCATION:{event.location}")
            if event.start_time:
                dt_format = "%Y%m%dT%H%M%S"
                lines.append(f"DTSTART:{event.start_time.strftime(dt_format)}")
                if event.end_time:
                    lines.append(f"DTEND:{event.end_time.strftime(dt_format)}")
            lines.append("END:VEVENT")
        
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)
    
    def shutdown(self):
        """Shutdown the calendar manager gracefully"""
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
                logger.info("[CALENDAR] Scheduler shut down")
            except Exception as e:
                logger.warning(f"[CALENDAR] Scheduler shutdown error: {e}")


# =============================================================================
# GLOBAL INSTANCE AND LEGACY API
# =============================================================================

_calendar_manager: Optional[CalendarManager] = None
_scheduler: Optional[BackgroundScheduler] = None


def get_calendar_manager() -> CalendarManager:
    """Get or create the global calendar manager instance"""
    global _calendar_manager
    if _calendar_manager is None:
        _calendar_manager = CalendarManager()
    return _calendar_manager


def init_db():
    """Initialize the database (legacy function)"""
    get_calendar_manager()
    logger.info("[REMINDER] Database initialized")


def log_reminder_event(event: str, description: str):
    """Log a reminder event (legacy function)"""
    manager = get_calendar_manager()
    manager.db.log_event("reminder", "", event, description)


def add_reminder(reminder_id: str, message: str, remind_time: datetime.datetime) -> bool:
    """Add a reminder (legacy function)"""
    manager = get_calendar_manager()
    return manager.add_reminder(reminder_id, message, remind_time)


def reminder_action(reminder_id: str, message: str):
    """Reminder action callback (legacy function)"""
    logger.info(f"[REMINDER TRIGGERED] {reminder_id}: {message}")
    print(f"REMINDER: {message}")
    log_reminder_event("Reminder Triggered", f"Triggered reminder '{reminder_id}': {message}")


def store_contact(name: str, email: str, phone: str, address: str):
    """Store a contact (legacy function)"""
    manager = get_calendar_manager()
    manager.store_contact(name, email, phone, address)


def store_webpage(label: str, url: str):
    """Store a webpage bookmark (legacy function)"""
    manager = get_calendar_manager()
    manager.store_webpage(label, url)


def store_password(label: str, plaintext_password: str):
    """Store an encrypted password (legacy function)"""
    manager = get_calendar_manager()
    manager.store_password(label, plaintext_password)


def start_scheduler():
    """Start the background scheduler (legacy function)"""
    global _scheduler
    if APSCHEDULER_AVAILABLE and _scheduler is None:
        try:
            _scheduler = BackgroundScheduler()
            _scheduler.start()
            logger.info("[REMINDER] Scheduler started successfully")
            log_reminder_event("Start Scheduler", "Reminder scheduler started successfully")
        except Exception as e:
            logger.error(f"[REMINDER] Error starting scheduler: {e}")
            log_reminder_event("Start Scheduler Error", str(e))


def load_reminders():
    """Load and schedule pending reminders (legacy function)"""
    manager = get_calendar_manager()
    manager.load_and_schedule_pending()


def start_reminder_monitor():
    """Start the reminder monitoring system (legacy function)"""
    init_db()
    manager = get_calendar_manager()
    if manager.scheduler is None and APSCHEDULER_AVAILABLE:
        manager._init_scheduler()
    load_reminders()
    logger.info("[REMINDER] Reminder monitor started")


# =============================================================================
# WEBUI API HELPER FUNCTIONS
# =============================================================================

def get_reminders_for_api(user_id: str = "default") -> List[Dict[str, Any]]:
    """Get reminders formatted for WebUI API"""
    manager = get_calendar_manager()
    reminders = manager.get_all_reminders(user_id)
    
    return [{
        'id': rem.get('id'),
        'title': rem.get('title') or rem.get('message', '')[:50],
        'time': rem.get('remind_time'),
        'note': rem.get('note', '')
    } for rem in reminders]


def save_reminder_from_api(title: str, time_str: str, note: str = "", user_id: str = "default") -> Dict[str, Any]:
    """Save a reminder from WebUI API"""
    try:
        remind_time = datetime.datetime.fromisoformat(time_str)
        reminder_id = str(uuid.uuid4())
        
        manager = get_calendar_manager()
        success = manager.add_reminder(reminder_id, title, remind_time, title, note, user_id)
        
        if success:
            return {"status": "ok", "id": reminder_id}
        return {"status": "error", "error": "Failed to save reminder"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def delete_reminder_from_api(reminder_id: str) -> Dict[str, Any]:
    """Delete a reminder from WebUI API"""
    manager = get_calendar_manager()
    success = manager.delete_reminder(reminder_id)
    return {"status": "deleted"} if success else {"status": "error", "error": "Reminder not found"}


def get_events_for_api(date_filter: str = None, user_id: str = "default") -> List[Dict[str, Any]]:
    """Get calendar events formatted for WebUI API"""
    manager = get_calendar_manager()
    
    if date_filter:
        try:
            date = datetime.datetime.strptime(date_filter, "%Y-%m-%d").date()
            events = manager.get_events_for_day(date, user_id)
        except ValueError:
            events = manager.get_upcoming_events(30, user_id)
    else:
        events = manager.get_upcoming_events(30, user_id)
    
    return [e.to_dict() for e in events]


def save_event_from_api(title: str, start_time: str, end_time: str = None, description: str = "",
                        location: str = "", event_type: str = "appointment", all_day: bool = False,
                        reminder_minutes: List[int] = None, recurrence: str = "none",
                        user_id: str = "default") -> Dict[str, Any]:
    """Save a calendar event from WebUI API"""
    try:
        event = CalendarEvent(
            title=title,
            description=description,
            location=location,
            event_type=EventType(event_type) if event_type else EventType.APPOINTMENT,
            start_time=datetime.datetime.fromisoformat(start_time) if start_time else None,
            end_time=datetime.datetime.fromisoformat(end_time) if end_time else None,
            all_day=all_day,
            reminder_minutes=reminder_minutes or [15],
            recurrence=RecurrencePattern(recurrence) if recurrence else RecurrencePattern.NONE,
            user_id=user_id
        )
        
        manager = get_calendar_manager()
        event_id = manager.create_event(event)
        
        return {"status": "ok", "id": event_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def delete_event_from_api(event_id: str) -> Dict[str, Any]:
    """Delete a calendar event from WebUI API"""
    manager = get_calendar_manager()
    success = manager.delete_event(event_id)
    return {"status": "deleted"} if success else {"status": "error", "error": "Event not found"}


def get_tasks_for_api(include_completed: bool = False, user_id: str = "default") -> List[Dict[str, Any]]:
    """Get tasks formatted for WebUI API"""
    manager = get_calendar_manager()
    tasks = manager.get_tasks_by_due_date(user_id=user_id, include_completed=include_completed)
    return [t.to_dict() for t in tasks]


def save_task_from_api(title: str, due_date: str = None, description: str = "",
                       priority: int = 3, user_id: str = "default") -> Dict[str, Any]:
    """Save a task from WebUI API"""
    try:
        task = Task(
            title=title,
            description=description,
            due_date=datetime.datetime.fromisoformat(due_date) if due_date else None,
            priority=Priority(priority),
            user_id=user_id
        )
        
        manager = get_calendar_manager()
        task_id = manager.create_task(task)
        
        return {"status": "ok", "id": task_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def complete_task_from_api(task_id: str) -> Dict[str, Any]:
    """Complete a task from WebUI API"""
    manager = get_calendar_manager()
    success = manager.complete_task(task_id)
    return {"status": "completed"} if success else {"status": "error", "error": "Task not found"}


def delete_task_from_api(task_id: str) -> Dict[str, Any]:
    """Delete a task from WebUI API"""
    manager = get_calendar_manager()
    success = manager.delete_task(task_id)
    return {"status": "deleted"} if success else {"status": "error", "error": "Task not found"}


def get_calendar_month_for_api(year: int, month: int, user_id: str = "default") -> Dict[str, Any]:
    """Get calendar month data formatted for WebUI"""
    manager = get_calendar_manager()
    
    grid = manager.get_month_calendar_grid(year, month, user_id)
    holidays = get_us_holidays(year)
    
    return {
        "year": year,
        "month": month,
        "month_name": cal_module.month_name[month],
        "weeks": grid,
        "holidays": {d.isoformat(): name for d, name in holidays.items() if d.month == month}
    }


def get_statistics_for_api(user_id: str = "default") -> Dict[str, Any]:
    """Get calendar/task statistics for WebUI"""
    manager = get_calendar_manager()
    return manager.get_statistics(user_id)


def parse_natural_date_for_api(text: str) -> Dict[str, Any]:
    """Parse natural language date for WebUI"""
    manager = get_calendar_manager()
    result = manager.parse_natural_date(text)
    
    if result:
        return {
            "status": "ok",
            "datetime": result.isoformat(),
            "formatted": result.strftime("%A, %B %d, %Y at %I:%M %p")
        }
    return {"status": "error", "error": "Could not parse date"}


def check_conflicts_for_api(start_time: str, end_time: str, exclude_event_id: str = None,
                            user_id: str = "default") -> Dict[str, Any]:
    """Check for scheduling conflicts for WebUI"""
    try:
        start = datetime.datetime.fromisoformat(start_time)
        end = datetime.datetime.fromisoformat(end_time)
        
        manager = get_calendar_manager()
        conflicts = manager.check_conflicts(start, end, exclude_event_id, user_id)
        
        return {"has_conflicts": len(conflicts) > 0, "conflicts": [e.to_dict() for e in conflicts]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_upcoming_for_api(days: int = 7, user_id: str = "default") -> Dict[str, Any]:
    """Get upcoming events and tasks for WebUI dashboard"""
    manager = get_calendar_manager()
    
    events = manager.get_upcoming_events(days, user_id)
    now = datetime.datetime.now()
    end = now + datetime.timedelta(days=days)
    tasks = manager.get_tasks_by_due_date(now, end, user_id)
    reminders = manager.get_all_reminders(user_id)
    
    upcoming_reminders = []
    for rem in reminders:
        try:
            remind_time = datetime.datetime.fromisoformat(rem['remind_time'])
            if now <= remind_time <= end:
                upcoming_reminders.append(rem)
        except:
            pass
    
    return {
        "events": [e.to_dict() for e in events[:10]],
        "tasks": [t.to_dict() for t in tasks[:10]],
        "reminders": upcoming_reminders[:10],
        "statistics": manager.get_statistics(user_id)
    }


def get_contacts_for_api(user_id: str = "default") -> List[Dict[str, Any]]:
    """Get contacts formatted for WebUI API"""
    manager = get_calendar_manager()
    return manager.get_contacts(user_id)


def save_contact_from_api(name: str, email: str = "", phone: str = "", address: str = "",
                          birthday: str = None, notes: str = "", tags: List[str] = None,
                          user_id: str = "default") -> Dict[str, Any]:
    """Save a contact from WebUI API"""
    try:
        birthday_date = None
        if birthday:
            try:
                birthday_date = datetime.datetime.strptime(birthday, "%Y-%m-%d").date()
            except:
                pass
        
        manager = get_calendar_manager()
        contact_id = manager.store_contact(name, email, phone, address, birthday_date, notes, tags, user_id)
        
        return {"status": "ok", "id": contact_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def delete_contact_from_api(contact_id: str) -> Dict[str, Any]:
    """Delete a contact from WebUI API"""
    manager = get_calendar_manager()
    success = manager.delete_contact(contact_id)
    return {"status": "deleted"} if success else {"status": "error", "error": "Contact not found"}


# =============================================================================
# AI ASSISTANT INTEGRATION
# =============================================================================

def process_calendar_command(command: str, user_id: str = "default") -> Dict[str, Any]:
    """Process natural language calendar commands from AI assistant"""
    command_lower = command.lower().strip()
    manager = get_calendar_manager()
    
    if any(word in command_lower for word in ['remind', 'reminder', 'alert']):
        return _process_reminder_command(command, manager, user_id)
    elif any(word in command_lower for word in ['schedule', 'meeting', 'appointment', 'event']):
        return _process_event_command(command, manager, user_id)
    elif any(word in command_lower for word in ['task', 'todo', 'to-do', 'to do']):
        return _process_task_command(command, manager, user_id)
    elif any(word in command_lower for word in ['what', 'show', 'list', 'view', 'calendar']):
        return _process_query_command(command, manager, user_id)
    else:
        return {
            "status": "unknown",
            "message": "I'm not sure what calendar action you want. Try:\n"
                      "- 'Remind me to [task] at [time]'\n"
                      "- 'Schedule [event] for [date/time]'\n"
                      "- 'Create task: [description] due [date]'\n"
                      "- 'What's on my calendar [today/tomorrow/this week]?'"
        }


def _process_reminder_command(command: str, manager: CalendarManager, user_id: str) -> Dict[str, Any]:
    """Process a reminder command"""
    patterns = [
        r'remind(?:\s+me)?\s+to\s+(.+?)\s+(?:at|on|in)\s+(.+)',
        r'remind(?:\s+me)?\s+(?:at|on|in)\s+(.+?)\s+to\s+(.+)',
        r'set\s+(?:a\s+)?reminder\s+(?:for\s+)?(.+?)\s+(?:at|on|in)\s+(.+)',
    ]
    
    message = None
    time_str = None
    
    for pattern in patterns:
        match = re.search(pattern, command.lower())
        if match:
            groups = match.groups()
            if any(word in groups[0] for word in ['tomorrow', 'today', 'hour', 'minute', 'pm', 'am', ':']):
                time_str, message = groups
            else:
                message, time_str = groups
            break
    
    if not message or not time_str:
        return {"status": "error", "message": "Please try: 'Remind me to [task] at [time]'"}
    
    remind_time = manager.parse_natural_date(time_str)
    if not remind_time:
        return {"status": "error", "message": f"Could not understand time '{time_str}'"}
    
    reminder_id = str(uuid.uuid4())
    success = manager.add_reminder(reminder_id, message.strip(), remind_time, message.strip()[:50], "", user_id)
    
    if success:
        formatted_time = remind_time.strftime("%A, %B %d at %I:%M %p")
        return {
            "status": "ok",
            "message": f"✅ Reminder set: '{message.strip()}' for {formatted_time}",
            "reminder_id": reminder_id,
            "remind_time": remind_time.isoformat()
        }
    
    return {"status": "error", "message": "Failed to create reminder"}


def _process_event_command(command: str, manager: CalendarManager, user_id: str) -> Dict[str, Any]:
    """Process an event scheduling command"""
    patterns = [
        r'schedule\s+(?:a\s+)?(.+?)\s+(?:for|on|at)\s+(.+)',
        r'(?:create|add)\s+(?:a\s+)?(?:meeting|appointment|event)\s+(?:called\s+)?(.+?)\s+(?:for|on|at)\s+(.+)',
    ]
    
    title = None
    time_str = None
    
    for pattern in patterns:
        match = re.search(pattern, command.lower())
        if match:
            title, time_str = match.groups()
            break
    
    if not title or not time_str:
        return {"status": "error", "message": "Please try: 'Schedule [event name] for [date/time]'"}
    
    start_time = manager.parse_natural_date(time_str)
    if not start_time:
        return {"status": "error", "message": f"Could not understand time '{time_str}'"}
    
    end_time = start_time + datetime.timedelta(hours=1)
    
    event = CalendarEvent(
        title=title.strip().title(),
        start_time=start_time,
        end_time=end_time,
        event_type=EventType.MEETING if 'meeting' in command.lower() else EventType.APPOINTMENT,
        user_id=user_id
    )
    
    event_id = manager.create_event(event)
    formatted_time = start_time.strftime("%A, %B %d at %I:%M %p")
    
    return {
        "status": "ok",
        "message": f"✅ Scheduled: '{title.strip().title()}' for {formatted_time}",
        "event_id": event_id,
        "start_time": start_time.isoformat()
    }


def _process_task_command(command: str, manager: CalendarManager, user_id: str) -> Dict[str, Any]:
    """Process a task creation command"""
    patterns = [
        r'(?:create|add)\s+(?:a\s+)?task[:\s]+(.+?)(?:\s+due\s+(.+))?$',
        r'task[:\s]+(.+?)(?:\s+due\s+(.+))?$',
    ]
    
    title = None
    due_str = None
    
    for pattern in patterns:
        match = re.search(pattern, command.lower())
        if match:
            title = match.group(1)
            due_str = match.group(2) if len(match.groups()) > 1 else None
            break
    
    if not title:
        return {"status": "error", "message": "Please try: 'Create task: [description] due [date]'"}
    
    due_date = manager.parse_natural_date(due_str) if due_str else None
    
    task = Task(title=title.strip().title(), due_date=due_date, user_id=user_id)
    task_id = manager.create_task(task)
    
    if due_date:
        formatted_date = due_date.strftime("%A, %B %d")
        message = f"✅ Task created: '{title.strip().title()}' due {formatted_date}"
    else:
        message = f"✅ Task created: '{title.strip().title()}' (no due date)"
    
    return {"status": "ok", "message": message, "task_id": task_id}


def _process_query_command(command: str, manager: CalendarManager, user_id: str) -> Dict[str, Any]:
    """Process a calendar query command"""
    command_lower = command.lower()
    
    if 'today' in command_lower:
        events = manager.get_events_for_day(datetime.date.today(), user_id)
        period = "today"
    elif 'tomorrow' in command_lower:
        events = manager.get_events_for_day(datetime.date.today() + datetime.timedelta(days=1), user_id)
        period = "tomorrow"
    elif 'week' in command_lower:
        events = manager.get_events_for_week(datetime.date.today(), user_id)
        period = "this week"
    elif 'month' in command_lower:
        today = datetime.date.today()
        events = manager.get_events_for_month(today.year, today.month, user_id)
        period = "this month"
    else:
        events = manager.get_upcoming_events(7, user_id)
        period = "the next 7 days"
    
    if not events:
        return {"status": "ok", "message": f"📅 No events scheduled for {period}.", "events": []}
    
    event_list = []
    for event in events[:10]:
        if event.start_time:
            time_str = event.start_time.strftime("%a %m/%d %I:%M %p")
            event_list.append(f"• {event.title} - {time_str}")
        else:
            event_list.append(f"• {event.title}")
    
    message = f"📅 Events for {period}:\n" + "\n".join(event_list)
    if len(events) > 10:
        message += f"\n... and {len(events) - 10} more events"
    
    return {"status": "ok", "message": message, "events": [e.to_dict() for e in events]}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    """Test the SarahMemoryReminder module"""
    print("=" * 70)
    print("SarahMemory Calendar & Reminder System")
    print(f"Version: {PROJECT_VERSION}")
    print("=" * 70)
    
    # Initialize the system
    start_reminder_monitor()
    
    # Get the manager
    manager = get_calendar_manager()
    
    # Test: Create a reminder
    print("\n[TEST] Creating a test reminder...")
    test_reminder_id = "test_" + str(uuid.uuid4())[:8]
    test_message = "This is a test reminder from SarahMemory"
    remind_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
    
    success = add_reminder(test_reminder_id, test_message, remind_time)
    print(f"Reminder created: {success}")
    print(f"  ID: {test_reminder_id}")
    print(f"  Time: {remind_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test: Create a calendar event
    print("\n[TEST] Creating a test calendar event...")
    event = CalendarEvent(
        title="Test Meeting",
        description="A test meeting created by the system",
        start_time=datetime.datetime.now() + datetime.timedelta(hours=1),
        end_time=datetime.datetime.now() + datetime.timedelta(hours=2),
        event_type=EventType.MEETING,
        reminder_minutes=[15, 5]
    )
    event_id = manager.create_event(event)
    print(f"Event created: {event_id}")
    print(f"  Title: {event.title}")
    print(f"  Start: {event.start_time}")
    
    # Test: Create a task
    print("\n[TEST] Creating a test task...")
    task = Task(
        title="Complete SarahMemory Calendar Module",
        description="Finish implementing all calendar features",
        due_date=datetime.datetime.now() + datetime.timedelta(days=1),
        priority=Priority.HIGH
    )
    task_id = manager.create_task(task)
    print(f"Task created: {task_id}")
    print(f"  Title: {task.title}")
    print(f"  Due: {task.due_date}")
    
    # Test: Get statistics
    print("\n[TEST] Calendar Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test: Natural language parsing
    print("\n[TEST] Natural Language Date Parsing:")
    test_phrases = ["tomorrow at 3pm", "next week", "in 2 hours", "today at 10am"]
    for phrase in test_phrases:
        result = manager.parse_natural_date(phrase)
        if result:
            print(f"  '{phrase}' -> {result.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  '{phrase}' -> Could not parse")
    
    # Test: AI command processing
    print("\n[TEST] AI Command Processing:")
    test_commands = [
        "Remind me to call John tomorrow at 2pm",
        "Schedule a meeting with team for next Monday at 10am",
        "What's on my calendar this week?",
    ]
    for cmd in test_commands:
        print(f"\n  Command: '{cmd}'")
        result = process_calendar_command(cmd)
        print(f"  Response: {result.get('message', result)}")
    
    # Test: Month calendar grid
    print("\n[TEST] Month Calendar Grid (current month):")
    today = datetime.date.today()
    grid = manager.get_month_calendar_grid(today.year, today.month)
    print(f"  Month: {cal_module.month_name[today.month]} {today.year}")
    print(f"  Weeks: {len(grid)}")
    
    # Keep running for reminder test
    print("\n" + "=" * 70)
    print("Waiting for test reminder to trigger (10 seconds)...")
    print("Press Ctrl+C to exit early.")
    print("=" * 70)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down...")
        manager.shutdown()
        print("Done.")

# ====================================================================
# END OF SarahMemoryReminder.py v8.0.0
# ====================================================================