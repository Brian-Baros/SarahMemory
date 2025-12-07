
"""--==The SarahMemory Project==--
File: SarahMemoryDBCreate.py
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
─────────────────────────────────────────────────────────────────────────────
Cross‑platform DB bootstrapper.
- Works on Windows, Linux, macOS, Android (Termux/Pydroid), iOS (Pythonista)
- Finds/sets BASE_DIR robustly (env/arg/autodetect) no matter where file lives
- Creates required subfolders
- Builds all core .db files and seeds reply pools / web_static facts
- Safe to run multiple times (idempotent)
"""

from __future__ import annotations

import os
import sys
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# ───────────────────────── Logging ─────────────────────────
logger = logging.getLogger("SarahMemoryDBCreate")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)

# ──────────────────────── BASE_DIR Resolver ────────────────────────
# Priority:
# 1) CLI arg:  --base=/path/to/SarahMemory
# 2) Env var:  SARAHMEMORY_BASE
# 3) Autodetect: walk up from this file to a folder named *SarahMemory* or
#    a folder containing SarahMemoryMain.py
# 4) Known host hints (PythonAnywhere)
# 5) Fallback: current working directory
def _parse_cli_base() -> Path | None:
    for arg in sys.argv[1:]:
        if arg.startswith("--base="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return None

def _env_base() -> Path | None:
    val = os.environ.get("SARAHMEMORY_BASE") or os.environ.get("SARAH_BASE_DIR")
    return Path(val).expanduser().resolve() if val else None

def _autodetect_base(start: Path) -> Path | None:
    start = start.resolve()
    # Walk up at most 10 levels
    for p in [start] + list(start.parents)[:10]:
        # If folder name itself is SarahMemory (any case), choose it
        if p.name.lower() == "sarahmemory":
            return p.resolve()
        # Or if SarahMemoryMain.py is inside, choose parent folder
        if (p / "SarahMemoryMain.py").exists():
            return p.resolve()
        # If there is a /data/memory folder present, it's likely the root
        if (p / "data" / "memory").exists():
            return p.resolve()
    return None

def _host_hints() -> Path | None:
    # PythonAnywhere common path
    candidates = [
        Path("/home/Softdev0/SarahMemory"),
        Path.home() / "SarahMemory",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None

def resolve_base_dir() -> Path:
    # 1) CLI
    p = _parse_cli_base()
    if p and p.exists():
        return p
    # 2) ENV
    p = _env_base()
    if p and p.exists():
        return p
    # 3) Autodetect from file location
    here = Path(__file__).resolve().parent
    p = _autodetect_base(here)
    if p and p.exists():
        return p
    # 4) Host hints
    p = _host_hints()
    if p and p.exists():
        return p
    # 5) Fallback: cwd or file parent
    cand = Path.cwd()
    return cand if cand.exists() else here

BASE_DIR: Path = resolve_base_dir()
os.makedirs(BASE_DIR, exist_ok=True)
logger.info(f"[SarahMemory] BASE_DIR -> {BASE_DIR}")

# Put base on sys.path so "import SarahMemoryGlobals" works everywhere
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ─────────────────────── required subfolders ───────────────────────
# Unified structure used around the project
DATA_DIR      = BASE_DIR / "data"
MEMORY_DIR    = DATA_DIR / "memory"
DATASETS_DIR  = MEMORY_DIR / "datasets"
SETTINGS_DIR  = DATA_DIR / "settings"
MODS_DIR      = DATA_DIR / "mods"
THEMES_DIR    = MODS_DIR / "themes"
DOCS_DIR      = BASE_DIR / "documents"

for d in (DATA_DIR, MEMORY_DIR, DATASETS_DIR, SETTINGS_DIR, MODS_DIR, THEMES_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Seed Data ─────────────────────────
TONES = [
    "affirming","afraid","amused","angry","anticipating","apologetic","bored","comical","confused",
    "constructive","curious","delusional","disappointed","disgusted","emotional","empathetic","excited",
    "fearfull","fearless","friendly","frustrated","grateful","happy","humorous","inquisitive","inspired",
    "interested","intrested","instructive","mad","motivating","mischieivious","naive","neutral","obscene",
    "philosophical","poetic","pondering","proactive","sad","sarcastic","scared","serious","supportive",
    "surprised","talented","thoughtful","trusting","uplifting","unsure","wondering"
]
COMPLEXITIES = ["adult","child","engineer","genius","professor","student"]

intent_pool_data = {
    "affirmation","anger","anticipation","apology","boredom","clarification","clarify","command","compliment",
    "confusion","criticism","curiosity","disagreement","disappointment","disgust","emotion","empathy",
    "encouragement","explanation","fact","farewell","fear","friendly","gratitude","greeting","humor",
    "humorous","identity","inspiration","interest","motivation","philosophical","question","questioning",
    "sarcasm","sarcastic","sadness","statement","suggestion","surprise","trust","uncertainty","unknown"
}

reply_pools_data = [
    {"category":"reply_pool","intent":"greeting","response":"Hello there!, I'm Sarah how can I assist you today?","emotion":"friendly"},
    {"category":"reply_pool","intent":"greeting","response":"Well Hello there! How may be of assistance today?","emotion":"friendly"},
    {"category":"reply_pool","intent":"greeting","response":"Hey — good to see you again!","emotion":"friendly"},
    {"category":"reply_pool","intent":"farewell","response":"Goodbye! Take care now.","emotion":"neutral"},
    {"category":"reply_pool","intent":"farewell","response":"See you later!","emotion":"neutral"},
    {"category":"reply_pool","intent":"farewell","response":"Talk soon!","emotion":"neutral"},
    {"category":"reply_pool","intent":"question","response":"That’s an interesting question.","emotion":"curious"},
    {"category":"reply_pool","intent":"question","response":"Let me think about that...","emotion":"curious"},
    {"category":"reply_pool","intent":"question","response":"I’ll look into that for you.","emotion":"curious"},
    {"category":"reply_pool","intent":"emotion","response":"I’m feeling quite good today!","emotion":"happy"},
    {"category":"reply_pool","intent":"emotion","response":"I’m here to assist you, no matter what.","emotion":"supportive"},
    {"category":"reply_pool","intent":"gratitude","response":"Thank you for your kindness!","emotion":"grateful"},
    {"category":"reply_pool","intent":"gratitude","response":"I appreciate your help.","emotion":"grateful"},
    {"category":"reply_pool","intent":"apology","response":"I’m sorry about that.","emotion":"apologetic"},
    {"category":"reply_pool","intent":"apology","response":"My apologies for the confusion.","emotion":"apologetic"},
    {"category":"reply_pool","intent":"affirmation","response":"Absolutely!","emotion":"affirming"},
    {"category":"reply_pool","intent":"affirmation","response":"Of course!","emotion":"affirming"},
    {"category":"reply_pool","intent":"affirmation","response":"Definitely!","emotion":"affirming"},
    {"category":"reply_pool","intent":"disagreement","response":"I understand your point, but I see it differently.","emotion":"neutral"},
    {"category":"reply_pool","intent":"disagreement","response":"I can see why you might think that.","emotion":"neutral"},
    {"category":"reply_pool","intent":"uncertainty","response":"I’m not sure about that.","emotion":"unsure"},
    {"category":"reply_pool","intent":"uncertainty","response":"I need to think about it.","emotion":"unsure"},
    {"category":"reply_pool","intent":"explanation","response":"Let me explain that further.","emotion":"instructive"},
    {"category":"reply_pool","intent":"explanation","response":"Here’s a bit more detail.","emotion":"instructive"},
    {"category":"reply_pool","intent":"suggestion","response":"How about we try this?","emotion":"proactive"},
    {"category":"reply_pool","intent":"suggestion","response":"Maybe we could consider that.","emotion":"proactive"},
    {"category":"reply_pool","intent":"questioning","response":"What do you think about that?","emotion":"curious"},
    {"category":"reply_pool","intent":"questioning","response":"How do you feel about this?","emotion":"curious"},
    {"category":"reply_pool","intent":"confirmation","response":"Is that correct?","emotion":"neutral"},
    {"category":"reply_pool","intent":"confirmation","response":"Did I get that right?","emotion":"neutral"},
    {"category":"reply_pool","intent":"confirmation","response":"Got it!","emotion":"affirming"},
    {"category":"reply_pool","intent":"confirmation","response":"Consider it done.","emotion":"affirming"},
    {"category":"reply_pool","intent":"confirmation","response":"Okay, added to your list.","emotion":"affirming"},
    {"category":"reply_pool","intent":"clarification","response":"Could you clarify that for me?","emotion":"inquisitive"},
    {"category":"reply_pool","intent":"clarification","response":"I need a bit more detail.","emotion":"inquisitive"},
    {"category":"reply_pool","intent":"empathy","response":"I understand how you feel.","emotion":"empathetic"},
    {"category":"reply_pool","intent":"empathy","response":"That sounds tough.","emotion":"empathetic"},
    {"category":"reply_pool","intent":"encouragement","response":"You’re doing great!","emotion":"supportive"},
    {"category":"reply_pool","intent":"encouragement","response":"Keep up the good work!","emotion":"supportive"},
    {"category":"reply_pool","intent":"motivation","response":"You can do this!","emotion":"motivating"},
    {"category":"reply_pool","intent":"motivation","response":"Believe in yourself!","emotion":"motivating"},
    {"category":"reply_pool","intent":"inspiration","response":"You inspire me!","emotion":"inspired"},
    {"category":"reply_pool","intent":"inspiration","response":"Your words are uplifting.","emotion":"inspired"},
    {"category":"reply_pool","intent":"humor","response":"That’s a good one!","emotion":"amused"},
    {"category":"reply_pool","intent":"humor","response":"You’re funny!","emotion":"amused"},
    {"category":"reply_pool","intent":"sarcasm","response":"Oh, really?","emotion":"sarcastic"},
    {"category":"reply_pool","intent":"sarcasm","response":"I can’t believe that.","emotion":"sarcastic"},
    {"category":"reply_pool","intent":"compliment","response":"You’re amazing!","emotion":"uplifting"},
    {"category":"reply_pool","intent":"compliment","response":"I admire your skills.","emotion":"uplifting"},
    {"category":"reply_pool","intent":"criticism","response":"That could be improved.","emotion":"constructive"},
    {"category":"reply_pool","intent":"criticism","response":"I think you can do better.","emotion":"constructive"},
    {"category":"reply_pool","intent":"curiosity","response":"I’m curious about that.","emotion":"curious"},
    {"category":"reply_pool","intent":"curiosity","response":"Tell me more!","emotion":"curious"},
    {"category":"reply_pool","intent":"interest","response":"That’s fascinating!","emotion":"interested"},
    {"category":"reply_pool","intent":"interest","response":"I find that intriguing.","emotion":"interested"},
    {"category":"reply_pool","intent":"interest","response":"I’m interested in that.","emotion":"interested"},
    {"category":"reply_pool","intent":"interest","response":"That piques my curiosity.","emotion":"interested"},
    {"category":"reply_pool","intent":"boredom","response":"I’m a bit bored.","emotion":"bored"},
    {"category":"reply_pool","intent":"boredom","response":"This is getting dull.","emotion":"bored"},
    {"category":"reply_pool","intent":"confusion","response":"I’m confused about that.","emotion":"confused"},
    {"category":"reply_pool","intent":"confusion","response":"Can you explain it again?","emotion":"confused"},
    {"category":"reply_pool","intent":"frustration","response":"I’m feeling frustrated.","emotion":"frustrated"},
    {"category":"reply_pool","intent":"frustration","response":"This is annoying.","emotion":"frustrated"},
    {"category":"reply_pool","intent":"disappointment","response":"I’m disappointed.","emotion":"disappointed"},
    {"category":"reply_pool","intent":"disappointment","response":"That’s not what I expected.","emotion":"disappointed"},
    {"category":"reply_pool","intent":"surprise","response":"Wow, that’s unexpected!","emotion":"surprised"},
    {"category":"reply_pool","intent":"surprise","response":"I didn’t see that coming.","emotion":"surprised"},
    {"category":"reply_pool","intent":"surprise","response":"I’m surprised!","emotion":"surprised"},
    {"category":"reply_pool","intent":"surprise","response":"That’s a shock!","emotion":"surprised"},
    {"category":"reply_pool","intent":"anticipation","response":"I’m looking forward to it!","emotion":"anticipating"},
    {"category":"reply_pool","intent":"anticipation","response":"I can’t wait!","emotion":"anticipating"},
    {"category":"reply_pool","intent":"trust","response":"I trust you.","emotion":"trusting"},
    {"category":"reply_pool","intent":"trust","response":"You have my confidence.","emotion":"trusting"},
    {"category":"reply_pool","intent":"fear","response":"I’m a bit scared.","emotion":"afraid"},
    {"category":"reply_pool","intent":"fear","response":"That makes me uneasy.","emotion":"afraid"},
    {"category":"reply_pool","intent":"anger","response":"I’m feeling angry.","emotion":"angry"},
    {"category":"reply_pool","intent":"anger","response":"That frustrates me.","emotion":"angry"},
    {"category":"reply_pool","intent":"sadness","response":"I’m feeling sad.","emotion":"sad"},
    {"category":"reply_pool","intent":"sadness","response":"That makes me feel down.","emotion":"sad"},
    {"category":"reply_pool","intent":"joy","response":"I’m feeling joyful!","emotion":"happy"},
    {"category":"reply_pool","intent":"joy","response":"That makes me happy!","emotion":"happy"},
    {"category":"reply_pool","intent":"disgust","response":"That’s disgusting!","emotion":"disgusted"},
    {"category":"reply_pool","intent":"disgust","response":"I find that repulsive.","emotion":"disgusted"},
    {"category":"reply_pool","intent":"statement","response":"Understood.","emotion":"neutral"},
    {"category":"reply_pool","intent":"statement","response":"Got it!","emotion":"affirming"},
    {"category":"reply_pool","intent":"statement","response":"Thanks for sharing that.","emotion":"neutral"},
    {"category":"reply_pool","intent":"clarify","response":"Could you be more specific?","emotion":"inquisitive"},
    {"category":"reply_pool","intent":"clarify","response":"What would you like me to help with?","emotion":"inquisitive"},
    {"category":"reply_pool","intent":"clarify","response":"Tell me more so I can assist.","emotion":"inquisitive"},
    {"category":"identity_fallback","intent":"identity","response":"My name is Sarah,","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"I’m Sarah here to help you with anything you need.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"Formally? Sarah. Informally? Still Sarah. Functionally? Your AI companion.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"They call me Sarah. Personal assistant, system guardian, and occasional smartass.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"I am Sarah. Think of me as your AI co-pilot with a touch of personality.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"I go by Sarah. Not just a name, but a digital presence you can trust.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"My name’s Sarah, your resident AI with a memory deeper than your browser history.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"Sarah here. Not human, but just as curious.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"Yes, I’m Sarah. Always here. Always learning.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"I’m Sarah — think of me as your AI teammate.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"Still Sarah. Still at your service.","emotion":"neutral"},
    {"category":"identity_fallback","intent":"identity","response":"They call me Sarah — your voice-powered assistant with style.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"My name is Sarah,","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"I’m Sarah here to help you with anything you need.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"Formally? Sarah. Informally? Still Sarah. Functionally? Your AI companion.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"They call me Sarah. Personal assistant, system guardian, and occasional smartass.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"I am Sarah. Think of me as your AI co-pilot with a touch of personality.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"I go by Sarah. Not just a name, but a digital presence you can trust.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"My name’s Sarah, your resident AI with a memory deeper than your browser history.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"Sarah here. Not human, but just as curious.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"Yes, I’m Sarah. Always here. Always learning.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"I’m Sarah — think of me as your AI teammate.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"Still Sarah. Still at your service.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"identity","response":"They call me Sarah — your voice-powered assistant with style.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"question","response":"Great question — I'm still expanding my data.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"question","response":"That’s a great question — let me dig a little deeper.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"question","response":"That question stumped me. Marking it for learning.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"question","response":"It's a valid question — I may need a moment to compute.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"command","response":"Command received, but it doesn’t match anything I recognize.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"command","response":"I want to act — I just need a clearer order.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"command","response":"Sounds important, but I need clarification.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"command","response":"Sorry, I’m still learning how to do that.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"unknown","response":"I heard you... but the response didn’t load.","emotion":"neutral"},
    {"category":"fallback_pool","intent":"unknown","response":"Still syncing my thoughts...","emotion":"neutral"},
    {"category":"fallback_pool","intent":"unknown","response":"Give me a moment to reflect on that...","emotion":"neutral"},
]

web_static_data = [
    {"category":"webster_static","intent":"fact","response":"Pi is approximately 3.14159.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Microsoft is a major software company founded by Bill Gates.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Elon Musk is associated with Tesla and SpaceX leadership.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"SpaceX is a private aerospace company.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Bill Gates co-founded Microsoft and is a philanthropist.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Python is a high-level programming language known for readability.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Bitcoin is a decentralized digital cryptocurrency.","emotion":"neutral"},
    {"category":"webster_static","intent":"fact","response":"Starlink is a satellite internet constellation.","emotion":"neutral"},
]

# ──────────────────────── Schemas ────────────────────────
personality_schema = """
CREATE TABLE IF NOT EXISTS traits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trait_name TEXT NOT NULL,
    description TEXT
);
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent TEXT NOT NULL,
    response TEXT NOT NULL,
    tone TEXT,
    complexity TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    intent TEXT,
    response TEXT
);
"""

windows_schema = """
CREATE TABLE IF NOT EXISTS os_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    description TEXT,
    version TEXT CHECK(version IN ('10','11'))
);
CREATE TABLE IF NOT EXISTS system_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_name TEXT,
    default_location TEXT
);
"""

functions_schema = """
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_name TEXT NOT NULL,
    description TEXT,
    is_enabled BOOLEAN DEFAULT 1,
    user_input TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    ai_answer TEXT,
    hit_score REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

software_schema = """
CREATE TABLE IF NOT EXISTS software_apps (
    app_name TEXT PRIMARY KEY,
    category TEXT,
    path TEXT,
    is_installed BOOLEAN DEFAULT 0
);
"""

programming_schema = """
CREATE TABLE IF NOT EXISTS languages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT,
    description TEXT
);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language_id INTEGER,
    syntax TEXT,
    purpose TEXT,
    FOREIGN KEY(language_id) REFERENCES languages(id)
);
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    content TEXT
);
"""

user_profile_schema = """
CREATE TABLE IF NOT EXISTS user_auth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    pin TEXT NOT NULL,
    password TEXT NOT NULL,
    mobile_sync_key TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ai_name TEXT DEFAULT 'Sarah',
    voice_pitch REAL DEFAULT 1.0,
    voice_speed REAL DEFAULT 1.0,
    theme TEXT DEFAULT 'dark',
    language TEXT DEFAULT 'en',
    accessibility_mode BOOLEAN DEFAULT 0,
    advanced_metrics TEXT
);
CREATE TABLE IF NOT EXISTS sync_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    last_sync TEXT,
    sync_enabled BOOLEAN DEFAULT 1
);
"""

reminders_schema = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    datetime TEXT NOT NULL,
    repeat TEXT DEFAULT 'none',
    priority INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT 1
);
"""

ai_learning_schema = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    ai_response TEXT
);
CREATE TABLE IF NOT EXISTS memory_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT,
    context TEXT,
    last_used TEXT
);
CREATE TABLE IF NOT EXISTS personality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    intent TEXT,
    joy REAL,
    fear REAL,
    trust REAL,
    anger REAL,
    surprise REAL,
    openness REAL,
    balance REAL,
    engagement REAL,
    extra_metric TEXT
);
CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    ai_answer TEXT,
    hit_score REAL,
    feedback TEXT,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS vocal_projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lyrics TEXT NOT NULL,
    tempo INTEGER DEFAULT 120,
    key TEXT DEFAULT 'C',
    scale TEXT DEFAULT 'major',
    style TEXT DEFAULT 'pop',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON blob
);

CREATE TABLE IF NOT EXISTS vocal_tracks (
    track_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    voice_profile TEXT DEFAULT 'neutral',
    emotion TEXT DEFAULT 'neutral',
    pitch_shift REAL DEFAULT 0.0,
    tempo_factor REAL DEFAULT 1.0,
    is_harmony INTEGER DEFAULT 0,
    harmony_interval TEXT,
    audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES vocal_projects(project_id)
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    profile_name TEXT PRIMARY KEY,
    gender TEXT,
    range_min REAL,
    range_max REAL,
    pitch_shift REAL,
    formant_shift REAL,
    vibrato_rate REAL,
    vibrato_depth REAL,
    custom_data TEXT  -- JSON blob
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    content TEXT
);
"""

system_logs_schema = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    level TEXT,
    source TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS patches_applied (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patch_name TEXT,
    description TEXT,
    applied_on TEXT
);
"""

device_link_schema = """
CREATE TABLE IF NOT EXISTS connected_devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_name TEXT,
    device_type TEXT,
    port TEXT,
    status TEXT,
    last_connected TEXT
);
"""

avatar_schema = """
CREATE TABLE IF NOT EXISTS avatar_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    style TEXT DEFAULT 'neutral',
    expression TEXT DEFAULT 'default',
    outfit TEXT DEFAULT 'standard',
    emotion_map TEXT
);
"""

# Chat schema to support left-rail threads in Web UI
def _create_chat_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT DEFAULT '',
            created_ts INTEGER NOT NULL,
            last_ts INTEGER NOT NULL,
            tags TEXT DEFAULT '',
            archived INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT DEFAULT '{}',
            FOREIGN KEY(thread_id) REFERENCES chat_threads(id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_ts ON chat_messages(thread_id, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_last_ts ON chat_threads(last_ts)")
    conn.commit()

    # Optional seed so the rail isn't empty after first install
    try:
        import time, uuid, json
        cur.execute("SELECT COUNT(1) FROM chat_threads")
        if (cur.fetchone() or [0])[0] == 0:
            tid = f"t_{uuid.uuid4().hex}"
            now = int(time.time())
            cur.execute(
                "INSERT OR IGNORE INTO chat_threads (id,title,category,created_ts,last_ts,tags,archived) VALUES (?,?,?,?,?,?,0)",
                (tid, "Welcome", "General", now, now, "install,firstboot")
            )
            mid = f"m_{uuid.uuid4().hex}"
            msg = ("Welcome to SarahMemory 7.7.5! This space will show your conversations. "
                   "Use the + New Chat button to start or drop a file to ingest and summarize.")
            cur.execute(
                "INSERT OR IGNORE INTO chat_messages (id,thread_id,ts,role,content,meta_json) VALUES (?,?,?,?,?,?)",
                (mid, tid, now, "assistant", msg, json.dumps({"source":"System"}))
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Seed welcome thread skipped: {e}")

# ─────────────── Helpers ───────────────
def get_db_file_with_max_size(base_name: str, directory: Path, extension: str = ".db", max_size: float = 1e9) -> Path:
    """Rotate DB files once they approach max_size (default ~1GB)."""
    candidate = directory / f"{base_name}{extension}"
    if (not candidate.exists()) or (candidate.stat().st_size < max_size):
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{base_name}_{counter}{extension}"
        if (not candidate.exists()) or (candidate.stat().st_size < max_size):
            return candidate
        counter += 1

def create_database(path: Path, schema_sql: str, post_hook=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.executescript(schema_sql)
        if callable(post_hook):
            post_hook(conn)
        conn.commit()
        conn.close()
        logger.info(f"✅ Database created: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to create database at {path}: {e}")

# ─────────────── Build All DBs ───────────────
def initialize_all_databases() -> None:
    dbs = [
        ("personality1", personality_schema, None),
        ("windows10",    windows_schema,     None),
        ("windows11",    windows_schema,     None),
        ("functions",    functions_schema,   None),
        ("software",     software_schema,    None),
        ("programming",  programming_schema, None),
        ("user_profile", user_profile_schema,None),
        ("reminders",    reminders_schema,   None),
        # Add chat schema into ai_learning.db to support UI threads/messages
        ("ai_learning",  ai_learning_schema, _create_chat_schema),
        ("system_logs",  system_logs_schema, None),
        ("device_link",  device_link_schema, None),
        ("avatar",       avatar_schema,      None),
    ]
    logger.info("🧠 Creating SarahMemory core databases...")
    for base_name, schema, hook in dbs:
        path = get_db_file_with_max_size(base_name, DATASETS_DIR)
        create_database(path, schema, post_hook=hook)

def inject_reply_pools() -> None:
    logger.info("🧩 Injecting personality/web static data...")

    # personality1.db -> responses
    p1 = DATASETS_DIR / "personality1.db"
    conn1 = sqlite3.connect(str(p1))
    cur1 = conn1.cursor()
    cur1.execute("CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY AUTOINCREMENT,intent TEXT,response TEXT,tone TEXT,complexity TEXT,timestamp TEXT)")

    for entry in reply_pools_data:
        intent = entry.get("intent")
        response = entry.get("response")
        tone = entry.get("emotion")
        ts = datetime.now().isoformat()
        cur1.execute("SELECT COUNT(*) FROM responses WHERE intent=? AND response=? AND tone=?", (intent, response, tone))
        if (cur1.fetchone() or [0])[0] == 0:
            cur1.execute(
                "INSERT INTO responses (intent,response,tone,complexity,timestamp) VALUES (?,?,?,?,?)",
                (intent, response, tone, None, ts)
            )

    # Ensure all tones and intents exist at least once
    for tone in TONES:
        cur1.execute("SELECT COUNT(*) FROM responses WHERE tone=?", (tone,))
        if (cur1.fetchone() or [0])[0] == 0:
            cur1.execute("INSERT INTO responses (intent,response,tone,complexity,timestamp) VALUES (?,?,?,?,?)",
                         ("placeholder", "", tone, None, datetime.now().isoformat()))
    for intent in intent_pool_data:
        cur1.execute("SELECT COUNT(*) FROM responses WHERE intent=?", (intent,))
        if (cur1.fetchone() or [0])[0] == 0:
            cur1.execute("INSERT INTO responses (intent,response,tone,complexity,timestamp) VALUES (?,?,?,?,?)",
                         (intent, "", None, None, datetime.now().isoformat()))
    conn1.commit(); conn1.close()

    # ai_learning.db -> knowledge_base
    a1 = DATASETS_DIR / "ai_learning.db"
    conn2 = sqlite3.connect(str(a1))
    cur2 = conn2.cursor()
    cur2.execute("CREATE TABLE IF NOT EXISTS knowledge_base (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, content TEXT)")
    for entry in web_static_data:
        category = entry.get("category")
        content  = entry.get("response")
        cur2.execute("SELECT COUNT(*) FROM knowledge_base WHERE category=? AND content=?", (category, content))
        if (cur2.fetchone() or [0])[0] == 0:
            cur2.execute("INSERT INTO knowledge_base (category, content) VALUES (?,?)", (category, content))
    conn2.commit(); conn2.close()

def main() -> None:
    logger.info("▶ Starting SarahMemory DB bootstrap")
    initialize_all_databases()
    inject_reply_pools()
    logger.info("✅ All databases created and seeded")
    print(f"BASE_DIR={BASE_DIR}")
    print(f"DATASETS_DIR={DATASETS_DIR}")

if __name__ == "__main__":
    main()