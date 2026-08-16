"""--==The SarahMemory Project==--
File: SarahMemoryCleanup.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
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
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "maintenance_tool"
# CATEGORY = "cleanup_and_restore"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "classic_ui"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "cleanup_gui"
# FAMILY = "maintenance"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "User-facing cleanup, backup, restore, and log maintenance GUI for databases and local data retention management."
# --- SARAHMETA END ---

import os, sqlite3, shutil, time, traceback, json, re, sys
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ARILE protected cleanup/update guard. Protected immune-system files are never
# cleanup/update targets from automated maintenance lanes.
try:
    from SarahMemoryARILE import arile_is_protected_core_file, arile_emit
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    def arile_is_protected_core_file(path_value):
        try:
            return Path(path_value).name.lower() in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}
        except Exception:
            return False

def _arile_block_if_protected_target(path_value: str, operation: str = "maintenance") -> None:
    if arile_is_protected_core_file(path_value):
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, kind="protected_core_variance", failure_type="maintenance_target_blocked", severity=0.92, confidence=0.90, risk="critical", summary=f"Blocked {operation} against protected core file.", requires_governance=True, retention="security_audit", data={"target": str(path_value)})
        except Exception:
            pass
        raise PermissionError(f"Protected core maintenance target blocked: {path_value}")


try:
    from PIL import Image, ImageTk  # optional for icons
except Exception:
    Image = ImageTk = None

try:
    import SarahMemoryGlobals as config
except Exception:
    class config:
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MEMORY_DIR = os.path.join(DATA_DIR, "memory")
        DATASETS_DIR = os.path.join(MEMORY_DIR, "datasets")
        LOGS_DIR = os.path.join(DATA_DIR, "logs")

BACKUP_DIR = os.path.join(config.DATASETS_DIR, "_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

DBS = {
    "context_history.db": {
        "path": os.path.join(config.DATASETS_DIR, "context_history.db"),
        "ranges": [("context_history","timestamp")]
    },
    "ai_learning.db": {
        "path": os.path.join(config.DATASETS_DIR, "ai_learning.db"),
        "ranges": [("intent_logs","timestamp")]
    },
    "personality1.db": {
        "path": os.path.join(config.DATASETS_DIR, "personality1.db"),
        "ranges": [("emotion_states","timestamp")]  # `responses` may lack timestamp; cleared on ALL only
    },
    "functions.db": {
        "path": os.path.join(config.DATASETS_DIR, "functions.db"),
        "ranges": [("dl_cache","ts")]
    },
    "system_logs.db": {
        "path": os.path.join(config.DATASETS_DIR, "system_logs.db"),
        "ranges": [("events","timestamp"), ("response","timestamp"), ("responses","timestamp")]
    },
}


def _cleanup_audit_event(event_type: str, payload: dict) -> None:
    """Write a compact cleanup accountability record; never deletes truth records."""
    try:
        audit_dir = os.path.join(getattr(config, "DATA_DIR", os.getcwd()), "audit", "cleanup")
        os.makedirs(audit_dir, exist_ok=True)
        path = os.path.join(audit_dir, "cleanup_events.jsonl")
        row = {"event_type": event_type, "timestamp": datetime.utcnow().isoformat(), **(payload or {})}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _safe_sql_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# QA CACHE POISON QUARANTINE - v9 Semantic Memory Hygiene
# ---------------------------------------------------------------------------
# These helpers do not perform broad filesystem scans. They inspect known
# project SQLite DBs only, backup before mutation, quarantine before delete, and
# write an audit event.  This is cleanup, not answer routing authority.

def _qa_cache_tables(cur) -> list[str]:
    try:
        rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    except Exception:
        return []
    names = []
    for r in rows or []:
        name = str(r[0] or "")
        if name.lower() in {"qa_cache", "qacache", "sm_qa_cache"}:
            names.append(name)
    return names


def _qa_cache_columns(cur, table: str) -> dict:
    try:
        cols = [str(r[1]) for r in cur.execute(f"PRAGMA table_info({_safe_sql_identifier(table)})").fetchall()]
    except Exception:
        cols = []
    low = {c.lower(): c for c in cols}
    return {
        "all": cols,
        "query": low.get("query") or low.get("question") or low.get("prompt") or low.get("user_input") or low.get("input"),
        "answer": low.get("ai_answer") or low.get("answer") or low.get("response") or low.get("reply") or low.get("content") or low.get("output"),
        "timestamp": low.get("timestamp") or low.get("created_at") or low.get("ts"),
    }


def _normal_text(value) -> str:
    try:
        return re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    except Exception:
        return str(value or "").strip()


def _definition_subject(query: str) -> str:
    q = _normal_text(query).lower().strip(" ?.!")
    for pat in (
        r"^what\s+is\s+an?\s+(.+)$",
        r"^what\s+is\s+(.+)$",
        r"^what\s+are\s+(.+)$",
        r"^define\s+(.+)$",
        r"^explain\s+(.+)$",
        r"^describe\s+(.+)$",
        r"^tell\s+me\s+about\s+(.+)$",
    ):
        m = re.match(pat, q)
        if m:
            return re.sub(r"\b(the|a|an)\b", " ", m.group(1)).strip()
    return ""


def _qa_poison_reason(query: str, answer: str) -> str:
    q = _normal_text(query).lower()
    raw = str(answer or "").strip()
    a = raw.lower()
    if not raw:
        return "empty_answer"
    if any(x in a for x in ("request denied by policy", "user confirmation required", "could you rephrase", "please provide more details", "unable to answer")):
        return "cached_failure_or_policy_text"
    paper_hits = sum(1 for x in ("abstract", "1. introduction", "1 introduction", "references", "bibliography", "we present", "we propose", "this paper", "doi", "arxiv", "@", "compiler from llvm", "mozilla") if x in a)
    if paper_hits >= 2:
        return "paper_or_corpus_blob"
    if len(raw) > 1200 and _definition_subject(q):
        return "oversized_definition_cache_answer"
    if "python" in q and (a.startswith("emscripten") or "llvm-to-javascript compiler" in a):
        return "wrong_topic_python_emscripten_blob"
    subject = _definition_subject(q)
    if subject and len(raw) > 120:
        first = _normal_text(raw[:320]).lower()
        terms = [x for x in re.findall(r"[a-z0-9]+", subject) if len(x) > 1]
        if terms and not any(t in first for t in terms[:3]):
            return "definition_subject_missing_from_opening"
    return ""


def scan_poisoned_qa_cache(*, db_names: list[str] | None = None, max_rows: int = 5000) -> dict:
    """Read-only scan for poisoned QA cache rows in known local SQLite DBs."""
    names = db_names or ["ai_learning.db", "ailearning.db", "personality1.db", "context_history.db", "functions.db"]
    report = {"ok": True, "mode": "read_only", "poisoned": [], "checked_rows": 0, "errors": [], "timestamp": datetime.utcnow().isoformat()}
    for dbname in names:
        db_path = os.path.join(config.DATASETS_DIR, dbname)
        if not os.path.isfile(db_path):
            continue
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
            cur = conn.cursor()
            for table in _qa_cache_tables(cur):
                cols = _qa_cache_columns(cur, table)
                qcol, acol = cols.get("query"), cols.get("answer")
                if not qcol or not acol:
                    continue
                sql = f"SELECT rowid, {_safe_sql_identifier(qcol)}, {_safe_sql_identifier(acol)} FROM {_safe_sql_identifier(table)} LIMIT ?"
                for rowid, query, answer in cur.execute(sql, (max(1, int(max_rows)),)).fetchall():
                    report["checked_rows"] += 1
                    reason = _qa_poison_reason(str(query or ""), str(answer or ""))
                    if reason:
                        report["poisoned"].append({
                            "db": dbname,
                            "table": table,
                            "rowid": int(rowid),
                            "query": _normal_text(query)[:240],
                            "reason": reason,
                            "answer_preview": _normal_text(answer)[:260],
                        })
        except Exception as exc:
            report["errors"].append(f"{dbname}: {exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
    report["poison_count"] = len(report["poisoned"])
    return report


def _backup_db_for_quarantine(db_path: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(BACKUP_DIR, f"poison-quarantine-{ts}")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, os.path.basename(db_path))
    shutil.copy2(db_path, dst)
    return dst


def quarantine_poisoned_qa_cache(*, dry_run: bool = True, max_rows: int = 5000, backup: bool = True) -> dict:
    """Quarantine poisoned QA rows. Defaults to dry-run; never deletes without quarantine."""
    scan = scan_poisoned_qa_cache(max_rows=max_rows)
    out = {"ok": True, "dry_run": bool(dry_run), "scan": scan, "quarantined": 0, "deleted": 0, "backups": [], "errors": [], "timestamp": datetime.utcnow().isoformat()}
    if dry_run or not scan.get("poisoned"):
        _cleanup_audit_event("qa_poison_scan", {"dry_run": True, "poison_count": scan.get("poison_count", 0)})
        return out

    by_db: dict[str, list[dict]] = {}
    for item in scan.get("poisoned") or []:
        by_db.setdefault(str(item.get("db")), []).append(item)

    for dbname, items in by_db.items():
        db_path = os.path.join(config.DATASETS_DIR, dbname)
        if not os.path.isfile(db_path):
            continue
        conn = None
        try:
            if backup:
                out["backups"].append(_backup_db_for_quarantine(db_path))
            conn = sqlite3.connect(db_path, timeout=2.0)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_cache_poison_quarantine (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_db TEXT,
                    source_table TEXT,
                    source_rowid INTEGER,
                    query TEXT,
                    ai_answer TEXT,
                    reason TEXT,
                    quarantined_at TEXT
                )
            """)
            for item in items:
                table = str(item.get("table") or "")
                rowid = int(item.get("rowid") or 0)
                if not table or rowid <= 0:
                    continue
                cols = _qa_cache_columns(cur, table)
                qcol, acol = cols.get("query"), cols.get("answer")
                if not qcol or not acol:
                    continue
                row = cur.execute(f"SELECT {_safe_sql_identifier(qcol)}, {_safe_sql_identifier(acol)} FROM {_safe_sql_identifier(table)} WHERE rowid=?", (rowid,)).fetchone()
                if not row:
                    continue
                cur.execute(
                    "INSERT INTO qa_cache_poison_quarantine (source_db, source_table, source_rowid, query, ai_answer, reason, quarantined_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (dbname, table, rowid, str(row[0] or ""), str(row[1] or ""), str(item.get("reason") or ""), datetime.utcnow().isoformat()),
                )
                cur.execute(f"DELETE FROM {_safe_sql_identifier(table)} WHERE rowid=?", (rowid,))
                out["quarantined"] += 1
                out["deleted"] += 1
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        except Exception as exc:
            out["errors"].append(f"{dbname}: {exc}")
            try:
                if conn is not None:
                    conn.rollback()
            except Exception:
                pass
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
    _cleanup_audit_event("qa_poison_quarantine", {"dry_run": False, "quarantined": out["quarantined"], "deleted": out["deleted"], "backups": out["backups"], "errors": out["errors"]})
    return out


RANGES = [
    ("5 minutes", 5*60),
    ("10 minutes", 10*60),
    ("30 minutes", 30*60),
    ("1 hour", 60*60),
    ("3 hours", 3*60*60),
    ("5 hours", 5*60*60),
    ("12 hours", 12*60*60),
    ("1 day", 24*60*60),
    ("3 days", 3*24*60*60),
    ("1 week", 7*24*60*60),
    ("1 month (~30d)", 30*24*60*60),
    ("3 months", 90*24*60*60),
    ("6 months", 180*24*60*60),
    ("1 year", 365*24*60*60),
    ("ALL DATA", None),
]

def backup_all():
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    bundle = os.path.join(BACKUP_DIR, f"backup-{ts}")
    os.makedirs(bundle, exist_ok=True)
    manifest = {"created_at": datetime.utcnow().isoformat(), "files": [], "purpose": "SarahMemoryCleanup pre-change backup"}
    for name, meta in DBS.items():
        src = meta["path"]
        if os.path.exists(src):
            dst = os.path.join(bundle, name)
            shutil.copy2(src, dst)
            manifest["files"].append({"name": name, "source": src, "backup": dst, "bytes": os.path.getsize(dst)})
    # copy logs too
    if os.path.isdir(config.LOGS_DIR):
        zipname = os.path.join(bundle, "logs.zip")
        shutil.make_archive(zipname[:-4], "zip", config.LOGS_DIR)
        manifest["files"].append({"name": "logs.zip", "source": config.LOGS_DIR, "backup": zipname})
    with open(os.path.join(bundle, "backup_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    _cleanup_audit_event("backup_created", {"bundle": bundle, "files": len(manifest["files"])})
    messagebox.showinfo("Backup", f"Backup created: {bundle}")
    return bundle

def restore_backup():
    folder = filedialog.askdirectory(initialdir=BACKUP_DIR, title="Select backup folder")
    if not folder:
        return
    restored = []
    for name, meta in DBS.items():
        src = os.path.join(folder, name)
        dst = meta["path"]
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(name)
    messagebox.showinfo("Restore", "Restored: " + ", ".join(restored) if restored else "No DB files in chosen backup.")

def clear_range(seconds=None):
    now = datetime.utcnow()
    cutoff = None if seconds is None else now - timedelta(seconds=seconds)
    # SARAHMEMORY_PATCH_NOTE 2026-06-28:
    # Cleanup is now accountable and survivable. Any destructive range clear
    # creates a backup first and records an audit event.
    backup_bundle = backup_all()
    _cleanup_audit_event("clear_range_requested", {"seconds": seconds, "cutoff": cutoff.isoformat() if cutoff else "ALL_DATA", "backup_bundle": backup_bundle})
    for dbname, meta in DBS.items():
        path = meta["path"]
        if not os.path.exists(path):
            continue
        try:
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                if seconds is None:
                    # Clear everything safely
                    for table, _tscol in meta["ranges"]:
                        try:
                            cur.execute(f"DELETE FROM {_safe_sql_identifier(table)}")
                        except Exception:
                            pass
                    # handle responses in personality if present
                    try:
                        cur.execute(f"DELETE FROM {_safe_sql_identifier("responses")}")
                    except Exception:
                        pass
                else:
                    for table, tscol in meta["ranges"]:
                        try:
                            # Only delete where timestamp column exists
                            cur.execute(f"PRAGMA table_info({_safe_sql_identifier(table)})")
                            cols = [r[1] for r in cur.fetchall()]
                            if tscol in cols:
                                cur.execute(f"DELETE FROM {_safe_sql_identifier(table)} WHERE {_safe_sql_identifier(tscol)} >= ?", (cutoff.isoformat(),))
                        except Exception:
                            pass
                con.commit()
                con.execute("VACUUM")
        except Exception as e:
            print("[Cleanup] Failed clearing", dbname, ":", e)
    messagebox.showinfo("Cleanup", "Cleanup completed.")

def tidy_logs():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    for fn in os.listdir(config.LOGS_DIR):
        p = os.path.join(config.LOGS_DIR, fn)
        try:
            if os.path.isfile(p) and (fn.lower().endswith(".log") or fn.lower().endswith(".txt")):
                # Truncate if > 5MB
                if os.path.getsize(p) > 5*1024*1024:
                    with open(p, "rb+") as f:
                        f.seek(-1024*1024, os.SEEK_END)
                        tail = f.read()
                        f.seek(0); f.truncate()
                        f.write(b"...[truncated]\n" + tail)
        except Exception as e:
            print("[Cleanup] tidy_logs:", e)
    messagebox.showinfo("Logs", "Logs tidied.")

def launch_cleanup_gui():
    root = tk.Tk()
    root.title("SarahMemory Cleanup & Restore")
    root.geometry("520x520")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Select a time range to clear across DBs").pack(pady=6)

    range_var = tk.StringVar(value=RANGES[0][0])
    combo = ttk.Combobox(frm, textvariable=range_var, values=[r[0] for r in RANGES], state="readonly")
    combo.pack(pady=6, fill="x")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=10)
    ttk.Button(btns, text="Create Backup", command=backup_all).pack(side="left", padx=4)
    ttk.Button(btns, text="Restore Backup", command=restore_backup).pack(side="left", padx=4)
    ttk.Button(btns, text="Tidy Logs", command=tidy_logs).pack(side="left", padx=4)

    def on_clear():
        label = range_var.get()
        seconds = next((s for lbl, s in RANGES if lbl == label), None)
        if messagebox.askyesno("Confirm", f"Proceed to clear: {label}?"):
            clear_range(seconds)

    def on_scan_poison():
        report = scan_poisoned_qa_cache(max_rows=5000)
        messagebox.showinfo("Poison Cache Scan", f"Checked rows: {report.get('checked_rows', 0)}\nPoison rows: {report.get('poison_count', 0)}")

    def on_quarantine_poison():
        if messagebox.askyesno("Confirm", "Backup and quarantine poisoned QA cache rows? This does not vacuum databases."):
            report = quarantine_poisoned_qa_cache(dry_run=False, max_rows=5000, backup=True)
            messagebox.showinfo("Poison Cache Quarantine", f"Quarantined: {report.get('quarantined', 0)}\nBackups: {len(report.get('backups') or [])}\nErrors: {len(report.get('errors') or [])}")

    ttk.Button(frm, text="SCAN POISONED QA CACHE", command=on_scan_poison).pack(pady=4, fill="x")
    ttk.Button(frm, text="QUARANTINE POISONED QA CACHE", command=on_quarantine_poison).pack(pady=4, fill="x")

    ttk.Button(frm, text="CLEAR SELECTED RANGE", command=on_clear).pack(pady=16, fill="x")

    ttk.Label(frm, text="DB Directory: " + config.DATASETS_DIR).pack(anchor="w", pady=6)
    ttk.Label(frm, text="Logs Directory: " + config.LOGS_DIR).pack(anchor="w")

    root.mainloop()

if __name__ == "__main__":
    if "--scan-poison" in sys.argv:
        print(json.dumps(scan_poisoned_qa_cache(max_rows=5000), indent=2, ensure_ascii=False, default=str))
    elif "--quarantine-poison" in sys.argv:
        print(json.dumps(quarantine_poisoned_qa_cache(dry_run=False, max_rows=5000, backup=True), indent=2, ensure_ascii=False, default=str))
    elif "--dry-run-quarantine-poison" in sys.argv:
        print(json.dumps(quarantine_poisoned_qa_cache(dry_run=True, max_rows=5000, backup=True), indent=2, ensure_ascii=False, default=str))
    else:
        launch_cleanup_gui()

# ====================================================================
# END OF SarahMemoryCleanup.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCleanup',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Diagnostics',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['diagnostics', 'health_maintenance'],
    "supported_missions": ['Conversation', 'Diagnostics', 'Repair'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω090', 'Ω110'],
    "required_authority": ['Diagnostics', 'Read'],
    "priority": 85,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryCleanup.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCleanup',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCleanup', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

