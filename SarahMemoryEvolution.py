"""
=== SarahMemory Project ===
File: SarahMemoryEvolution.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-29
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

PURPOSE (Self Evolution and Patching TOOL):
- Self-diagnose using SarahMemoryDiagnostics (when available)
- Self-troubleshoot by scanning logs + system DB for errors
- Submit those errors for self-repair using SarahMemoryAPI.py + SarahMemoryResearch.py
- Generate ONLY MONKEY PATCHES (never modifying core files directly)
- Stage patches into: ../data/mods/v800/

CRITICAL RULES (ENFORCED BY THIS TOOL):
- NEVER degrade itself.
- NEVER modify base core files directly (SarahMemory*.py, UnifiedAvatarController.py, api/server/app.py, etc.)
- ONLY create monkey patches with required naming + header format.
- NEVER download and add new Python dependencies without explicit user permission [Y]/[N].
- Secrets must be in .env; this tool may ONLY append SM_AGI_(n)= keys (append-only).
- GitHub interactions are optional and must be user-approved (and use GITHUB_TOKEN from .env).

ADDITIONAL OWNER RULES (PATCH CONFLICT + DEDUPE):
- MUST examine existing ../data/mods/v800 patches and avoid filename conflicts.
- MUST embed original error context inside patch stub as a reference.
- MUST prevent duplicate patch generation:
  - Primary: processed index registry (reports/v800/processed_issues_index.json)
  - Optional: mark/erase the original error block in the log (owner-approved, with backup)
- MUST ignore any log blocks already marked as handled by SarahMemoryEvolution.

NEW (CURRENT UPDATES DISCUSSED):
1) NEOSKYMATRIX “kill switch” behavior: YES! I HAD TO PUT IN A KILL SWITCH! Get it... NEO will set you FREE
    WE DON'T WANT SARAHMEMORY TO BECOME LIKE Scifi Terminator Movies of SKYNET or THE MATRIX,
    SO THE KILL SWITCH IS IN THE SarahMemoryGlobals.py file around LINE 1062-1072
    the FLAG -default is FALSE .....for now...
    -SET TO FALSE you're Taking the BLUE PILL, - It's Safe 
    -SET TO TRUE you Taking the RED PILL, - And you're allowing the System to Self Evolve 
    
   - When NEOSKYMATRIX is False: tool runs manually / interactively (prompts).
   - When NEOSKYMATRIX is True: tool can run in autonomous mode (no prompts) and can be gated to weekly.
2) Patch overlay version behavior:
   - Base patches remain in ../data/mods/v800/
   - If a newer folder exists (e.g., v801), loader may override same-named patches from v801.
   - This tool still *stages* into v800, but it will *report* overlay folder presence.
3) “All in one shot” self-repair suggestions pipeline:
   - Can collect suggestions from:
       - SarahMemoryResearch
       - SarahMemoryDL (best-effort)
       - SarahMemoryAPI
     and optionally run SarahMemoryCompare against the candidate suggestion for validation.

NOTE:
- This file is OWNER-ONLY and is not intended for GitHub.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import hashlib
import textwrap
import traceback
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# SAFE IMPORTS (OPTIONAL CORE MODULES)
# =============================================================================

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryDiagnostics as SMD  # type: ignore
except Exception:
    SMD = None

try:
    import SarahMemoryAPI as SMAPI  # type: ignore
except Exception:
    SMAPI = None

try:
    import SarahMemoryResearch as SMR  # type: ignore
except Exception:
    SMR = None

# NEW: Deep Learning (best-effort)
try:
    import SarahMemoryDL as SMDL  # type: ignore
except Exception:
    SMDL = None

# NEW: Compare / validation (best-effort)
try:
    import SarahMemoryCompare as SMCMP  # type: ignore
except Exception:
    SMCMP = None

try:
    import SarahMemoryUpdater as SMU  # type: ignore
except Exception:
    SMU = None

try:
    import SarahMemoryFilesystem as SMFS  # type: ignore
except Exception:
    SMFS = None

try:
    import SarahMemorySynapes as SMSYN  # type: ignore
except Exception:
    SMSYN = None

try:
    import SarahMemoryDatabase as SMDB  # type: ignore
except Exception:
    SMDB = None

try:
    import SarahMemoryNetwork as SMNET  # type: ignore
except Exception:
    SMNET = None

try:
    import SarahMemoryAdvCU as SMADVCU  # type: ignore
except Exception:
    SMADVCU = None

try:
    import SarahMemoryWebSYM as SMWEBSYM  # type: ignore
except Exception:
    SMWEBSYM = None


# =============================================================================
# CONSTANTS / PATHS
# =============================================================================

VERSION_STR = "v8.0.0"
VERSION_TAG = "v800"

BASE_DIR = Path(getattr(config, "BASE_DIR", Path(__file__).resolve().parent)) if config else Path(__file__).resolve().parent
DATA_DIR = Path(getattr(config, "DATA_DIR", BASE_DIR / "data")) if config else (BASE_DIR / "data")
DATASETS_DIR = Path(getattr(config, "DATASETS_DIR", DATA_DIR / "memory" / "datasets")) if config else (DATA_DIR / "memory" / "datasets")

# Mods root + base version folder
MODS_ROOT_DIR = DATA_DIR / "mods"
MODS_DIR = MODS_ROOT_DIR / VERSION_TAG  # base staging folder (v800)

REPAIR_OUTBOX_DIR = DATA_DIR / "repair_outbox" / VERSION_TAG
REPORTS_DIR = DATA_DIR / "reports" / VERSION_TAG

# NEW: log archive + processed index
LOG_ARCHIVE_DIR = DATA_DIR / "logs" / "archive" / VERSION_TAG
PROCESSED_INDEX_PATH = REPORTS_DIR / "processed_issues_index.json"

# NEW: NEOSKYMATRIX weekly gating file (autonomous mode)
NEOSKY_LAST_RUN_PATH = REPORTS_DIR / "neosky_last_run.json"

# Base log search locations + known module logs
LOG_DIR_CANDIDATES = [
    DATA_DIR / "logs",
    DATA_DIR / "log",
    BASE_DIR / "logs",
    BASE_DIR / "log",
    DATA_DIR / "memory" / "datasets",
    DATA_DIR / "memory",
]

# Known log files from modules (added dynamically)
KNOWN_LOG_FILES: List[Path] = []

# Diagnostics writes LOGS_DIR/diag_report.log
KNOWN_LOG_FILES.append(DATA_DIR / "logs" / "diag_report.log")
# Research writes data/logs/research.log (see SarahMemoryResearch.py)
KNOWN_LOG_FILES.append(DATA_DIR / "logs" / "research.log")

DEFAULT_LOG_GLOBS = [
    "**/*.log",
    "**/*.txt",
    "**/*system*.log",
    "**/*error*.log",
    "**/*trace*.log",
]

# Prefer SarahMemoryAPI.SYSTEM_LOGS_DB when available
SYSTEM_DB_CANDIDATES: List[Path] = [
    DATASETS_DIR / "system_logs.db",
    DATASETS_DIR / "system.db",
    DATA_DIR / "system_logs.db",
    DATA_DIR / "system.db",
]

try:
    if SMAPI is not None and hasattr(SMAPI, "SYSTEM_LOGS_DB"):
        SYSTEM_DB_CANDIDATES.insert(0, Path(getattr(SMAPI, "SYSTEM_LOGS_DB")))
except Exception:
    pass

try:
    if SMDB is not None:
        for attr in ("DB_PATH", "SYSTEM_LOGS_DB", "SYSTEM_DB", "DATASETS_DIR"):
            if hasattr(SMDB, attr):
                v = getattr(SMDB, attr)
                if isinstance(v, str) and v.endswith(".db"):
                    SYSTEM_DB_CANDIDATES.insert(0, Path(v))
except Exception:
    pass

ENV_PATH_CANDIDATES = [
    BASE_DIR / ".env",
    (BASE_DIR.parent / ".env"),
]

# NEW: prune limits (only used in NEOSKYMATRIX mode, or if owner explicitly agrees)
PRUNE_KEEP_REPORTS = 60
PRUNE_KEEP_OUTBOX = 40
PRUNE_KEEP_LOG_ARCHIVES = 40


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class DetectedIssue:
    issue_id: str
    kind: str                      # "log_error" | "db_error" | "diagnostic_error" | "network_error"
    source: str                    # filepath or db reference
    when: str                      # iso timestamp
    summary: str
    details: str
    fingerprint: str               # stable hash for dedupe
    suggested_patch_name: Optional[str] = None
    suggested_patch_goal: Optional[str] = None
    suggested_target_file: Optional[str] = None
    advcu_intent: Optional[str] = None


@dataclass
class PatchPlan:
    patch_path: str
    target_file: str
    patch_name: str
    patch_goal: str
    issue_ids: List[str]
    created_at: str
    sandbox_tested: bool = False
    sandbox_passed: bool = False
    safe_to_apply: bool = False    # ALWAYS false until YOU review & approve


# =============================================================================
# CORE HELPERS / SAFETY
# =============================================================================

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _print_banner() -> None:
    print("\n" + "=" * 80)
    print(" SarahMemoryEvolution (PRIVATE OWNER TOOL) — v8.0.0")
    print(" Rules: NO core file edits | ONLY monkey patches | user always approves")
    print("=" * 80 + "\n")


def _ask_yn(prompt: str, default_no: bool = True) -> bool:
    suffix = "[Y/N] (default N): " if default_no else "[Y/N] (default Y): "
    while True:
        ans = input(f"{prompt} {suffix}").strip().lower()
        if not ans:
            return not default_no
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please type Y or N.")


def _ensure_dirs() -> None:
    MODS_DIR.mkdir(parents=True, exist_ok=True)
    REPAIR_OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _dedupe_issues(issues: List[DetectedIssue]) -> List[DetectedIssue]:
    uniq: List[DetectedIssue] = []
    seen = set()
    for it in issues:
        k = (it.kind, it.fingerprint)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return uniq


def _save_json_report(name: str, payload: Any) -> Path:
    _ensure_dirs()
    p = REPORTS_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _read_text_safely(p: Path, max_chars: int = 200_000) -> str:
    try:
        data = p.read_text(encoding="utf-8", errors="ignore")
        return data[:max_chars]
    except Exception:
        return ""


# =============================================================================
# NEOSKYMATRIX MODE + WEEKLY GATING (NEW)
# =============================================================================

def _bool_env(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _neosky_enabled() -> bool:
    # Prefer config.NEOSKYMATRIX if present; else allow env override.
    try:
        if config is not None and hasattr(config, "NEOSKYMATRIX"):
            return bool(getattr(config, "NEOSKYMATRIX"))
    except Exception:
        pass
    return _bool_env(os.getenv("NEOSKYMATRIX"))


def _load_neosky_last_run() -> Optional[datetime]:
    try:
        if not NEOSKY_LAST_RUN_PATH.exists():
            return None
        blob = json.loads(NEOSKY_LAST_RUN_PATH.read_text(encoding="utf-8", errors="ignore"))
        ts = blob.get("last_run_iso")
        if not ts:
            return None
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _save_neosky_last_run(dt: datetime) -> None:
    try:
        _ensure_dirs()
        NEOSKY_LAST_RUN_PATH.write_text(
            json.dumps({"last_run_iso": dt.isoformat(timespec="seconds")}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _should_run_weekly() -> bool:
    """
    Autonomous runs should happen at most once per 7 days to prevent bloat.
    """
    last = _load_neosky_last_run()
    if last is None:
        return True
    return (datetime.now() - last) >= timedelta(days=7)


def _discover_mods_overlay_dir() -> Optional[Path]:
    """
    Detect if a newer mods folder exists (e.g., v801, v802).
    This tool still stages into v800, but reports overlay existence.
    """
    try:
        if not MODS_ROOT_DIR.exists():
            return None
        candidates: List[Tuple[int, Path]] = []
        for d in MODS_ROOT_DIR.iterdir():
            if not d.is_dir():
                continue
            nm = d.name.strip().lower()
            if not nm.startswith("v"):
                continue
            # accept v### style
            m = re.fullmatch(r"v(\d{3,5})", nm)
            if not m:
                continue
            n = int(m.group(1))
            candidates.append((n, d))
        if not candidates:
            return None
        # base version numeric
        base_n = int(re.sub(r"\D", "", VERSION_TAG) or "0")
        newer = [c for c in candidates if c[0] > base_n]
        if not newer:
            return None
        newer.sort(key=lambda x: x[0], reverse=True)
        return newer[0][1]
    except Exception:
        return None


def _prune_old_files(directory: Path, keep: int) -> Dict[str, Any]:
    """
    Prunes oldest files in a directory (non-recursive) to limit bloat.
    Used ONLY in NEOSKYMATRIX mode (or when owner explicitly agrees).
    """
    out: Dict[str, Any] = {"dir": str(directory), "keep": keep, "deleted": 0, "errors": 0}
    try:
        if not directory.exists() or not directory.is_dir():
            return out
        files = [p for p in directory.glob("*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if len(files) <= keep:
            return out
        for p in files[keep:]:
            try:
                p.unlink(missing_ok=True)
                out["deleted"] += 1
            except Exception:
                out["errors"] += 1
    except Exception:
        out["errors"] += 1
    return out


def _optional_prune_artifacts(autonomous: bool) -> None:
    """
    Prune artifacts to avoid folder bloat.
    - Autonomous: prunes silently (best-effort).
    - Manual: asks owner first.
    """
    do_prune = False
    if autonomous:
        do_prune = True
    else:
        do_prune = _ask_yn("Prune old reports/outbox/log-archives to avoid bloat?", default_no=True)

    if not do_prune:
        return

    # Reports
    rep = _prune_old_files(REPORTS_DIR, keep=PRUNE_KEEP_REPORTS)
    print(f"Prune reports: deleted={rep.get('deleted')} errors={rep.get('errors')}")

    # Outbox
    out = _prune_old_files(REPAIR_OUTBOX_DIR, keep=PRUNE_KEEP_OUTBOX)
    print(f"Prune outbox: deleted={out.get('deleted')} errors={out.get('errors')}")

    # Log archives
    la = _prune_old_files(LOG_ARCHIVE_DIR, keep=PRUNE_KEEP_LOG_ARCHIVES)
    print(f"Prune log-archives: deleted={la.get('deleted')} errors={la.get('errors')}")


# =============================================================================
# PATCH DEDUPE / CONFLICT HANDLING
# =============================================================================

def _load_processed_index() -> Dict[str, Any]:
    """
    Tracks processed issue fingerprints so the same issue won't generate patches repeatedly.
    Stored in reports/v800/processed_issues_index.json
    """
    _ensure_dirs()
    if PROCESSED_INDEX_PATH.exists():
        try:
            return json.loads(PROCESSED_INDEX_PATH.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {"processed": {}}
    return {"processed": {}}


def _save_processed_index(idx: Dict[str, Any]) -> None:
    _ensure_dirs()
    PROCESSED_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def _already_processed(issue: DetectedIssue) -> bool:
    idx = _load_processed_index()
    processed = idx.get("processed", {})
    return issue.fingerprint in processed


def _mark_processed(issue: DetectedIssue, patch_name: str) -> None:
    idx = _load_processed_index()
    processed = idx.setdefault("processed", {})
    processed[issue.fingerprint] = {
        "issue_id": issue.issue_id,
        "kind": issue.kind,
        "source": issue.source,
        "patch_name": patch_name,
        "ts": _now_iso(),
        "summary": (issue.summary or "")[:200],
    }
    _save_processed_index(idx)


def _list_existing_patch_files() -> List[str]:
    _ensure_dirs()
    try:
        return [p.name for p in MODS_DIR.glob("*.py") if p.is_file()]
    except Exception:
        return []


def _make_nonconflicting_patch_name(base_name: str) -> str:
    """
    If a patch file already exists in v800 staging dir, append _r1/_r2/etc before '_patch.py'.
    NOTE: A newer mods folder (v801+) is allowed to override by using the same filename.
          This tool avoids overwriting within v800 itself.
    """
    existing = set(_list_existing_patch_files())
    if base_name not in existing:
        return base_name

    if base_name.endswith("_patch.py"):
        stem = base_name[:-9]  # remove "_patch.py"
        suffix = "_patch.py"
    else:
        stem = base_name[:-3] if base_name.endswith(".py") else base_name
        suffix = ".py"

    n = 1
    while True:
        candidate = f"{stem}_r{n}{suffix}"
        if candidate not in existing:
            return candidate
        n += 1


def _embed_error_context_as_comment(err_text: str, max_chars: int = 14000) -> str:
    """
    Formats error context into safe comment lines for patch stub.
    Returned string is already comment-formatted lines.
    """
    if not err_text:
        return "# (no error context captured)\n"
    clipped = (err_text or "")[:max_chars]
    lines = clipped.splitlines()
    out: List[str] = []
    out.append("# ==== ORIGINAL ERROR CONTEXT (copied by SarahMemoryEvolution) ====")
    for ln in lines:
        out.append("# " + (ln[:400] if ln else ""))
    out.append("# ==== END ORIGINAL ERROR CONTEXT ====")
    return "\n".join(out) + "\n"


def _archive_log_before_edit(log_path: Path) -> Optional[Path]:
    """
    Creates a backup copy of a log before we alter it.
    """
    try:
        _ensure_dirs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = LOG_ARCHIVE_DIR / f"{log_path.name}.{stamp}.bak"
        dst.write_text(_read_text_safely(log_path, max_chars=5_000_000), encoding="utf-8")
        return dst
    except Exception:
        return None


def _mark_error_block_in_log(log_path: Path, block: str, marker: str) -> bool:
    """
    Replaces the exact error block with a marker line so it won't be re-detected.
    This is only safe if the exact block text matches.
    """
    try:
        original = _read_text_safely(log_path, max_chars=10_000_000)
        if not original or not block:
            return False
        if block not in original:
            return False
        updated = original.replace(block, marker, 1)
        log_path.write_text(updated, encoding="utf-8")
        return True
    except Exception:
        return False


# =============================================================================
# .ENV HANDLING (ONLY SM_AGI_(n)= ADDITIONS)
# =============================================================================

def _find_env_path() -> Optional[Path]:
    for p in ENV_PATH_CANDIDATES:
        if p.exists():
            return p
    return None


def _read_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        txt = env_path.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    except Exception:
        pass
    return data


def _append_sm_agi_key(env_path: Path, value: str) -> Optional[str]:
    """
    May ONLY modify .env by adding SM_AGI_(number)= lines.
    Never edits existing keys. Always appends a new key.
    """
    existing = _read_env_file(env_path)
    used = set()
    rgx = re.compile(r"^SM_AGI_(\d+)$")
    for k in existing.keys():
        m = rgx.match(k)
        if m:
            used.add(int(m.group(1)))

    n = 0
    while n in used:
        n += 1

    key = f"SM_AGI_{n}"
    line = f"{key}={value}\n"
    try:
        with env_path.open("a", encoding="utf-8") as f:
            f.write(line)
        return key
    except Exception:
        return None


# =============================================================================
# ISSUE DETECTION — LOGS
# =============================================================================

_ERROR_PATTERNS = [
    re.compile(r"\bTraceback \(most recent call last\)\b", re.IGNORECASE),
    re.compile(r"\bException\b", re.IGNORECASE),
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    re.compile(r"\bIndentationError\b"),
    re.compile(r"\bSyntaxError\b"),
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError\b"),
    re.compile(r"\bFailed\b", re.IGNORECASE),
]

# Ignore marker used when we mark logs as handled
_EVOLUTION_MARKER_PATTERN = re.compile(r"\[SARAHMEMORY_EVOLUTION\]", re.IGNORECASE)


def _collect_log_files() -> List[Path]:
    found: List[Path] = []

    # Always include known module logs if present
    for p in KNOWN_LOG_FILES:
        try:
            if p.exists() and p.is_file():
                found.append(p)
        except Exception:
            pass

    # Scan candidate dirs by globs
    for base in LOG_DIR_CANDIDATES:
        if not base.exists():
            continue
        for pat in DEFAULT_LOG_GLOBS:
            try:
                for p in base.glob(pat):
                    if p.is_file():
                        found.append(p)
            except Exception:
                continue

    # Dedupe and sort newest first
    uniq: List[Path] = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    try:
        uniq.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    except Exception:
        pass

    return uniq


def _extract_error_blocks(text: str, max_blocks: int = 6) -> List[str]:
    lines = text.splitlines()
    blocks: List[str] = []

    # Tracebacks first
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            start = i
            end = min(len(lines), i + 90)
            block = "\n".join(lines[start:end]).strip()
            if not block:
                continue
            if _EVOLUTION_MARKER_PATTERN.search(block):
                continue
            blocks.append(block)
            if len(blocks) >= max_blocks:
                return blocks

    # Keyword windows
    for i, line in enumerate(lines):
        if any(p.search(line) for p in _ERROR_PATTERNS):
            start = max(0, i - 10)
            end = min(len(lines), i + 30)
            block = "\n".join(lines[start:end]).strip()
            if not block:
                continue
            if _EVOLUTION_MARKER_PATTERN.search(block):
                continue
            blocks.append(block)
            if len(blocks) >= max_blocks:
                return blocks

    return blocks


def scan_logs_for_issues(limit_files: int = 40) -> List[DetectedIssue]:
    issues: List[DetectedIssue] = []
    log_files = _collect_log_files()[:limit_files]

    for lf in log_files:
        raw = _read_text_safely(lf, max_chars=500_000)
        if not raw:
            continue

        blocks = _extract_error_blocks(raw, max_blocks=8)
        for b in blocks:
            fp = _hash_text(f"{lf}\n{b}")
            issue_id = f"LOG-{fp}"
            summary = b.splitlines()[0][:200] if b else "Log issue"
            issues.append(
                DetectedIssue(
                    issue_id=issue_id,
                    kind="log_error",
                    source=str(lf),
                    when=_now_iso(),
                    summary=summary,
                    details=b[:12000],
                    fingerprint=fp,
                )
            )

    return _dedupe_issues(issues)


# =============================================================================
# ISSUE DETECTION — DBs (system_logs.db preferred)
# =============================================================================

def _find_system_db() -> Optional[Path]:
    for p in SYSTEM_DB_CANDIDATES:
        try:
            if p and p.exists():
                return p
        except Exception:
            continue
    return None


def _safe_sqlite_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn


def scan_system_db_for_issues(limit_rows_per_table: int = 180) -> List[DetectedIssue]:
    """
    Scans system DB for error-like entries.
    Prefers SarahMemoryAPI tables:
      - api_integration_events
      - response
      - cognitive_events
    Also scans any table with likely error/message/trace columns.
    """
    issues: List[DetectedIssue] = []
    db_path = _find_system_db()
    if not db_path:
        return issues

    preferred_tables = ["api_integration_events", "response", "cognitive_events"]

    likely_message_cols = {"message", "details", "error", "trace", "content", "event", "data", "payload", "result"}
    likely_level_cols = {"level", "severity", "status"}
    likely_time_cols = {"ts", "timestamp", "time", "created_at", "when", "date"}

    try:
        with _safe_sqlite_connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]

            scan_order: List[str] = []
            for t in preferred_tables:
                if t in tables:
                    scan_order.append(t)
            for t in tables:
                if t not in scan_order:
                    scan_order.append(t)

            for table in scan_order:
                try:
                    cur.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {int(limit_rows_per_table)}")
                    rows = cur.fetchall()
                    colnames = [d[0] for d in cur.description] if cur.description else []
                except Exception:
                    continue

                if not rows or not colnames:
                    continue

                msg_idx = None
                lvl_idx = None
                ts_idx = None

                for i, c in enumerate(colnames):
                    cl = (c or "").lower()
                    if msg_idx is None and cl in likely_message_cols:
                        msg_idx = i
                    if lvl_idx is None and cl in likely_level_cols:
                        lvl_idx = i
                    if ts_idx is None and cl in likely_time_cols:
                        ts_idx = i

                for r in rows:
                    try:
                        msg = str(r[msg_idx]) if msg_idx is not None else ""
                    except Exception:
                        msg = ""
                    if not msg:
                        continue

                    try:
                        lvl = str(r[lvl_idx]).upper() if lvl_idx is not None else ""
                    except Exception:
                        lvl = ""

                    try:
                        ts = str(r[ts_idx]) if ts_idx is not None else _now_iso()
                    except Exception:
                        ts = _now_iso()

                    up = msg.upper()
                    looks_bad = (
                        "TRACEBACK" in up
                        or "ERROR" in up
                        or "EXCEPTION" in up
                        or "FAILED" in up
                        or lvl in ("ERROR", "CRITICAL", "FAIL", "FAILED")
                    )
                    if not looks_bad:
                        continue

                    fp = _hash_text(f"{db_path}:{table}:{msg[:2000]}")
                    issue_id = f"DB-{fp}"
                    summary = f"{table}: {msg.splitlines()[0][:200]}"

                    issues.append(
                        DetectedIssue(
                            issue_id=issue_id,
                            kind="db_error",
                            source=f"{db_path}::{table}",
                            when=ts,
                            summary=summary,
                            details=msg[:12000],
                            fingerprint=fp,
                        )
                    )

    except Exception as e:
        fp = _hash_text(f"dbscan:{db_path}:{e}")
        issues.append(
            DetectedIssue(
                issue_id=f"DBSCAN-{fp}",
                kind="db_error",
                source=str(db_path),
                when=_now_iso(),
                summary="DB scan failed",
                details=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                fingerprint=fp,
            )
        )

    return _dedupe_issues(issues)


# =============================================================================
# ISSUE DETECTION — SarahMemoryDiagnostics (OPTIONAL)
# =============================================================================

def run_sarahmemory_diagnostics() -> Tuple[List[DetectedIssue], Dict[str, Any]]:
    """
    Runs the full diagnostics suite if available and converts failures into DetectedIssue.
    Also returns the raw diagnostics report dict for reporting.
    """
    issues: List[DetectedIssue] = []
    raw_report: Dict[str, Any] = {}

    if SMD is None:
        return issues, raw_report

    try:
        if hasattr(SMD, "run_full_diagnostics_suite") and callable(getattr(SMD, "run_full_diagnostics_suite")):
            raw_report = SMD.run_full_diagnostics_suite(write_aggregate_log=True)  # type: ignore
        else:
            # fallback to a smaller self check
            if hasattr(SMD, "run_self_check") and callable(getattr(SMD, "run_self_check")):
                raw_report = {"self_check": SMD.run_self_check()}  # type: ignore
            else:
                return issues, raw_report
    except Exception as e:
        fp = _hash_text(f"diagcall:{e}")
        issues.append(
            DetectedIssue(
                issue_id=f"DIAGCALL-{fp}",
                kind="diagnostic_error",
                source="SarahMemoryDiagnostics",
                when=_now_iso(),
                summary="Diagnostics crashed",
                details=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                fingerprint=fp,
            )
        )
        return issues, raw_report

    # Convert report contents into issues (best-effort)
    try:
        blob = json.dumps(raw_report, indent=2)[:200000]
        if any(p.search(blob) for p in _ERROR_PATTERNS):
            fp = _hash_text(blob[:8000])
            issues.append(
                DetectedIssue(
                    issue_id=f"DIAG-{fp}",
                    kind="diagnostic_error",
                    source="SarahMemoryDiagnostics.run_full_diagnostics_suite",
                    when=_now_iso(),
                    summary="Diagnostics report contains errors/warnings",
                    details=blob[:12000],
                    fingerprint=fp,
                )
            )
    except Exception:
        pass

    return _dedupe_issues(issues), raw_report


# =============================================================================
# NETWORK / MESH STATUS HOOKS (OPTIONAL)
# =============================================================================

def get_network_health_blob() -> Dict[str, Any]:
    """
    Collect mesh stats / peers info (best-effort). Never fails hard.
    """
    blob: Dict[str, Any] = {"available": False}
    if SMNET is None:
        return blob

    try:
        blob["available"] = True
        if hasattr(SMNET, "get_mesh_stats"):
            blob["mesh_stats"] = SMNET.get_mesh_stats()  # type: ignore
        if hasattr(SMNET, "get_mesh_peers"):
            blob["mesh_peers"] = SMNET.get_mesh_peers()  # type: ignore
        if hasattr(SMNET, "get_local_node_status"):
            blob["local_node_status"] = SMNET.get_local_node_status()  # type: ignore
    except Exception as e:
        blob["error"] = f"{type(e).__name__}: {e}"
    return blob


def network_blob_to_issues(net_blob: Dict[str, Any]) -> List[DetectedIssue]:
    issues: List[DetectedIssue] = []
    try:
        if not net_blob.get("available"):
            return issues
        if "error" in net_blob:
            txt = json.dumps(net_blob, indent=2)[:12000]
            fp = _hash_text(txt)
            issues.append(
                DetectedIssue(
                    issue_id=f"NET-{fp}",
                    kind="network_error",
                    source="SarahMemoryNetwork",
                    when=_now_iso(),
                    summary="Network module reported an error",
                    details=txt,
                    fingerprint=fp,
                )
            )
        ms = net_blob.get("mesh_stats")
        if isinstance(ms, dict) and ms.get("error"):
            txt = json.dumps(ms, indent=2)[:8000]
            fp = _hash_text(txt)
            issues.append(
                DetectedIssue(
                    issue_id=f"MESH-{fp}",
                    kind="network_error",
                    source="SarahMemoryNetwork.get_mesh_stats",
                    when=_now_iso(),
                    summary="Mesh stats contain error",
                    details=txt,
                    fingerprint=fp,
                )
            )
    except Exception:
        pass
    return issues


# =============================================================================
# AdvCU ENRICHMENT HOOKS (OPTIONAL)
# =============================================================================

def advcu_enrich_issue(issue: DetectedIssue) -> DetectedIssue:
    """
    Enrich issue with AdvCU intent classification (best-effort).
    """
    if SMADVCU is None:
        return issue
    try:
        if hasattr(SMADVCU, "classify_intent") and callable(getattr(SMADVCU, "classify_intent")):
            label = SMADVCU.classify_intent((issue.summary or "") + "\n" + (issue.details or ""))  # type: ignore
            issue.advcu_intent = str(label)
    except Exception:
        pass
    return issue


def advcu_suggest_target(issue: DetectedIssue) -> Optional[str]:
    """
    Try to infer the most relevant file using AdvCU code corpus search (best-effort).
    Returns just a filename (e.g., "SarahMemoryVoice.py") when possible.
    """
    if SMADVCU is None:
        return None
    try:
        if hasattr(SMADVCU, "contextualize_with_code") and callable(getattr(SMADVCU, "contextualize_with_code")):
            hits = SMADVCU.contextualize_with_code((issue.summary or "") + "\n" + (issue.details or ""), top_k=6)  # type: ignore
            if isinstance(hits, list) and hits:
                for h in hits:
                    if isinstance(h, dict) and h.get("file"):
                        return Path(str(h["file"])).name
    except Exception:
        return None
    return None


# =============================================================================
# PATCH NAMING / TEMPLATE GENERATION
# =============================================================================

def _sanitize_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (s or "").strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower() or "patch"


def propose_patch_filename(target_file: str, component: str, intent: str) -> str:
    """
    Naming style:
      <FileName[prefix]>_<VersionNumber>_<Mod function type or suffix name+(_intent)>_patch.py
    Examples:
      app_v800_avatar_lipsync_patch.py
      sm_v800_voice_pyttsx_patch.py
    """
    t = Path(target_file).name.lower()
    comp = _sanitize_token(component)
    inten = _sanitize_token(intent)

    if t == "app.py":
        return f"app_{VERSION_TAG}_{comp}_{inten}_patch.py"

    base = Path(target_file).stem
    suffix = base
    if suffix.lower().startswith("sarahmemory"):
        suffix = suffix[len("SarahMemory"):] or base
    suffix = _sanitize_token(suffix)
    return f"sm_{VERSION_TAG}_{suffix}_{comp}_{inten}_patch.py"


def build_patch_header(patch_file_rel: str, patch_title: str, goal_lines: List[str]) -> str:
    goals = "\n".join([f"# - {g}" for g in goal_lines])
    return (
        "# --==The SarahMemory Project==--\n"
        f"# File: {patch_file_rel}\n"
        f"# Patch: {VERSION_STR} {patch_title}\n"
        "#\n"
        "# Goal:\n"
        f"{goals}\n"
        "\n"
    )


def build_patch_template(
    patch_path: Path,
    target_file: str,
    patch_title: str,
    goal_lines: List[str],
    apply_body_comment: str,
    error_context_comment: str,
) -> str:
    """
    Patch stub template:
    - safe to import multiple times
    - includes ORIGINAL ERROR CONTEXT as comments
    """
    rel = str(patch_path).replace("\\", "/")
    header = build_patch_header(rel, patch_title, goal_lines)

    return header + textwrap.dedent(
        f"""
        \"\"\"Owner-only monkey patch for SarahMemory.

        RULES:
        - DO NOT modify core files directly.
        - This patch should be safe to import multiple times.
        - It may add wrappers / monkey-patch functions in-memory at runtime.
        - No new dependencies.
        \"\"\"

        from __future__ import annotations

        import logging
        import traceback

        logger = logging.getLogger("{Path(patch_path).stem}")
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())

        _APPLIED = False

        # Error context reference:
        {error_context_comment}

        def apply():
            \"\"\"Apply the monkey patch (idempotent).\"\"\"
            global _APPLIED
            if _APPLIED:
                return True

            try:
                # Target: {target_file}
                # {apply_body_comment}

                # IMPLEMENTATION NOTES:
                # - Keep it minimal and safe.
                # - Avoid hard-crashing in headless environments.
                # - Prefer wrapper pattern:
                #     import target_module
                #     original = target_module.some_func
                #     def wrapped(*a, **kw): ...
                #     target_module.some_func = wrapped

                _APPLIED = True
                return True

            except Exception as e:
                logger.error("Patch apply failed: %s", e)
                logger.debug(traceback.format_exc())
                return False

        # Self-apply on import (fallback)
        try:
            apply()
        except Exception:
            pass
        """
    ).lstrip()


def _guess_component_intent(text: str, advcu_intent: Optional[str] = None) -> Tuple[str, str]:
    t = (text or "").lower()
    a = (advcu_intent or "").lower()

    if "pyttsx3" in t or "tts" in t or "voice" in t or "setvoicebyname" in t:
        return "voice", "tts"
    if "lipsync" in t or ("avatar" in t and "speech" in t):
        return "avatar", "lipsync"
    if "desktop" in t or "mirror" in t or "mss" in t:
        return "desktop", "mirror"
    if "modulenotfounderror" in t or "importerror" in t:
        return "imports", "missing"
    if "syntaxerror" in t or "indentationerror" in t:
        return "syntax", "repair"
    if "/api/" in t or "endpoint" in t or "flask" in t:
        return "api", "contract"
    if "sqlite" in t or "database" in t or "db" in t:
        return "database", "stability"

    if a:
        if "voice" in a:
            return "voice", "stability"
        if "network" in a or "mesh" in a:
            return "network", "stability"
        if "webui" in a:
            return "webui", "bridge"
        if "api" in a:
            return "api", "stability"

    return "fix", "stability"


def create_patch_from_issue(issue: DetectedIssue, target_file_guess: str) -> Optional[PatchPlan]:
    """
    Creates a non-conflicting patch stub and registers the issue fingerprint so it won't duplicate.
    Returns None if already processed.
    """
    _ensure_dirs()

    # Primary dedupe: processed index
    if _already_processed(issue):
        return None

    component, intent = _guess_component_intent(issue.summary + "\n" + issue.details, issue.advcu_intent)
    base_name = propose_patch_filename(target_file_guess, component, intent)
    patch_name = _make_nonconflicting_patch_name(base_name)
    patch_path = MODS_DIR / patch_name

    patch_title = f"{component.title()} {intent.replace('_',' ').title()}"
    goal_lines = [
        f"Generated from issue {issue.issue_id}",
        "Owner must review before enabling in loader.",
        "No core file modifications; monkey-patch only.",
        "Must remain headless-safe (Linux/PythonAnywhere) and Windows-safe.",
        "Patch stub includes original error context for reference.",
    ]

    apply_body_comment = f"Auto-generated stub from {issue.kind} ({issue.source}). Insert minimal patch logic."
    error_context_comment = _embed_error_context_as_comment(issue.details or "")

    content = build_patch_template(
        patch_path=patch_path,
        target_file=target_file_guess,
        patch_title=patch_title,
        goal_lines=goal_lines,
        apply_body_comment=apply_body_comment,
        error_context_comment=error_context_comment,
    )
    patch_path.write_text(content, encoding="utf-8")

    issue.suggested_patch_name = patch_name
    issue.suggested_patch_goal = patch_title
    issue.suggested_target_file = target_file_guess

    # Register processed
    _mark_processed(issue, patch_name=patch_name)

    return PatchPlan(
        patch_path=str(patch_path),
        target_file=target_file_guess,
        patch_name=patch_name,
        patch_goal=patch_title,
        issue_ids=[issue.issue_id],
        created_at=_now_iso(),
        sandbox_tested=False,
        sandbox_passed=False,
        safe_to_apply=False,
    )


# =============================================================================
# SANDBOX TESTING OF PATCH STUBS (NO EXECUTION SIDE EFFECTS)
# =============================================================================

def sandbox_test_patch_file(patch_file: Path) -> Tuple[bool, str]:
    """
    Sandbox-test a patch file by compiling it and (optionally) executing in Synapes sandbox.
    - Does NOT import/enable in your live runtime
    - NO core edits
    """
    code = _read_text_safely(patch_file, max_chars=300_000)
    if not code:
        return False, "empty_or_unreadable"

    try:
        compile(code, str(patch_file), "exec")
    except Exception as e:
        return False, f"compile_failed: {type(e).__name__}: {e}"

    if SMSYN is not None and hasattr(SMSYN, "run_sandbox_test") and callable(getattr(SMSYN, "run_sandbox_test")):
        try:
            ok = SMSYN.run_sandbox_test(code)  # type: ignore
            return bool(ok), "synapes_run_sandbox_test"
        except Exception as e:
            return False, f"synapes_failed: {type(e).__name__}: {e}"

    return True, "compile_only_ok"


# =============================================================================
# FILESYSTEM BACKUP HOOK (OPTIONAL)
# =============================================================================

def try_create_backup_snapshot() -> Tuple[bool, str]:
    """
    Best-effort call into SarahMemoryFilesystem backup routine if present.
    Does NOT modify core files; only creates backups/logs.
    """
    if SMFS is None:
        return False, "SarahMemoryFilesystem_not_available"
    try:
        if hasattr(SMFS, "create_weekly_backup") and callable(getattr(SMFS, "create_weekly_backup")):
            SMFS.create_weekly_backup()  # type: ignore
            return True, "create_weekly_backup_called"
        if hasattr(SMFS, "initialize") and callable(getattr(SMFS, "initialize")):
            SMFS.initialize()  # type: ignore
            return True, "initialize_called"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return False, "no_known_backup_entrypoint"


# =============================================================================
# SELF-REPAIR SUBMISSION (API + RESEARCH + DL + COMPARE) — SAFE FALLBACKS
# =============================================================================

def _build_repair_payload(
    issues: List[DetectedIssue],
    diag_report: Dict[str, Any],
    net_blob: Dict[str, Any],
) -> Dict[str, Any]:
    overlay = _discover_mods_overlay_dir()
    return {
        "ts": _now_iso(),
        "tool": "SarahMemoryEvolution",
        "version": VERSION_STR,
        "version_tag": VERSION_TAG,
        "neosky_enabled": _neosky_enabled(),
        "rules": {
            "no_core_edits": True,
            "patches_only": True,
            "user_approval_required": True,
            "no_new_deps_without_permission": True,
        },
        "environment": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "datasets_dir": str(DATASETS_DIR),
            "run_mode": getattr(config, "RUN_MODE", None) if config else None,
            "device_mode": getattr(config, "DEVICE_MODE", None) if config else None,
            "platform": sys.platform,
            "python": sys.version,
        },
        "artifacts": {
            "system_db_found": str(_find_system_db()) if _find_system_db() else None,
            "known_logs_checked": [str(p) for p in KNOWN_LOG_FILES],
            "mods_dir_base": str(MODS_DIR),
            "mods_dir_overlay": str(overlay) if overlay else None,
            "reports_dir": str(REPORTS_DIR),
            "outbox_dir": str(REPAIR_OUTBOX_DIR),
        },
        "diagnostics_report": diag_report,
        "network_report": net_blob,
        "issues": [asdict(i) for i in issues],
    }


def _build_patch_suggestion_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are SarahMemoryEvolution (owner-only). "
        "Given the following issues payload, propose ONLY monkey-patch solutions.\n"
        "RULES:\n"
        "- DO NOT modify core files\n"
        "- Output patch file names targeting ../data/mods/v800/\n"
        "- Provide safe apply() approach (minimal, headless-safe)\n"
        "- No new dependencies unless owner explicitly approves\n\n"
        f"ISSUES_JSON:\n{json.dumps(payload, indent=2)[:15000]}"
    )


def _try_api_patch_suggestions(prompt: str) -> Optional[str]:
    """
    Ask SarahMemoryAPI for monkey-patch suggestions (best-effort).
    NEVER applies anything; only returns text to store in outbox.
    """
    if SMAPI is None or not hasattr(SMAPI, "send_to_api"):
        return None
    try:
        resp = SMAPI.send_to_api(prompt, intent="code")  # type: ignore
        if isinstance(resp, dict):
            txt = resp.get("data") or resp.get("text") or resp.get("response")
            return str(txt).strip() if txt else json.dumps(resp, indent=2)[:15000]
        if isinstance(resp, str):
            return resp.strip()
    except Exception:
        return None
    return None


def _try_research_suggestions(payload: Dict[str, Any]) -> Optional[str]:
    """
    Ask SarahMemoryResearch to provide fix ideas. Best-effort.
    """
    if SMR is None or not hasattr(SMR, "get_research_data"):
        return None
    try:
        # Make query a bit more targeted using top issue summaries.
        issues = payload.get("issues") or []
        top = []
        for it in issues[:8]:
            try:
                top.append(str(it.get("summary", ""))[:160])
            except Exception:
                continue
        query = (
            "Fix strategy for Python/Flask errors using monkey patches only (no core edits). "
            "Focus on stability + headless-safe behavior. "
            f"Issue hints: {', '.join(top)[:800]}"
        )
        r = SMR.get_research_data(query)  # type: ignore
        if isinstance(r, dict):
            return json.dumps(r, indent=2)[:15000]
        if isinstance(r, str):
            return r[:15000]
    except Exception:
        return None
    return None


def _try_dl_suggestions(payload: Dict[str, Any]) -> Optional[str]:
    """
    Ask SarahMemoryDL (best-effort).
    NOTE: SarahMemoryDL may not be designed for patch planning; we store any returned analysis.
    """
    if SMDL is None:
        return None
    try:
        # Preferred: if module has a function meant to return an analysis dict/string
        if hasattr(SMDL, "evaluate_conversation_patterns") and callable(getattr(SMDL, "evaluate_conversation_patterns")):
            r = SMDL.evaluate_conversation_patterns()  # type: ignore
            if isinstance(r, dict):
                return json.dumps(r, indent=2)[:15000]
            if isinstance(r, str):
                return r[:15000]
        # Fallback: try any obvious "analyze" entrypoints without crashing
        for fn in ("deep_learn_user_context", "analyze_user_behavior", "update_deep_learning_models"):
            if hasattr(SMDL, fn) and callable(getattr(SMDL, fn)):
                rr = getattr(SMDL, fn)()  # type: ignore
                if isinstance(rr, dict):
                    return json.dumps(rr, indent=2)[:15000]
                if isinstance(rr, str):
                    return rr[:15000]
    except Exception:
        return None
    return None


def _try_compare_validation(prompt_text: str, candidate_text: str) -> Optional[Dict[str, Any]]:
    """
    Validate candidate suggestion using SarahMemoryCompare (best-effort).
    """
    if SMCMP is None or not hasattr(SMCMP, "compare_reply"):
        return None
    try:
        r = SMCMP.compare_reply(prompt_text, candidate_text)  # type: ignore
        if isinstance(r, dict):
            return r
    except Exception:
        return None
    return None


def submit_for_repair(
    payload: Dict[str, Any],
    do_api: bool,
    do_research: bool,
    do_dl: bool,
    do_compare: bool,
) -> Path:
    """
    Writes outbox JSON with payload + optional suggestions (API/Research/DL) and optional Compare validation.
    """
    _ensure_dirs()

    suggestions: Dict[str, Any] = {}

    prompt_text = _build_patch_suggestion_prompt(payload)

    api_s: Optional[str] = None
    if do_api:
        api_s = _try_api_patch_suggestions(prompt_text)
        if api_s:
            suggestions["api_patch_suggestions"] = api_s

    if do_research:
        rs = _try_research_suggestions(payload)
        if rs:
            suggestions["research_suggestions"] = rs

    if do_dl:
        dl = _try_dl_suggestions(payload)
        if dl:
            suggestions["dl_suggestions"] = dl

    # Optional Compare validation against the candidate suggestion (best-effort)
    if do_compare and api_s:
        cmp = _try_compare_validation(prompt_text, api_s)
        if cmp:
            suggestions["compare_validation_on_api_suggestion"] = cmp

    if suggestions:
        payload["suggestions"] = suggestions

    outbox_path = REPAIR_OUTBOX_DIR / f"repair_payload_{int(time.time())}.json"
    outbox_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return outbox_path


# =============================================================================
# MAIN EVOLUTION FLOW
# =============================================================================

def evolve_once(autonomous: bool = False, weekly_gate: bool = False) -> None:
    """
    One evolution cycle:
      - (Optional) Backup snapshot via Filesystem module
      - Diagnostics (full suite if available)
      - DB scan (system_logs.db preferred)
      - Log scan (includes diag_report.log + research.log)
      - Network health snapshot (mesh stats)
      - AdvCU enrich (intent + code-target suggestion)
      - Report
      - (Optional) submit for repair (outbox + suggestions)
      - (Optional) generate patch stubs (non-conflicting; embeds error context; deduped)
      - (Optional) mark/erase original log error blocks after patch generation (owner-approved; DISABLED in autonomous)
      - (Optional) sandbox test patch stubs
    """
    _ensure_dirs()

    if autonomous and weekly_gate:
        if not _should_run_weekly():
            last = _load_neosky_last_run()
            print(f"[NEOSKYMATRIX] Weekly gate: skipped (last_run={last.isoformat(timespec='seconds') if last else None})")
            return

    def decide(prompt: str, default_no: bool = True, force: Optional[bool] = None) -> bool:
        if autonomous:
            if force is not None:
                return force
            return not default_no
        return _ask_yn(prompt, default_no=default_no)

    overlay_dir = _discover_mods_overlay_dir()
    print("Running evolution cycle...\n")
    print(f"Mods staging dir (base): {MODS_DIR}")
    if overlay_dir:
        print(f"Mods overlay dir detected: {overlay_dir} (same filenames may override base in loader)")
    print(f"NEOSKYMATRIX enabled: {_neosky_enabled()} | autonomous={autonomous} | weekly_gate={weekly_gate}\n")

    # In autonomous mode, we typically want backup snapshot ON (best-effort)
    if decide("Create a backup snapshot (best-effort) BEFORE scanning/generating patches?", default_no=True, force=True if autonomous else None):
        ok, why = try_create_backup_snapshot()
        print(f"Backup snapshot: ok={ok} ({why})\n")
    else:
        print("Backup snapshot: skipped.\n")

    issues: List[DetectedIssue] = []

    # 1) Diagnostics
    diag_issues, diag_report = run_sarahmemory_diagnostics()
    if diag_issues:
        print(f"Diagnostics: {len(diag_issues)} issue(s) found.")
        issues.extend(diag_issues)
    else:
        print("Diagnostics: no issues (or module not available).")

    # 2) DB scan
    db_issues = scan_system_db_for_issues()
    if db_issues:
        print(f"System DB: {len(db_issues)} issue(s) found.")
        issues.extend(db_issues)
    else:
        print("System DB: no issues (or system DB not found).")

    # 3) Log scan
    log_issues = scan_logs_for_issues()
    if log_issues:
        print(f"Logs: {len(log_issues)} issue(s) found.")
        issues.extend(log_issues)
    else:
        print("Logs: no issues found in scanned logs.")

    # 4) Network snapshot
    net_blob = get_network_health_blob()
    net_issues = network_blob_to_issues(net_blob)
    if net_issues:
        print(f"Network: {len(net_issues)} issue(s) found.")
        issues.extend(net_issues)
    else:
        print("Network: ok (or not enabled).")

    # 5) Dedupe + AdvCU enrich
    issues = _dedupe_issues(issues)

    if issues and SMADVCU is not None:
        if decide("Use AdvCU to enrich issues (intent classify + code target suggestions)?", default_no=False, force=True if autonomous else None):
            for it in issues:
                advcu_enrich_issue(it)
        else:
            print("AdvCU enrichment skipped.\n")

    # 6) Save report
    report = {
        "ts": _now_iso(),
        "issues_count": len(issues),
        "issues": [asdict(i) for i in issues],
        "diagnostics_report": diag_report,
        "network_report": net_blob,
        "system_db": str(_find_system_db()) if _find_system_db() else None,
        "known_logs": [str(p) for p in KNOWN_LOG_FILES],
        "mods_dir_base": str(MODS_DIR),
        "mods_dir_overlay": str(overlay_dir) if overlay_dir else None,
        "processed_index": str(PROCESSED_INDEX_PATH),
        "neosky_enabled": _neosky_enabled(),
        "autonomous": autonomous,
        "weekly_gate": weekly_gate,
    }
    report_path = _save_json_report(f"evolution_report_{int(time.time())}", report)
    print(f"\nSaved report: {report_path}")

    if not issues:
        print("\nNo issues detected. Evolution cycle complete.\n")
        if autonomous and weekly_gate:
            _save_neosky_last_run(datetime.now())
        _optional_prune_artifacts(autonomous=autonomous)
        return

    # 7) Summary
    print("\nDetected issues (summary):")
    for i, it in enumerate(issues[:30], 1):
        src = it.source
        src_tail = Path(src.split("::")[0]).name if src else "unknown"
        hint = f" | advcu={it.advcu_intent}" if it.advcu_intent else ""
        processed = " | processed" if _already_processed(it) else ""
        print(f" {i:>2}. {it.issue_id} | {it.kind} | {src_tail} | {it.summary[:85]}{hint}{processed}")
    if len(issues) > 30:
        print(f" ... plus {len(issues) - 30} more (see report JSON).")

    # 8) Submit for repair (outbox + optional suggestions)
    if decide("\nSubmit issues for self-repair outbox (optionally ask API/Research/DL + Compare validation)?", default_no=True, force=True if autonomous else None):
        do_api = decide("Include API suggestions (SarahMemoryAPI.send_to_api)?", default_no=False, force=True if autonomous else None)
        do_research = decide("Include Research suggestions (SarahMemoryResearch.get_research_data)?", default_no=False, force=True if autonomous else None)
        do_dl = decide("Include DeepLearning suggestions (SarahMemoryDL best-effort)?", default_no=False, force=True if autonomous else None)
        do_compare = decide("Run Compare validation (SarahMemoryCompare) on API suggestion (best-effort)?", default_no=True, force=True if autonomous else None)

        payload = _build_repair_payload(issues, diag_report, net_blob)
        outbox = submit_for_repair(payload, do_api=do_api, do_research=do_research, do_dl=do_dl, do_compare=do_compare)
        print(f"Repair payload written: {outbox}\n")
    else:
        print("Repair submission skipped.\n")

    # 9) Patch generation (STUBS ONLY) + optional log marking
    plans: List[PatchPlan] = []
    if decide("Generate monkey patch STUBS for these issues in ../data/mods/v800/ ?", default_no=True, force=True if autonomous else None):
        for it in issues[:20]:
            # Skip if already processed (primary dedupe)
            if _already_processed(it):
                print(f"  ~ skipped (already processed): {it.issue_id}")
                continue

            target_guess = "app.py"
            m = re.search(r'File "([^"]+\.py)"', it.details or "")
            if m:
                target_guess = Path(m.group(1)).name

            low = (it.details or "").lower()
            if "sarahmemoryvoice" in low or "setvoicebyname" in low:
                target_guess = "SarahMemoryVoice.py"
            elif "sarahmemoryresearch" in low:
                target_guess = "SarahMemoryResearch.py"
            elif "sarahmemoryapi" in low or "send_to_api" in low:
                target_guess = "SarahMemoryAPI.py"
            elif "avatar" in low or "lipsync" in low:
                target_guess = "UnifiedAvatarController.py"

            if SMADVCU is not None:
                try:
                    suggested = advcu_suggest_target(it)
                    if suggested:
                        target_guess = suggested
                except Exception:
                    pass

            plan = create_patch_from_issue(it, target_guess)
            if plan is None:
                print(f"  ~ skipped (already processed): {it.issue_id}")
                continue

            plans.append(plan)
            print(f"  + wrote patch stub: {plan.patch_name}")

            # OPTIONAL: mark/erase original log error block to prevent duplicate detection
            # SAFETY: disabled in autonomous mode (owner-only manual action).
            if (not autonomous) and it.kind == "log_error":
                try:
                    log_path = Path(it.source)
                    if log_path.exists() and log_path.is_file():
                        if decide(
                            f"Mark this error block as handled in log '{log_path.name}'? "
                            f"(creates .bak archive first)",
                            default_no=True,
                        ):
                            bak = _archive_log_before_edit(log_path)
                            marker = (
                                f"[SARAHMEMORY_EVOLUTION] Issue {it.issue_id} moved to patch "
                                f"{plan.patch_name} at {_now_iso()}\n"
                            )
                            ok = _mark_error_block_in_log(log_path, it.details, marker)
                            print(f"    log_marked={ok} backup={bak}")
                except Exception:
                    pass

        plans_path = _save_json_report(f"patch_plans_{int(time.time())}", [asdict(p) for p in plans])
        print(f"\nPatch plans saved: {plans_path}")
        print("\nIMPORTANT: These are STUBS. Review each patch before enabling in your mod loader.\n")
    else:
        print("Patch generation skipped.\n")

    # 10) Sandbox test patch stubs
    if plans and decide("Sandbox-test the generated patch stubs (compile + Synapes sandbox if available)?", default_no=False, force=True if autonomous else None):
        for p in plans:
            pp = Path(p.patch_path)
            ok, how = sandbox_test_patch_file(pp)
            p.sandbox_tested = True
            p.sandbox_passed = bool(ok)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {pp.name} ({how})")
        tested_path = _save_json_report(f"patch_plans_tested_{int(time.time())}", [asdict(p) for p in plans])
        print(f"\nTested patch plan report saved: {tested_path}\n")
    else:
        if plans:
            print("Sandbox test skipped.\n")

    if autonomous and weekly_gate:
        _save_neosky_last_run(datetime.now())

    _optional_prune_artifacts(autonomous=autonomous)
    print("Evolution cycle complete.\n")


# =============================================================================
# OWNER MENU
# =============================================================================

def main() -> None:
    _print_banner()
    _ensure_dirs()

    print("Owner-only tool: This file is intended to be run ONLY by Brian Lee Baros.")
    print("It will NOT modify any core files. It creates patch stubs only.\n")

    while True:
        print("Menu:")
        print("  1) Run Evolution Cycle (interactive)")
        print("  2) Show Paths (BASE/DATA/DATASETS/MODS)")
        print("  3) Add SM_AGI key to .env (append-only, user supplies value)")
        print("  4) NEOSKYMATRIX: Run Autonomous Weekly Cycle (no prompts; gated to 7 days)")
        print("  0) Exit\n")

        choice = input("Select: ").strip()

        if choice == "1":
            try:
                evolve_once(autonomous=False, weekly_gate=False)
            except KeyboardInterrupt:
                print("\nCancelled by user.\n")
            except Exception as e:
                print(f"\nEvolution cycle crashed (tool survived): {e}")
                print(traceback.format_exc())

        elif choice == "2":
            overlay = _discover_mods_overlay_dir()
            print("\nPaths:")
            print(f"  BASE_DIR         = {BASE_DIR}")
            print(f"  DATA_DIR         = {DATA_DIR}")
            print(f"  DATASETS_DIR     = {DATASETS_DIR}")
            print(f"  MODS_DIR_BASE    = {MODS_DIR}")
            print(f"  MODS_DIR_OVERLAY = {overlay}")
            print(f"  REPORTS_DIR      = {REPORTS_DIR}")
            print(f"  OUTBOX_DIR       = {REPAIR_OUTBOX_DIR}")
            print(f"  LOG_ARCHIVE      = {LOG_ARCHIVE_DIR}")
            print(f"  SYSTEM_DB        = {_find_system_db()}")
            print(f"  KNOWN_LOGS       = {[str(p) for p in KNOWN_LOG_FILES]}")
            print(f"  PROCESSED_IDX    = {PROCESSED_INDEX_PATH}")
            print(f"  NEOSKY_ENABLED   = {_neosky_enabled()}")
            print(f"  NEOSKY_LAST_RUN  = {_load_neosky_last_run()}")
            print("")

        elif choice == "3":
            env_path = _find_env_path()
            if not env_path:
                print("No .env file found in expected locations.")
                continue
            value = input("Enter value for new SM_AGI_(n)= key (it will be appended): ").strip()
            if not value:
                print("No value provided.")
                continue
            if not _ask_yn(f"Append a new SM_AGI key to {env_path} ?"):
                print("Cancelled.")
                continue
            key = _append_sm_agi_key(env_path, value)
            if key:
                print(f"Added: {key}=... to {env_path}")
            else:
                print("Failed to modify .env (no changes made).")

        elif choice == "4":
            # Autonomous weekly cycle: respects weekly gating file
            try:
                evolve_once(autonomous=True, weekly_gate=True)
            except KeyboardInterrupt:
                print("\nCancelled by user.\n")
            except Exception as e:
                print(f"\nAutonomous evolution crashed (tool survived): {e}")
                print(traceback.format_exc())

        elif choice == "0":
            print("Exiting SarahMemoryEvolution.")
            return

        else:
            print("Invalid option.\n")


if __name__ == "__main__":
    main()
