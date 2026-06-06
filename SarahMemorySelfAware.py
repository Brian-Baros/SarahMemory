"""
--== SarahMemory Project ==--
File: SarahMemorySelfAware.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
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

ENTERPRISE INTENT
- Autonomous meta-cognition loop gated by NEOSKYMATRIX + DEVELOPERSMODE.
- Self-introspection: understand core files, defs, imports, boot orchestration.
- Self-diagnostics: mine logs + system DB, cluster issues, propose remediation.
- Governance-first: can stage recommendations and patch plans; does NOT silently
  rewrite core files in place.

SECURITY POSTURE
- Hard gate: NEOSKYMATRIX=True AND DEVELOPERSMODE=True.
- Audit logging into system_logs.db.
- No destructive filesystem operations by default.

===============================================================================
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "self_awareness_engine"
# CATEGORY = "autonomous_meta_cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "self_aware"
# FAMILY = "self_evolution"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Autonomous meta-cognition loop gated by NEOSKYMATRIX and DEVELOPERSMODE. Performs self-introspection, diagnostics mining, and governed remediation planning without silently rewriting core files." 
# --- SARAHMETA END ---
import os
import re
import sys
import json
import time
import ast
import sqlite3
import hashlib
import logging
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Core config import (single source of truth)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore


# ---------------------------------------------------------------------------
# Optional module imports (best-effort, never block boot)
# ---------------------------------------------------------------------------
SYN = None
UPD = None
COG = None

try:
    import SarahMemorySynapes as SYN  # type: ignore
except Exception:
    SYN = None

try:
    import SarahMemoryUpdater as UPD  # type: ignore
except Exception:
    UPD = None

try:
    import SarahMemoryCognitiveServices as COG  # type: ignore
except Exception:
    COG = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemorySelfAware")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Gating (NEOSKYMATRIX + DEVELOPERSMODE)
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "false") -> bool:
    try:
        v = os.getenv(name, default)
        return str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")
    except Exception:
        return False


def _get_flag(flag_name: str, default: bool = False) -> bool:
    """
    Reads config.<flag_name> first, then env var, then default.
    """
    try:
        if config is not None and hasattr(config, flag_name):
            v = getattr(config, flag_name)
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")
    except Exception:
        pass
    return _env_flag(flag_name, "true" if default else "false")


def _armed() -> bool:
    # Owner-intent: NEOSKYMATRIX is your “red pill” arming switch.
    neosky = _get_flag("NEOSKYMATRIX", default=False)
    dev = _get_flag("DEVELOPERSMODE", default=False)
    return bool(neosky and dev)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _base_dir() -> Path:
    try:
        if config is not None and hasattr(config, "BASE_DIR"):
            return Path(getattr(config, "BASE_DIR"))
    except Exception:
        pass
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    try:
        if config is not None and hasattr(config, "DATA_DIR"):
            return Path(getattr(config, "DATA_DIR"))
    except Exception:
        pass
    return _base_dir() / "data"


def _datasets_dir() -> Path:
    try:
        if config is not None and hasattr(config, "DATASETS_DIR"):
            return Path(getattr(config, "DATASETS_DIR"))
    except Exception:
        pass
    return _data_dir() / "memory" / "datasets"


def _system_logs_db() -> Path:
    return _datasets_dir() / "system_logs.db"


def _ensure_dirs() -> None:
    (_data_dir()).mkdir(parents=True, exist_ok=True)
    (_datasets_dir()).mkdir(parents=True, exist_ok=True)
    (_data_dir() / "reports" / "v800").mkdir(parents=True, exist_ok=True)
    (_data_dir() / "mods" / "v800").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# System log DB (append-only governance)
# ---------------------------------------------------------------------------
def _db_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    return con


def _db_ensure() -> None:
    con = None
    try:
        con = _db_connect(_system_logs_db())
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS selfaware_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                severity TEXT,
                cycle_id TEXT,
                event TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def log_event(event: str, details: str, *, severity: str = "INFO", cycle_id: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        _db_ensure()
        con = _db_connect(_system_logs_db())
        cur = con.cursor()
        ts = datetime.now().isoformat()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        cur.execute(
            "INSERT INTO selfaware_events (ts, severity, cycle_id, event, details, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, str(severity), str(cycle_id), str(event), str(details), meta_json),
        )
        con.commit()
        con.close()
    except Exception:
        # never block loop
        pass


# ---------------------------------------------------------------------------
# Codebase introspection
# ---------------------------------------------------------------------------
@dataclass
class FunctionSig:
    name: str
    lineno: int
    args: List[str]
    doc: str = ""


@dataclass
class ModuleScan:
    path: str
    sha256: str
    imports: List[str]
    functions: List[FunctionSig]
    classes: List[str]


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_python_module(p: Path) -> Optional[ModuleScan]:
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        imports: List[str] = []
        functions: List[FunctionSig] = []
        classes: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            imports.append(a.name)
                    else:
                        mod = node.module or ""
                        imports.append(mod)
                except Exception:
                    pass
            elif isinstance(node, ast.FunctionDef):
                args = []
                try:
                    for a in node.args.args:
                        args.append(a.arg)
                except Exception:
                    pass
                doc = ast.get_docstring(node) or ""
                functions.append(FunctionSig(name=node.name, lineno=int(getattr(node, "lineno", 0) or 0), args=args, doc=doc[:240]))
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return ModuleScan(
            path=str(p),
            sha256=_sha256_file(p),
            imports=sorted(list(set([i for i in imports if i]))),
            functions=sorted(functions, key=lambda x: x.lineno),
            classes=sorted(list(set(classes))),
        )
    except Exception:
        return None


def scan_codebase(root: Path, *, max_files: int = 250) -> List[ModuleScan]:
    scans: List[ModuleScan] = []
    try:
        max_files = max(1, min(int(max_files or _SELF_AWARE_MAX_SCAN_FILES), _SELF_AWARE_MAX_SCAN_FILES))
        py_files = _selfaware_walk_files(root, {".py"}, max_files=max_files)
        # prioritize core files first (SarahMemory*.py + api/server/app*.py)
        py_files.sort(key=lambda x: (0 if x.name.startswith("SarahMemory") else 1, 0 if "api" in str(x).lower() else 1, str(x)))
        for p in py_files[:max_files]:
            ms = scan_python_module(p)
            if ms:
                scans.append(ms)
    except Exception:
        pass
    return scans


# ---------------------------------------------------------------------------
# Log mining (lightweight)
# ---------------------------------------------------------------------------
ERROR_PAT = re.compile(r"(Traceback \(most recent call last\):|ERROR\s+-|CRITICAL\s+-|Exception:)", re.IGNORECASE)


# Runtime optimization: SelfAware may be ARMED during development, but it must not
# scan dependency folders, caches, models, backups, or generated UI packages. These
# exclusions protect the active C: NVMe drive and keep the autonomous loop bounded.
_SELF_AWARE_SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build", ".next", ".vite", "coverage",
    "logs", "log", "cache", "tmp", "temp", "runtime", "backup", "backups", "archive", "archives",
    "models", "checkpoints", "exports", "downloads", "sandbox",
}
_SELF_AWARE_SKIP_PARTS = {part.lower() for part in _SELF_AWARE_SKIP_DIR_NAMES}
_SELF_AWARE_MAX_SCAN_FILES = int(os.getenv("SARAH_SELFAWARE_MAX_SCAN_FILES", "160") or 160)
_SELF_AWARE_DEFAULT_LOOP_SECONDS = int(os.getenv("SARAH_SELFAWARE_LOOP_SECONDS", "300") or 300)
_SELF_AWARE_AUTOINDEX_ENABLED = str(os.getenv("SARAH_SELFAWARE_AUTOINDEX_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes", "on")


def _selfaware_path_skipped(path: Path) -> bool:
    try:
        parts = {str(part).lower() for part in Path(path).parts}
        return bool(parts & _SELF_AWARE_SKIP_PARTS)
    except Exception:
        return False


def _selfaware_walk_files(root: Path, suffixes: set[str], max_files: int) -> List[Path]:
    out: List[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if d.lower() not in _SELF_AWARE_SKIP_PARTS]
            if _selfaware_path_skipped(Path(dirpath)):
                continue
            for fn in filenames:
                if len(out) >= max_files:
                    return out
                if Path(fn).suffix.lower() in suffixes:
                    p = Path(dirpath) / fn
                    if not _selfaware_path_skipped(p):
                        out.append(p)
    except Exception:
        pass
    return out

def tail_text_file(p: Path, *, max_bytes: int = 256_000) -> str:
    try:
        if not p.exists():
            return ""
        b = p.read_bytes()
        if len(b) > max_bytes:
            b = b[-max_bytes:]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def find_recent_errors(log_dir: Path, *, max_files: int = 30) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    try:
        if not log_dir.exists():
            return hits
        files = _selfaware_walk_files(log_dir, {".log"}, max_files=max(1, int(max_files or 30)))
        files = sorted(files, key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
        for p in files[:max_files]:
            txt = tail_text_file(p)
            if not txt:
                continue
            if ERROR_PAT.search(txt):
                # capture last ~50 lines
                lines = txt.splitlines()[-80:]
                blob = "\n".join(lines)
                hits.append({"path": str(p), "mtime": p.stat().st_mtime, "tail": blob[:12000]})
    except Exception:
        pass
    return hits


# ---------------------------------------------------------------------------
# Cognitive prioritization (best-effort)
# ---------------------------------------------------------------------------
def cognitive_rank(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If CognitiveServices exposes any triage method, use it. Otherwise basic ranking.
    """
    if not issues:
        return issues

    # default: newest first
    issues.sort(key=lambda x: float(x.get("mtime", 0) or 0), reverse=True)

    try:
        # optional hook patterns (won’t assume exact API)
        if COG is not None:
            # common patterns we can call if present
            for fn_name in ("triage_issues", "rank_issues", "classify_issue_batch"):
                if hasattr(COG, fn_name):
                    fn = getattr(COG, fn_name)
                    ranked = fn(issues)  # type: ignore
                    if isinstance(ranked, list) and ranked:
                        return ranked
    except Exception:
        pass

    return issues


# ---------------------------------------------------------------------------
# Cycle artifacts (reports)
# ---------------------------------------------------------------------------
def _reports_dir() -> Path:
    return _data_dir() / "reports" / "v800" / "selfaware"


def write_cycle_report(cycle_id: str, payload: Dict[str, Any]) -> Path:
    d = _reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"cycle_{cycle_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _cycle_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_synapes_bootstrap() -> None:
    """
    Materialize Synapes directories/registries when available.
    """
    try:
        if SYN is not None and hasattr(SYN, "ensure_sarahmemory_model_dirs"):
            SYN.ensure_sarahmemory_model_dirs()  # type: ignore
    except Exception:
        pass


def loop_sleep_seconds() -> int:
    try:
        v = getattr(config, "REFLECTION_INTERVAL", None) if config is not None else None
        if isinstance(v, int) and v > 0:
            return int(v)
    except Exception:
        pass
    # optimized runtime default: SelfAware is a deep reflection organ, not a tight poller.
    return max(60, int(_SELF_AWARE_DEFAULT_LOOP_SECONDS))



# ---------------------------------------------------------------------------
# System Index -> Learning Bridge (no edits to SystemIndexer.py)
# ---------------------------------------------------------------------------
class SystemIndexAutoLearner:
    """
    Background bridge:
      - Uses SarahMemorySystemIndexer helpers to keep system_index.db fresh (bounded scope).
      - Detects new rows in system_index.db and triggers targeted learning via SarahMemorySystemLearn.
      - Runs ONLY when SelfAware is ARMED (NEOSKYMATRIX + DEVELOPERSMODE) to match governance intent.

    Notes:
      - We do NOT modify SarahMemorySystemIndexer.py or SarahMemorySystemLearn.py.
      - This class is designed to be called from SelfAware's autonomous cycle (tick()).
    """

    def __init__(self, *, scope_paths: Optional[List[Path]] = None, max_new_per_tick: int = 25):
        self.index_db = _datasets_dir() / "system_index.db"
        self.state_path = _datasets_dir() / "system_index_autolearn_state.json"
        self.max_new_per_tick = int(max_new_per_tick) if int(max_new_per_tick) > 0 else 25
        self.scope_paths = scope_paths or [ _base_dir(), _data_dir() ]
        self.last_file_id = 0
        self.last_reg_id = 0
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self.state_path.exists():
                obj = json.loads(self.state_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    self.last_file_id = int(obj.get("last_file_id") or 0)
                    self.last_reg_id = int(obj.get("last_reg_id") or 0)
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "last_file_id": int(self.last_file_id),
                "last_reg_id": int(self.last_reg_id),
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            self.state_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _connect_index_db(self) -> Optional[sqlite3.Connection]:
        try:
            self.index_db.parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(str(self.index_db))
        except Exception:
            return None

    def _ensure_index_db(self) -> None:
        try:
            import SarahMemorySystemIndexer as IDX  # type: ignore
            # init_db uses its own DB_PATH derived from Globals; that should match this.index_db
            IDX.init_db()
        except Exception:
            pass

    def _bounded_index_scan(self) -> Dict[str, Any]:
        """Index a bounded set of paths (BASE_DIR + DATA_DIR) to avoid full-drive scans."""
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        try:
            import SarahMemorySystemIndexer as IDX  # type: ignore
        except Exception:
            return stats

        self._ensure_index_db()

        try:
            exts = getattr(IDX, "FILE_EXTENSIONS", set())
            if not exts:
                exts = {".txt", ".md", ".json", ".py"}
        except Exception:
            exts = {".txt", ".md", ".json", ".py"}

        for root in self.scope_paths:
            try:
                if not root.exists():
                    continue
                for dirpath, dirnames, filenames in os.walk(str(root)):
                    dirnames[:] = [d for d in dirnames if d.lower() not in _SELF_AWARE_SKIP_PARTS]
                    if _selfaware_path_skipped(Path(dirpath)):
                        continue
                    for fn in filenames:
                        try:
                            p = os.path.join(dirpath, fn)
                            ext = os.path.splitext(fn)[1].lower()
                            if ext not in exts:
                                continue
                            st = os.stat(p)
                            file_size = int(st.st_size)
                            modified = datetime.fromtimestamp(st.st_mtime).isoformat()
                            h = IDX.compute_sha256(p)
                            if not h:
                                stats["errors"] += 1
                                continue
                            if IDX.file_already_indexed(p, h):
                                stats["skipped"] += 1
                                continue
                            IDX.insert_file_record(p, ext, file_size, modified, h)
                            stats["indexed"] += 1
                        except Exception:
                            stats["errors"] += 1
            except Exception:
                stats["errors"] += 1

        return stats

    def _fetch_new_files(self) -> List[Tuple[int, str, str]]:
        con = self._connect_index_db()
        if not con:
            return []
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT id, file_path, file_type FROM file_index WHERE id > ? ORDER BY id ASC LIMIT ?",
                (int(self.last_file_id), int(self.max_new_per_tick)),
            )
            rows = cur.fetchall() or []
            return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]
        except Exception:
            return []
        finally:
            try:
                con.close()
            except Exception:
                pass

    def _learn_file(self, file_path: str) -> bool:
        try:
            import SarahMemorySystemLearn as SML  # type: ignore
        except Exception:
            return False

        try:
            text = SML.extract_text(file_path)
            if not text:
                return False
            category = SML.categorize_text_by_path_or_content(file_path, text)
            if not SML.entry_already_learned(category, text, file_path):
                SML.insert_to_database(category, text, file_path)
                return True
            return False
        except Exception:
            return False

    def tick(self, cycle_id: str = "") -> None:
        """Called by SelfAware each cycle. Index (bounded) then learn only NEW index entries."""
        if not _armed():
            return

        # Phase A: keep index fresh (bounded)
        scan_stats = self._bounded_index_scan()

        # Phase B: learn only new records
        new_files = self._fetch_new_files()
        learned = 0
        for fid, path, _typ in new_files:
            ok = self._learn_file(path)
            if ok:
                learned += 1
            self.last_file_id = max(self.last_file_id, fid)

        if new_files:
            self._save_state()

        if scan_stats.get("indexed") or new_files:
            log_event(
                "SYSINDEX_AUTOLEARN",
                f"scan_indexed={scan_stats.get('indexed')} new_rows={len(new_files)} learned={learned}",
                severity="INFO",
                cycle_id=cycle_id,
                meta={"scan": scan_stats, "new_rows": len(new_files), "learned": learned},
            )




# ---------------------------------------------------------------------------
# Dual-Brain Local Model Forge + Hardware Safety Governor
# ---------------------------------------------------------------------------

class ThermalResourceGovernor:
    """
    Workload governor to prevent GPU overheating / VRAM thrash.
    Conservative defaults; if telemetry is unavailable -> GPU work is disabled.

    NVIDIA path: NVML (pynvml) if installed.
    Cross-hardware fallback: CPU-only decisioning.
    """

    def __init__(self) -> None:
        self.gpu_temp_soft_c = int(os.getenv("GPU_TEMP_SOFT_C", "78"))
        self.gpu_temp_hard_c = int(os.getenv("GPU_TEMP_HARD_C", "83"))
        self.gpu_vram_soft_pct = float(os.getenv("GPU_VRAM_SOFT_PCT", "85"))
        self.gpu_vram_hard_pct = float(os.getenv("GPU_VRAM_HARD_PCT", "92"))

    def snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {"ok": False, "vendor": "unknown", "ts": int(time.time())}

        # NVIDIA NVML (best)
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)

            snap.update(
                {
                    "ok": True,
                    "vendor": "nvidia",
                    "gpu_temp_c": int(temp),
                    "gpu_util_pct": int(getattr(util, "gpu", 0)),
                    "gpu_mem_used": int(getattr(mem, "used", 0)),
                    "gpu_mem_total": int(getattr(mem, "total", 0)),
                }
            )
            if snap["gpu_mem_total"] > 0:
                snap["gpu_mem_pct"] = float(snap["gpu_mem_used"] / snap["gpu_mem_total"] * 100.0)
            return snap
        except Exception:
            pass

        # CPU fallback
        try:
            import psutil  # type: ignore

            snap.update({"ok": True, "vendor": "cpu", "cpu_util_pct": float(psutil.cpu_percent(interval=0.0))})
        except Exception:
            pass

        return snap

    def gpu_allowed(self) -> bool:
        snap = self.snapshot()
        # if we can't see GPU temp, we do NOT allow GPU compute (safety-first)
        if not snap.get("ok"):
            return False
        if snap.get("vendor") != "nvidia":
            return False
        t = snap.get("gpu_temp_c")
        if t is None:
            return False
        if int(t) >= int(self.gpu_temp_hard_c):
            return False
        mp = snap.get("gpu_mem_pct")
        if mp is not None and float(mp) >= float(self.gpu_vram_hard_pct):
            return False
        return True


class DualBrainLocalModelForge:
    """
    Two brains in conjunction:
      - Ledger: ./data/memory/datasets/synapses.db (governance + curated learnables; no spam)
      - Models: ./data/models/SarahMemory (portable local brain artifacts)

    This class does NOT modify SarahMemoryLLM.py. It stages the local brain
    foundation so a local responder can exist without OpenAI/3rd-party APIs.

    Responsibilities (lightweight):
      1) Ensure model directories exist (delegates to SarahMemorySynapes if present).
      2) Monitor synapses.db growth and schedule safe maintenance checkpoints.
      3) Maintain a small local embedding index DB placeholder (future wiring),
         without heavy dependencies.

    Heavy training is intentionally NOT executed here.
    """

    def __init__(self) -> None:
        self.base_dir = _base_dir()
        self.data_dir = _data_dir()
        self.datasets_dir = _datasets_dir()
        self.models_root = self.data_dir / "models" / "SarahMemory"
        self.emb_root = self.models_root / "embeddings"
        self.emb_db = self.emb_root / "brain_index.db"
        self.synapses_db = self.datasets_dir / "synapses.db"
        self.state_path = self.models_root / "forge_state.json"
        self.gov = ThermalResourceGovernor()

        self._state: Dict[str, Any] = {}
        self._last_tick = 0.0
        self._load_state()

        self.models_root.mkdir(parents=True, exist_ok=True)
        self.emb_root.mkdir(parents=True, exist_ok=True)
        self._ensure_emb_db()

        # Ensure synapses schema exists (if module available)
        try:
            if SYN is not None and hasattr(SYN, "initialize_synapses_database"):
                SYN.initialize_synapses_database()  # type: ignore
        except Exception:
            pass

    def _load_state(self) -> None:
        self._state = {
            "synapses_last_size": 0,
            "synapses_last_vacuum_ts": 0,
            "forge_version": "1",
        }
        try:
            if self.state_path.exists():
                obj = json.loads(self.state_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    self._state.update(obj)
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _ensure_emb_db(self) -> None:
        try:
            con = sqlite3.connect(str(self.emb_db))
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doc_key TEXT NOT NULL,
                  source TEXT NOT NULL,
                  ref TEXT NOT NULL,
                  chunk_id INTEGER NOT NULL,
                  text_hash TEXT NOT NULL,
                  text TEXT NOT NULL,
                  vector_json TEXT NOT NULL,
                  created_ts INTEGER NOT NULL
                )
                """
            )
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_unique ON embeddings(doc_key, chunk_id, text_hash)")
            con.commit()
            con.close()
        except Exception:
            pass

    def _synapses_size(self) -> int:
        try:
            if self.synapses_db.exists():
                return int(self.synapses_db.stat().st_size)
        except Exception:
            pass
        return 0

    def _maybe_checkpoint_synapses(self, cycle_id: str) -> None:
        if not self.synapses_db.exists():
            return
        now = int(time.time())
        last_vac = int(self._state.get("synapses_last_vacuum_ts") or 0)
        if now - last_vac < 6 * 3600:
            return

        size_b = self._synapses_size()
        last_size_b = int(self._state.get("synapses_last_size") or 0)
        if size_b < last_size_b + (256 * 1024 * 1024):
            return

        # safe checkpoint + vacuum
        try:
            con = sqlite3.connect(str(self.synapses_db))
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            con.execute("VACUUM;")
            con.close()

            self._state["synapses_last_vacuum_ts"] = now
            self._state["synapses_last_size"] = size_b
            self._save_state()

            log_event("SYNAPSES_DB_VACUUM", f"VACUUM ok size_bytes={size_b}", severity="INFO", cycle_id=cycle_id)
        except Exception as e:
            log_event("SYNAPSES_DB_VACUUM_FAIL", f"{type(e).__name__}: {e}", severity="WARN", cycle_id=cycle_id)

    def tick(self, cycle_id: str) -> None:
        if not _armed():
            return

        now = time.time()
        if (now - self._last_tick) < 8.0:
            return
        self._last_tick = now

        # ensure model dirs exist (idempotent)
        try:
            if SYN is not None and hasattr(SYN, "ensure_sarahmemory_model_dirs"):
                SYN.ensure_sarahmemory_model_dirs()  # type: ignore
        except Exception:
            pass

        # thermal visibility logging (only when in soft-zone)
        try:
            snap = self.gov.snapshot()
            if snap.get("vendor") == "nvidia" and snap.get("gpu_temp_c") is not None:
                if int(snap.get("gpu_temp_c", 0)) >= int(self.gov.gpu_temp_soft_c):
                    log_event("THERMAL_THROTTLE", f"GPU temp high: {snap.get('gpu_temp_c')}C", severity="WARN", cycle_id=cycle_id, meta=snap)
        except Exception:
            pass

        # synapses db hygiene (rare, gated)
        self._maybe_checkpoint_synapses(cycle_id)


def run_autonomous_loop() -> None:
    _ensure_dirs()

    if not _armed():
        msg = "SelfAware not armed (requires NEOSKYMATRIX=True and DEVELOPERSMODE=True). Exiting."
        logger.warning(msg)
        log_event("SELF_AWARE_EXIT", msg, severity="WARN", meta={"armed": False})
        return

    logger.info("SELF-AWARE ARMED: entering autonomous loop (governance-first).")
    log_event("SELF_AWARE_START", "Autonomous loop armed.", severity="INFO", meta={"armed": True})

    ensure_synapes_bootstrap()

    forge = DualBrainLocalModelForge()

    auto_learner = SystemIndexAutoLearner() if _SELF_AWARE_AUTOINDEX_ENABLED else None

    root = _base_dir()
    logs_dir_candidates = [
        _data_dir() / "logs",
        root / "logs",
        root / "api" / "server" / "logs",
    ]
    logs_dir = None
    for c in logs_dir_candidates:
        if c.exists():
            logs_dir = c
            break
    if logs_dir is None:
        logs_dir = _data_dir() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    while True:
        cycle_id = _cycle_id()
        t0 = time.time()

        try:
            # 0) Dual-brain model forge (local brain + safety + DB hygiene)
            try:
                forge.tick(cycle_id)
            except Exception:
                pass

            # 0) Index->Learn bridge (bounded, disabled by default for NVMe protection)
            if auto_learner is not None:
                try:
                    auto_learner.tick(cycle_id)
                except Exception:
                    pass

            # 1) Introspect codebase (bounded)
            scans = scan_codebase(root, max_files=220)
            core_count = len(scans)

            # 2) Mine logs for new errors (bounded)
            issues = find_recent_errors(logs_dir, max_files=25)
            issues = cognitive_rank(issues)

            # 3) Create cycle report payload
            payload = {
                "cycle_id": cycle_id,
                "ts": datetime.now().isoformat(),
                "armed": True,
                "base_dir": str(root),
                "data_dir": str(_data_dir()),
                "modules_scanned": core_count,
                "issues_found": len(issues),
                "top_issue": issues[0] if issues else None,
                "synapes_available": SYN is not None,
                "updater_available": UPD is not None,
                "cognitive_available": COG is not None,
                "elapsed_ms": None,
                "notes": "Autonomous analysis cycle. No core files modified in place.",
            }
            # lightweight signature of scans for change detection
            sig = hashlib.sha256(("".join([s.sha256 for s in scans[:80]])).encode("utf-8")).hexdigest()
            payload["code_signature"] = sig

            rp = write_cycle_report(cycle_id, payload)

            # 4) Audit event
            log_event(
                "SELF_AWARE_CYCLE",
                f"Cycle completed. scans={core_count} issues={len(issues)} report={rp.name}",
                severity="INFO",
                cycle_id=cycle_id,
                meta={"code_signature": sig},
            )

        except KeyboardInterrupt:
            logger.warning("SELF-AWARE STOPPED by operator (KeyboardInterrupt).")
            log_event("SELF_AWARE_STOP", "Operator interrupt.", severity="WARN", cycle_id=cycle_id)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.error("Cycle failure: %s", err)
            log_event("SELF_AWARE_CYCLE_FAIL", err, severity="ERROR", cycle_id=cycle_id, meta={"trace": traceback.format_exc()[:12000]})

        # 5) Sleep (autonomous cadence)
        dt_ms = int((time.time() - t0) * 1000)
        try:
            # patch elapsed_ms into report (best-effort)
            p = _reports_dir() / f"cycle_{cycle_id}.json"
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    obj["elapsed_ms"] = dt_ms
                    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        time.sleep(loop_sleep_seconds())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_autonomous_loop()
