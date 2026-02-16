"""
--== SarahMemory Project ==--
File: SarahMemorySelfAware.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0 (Experimental Autonomous Loop)
Date: 2026-02-16
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
https://www.sarahmemory.com
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com

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
        py_files = list(root.rglob("*.py"))
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
        files = sorted([p for p in log_dir.rglob("*.log")], key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
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
    # default: tight enough for lab testing, not insane
    return 10


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
