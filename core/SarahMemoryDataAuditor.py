"""--==The SarahMemory Project==--
File: SarahMemoryDataAuditor.py
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

Read-only SarahMemory data-layout auditor, focused tree generator, and future
bootable-AiOS installation verifier foundation.

Doctrine:
- Audit first. Do not move, delete, rewrite, or migrate files automatically.
- ../data/settings is the runtime configuration/state JSON home.
- ../data/memory/datasets is the SQLite DB home.
- ../data/logs is the log home.
- ../data/reports is the report home.
- ../data/drivers and ../data/boot/drivers are driver-package homes.
- ../data/models is the default local model inventory root.
- ../data/registry is deprecated for runtime JSON such as body_map.json and
vision_policy.json unless a future explicit registry-only contract is added.

Usage examples:
python SarahMemoryDataAuditor.py
python SarahMemoryDataAuditor.py --audit
python SarahMemoryDataAuditor.py --tree settings
python SarahMemoryDataAuditor.py --tree drivers --max-depth 4
python SarahMemoryDataAuditor.py --tree models --dirs-only
python SarahMemoryDataAuditor.py --deprecated-registry
python SarahMemoryDataAuditor.py --orphans
python SarahMemoryDataAuditor.py --gui

Notes:
This module is deliberately dependency-light and headless-safe. The optional
Tkinter GUI is launched only when requested and fails gracefully if Tkinter is
unavailable.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "data_layout_auditor"
# CATEGORY = "filesystem_integrity_and_data_layout"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "core_utility"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem_storage"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "data_auditor"
# FAMILY = "core_maintenance"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Read-only data layout auditor and focused tree/report generator. Future bootable AiOS installation verifier foundation. No automatic moves/deletes."
# --- SARAHMETA END ---

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover - expected to be usable standalone
    config = None  # type: ignore


VERSION_STR = "v9.0.0"
VERSION_TAG = "v800"
MODULE_NAME = "SarahMemoryDataAuditor"

DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".npm",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}

DEFAULT_EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

# These JSON names are runtime settings/state and should not live in data/registry.
DEPRECATED_REGISTRY_JSON_TARGETS = {
    "body_map.json": "settings",
    "vision_policy.json": "settings",
    "drivers.json": "settings",
    "driver_registry.json": "settings",
    "model_registry.json": "settings",
    "runtime_capabilities.json": "settings",
    "device_registry.json": "settings",
}

SETTINGS_JSON_NAMES = {
    "settings.json",
    "sarahnet.config.json",
    "sarahnet.comms.json",
    "contacts.json",
    "conversation_analysis.json",
    "runtime_environment_snapshot.json",
    "assurance_gate_snapshot.json",
    "body_map.json",
    "vision_policy.json",
    "drivers.json",
    "driver_registry.json",
    "model_registry.json",
    "runtime_capabilities.json",
    "device_registry.json",
    "audio_policy.json",
    "network_policy.json",
    "ui_runtime_profile.json",
}

DATASET_DB_NAMES_HINTS = {
    "ai_learning.db",
    "context_history.db",
    "cognitive_compass.db",
    "cognitive_self.db",
    "cognitive_thinker.db",
    "device_link.db",
    "email_system.db",
    "filesystem_logs.db",
    "functions.db",
    "meta.db",
    "migration_history.db",
    "neuron_axis.db",
    "operator_core.db",
    "personality1.db",
    "phase_c_sync.db",
    "phase_d_mesh.db",
    "programming.db",
    "reminders.db",
    "runtime_environment.db",
    "safety_policies.db",
    "security_governor.db",
    "selfaware_tickets.db",
    "software.db",
    "synapses.db",
    "system_index.db",
    "system_logs.db",
    "trust_registry.db",
    "user_profile.db",
    "windows10.db",
    "windows11.db",
}

TEXT_REPORT_SUFFIXES = {".txt", ".md", ".rst"}
LOG_SUFFIXES = {".log"}
DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
JSON_SUFFIXES = {".json"}
MODEL_SUFFIXES = {
    ".bin",
    ".safetensors",
    ".pth",
    ".pt",
    ".onnx",
    ".gguf",
    ".tflite",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico"}
CACHE_SUFFIXES = {".cache", ".tmp", ".temp", ".bak"}


@dataclass
class AuditFinding:
    severity: str                 # PASS | INFO | WARN | FAIL
    category: str                 # layout | deprecated_registry | orphan | missing_dir | etc.
    path: str
    message: str
    recommendation: str = ""
    suggested_destination: str = ""
    safe_to_auto_apply: bool = False


@dataclass
class TreeOptions:
    scope: str = "data-root"
    max_depth: int = 4
    include_files: bool = True
    include_sizes: bool = True
    dirs_only: bool = False
    max_entries: int = 1200
    compact: bool = False


class SarahMemoryDataAuditor:
    """Read-only data-layout auditor and filtered tree generator.

    The class is intentionally side-effect-light:
    - read-only scans by default;
    - writes reports only when write_report()/write_tree_report() is explicitly called;
    - never moves/deletes files;
    - never modifies SarahMemoryGlobals.py.
    """

    def __init__(self, base_dir: Optional[str | Path] = None, data_dir: Optional[str | Path] = None) -> None:
        self.base_dir = Path(base_dir or self._config_path("BASE_DIR") or Path(__file__).resolve().parent).resolve()
        self.data_dir = Path(data_dir or self._config_path("DATA_DIR") or (self.base_dir / "data")).resolve()
        self.settings_dir = Path(self._config_path("SETTINGS_DIR") or (self.data_dir / "settings")).resolve()
        self.datasets_dir = Path(self._config_path("DATASETS_DIR") or (self.data_dir / "memory" / "datasets")).resolve()
        self.models_dir = Path(self._config_path("MODELS_DIR") or (self.data_dir / "models")).resolve()
        self.logs_dir = Path(self._config_path("LOGS_DIR") or (self.data_dir / "logs")).resolve()
        self.reports_dir = Path(self._config_path("REPORTS_DIR") or (self.data_dir / "reports")).resolve()
        self.drivers_dir = (self.data_dir / "drivers").resolve()
        self.boot_drivers_dir = (self.data_dir / "boot" / "drivers").resolve()
        self.registry_dir = (self.data_dir / "registry").resolve()
        self.cache_dir = (self.data_dir / "cache").resolve()
        self.ui_dir = (self.data_dir / "ui").resolve()
        self.report_version_dir = (self.reports_dir / VERSION_TAG).resolve()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def sizeof_fmt(num: int | float) -> str:
        try:
            n = float(num)
        except Exception:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(n) < 1024.0:
                return f"{n:,.1f}{unit}"
            n /= 1024.0
        return f"{n:,.1f}PB"

    @staticmethod
    def safe_getsize(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    @staticmethod
    def safe_rel(path: Path, root: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    @staticmethod
    def _json_load(path: Path) -> Optional[Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(str(tmp), str(path))

    def _config_path(self, attr: str) -> Optional[str]:
        try:
            if config is not None and hasattr(config, attr):
                value = getattr(config, attr)
                if value:
                    return str(value)
        except Exception:
            pass
        return None

    def _path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except Exception:
            return False

    def _is_under(self, child: Path, parent: Path) -> bool:
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Scope mapping
    # ------------------------------------------------------------------

    def available_scopes(self) -> Dict[str, List[Path]]:
        return {
            "data-root": [self.data_dir],
            "settings": [self.settings_dir],
            "drivers": [self.drivers_dir, self.boot_drivers_dir],
            "runtime-drivers": [self.drivers_dir],
            "boot-drivers": [self.boot_drivers_dir],
            "models": [self.models_dir],
            "datasets": [self.datasets_dir],
            "memory": [(self.data_dir / "memory").resolve()],
            "logs": [self.logs_dir],
            "reports": [self.reports_dir],
            "registry": [self.registry_dir],
            "ui": [self.ui_dir],
            "cache": [self.cache_dir],
            "all-data": [self.data_dir],
        }

    def scope_roots(self, scope: str) -> List[Path]:
        key = (scope or "data-root").strip().lower()
        scopes = self.available_scopes()
        return scopes.get(key, [self.data_dir])

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def classify_path(self, path: Path) -> Dict[str, Any]:
        """Classify a path and return expected storage family information."""
        p = Path(path).resolve()
        name = p.name
        suffix = p.suffix.lower()
        rel_data = self.safe_rel(p, self.data_dir) if self._is_under(p, self.data_dir) else str(p)

        is_dir = False
        try:
            is_dir = p.is_dir()
        except Exception:
            pass

        expected_root = "review"
        kind = "unknown"
        reason = "No rule matched. Manual review recommended."

        if is_dir:
            kind = "directory"
            if p == self.settings_dir or self._is_under(p, self.settings_dir):
                expected_root = "settings"
                reason = "Runtime settings/configuration directory."
            elif p == self.datasets_dir or self._is_under(p, self.datasets_dir):
                expected_root = "memory/datasets"
                reason = "SQLite datasets/memory directory."
            elif p == self.models_dir or self._is_under(p, self.models_dir):
                expected_root = "models"
                reason = "Local model inventory directory."
            elif p == self.logs_dir or self._is_under(p, self.logs_dir):
                expected_root = "logs"
                reason = "Log directory."
            elif p == self.reports_dir or self._is_under(p, self.reports_dir):
                expected_root = "reports"
                reason = "Report directory."
            elif p == self.drivers_dir or self._is_under(p, self.drivers_dir):
                expected_root = "drivers"
                reason = "Runtime driver package directory."
            elif p == self.boot_drivers_dir or self._is_under(p, self.boot_drivers_dir):
                expected_root = "boot/drivers"
                reason = "Boot driver package directory."
            elif p == self.registry_dir or self._is_under(p, self.registry_dir):
                expected_root = "deprecated_registry"
                reason = "data/registry is deprecated for current runtime JSON files."
            return {
                "path": str(p),
                "relative_to_data": rel_data,
                "name": name,
                "kind": kind,
                "expected_root": expected_root,
                "reason": reason,
                "size_bytes": 0,
            }

        size = self.safe_getsize(p)

        if suffix in DB_SUFFIXES:
            kind = "sqlite_database"
            expected_root = "memory/datasets"
            reason = "Database files belong in data/memory/datasets."
        elif suffix in LOG_SUFFIXES:
            kind = "log"
            expected_root = "logs"
            reason = "Log files belong in data/logs."
        elif suffix in JSON_SUFFIXES:
            kind = "json_settings_or_state"
            if name in SETTINGS_JSON_NAMES:
                expected_root = "settings"
                reason = "Known runtime settings/state JSON belongs in data/settings."
            elif name in DEPRECATED_REGISTRY_JSON_TARGETS:
                expected_root = DEPRECATED_REGISTRY_JSON_TARGETS[name]
                reason = "Deprecated registry JSON should be generated/stored under data/settings."
            elif self._is_under(p, self.settings_dir):
                expected_root = "settings"
                reason = "JSON under data/settings is treated as runtime settings/state."
            elif self._is_under(p, self.models_dir):
                expected_root = "models"
                reason = "JSON under model folder may be model configuration/manifest."
            elif self._is_under(p, self.drivers_dir) or self._is_under(p, self.boot_drivers_dir):
                expected_root = "drivers"
                reason = "JSON under driver folder may be manifest/config."
            else:
                expected_root = "settings"
                reason = "Unclassified JSON in data should usually be reviewed for data/settings."
        elif suffix in MODEL_SUFFIXES:
            kind = "model_artifact"
            expected_root = "models"
            reason = "Model weight/runtime artifacts belong under data/models."
        elif suffix in TEXT_REPORT_SUFFIXES:
            kind = "text_or_report"
            expected_root = "reports"
            reason = "Text reports/snapshots generally belong in data/reports unless they are documentation."
        elif suffix in IMAGE_SUFFIXES:
            kind = "image_asset"
            expected_root = "review"
            reason = "Image may be UI/avatar/document asset; destination depends on purpose."
        elif suffix in CACHE_SUFFIXES or name.lower().endswith((".tmp", ".bak")):
            kind = "cache_or_backup"
            expected_root = "cache"
            reason = "Temporary/cache/backup artifacts should not stay loose in data root."
        else:
            kind = "file"
            expected_root = "review"
            reason = "Unknown file type; manual review recommended."

        return {
            "path": str(p),
            "relative_to_data": rel_data,
            "name": name,
            "kind": kind,
            "expected_root": expected_root,
            "reason": reason,
            "size_bytes": size,
        }

    def expected_destination_for(self, path: Path) -> str:
        cls = self.classify_path(path)
        root = cls.get("expected_root")
        name = Path(path).name
        mapping = {
            "settings": self.settings_dir / name,
            "memory/datasets": self.datasets_dir / name,
            "logs": self.logs_dir / name,
            "reports": self.reports_dir / name,
            "models": self.models_dir / name,
            "drivers": self.drivers_dir / name,
            "boot/drivers": self.boot_drivers_dir / name,
            "cache": self.cache_dir / name,
        }
        dest = mapping.get(str(root))
        return str(dest) if dest else "manual_review"

    # ------------------------------------------------------------------
    # Audit checks
    # ------------------------------------------------------------------

    def check_directory_health(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        required = [
            ("data", self.data_dir),
            ("settings", self.settings_dir),
            ("memory_datasets", self.datasets_dir),
            ("models", self.models_dir),
            ("logs", self.logs_dir),
            ("reports", self.reports_dir),
            ("drivers", self.drivers_dir),
            ("boot_drivers", self.boot_drivers_dir),
        ]
        for label, path in required:
            if self._path_exists(path):
                findings.append(AuditFinding(
                    severity="PASS",
                    category="directory_health",
                    path=str(path),
                    message=f"Required directory exists: {label}",
                ))
            else:
                findings.append(AuditFinding(
                    severity="WARN",
                    category="missing_directory",
                    path=str(path),
                    message=f"Directory missing: {label}",
                    recommendation="Create during runtime bootstrap or next patch if this feature is enabled.",
                    suggested_destination=str(path),
                    safe_to_auto_apply=False,
                ))
        return findings

    def find_deprecated_registry_files(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        if not self.registry_dir.exists():
            findings.append(AuditFinding(
                severity="PASS",
                category="deprecated_registry",
                path=str(self.registry_dir),
                message="data/registry does not exist. No deprecated registry files found.",
            ))
            return findings

        try:
            entries = sorted([p for p in self.registry_dir.rglob("*") if p.is_file()])
        except Exception as e:
            findings.append(AuditFinding(
                severity="WARN",
                category="deprecated_registry",
                path=str(self.registry_dir),
                message=f"Could not scan data/registry: {e}",
            ))
            return findings

        if not entries:
            findings.append(AuditFinding(
                severity="INFO",
                category="deprecated_registry",
                path=str(self.registry_dir),
                message="data/registry exists but has no files.",
                recommendation="Directory may be removed later after callers are migrated, but no automatic delete should occur now.",
            ))
            return findings

        for p in entries:
            name = p.name
            if name in DEPRECATED_REGISTRY_JSON_TARGETS or p.suffix.lower() == ".json":
                dest = self.expected_destination_for(self.settings_dir / name)
                if name in DEPRECATED_REGISTRY_JSON_TARGETS:
                    dest = str(self.settings_dir / name)
                findings.append(AuditFinding(
                    severity="WARN",
                    category="deprecated_registry_file",
                    path=str(p),
                    message=f"Runtime JSON found in deprecated data/registry path: {name}",
                    recommendation="Prefer data/settings first; migrate legacy file into data/settings; write future updates only to data/settings.",
                    suggested_destination=dest,
                    safe_to_auto_apply=False,
                ))
            else:
                findings.append(AuditFinding(
                    severity="INFO",
                    category="registry_manual_review",
                    path=str(p),
                    message="Non-JSON file found under data/registry; manual review recommended.",
                    recommendation="Keep only if a future registry-only contract is explicitly defined.",
                    safe_to_auto_apply=False,
                ))
        return findings

    def find_data_root_orphans(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        if not self.data_dir.exists():
            findings.append(AuditFinding(
                severity="WARN",
                category="data_root",
                path=str(self.data_dir),
                message="data directory does not exist.",
            ))
            return findings

        try:
            root_files = sorted([p for p in self.data_dir.iterdir() if p.is_file()])
        except Exception as e:
            findings.append(AuditFinding(
                severity="WARN",
                category="data_root",
                path=str(self.data_dir),
                message=f"Could not scan data root: {e}",
            ))
            return findings

        if not root_files:
            findings.append(AuditFinding(
                severity="PASS",
                category="data_root_orphans",
                path=str(self.data_dir),
                message="No loose files found directly under data root.",
            ))
            return findings

        for p in root_files:
            cls = self.classify_path(p)
            expected = str(cls.get("expected_root") or "review")
            dest = self.expected_destination_for(p)
            severity = "INFO" if expected == "review" else "WARN"
            findings.append(AuditFinding(
                severity=severity,
                category="data_root_orphan",
                path=str(p),
                message=f"Loose file in data root: {p.name} ({cls.get('kind')})",
                recommendation=cls.get("reason", "Manual review recommended."),
                suggested_destination=dest,
                safe_to_auto_apply=False,
            ))
        return findings

    def find_settings_contract_files(self) -> List[AuditFinding]:
        findings: List[AuditFinding] = []
        expected_runtime_files = [
            "settings.json",
            "body_map.json",
            "vision_policy.json",
            "model_registry.json",
            "drivers.json",
        ]
        for name in expected_runtime_files:
            p = self.settings_dir / name
            legacy = self.registry_dir / name
            if p.exists():
                findings.append(AuditFinding(
                    severity="PASS",
                    category="settings_contract",
                    path=str(p),
                    message=f"Settings/runtime file present: {name}",
                ))
            elif legacy.exists():
                findings.append(AuditFinding(
                    severity="WARN",
                    category="settings_contract_legacy_only",
                    path=str(legacy),
                    message=f"{name} exists only in deprecated data/registry path.",
                    recommendation="Read settings path first, migrate/copy legacy file into data/settings, then write only to settings.",
                    suggested_destination=str(p),
                    safe_to_auto_apply=False,
                ))
            else:
                findings.append(AuditFinding(
                    severity="INFO",
                    category="settings_contract_missing",
                    path=str(p),
                    message=f"Runtime settings/state file missing and should be generated when needed: {name}",
                    recommendation="Generate at runtime when the owning subsystem initializes; do not ship machine-specific JSON from GitHub.",
                    suggested_destination=str(p),
                    safe_to_auto_apply=False,
                ))
        return findings

    def audit(self) -> Dict[str, Any]:
        findings: List[AuditFinding] = []
        findings.extend(self.check_directory_health())
        findings.extend(self.find_deprecated_registry_files())
        findings.extend(self.find_data_root_orphans())
        findings.extend(self.find_settings_contract_files())

        counts: Dict[str, int] = {"PASS": 0, "INFO": 0, "WARN": 0, "FAIL": 0}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        ok = counts.get("FAIL", 0) == 0
        status = "ok" if ok and counts.get("WARN", 0) == 0 else "review_required" if ok else "fail"

        return {
            "ok": ok,
            "status": status,
            "module": MODULE_NAME,
            "version": VERSION_STR,
            "version_tag": VERSION_TAG,
            "generated_at": self.now_iso(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "python": platform.python_version(),
            },
            "paths": {
                "base_dir": str(self.base_dir),
                "data_dir": str(self.data_dir),
                "settings_dir": str(self.settings_dir),
                "datasets_dir": str(self.datasets_dir),
                "models_dir": str(self.models_dir),
                "logs_dir": str(self.logs_dir),
                "reports_dir": str(self.reports_dir),
                "drivers_dir": str(self.drivers_dir),
                "boot_drivers_dir": str(self.boot_drivers_dir),
                "registry_dir_deprecated": str(self.registry_dir),
            },
            "doctrine": {
                "read_only": True,
                "auto_move": False,
                "auto_delete": False,
                "settings_home": "data/settings",
                "datasets_home": "data/memory/datasets",
                "deprecated_registry": "data/registry",
            },
            "summary": counts,
            "findings": [asdict(f) for f in findings],
        }

    # ------------------------------------------------------------------
    # Tree rendering
    # ------------------------------------------------------------------

    def _iter_tree_entries(
        self,
        root: Path,
        max_depth: int,
        include_files: bool,
        dirs_only: bool,
        max_entries: int,
    ) -> Iterable[Tuple[Path, int, bool]]:
        """Yield (path, depth, is_dir) in stable order."""
        root = root.resolve()
        if not root.exists():
            return

        yielded = 0
        stack: List[Tuple[Path, int]] = [(root, 0)]
        while stack and yielded < max_entries:
            current, depth = stack.pop()
            try:
                is_dir = current.is_dir()
            except Exception:
                is_dir = False
            yield current, depth, is_dir
            yielded += 1

            if not is_dir:
                continue
            if depth >= max_depth:
                continue

            try:
                children = list(current.iterdir())
            except Exception:
                continue

            dirs: List[Path] = []
            files: List[Path] = []
            for child in children:
                nm = child.name
                if nm in DEFAULT_EXCLUDE_DIRS or nm in DEFAULT_EXCLUDE_FILES:
                    continue
                try:
                    if child.is_dir():
                        dirs.append(child)
                    elif include_files and not dirs_only:
                        files.append(child)
                except Exception:
                    continue
            # reverse because stack is LIFO
            for item in sorted(files, key=lambda p: p.name.lower(), reverse=True):
                stack.append((item, depth + 1))
            for item in sorted(dirs, key=lambda p: p.name.lower(), reverse=True):
                stack.append((item, depth + 1))

    def build_tree_text(self, options: Optional[TreeOptions] = None) -> str:
        opts = options or TreeOptions()
        roots = self.scope_roots(opts.scope)
        lines: List[str] = []
        lines.append(f"SarahMemory Focused Tree — scope={opts.scope}")
        lines.append("=" * 72)
        lines.append(f"Generated: {self.now_iso()}")
        lines.append(f"Base Dir: {self.base_dir}")
        lines.append(f"Data Dir: {self.data_dir}")
        lines.append(f"Options: max_depth={opts.max_depth}, include_files={opts.include_files and not opts.dirs_only}, include_sizes={opts.include_sizes}, max_entries={opts.max_entries}")
        lines.append("=" * 72)
        lines.append("")

        for root in roots:
            lines.append(f"ROOT: {root}")
            if not root.exists():
                lines.append("  [MISSING]")
                lines.append("")
                continue

            count = 0
            for path, depth, is_dir in self._iter_tree_entries(
                root=root,
                max_depth=max(0, int(opts.max_depth)),
                include_files=bool(opts.include_files),
                dirs_only=bool(opts.dirs_only),
                max_entries=max(1, int(opts.max_entries)),
            ):
                indent = "│   " * depth
                icon = "📁" if is_dir else "📄"
                name = path.name if depth > 0 else (path.name or str(path))
                size = ""
                if opts.include_sizes and not is_dir:
                    size = f" ({self.sizeof_fmt(self.safe_getsize(path))})"
                if opts.compact:
                    rel = self.safe_rel(path, root)
                    lines.append(f"{rel}{'/' if is_dir else ''}{size}")
                else:
                    lines.append(f"{indent}{icon} {name}{'/' if is_dir else ''}{size}")
                count += 1
            if count >= opts.max_entries:
                lines.append(f"  [TRUNCATED at max_entries={opts.max_entries}]")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def write_tree_report(self, options: Optional[TreeOptions] = None, output_path: Optional[str | Path] = None) -> Path:
        opts = options or TreeOptions()
        self.report_version_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path(output_path) if output_path else self.report_version_dir / f"data_tree_{opts.scope}_{ts}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.build_tree_text(opts), encoding="utf-8")
        return out

    def write_audit_report(self, output_path: Optional[str | Path] = None) -> Path:
        payload = self.audit()
        self.report_version_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path(output_path) if output_path else self.report_version_dir / f"data_layout_audit_{ts}.json"
        self._atomic_write_json(out, payload)
        return out

    # ------------------------------------------------------------------
    # Pretty printers
    # ------------------------------------------------------------------

    @staticmethod
    def _shorten(text: str, width: int = 110) -> str:
        t = str(text or "").replace("\n", " ").strip()
        return t if len(t) <= width else t[: max(0, width - 3)] + "..."

    def print_findings_table(self, findings: Sequence[Dict[str, Any] | AuditFinding]) -> None:
        rows: List[Dict[str, Any]] = []
        for item in findings:
            if isinstance(item, AuditFinding):
                rows.append(asdict(item))
            elif isinstance(item, dict):
                rows.append(item)
        if not rows:
            print("No findings.")
            return

        sev_w = max(len("SEV"), max(len(str(r.get("severity", ""))) for r in rows))
        cat_w = max(len("CATEGORY"), min(34, max(len(str(r.get("category", ""))) for r in rows)))
        path_w = 52
        header = f"{'SEV'.ljust(sev_w)}  {'CATEGORY'.ljust(cat_w)}  {'PATH'.ljust(path_w)}  MESSAGE"
        print(header)
        print("-" * len(header))
        for r in rows:
            sev = str(r.get("severity", ""))
            cat = self._shorten(str(r.get("category", "")), cat_w)
            path = self._shorten(str(r.get("path", "")), path_w)
            msg = self._shorten(str(r.get("message", "")), 120)
            print(f"{sev.ljust(sev_w)}  {cat.ljust(cat_w)}  {path.ljust(path_w)}  {msg}")

    def print_audit_summary(self, payload: Optional[Dict[str, Any]] = None) -> None:
        report = payload or self.audit()
        print(f"SarahMemory Data Layout Audit — {report.get('status')}")
        print("=" * 72)
        print(f"Generated: {report.get('generated_at')}")
        print(f"Base Dir : {report.get('paths', {}).get('base_dir')}")
        print(f"Data Dir : {report.get('paths', {}).get('data_dir')}")
        print(f"Summary  : {report.get('summary')}")
        print("")
        self.print_findings_table(report.get("findings") or [])

    # ------------------------------------------------------------------
    # Optional Tkinter GUI
    # ------------------------------------------------------------------

    def launch_gui(self) -> None:
        """Launch optional local Tkinter GUI. Headless-safe: only called explicitly."""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
        except Exception as e:
            print(f"[GUI unavailable] Tkinter import failed: {e}")
            return

        root = tk.Tk()
        root.title("SarahMemory Data Auditor")
        root.geometry("980x680")

        scope_var = tk.StringVar(value="settings")
        depth_var = tk.StringVar(value="4")
        include_files_var = tk.BooleanVar(value=True)
        include_sizes_var = tk.BooleanVar(value=True)
        dirs_only_var = tk.BooleanVar(value=False)
        max_entries_var = tk.StringVar(value="1200")

        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Scope:").pack(side=tk.LEFT)
        scopes = sorted(self.available_scopes().keys())
        scope_combo = ttk.Combobox(top, textvariable=scope_var, values=scopes, width=24, state="readonly")
        scope_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Max Depth:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=depth_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Max Entries:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(top, textvariable=max_entries_var, width=8).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(top, text="Files", variable=include_files_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Sizes", variable=include_sizes_var).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Dirs Only", variable=dirs_only_var).pack(side=tk.LEFT, padx=4)

        text = tk.Text(root, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        xscroll = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=text.xview)
        xscroll.pack(fill=tk.X, padx=8)
        text.configure(xscrollcommand=xscroll.set)

        def _opts() -> TreeOptions:
            try:
                depth = int(depth_var.get())
            except Exception:
                depth = 4
            try:
                max_entries = int(max_entries_var.get())
            except Exception:
                max_entries = 1200
            return TreeOptions(
                scope=scope_var.get(),
                max_depth=max(0, depth),
                include_files=bool(include_files_var.get()),
                include_sizes=bool(include_sizes_var.get()),
                dirs_only=bool(dirs_only_var.get()),
                max_entries=max(1, max_entries),
            )

        def show_tree() -> None:
            text.delete("1.0", tk.END)
            text.insert(tk.END, self.build_tree_text(_opts()))

        def show_audit() -> None:
            text.delete("1.0", tk.END)
            report = self.audit()
            text.insert(tk.END, json.dumps(report, indent=2))

        def save_tree() -> None:
            try:
                out = self.write_tree_report(_opts())
                messagebox.showinfo("Saved", f"Tree report saved:\n{out}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

        def save_audit() -> None:
            try:
                out = self.write_audit_report()
                messagebox.showinfo("Saved", f"Audit report saved:\n{out}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))

        def choose_data_dir() -> None:
            selected = filedialog.askdirectory(title="Select SarahMemory data directory")
            if selected:
                self.data_dir = Path(selected).resolve()
                self.settings_dir = self.data_dir / "settings"
                self.datasets_dir = self.data_dir / "memory" / "datasets"
                self.models_dir = self.data_dir / "models"
                self.logs_dir = self.data_dir / "logs"
                self.reports_dir = self.data_dir / "reports"
                self.drivers_dir = self.data_dir / "drivers"
                self.boot_drivers_dir = self.data_dir / "boot" / "drivers"
                self.registry_dir = self.data_dir / "registry"
                self.cache_dir = self.data_dir / "cache"
                self.ui_dir = self.data_dir / "ui"
                self.report_version_dir = self.reports_dir / VERSION_TAG
                show_tree()

        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="Show Tree", command=show_tree).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Show Audit JSON", command=show_audit).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Save Tree", command=save_tree).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Save Audit", command=save_audit).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Choose Data Dir", command=choose_data_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="Close", command=root.destroy).pack(side=tk.RIGHT, padx=4)

        show_tree()
        root.mainloop()


# ---------------------------------------------------------------------------
# CLI / menu entry points
# ---------------------------------------------------------------------------


def _print_menu() -> None:
    print("\nSarahMemory Data Auditor")
    print("=" * 72)
    print("1) Show data/settings tree")
    print("2) Show data/drivers + data/boot/drivers tree")
    print("3) Show data/models tree")
    print("4) Show data root orphan files")
    print("5) Show deprecated data/registry usage")
    print("6) Run full data layout audit")
    print("7) Write JSON audit report")
    print("8) Launch optional Tkinter GUI")
    print("9) Write focused tree report")
    print("0) Exit")


def run_interactive_menu() -> None:
    auditor = SarahMemoryDataAuditor()
    while True:
        _print_menu()
        choice = input("Select option: ").strip().lower()
        if choice in ("0", "q", "quit", "exit"):
            return
        if choice == "1":
            print(auditor.build_tree_text(TreeOptions(scope="settings", max_depth=5, include_files=True)))
        elif choice == "2":
            print(auditor.build_tree_text(TreeOptions(scope="drivers", max_depth=5, include_files=True)))
        elif choice == "3":
            print(auditor.build_tree_text(TreeOptions(scope="models", max_depth=3, include_files=True, max_entries=800)))
        elif choice == "4":
            auditor.print_findings_table(auditor.find_data_root_orphans())
        elif choice == "5":
            auditor.print_findings_table(auditor.find_deprecated_registry_files())
        elif choice == "6":
            auditor.print_audit_summary()
        elif choice == "7":
            out = auditor.write_audit_report()
            print(f"Audit report written: {out}")
        elif choice == "8":
            auditor.launch_gui()
        elif choice == "9":
            scope = input("Scope (settings/drivers/models/datasets/logs/reports/registry/data-root/all-data): ").strip() or "settings"
            try:
                depth = int(input("Max depth [4]: ").strip() or "4")
            except Exception:
                depth = 4
            out = auditor.write_tree_report(TreeOptions(scope=scope, max_depth=depth, include_files=True))
            print(f"Tree report written: {out}")
        else:
            print("Invalid selection.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SarahMemory data layout auditor and focused tree tool.")
    parser.add_argument("--base-dir", default=None, help="Override SarahMemory BASE_DIR.")
    parser.add_argument("--data-dir", default=None, help="Override SarahMemory DATA_DIR.")
    parser.add_argument("--audit", action="store_true", help="Run full data layout audit and print summary.")
    parser.add_argument("--write-audit", action="store_true", help="Run audit and write JSON report.")
    parser.add_argument("--tree", default=None, help="Render focused tree for scope. Example: settings, drivers, models, data-root.")
    parser.add_argument("--write-tree", default=None, help="Write focused tree report for scope.")
    parser.add_argument("--max-depth", type=int, default=4, help="Tree max depth.")
    parser.add_argument("--max-entries", type=int, default=1200, help="Tree max entries before truncation.")
    parser.add_argument("--dirs-only", action="store_true", help="Tree output should list directories only.")
    parser.add_argument("--no-files", action="store_true", help="Tree output should not include files.")
    parser.add_argument("--no-sizes", action="store_true", help="Tree output should not include file sizes.")
    parser.add_argument("--compact", action="store_true", help="Compact tree output.")
    parser.add_argument("--orphans", action="store_true", help="Show loose files directly under data root.")
    parser.add_argument("--deprecated-registry", action="store_true", help="Show deprecated data/registry usage.")
    parser.add_argument("--gui", action="store_true", help="Launch optional Tkinter GUI.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload for audit output.")
    parser.add_argument("--output", default=None, help="Optional output path for --write-audit or --write-tree.")
    parser.add_argument("--menu", action="store_true", help="Force interactive menu.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    auditor = SarahMemoryDataAuditor(base_dir=args.base_dir, data_dir=args.data_dir)

    if args.menu or len(sys.argv) <= 1:
        run_interactive_menu()
        return 0

    if args.gui:
        auditor.launch_gui()
        return 0

    tree_scope = args.tree or args.write_tree
    if tree_scope:
        opts = TreeOptions(
            scope=tree_scope,
            max_depth=max(0, int(args.max_depth)),
            include_files=not bool(args.no_files),
            include_sizes=not bool(args.no_sizes),
            dirs_only=bool(args.dirs_only),
            max_entries=max(1, int(args.max_entries)),
            compact=bool(args.compact),
        )
        if args.write_tree:
            out = auditor.write_tree_report(opts, output_path=args.output)
            print(f"Tree report written: {out}")
        else:
            print(auditor.build_tree_text(opts))
        return 0

    if args.orphans:
        auditor.print_findings_table(auditor.find_data_root_orphans())
        return 0

    if args.deprecated_registry:
        auditor.print_findings_table(auditor.find_deprecated_registry_files())
        return 0

    if args.write_audit:
        out = auditor.write_audit_report(output_path=args.output)
        print(f"Audit report written: {out}")
        return 0

    if args.audit:
        report = auditor.audit()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            auditor.print_audit_summary(report)
        return 0 if report.get("ok") else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Local-first semantic telemetry. Default behavior is RAM-buffered only; disk
# export is explicit and compact to avoid log/write storms.
import threading as _sm_semantic_threading
import time as _sm_semantic_time
import uuid as _sm_semantic_uuid

class SemanticTelemetryRecorder:
    """Low-thrash semantic trace recorder for agentic runtime events."""

    def __init__(self, max_events: int = 256) -> None:
        self.max_events = max(16, int(max_events or 256))
        self._events: List[Dict[str, Any]] = []
        self._lock = _sm_semantic_threading.RLock()

    def record(self, *, organ: str, action: str, verdict: str = "OBSERVED", task_id: str = "", trace_id: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {
            "ok": True,
            "ts": datetime.now().isoformat(),
            "trace_id": trace_id or _sm_semantic_uuid.uuid4().hex,
            "task_id": str(task_id or ""),
            "organ": str(organ or "unknown")[:120],
            "action": str(action or "unknown")[:160],
            "verdict": str(verdict or "OBSERVED")[:80],
            "risk_tier": str((meta or {}).get("risk_tier") or ""),
            "write_bytes": int((meta or {}).get("write_bytes") or 0),
            "network_used": bool((meta or {}).get("network_used") or False),
            "rollback_available": bool((meta or {}).get("rollback_available") or False),
            "meta": dict(meta or {}),
        }
        if self._lock:
            with self._lock:
                self._events.append(rec)
                if len(self._events) > self.max_events:
                    self._events = self._events[-self.max_events:]
        else:
            self._events.append(rec)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
        return rec

    def snapshot(self) -> Dict[str, Any]:
        events = list(self._events)
        return {"ok": True, "schema": "SarahMemory.semantic_telemetry.v1", "count": len(events), "events": events[-self.max_events:]}

    def export_compact_jsonl(self, path: str, *, max_events: int = 128) -> Dict[str, Any]:
        # Explicit export only. No automatic telemetry file writes.
        try:
            events = list(self._events)[-int(max_events or 128):]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            return {"ok": True, "path": path, "events_written": len(events)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


_SEMANTIC_TELEMETRY = SemanticTelemetryRecorder()


def record_semantic_telemetry(**kwargs: Any) -> Dict[str, Any]:
    return _SEMANTIC_TELEMETRY.record(**kwargs)


def get_semantic_telemetry_snapshot() -> Dict[str, Any]:
    return _SEMANTIC_TELEMETRY.snapshot()
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemoryDataAuditor.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryDataAuditor',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Governance',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['governance'],
    "supported_missions": ['Conversation', 'Execution', 'Governance', 'Security'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω060'],
    "required_authority": ['Read', 'Research'],
    "priority": 90,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryDataAuditor.py'},
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
        "component": 'SarahMemoryDataAuditor',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryDataAuditor', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

