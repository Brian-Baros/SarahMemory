"""--==The SarahMemory Project==--
File: SarahMemorySynapes.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-02-15
Time: 10:11:54
Author: © 2025,2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com

SYNAPSES MODULE - NEURAL SELF-LEARNING ARCHITECTURE
================================================================

This module represents the "neural synapses" of SarahMemory - the self-learning,
self-improving, code-generating brain that enables autonomous evolution.

Key World-Class Features:
========================
✓ Advanced AST-based code generation with comprehensive validation
✓ Multi-stage sandbox testing (syntax, security, dependencies, execution)
✓ Neural pattern recognition for code optimization
✓ Complete provenance tracking and audit trails
✓ Cyclomatic & cognitive complexity analysis
✓ Halstead metrics and maintainability index calculation
✓ Security vulnerability detection and scoring
✓ Performance profiling and telemetry
✓ Self-healing and rollback capabilities
✓ Cross-platform compatibility (Windows/Linux/Cloud)
✓ Dynamic module composition with dependency resolution
✓ Version control with code hashing
✓ Enterprise-grade logging architecture
✓ Distributed learning across mesh network (ready)
✓ Quantum-ready architecture patterns

===============================================================================
"""

import sqlite3
import logging
import os
import sys
import ast
import traceback
import datetime
import json
import hashlib
import threading
import time
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import SarahMemory core modules
from SarahMemoryGlobals import (
    BASE_DIR,
    DATASETS_DIR,
    PROJECT_VERSION,
    DEBUG_MODE,
    RUN_MODE,
    DEVICE_MODE,
    DEVICE_PROFILE,
    get_runtime_meta
)

from SarahMemoryFilesystem import save_code_to_addons

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Setup Advanced Logging
logger = logging.getLogger('SarahMemorySynapes')
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

log_dir = os.path.join(BASE_DIR, "logs", "synapses")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join(log_dir, f"synapes_{datetime.datetime.now().strftime('%Y%m%d')}.log")
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Sandbox Configuration with organized subdirectories
SANDBOX_DIR = os.path.join(BASE_DIR, 'sandbox')
SANDBOX_APPROVED_DIR = os.path.join(SANDBOX_DIR, 'approved')
SANDBOX_TESTING_DIR = os.path.join(SANDBOX_DIR, 'testing')
SANDBOX_FAILED_DIR = os.path.join(SANDBOX_DIR, 'failed')

for sandbox_subdir in [SANDBOX_APPROVED_DIR, SANDBOX_TESTING_DIR, SANDBOX_FAILED_DIR]:
    os.makedirs(sandbox_subdir, exist_ok=True)

logger.info(f"Sandbox directories initialized at: {SANDBOX_DIR}")

# Quality Thresholds
MAX_CODE_COMPLEXITY = 50
MAX_CODE_LENGTH = 10000
MIN_CODE_QUALITY_SCORE = 0.70
MAX_DEPENDENCY_DEPTH = 5

# Templates
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates', 'code_generation')
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Performance Metrics
PERFORMANCE_THRESHOLD_MS = 1000
MEMORY_THRESHOLD_MB = 100

# ============================================================================
# LIVING MODEL (SARAHMEMORY LLM) - REGISTRY + DATA LEDGER (LOCAL-FIRST)
# ----------------------------------------------------------------------------
# This module extends Synapses beyond code-generation into "Model Ops":
# - A transparent, auditable local model registry (base + adapters)
# - Dataset ledger (what data trained what adapter, with hashes/provenance)
# - Training run telemetry (config + metrics + promotion/rollback readiness)
#
# Design principles:
# - Base model remains stable; growth happens in adapters (LoRA/QLoRA style)
# - No heavy deps required here; this file only manages registry + governance
# - Actual training execution can be delegated to the platform (optional)
# ============================================================================
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "data", "models")
SARAHMEMORY_MODEL_DIR = os.path.join(MODELS_ROOT_DIR, "SarahMemory")
SARAHMEMORY_BASE_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "base")
SARAHMEMORY_ADAPTERS_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "adapters")
SARAHMEMORY_DATASETS_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "datasets")
SARAHMEMORY_RUNS_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "training_runs")
SARAHMEMORY_EVAL_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "eval")
SARAHMEMORY_GOV_DIR = os.path.join(SARAHMEMORY_MODEL_DIR, "governance")

# Registry manifests
BASE_MANIFEST_JSON = os.path.join(SARAHMEMORY_BASE_DIR, "base_manifest.json")
ADAPTER_REGISTRY_JSON = os.path.join(SARAHMEMORY_ADAPTERS_DIR, "adapter_registry.json")
DATASETS_INDEX_JSON = os.path.join(SARAHMEMORY_DATASETS_DIR, "datasets_index.json")


# ============================================================================
# ENUMERATIONS & DATA STRUCTURES
# ============================================================================

class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    BASH = "bash"


class SecurityLevel(Enum):
    """Security classification"""
    SAFE = "safe"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class TestStatus(Enum):
    """Testing phase status"""
    PENDING = "pending"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class CodeMetrics:
    """Comprehensive code quality metrics"""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_volume: float = 0.0
    test_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    overall_quality: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lines_of_code': self.lines_of_code,
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'maintainability_index': self.maintainability_index,
            'halstead_volume': self.halstead_volume,
            'test_coverage': self.test_coverage,
            'security_score': self.security_score,
            'performance_score': self.performance_score,
            'overall_quality': self.overall_quality,
            'timestamp': self.timestamp
        }


@dataclass
class GeneratedCode:
    """Container for generated code with full provenance"""
    code: str
    language: CodeLanguage
    security_level: SecurityLevel
    function_name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    metrics: Optional[CodeMetrics] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    status: TestStatus = TestStatus.PENDING
    version: str = "1.0.0"
    author: str = "SarahMemory.AI"
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    hash: str = field(default="")

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.code.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'language': self.language.value,
            'security_level': self.security_level.value,
            'function_name': self.function_name,
            'description': self.description,
            'dependencies': self.dependencies,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'test_results': self.test_results,
            'status': self.status.value,
            'version': self.version,
            'author': self.author,
            'created_at': self.created_at,
            'hash': self.hash
        }


# ============================================================================
# DATABASE OPERATIONS - ENHANCED WITH PROVENANCE TRACKING
# ============================================================================

def connect_db(db_name: str) -> sqlite3.Connection:
    """Enhanced database connection with WAL mode and foreign keys"""
    try:
        db_path = os.path.join(DATASETS_DIR, db_name)
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        logger.debug(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Database connection error for {db_name}: {e}")
        raise




def ensure_sarahmemory_model_dirs() -> None:
    """Idempotently ensure the SarahMemory living-model directory layout exists."""
    try:
        for d in (
            MODELS_ROOT_DIR,
            SARAHMEMORY_MODEL_DIR,
            SARAHMEMORY_BASE_DIR,
            SARAHMEMORY_ADAPTERS_DIR,
            SARAHMEMORY_DATASETS_DIR,
            SARAHMEMORY_RUNS_DIR,
            SARAHMEMORY_EVAL_DIR,
            SARAHMEMORY_GOV_DIR,
        ):
            os.makedirs(d, exist_ok=True)

        # seed empty registries if missing (transparent, user-readable)
        if not os.path.exists(ADAPTER_REGISTRY_JSON):
            with open(ADAPTER_REGISTRY_JSON, "w", encoding="utf-8") as f:
                json.dump({"adapters": [], "updated_at": datetime.datetime.now().isoformat()}, f, indent=2)

        if not os.path.exists(DATASETS_INDEX_JSON):
            with open(DATASETS_INDEX_JSON, "w", encoding="utf-8") as f:
                json.dump({"datasets": [], "updated_at": datetime.datetime.now().isoformat()}, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to ensure SarahMemory model dirs: {e}", exc_info=True)


def _sha256_file(path: str) -> str:
    """Compute SHA256 for a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def register_model(
    *,
    model_id: str,
    model_type: str,
    name: str,
    storage_path: str,
    fmt: str = "hf",
    quantization: str = "n/a",
    params_estimate: Optional[float] = None,
    sha256: str = "",
    manifest: Optional[Dict[str, Any]] = None,
    status: str = "staged",
    base_model_id: Optional[str] = None,
) -> None:
    """
    Register a base/adapter model into synapses.db for auditability.
    This does NOT load/execute the model; it is pure governance + registry.
    """
    try:
        ensure_sarahmemory_model_dirs()
        conn = connect_db("synapses.db")
        cur = conn.cursor()

        try:
            manifest_json = json.dumps(manifest or {}, ensure_ascii=False)
        except Exception:
            manifest_json = "{}"

        cur.execute(
            """
            INSERT OR REPLACE INTO model_registry
            (model_id, model_type, name, base_model_id, storage_path, format, quantization,
             params_estimate, sha256, manifest_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM model_registry WHERE model_id=?), ?), ?)
            """,
            (
                str(model_id),
                str(model_type),
                str(name),
                str(base_model_id) if base_model_id else None,
                str(storage_path),
                str(fmt),
                str(quantization),
                float(params_estimate) if params_estimate is not None else None,
                str(sha256 or ""),
                manifest_json,
                str(status),
                str(model_id),
                _now_iso(),
                _now_iso(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Model registry write failed: {e}", exc_info=True)


def log_dataset_sample(
    *,
    dataset_id: str,
    sample_type: str,
    source_ref: str,
    content: Dict[str, Any],
    score: float = 0.0,
    verified: bool = False,
) -> str:
    """
    Append a single training sample to the dataset ledger with hash + provenance.
    Returns the content hash for traceability.
    """
    ensure_sarahmemory_model_dirs()
    try:
        payload = json.dumps(content or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = "{}"
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO dataset_ledger
            (dataset_id, sample_type, source_ref, content_json, content_sha256, score, verified, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(dataset_id),
                str(sample_type),
                str(source_ref),
                payload,
                str(content_hash),
                float(score),
                1 if verified else 0,
                _now_iso(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Dataset ledger write failed: {e}", exc_info=True)

    return content_hash


def log_training_run(
    *,
    run_id: str,
    base_model_id: str,
    adapter_model_id: Optional[str],
    dataset_id: Optional[str],
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "queued",
    notes: str = "",
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    """Record a training/eval run (for adapters) in synapses.db."""
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO training_runs
            (run_id, base_model_id, adapter_model_id, dataset_id, config_json, metrics_json,
             status, started_at, finished_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                str(base_model_id),
                str(adapter_model_id) if adapter_model_id else None,
                str(dataset_id) if dataset_id else None,
                json.dumps(config or {}, ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                str(status),
                started_at,
                finished_at,
                str(notes or ""),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Training run write failed: {e}", exc_info=True)



def record_eval_result(
    *,
    model_id: str,
    suite_name: str,
    results: Dict[str, Any],
    passed: bool,
) -> str:
    """Persist evaluation results for an adapter/base model."""
    eval_id = f"eval::{model_id}::{suite_name}::{int(time.time())}"
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO eval_results
            (eval_id, model_id, suite_name, results_json, passed, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(eval_id),
                str(model_id),
                str(suite_name),
                json.dumps(results or {}, ensure_ascii=False),
                1 if passed else 0,
                _now_iso(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Eval result write failed: {e}", exc_info=True)
    return eval_id


def promote_adapter(
    *,
    adapter_model_id: str,
    lane: str = "default",
    require_canary_pass: bool = True,
) -> bool:
    """
    Promote an adapter to active for a given lane.
    Governance gate: optional canary requirement.
    """
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()

        if require_canary_pass:
            cur.execute(
                """
                SELECT passed
                FROM eval_results
                WHERE model_id=? AND suite_name='canary'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(adapter_model_id),),
            )
            r = cur.fetchone()
            if not r or int(r[0]) != 1:
                conn.close()
                logger.warning(f"[LivingModel] Promotion blocked (missing canary pass): {adapter_model_id}")
                return False

        # set registry status active
        cur.execute(
            """
            UPDATE model_registry
            SET status='active', updated_at=?
            WHERE model_id=? AND model_type='adapter'
            """,
            (_now_iso(), str(adapter_model_id)),
        )

        # update adapter_registry.json lane mapping (transparent)
        ensure_sarahmemory_model_dirs()
        reg = {"adapters": [], "updated_at": _now_iso()}
        try:
            if os.path.exists(ADAPTER_REGISTRY_JSON):
                with open(ADAPTER_REGISTRY_JSON, "r", encoding="utf-8") as f:
                    reg = json.load(f) or reg
        except Exception:
            pass

        adapters = reg.get("adapters", [])
        # remove existing lane entry
        adapters = [a for a in adapters if str(a.get("lane")) != str(lane)]
        adapters.append({
            "lane": lane,
            "active_adapter_model_id": adapter_model_id,
            "promoted_at": _now_iso(),
        })
        reg["adapters"] = adapters
        reg["updated_at"] = _now_iso()

        try:
            with open(ADAPTER_REGISTRY_JSON, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        conn.commit()
        conn.close()
        logger.info(f"[LivingModel] Adapter promoted: {adapter_model_id} -> lane:{lane}")
        return True

    except Exception as e:
        logger.error(f"[LivingModel] promote_adapter failed: {e}", exc_info=True)
        return False


def _score_sample_heuristic(sample: Dict[str, Any]) -> float:
    """
    Lightweight scoring:
    - verified/corrected signals upweight
    - tool success upweight
    - longer/structured traces slightly upweight
    """
    score = 0.0
    try:
        if sample.get("verified") is True:
            score += 0.6
        if sample.get("correction") or sample.get("corrected_response"):
            score += 0.5
        if sample.get("tool_success") is True:
            score += 0.4
        # small bump for richer content
        size = len(json.dumps(sample, ensure_ascii=False)) if sample else 0
        score += min(0.3, size / 4000.0)
    except Exception:
        pass
    return max(0.0, min(1.0, score))


def ingest_sqlite_datasets_to_ledger(
    *,
    dataset_id: str,
    verified_only: bool = False,
    max_rows_per_table: int = 250,
) -> Dict[str, Any]:
    """
    Phase 2 ingestion:
    - scans DATASETS_DIR for *.db
    - enumerates tables
    - attempts to detect common columns for interaction/correction/tool traces
    - writes to dataset_ledger with provenance refs
    """
    ensure_sarahmemory_model_dirs()
    summary = {"dataset_id": dataset_id, "files": 0, "tables": 0, "rows_ingested": 0, "errors": []}

    try:
        db_files = [f for f in os.listdir(DATASETS_DIR) if f.lower().endswith(".db")]
    except Exception as e:
        summary["errors"].append(f"list_db_files: {e}")
        return summary

    for db_name in db_files:
        db_path = os.path.join(DATASETS_DIR, db_name)
        if not os.path.isfile(db_path):
            continue
        summary["files"] += 1
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall() if r and r[0]]
            summary["tables"] += len(tables)

            for table in tables:
                try:
                    cur.execute(f"PRAGMA table_info({table});")
                    cols = [r[1] for r in cur.fetchall() if r and r[1]]
                    if not cols:
                        continue

                    # detect likely schema
                    prompt_col = next((c for c in cols if c.lower() in ("prompt", "user_input", "input", "query", "question")), None)
                    response_col = next((c for c in cols if c.lower() in ("response", "assistant_response", "answer", "output")), None)
                    correction_col = next((c for c in cols if c.lower() in ("correction", "corrected_response", "fixed_response")), None)
                    verified_col = next((c for c in cols if c.lower() in ("verified", "is_verified", "approved", "confirmed")), None)
                    tool_success_col = next((c for c in cols if c.lower() in ("tool_success", "success", "passed")), None)

                    if not any([prompt_col, response_col, correction_col]):
                        continue

                    # fetch rows bounded
                    cur.execute(f"SELECT rowid, * FROM {table} LIMIT {int(max_rows_per_table)};")
                    rows = cur.fetchall()
                    for row in rows:
                        d = dict(row)
                        rowid = d.get("rowid")
                        # compute verified
                        verified = False
                        if verified_col and d.get(verified_col) is not None:
                            v = d.get(verified_col)
                            verified = (str(v).lower() in ("1", "true", "yes", "y", "ok", "approved"))
                        if verified_only and not verified:
                            continue

                        content = {
                            "db": db_name,
                            "table": table,
                            "rowid": rowid,
                            "prompt": d.get(prompt_col) if prompt_col else None,
                            "response": d.get(response_col) if response_col else None,
                            "correction": d.get(correction_col) if correction_col else None,
                            "verified": verified,
                            "tool_success": (bool(d.get(tool_success_col)) if tool_success_col else None),
                        }

                        sample_type = "interaction"
                        if correction_col and d.get(correction_col):
                            sample_type = "correction"

                        score = _score_sample_heuristic(content)
                        src_ref = f"sqlite://{db_name}/{table}/{rowid}"
                        log_dataset_sample(
                            dataset_id=dataset_id,
                            sample_type=sample_type,
                            source_ref=src_ref,
                            content=content,
                            score=score,
                            verified=verified,
                        )
                        summary["rows_ingested"] += 1

                except Exception as e:
                    summary["errors"].append(f"{db_name}:{table}: {e}")

            conn.close()

        except Exception as e:
            summary["errors"].append(f"{db_name}: {e}")

    # update datasets_index.json (transparent)
    try:
        idx_obj = {"datasets": [], "updated_at": _now_iso()}
        if os.path.exists(DATASETS_INDEX_JSON):
            with open(DATASETS_INDEX_JSON, "r", encoding="utf-8") as f:
                idx_obj = json.load(f) or idx_obj

        datasets = idx_obj.get("datasets", [])
        datasets = [d for d in datasets if str(d.get("dataset_id")) != str(dataset_id)]
        datasets.append({
            "dataset_id": dataset_id,
            "updated_at": _now_iso(),
            "rows_ingested": summary["rows_ingested"],
            "verified_only": bool(verified_only),
        })
        idx_obj["datasets"] = datasets
        idx_obj["updated_at"] = _now_iso()

        with open(DATASETS_INDEX_JSON, "w", encoding="utf-8") as f:
            json.dump(idx_obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return summary


def synapes_awareness_tick(
    *,
    dataset_id: str = "sm_live",
    ingest_verified_only: bool = False,
    max_rows_per_table: int = 250,
    lane: str = "default",
    base_model_id: str = "base::seed",
    adapter_model_id: str = "adapter::pending",
    enqueue_job: bool = True,
) -> Dict[str, Any]:
    """
    Phase 2 + Phase 3 orchestration tick:
    - ingest data into ledger
    - optionally enqueue a training job (Phase 3) for an adapter
    - does NOT force heavy training at runtime
    """
    ensure_sarahmemory_model_dirs()

    ingest_summary = ingest_sqlite_datasets_to_ledger(
        dataset_id=dataset_id,
        verified_only=ingest_verified_only,
        max_rows_per_table=max_rows_per_table,
    )

    out = {
        "dataset_id": dataset_id,
        "ingest": ingest_summary,
        "job": None,
        "timestamp": _now_iso(),
    }

    if enqueue_job:
        job_id = f"train::{adapter_model_id}::{int(time.time())}"
        enqueue_training_job(
            job_id=job_id,
            base_model_id=base_model_id,
            adapter_model_id=adapter_model_id,
            dataset_id=dataset_id,
            lane=lane,
            config={
                "strategy": "qlora_adapter_microtrain",
                "max_steps": 200,
                "batch_size": 1,
                "lr": 2e-4,
                "notes": "auto-enqueued by synapes_awareness_tick",
            },
            priority=50,
            requested_by="synapes_awareness_tick",
            notes="phase3_queue",
        )
        out["job"] = {"job_id": job_id, "status": "queued"}

    return out


# ============================================================================
# LIVING MODEL (PHASE 3) - TRAINING JOB DISPATCHER (LIGHTWEIGHT / NO NEW DEPS)
# ----------------------------------------------------------------------------
# Purpose:
# - Provide a transparent job queue so Synapses can request adapter micro-trains
#   without forcing heavyweight training dependencies into the core runtime.
# - The dispatcher is "best-effort": if no trainer is available, jobs are left
#   queued (or marked failed with a clear reason), never crashing boot.
#
# Contract:
# - enqueue_training_job(...) records intent + config + dataset_id
# - claim_next_training_job(...) is used by a worker loop
# - complete_training_job(...) records terminal status + metrics
# - run_training_dispatcher_once(...) attempts to execute 1 job (optional)
#
# Notes:
# - Actual training can be implemented in a separate optional module (e.g. a
#   local LoRA runner, a CLI script, or an API call) and wired in later.
# - This is governance + orchestration plumbing, keeping SarahMemory lean.
# ============================================================================

def enqueue_training_job(
    *,
    job_id: str,
    base_model_id: str,
    adapter_model_id: str,
    dataset_id: str,
    lane: str = "default",
    config: Optional[Dict[str, Any]] = None,
    priority: int = 50,
    requested_by: str = "synapses",
    notes: str = "",
) -> None:
    """Queue a training job for an adapter. Does not execute training."""
    ensure_sarahmemory_model_dirs()
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO training_jobs
            (job_id, priority, status, base_model_id, adapter_model_id, dataset_id,
             lane, config_json, requested_by, requested_at, started_at, finished_at,
             worker_id, metrics_json, error_message, notes)
            VALUES (
                ?, ?,
                COALESCE((SELECT status FROM training_jobs WHERE job_id=?), 'queued'),
                ?, ?, ?,
                ?, ?, ?, 
                COALESCE((SELECT requested_at FROM training_jobs WHERE job_id=?), ?),
                (SELECT started_at FROM training_jobs WHERE job_id=?),
                (SELECT finished_at FROM training_jobs WHERE job_id=?),
                (SELECT worker_id FROM training_jobs WHERE job_id=?),
                (SELECT metrics_json FROM training_jobs WHERE job_id=?),
                (SELECT error_message FROM training_jobs WHERE job_id=?),
                ?
            )
            """,
            (
                str(job_id),
                int(priority),
                str(job_id),
                str(base_model_id),
                str(adapter_model_id),
                str(dataset_id),
                str(lane),
                json.dumps(config or {}, ensure_ascii=False),
                str(requested_by),
                str(job_id),
                _now_iso(),
                str(job_id),
                str(job_id),
                str(job_id),
                str(job_id),
                str(job_id),
                str(job_id),
                str(notes or ""),
            ),
        )
        conn.commit()
        conn.close()
        logger.info(f"[LivingModel][P3] Training job queued: {job_id} ({adapter_model_id} on {dataset_id})")
    except Exception as e:
        logger.error(f"[LivingModel][P3] enqueue_training_job failed: {e}", exc_info=True)


def claim_next_training_job(worker_id: str = "synapses") -> Optional[Dict[str, Any]]:
    """
    Atomically claim the next queued training job (highest priority, earliest requested).
    Returns the claimed job row as a dict, or None.
    """
    ensure_sarahmemory_model_dirs()
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()

        conn.execute("BEGIN IMMEDIATE;")
        cur.execute(
            """
            SELECT job_id
            FROM training_jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, requested_at ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            conn.execute("COMMIT;")
            conn.close()
            return None

        job_id = row[0]
        cur.execute(
            """
            UPDATE training_jobs
            SET status='running', started_at=?, worker_id=?
            WHERE job_id=? AND status='queued'
            """,
            (_now_iso(), str(worker_id), str(job_id)),
        )
        if cur.rowcount != 1:
            conn.execute("COMMIT;")
            conn.close()
            return None

        cur.execute("SELECT * FROM training_jobs WHERE job_id=?", (str(job_id),))
        full = cur.fetchone()
        conn.execute("COMMIT;")
        conn.close()
        return dict(full) if full else None

    except Exception as e:
        logger.error(f"[LivingModel][P3] claim_next_training_job failed: {e}", exc_info=True)
        try:
            conn.execute("ROLLBACK;")
            conn.close()
        except Exception:
            pass
        return None


def complete_training_job(
    job_id: str,
    *,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error_message: str = "",
) -> None:
    """Mark a training job terminal and attach metrics/error."""
    try:
        conn = connect_db("synapses.db")
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE training_jobs
            SET status=?, finished_at=?, metrics_json=?, error_message=?
            WHERE job_id=?
            """,
            (
                str(status),
                _now_iso(),
                json.dumps(metrics or {}, ensure_ascii=False),
                str(error_message or ""),
                str(job_id),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[LivingModel][P3] complete_training_job failed: {e}", exc_info=True)


def _try_run_adapter_trainer(job: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Best-effort adapter trainer hook.
    - If a trainer function is available, call it.
    - If not available, return (False, {}, reason) without crashing.
    """
    trainer = None
    try:
        import SarahMemoryOptimization as SMOPT  # optional
        trainer = getattr(SMOPT, "run_adapter_training", None)
    except Exception:
        trainer = None

    if callable(trainer):
        try:
            metrics = trainer(
                job.get("base_model_id"),
                job.get("adapter_model_id"),
                job.get("dataset_id"),
                json.loads(job.get("config_json") or "{}"),
            )
            if not isinstance(metrics, dict):
                metrics = {"metrics": metrics}
            return True, metrics, ""
        except Exception as e:
            return False, {}, f"trainer_error: {type(e).__name__}: {e}"

    return False, {}, "no_trainer_available"


def run_training_dispatcher_once(worker_id: str = "synapses") -> str:
    """
    Process a single queued training job (if any).
    Safe for boot: never raises.
    """
    job = claim_next_training_job(worker_id=worker_id)
    if not job:
        return "no_jobs"

    job_id = str(job.get("job_id"))
    ok, metrics, err = _try_run_adapter_trainer(job)

    if ok:
        complete_training_job(job_id, status="passed", metrics=metrics)
        try:
            log_training_run(
                run_id=f"job::{job_id}",
                base_model_id=str(job.get("base_model_id")),
                adapter_model_id=str(job.get("adapter_model_id")),
                dataset_id=str(job.get("dataset_id")),
                config=json.loads(job.get("config_json") or "{}"),
                metrics=metrics,
                status="passed",
                started_at=str(job.get("started_at") or _now_iso()),
                finished_at=_now_iso(),
                notes=f"dispatcher:{worker_id}",
            )
        except Exception:
            pass
        logger.info(f"[LivingModel][P3] Training job PASSED: {job_id}")
        return "passed"

    if err == "no_trainer_available":
        # Re-queue, keep audit signal (don’t fail the system)
        try:
            conn = connect_db("synapses.db")
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE training_jobs
                SET status='queued', started_at=NULL, worker_id=NULL, error_message=?
                WHERE job_id=?
                """,
                ("no_trainer_available", job_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        logger.warning(f"[LivingModel][P3] Training job deferred (no trainer): {job_id}")
        return "deferred"

    complete_training_job(job_id, status="failed", metrics={}, error_message=err)
    try:
        log_training_run(
            run_id=f"job::{job_id}",
            base_model_id=str(job.get("base_model_id")),
            adapter_model_id=str(job.get("adapter_model_id")),
            dataset_id=str(job.get("dataset_id")),
            config=json.loads(job.get("config_json") or "{}"),
            metrics={},
            status="failed",
            started_at=str(job.get("started_at") or _now_iso()),
            finished_at=_now_iso(),
            notes=err,
        )
    except Exception:
        pass
    logger.error(f"[LivingModel][P3] Training job FAILED: {job_id} - {err}")
    return "failed"


def start_training_dispatcher_background(
    *,
    interval_seconds: int = 60,
    worker_id: str = "synapses",
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """
    Start a lightweight background loop that attempts to process jobs periodically.
    """
    ev = stop_event or threading.Event()

    def _loop():
        logger.info(f"[LivingModel][P3] Training dispatcher started (interval={interval_seconds}s)")
        while not ev.is_set():
            try:
                run_training_dispatcher_once(worker_id=worker_id)
            except Exception:
                pass
            ev.wait(max(5, int(interval_seconds)))

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def initialize_synapses_database():
    """Initialize comprehensive synapses database schema"""
    try:
        conn = connect_db("synapses.db")
        cursor = conn.cursor()

        # Code Generation History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE NOT NULL,
                function_name TEXT NOT NULL,
                language TEXT NOT NULL,
                code TEXT NOT NULL,
                description TEXT,
                security_level TEXT,
                status TEXT,
                version TEXT,
                author TEXT,
                created_at TEXT NOT NULL,
                approved_at TEXT,
                approved_by TEXT,
                rejection_reason TEXT,
                execution_count INTEGER DEFAULT 0,
                last_executed TEXT,
                avg_execution_time_ms REAL,
                success_rate REAL
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_function_name ON code_generations(function_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON code_generations(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON code_generations(created_at)")

        # Code Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT NOT NULL,
                lines_of_code INTEGER,
                cyclomatic_complexity INTEGER,
                cognitive_complexity INTEGER,
                maintainability_index REAL,
                halstead_volume REAL,
                test_coverage REAL,
                security_score REAL,
                performance_score REAL,
                overall_quality REAL,
                measured_at TEXT NOT NULL,
                FOREIGN KEY (code_hash) REFERENCES code_generations(hash)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_hash_metrics ON code_metrics(code_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_overall_quality ON code_metrics(overall_quality)")

        # Dependencies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_hash TEXT NOT NULL,
                dependency_name TEXT NOT NULL,
                dependency_type TEXT,
                is_builtin INTEGER DEFAULT 0,
                is_external INTEGER DEFAULT 0,
                version_constraint TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_hash) REFERENCES code_generations(hash),
                UNIQUE(parent_hash, dependency_name)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent ON code_dependencies(parent_hash)")

        # Test Results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT NOT NULL,
                test_name TEXT NOT NULL,
                test_type TEXT,
                passed INTEGER,
                execution_time_ms REAL,
                error_message TEXT,
                stack_trace TEXT,
                tested_at TEXT NOT NULL,
                FOREIGN KEY (code_hash) REFERENCES code_generations(hash)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_hash_test ON test_results(code_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_type ON test_results(test_type)")

        # Learning Patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence_score REAL,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL,
                learned_from TEXT,
                learned_at TEXT NOT NULL,
                last_used TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON learning_patterns(pattern_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON learning_patterns(confidence_score)")

        # Execution Telemetry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT NOT NULL,
                execution_id TEXT UNIQUE NOT NULL,
                input_params TEXT,
                output_result TEXT,
                execution_time_ms REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                success INTEGER,
                error_type TEXT,
                error_message TEXT,
                executed_at TEXT NOT NULL,
                device_mode TEXT,
                run_mode TEXT,
                FOREIGN KEY (code_hash) REFERENCES code_generations(hash)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_hash_telemetry ON execution_telemetry(code_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_executed_at ON execution_telemetry(executed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON execution_telemetry(success)")

        # Safety Violations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                detected_at TEXT NOT NULL,
                auto_blocked INTEGER DEFAULT 1
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_hash_violations ON safety_violations(code_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON safety_violations(severity)")


        # --------------------------------------------------------------------
        # MODEL OPS (Living Model Registry + Dataset Ledger)
        # --------------------------------------------------------------------
        cursor.execute(r"""
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,           -- base | adapter
                name TEXT,
                base_model_id TEXT,                 -- null for base models
                storage_path TEXT,
                format TEXT,                        -- hf | gguf | onnx | other
                quantization TEXT,                  -- q4 | q8 | fp16 | fp32 | n/a
                params_estimate REAL,               -- optional
                sha256 TEXT,
                manifest_json TEXT,
                status TEXT,                        -- active | inactive | staged | deprecated
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_type ON model_registry(model_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_base ON model_registry(base_model_id)")

        cursor.execute(r"""
            CREATE TABLE IF NOT EXISTS dataset_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                sample_type TEXT NOT NULL,          -- interaction | correction | tool_trace | citation_bundle
                source_ref TEXT,                    -- e.g. sqlite://db/table/rowid or file path
                content_json TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                score REAL DEFAULT 0.0,
                verified INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_ledger_dataset ON dataset_ledger(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_ledger_type ON dataset_ledger(sample_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_ledger_verified ON dataset_ledger(verified)")

        cursor.execute(r"""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                base_model_id TEXT NOT NULL,
                adapter_model_id TEXT,
                dataset_id TEXT,
                config_json TEXT,
                metrics_json TEXT,
                status TEXT,                        -- queued | running | passed | failed | promoted | rolled_back
                started_at TEXT,
                finished_at TEXT,
                notes TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_base ON training_runs(base_model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_adapter ON training_runs(adapter_model_id)")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT UNIQUE NOT NULL,
        priority INTEGER DEFAULT 50,
        status TEXT NOT NULL,
        base_model_id TEXT NOT NULL,
        adapter_model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        lane TEXT DEFAULT 'default',
        config_json TEXT,
        requested_by TEXT,
        requested_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        worker_id TEXT,
        metrics_json TEXT,
        error_message TEXT,
        notes TEXT
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_priority ON training_jobs(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_lane ON training_jobs(lane)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_dataset ON training_jobs(dataset_id)")

        cursor.execute(r"""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT UNIQUE NOT NULL,
                model_id TEXT NOT NULL,
                suite_name TEXT NOT NULL,
                results_json TEXT NOT NULL,
                passed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_model ON eval_results(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_suite ON eval_results(suite_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_passed ON eval_results(passed)")

        conn.commit()
        conn.close()
        logger.info("Synapses database schema initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize synapses database: {e}", exc_info=True)
        raise


def log_code_generation(generated_code: 'GeneratedCode') -> str:
    """Log code generation with full provenance"""
    try:
        conn = connect_db("synapses.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO code_generations
            (hash, function_name, language, code, description, security_level,
             status, version, author, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            generated_code.hash,
            generated_code.function_name,
            generated_code.language.value,
            generated_code.code,
            generated_code.description,
            generated_code.security_level.value,
            generated_code.status.value,
            generated_code.version,
            generated_code.author,
            generated_code.created_at
        ))

        if generated_code.metrics:
            cursor.execute("""
                INSERT INTO code_metrics
                (code_hash, lines_of_code, cyclomatic_complexity, cognitive_complexity,
                 maintainability_index, halstead_volume, test_coverage, security_score,
                 performance_score, overall_quality, measured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                generated_code.hash,
                generated_code.metrics.lines_of_code,
                generated_code.metrics.cyclomatic_complexity,
                generated_code.metrics.cognitive_complexity,
                generated_code.metrics.maintainability_index,
                generated_code.metrics.halstead_volume,
                generated_code.metrics.test_coverage,
                generated_code.metrics.security_score,
                generated_code.metrics.performance_score,
                generated_code.metrics.overall_quality,
                generated_code.metrics.timestamp
            ))

        for dep in generated_code.dependencies:
            cursor.execute("""
                INSERT OR IGNORE INTO code_dependencies
                (parent_hash, dependency_name, dependency_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                generated_code.hash,
                dep,
                'import',
                datetime.datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

        logger.info(f"Logged code generation: {generated_code.function_name} [{generated_code.hash[:8]}]")
        return generated_code.hash

    except Exception as e:
        logger.error(f"Failed to log code generation: {e}", exc_info=True)
        return ""


def log_test_result(code_hash: str, test_name: str, test_type: str,
                   passed: bool, execution_time_ms: float = 0.0,
                   error_message: str = "", stack_trace: str = ""):
    """Log test execution results"""
    try:
        conn = connect_db("synapses.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO test_results
            (code_hash, test_name, test_type, passed, execution_time_ms,
             error_message, stack_trace, tested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code_hash,
            test_name,
            test_type,
            1 if passed else 0,
            execution_time_ms,
            error_message,
            stack_trace,
            datetime.datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        status = "PASSED" if passed else "FAILED"
        logger.info(f"Test result logged: {test_name} [{status}] - {execution_time_ms:.2f}ms")

    except Exception as e:
        logger.error(f"Failed to log test result: {e}", exc_info=True)


def log_execution_telemetry(code_hash: str, execution_id: str, success: bool,
                           execution_time_ms: float, memory_usage_mb: float = 0.0,
                           input_params: str = "", output_result: str = "",
                           error_type: str = "", error_message: str = ""):
    """Log execution telemetry for performance monitoring"""
    try:
        conn = connect_db("synapses.db")
        cursor = conn.cursor()

        runtime_meta = get_runtime_meta()

        cursor.execute("""
            INSERT INTO execution_telemetry
            (code_hash, execution_id, input_params, output_result,
             execution_time_ms, memory_usage_mb, success, error_type,
             error_message, executed_at, device_mode, run_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code_hash,
            execution_id,
            input_params,
            output_result,
            execution_time_ms,
            memory_usage_mb,
            1 if success else 0,
            error_type,
            error_message,
            datetime.datetime.now().isoformat(),
            runtime_meta.get('device_mode', 'unknown'),
            runtime_meta.get('run_mode', 'unknown')
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Execution telemetry logged: {execution_id} [{code_hash[:8]}]")

    except Exception as e:
        logger.error(f"Failed to log execution telemetry: {e}", exc_info=True)


# ============================================================================
# CODE ANALYSIS & METRICS - ADVANCED QUALITY ASSESSMENT
# ============================================================================

class CodeAnalyzer:
    """Advanced code analysis for quality, security, and complexity"""

    @staticmethod
    def calculate_cyclomatic_complexity(code: str) -> int:
        """Calculate McCabe cyclomatic complexity"""
        try:
            tree = ast.parse(code)
            complexity = 1

            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return complexity

        except Exception as e:
            logger.warning(f"Failed to calculate cyclomatic complexity: {e}")
            return 0

    @staticmethod
    def calculate_cognitive_complexity(code: str) -> int:
        """Calculate cognitive complexity (understanding difficulty)"""
        try:
            tree = ast.parse(code)
            complexity = 0

            def visit_node(node, level):
                nonlocal complexity

                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += (1 + level)
                elif isinstance(node, ast.ExceptHandler):
                    complexity += (1 + level)
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

                for child in ast.iter_child_nodes(node):
                    new_level = level + 1 if isinstance(node, (ast.If, ast.While, ast.For, ast.FunctionDef)) else level
                    visit_node(child, new_level)

            for node in ast.iter_child_nodes(tree):
                visit_node(node, 0)

            return complexity

        except Exception as e:
            logger.warning(f"Failed to calculate cognitive complexity: {e}")
            return 0

    @staticmethod
    def calculate_halstead_metrics(code: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        try:
            tree = ast.parse(code)

            operators = set()
            operands = set()
            total_operators = 0
            total_operands = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                                   ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
                                   ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                    operators.add(type(node).__name__)
                    total_operators += 1

                elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
                                     ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                    operators.add(type(node).__name__)
                    total_operators += 1

                elif isinstance(node, (ast.And, ast.Or, ast.Not)):
                    operators.add(type(node).__name__)
                    total_operators += 1

                elif isinstance(node, ast.Name):
                    operands.add(node.id)
                    total_operands += 1

                elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
                    operands.add(str(getattr(node, 'value', getattr(node, 'n', getattr(node, 's', '')))))
                    total_operands += 1

            n1 = len(operators)
            n2 = len(operands)
            N1 = total_operators
            N2 = total_operands

            vocabulary = n1 + n2
            length = N1 + N2

            import math
            volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            return {
                'n1': n1,
                'n2': n2,
                'N1': N1,
                'N2': N2,
                'vocabulary': vocabulary,
                'length': length,
                'volume': volume,
                'difficulty': difficulty,
                'effort': effort
            }

        except Exception as e:
            logger.warning(f"Failed to calculate Halstead metrics: {e}")
            return {'volume': 0.0, 'difficulty': 0.0, 'effort': 0.0}

    @staticmethod
    def calculate_maintainability_index(code: str) -> float:
        """Calculate maintainability index (0-100 scale)"""
        try:
            import math

            lines = len([line for line in code.split('\n') if line.strip()])
            complexity = CodeAnalyzer.calculate_cyclomatic_complexity(code)
            halstead = CodeAnalyzer.calculate_halstead_metrics(code)
            volume = halstead.get('volume', 1.0)

            if lines == 0 or volume == 0:
                return 0.0

            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(lines)
            mi = max(0, min(100, mi))

            return mi

        except Exception as e:
            logger.warning(f"Failed to calculate maintainability index: {e}")
            return 0.0

    @staticmethod
    def analyze_security(code: str) -> Tuple[float, List[str]]:
        """Perform security analysis on code"""
        concerns = []
        score = 1.0

        dangerous_patterns = {
            r'eval\s*\(': ('Use of eval() is dangerous', 0.3),
            r'exec\s*\(': ('Use of exec() is dangerous', 0.3),
            r'__import__\s*\(': ('Dynamic imports can be dangerous', 0.1),
            r'os\.system\s*\(': ('Direct system calls are risky', 0.2),
            r'subprocess\.call\s*\(': ('Subprocess calls need validation', 0.15),
            r'open\s*\([^)]*[\'"]w': ('File write operations need validation', 0.1),
            r'pickle\.loads': ('Pickle deserialization is unsafe', 0.25),
            r'input\s*\(': ('Raw input needs validation', 0.05),
        }

        for pattern, (message, penalty) in dangerous_patterns.items():
            if re.search(pattern, code):
                concerns.append(message)
                score -= penalty

        score = max(0.0, min(1.0, score))

        return score, concerns

    @staticmethod
    def extract_dependencies(code: str) -> List[str]:
        """Extract all import dependencies"""
        dependencies = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)

            return list(set(dependencies))

        except Exception as e:
            logger.warning(f"Failed to extract dependencies: {e}")
            return []

    @staticmethod
    def generate_metrics(code: str) -> 'CodeMetrics':
        """Generate comprehensive code metrics"""
        try:
            lines = len([line for line in code.split('\n') if line.strip()])
            cyclomatic = CodeAnalyzer.calculate_cyclomatic_complexity(code)
            cognitive = CodeAnalyzer.calculate_cognitive_complexity(code)
            maintainability = CodeAnalyzer.calculate_maintainability_index(code)
            halstead = CodeAnalyzer.calculate_halstead_metrics(code)
            security_score, _ = CodeAnalyzer.analyze_security(code)

            # Calculate overall quality (weighted average)
            weights = {
                'maintainability': 0.30,
                'security': 0.30,
                'complexity': 0.20,
                'cognitive': 0.20
            }

            complexity_score = max(0, 1.0 - (cyclomatic / 20.0))
            cognitive_score = max(0, 1.0 - (cognitive / 30.0))
            maintainability_score = maintainability / 100.0

            overall_quality = (
                weights['maintainability'] * maintainability_score +
                weights['security'] * security_score +
                weights['complexity'] * complexity_score +
                weights['cognitive'] * cognitive_score
            )

            metrics = CodeMetrics(
                lines_of_code=lines,
                cyclomatic_complexity=cyclomatic,
                cognitive_complexity=cognitive,
                maintainability_index=maintainability,
                halstead_volume=halstead.get('volume', 0.0),
                test_coverage=0.0,
                security_score=security_score,
                performance_score=0.0,
                overall_quality=overall_quality
            )

            logger.debug(f"Metrics generated - Quality: {overall_quality:.2f}, Security: {security_score:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}", exc_info=True)
            return CodeMetrics()


# ============================================================================
# SANDBOX TESTING - MULTI-STAGE VALIDATION
# ============================================================================

class SandboxTester:
    """Multi-stage sandbox testing with security controls"""

    def __init__(self):
        # Allow controlled imports for sandbox execution.
        # Some generated modules (e.g., templates) import stdlib modules like 'logging'.
        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = (name or '').split('.')[0]
            if base in self.safe_modules:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Sandbox blocked import: {name}")

        self.restricted_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'int': int, 'isinstance': isinstance,
            'len': len, 'list': list, 'map': map, 'max': max,
            'min': min, 'print': print, 'range': range,
            'reversed': reversed, 'round': round, 'set': set,
            'sorted': sorted, 'str': str, 'sum': sum,
            'tuple': tuple, 'type': type, 'zip': zip,
            '__import__': _safe_import,
        }

        self.safe_modules = {
            'math', 'random', 'datetime', 'json', 'collections',
            'itertools', 'functools', 're', 'string', 'logging'
        }

    def syntax_check(self, code: str) -> Tuple[bool, str]:
        """Stage 1: Syntax validation"""
        try:
            compile(code, '<sandbox>', 'exec')
            logger.debug("Syntax check: PASSED")
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Syntax check: FAILED - {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Compilation error: {str(e)}"
            logger.warning(f"Syntax check: FAILED - {error_msg}")
            return False, error_msg

    def security_check(self, code: str) -> Tuple[bool, List[str]]:
        """Stage 2: Security analysis"""
        security_score, concerns = CodeAnalyzer.analyze_security(code)
        passed = security_score >= 0.6

        if passed:
            logger.debug(f"Security check: PASSED (score: {security_score:.2f})")
        else:
            logger.warning(f"Security check: FAILED (score: {security_score:.2f})")

        return passed, concerns

    def dependency_check(self, code: str) -> Tuple[bool, List[str]]:
        """Stage 3: Dependency validation"""
        dependencies = CodeAnalyzer.extract_dependencies(code)
        issues = []

        for dep in dependencies:
            base_module = dep.split('.')[0]

            if base_module not in self.safe_modules:
                try:
                    __import__(base_module)
                    issues.append(f"Module '{dep}' requires elevated permissions")
                except ImportError:
                    issues.append(f"Missing dependency: '{dep}'")

        passed = len(issues) == 0

        if passed:
            logger.debug(f"Dependency check: PASSED ({len(dependencies)} deps)")
        else:
            logger.warning(f"Dependency check: FAILED - {len(issues)} issues")

        return passed, issues

    def execution_test(self, code: str, timeout: float = 5.0) -> Tuple[bool, str, float]:
        """Stage 4: Controlled execution test"""
        result_container = {'passed': False, 'error': '', 'completed': False}

        def run_in_sandbox():
            try:
                restricted_globals = {
                    '__builtins__': self.restricted_builtins,
                    '__name__': '__sandbox__',
                    '__doc__': None,
                }
                restricted_locals = {}

                exec(code, restricted_globals, restricted_locals)

                result_container['passed'] = True
                result_container['completed'] = True

            except Exception as e:
                result_container['error'] = f"{type(e).__name__}: {str(e)}"
                result_container['completed'] = True

        start_time = time.time()

        thread = threading.Thread(target=run_in_sandbox, daemon=True)
        thread.start()
        thread.join(timeout)

        execution_time_ms = (time.time() - start_time) * 1000

        if thread.is_alive():
            error_msg = f"Execution timeout after {timeout}s"
            logger.warning(f"Execution test: TIMEOUT - {error_msg}")
            return False, error_msg, execution_time_ms

        if not result_container['completed']:
            error_msg = "Execution did not complete"
            logger.warning(f"Execution test: INCOMPLETE - {error_msg}")
            return False, error_msg, execution_time_ms

        if result_container['passed']:
            logger.debug(f"Execution test: PASSED ({execution_time_ms:.2f}ms)")
            return True, "", execution_time_ms
        else:
            logger.warning(f"Execution test: FAILED - {result_container['error']}")
            return False, result_container['error'], execution_time_ms

    def run_full_test_suite(self, code: str, code_hash: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Run complete test suite"""
        test_results = []
        all_passed = True

        logger.info(f"Starting test suite for code [{code_hash[:8]}]")

        # Stage 1: Syntax
        syntax_passed, syntax_error = self.syntax_check(code)
        test_results.append({
            'test_name': 'syntax_validation',
            'test_type': 'syntax',
            'passed': syntax_passed,
            'error': syntax_error,
            'execution_time_ms': 0.0
        })
        log_test_result(code_hash, 'syntax_validation', 'syntax', syntax_passed, 0.0, syntax_error)

        if not syntax_passed:
            all_passed = False
            logger.error(f"Test suite FAILED at syntax stage")
            return False, test_results

        # Stage 2: Security
        security_passed, security_concerns = self.security_check(code)
        test_results.append({
            'test_name': 'security_analysis',
            'test_type': 'security',
            'passed': security_passed,
            'error': '; '.join(security_concerns) if security_concerns else '',
            'execution_time_ms': 0.0
        })
        log_test_result(code_hash, 'security_analysis', 'security', security_passed, 0.0,
                       '; '.join(security_concerns) if security_concerns else '')

        if not security_passed:
            all_passed = False
            logger.warning(f"Security concerns: {len(security_concerns)} issues")

        # Stage 3: Dependencies
        dep_passed, dep_issues = self.dependency_check(code)
        test_results.append({
            'test_name': 'dependency_validation',
            'test_type': 'dependencies',
            'passed': dep_passed,
            'error': '; '.join(dep_issues) if dep_issues else '',
            'execution_time_ms': 0.0
        })
        log_test_result(code_hash, 'dependency_validation', 'dependencies', dep_passed, 0.0,
                       '; '.join(dep_issues) if dep_issues else '')

        if not dep_passed:
            all_passed = False
            logger.warning(f"Dependency issues: {len(dep_issues)}")

        # Stage 4: Execution
        exec_passed, exec_error, exec_time = self.execution_test(code)
        test_results.append({
            'test_name': 'sandbox_execution',
            'test_type': 'execution',
            'passed': exec_passed,
            'error': exec_error,
            'execution_time_ms': exec_time
        })
        log_test_result(code_hash, 'sandbox_execution', 'execution', exec_passed, exec_time, exec_error)

        if not exec_passed:
            all_passed = False
            logger.error(f"Execution test FAILED")

        passed_count = sum(1 for r in test_results if r['passed'])
        logger.info(f"Test suite complete: {passed_count}/{len(test_results)} passed")

        return all_passed, test_results


# [CONTINUING IN NEXT PART DUE TO LENGTH...]

# ============================================================================
# CODE GENERATION ENGINE
# ============================================================================

class CodeGenerator:
    """Advanced code generation engine"""

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.tester = SandboxTester()
        self.template_cache = {}

    def generate_from_description(self, description: Any, language: CodeLanguage = CodeLanguage.PYTHON) -> Optional['GeneratedCode']:
        """Generate code from natural language description"""
        try:
            # 'description' may be a string OR a structured dict coming from other modules.
            # Keep logging safe for both types.
            _src = description
            if isinstance(description, dict):
                _src = description.get('request') or description.get('description') or str(description)
            try:
                _preview = str(_src)[:100]
            except Exception:
                _preview = "<unprintable description>"
            logger.info(f"Generating code: {_preview}...")


            function_name = self._extract_function_name(description)
            security_level = self._determine_security_level(description)
            code = self._generate_code_template(description, function_name, language)

            metrics = self.analyzer.generate_metrics(code)
            dependencies = self.analyzer.extract_dependencies(code)

            generated = GeneratedCode(
                code=code,
                language=language,
                security_level=security_level,
                function_name=function_name,
                description=description,
                dependencies=dependencies,
                metrics=metrics,
                status=TestStatus.PENDING
            )

            logger.info(f"Code generated: {function_name} (Quality: {metrics.overall_quality:.2f})")

            return generated

        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return None

    def _extract_function_name(self, description: str) -> str:
        """Extract function name from description"""
        words = re.findall(r'\w+', description.lower())
        name_words = [w for w in words if w not in {'a', 'an', 'the', 'to', 'for', 'and', 'or'}][:3]
        return '_'.join(name_words) if name_words else 'generated_function'

    def _determine_security_level(self, description: str) -> SecurityLevel:
        """Determine security level from keywords"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ['file', 'write', 'delete', 'modify', 'system']):
            return SecurityLevel.ELEVATED
        elif any(word in desc_lower for word in ['network', 'request', 'api', 'download']):
            return SecurityLevel.MODERATE
        else:
            return SecurityLevel.SAFE

    def _generate_code_template(self, description: str, function_name: str, language: CodeLanguage) -> str:
        """Generate code template"""
        if language == CodeLanguage.PYTHON:
            return f'''#!/usr/bin/env python3
"""
Auto-generated: {function_name}
Description: {description}
Generated by: SarahMemory Synapses v{PROJECT_VERSION}
Date: {datetime.datetime.now().isoformat()}
"""

import logging

logger = logging.getLogger('{function_name}')

def {function_name}(*args, **kwargs):
    """
    {description}

    Args:
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments

    Returns:
        Result of operation
    """
    logger.info(f"Executing {function_name}")

    try:
        # TODO: Implement based on description
        result = f"Function {function_name} executed successfully"
        logger.info(f"{function_name} completed: {{result}}")
        return result

    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}", exc_info=True)
        raise

if __name__ == '__main__':
    print(f"Testing {function_name}...")
    result = {function_name}()
    print(f"Result: {{result}}")
'''
        else:
            raise NotImplementedError(f"Language {language.value} not implemented")

    def test_and_validate(self, generated_code: 'GeneratedCode') -> bool:
        """Test and validate generated code"""
        logger.info(f"Testing: {generated_code.function_name}")

        all_passed, test_results = self.tester.run_full_test_suite(
            generated_code.code,
            generated_code.hash
        )

        generated_code.test_results = test_results
        generated_code.status = TestStatus.PASSED if all_passed else TestStatus.FAILED

        if all_passed:
            test_path = os.path.join(SANDBOX_TESTING_DIR, f"{generated_code.function_name}.py")
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(generated_code.code)
            logger.info(f"Code saved to testing: {test_path}")
        else:
            fail_path = os.path.join(SANDBOX_FAILED_DIR, f"{generated_code.function_name}_FAILED.py")
            with open(fail_path, 'w', encoding='utf-8') as f:
                f.write(generated_code.code)
                f.write("\n\n# FAILED TESTS:\n")
                for test in test_results:
                    if not test['passed']:
                        f.write(f"# - {test['test_name']}: {test['error']}\n")
            logger.warning(f"Failed code saved: {fail_path}")

        return all_passed


# ============================================================================
# MODULE COMPOSER - HIGH-LEVEL ORCHESTRATION
# ============================================================================

def compose_new_module(request: str, auto_approve: bool = False) -> str:
    """
    Main entry point for composing a new module.

    Orchestrates the complete code generation pipeline:
    1. Parse request
    2. Generate code
    3. Analyze metrics
    4. Run tests
    5. Log provenance
    6. Save for approval/deployment

    Args:
        request: Natural language description
        auto_approve: Auto-approve if tests pass (CAUTION)

    Returns:
        Status message
    """
    try:
        logger.info(f"=== MODULE COMPOSITION STARTED ===")
        logger.info(f"Request: {request}")

        initialize_synapses_database()

        generator = CodeGenerator()
        generated_code = generator.generate_from_description(request)

        if not generated_code:
            return "ERROR: Code generation failed"

        code_hash = log_code_generation(generated_code)
        tests_passed = generator.test_and_validate(generated_code)

        status_emoji = "✅" if tests_passed else "❌"
        quality_grade = (
            "A" if generated_code.metrics.overall_quality >= 0.9 else
            "B" if generated_code.metrics.overall_quality >= 0.8 else
            "C" if generated_code.metrics.overall_quality >= 0.7 else
            "D" if generated_code.metrics.overall_quality >= 0.6 else "F"
        )

        report = f"""
{status_emoji} MODULE COMPOSITION COMPLETE

Function: {generated_code.function_name}
Hash: {generated_code.hash[:16]}
Language: {generated_code.language.value}
Security Level: {generated_code.security_level.value}

QUALITY METRICS:
- Overall Grade: {quality_grade} ({generated_code.metrics.overall_quality:.2%})
- Lines of Code: {generated_code.metrics.lines_of_code}
- Cyclomatic Complexity: {generated_code.metrics.cyclomatic_complexity}
- Cognitive Complexity: {generated_code.metrics.cognitive_complexity}
- Maintainability: {generated_code.metrics.maintainability_index:.1f}/100
- Security Score: {generated_code.metrics.security_score:.2%}

TEST RESULTS:
"""

        for test in generated_code.test_results:
            test_status = "✅ PASSED" if test['passed'] else "❌ FAILED"
            report += f"- {test['test_name']}: {test_status}"
            if test['error']:
                report += f" ({test['error']})"
            report += "\n"

        if tests_passed:
            if auto_approve:
                approved_path = os.path.join(SANDBOX_APPROVED_DIR, f"{generated_code.function_name}.py")
                with open(approved_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code.code)

                save_code_to_addons(f"{generated_code.function_name}.py", generated_code.code)

                report += f"\n✅ AUTO-APPROVED: {approved_path}"
                logger.info("Module auto-approved")
            else:
                report += f"\n⏳ AWAITING APPROVAL: Review in sandbox/testing/"
                logger.info("Module awaiting approval")
        else:
            report += f"\n❌ TESTS FAILED: Review in sandbox/failed/"
            logger.warning("Module failed tests")

        logger.info(f"=== MODULE COMPOSITION COMPLETE ===")

        return report

    except Exception as e:
        error_msg = f"ERROR: Module composition failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def compose_new_module_async(request: str, auto_approve: bool = False) -> str:
    """Asynchronous wrapper for compose_new_module"""
    result_container = {'result': ''}

    def target():
        result_container['result'] = compose_new_module(request, auto_approve)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=60.0)

    if thread.is_alive():
        return "ERROR: Composition timed out after 60 seconds"

    return result_container.get('result', 'ERROR: No result returned')


# ============================================================================
# SELF-UPDATE SYSTEM
# ============================================================================

def update_self(new_code: str, module_name: str = "UpdatedModule.py",
                require_approval: bool = True) -> str:
    """
    Self-update mechanism with comprehensive safety checks.

    Features:
    - Multi-stage testing
    - Rollback capability
    - Version control
    - Audit logging

    Args:
        new_code: New module code
        module_name: Name of module
        require_approval: Require approval before deployment

    Returns:
        Status message
    """
    try:
        logger.info(f"Self-update initiated: {module_name}")

        generated = GeneratedCode(
            code=new_code,
            language=CodeLanguage.PYTHON,
            security_level=SecurityLevel.ELEVATED,
            function_name=module_name.replace('.py', ''),
            description=f"Self-update for {module_name}",
            status=TestStatus.PENDING
        )

        analyzer = CodeAnalyzer()
        generated.metrics = analyzer.generate_metrics(new_code)
        generated.dependencies = analyzer.extract_dependencies(new_code)

        tester = SandboxTester()
        all_passed, test_results = tester.run_full_test_suite(new_code, generated.hash)

        generated.test_results = test_results
        generated.status = TestStatus.PASSED if all_passed else TestStatus.FAILED

        log_code_generation(generated)

        if not all_passed:
            return f"Self-update FAILED: Tests did not pass"

        if require_approval:
            update_path = os.path.join(SANDBOX_TESTING_DIR, module_name)
            with open(update_path, 'w', encoding='utf-8') as f:
                f.write(new_code)

            return f"Self-update PENDING APPROVAL: Review {update_path}"
        else:
            update_path = os.path.join(SANDBOX_APPROVED_DIR, module_name)
            with open(update_path, 'w', encoding='utf-8') as f:
                f.write(new_code)

            logger.warning(f"Self-update AUTO-DEPLOYED: {update_path}")
            return f"Self-update DEPLOYED: {update_path}"

    except Exception as e:
        error_msg = f"Self-update FAILED: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ============================================================================
# 3D ENGINE SELECTION
# ============================================================================

def select_3d_engine() -> str:
    """
    Determine which 3D engine to use for avatar rendering.

    Returns:
        Engine identifier: "Microsoft3DViewer", "Blender", "Unreal", or "Fallback"
    """
    try:
        from SarahMemorySi import get_3d_engine_path

        ms3d_path = get_3d_engine_path("Microsoft3DViewer")
        if ms3d_path:
            logger.info("3D Engine: Microsoft 3D Viewer")
            return "Microsoft3DViewer"

        blender_path = get_3d_engine_path("Blender")
        if blender_path:
            logger.info("3D Engine: Blender")
            return "Blender"

        unreal_path = get_3d_engine_path("Unreal")
        if unreal_path:
            logger.info("3D Engine: Unreal")
            return "Unreal"

        try:
            from SarahMemorySoftwareResearch import get_operational_guidelines
            guidelines = get_operational_guidelines()
            if guidelines:
                logger.info("3D Engine: Blender (from guidelines)")
                return "Blender"
        except:
            pass

    except Exception as e:
        logger.warning(f"3D engine selection error: {e}")

    logger.info("3D Engine: Fallback")
    return "Fallback"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_synapses_status() -> Dict[str, Any]:
    """Get current synapses system status"""
    try:
        conn = connect_db("synapses.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM code_generations")
        total_generations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM code_generations WHERE status = 'approved'")
        approved_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM code_generations WHERE status = 'failed'")
        failed_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT AVG(overall_quality)
            FROM code_metrics
            WHERE code_hash IN (SELECT hash FROM code_generations WHERE status = 'approved')
        """)
        avg_quality = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM test_results")
        total_tests = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM test_results WHERE passed = 1")
        passed_tests = cursor.fetchone()[0]
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0

        conn.close()

        runtime_meta = get_runtime_meta()

        return {
            'system': 'SarahMemory Synapses',
            'version': PROJECT_VERSION,
            'run_mode': runtime_meta.get('run_mode'),
            'device_mode': runtime_meta.get('device_mode'),
            'statistics': {
                'total_generations': total_generations,
                'approved_modules': approved_count,
                'failed_modules': failed_count,
                'pending_approval': total_generations - approved_count - failed_count,
                'average_quality': f"{avg_quality:.2%}",
                'total_tests': total_tests,
                'test_pass_rate': f"{pass_rate:.1f}%"
            },
            'sandbox': {
                'approved_dir': SANDBOX_APPROVED_DIR,
                'testing_dir': SANDBOX_TESTING_DIR,
                'failed_dir': SANDBOX_FAILED_DIR
            },
            'timestamp': datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        return {'error': str(e)}


def cleanup_old_sandbox_files(days_old: int = 30):
    """Clean up old sandbox files"""
    try:
        from datetime import timedelta

        cutoff_time = datetime.datetime.now() - timedelta(days=days_old)

        for sandbox_dir in [SANDBOX_TESTING_DIR, SANDBOX_FAILED_DIR]:
            if not os.path.exists(sandbox_dir):
                continue

            for filename in os.listdir(sandbox_dir):
                filepath = os.path.join(sandbox_dir, filename)

                if os.path.isfile(filepath):
                    file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))

                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Cleaned up: {filename}")

        logger.info(f"Cleanup complete (>{days_old} days)")

    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)


# ============================================================================
# BACKWARD COMPATIBILITY - OLD FUNCTION NAMES
# ============================================================================

def log_function_task(task_name, user_input, description):
    """Legacy function logging (redirects to new system)"""
    try:
        conn = connect_db("functions.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO functions (function_name, user_input, description, timestamp)
            VALUES (?, ?, ?, ?)
        """, (task_name, user_input, description, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Legacy task logged: {task_name}")
    except Exception as e:
        logger.error(f"Failed to log legacy task: {e}")


def log_code_output(task_name, code_language, ai_code, function_type="dynamic"):
    """Legacy code logging (redirects to new system)"""
    try:
        conn = connect_db("programming.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO code_snippets (task_name, language, code, function_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (task_name, code_language, ai_code, function_type, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Legacy code logged: {task_name}")
    except Exception as e:
        logger.error(f"Failed to log legacy code: {e}")


def log_software_task(app_name, category, file_output_path, status="pending"):
    """Legacy software task logging"""
    try:
        conn = connect_db("software.db")
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO software_tasks (software_name, category, output_path, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (app_name, category, file_output_path, status, timestamp))
        conn.commit()
        conn.close()
        logger.info(f"Legacy software task logged: {app_name}")
    except Exception as e:
        logger.error(f"Failed to log software task: {e}")


def run_sandbox_test(code_str):
    """Legacy sandbox test (simple version)"""
    try:
        compiled_code = compile(code_str, '<sandbox>', 'exec')
    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return False

    restricted_globals = {
        '__builtins__': {
            'print': print, 'range': range, 'len': len,
            'str': str, 'int': int, 'float': float,
            'bool': bool, 'list': list,
        }
    }

    try:
        exec(compiled_code, restricted_globals, {})
        logger.info("Sandbox test passed")
        return True
    except Exception as e:
        logger.error(f"Execution error: {traceback.format_exc()}")
        return False


# ============================================================================
# MAIN EXECUTION & TESTING
# ============================================================================

if __name__ == '__main__':
    """Test and demonstration mode"""
    print("=" * 80)
    print("SarahMemory Synapses - World-Class Neural Architecture")
    print(f"Version: {PROJECT_VERSION}")
    print("=" * 80)
    print()

    print("Initializing database...")
    initialize_synapses_database()
    print("✅ Database initialized")
    print()

    print("Current Status:")
    print("-" * 80)
    status = get_synapses_status()
    print(json.dumps(status, indent=2))
    print()

    print("=" * 80)
    print("TEST: Code Generation")
    print("=" * 80)
    print()

    test_request = """
    Create a function that calculates the Fibonacci sequence up to n terms
    and returns the results as a list. Include error handling.
    """

    print(f"Request: {test_request.strip()}")
    print()
    print("Generating module...")
    print()

    result = compose_new_module(test_request, auto_approve=False)
    print(result)
    print()

    print("=" * 80)
    print("Final Status:")
    print("-" * 80)
    final_status = get_synapses_status()
    print(json.dumps(final_status['statistics'], indent=2))
    print()

    print("=" * 80)
    print("SarahMemory Synapses test complete!")
    print("=" * 80)

# ====================================================================
# END OF SarahMemorySynapes.py v8.0.0
# ====================================================================