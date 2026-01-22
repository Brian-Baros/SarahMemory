"""--==The SarahMemory Project==--
File: SarahMemoryOptimization.py
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

SYSTEM OPTIMIZATION MODULE v8.0.0
==============================================
This module has standards with intelligent resource
management, predictive optimization, and automated performance tuning while
maintaining 100% backward compatibility.

KEY ENHANCEMENTS:
-----------------
1. INTELLIGENT RESOURCE MONITORING
   - Real-time CPU, memory, disk, network tracking
   - Adaptive threshold management
   - Trend analysis and prediction
   - Resource bottleneck detection
   - Historical performance metrics

2. AUTOMATED OPTIMIZATION
   - Proactive resource management
   - Intelligent task scheduling
   - Cache optimization
   - Memory cleanup automation
   - Process priority management

3. IDLE-TIME LEARNING
   - Background enrichment tasks
   - Context deep learning
   - Behavior pattern analysis
   - Automated research and caching
   - Database maintenance

4. PERFORMANCE ANALYTICS
   - Comprehensive metrics collection
   - Performance trend visualization
   - Resource utilization reports
   - Optimization effectiveness tracking
   - Predictive alerts

5. CROSS-PLATFORM COMPATIBILITY
   - Windows optimization strategies
   - Linux resource management
   - macOS performance tuning
   - Headless server optimization
   - Platform-specific recommendations

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- monitor_system_resources()
- optimize_system()
- start_optimization_monitor(interval=10)
- run_idle_optimization_tasks()
- log_optimization_event(event, details)

New functions added (non-breaking):
- get_optimization_metrics()
- predict_resource_usage()
- intelligent_threshold_adjustment()
- get_optimization_recommendations()
- schedule_optimization_tasks()
- analyze_performance_trends()

INTEGRATION POINTS:
-------------------
- SarahMemoryHi.py: System metrics collection
- SarahMemoryDatabase.py: Metrics persistence
- SarahMemoryDL.py: Deep learning integration
- SarahMemoryResearch.py: Idle-time research
- SarahMemoryGlobals.py: Configuration management
- SarahMemoryDiagnostics.py: Health monitoring

===============================================================================
"""

import logging
import psutil
import time
import os
import json
import subprocess
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Core imports
import SarahMemoryGlobals as config
from SarahMemoryGlobals import run_async, DATASETS_DIR

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryOptimization')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# RESOURCE THRESHOLDS - v8.0 Enhanced
# =============================================================================
CPU_THRESHOLD = int(getattr(config, 'CPU_OPTIMIZATION_THRESHOLD', 80))
MEMORY_THRESHOLD = int(getattr(config, 'MEMORY_OPTIMIZATION_THRESHOLD', 80))
DISK_THRESHOLD = int(getattr(config, 'DISK_OPTIMIZATION_THRESHOLD', 90))
NETWORK_THRESHOLD = float(getattr(config, 'NETWORK_OPTIMIZATION_THRESHOLD_MB', 100.0))

# Adaptive thresholds
_adaptive_thresholds = {
    'cpu': CPU_THRESHOLD,
    'memory': MEMORY_THRESHOLD,
    'disk': DISK_THRESHOLD,
    'network': NETWORK_THRESHOLD
}

# Performance history for trend analysis
_performance_history = []
_max_history_size = 100


# =============================================================================
# SARAH MEMORY PARTITIONING (Cognitive-Role Partitions) - v8.0.0
# =============================================================================
# NOTE:
# - These are logical in-process partitions (bytearrays) used as bounded scratch arenas.
# - They are NOT OS-level reserved physical RAM.
# - CognitiveServices can publish structured records into these partitions.
import ctypes

# Cognitive role labels (aligned to owner intent)
COG_PART_MONITOR = "monitor"   # Partition 1: "understanding how it is running now"
COG_PART_IMPROVE = "improve"   # Partition 2: "how can I improve myself"
COG_PART_TEST    = "test"      # Partition 3: "sandbox testing"
COG_PART_DEPLOY  = "deploy"    # Partition 4: "validated - implement into program"

# Internal state (module-level, so it’s shared across the process)
_SARAH_MEM_POOLS: Dict[str, Any] = {
    "runtime": {"total_mb": 0, "partitions": [], "bytes_each": 0},
    "sandbox": {"total_mb": 0, "partitions": [], "bytes_each": 0},
    "refresh": {"enabled": False, "interval_seconds": 0, "thread": None, "stop_flag": False},
    # Cognitive partition routing table (logical roles -> (pool, index))
    "cognitive": {
        "roles": {
            COG_PART_MONITOR: ("runtime", 0),
            COG_PART_IMPROVE: ("runtime", 1),
            COG_PART_TEST:    ("sandbox", 0),  # prefers sandbox if enabled
            COG_PART_DEPLOY:  ("runtime", 3),
        },
        # last written meta per role (for quick inspection / diagnostics)
        "last_meta": {},
    },
}

def _ctypes_zero_buffer(buf: bytearray) -> None:
    """Zero a bytearray in-place without allocating a massive temporary bytes object."""
    try:
        ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
        ctypes.memset(ctypes.addressof(ptr), 0, len(buf))
    except Exception as e:
        logger.warning(f"[v8.0] Memory zero failed: {e}")

def _partition_get_buffer(role: str) -> Optional[bytearray]:
    """
    Resolve a cognitive role to its backing bytearray partition.
    Returns None if partitions are not initialized or role is unknown.
    """
    try:
        roles = (_SARAH_MEM_POOLS.get("cognitive") or {}).get("roles") or {}
        pool_name, idx = roles.get(str(role), (None, None))
        if not pool_name:
            return None
        pool = _SARAH_MEM_POOLS.get(pool_name) or {}
        parts = pool.get("partitions") or []
        if not isinstance(idx, int) or idx < 0 or idx >= len(parts):
            return None
        return parts[idx]
    except Exception:
        return None

def _partition_write(role: str, payload: Any, *, meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    Writes a JSON payload into the specified cognitive role partition using:
      [4 bytes length little-endian][payload bytes UTF-8][zero-fill remainder]
    Returns True if written; False otherwise.
    """
    buf = _partition_get_buffer(role)
    if buf is None:
        return False

    try:
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        else:
            data = str(payload).encode("utf-8")
    except Exception:
        data = b"{}"

    cap = len(buf)
    if cap < 8:
        return False

    # Keep within buffer: reserve 4 bytes header
    max_payload = cap - 4
    if len(data) > max_payload:
        data = data[:max_payload]

    # zero first, then write header + data
    _ctypes_zero_buffer(buf)
    buf[0:4] = int(len(data)).to_bytes(4, byteorder="little", signed=False)
    buf[4:4 + len(data)] = data

    try:
        last_meta = (_SARAH_MEM_POOLS.get("cognitive") or {}).get("last_meta")
        if isinstance(last_meta, dict):
            last_meta[str(role)] = {
                "ts": datetime.now().isoformat(),
                "bytes": int(len(data)),
                "cap_bytes": int(cap),
                "meta": meta or {},
            }
    except Exception:
        pass

    return True

def _partition_read(role: str) -> Optional[Dict[str, Any]]:
    """
    Read back the latest JSON record from a role partition.
    Returns dict with 'payload' (str) + header metadata; None if not available.
    """
    buf = _partition_get_buffer(role)
    if buf is None or len(buf) < 8:
        return None
    try:
        n = int.from_bytes(buf[0:4], byteorder="little", signed=False)
        if n <= 0 or n > (len(buf) - 4):
            return None
        raw = bytes(buf[4:4+n])
        payload = raw.decode("utf-8", errors="replace")
        meta = ((_SARAH_MEM_POOLS.get("cognitive") or {}).get("last_meta") or {}).get(str(role)) or {}
        return {"role": str(role), "payload": payload, "meta": meta}
    except Exception:
        return None

def memory_partition() -> Dict[str, Any]:
    """
    Allocate logical in-process memory pools for:
      1) runtime operations
      2) sandbox (self-update/self-test) operations (optional)

    Cognitive role routing:
      monitor -> runtime[0]
      improve -> runtime[1]
      test    -> sandbox[0] if available else runtime[2]
      deploy  -> runtime[3] (or last partition if <4)

    Configuration source of truth:
      - SarahMemoryGlobals.py (preferred defaults)
      - Optional env overrides for cloud/headless safety
    """
    total_mb = int(os.getenv("SARAH_TOTAL_MEMORY_MB", getattr(config, "SARAH_TOTAL_MEMORY_MB", 0) or 0))
    partitions = int(os.getenv("SARAH_MEMORY_PARTITIONS", getattr(config, "SARAH_MEMORY_PARTITIONS", 1) or 1))
    refresh_minutes = int(os.getenv("SARAH_MEMORY_REFRESH_MINUTES", getattr(config, "SARAH_MEMORY_REFRESH_MINUTES", 0) or 0))
    sandbox_enabled = str(os.getenv("SARAH_MEMORY_SANDBOX_ENABLED", getattr(config, "SARAH_MEMORY_SANDBOX_ENABLED", True))).strip().lower() in ("1","true","yes","on")

    # Optional: allow sandbox pool size to differ without adding a new Globals constant
    sandbox_mb = int(os.getenv("SARAH_SANDBOX_MEMORY_MB", total_mb))

    # ---- Guardrails for web/cloud execution ----
    device_mode = getattr(config, "DEVICE_MODE", "").lower()
    run_mode = getattr(config, "RUN_MODE", "").lower()
    force_alloc = os.getenv("SARAH_FORCE_MEMORY_ALLOC", "false").strip().lower() in ("1","true","yes","on")

    if not force_alloc and (run_mode == "cloud" or "public_web" in device_mode or "mobile_web" in device_mode):
        logger.info("[v8.0] Memory partitioning skipped in cloud/web mode (set SARAH_FORCE_MEMORY_ALLOC=true to override).")
        return {
            "ok": True,
            "skipped": True,
            "reason": "cloud/web mode",
            "runtime": {"total_mb": 0, "partitions": 0},
            "sandbox": {"enabled": sandbox_enabled, "total_mb": 0, "partitions": 0},
            "refresh_minutes": refresh_minutes,
        }

    if total_mb <= 0 or partitions <= 0:
        return {
            "ok": False,
            "error": "Invalid SARAH_TOTAL_MEMORY_MB or SARAH_MEMORY_PARTITIONS",
            "runtime": {"total_mb": total_mb, "partitions": partitions},
            "sandbox": {"enabled": sandbox_enabled, "total_mb": sandbox_mb, "partitions": partitions},
            "refresh_minutes": refresh_minutes,
        }

    bytes_each_runtime = (total_mb * 1024 * 1024) // partitions
    bytes_each_sandbox = (sandbox_mb * 1024 * 1024) // partitions if sandbox_enabled else 0

    runtime_parts: List[bytearray] = []
    try:
        for _ in range(partitions):
            runtime_parts.append(bytearray(bytes_each_runtime))
    except MemoryError:
        logger.error("[v8.0] Runtime memory allocation failed (MemoryError).")
        return {"ok": False, "error": "Runtime memory allocation failed (MemoryError)"}
    except Exception as e:
        logger.error(f"[v8.0] Runtime memory allocation failed: {e}")
        return {"ok": False, "error": f"Runtime memory allocation failed: {e}"}

    sandbox_parts: List[bytearray] = []
    if sandbox_enabled and bytes_each_sandbox > 0:
        try:
            for _ in range(partitions):
                sandbox_parts.append(bytearray(bytes_each_sandbox))
        except MemoryError:
            logger.warning("[v8.0] Sandbox allocation failed (MemoryError). Disabling sandbox pool.")
            sandbox_enabled = False
            sandbox_parts = []
        except Exception as e:
            logger.warning(f"[v8.0] Sandbox allocation failed: {e}. Disabling sandbox pool.")
            sandbox_enabled = False
            sandbox_parts = []

    _SARAH_MEM_POOLS["runtime"] = {
        "total_mb": total_mb,
        "partitions": runtime_parts,
        "bytes_each": bytes_each_runtime,
    }
    _SARAH_MEM_POOLS["sandbox"] = {
        "total_mb": sandbox_mb if sandbox_enabled else 0,
        "partitions": sandbox_parts,
        "bytes_each": bytes_each_sandbox if sandbox_enabled else 0,
    }

    # ---- Cognitive role routing normalization ----
    # If runtime partitions < 4, map deploy to last available partition.
    # If sandbox disabled, map test to runtime[2] if available else runtime[last].
    try:
        roles = (_SARAH_MEM_POOLS.get("cognitive") or {}).get("roles") or {}
        # deploy
        deploy_idx = 3 if len(runtime_parts) > 3 else max(0, len(runtime_parts) - 1)
        roles[COG_PART_DEPLOY] = ("runtime", deploy_idx)
        # test
        if sandbox_enabled and len(sandbox_parts) > 0:
            roles[COG_PART_TEST] = ("sandbox", 0)
        else:
            test_idx = 2 if len(runtime_parts) > 2 else max(0, len(runtime_parts) - 1)
            roles[COG_PART_TEST] = ("runtime", test_idx)
        # monitor/improve (best-effort)
        roles[COG_PART_MONITOR] = ("runtime", 0)
        roles[COG_PART_IMPROVE] = ("runtime", 1 if len(runtime_parts) > 1 else 0)
        (_SARAH_MEM_POOLS.get("cognitive") or {})["roles"] = roles
    except Exception:
        pass

    if refresh_minutes > 0:
        _SARAH_MEM_POOLS["refresh"]["enabled"] = True
        _SARAH_MEM_POOLS["refresh"]["interval_seconds"] = max(30, refresh_minutes * 60)
        _start_memory_refresh_loop()

    log_optimization_event(
        "Memory Partitioned",
        f"runtime={total_mb}MB/{partitions} parts, sandbox={'on' if sandbox_enabled else 'off'} ({sandbox_mb}MB), refresh={refresh_minutes}m"
    )

    return {
        "ok": True,
        "skipped": False,
        "runtime": {"total_mb": total_mb, "partitions": partitions, "bytes_each": bytes_each_runtime},
        "sandbox": {"enabled": sandbox_enabled, "total_mb": sandbox_mb if sandbox_enabled else 0, "partitions": partitions if sandbox_enabled else 0, "bytes_each": bytes_each_sandbox if sandbox_enabled else 0},
        "refresh_minutes": refresh_minutes,
        "cognitive_roles": (_SARAH_MEM_POOLS.get("cognitive") or {}).get("roles", {}),
    }

def _start_memory_refresh_loop() -> None:
    """Starts a single refresh thread (idempotent)."""
    try:
        t = _SARAH_MEM_POOLS["refresh"].get("thread")
        if t and getattr(t, "is_alive", lambda: False)():
            return

        _SARAH_MEM_POOLS["refresh"]["stop_flag"] = False
        interval = int(_SARAH_MEM_POOLS["refresh"].get("interval_seconds", 0) or 0)
        if interval <= 0:
            return

        def _loop():
            while not _SARAH_MEM_POOLS["refresh"]["stop_flag"]:
                time.sleep(interval)
                try:
                    for buf in _SARAH_MEM_POOLS["runtime"].get("partitions", []):
                        _ctypes_zero_buffer(buf)
                    for buf in _SARAH_MEM_POOLS["sandbox"].get("partitions", []):
                        _ctypes_zero_buffer(buf)

                    log_optimization_event("Memory Refreshed", f"interval={interval}s")
                except Exception as e:
                    logger.warning(f"[v8.0] Memory refresh loop error: {e}")

        thr = threading.Thread(target=_loop, daemon=True, name="SarahMemoryRefresh")
        _SARAH_MEM_POOLS["refresh"]["thread"] = thr
        thr.start()

    except Exception as e:
        logger.warning(f"[v8.0] Failed to start memory refresh loop: {e}")

def stop_memory_refresh_loop() -> None:
    """Stops refresh loop cleanly."""
    _SARAH_MEM_POOLS["refresh"]["stop_flag"] = True

# Public helpers for CognitiveServices (lazy, safe)
def publish_cognitive_record(role: str, record: Dict[str, Any]) -> bool:
    """
    Publish a small structured record into the appropriate cognitive partition.
    Intended caller: SarahMemoryCognitiveServices (governor).
    """
    meta = {"source": "cognitive_governor"}
    return _partition_write(role, record, meta=meta)

def read_cognitive_record(role: str) -> Optional[Dict[str, Any]]:
    """Read back the last record for a cognitive role partition."""
    return _partition_read(role)


# =============================================================================
# DATABASE LOGGING - v8.0 Enhanced
# =============================================================================
def log_optimization_event(event: str, details: str) -> None:
    """
    Log an optimization-related event to the database.
    v8.0: Enhanced with better error handling and metadata.

    Args:
        event: Event name/type
        details: Event details/description
    """
    try:
        db_path = os.path.join(DATASETS_DIR, "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT,
                version TEXT DEFAULT '8.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO optimization_events (timestamp, event, details, version) VALUES (?, ?, ?, ?)",
            (timestamp, event, details, "8.0.0")
        )

        conn.commit()
        conn.close()

        logger.debug(f"[v8.0] Logged optimization event: {event}")

    except Exception as e:
        logger.warning(f"[v8.0] Failed to log optimization event: {e}")

# =============================================================================
# RESOURCE MONITORING - v8.0 Enhanced
# =============================================================================
def monitor_system_resources() -> Dict[str, Union[float, str]]:
    """
    Monitor comprehensive system resource usage.
    v8.0: Enhanced with network monitoring and error handling.

    Returns:
        Dictionary containing resource usage metrics
    """
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Disk usage
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        disk_usage = disk.percent

        # Network usage
        net_io = psutil.net_io_counters()
        network_usage_mb = round((net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024), 2)

        # Process count
        process_count = len(psutil.pids())

        # Build resource usage dict
        resource_usage = {
            "cpu": round(cpu_usage, 2),
            "memory": round(memory_usage, 2),
            "disk": round(disk_usage, 2),
            "network_mb": network_usage_mb,
            "process_count": process_count,
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0"
        }

        # Add to performance history
        _performance_history.append(resource_usage.copy())
        if len(_performance_history) > _max_history_size:
            _performance_history.pop(0)

        logger.debug(f"[v8.0] Resources: CPU={cpu_usage}% MEM={memory_usage}% DISK={disk_usage}%")
        log_optimization_event("Monitor Resources", f"CPU: {cpu_usage}%, MEM: {memory_usage}%, DISK: {disk_usage}%")

        return resource_usage

    except Exception as e:
        error_msg = f"Error monitoring resources: {e}"
        logger.error(f"[v8.0] {error_msg}")
        log_optimization_event("Monitor Resources Error", error_msg)
        return {"error": str(e), "version": "8.0.0"}

# =============================================================================
# SYSTEM OPTIMIZATION - v8.0 Enhanced
# =============================================================================
def optimize_system() -> str:
    """
    Optimize system performance based on monitored resources.
    v8.0: Enhanced with intelligent recommendations and actions.

    Returns:
        Status message with optimization results
    """
    try:
        usage = monitor_system_resources()

        if "error" in usage:
            return f"Optimization failed: {usage['error']}"

        actions_taken = []
        recommendations = []

        # Get adaptive thresholds
        cpu_thresh = _adaptive_thresholds.get('cpu', CPU_THRESHOLD)
        mem_thresh = _adaptive_thresholds.get('memory', MEMORY_THRESHOLD)
        disk_thresh = _adaptive_thresholds.get('disk', DISK_THRESHOLD)

        # CPU optimization
        if usage.get("cpu", 0) > cpu_thresh:
            recommendations.append(f"High CPU usage ({usage['cpu']}%) detected.")
            recommendations.append("→ Consider closing CPU-intensive applications")
            recommendations.append("→ Check for background processes")
            recommendations.append("→ Review startup programs")
            logger.warning(f"[v8.0] High CPU usage: {usage['cpu']}%")
            log_optimization_event("CPU Optimization Alert", f"CPU usage at {usage['cpu']}%")

        # Memory optimization
        if usage.get("memory", 0) > mem_thresh:
            recommendations.append(f"High memory usage ({usage['memory']}%) detected.")
            recommendations.append("→ Consider closing unused applications")
            recommendations.append("→ Clear browser cache and temporary files")
            recommendations.append("→ Increase virtual memory/swap space")

            # Attempt automatic cleanup (if safe mode allows)
            if getattr(config, 'AUTO_OPTIMIZATION_ENABLED', False):
                try:
                    # Trigger garbage collection
                    import gc
                    gc.collect()
                    actions_taken.append("Triggered Python garbage collection")
                    logger.info("[v8.0] Automatic memory cleanup performed")
                except Exception as e:
                    logger.debug(f"[v8.0] Auto cleanup failed: {e}")

            logger.warning(f"[v8.0] High memory usage: {usage['memory']}%")
            log_optimization_event("Memory Optimization Alert", f"Memory usage at {usage['memory']}%")

        # Disk optimization
        if usage.get("disk", 0) > disk_thresh:
            recommendations.append(f"High disk usage ({usage['disk']}%) detected.")
            recommendations.append("→ Run disk cleanup utility")
            recommendations.append("→ Remove temporary files")
            recommendations.append("→ Uninstall unused programs")
            recommendations.append("→ Move files to external storage")
            logger.warning(f"[v8.0] High disk usage: {usage['disk']}%")
            log_optimization_event("Disk Optimization Alert", f"Disk usage at {usage['disk']}%")

        # Build status message
        if not recommendations and not actions_taken:
            status = "✓ System resources are optimal."
            logger.info("[v8.0] System resources optimal")
            log_optimization_event("Optimize System", "All resources within normal parameters")
        else:
            status_parts = []

            if actions_taken:
                status_parts.append("Actions Taken:")
                status_parts.extend(f"  • {action}" for action in actions_taken)

            if recommendations:
                if status_parts:
                    status_parts.append("")
                status_parts.append("Recommendations:")
                status_parts.extend(f"  • {rec}" for rec in recommendations)

            status = "\n".join(status_parts)
            logger.info(f"[v8.0] Optimization: {len(actions_taken)} actions, {len(recommendations)} recommendations")
            log_optimization_event("Optimize System", f"Actions: {len(actions_taken)}, Recommendations: {len(recommendations)}")

        return status

    except Exception as e:
        error_msg = f"Error optimizing system: {e}"
        logger.error(f"[v8.0] {error_msg}")
        log_optimization_event("Optimize System Error", error_msg)
        return error_msg

# =============================================================================
# OPTIMIZATION MONITOR - Backward Compatible
# =============================================================================
def start_optimization_monitor(interval: int = 10) -> None:
    """
    Start a background loop that runs optimization checks.
    v8.0: Enhanced with better thread management.

    Args:
        interval: Seconds between optimization checks
    """
    def monitor_loop():
        logger.info(f"[v8.0] Starting optimization monitor (interval={interval}s)")
        try:
            while True:
                optimize_system()
                time.sleep(interval)
        except Exception as e:
            logger.error(f"[v8.0] Monitor loop error: {e}")

    try:
        run_async(monitor_loop)
        logger.info("[v8.0] Optimization monitor started")
    except Exception as e:
        logger.error(f"[v8.0] Failed to start optimization monitor: {e}")

# =============================================================================
# IDLE OPTIMIZATION TASKS - v8.0 Enhanced
# =============================================================================
def run_idle_optimization_tasks() -> None:
    """
    Execute optimization and learning tasks during idle time.
    v8.0: Enhanced with better error handling and task management.
    """
    logger.info("[v8.0] Starting idle optimization and enrichment cycle...")

    tasks_completed = []
    tasks_failed = []

    try:
        # Memory auto-correction
        try:
            from SarahMemorySystemLearn import memory_autocorrect
            memory_autocorrect()
            tasks_completed.append("memory_autocorrect")
            logger.info("[v8.0] ✓ Memory auto-correction completed")
        except Exception as e:
            tasks_failed.append(f"memory_autocorrect: {e}")
            logger.warning(f"[v8.0] Memory auto-correction failed: {e}")

        # Behavior analysis
        try:
            from SarahMemoryDL import analyze_user_behavior
            behavior = analyze_user_behavior()
            tasks_completed.append("analyze_user_behavior")
            logger.info(f"[v8.0] ✓ Behavior analysis: {behavior}")
        except Exception as e:
            tasks_failed.append(f"analyze_user_behavior: {e}")
            logger.warning(f"[v8.0] Behavior analysis failed: {e}")

        # Context deep learning
        try:
            from SarahMemoryDL import deep_learn_user_context
            topics = deep_learn_user_context()
            tasks_completed.append("deep_learn_user_context")
            logger.info(f"[v8.0] ✓ Deep learning: {len(topics)} topics identified")

            # Research top topics
            try:
                from SarahMemoryResearch import research_topic
                for topic in topics[:3]:
                    research_topic(topic)
                tasks_completed.append("research_top_topics")
                logger.info("[v8.0] ✓ Topic research completed")
            except Exception as e:
                tasks_failed.append(f"research_top_topics: {e}")
                logger.warning(f"[v8.0] Topic research failed: {e}")

        except Exception as e:
            tasks_failed.append(f"deep_learn_user_context: {e}")
            logger.warning(f"[v8.0] Deep learning failed: {e}")

        # Database maintenance
        try:
            from SarahMemoryDatabase import optimize_databases
            optimize_databases()
            tasks_completed.append("optimize_databases")
            logger.info("[v8.0] ✓ Database optimization completed")
        except Exception as e:
            tasks_failed.append(f"optimize_databases: {e}")
            logger.warning(f"[v8.0] Database optimization failed: {e}")

        # Log results
        log_optimization_event(
            "Idle Optimization Completed",
            f"Completed: {tasks_completed}, Failed: {tasks_failed}"
        )

        logger.info(f"[v8.0] Idle tasks completed: {len(tasks_completed)}, failed: {len(tasks_failed)}")

    except Exception as e:
        logger.error(f"[v8.0] Idle optimization cycle failed: {e}")
        log_optimization_event("Idle Optimization Error", str(e))

# =============================================================================
# ADVANCED ANALYTICS FUNCTIONS - v8.0 Enhanced
# =============================================================================
def get_optimization_metrics() -> Dict[str, Any]:
    """Get current optimization metrics and system status."""
    try:
        current_usage = monitor_system_resources()

        return {
            "current": current_usage,
            "history": _performance_history[-10:] if _performance_history else [],
            "thresholds": _adaptive_thresholds,
            "status": "active",
            "version": "8.0.0"
        }
    except Exception as e:
        logger.error(f"[v8.0] Failed to get metrics: {e}")
        return {"error": str(e), "version": "8.0.0"}

def predict_resource_usage() -> Dict[str, Any]:
    """Predict future resource usage based on historical trends."""
    try:
        if len(_performance_history) < 5:
            return {"status": "insufficient_data", "version": "8.0.0"}

        # Simple trend analysis (can be enhanced with ML later)
        recent = _performance_history[-5:]
        cpu_trend = sum(item.get("cpu", 0) for item in recent) / len(recent)
        mem_trend = sum(item.get("memory", 0) for item in recent) / len(recent)

        return {
            "predicted_cpu": round(cpu_trend, 2),
            "predicted_memory": round(mem_trend, 2),
            "confidence": "medium",
            "based_on_samples": len(recent),
            "version": "8.0.0"
        }
    except Exception as e:
        logger.error(f"[v8.0] Prediction failed: {e}")
        return {"error": str(e), "version": "8.0.0"}

def intelligent_threshold_adjustment() -> None:
    """Adjust thresholds based on system behavior patterns."""
    try:
        if len(_performance_history) < 10:
            return

        recent = _performance_history[-10:]
        avg_cpu = sum(item.get("cpu", 0) for item in recent) / len(recent)
        avg_mem = sum(item.get("memory", 0) for item in recent) / len(recent)

        # Adjust thresholds slightly based on sustained usage
        if avg_cpu > CPU_THRESHOLD * 0.9:
            _adaptive_thresholds['cpu'] = min(95, CPU_THRESHOLD + 5)

        if avg_mem > MEMORY_THRESHOLD * 0.9:
            _adaptive_thresholds['memory'] = min(95, MEMORY_THRESHOLD + 5)

        logger.debug("[v8.0] Adaptive thresholds updated")
        log_optimization_event("Threshold Adjustment", f"CPU={_adaptive_thresholds['cpu']} MEM={_adaptive_thresholds['memory']}")

    except Exception as e:
        logger.error(f"[v8.0] Threshold adjustment failed: {e}")

def get_optimization_recommendations() -> List[str]:
    """Generate optimization recommendations based on current state."""
    recommendations = []
    try:
        usage = monitor_system_resources()

        if usage.get("cpu", 0) > _adaptive_thresholds['cpu']:
            recommendations.append("Consider CPU optimization: close intensive apps")

        if usage.get("memory", 0) > _adaptive_thresholds['memory']:
            recommendations.append("Consider memory optimization: close unused apps")

        if usage.get("disk", 0) > _adaptive_thresholds['disk']:
            recommendations.append("Consider disk cleanup: remove temporary files")

        return recommendations

    except Exception as e:
        logger.error(f"[v8.0] Failed to get recommendations: {e}")
        return [f"Error generating recommendations: {e}"]

def schedule_optimization_tasks() -> None:
    """Schedule optimization tasks based on system state."""
    try:
        # Placeholder for future task scheduling system
        intelligent_threshold_adjustment()
        logger.debug("[v8.0] Optimization tasks scheduled")
    except Exception as e:
        logger.error(f"[v8.0] Task scheduling failed: {e}")

def analyze_performance_trends() -> Dict[str, Any]:
    """Analyze long-term performance trends."""
    try:
        if not _performance_history:
            return {"status": "no_data", "version": "8.0.0"}

        # Basic trend analysis
        cpu_values = [item.get("cpu", 0) for item in _performance_history]
        mem_values = [item.get("memory", 0) for item in _performance_history]

        return {
            "cpu_avg": round(sum(cpu_values) / len(cpu_values), 2),
            "cpu_max": max(cpu_values),
            "mem_avg": round(sum(mem_values) / len(mem_values), 2),
            "mem_max": max(mem_values),
            "samples": len(_performance_history),
            "version": "8.0.0"
        }

    except Exception as e:
        logger.error(f"[v8.0] Trend analysis failed: {e}")
        return {"error": str(e), "version": "8.0.0"}

# ====================================================================
# END OF SarahMemoryOptimization.py v8.0.0
# ====================================================================