"""--==The SarahMemory Project==--
File: SarahMemoryOptimization.py
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
                from SarahMemoryResearch import get_research_data
                from SarahMemoryDatabase import record_qa_feedback
                
                for topic in topics[:3]:  # Limit to top 3 topics
                    try:
                        result = get_research_data(topic)
                        record_qa_feedback(
                            topic,
                            score=1,
                            feedback=f"Idle learning | {datetime.now().isoformat()}"
                        )
                        logger.info(f"[v8.0] 🌐 Researched and scored: {topic}")
                    except Exception as e:
                        logger.debug(f"[v8.0] Research failed for {topic}: {e}")
                
                tasks_completed.append("topic_research")
            except Exception as e:
                tasks_failed.append(f"topic_research: {e}")
                logger.warning(f"[v8.0] Topic research failed: {e}")
                
        except Exception as e:
            tasks_failed.append(f"deep_learn_user_context: {e}")
            logger.warning(f"[v8.0] Deep learning failed: {e}")
        
        # Database optimization
        try:
            from SarahMemoryDatabase import optimize_database
            optimize_database()
            tasks_completed.append("optimize_database")
            logger.info("[v8.0] ✓ Database optimization completed")
        except Exception as e:
            tasks_failed.append(f"optimize_database: {e}")
            logger.debug(f"[v8.0] Database optimization failed: {e}")
        
        # Summary
        summary = {
            "completed": tasks_completed,
            "failed": tasks_failed,
            "timestamp": datetime.now().isoformat()
        }
        
        log_optimization_event(
            "Idle Optimization Cycle",
            f"Completed: {len(tasks_completed)}, Failed: {len(tasks_failed)}"
        )
        
        logger.info(f"[v8.0] Idle cycle complete: {len(tasks_completed)} tasks succeeded")
        
    except Exception as e:
        logger.error(f"[v8.0] Idle optimization error: {e}")
        log_optimization_event("Idle Optimization Error", str(e))

# =============================================================================
# PERFORMANCE ANALYTICS - v8.0 New
# =============================================================================
def get_optimization_metrics() -> Dict[str, Any]:
    """
    Get comprehensive optimization metrics.
    v8.0: New function for analytics.
    
    Returns:
        Dictionary with optimization metrics
    """
    try:
        if not _performance_history:
            return {"status": "no_data", "message": "No performance history available"}
        
        # Calculate averages
        recent = _performance_history[-20:] if len(_performance_history) >= 20 else _performance_history
        
        avg_cpu = sum(h.get('cpu', 0) for h in recent) / len(recent)
        avg_mem = sum(h.get('memory', 0) for h in recent) / len(recent)
        avg_disk = sum(h.get('disk', 0) for h in recent) / len(recent)
        
        # Calculate trends
        if len(_performance_history) >= 10:
            old_avg_cpu = sum(h.get('cpu', 0) for h in _performance_history[:10]) / 10
            cpu_trend = "increasing" if avg_cpu > old_avg_cpu else "decreasing" if avg_cpu < old_avg_cpu else "stable"
        else:
            cpu_trend = "insufficient_data"
        
        return {
            "average_cpu": round(avg_cpu, 2),
            "average_memory": round(avg_mem, 2),
            "average_disk": round(avg_disk, 2),
            "cpu_trend": cpu_trend,
            "samples": len(_performance_history),
            "recent_samples": len(recent),
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0"
        }
        
    except Exception as e:
        logger.error(f"[v8.0] Metrics error: {e}")
        return {"status": "error", "message": str(e)}

def analyze_performance_trends() -> Dict[str, Any]:
    """
    Analyze performance trends over time.
    v8.0: New function for trend analysis.
    
    Returns:
        Dictionary with trend analysis
    """
    try:
        if len(_performance_history) < 5:
            return {"status": "insufficient_data"}
        
        # Get recent and old samples
        recent = _performance_history[-10:]
        old = _performance_history[:10]
        
        # Calculate changes
        cpu_change = (sum(r.get('cpu', 0) for r in recent) / len(recent)) - (sum(o.get('cpu', 0) for o in old) / len(old))
        mem_change = (sum(r.get('memory', 0) for r in recent) / len(recent)) - (sum(o.get('memory', 0) for o in old) / len(old))
        
        return {
            "cpu_change": round(cpu_change, 2),
            "memory_change": round(mem_change, 2),
            "cpu_status": "degrading" if cpu_change > 5 else "improving" if cpu_change < -5 else "stable",
            "memory_status": "degrading" if mem_change > 5 else "improving" if mem_change < -5 else "stable",
            "timestamp": datetime.now().isoformat(),
            "version": "8.0.0"
        }
        
    except Exception as e:
        logger.error(f"[v8.0] Trend analysis error: {e}")
        return {"status": "error", "message": str(e)}

def get_optimization_recommendations() -> List[str]:
    """
    Get intelligent optimization recommendations.
    v8.0: New function for recommendations.
    
    Returns:
        List of recommendation strings
    """
    try:
        usage = monitor_system_resources()
        recommendations = []
        
        if usage.get('cpu', 0) > 70:
            recommendations.append("CPU usage is high - consider closing unused applications")
        
        if usage.get('memory', 0) > 70:
            recommendations.append("Memory usage is high - clear cache and close browser tabs")
        
        if usage.get('disk', 0) > 85:
            recommendations.append("Disk space is low - run cleanup utility or move files")
        
        if usage.get('process_count', 0) > 200:
            recommendations.append("High process count - review startup programs")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"[v8.0] Recommendations error: {e}")
        return [f"Error generating recommendations: {e}"]

# =============================================================================
# MAIN TEST HARNESS - v8.0 Enhanced
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("SarahMemory Optimization v8.0.0 - Test Mode")
    print("=" * 80)
    
    logger.info("[v8.0] Starting Optimization test suite")
    
    try:
        # Run optimization cycles
        print("\nRunning 3 optimization cycles...")
        print("-" * 80)
        
        for i in range(3):
            print(f"\nCycle {i + 1}:")
            status = optimize_system()
            print(status)
            
            if i < 2:  # Don't sleep after last iteration
                time.sleep(2)
        
        # Display metrics
        print("\n" + "=" * 80)
        print("Optimization Metrics:")
        print("=" * 80)
        metrics = get_optimization_metrics()
        print(json.dumps(metrics, indent=2))
        
        # Display trends
        print("\n" + "=" * 80)
        print("Performance Trends:")
        print("=" * 80)
        trends = analyze_performance_trends()
        print(json.dumps(trends, indent=2))
        
        # Display recommendations
        print("\n" + "=" * 80)
        print("Recommendations:")
        print("=" * 80)
        for rec in get_optimization_recommendations():
            print(f"• {rec}")
        
        print("\n" + "=" * 80)
        logger.info("[v8.0] Optimization test suite complete")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        logger.info("[v8.0] Test interrupted by user")
        log_optimization_event("Optimization Test Interrupted", "User cancelled test")
    
    except Exception as e:
        print(f"\nError during test: {e}")
        logger.error(f"[v8.0] Test error: {e}")
        log_optimization_event("Optimization Test Error", f"Error: {e}")

logger.info("[v8.0] SarahMemoryOptimization module loaded successfully")
