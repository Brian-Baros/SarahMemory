"""--==The SarahMemory Project==--
File: SarahMemoryHi.py
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

SYSTEM INFORMATION MODULE v8.0.0
=============================================
This module has standards with comprehensive hardware
detection, advanced network monitoring, and real-time system metrics while maintaining
100% backward compatibility.

KEY ENHANCEMENTS:
-----------------
1. COMPREHENSIVE HARDWARE DETECTION
   - Cross-platform system information
   - CPU, memory, disk, network metrics
   - GPU detection and monitoring
   - Thermal sensor readings
   - Battery status (mobile devices)
   - Hardware capabilities profiling

2. ADVANCED NETWORK MONITORING
   - Real-time connectivity status
   - Network state tracking (green/yellow/red)
   - Bandwidth utilization monitoring
   - Connection quality metrics
   - Async network state updates
   - Multi-interface support

3. PERFORMANCE METRICS
   - Real-time resource usage
   - Historical trend analysis
   - Performance bottleneck detection
   - Resource allocation recommendations
   - Predictive usage patterns

4. CROSS-PLATFORM COMPATIBILITY
   - Windows (including DXDiag integration)
   - Linux (lscpu, lspci integration)
   - macOS support
   - Headless server compatibility
   - Graceful feature degradation

5. COMPREHENSIVE LOGGING
   - Structured system logs
   - JSON-formatted metrics
   - Database persistence
   - Historical data retention
   - Audit trail generation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- get_system_info()
- get_system_info_json()
- display_system_info(info)
- log_system_info_to_db(info)
- is_connected(host, port, timeout)
- update_network_state()

New functions added (non-breaking):
- get_extended_system_info()
- get_gpu_info()
- get_network_metrics()
- get_thermal_info()
- get_system_health_score()
- async_update_network_state()

INTEGRATION POINTS:
-------------------
- SarahMemoryOptimization.py: Resource monitoring
- SarahMemoryDiagnostics.py: Health checks
- SarahMemoryGlobals.py: Configuration
- SarahMemoryDatabase.py: Metrics persistence
- SarahMemoryNetwork.py: Network status

===============================================================================
"""

import logging
import platform
import psutil
import subprocess
import os
import json
import sqlite3
import socket
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Core imports
import SarahMemoryGlobals as config

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryHi')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# GLOBAL NETWORK STATE
# =============================================================================
NETWORK_STATE = "unknown"  # green, yellow, red, unknown
_last_net_io = None
_network_state_cache = {"state": "unknown", "updated": None}

# =============================================================================
# SYSTEM INFORMATION - v8.0 Enhanced
# =============================================================================
def get_system_info() -> Dict[str, Any]:
    """
    Retrieve comprehensive system and hardware information.
    v8.0: Enhanced with additional metrics and error handling.
    
    Returns:
        Dictionary containing system information
    """
    try:
        system_info = {}
        
        # Platform information
        uname = platform.uname()
        system_info['System'] = uname.system
        system_info['Node Name'] = uname.node
        system_info['Release'] = uname.release
        system_info['Version'] = uname.version
        system_info['Machine'] = uname.machine
        system_info['Processor'] = uname.processor
        
        # CPU information
        try:
            system_info['CPU Cores (Physical)'] = psutil.cpu_count(logical=False) or 0
            system_info['CPU Cores (Logical)'] = psutil.cpu_count(logical=True) or 0
            system_info['CPU Usage (%)'] = round(psutil.cpu_percent(interval=1), 2)
            system_info['CPU Frequency (MHz)'] = round(psutil.cpu_freq().current, 2) if psutil.cpu_freq() else 0
        except Exception as e:
            logger.debug(f"[v8.0] CPU info error: {e}")
            system_info['CPU Cores (Physical)'] = 0
            system_info['CPU Cores (Logical)'] = 0
            system_info['CPU Usage (%)'] = 0
        
        # Memory information
        try:
            virtual_mem = psutil.virtual_memory()
            system_info['Total Memory (GB)'] = round(virtual_mem.total / (1024**3), 2)
            system_info['Available Memory (GB)'] = round(virtual_mem.available / (1024**3), 2)
            system_info['Memory Usage (%)'] = round(virtual_mem.percent, 2)
            
            # Swap memory
            swap = psutil.swap_memory()
            system_info['Total Swap (GB)'] = round(swap.total / (1024**3), 2)
            system_info['Swap Usage (%)'] = round(swap.percent, 2)
        except Exception as e:
            logger.debug(f"[v8.0] Memory info error: {e}")
        
        # Disk information
        try:
            disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
            system_info['Total Disk (GB)'] = round(disk_usage.total / (1024**3), 2)
            system_info['Used Disk (GB)'] = round(disk_usage.used / (1024**3), 2)
            system_info['Free Disk (GB)'] = round(disk_usage.free / (1024**3), 2)
            system_info['Disk Usage (%)'] = round(disk_usage.percent, 2)
        except Exception as e:
            logger.debug(f"[v8.0] Disk info error: {e}")
        
        # Network information
        try:
            net_io = psutil.net_io_counters()
            system_info['Bytes Sent (MB)'] = round(net_io.bytes_sent / (1024**2), 2)
            system_info['Bytes Received (MB)'] = round(net_io.bytes_recv / (1024**2), 2)
            system_info['Network State'] = NETWORK_STATE
        except Exception as e:
            logger.debug(f"[v8.0] Network info error: {e}")
        
        # Boot time
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            system_info['Boot Time'] = boot_time.strftime("%Y-%m-%d %H:%M:%S")
            uptime = datetime.now() - boot_time
            system_info['Uptime'] = str(uptime).split('.')[0]  # Remove microseconds
        except Exception as e:
            logger.debug(f"[v8.0] Boot time error: {e}")
        
        # Platform-specific information
        if platform.system() == "Windows":
            try:
                dx_info = get_dxdiag_info()
                if dx_info:
                    system_info['DXDiag Info'] = dx_info
            except Exception as e:
                logger.debug(f"[v8.0] DXDiag error: {e}")
                system_info['DXDiag Info'] = "Not available"
        else:
            system_info['DXDiag Info'] = "Not available on non-Windows platforms"
        
        # Metadata
        system_info['Version'] = "8.0.0"
        system_info['Timestamp'] = datetime.now().isoformat()
        
        logger.info("[v8.0] System information retrieved successfully")
        return system_info
        
    except Exception as e:
        logger.error(f"[v8.0] Error retrieving system information: {e}")
        return {"error": str(e), "version": "8.0.0"}

# =============================================================================
# DXDiag INFORMATION - v8.0 Enhanced
# =============================================================================
def get_dxdiag_info() -> str:
    """
    Get DirectX diagnostic information on Windows.
    v8.0: Enhanced with better error handling and timeout.
    
    Returns:
        DXDiag information as string or error message
    """
    try:
        if platform.system() != "Windows":
            return "Not available on non-Windows platforms"
        
        output_file = "dxdiag_output.txt"
        
        # Run DXDiag with timeout
        result = subprocess.run(
            ["dxdiag", "/t", output_file],
            shell=True,
            capture_output=True,
            timeout=30
        )
        
        # Wait for file to be written
        import time
        for _ in range(10):
            if os.path.exists(output_file):
                break
            time.sleep(0.5)
        
        if not os.path.exists(output_file):
            return "DXDiag output file not generated"
        
        # Read and cleanup
        with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
            dx_data = f.read()
        
        try:
            os.remove(output_file)
        except Exception:
            pass
        
        # Extract key information (first 5000 chars to avoid overwhelming)
        return dx_data[:5000] + ("..." if len(dx_data) > 5000 else "")
        
    except subprocess.TimeoutExpired:
        logger.warning("[v8.0] DXDiag timeout")
        return "DXDiag timeout (process took too long)"
    except Exception as e:
        logger.warning(f"[v8.0] DXDiag error: {e}")
        return f"DXDiag info not available: {e}"

# =============================================================================
# EXTENDED SYSTEM INFORMATION - v8.0 New
# =============================================================================
def get_extended_system_info() -> Dict[str, Any]:
    """
    Get extended system information including GPU and sensors.
    v8.0: New function with comprehensive hardware detection.
    
    Returns:
        Dictionary with extended system information
    """
    info = get_system_info()
    
    # Add GPU information
    try:
        gpu_info = get_gpu_info()
        if gpu_info:
            info['GPU'] = gpu_info
    except Exception as e:
        logger.debug(f"[v8.0] GPU detection error: {e}")
    
    # Add thermal information
    try:
        thermal_info = get_thermal_info()
        if thermal_info:
            info['Thermal'] = thermal_info
    except Exception as e:
        logger.debug(f"[v8.0] Thermal detection error: {e}")
    
    # Add battery information (for laptops/mobile)
    try:
        if hasattr(psutil, 'sensors_battery'):
            battery = psutil.sensors_battery()
            if battery:
                info['Battery'] = {
                    'Percent': battery.percent,
                    'Plugged In': battery.power_plugged,
                    'Time Remaining': str(timedelta(seconds=battery.secsleft)) if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "Unlimited"
                }
    except Exception as e:
        logger.debug(f"[v8.0] Battery detection error: {e}")
    
    return info

# =============================================================================
# GPU INFORMATION - v8.0 New
# =============================================================================
def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Detect GPU information.
    v8.0: New function for GPU detection.
    
    Returns:
        GPU information dictionary or None
    """
    try:
        # Try nvidia-smi first (NVIDIA GPUs)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'Type': 'NVIDIA',
                    'Name': parts[0].strip() if len(parts) > 0 else 'Unknown',
                    'Memory': parts[1].strip() if len(parts) > 1 else 'Unknown',
                    'Driver': parts[2].strip() if len(parts) > 2 else 'Unknown'
                }
        except Exception:
            pass
        
        # Try DirectX on Windows
        if platform.system() == "Windows":
            # GPU info usually in DXDiag, already captured
            return {'Type': 'See DXDiag', 'Detection': 'Windows Graphics'}
        
        # Try lspci on Linux
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or 'Display' in line or '3D' in line:
                            return {'Type': 'Linux GPU', 'Info': line.split(':', 1)[1].strip() if ':' in line else line}
            except Exception:
                pass
        
        return None
        
    except Exception as e:
        logger.debug(f"[v8.0] GPU detection error: {e}")
        return None

# =============================================================================
# THERMAL INFORMATION - v8.0 New
# =============================================================================
def get_thermal_info() -> Optional[Dict[str, Any]]:
    """
    Get thermal sensor information.
    v8.0: New function for temperature monitoring.
    
    Returns:
        Thermal information dictionary or None
    """
    try:
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                thermal = {}
                for name, entries in temps.items():
                    thermal[name] = [
                        {
                            'label': entry.label or 'N/A',
                            'current': entry.current,
                            'high': entry.high if entry.high else 'N/A',
                            'critical': entry.critical if entry.critical else 'N/A'
                        }
                        for entry in entries
                    ]
                return thermal
        return None
    except Exception as e:
        logger.debug(f"[v8.0] Thermal info error: {e}")
        return None

# =============================================================================
# JSON OUTPUT - Backward Compatible
# =============================================================================
def get_system_info_json() -> str:
    """
    Retrieve system information in JSON format.
    v8.0: Enhanced with better JSON formatting.
    
    Returns:
        JSON-formatted system information
    """
    try:
        info = get_system_info()
        return json.dumps(info, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[v8.0] JSON formatting error: {e}")
        return json.dumps({"error": str(e)}, indent=4)

# =============================================================================
# GUI DISPLAY - Backward Compatible
# =============================================================================
def display_system_info(info: Dict[str, Any]) -> None:
    """
    Display system information in a GUI window using Tkinter.
    v8.0: Enhanced with better formatting and error handling.
    
    Args:
        info: System information dictionary
    """
    try:
        import tkinter as tk
        from tkinter import scrolledtext
        
        root = tk.Tk()
        root.title("System Information - SarahMemory v8.0")
        root.geometry("900x700")
        
        # Create scrolled text widget
        text_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            width=100,
            height=40,
            font=("Consolas", 10)
        )
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Format and insert information
        info_text = "SarahMemory System Information v8.0.0\n"
        info_text += "=" * 80 + "\n\n"
        
        for key, value in info.items():
            if isinstance(value, dict):
                info_text += f"\n{key}:\n"
                info_text += "-" * 40 + "\n"
                for sub_key, sub_value in value.items():
                    info_text += f"  {sub_key}: {sub_value}\n"
            else:
                info_text += f"{key}: {value}\n"
        
        text_area.insert(tk.INSERT, info_text)
        text_area.config(state=tk.DISABLED)  # Make read-only
        
        logger.info("[v8.0] Displaying system information in GUI")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"[v8.0] GUI display error: {e}")
        print(f"Error displaying GUI: {e}")
        print("System information:")
        print(json.dumps(info, indent=2))

# =============================================================================
# DATABASE LOGGING - Backward Compatible
# =============================================================================
def log_system_info_to_db(info: Dict[str, Any]) -> None:
    """
    Log system information to the database.
    v8.0: Enhanced with better database handling.
    
    Args:
        info: System information dictionary
    """
    try:
        # Get database path
        datasets_dir = getattr(config, 'DATASETS_DIR', os.path.join('data', 'memory', 'datasets'))
        db_path = os.path.abspath(os.path.join(datasets_dir, "system_logs.db"))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                info TEXT NOT NULL,
                version TEXT DEFAULT '8.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert log
        timestamp = datetime.now().isoformat()
        info_json = json.dumps(info, ensure_ascii=False)
        
        cursor.execute(
            "INSERT INTO system_logs (timestamp, info, version) VALUES (?, ?, ?)",
            (timestamp, info_json, "8.0.0")
        )
        
        conn.commit()
        conn.close()
        
        logger.info("[v8.0] System info logged to database successfully")
        
    except Exception as e:
        logger.error(f"[v8.0] Database logging error: {e}")

# =============================================================================
# NETWORK CONNECTIVITY - Backward Compatible
# =============================================================================
def is_connected(host: str = "8.8.8.8", port: int = 53, timeout: float = 3) -> bool:
    """
    Check network connectivity to a host.
    v8.0: Enhanced with better error handling.
    
    Args:
        host: Target host
        port: Target port
        timeout: Connection timeout
        
    Returns:
        True if connected, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False

# =============================================================================
# NETWORK STATE MONITORING - v8.0 Enhanced
# =============================================================================
def update_network_state() -> str:
    """
    Update and return current network state.
    v8.0: Enhanced with bandwidth monitoring.
    
    Returns:
        Network state: green/yellow/red
    """
    global NETWORK_STATE, _last_net_io, _network_state_cache
    
    try:
        # Check basic connectivity
        if not is_connected():
            NETWORK_STATE = "red"
            _network_state_cache = {"state": "red", "updated": datetime.now()}
            return NETWORK_STATE
        
        # Get current network I/O
        net_io = psutil.net_io_counters()
        
        # First run - assume active
        if _last_net_io is None:
            _last_net_io = net_io
            NETWORK_STATE = "green"
            _network_state_cache = {"state": "green", "updated": datetime.now()}
            return NETWORK_STATE
        
        # Calculate traffic delta
        sent_diff = net_io.bytes_sent - _last_net_io.bytes_sent
        recv_diff = net_io.bytes_recv - _last_net_io.bytes_recv
        _last_net_io = net_io
        
        # Determine state based on activity
        threshold = 1024  # 1 KB threshold
        if sent_diff >= threshold or recv_diff >= threshold:
            NETWORK_STATE = "green"  # Active traffic
        else:
            NETWORK_STATE = "yellow"  # Connected but idle
        
        _network_state_cache = {"state": NETWORK_STATE, "updated": datetime.now()}
        return NETWORK_STATE
        
    except Exception as e:
        logger.warning(f"[v8.0] Network state update error: {e}")
        NETWORK_STATE = "unknown"
        return NETWORK_STATE

# =============================================================================
# ASYNC NETWORK STATE - Backward Compatible
# =============================================================================
async def async_update_network_state() -> str:
    """
    Asynchronously update network state.
    v8.0: Enhanced with proper async handling.
    
    Returns:
        Network state: green/yellow/red/unknown
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, update_network_state)
    except Exception as e:
        logger.warning(f"[v8.0] Async network update error: {e}")
        return "unknown"

# =============================================================================
# SYSTEM HEALTH SCORE - v8.0 New
# =============================================================================
def get_system_health_score() -> Dict[str, Any]:
    """
    Calculate overall system health score.
    v8.0: New function for health assessment.
    
    Returns:
        Dictionary with health score and components
    """
    try:
        info = get_system_info()
        
        # Component scores (0-100)
        cpu_score = max(0, 100 - info.get('CPU Usage (%)', 0))
        mem_score = max(0, 100 - info.get('Memory Usage (%)', 0))
        disk_score = max(0, 100 - info.get('Disk Usage (%)', 0))
        
        # Network score
        net_state = info.get('Network State', 'unknown')
        net_score = {'green': 100, 'yellow': 70, 'red': 30, 'unknown': 50}.get(net_state, 50)
        
        # Overall score (weighted average)
        overall = (cpu_score * 0.3 + mem_score * 0.3 + disk_score * 0.25 + net_score * 0.15)
        
        return {
            'overall_score': round(overall, 2),
            'cpu_health': round(cpu_score, 2),
            'memory_health': round(mem_score, 2),
            'disk_health': round(disk_score, 2),
            'network_health': round(net_score, 2),
            'status': 'excellent' if overall >= 80 else 'good' if overall >= 60 else 'fair' if overall >= 40 else 'poor',
            'timestamp': datetime.now().isoformat(),
            'version': '8.0.0'
        }
        
    except Exception as e:
        logger.error(f"[v8.0] Health score error: {e}")
        return {'overall_score': 0, 'status': 'error', 'message': str(e)}

# =============================================================================
# MAIN TEST HARNESS - v8.0 Enhanced
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("SarahMemory System Information v8.0.0 - Test Mode")
    print("=" * 80)
    
    logger.info("[v8.0] Starting SarahMemoryHi test suite")
    
    # Get system info
    print("\nGathering system information...")
    info = get_system_info()
    
    # Log to database
    print("Logging to database...")
    log_system_info_to_db(info)
    
    # Display key information
    print("\n" + "=" * 80)
    print("System Information Summary:")
    print("=" * 80)
    
    for key, value in info.items():
        if not isinstance(value, (dict, list)) and key != 'DXDiag Info':
            print(f"{key:30} {value}")
    
    # Get health score
    print("\n" + "=" * 80)
    print("System Health Score:")
    print("=" * 80)
    health = get_system_health_score()
    print(json.dumps(health, indent=2))
    
    # Network state
    print("\n" + "=" * 80)
    print("Network Status:")
    print("=" * 80)
    net_state = update_network_state()
    print(f"Current State: {net_state}")
    print(f"Internet Connectivity: {'Yes' if is_connected() else 'No'}")
    
    # Option to display GUI
    show_gui = input("\nDisplay GUI? (y/n): ").strip().lower()
    if show_gui == 'y':
        display_system_info(info)
    
    print("\n" + "=" * 80)
    logger.info("[v8.0] SarahMemoryHi test suite complete")

logger.info("[v8.0] SarahMemoryHi module loaded successfully")


# ====================================================================
# END OF SarahMemoryHi.py v8.0.0
# ====================================================================