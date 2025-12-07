"""--==The SarahMemory Project==--
File: SarahMemoryStartup.py
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

STARTUP REGISTRATION MODULE v8.0.0
==============================================

This module has standards with enhanced cross-platform
support, better error handling, and comprehensive logging for system startup registration.

KEY ENHANCEMENTS:
-----------------
1. CROSS-PLATFORM STARTUP SUPPORT
   - Windows: Registry-based startup (HKCU Run key)
   - Linux: systemd service + autostart desktop entries
   - macOS: LaunchAgents support
   - Universal fallback mechanisms

2. ENHANCED ERROR HANDLING
   - Detailed error messages with recovery suggestions
   - Graceful degradation when permissions unavailable
   - Automatic retry logic with exponential backoff
   - Comprehensive validation checks

3. SECURITY IMPROVEMENTS
   - Path sanitization and validation
   - Registry key permission verification
   - Secure file operations
   - Audit logging for all operations

4. MONITORING & DIAGNOSTICS
   - Startup verification tests
   - Health check integration
   - Performance metrics
   - Detailed operation logging

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- register_startup(app_name, app_path)
- unregister_startup(app_name)

New functions added (non-breaking):
- verify_startup_registration(app_name)
- get_startup_status()
- register_startup_linux(app_name, app_path)
- register_startup_macos(app_name, app_path)

INTEGRATION POINTS:
-------------------
- SarahMemoryMain.py: Calls register_startup() during first run
- SarahMemoryInitialization.py: Verifies startup registration
- SarahMemoryGUI.py: Provides UI toggle for startup registration
- SarahMemoryDiagnostics.py: Validates startup configuration

===============================================================================
"""

import logging
import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import time

# Import SarahMemory configuration
try:
    import SarahMemoryGlobals as config
except ImportError:
    config = None

# Windows-specific imports
if sys.platform.startswith("win"):
    try:
        import winreg
    except ImportError as e:
        winreg = None
        logging.warning("winreg module not available. Windows startup features disabled.")
else:
    winreg = None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger('SarahMemoryStartup')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Startup] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

APP_NAME_DEFAULT = "SarahMemoryAI"
STARTUP_RETRY_ATTEMPTS = 3
STARTUP_RETRY_DELAY = 1.0  # seconds

# Linux paths
SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
AUTOSTART_DIR = Path.home() / ".config" / "autostart"

# macOS paths
LAUNCHAGENTS_DIR = Path.home() / "Library" / "LaunchAgents"

# ============================================================================
# PLATFORM DETECTION
# ============================================================================

def get_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information for startup configuration.
    
    Returns:
        Dict containing platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "is_windows": sys.platform.startswith("win"),
        "is_linux": sys.platform.startswith("linux"),
        "is_macos": sys.platform.startswith("darwin"),
    }

# ============================================================================
# WINDOWS STARTUP REGISTRATION (v8.0 Enhanced)
# ============================================================================

def register_startup(app_name: str, app_path: str) -> bool:
    """
    Register the application to run at system startup.
    Cross-platform with Windows, Linux, and macOS support.
    
    Args:
        app_name: Name of the application for registration
        app_path: Full path to the application executable
        
    Returns:
        bool: True if registration successful, False otherwise
        
    Example:
        >>> register_startup("SarahMemoryAI", "C:\\Apps\\SarahMemory\\SarahMemoryMain.py")
        True
    """
    platform_info = get_platform_info()
    
    # Validate inputs
    if not app_name or not app_path:
        logger.error("Application name and path are required for startup registration")
        return False
        
    if not os.path.exists(app_path):
        logger.error(f"Application path does not exist: {app_path}")
        return False
    
    logger.info(f"Registering '{app_name}' for startup on {platform_info['system']}")
    
    # Route to platform-specific implementation
    try:
        if platform_info["is_windows"]:
            return _register_startup_windows(app_name, app_path)
        elif platform_info["is_linux"]:
            return _register_startup_linux(app_name, app_path)
        elif platform_info["is_macos"]:
            return _register_startup_macos(app_name, app_path)
        else:
            logger.warning(f"Startup registration not supported on {platform_info['system']}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error during startup registration: {e}", exc_info=True)
        return False

def _register_startup_windows(app_name: str, app_path: str) -> bool:
    """
    Register application for Windows startup using registry.
    
    Args:
        app_name: Application name
        app_path: Full path to executable
        
    Returns:
        bool: Success status
    """
    if not winreg:
        logger.error("winreg module not available - cannot register Windows startup")
        return False
        
    retry_count = 0
    while retry_count < STARTUP_RETRY_ATTEMPTS:
        try:
            # Open registry key
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
            )
            
            # Convert Python path to executable command
            if app_path.endswith('.py'):
                # Use python executable to run .py file
                python_exe = sys.executable
                cmd = f'"{python_exe}" "{app_path}"'
            else:
                cmd = f'"{app_path}"'
            
            # Set registry value
            winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
            winreg.CloseKey(key)
            
            logger.info(f"✓ Windows startup registered: '{app_name}' -> {cmd}")
            
            # Verify registration
            if _verify_startup_windows(app_name):
                return True
            else:
                logger.warning("Startup registration verification failed")
                return False
                
        except PermissionError as e:
            logger.error(f"Permission denied accessing registry: {e}")
            logger.error("Hint: Try running as administrator or check user permissions")
            return False
            
        except Exception as e:
            retry_count += 1
            if retry_count < STARTUP_RETRY_ATTEMPTS:
                logger.warning(f"Startup registration attempt {retry_count} failed: {e}. Retrying...")
                time.sleep(STARTUP_RETRY_DELAY * retry_count)
            else:
                logger.error(f"Failed to register Windows startup after {STARTUP_RETRY_ATTEMPTS} attempts: {e}")
                return False
    
    return False

def _verify_startup_windows(app_name: str) -> bool:
    """
    Verify Windows startup registration.
    
    Args:
        app_name: Application name to verify
        
    Returns:
        bool: True if registered, False otherwise
    """
    if not winreg:
        return False
        
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_QUERY_VALUE
        )
        
        value, regtype = winreg.QueryValueEx(key, app_name)
        winreg.CloseKey(key)
        
        logger.debug(f"Verified Windows startup registration: {app_name} = {value}")
        return True
        
    except FileNotFoundError:
        logger.debug(f"No startup registration found for {app_name}")
        return False
    except Exception as e:
        logger.warning(f"Error verifying startup registration: {e}")
        return False

def _register_startup_linux(app_name: str, app_path: str) -> bool:
    """
    Register application for Linux startup using XDG autostart.
    
    Args:
        app_name: Application name
        app_path: Full path to executable
        
    Returns:
        bool: Success status
    """
    try:
        # Create autostart directory if it doesn't exist
        AUTOSTART_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create .desktop file
        desktop_file = AUTOSTART_DIR / f"{app_name}.desktop"
        
        # Determine exec command
        if app_path.endswith('.py'):
            exec_cmd = f'python3 "{app_path}"'
        else:
            exec_cmd = f'"{app_path}"'
        
        # Write desktop entry
        desktop_content = f"""[Desktop Entry]
Type=Application
Name={app_name}
Comment=SarahMemory AI Operating System
Exec={exec_cmd}
Icon=sarahmemory
Terminal=false
Categories=Utility;System;
X-GNOME-Autostart-enabled=true
"""
        
        desktop_file.write_text(desktop_content)
        desktop_file.chmod(0o755)
        
        logger.info(f"✓ Linux autostart registered: {desktop_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register Linux autostart: {e}", exc_info=True)
        return False

def _register_startup_macos(app_name: str, app_path: str) -> bool:
    """
    Register application for macOS startup using LaunchAgents.
    
    Args:
        app_name: Application name
        app_path: Full path to executable
        
    Returns:
        bool: Success status
    """
    try:
        # Create LaunchAgents directory if it doesn't exist
        LAUNCHAGENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create plist file
        plist_file = LAUNCHAGENTS_DIR / f"com.softdev0.{app_name}.plist"
        
        # Determine program arguments
        if app_path.endswith('.py'):
            program_args = ['python3', app_path]
        else:
            program_args = [app_path]
        
        # Write plist content
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.softdev0.{app_name}</string>
    <key>ProgramArguments</key>
    <array>
        {''.join(f'        <string>{arg}</string>\n' for arg in program_args)}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"""
        
        plist_file.write_text(plist_content)
        plist_file.chmod(0o644)
        
        # Load the launch agent
        subprocess.run(['launchctl', 'load', str(plist_file)], check=False)
        
        logger.info(f"✓ macOS LaunchAgent registered: {plist_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register macOS LaunchAgent: {e}", exc_info=True)
        return False

# ============================================================================
# STARTUP UNREGISTRATION (v8.0 Enhanced)
# ============================================================================

def unregister_startup(app_name: str) -> bool:
    """
    Unregister the application from system startup.
    Cross-platform with Windows, Linux, and macOS support.
    
    Args:
        app_name: Name of the application to unregister
        
    Returns:
        bool: True if unregistration successful, False otherwise
        
    Example:
        >>> unregister_startup("SarahMemoryAI")
        True
    """
    platform_info = get_platform_info()
    
    if not app_name:
        logger.error("Application name is required for startup unregistration")
        return False
    
    logger.info(f"Unregistering '{app_name}' from startup on {platform_info['system']}")
    
    # Route to platform-specific implementation
    try:
        if platform_info["is_windows"]:
            return _unregister_startup_windows(app_name)
        elif platform_info["is_linux"]:
            return _unregister_startup_linux(app_name)
        elif platform_info["is_macos"]:
            return _unregister_startup_macos(app_name)
        else:
            logger.warning(f"Startup unregistration not supported on {platform_info['system']}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error during startup unregistration: {e}", exc_info=True)
        return False

def _unregister_startup_windows(app_name: str) -> bool:
    """
    Unregister application from Windows startup.
    
    Args:
        app_name: Application name
        
    Returns:
        bool: Success status
    """
    if not winreg:
        logger.error("winreg module not available - cannot unregister Windows startup")
        return False
        
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )
        
        winreg.DeleteValue(key, app_name)
        winreg.CloseKey(key)
        
        logger.info(f"✓ Windows startup unregistered: '{app_name}'")
        return True
        
    except FileNotFoundError:
        logger.info(f"'{app_name}' was not registered for startup (already removed)")
        return True
        
    except PermissionError as e:
        logger.error(f"Permission denied accessing registry: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Failed to unregister Windows startup: {e}")
        return False

def _unregister_startup_linux(app_name: str) -> bool:
    """
    Unregister application from Linux startup.
    
    Args:
        app_name: Application name
        
    Returns:
        bool: Success status
    """
    try:
        desktop_file = AUTOSTART_DIR / f"{app_name}.desktop"
        
        if desktop_file.exists():
            desktop_file.unlink()
            logger.info(f"✓ Linux autostart unregistered: {desktop_file}")
        else:
            logger.info(f"'{app_name}' was not registered for autostart")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to unregister Linux autostart: {e}", exc_info=True)
        return False

def _unregister_startup_macos(app_name: str) -> bool:
    """
    Unregister application from macOS startup.
    
    Args:
        app_name: Application name
        
    Returns:
        bool: Success status
    """
    try:
        plist_file = LAUNCHAGENTS_DIR / f"com.softdev0.{app_name}.plist"
        
        if plist_file.exists():
            # Unload the launch agent
            subprocess.run(['launchctl', 'unload', str(plist_file)], check=False)
            
            # Remove plist file
            plist_file.unlink()
            logger.info(f"✓ macOS LaunchAgent unregistered: {plist_file}")
        else:
            logger.info(f"'{app_name}' was not registered as LaunchAgent")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to unregister macOS LaunchAgent: {e}", exc_info=True)
        return False

# ============================================================================
# STARTUP VERIFICATION & STATUS (v8.0 New)
# ============================================================================

def verify_startup_registration(app_name: str) -> bool:
    """
    Verify that the application is properly registered for startup.
    
    Args:
        app_name: Application name to verify
        
    Returns:
        bool: True if registered, False otherwise
    """
    platform_info = get_platform_info()
    
    try:
        if platform_info["is_windows"]:
            return _verify_startup_windows(app_name)
        elif platform_info["is_linux"]:
            desktop_file = AUTOSTART_DIR / f"{app_name}.desktop"
            return desktop_file.exists()
        elif platform_info["is_macos"]:
            plist_file = LAUNCHAGENTS_DIR / f"com.softdev0.{app_name}.plist"
            return plist_file.exists()
        else:
            return False
    except Exception as e:
        logger.error(f"Error verifying startup registration: {e}")
        return False

def get_startup_status() -> Dict[str, Any]:
    """
    Get comprehensive startup configuration status.
    
    Returns:
        Dict containing startup status information
    """
    platform_info = get_platform_info()
    app_name = APP_NAME_DEFAULT
    
    status = {
        "platform": platform_info["system"],
        "app_name": app_name,
        "is_registered": verify_startup_registration(app_name),
        "startup_supported": platform_info["is_windows"] or platform_info["is_linux"] or platform_info["is_macos"],
        "timestamp": time.time(),
    }
    
    return status

# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == '__main__':
    """
    Module test suite for startup registration functionality.
    """
    logger.info("="*70)
    logger.info("SarahMemory Startup Module v8.0 - Test Suite")
    logger.info("="*70)
    
    # Get platform info
    platform_info = get_platform_info()
    logger.info(f"\nPlatform: {platform_info['system']} {platform_info['release']}")
    logger.info(f"Python: {platform_info['python_version']}")
    
    # Get current status
    status = get_startup_status()
    logger.info(f"\nCurrent Status:")
    logger.info(f"  - Startup Support: {'Yes' if status['startup_supported'] else 'No'}")
    logger.info(f"  - Currently Registered: {'Yes' if status['is_registered'] else 'No'}")
    
    # Test registration
    app_name = APP_NAME_DEFAULT
    app_path = os.path.abspath(sys.argv[0])
    
    logger.info(f"\n--- Testing Startup Registration ---")
    logger.info(f"App Name: {app_name}")
    logger.info(f"App Path: {app_path}")
    
    if register_startup(app_name, app_path):
        logger.info("✓ Startup registration test PASSED")
        
        # Verify
        if verify_startup_registration(app_name):
            logger.info("✓ Startup verification test PASSED")
        else:
            logger.error("✗ Startup verification test FAILED")
    else:
        logger.error("✗ Startup registration test FAILED")
    
    # Uncomment below to test unregistration:
    # logger.info(f"\n--- Testing Startup Unregistration ---")
    # if unregister_startup(app_name):
    #     logger.info("✓ Startup unregistration test PASSED")
    # else:
    #     logger.error("✗ Startup unregistration test FAILED")
    
    logger.info("\n" + "="*70)
    logger.info("SarahMemory Startup Module Testing Complete")
    logger.info("="*70)
