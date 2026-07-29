@echo off
setlocal EnableExtensions
Title SarahMemory AiOS v9 Setup Launcher
color 0A

REM --==The SarahMemory Project==--
REM File: SarahMemorySetup.bat
REM Part of the SarahMemory AiOS Governed Cognitive Runtime
REM Version: v9.0.0
REM Date: 2026-06-06
REM Time: 10:11:54
REM Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
REM
REM v9 operational setup shim. Creates/uses local venv, installs requirements
REM only when requirements.txt exists, then launches the root SarahMemory trigger.
REM Heavy indexing/model/database operations are intentionally NOT automatic here.

set "SARAH_ROOT=%~dp0"
cd /d "%SARAH_ROOT%"

echo [SarahMemory Setup] Root: %SARAH_ROOT%

if not exist "venv\Scripts\python.exe" (
    echo [SarahMemory Setup] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 goto fail
)

echo [SarahMemory Setup] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 goto fail

if exist "requirements.txt" (
    echo [SarahMemory Setup] Installing/updating Python requirements...
    python -m pip install --upgrade pip
    if errorlevel 1 goto fail
    pip install -r requirements.txt
    if errorlevel 1 goto fail
) else (
    echo [SarahMemory Setup] requirements.txt not found. Skipping package install.
)

echo [SarahMemory Setup] Launching SarahMemory...
python SarahMemory.py
exit /b %errorlevel%

:fail
echo [SarahMemory Setup] FAILED. Review the console output above.
pause
exit /b 1

REM ====================================================================
REM END OF SarahMemorySetup.bat v9.0.0
REM ====================================================================
