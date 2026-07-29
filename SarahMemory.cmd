@echo off
REM --==The SarahMemory Project==--
REM File: SarahMemory.cmd
REM Part of the SarahMemory AiOS Governed Cognitive Runtime
REM Version: v9.0.0
REM Date: 2026-06-06
REM Time: 10:11:54
REM Author: (c) 2025, 2026 Brian Lee Baros. All Rights Reserved.
REM Purpose: Windows CMD / double-click ignition trigger for SarahMemory AiOS.
REM This file is a thin launcher only. It enters .\core and runs SarahMemoryMain.py.

setlocal

set "SARAHMEMORY_ROOT=%~dp0"
set "SARAHMEMORY_CORE=%SARAHMEMORY_ROOT%core"
set "SARAHMEMORY_MAIN=%SARAHMEMORY_CORE%\SarahMemoryMain.py"
set "SARAHMEMORY_LAUNCHER=SarahMemory.cmd"

if not exist "%SARAHMEMORY_CORE%\" (
    echo [SarahMemory Launcher] ERROR: Missing core directory: "%SARAHMEMORY_CORE%"
    exit /b 2
)

if not exist "%SARAHMEMORY_MAIN%" (
    echo [SarahMemory Launcher] ERROR: Missing core entry file: "%SARAHMEMORY_MAIN%"
    exit /b 3
)

if exist "%SARAHMEMORY_ROOT%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SARAHMEMORY_ROOT%venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

pushd "%SARAHMEMORY_CORE%" >nul
"%PYTHON_EXE%" "SarahMemoryMain.py" %*
set "SARAHMEMORY_EXIT_CODE=%ERRORLEVEL%"
popd >nul

exit /b %SARAHMEMORY_EXIT_CODE%

REM ====================================================================
REM END OF SarahMemory.cmd v9.0.0
REM ====================================================================
