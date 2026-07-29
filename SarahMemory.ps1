<#
--==The SarahMemory Project==--
File: SarahMemory.ps1
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
Purpose: Windows PowerShell ignition trigger for SarahMemory AiOS.
This file is a thin launcher only. It enters .\core and runs SarahMemoryMain.py.
#>

$ErrorActionPreference = "Stop"

$SarahMemoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SarahMemoryCore = Join-Path $SarahMemoryRoot "core"
$SarahMemoryMain = Join-Path $SarahMemoryCore "SarahMemoryMain.py"
$SarahMemoryLauncher = "SarahMemory.ps1"

if (-not (Test-Path -LiteralPath $SarahMemoryCore -PathType Container)) {
    Write-Error "[SarahMemory Launcher] Missing core directory: $SarahMemoryCore"
    exit 2
}

if (-not (Test-Path -LiteralPath $SarahMemoryMain -PathType Leaf)) {
    Write-Error "[SarahMemory Launcher] Missing core entry file: $SarahMemoryMain"
    exit 3
}

$VenvPython = Join-Path $SarahMemoryRoot "venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    $PythonExe = $VenvPython
} else {
    $PythonExe = "python"
}

$env:SARAHMEMORY_ROOT = $SarahMemoryRoot
$env:SARAHMEMORY_CORE = $SarahMemoryCore
$env:SARAHMEMORY_LAUNCHER = $SarahMemoryLauncher

Push-Location $SarahMemoryCore
try {
    & $PythonExe "SarahMemoryMain.py" @args
    $SarahMemoryExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $SarahMemoryExitCode

# ====================================================================
# END OF SarahMemory.ps1 v9.0.0
# ====================================================================
