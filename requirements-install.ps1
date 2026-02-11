<# ==========================================================
 SarahMemory v8.x Dependency Installer (One-shot)
 - Menu-driven: VENV or Global
 - OS prompt: Windows runs; Linux/macOS prints commands
 - Installs requirements line-by-line
 - pip cache purge after each install
 - Fallbacks for known pain points (no extra downloads)

 Usage (PowerShell as Admin recommended):
   powershell -ExecutionPolicy Bypass -File .\Install-SarahMemoryDependencies.ps1

 Put this script in the same directory as:
   - requirements.txt   OR
   - req1.txt, req2.txt, ... (split files) (it installs them in order)
========================================================== #>

$ErrorActionPreference = "Stop"

function Write-Header($text) {
  Write-Host ""
  Write-Host "==========================================================" -ForegroundColor Green
  Write-Host "  $text" -ForegroundColor Green
  Write-Host "==========================================================" -ForegroundColor Green
}

function Pause-AnyKey($msg="Press Enter to continue...") {
  Read-Host $msg | Out-Null
}

function Test-Command($cmd) {
  return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Run($file, $args, [switch]$IgnoreError) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $file
  $psi.Arguments = $args
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  $null = $p.Start()
  $stdout = $p.StandardOutput.ReadToEnd()
  $stderr = $p.StandardError.ReadToEnd()
  $p.WaitForExit()

  if ($stdout) { Write-Host $stdout }
  if ($stderr) { Write-Host $stderr -ForegroundColor DarkYellow }

  if ($p.ExitCode -ne 0 -and -not $IgnoreError) {
    throw "Command failed ($file $args) exit=$($p.ExitCode)"
  }
  return $p.ExitCode
}

function Get-Python() {
  if (Test-Command "python") { return "python" }
  if (Test-Command "py")     { return "py" }
  throw "Python not found on PATH. Install Python and re-run."
}

function Ensure-Venv([string]$pythonExe, [string]$venvPath) {
  Write-Header "VENV Mode"
  if (-not (Test-Path $venvPath)) {
    Write-Host "[VENV] Creating $venvPath ..."
    Run $pythonExe "-m venv `"$venvPath`"" 
  }
  $activate = Join-Path $venvPath "Scripts\Activate.ps1"
  if (-not (Test-Path $activate)) {
    throw "[VENV] Activation script not found: $activate"
  }
  Write-Host "[VENV] Activating..."
  . $activate
  Write-Host "[VENV] Active: $venvPath"
}

function Upgrade-Toolchain([string]$pythonExe) {
  Write-Header "Upgrading pip/setuptools/wheel"
  Run $pythonExe "-m pip install --upgrade pip setuptools wheel" -IgnoreError
}

function Pip-Cache-Purge([string]$pythonExe) {
  Write-Host "[CACHE] pip cache purge"
  Run $pythonExe "-m pip cache purge" -IgnoreError | Out-Null
}

function Normalize-BaseName([string]$spec) {
  # Remove markers after ';'
  $base = ($spec -split ';')[0].Trim()
  # Remove version constraints and extras
  $base = ($base -split '[<>=!~\s]')[0].Trim()
  return $base
}

function Install-One([string]$pythonExe, [string]$spec) {
  $specTrim = $spec.Trim()
  if ([string]::IsNullOrWhiteSpace($specTrim)) { return $true }
  if ($specTrim.StartsWith("#")) { return $true }

  Write-Host ""
  Write-Host "[INSTALL] $specTrim" -ForegroundColor Cyan

  $exit = Run $pythonExe "-m pip install `"$specTrim`"" -IgnoreError
  if ($exit -eq 0) {
    Pip-Cache-Purge $pythonExe
    return $true
  }

  $base = Normalize-BaseName $specTrim
  Write-Host "[WARN] Failed: $specTrim" -ForegroundColor Yellow
  Write-Host "[WORKAROUND] Evaluating fallback for: $base" -ForegroundColor Yellow

  switch -Regex ($base.ToLower()) {

    "^pyaudio$" {
      Write-Host "[FALLBACK] pyaudio -> sounddevice (no PortAudio headers needed)" -ForegroundColor Green
      Run $pythonExe "-m pip uninstall -y pyaudio" -IgnoreError | Out-Null
      $exit2 = Run $pythonExe "-m pip install sounddevice" -IgnoreError
      if ($exit2 -eq 0) { Pip-Cache-Purge $pythonExe; return $true }
      return $false
    }

    "^pygame$" {
      Write-Host "[FALLBACK] pygame -> pygame-ce (wheel-first strategy)" -ForegroundColor Green
      Run $pythonExe "-m pip uninstall -y pygame" -IgnoreError | Out-Null
      $exit2 = Run $pythonExe "-m pip install pygame-ce" -IgnoreError
      if ($exit2 -eq 0) { Pip-Cache-Purge $pythonExe; return $true }
      return $false
    }

    "^pywebview$" {
      Write-Host "[FALLBACK] pywebview (skip pythonnet): install --no-deps + bottle/proxy_tools" -ForegroundColor Green
      Run $pythonExe "-m pip uninstall -y pywebview pythonnet" -IgnoreError | Out-Null
      $exit2 = Run $pythonExe "-m pip install --no-deps pywebview" -IgnoreError
      if ($exit2 -ne 0) { return $false }
      $exit3 = Run $pythonExe "-m pip install bottle proxy_tools" -IgnoreError
      if ($exit3 -eq 0) { Pip-Cache-Purge $pythonExe; return $true }
      return $false
    }

    "^pythonnet$" {
      Write-Host "[FALLBACK] Skipping pythonnet on this runtime (non-blocking)." -ForegroundColor Green
      return $true
    }

    "^torch$" {
      Write-Host "[FALLBACK] torch: try CPU wheels index-url" -ForegroundColor Green
      Run $pythonExe "-m pip uninstall -y torch torchvision torchaudio" -IgnoreError | Out-Null
      $exit2 = Run $pythonExe "-m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -IgnoreError
      if ($exit2 -eq 0) { Pip-Cache-Purge $pythonExe; return $true }
      return $false
    }

    default {
      Write-Host "[FAIL] No fallback rule for: $base" -ForegroundColor Red
      return $false
    }
  }
}

function Install-RequirementsFile([string]$pythonExe, [string]$path) {
  Write-Header "Processing $path"
  if (-not (Test-Path $path)) { throw "Missing file: $path" }

  $lines = Get-Content -LiteralPath $path -ErrorAction Stop
  foreach ($line in $lines) {
    $ok = Install-One $pythonExe $line
    if (-not $ok) {
      throw "Dependency failed with no viable fallback: $line"
    }
  }
}

function Print-LinuxGuide {
  Write-Header "Linux / macOS Guidance"
  Write-Host "1) python3 -m venv .venv"
  Write-Host "2) source .venv/bin/activate"
  Write-Host "3) python -m pip install --upgrade pip setuptools wheel"
  Write-Host "4) pip install -r requirements.txt  (or req*.txt in order)"
  Write-Host ""
  Write-Host "Fallbacks:"
  Write-Host " - If pyaudio fails: pip install sounddevice"
  Write-Host " - If pygame fails:  pip install pygame-ce"
  Write-Host " - If pywebview/pythonnet fails: skip and use WebUI/Flask frontend"
}

# ==========================================================
# MAIN
# ==========================================================
Write-Header "SarahMemory Dependency Installer"

Write-Host "Select target OS:"
Write-Host "  [1] Windows (run install)"
Write-Host "  [2] Linux / macOS (print commands + exit)"
Write-Host "  [3] Other (print guidance + exit)"
$osChoice = Read-Host "Enter choice (1/2/3)"
if ($osChoice -eq "2") { Print-LinuxGuide; exit 0 }
if ($osChoice -eq "3") { Print-LinuxGuide; exit 0 }

$pythonExe = Get-Python

Write-Host ""
Write-Host "Select install mode:"
Write-Host "  [1] Virtual Environment (.venv)  (recommended)"
Write-Host "  [2] Current Python environment (no venv)"
$mode = Read-Host "Enter choice (1/2)"

if ($mode -eq "1") {
  Ensure-Venv $pythonExe (Join-Path (Get-Location) ".venv")
}

Upgrade-Toolchain $pythonExe

# Prefer split req files if present
$split = Get-ChildItem -Filter "req*.txt" -File -ErrorAction SilentlyContinue | Sort-Object Name
if ($split -and $split.Count -gt 0) {
  Write-Header "Plan: Split requirements detected"
  $split | ForEach-Object { Write-Host " - $($_.Name)" }
  foreach ($f in $split) {
    Install-RequirementsFile $pythonExe $f.FullName
  }
}
else {
  $master = Join-Path (Get-Location) "requirements.txt"
  if (-not (Test-Path $master)) {
    throw "requirements.txt not found and no req*.txt present. Put this script beside the requirements file(s)."
  }
  Install-RequirementsFile $pythonExe $master
}

Write-Header "Verification"
Run $pythonExe "-c `"import sys; print('Python:', sys.version)`"" -IgnoreError | Out-Null
Run $pythonExe "-c `"import pip; print('pip:', pip.__version__)`"" -IgnoreError | Out-Null
Run $pythonExe "-c `"import numpy; print('numpy OK')`"" -IgnoreError | Out-Null
Run $pythonExe "-c `"import requests; print('requests OK')`"" -IgnoreError | Out-Null

Write-Header "DONE"
Write-Host "All dependencies processed. If something was skipped (e.g., pythonnet), it was intentionally non-blocking."
exit 0
