@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ==========================================================
:: SarahMemory v8.x Dependency Installer (Windows BAT)
:: - Mode: VENV or GLOBAL
:: - OS prompt: Windows / Linux / Other
:: - Installs requirements line-by-line
:: - Purges pip cache after each install
:: - Applies workarounds for known failure-prone packages:
::     * pyaudio   -> sounddevice fallback (no extra downloads)
::     * pygame    -> pygame-ce fallback (esp. py>=3.14)
::     * pywebview -> install --no-deps + bottle/proxy_tools; skip pythonnet
::     * pythonnet -> skip (source build failure likely on bleeding-edge Python)
:: ==========================================================

title SarahMemory Dependency Installer
color a

echo ==========================================================
echo   SarahMemory Dependency Installer
echo ==========================================================
echo.

:: ---------------------------
:: OS Selection (BAT runs on Windows; Linux choice prints instructions)
:: ---------------------------
echo Select target OS:
echo   [1] Windows (this installer runs)
echo   [2] Linux / macOS (print commands + exit)
echo   [3] Other (print guidance + exit)
set /p SM_OS=Enter choice (1/2/3) :

if "%SM_OS%"=="2" goto LINUX_GUIDE
if "%SM_OS%"=="3" goto OTHER_GUIDE
if not "%SM_OS%"=="1" (
  echo Invalid choice. Defaulting to Windows.
)

:: ---------------------------
:: Environment Mode
:: ---------------------------
echo.
echo Select install mode:
echo   [1] Use Virtual Environment (.venv)  (recommended)
echo   [2] Install into current Python (no venv)
set /p SM_MODE=Enter choice (1/2) :

if "%SM_MODE%"=="1" (
  call :ENSURE_VENV
  if errorlevel 1 goto FAIL_EXIT
) else (
  echo [MODE] Global / current Python environment.
)

:: ---------------------------
:: Toolchain upgrade (pip/setuptools/wheel)
:: ---------------------------
call :UPGRADE_TOOLCHAIN
if errorlevel 1 (
  echo [WARN] Toolchain upgrade failed. Continuing (may cause build failures).
)

:: ---------------------------
:: Choose requirements input(s)
:: - If req*.txt exist, install them in filename order.
:: - Else install requirements.txt
:: ---------------------------
set "FOUND_SPLIT=0"
for %%F in (req*.txt) do (
  set "FOUND_SPLIT=1"
  goto HAVE_SPLIT
)

:HAVE_SPLIT
if "%FOUND_SPLIT%"=="1" (
  echo.
  echo [PLAN] Split requirements detected (req*.txt). Installing in order:
  for /f "delims=" %%F in ('dir /b /on req*.txt') do (
    echo   - %%F
  )
  echo.
  for /f "delims=" %%F in ('dir /b /on req*.txt') do (
    call :INSTALL_REQUIREMENTS_FILE "%%F"
    if errorlevel 1 (
      echo [ERROR] Failed while processing %%F
      goto FAIL_EXIT
    )
  )
) else (
  if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found and no req*.txt present.
    echo Put this BAT in the same folder as requirements.txt (or req*.txt files).
    goto FAIL_EXIT
  )
  echo.
  echo [PLAN] Installing requirements.txt
  call :INSTALL_REQUIREMENTS_FILE "requirements.txt"
  if errorlevel 1 goto FAIL_EXIT
)

:: ---------------------------
:: Final verification summary
:: ---------------------------
echo.
echo ==========================================================
echo   INSTALL COMPLETE
echo ==========================================================
echo Verifying key imports (best effort)...
python -c "import sys; print('Python:', sys.version)"
python -c "import pip; print('pip:', pip.__version__)"
python -c "import numpy; print('numpy OK')"
python -c "import requests; print('requests OK')"
echo.
echo Done.
goto OK_EXIT


:: ==========================================================
:: FUNCTIONS
:: ==========================================================

:ENSURE_VENV
echo.
echo [VENV] Ensuring .venv exists and activating...
if not exist ".venv\Scripts\activate.bat" (
  echo [VENV] Creating .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Check Python install.
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv.
  exit /b 1
)
echo [VENV] Active: %CD%\.venv
exit /b 0


:UPGRADE_TOOLCHAIN
echo.
echo [TOOLS] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
exit /b 0


:INSTALL_REQUIREMENTS_FILE
set "REQFILE=%~1"
if not exist "%REQFILE%" (
  echo [ERROR] Missing file: %REQFILE%
  exit /b 1
)

echo.
echo ==========================================================
echo   Processing: %REQFILE%
echo ==========================================================

:: Install line-by-line so we can:
:: - cache purge after each
:: - apply per-package fallbacks on failure
for /f "usebackq delims=" %%L in ("%REQFILE%") do (
  set "LINE=%%L"
  call :TRIM_LINE LINE

  :: skip blanks
  if "!LINE!"=="" (
    rem noop
  ) else (
    :: skip comments
    if "!LINE:~0,1!"=="#" (
      rem noop
    ) else (
      call :INSTALL_ONE "!LINE!"
      if errorlevel 1 (
        echo [ERROR] Line failed with no viable fallback: "!LINE!"
        exit /b 1
      )
    )
  )
)

exit /b 0


:TRIM_LINE
:: trims leading spaces only (good enough for req files)
set "S=!%~1!"
for /f "tokens=* delims= " %%A in ("!S!") do set "S=%%A"
set "%~1=!S!"
exit /b 0


:INSTALL_ONE
set "SPEC=%~1"

:: Normalize to a “base name” for fallback routing:
::  - remove markers (split on ;)
::  - remove version pins (split on <,>,=,~,! )
set "BASE=%SPEC%"
for /f "tokens=1 delims=;" %%A in ("%BASE%") do set "BASE=%%A"

for %%D in (^< ^> ^= ^~ ^! ^  ) do (
  for /f "tokens=1 delims=%%D" %%A in ("!BASE!") do set "BASE=%%A"
)

call :TRIM_LINE BASE
set "BASE_LOWER=!BASE!"
for %%Z in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
  set "BASE_LOWER=!BASE_LOWER:%%Z=%%Z!"
)
:: (above loop is a no-op in cmd; we'll just compare case-insensitive using /I)

echo.
echo [INSTALL] %SPEC%
python -m pip install "%SPEC%"
if not errorlevel 1 (
  call :CACHE_PURGE
  exit /b 0
)

echo [WARN] Install failed: %SPEC%
echo [WORKAROUND] Evaluating fallback for: !BASE!

:: ---------------------------
:: Fallbacks (no extra downloads)
:: ---------------------------

:: 1) PyAudio -> sounddevice (no PortAudio headers required)
if /I "!BASE!"=="pyaudio" (
  echo [FALLBACK] Replacing PyAudio with sounddevice...
  python -m pip uninstall -y pyaudio >nul 2>&1
  python -m pip install sounddevice
  if errorlevel 1 exit /b 1
  call :CACHE_PURGE
  exit /b 0
)

:: 2) pygame -> pygame-ce (wheel availability on newer Pythons)
if /I "!BASE!"=="pygame" (
  echo [FALLBACK] Replacing pygame with pygame-ce...
  python -m pip uninstall -y pygame >nul 2>&1
  python -m pip install pygame-ce
  if errorlevel 1 exit /b 1
  call :CACHE_PURGE
  exit /b 0
)

:: 3) pywebview -> install without deps, then add runtime deps (skip pythonnet)
if /I "!BASE!"=="pywebview" (
  echo [FALLBACK] Installing pywebview without pythonnet...
  python -m pip uninstall -y pywebview pythonnet >nul 2>&1
  python -m pip install --no-deps pywebview
  if errorlevel 1 exit /b 1
  python -m pip install bottle proxy_tools
  if errorlevel 1 exit /b 1
  call :CACHE_PURGE
  exit /b 0
)

:: 4) pythonnet -> skip (source builds fail frequently; keep pipeline moving)
if /I "!BASE!"=="pythonnet" (
  echo [FALLBACK] Skipping pythonnet on this runtime (non-blocking).
  exit /b 0
)

:: 5) torch -> if a normal install fails, try CPU index (no extra downloads)
if /I "!BASE!"=="torch" (
  echo [FALLBACK] Trying CPU wheels for torch...
  python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 exit /b 1
  call :CACHE_PURGE
  exit /b 0
)

:: If no fallback matched, fail hard
exit /b 1


:CACHE_PURGE
echo [CACHE] Purging pip cache...
python -m pip cache purge >nul 2>&1
exit /b 0


:: ==========================================================
:: NON-WINDOWS GUIDANCE
:: ==========================================================
:LINUX_GUIDE
echo.
echo ==========================================================
echo   Linux / macOS Install Guidance
echo ==========================================================
echo 1) python3 -m venv .venv
echo 2) source .venv/bin/activate
echo 3) python -m pip install --upgrade pip setuptools wheel
echo 4) pip install -r requirements.txt   (or your req*.txt files in order)
echo.
echo NOTE: If pyaudio fails, use: pip install sounddevice
echo NOTE: If pygame fails, use: pip install pygame-ce
echo NOTE: If pywebview/pythonnet fails, skip it and use WebUI/Flask frontend.
echo.
goto OK_EXIT

:OTHER_GUIDE
echo.
echo ==========================================================
echo   Other OS Guidance
echo ==========================================================
echo This is a Windows BAT installer. For other platforms:
echo - Use the Linux/macOS guidance (option 2) as your baseline.
echo - Or build a platform-native installer (bash / PowerShell).
echo.
goto OK_EXIT


:FAIL_EXIT
echo.
echo ==========================================================
echo   INSTALL FAILED
echo ==========================================================
echo Review the last package shown above. That is the blocker.
echo.
exit /b 1

:OK_EXIT
echo.
echo Exiting installer.
exit /b 0
