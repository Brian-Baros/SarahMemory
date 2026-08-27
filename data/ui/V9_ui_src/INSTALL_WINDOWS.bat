@echo off
setlocal
cd /d "%~dp0"

echo [SarahMemory] Installing frontend dependencies, including dev tools...
set NPM_CONFIG_PRODUCTION=false
call npm ci --include=dev --legacy-peer-deps
if errorlevel 1 (
  echo [SarahMemory] npm ci failed. Retrying with npm install to repair lock/cache drift.
  call npm install --include=dev --legacy-peer-deps || exit /b 1
)

echo [SarahMemory] Installed.
call node --version
call npm --version
call node_modules\.bin\vite.cmd --version
pause
