@echo off
setlocal
cd /d "%~dp0"

if not exist "node_modules\.bin\vite.cmd" (
  echo [SarahMemory] node_modules missing. Installing dependencies first.
  set NPM_CONFIG_PRODUCTION=false
  call npm ci --include=dev --legacy-peer-deps
  if errorlevel 1 (
    echo [SarahMemory] npm ci failed. Retrying with npm install to repair lock/cache drift.
    call npm install --include=dev --legacy-peer-deps || exit /b 1
  )
)

echo [SarahMemory] Starting Vite UI on http://127.0.0.1:5173
echo [SarahMemory] Make sure the SarahMemory API Bridge is running on VITE_SARAH_API_URL or http://127.0.0.1:8000.
call npm run dev
