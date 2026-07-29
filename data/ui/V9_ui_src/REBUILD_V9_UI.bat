@echo off
setlocal
cd /d "%~dp0\..\..\data\ui\V9_ui_src" 2>nul
if errorlevel 1 (
  cd /d "C:\SarahMemory\data\ui\V9_ui_src" || exit /b 1
)
echo [SarahMemory] Installing/updating V9 UI dependencies...
npm install || exit /b 1
echo [SarahMemory] Building V9 UI to data\ui\v9...
npm run build -- --outDir ..\v9 || exit /b 1
echo [SarahMemory] V9 UI rebuild complete.
pause
