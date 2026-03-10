@echo off
setlocal EnableExtensions

rem SarahMemory manual fresh-boot cleanup
rem Run this BAT from the SarahMemory BASE_DIR.
rem It uses the BAT file's own folder as BASE_DIR.

cd /d "%~dp0"
set "BASE_DIR=%CD%"

echo ============================================================
echo SarahMemory Manual Fresh-Boot Cleanup
echo BASE_DIR: %BASE_DIR%
echo ============================================================

echo [1/4] Stopping Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
taskkill /F /IM py.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo [2/4] Removing __pycache__ folders...
for /d /r "%BASE_DIR%" %%D in (__pycache__) do (
    if exist "%%~fD" rd /s /q "%%~fD"
)

echo [3/4] Removing compiled Python artifacts...
for /r "%BASE_DIR%" %%F in (*.pyc) do (
    if exist "%%~fF" del /f /q "%%~fF" >nul 2>&1
)
for /r "%BASE_DIR%" %%F in (*.pyo) do (
    if exist "%%~fF" del /f /q "%%~fF" >nul 2>&1
)

echo [4/4] Cleanup complete.
echo.
echo Next step:
echo     python SarahMemoryMain.py
echo.
endlocal