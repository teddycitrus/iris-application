@echo off
REM Launch the Iris eye-tracking app (visible console, shows live logs).
REM Prefers the local .venv; falls back to Python 3.12 via the py launcher.
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    py -3.12 main.py
)

echo.
echo Iris exited. Close this window or press a key.
pause >nul
