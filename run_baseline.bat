@echo off
setlocal
cd /d "%~dp0"

set "BASELINE_PYTHON=%CD%\.venv\Scripts\python.exe"
if exist "%BASELINE_PYTHON%" goto run

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import mujoco, gymnasium, numpy" >nul 2>nul
    if not errorlevel 1 (
        set "BASELINE_PYTHON=py -3.11"
        goto run
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import mujoco, gymnasium, numpy" >nul 2>nul
    if not errorlevel 1 (
        set "BASELINE_PYTHON=python"
        goto run
    )
)

echo No usable Python environment was found.
echo.
echo Open a terminal in this folder and run:
echo   py -3.11 -m venv .venv
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1

:run
%BASELINE_PYTHON% scripts\view_baseline.py
if errorlevel 1 pause
