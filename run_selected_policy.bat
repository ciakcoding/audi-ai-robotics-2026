@echo off
setlocal
cd /d "%~dp0"

set "PPO_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PPO_PYTHON%" (
    echo Missing .venv. Follow the setup instructions in README.md first.
    pause
    exit /b 1
)

"%PPO_PYTHON%" scripts\view_ppo.py
if errorlevel 1 pause
