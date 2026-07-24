@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~1"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

"%PYTHON_EXE%" evaluation\evaluate_extended_metrics.py --episodes 100 --seed 2026 --joint-noise 0.00 --output-dir outputs\extended_evaluation\nominal
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXE%" evaluation\evaluate_extended_metrics.py --episodes 100 --seed 2026 --joint-noise 0.08 --output-dir outputs\extended_evaluation\joint_noise_0p08
exit /b %errorlevel%
