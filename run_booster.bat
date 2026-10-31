@echo off
title Ollama Download Booster
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

python -c "import customtkinter" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies...
    python -m pip install -r requirements.txt
)

python main.py %*
if %errorlevel% neq 0 (
    pause
)
