@echo off
REM GrowiHF Bot - Windows Installation
REM Run as Administrator

setlocal enabledelayedexpansion

echo Installing GrowiHF Reactive Bot for Windows...

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges
    echo Please run Command Prompt as Administrator
    pause
    exit /b 1
)

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0..

REM Create venv
echo Creating Python virtual environment...
cd /d "%SCRIPT_DIR%"
python -m venv venv

REM Activate venv and install deps
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

REM Create run script
echo Creating run script...
(
    echo @echo off
    echo cd /d "%SCRIPT_DIR%"
    echo call venv\Scripts\activate.bat
    echo python growhf_reactive_bot.py
) > "%SCRIPT_DIR%\run_bot.bat"

echo.
echo Installation complete!
echo.
echo To run the bot:
echo   Option 1 - Manual: double-click run_bot.bat
echo   Option 2 - Always-on: use Windows Task Scheduler (see deploy\windows_scheduler.md^)
echo.
pause
