@echo off
REM ==========================================================================
REM  OKX Bots - Windows Installation
REM  Sets up both bots to run locally (no cloud droplet needed):
REM    - okx_tele_bot.py     : Telegram commands + OI/funding-flip alerts
REM    - okx_perp_screener.py : reactive price+volume spike alerts
REM  Run this from the repo's deploy\ folder (double-click, or in a terminal).
REM ==========================================================================

setlocal enabledelayedexpansion

echo Installing OKX bots for Windows...
echo.

REM Repo root is one level up from this deploy\ folder.
set SCRIPT_DIR=%~dp0..
cd /d "%SCRIPT_DIR%"

REM --- Check Python is available -------------------------------------------
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.10+ from https://python.org and tick "Add Python to PATH".
    pause
    exit /b 1
)

REM --- Virtual environment + dependencies ----------------------------------
echo Creating Python virtual environment...
python -m venv venv

echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM --- Run scripts (one per bot) -------------------------------------------
echo Creating run scripts...
(
    echo @echo off
    echo cd /d "%SCRIPT_DIR%"
    echo call venv\Scripts\activate.bat
    echo python okx_tele_bot.py
) > "%SCRIPT_DIR%\run_okx_bot.bat"

(
    echo @echo off
    echo cd /d "%SCRIPT_DIR%"
    echo call venv\Scripts\activate.bat
    echo python okx_perp_screener.py
) > "%SCRIPT_DIR%\run_okx_spike.bat"

echo.
echo ==========================================================================
echo  Installation complete!
echo ==========================================================================
echo.
echo Next steps:
echo   1. Edit config.json  ^-  set telegram_bot_token and telegram_chat_id
echo   2. Test the Telegram bot:  double-click run_okx_bot.bat
echo      ^(you should get "OKX Telegram bot online" in your chat; send /help^)
echo   3. Test the spike screener: double-click run_okx_spike.bat
echo   4. Always-on 24/7 without windows: see deploy\okx_windows_scheduler.md
echo.
pause
