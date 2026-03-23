@echo off
echo ============================================
echo   ReelTranscribe - Windows Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download from: https://python.org/downloads
    echo IMPORTANT: Check "Add Python to PATH" during install!
    pause
    exit /b 1
)
echo [OK] Python found

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FFmpeg is not installed or not in PATH.
    echo.
    echo To install FFmpeg:
    echo   1. Download from https://www.gyan.dev/ffmpeg/builds/
    echo   2. Download "ffmpeg-release-essentials.zip"
    echo   3. Extract to C:\ffmpeg
    echo   4. Add C:\ffmpeg\bin to System PATH
    echo   5. Restart this terminal and run setup again
    echo.
    pause
    exit /b 1
)
echo [OK] FFmpeg found

REM Check Instaloader (optional)
instaloader --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] instaloader is not installed or not in PATH.
    echo "pip install instaloader" to enable fallback Instagram downloads.
) else (
    echo [OK] Instaloader found
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create directories
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs
if not exist static mkdir static

REM Check .env
if not exist .env (
    echo.
    echo [WARNING] .env file needs your OpenAI API key!
    echo Open .env file and replace "sk-proj-paste-your-key-here" with your actual key.
    echo Get your key from: https://platform.openai.com/api-keys
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo NEXT STEPS:
echo   1. Open .env file and paste your OpenAI API key
echo   2. Run: start.bat
echo.
pause
