@echo off
echo ============================================
echo   ReelTranscribe - Starting Server
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check API key
findstr /C:"sk-proj-paste-your-key-here" .env >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] You haven't set your OpenAI API key!
    echo Open .env file and replace "sk-proj-paste-your-key-here" with your real key.
    echo Get your key from: https://platform.openai.com/api-keys
    echo.
    pause
    exit /b 1
)

echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python main.py
