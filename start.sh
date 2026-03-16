#!/bin/bash
echo "============================================"
echo "  ReelTranscribe - Starting Server"
echo "============================================"
echo ""

source venv/bin/activate

if grep -q "sk-proj-paste-your-key-here" .env 2>/dev/null; then
    echo "[ERROR] You haven't set your OpenAI API key!"
    echo "Open .env and replace the placeholder with your real key."
    exit 1
fi

echo "Starting server at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

python main.py
