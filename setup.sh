#!/bin/bash
echo "============================================"
echo "  ReelTranscribe - Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    echo "Install: https://python.org/downloads"
    exit 1
fi
echo "[OK] Python3 found"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[ERROR] FFmpeg is not installed."
    echo "Install:"
    echo "  Mac:    brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi
echo "[OK] FFmpeg found"

# Check Instaloader (optional)
if ! command -v instaloader &> /dev/null; then
    echo "[WARNING] instaloader is not installed."
    echo "Install: pip install instaloader"
else
    echo "[OK] Instaloader found"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
mkdir -p uploads outputs static

# Check .env
if grep -q "sk-proj-paste-your-key-here" .env 2>/dev/null; then
    echo ""
    echo "[WARNING] .env file needs your OpenAI API key!"
    echo "Open .env and replace the placeholder with your actual key."
    echo "Get your key from: https://platform.openai.com/api-keys"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Open .env and paste your OpenAI API key"
echo "  2. Run: ./start.sh"
