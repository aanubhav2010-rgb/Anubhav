# ReelTranscribe

Production-grade verbatim transcription tool for Instagram Reels, videos, and audio files.

**Stack:** FastAPI + OpenAI Whisper + FFmpeg + yt-dlp

---

## Folder Structure

```
reeltranscribe/
├── main.py              # FastAPI backend + full processing pipeline
├── templates/
│   └── index.html       # Frontend UI
├── static/              # Static assets
├── uploads/             # Temp uploaded files (auto-cleaned)
├── outputs/             # Transcription JSON results
├── requirements.txt     # Python dependencies
├── .env                 # Your API key goes here
├── setup.bat            # Windows one-click setup
├── start.bat            # Windows one-click start
├── setup.sh             # Mac/Linux setup
├── start.sh             # Mac/Linux start
├── Dockerfile           # Docker deployment
├── Procfile             # Render/Railway deployment
├── render.yaml          # Render config
├── railway.json         # Railway config
├── .gitignore
└── README.md
```

---

## Prerequisites (Install These First)

### 1. Python 3.10+
- Download: https://python.org/downloads
- IMPORTANT (Windows): Check "Add Python to PATH" during installation

### 2. FFmpeg
- **Windows:** Download from https://www.gyan.dev/ffmpeg/builds/ → extract to C:\ffmpeg → add C:\ffmpeg\bin to PATH
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 3. OpenAI API Key
- Get from: https://platform.openai.com/api-keys
- Needs credits loaded (Whisper costs ~$0.006/minute)

---

## Quick Start (Windows)

1. Extract the ZIP
2. Open `.env` file → replace `sk-proj-paste-your-key-here` with your real OpenAI key
3. Double-click `setup.bat` (installs everything)
4. Double-click `start.bat` (starts the server)
5. Open http://localhost:8000 in browser

## Quick Start (Mac/Linux)

```bash
cd reeltranscribe
chmod +x setup.sh start.sh
nano .env   # paste your OpenAI API key
./setup.sh
./start.sh
# Open http://localhost:8000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/transcribe/url` | POST | Transcribe from Instagram URL |
| `/api/transcribe/upload` | POST | Transcribe from file upload |
| `/api/status/{job_id}` | GET | Check job status & get results |
| `/api/health` | GET | System health check |

---

## Deployment

### Render
1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Add env var: OPENAI_API_KEY
4. Deploy

### Railway
1. Push to GitHub
2. railway.app → New Project → connect repo
3. Add env var: OPENAI_API_KEY
4. Deploy

### Docker
```bash
docker build -t reeltranscribe .
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-..." reeltranscribe
```
