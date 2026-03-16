"""
ReelTranscribe — Production-grade Instagram Reel / Video / Audio transcription tool.
Backend: FastAPI  |  STT: OpenAI Whisper large-v3  |  Video: FFmpeg + yt-dlp
"""

import os
import re
import uuid
import json
import shutil
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import openai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reeltranscribe")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
MAX_FILE_SIZE_MB = 500

# Filler words to strip for clean version
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "hmm", "hmmm", "ah", "ahh",
    "er", "err", "like", "matlab", "toh", "na", "haan",
    "you know", "i mean", "basically", "actually", "so yeah",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ReelTranscribe",
    description="100% accurate verbatim transcription from Instagram Reels, videos, and audio files.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory job store (swap for Redis/DB in prod)
jobs: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set.")
    return openai.OpenAI(api_key=api_key)


def run_cmd(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a shell command with timeout."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(result.stderr)
    return result


def download_instagram_reel(url: str, dest_dir: Path) -> Path:
    """Download Instagram Reel using yt-dlp and return the video file path."""
    output_template = str(dest_dir / "%(id)s.%(ext)s")
    
    # Try without cookies first
    cmd_no_cookies = [
        "yt-dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "-o", output_template,
        url,
    ]
    
    # Fallback with browser cookies
    cmd_with_cookies = [
        "yt-dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--cookies-from-browser", "chrome",
        url,
    ]
    
    try:
        run_cmd(cmd_no_cookies, timeout=120)
    except RuntimeError:
        logger.info("Retrying with browser cookies...")
        try:
            run_cmd(cmd_with_cookies, timeout=120)
        except RuntimeError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to download video. Instagram may require login. Try uploading the video directly. Error: {str(e)[:300]}"
            )

    # Find the downloaded file
    files = list(dest_dir.glob("*.*"))
    video_files = [f for f in files if f.suffix.lower() in ALLOWED_VIDEO_EXT]
    if not video_files:
        raise HTTPException(status_code=422, detail="Download succeeded but no video file found.")
    return video_files[0]


def extract_audio(video_path: Path, dest_dir: Path) -> Path:
    """Extract audio from video using FFmpeg -> 16kHz mono WAV (optimal for Whisper)."""
    audio_path = dest_dir / f"{video_path.stem}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16 kHz
        "-ac", "1",               # mono
        str(audio_path),
    ]
    run_cmd(cmd, timeout=120)
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Audio extraction failed.")
    return audio_path


def transcribe_with_whisper(audio_path: Path) -> dict:
    """
    Transcribe audio using OpenAI Whisper API (large-v3 / whisper-1).
    Returns {"text": ..., "segments": [...], "words": [...]}.
    """
    client = get_openai_client()

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    logger.info(f"Transcribing {audio_path.name} ({file_size_mb:.1f} MB)")

    # Whisper API limit is 25 MB. If larger, compress first.
    actual_path = audio_path
    if file_size_mb > 24:
        compressed = audio_path.parent / f"{audio_path.stem}_compressed.mp3"
        run_cmd([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-b:a", "64k", "-ar", "16000", "-ac", "1",
            str(compressed)
        ])
        actual_path = compressed

    with open(actual_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,
            prompt=(
                "Transcribe verbatim. Preserve every word exactly as spoken. "
                "Include all fillers: um, uh, hmm, like, matlab, toh, haan, na. "
                "Preserve Hinglish, code-switching, and mixed-language speech exactly. "
                "Do NOT correct grammar. Do NOT paraphrase. Do NOT summarize."
            ),
        )

    result = {
        "text": response.text,
        "language": getattr(response, "language", "unknown"),
        "duration": getattr(response, "duration", None),
        "segments": [],
        "words": [],
    }

    if hasattr(response, "segments") and response.segments:
        result["segments"] = [
            {
                "id": seg.id if hasattr(seg, "id") else i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for i, seg in enumerate(response.segments)
        ]

    if hasattr(response, "words") and response.words:
        result["words"] = [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in response.words
        ]

    return result


def generate_clean_text(verbatim: str) -> str:
    """
    Produce clean, copy-ready text:
      - Remove filler words
      - Fix punctuation
      - Paragraph format
    """
    text = verbatim

    # Remove filler words (case-insensitive, word-boundary)
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        pattern = r'\b' + re.escape(filler) + r'\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up multiple spaces and orphan punctuation
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
    text = text.strip()

    # Split into sentences for paragraph formatting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    paragraphs = []
    chunk = []
    for i, sentence in enumerate(sentences):
        chunk.append(sentence.strip())
        if len(chunk) >= 3 or i == len(sentences) - 1:
            paragraphs.append(' '.join(chunk))
            chunk = []

    return '\n\n'.join(paragraphs)


def cleanup_files(*paths):
    """Remove temporary files/dirs."""
    for p in paths:
        try:
            if p and Path(p).exists():
                if Path(p).is_dir():
                    shutil.rmtree(p)
                else:
                    Path(p).unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed for {p}: {e}")


# ---------------------------------------------------------------------------
# Processing Pipeline
# ---------------------------------------------------------------------------
async def process_job(job_id: str, source_type: str, source_path: str, url: Optional[str] = None):
    """Main transcription pipeline — runs in background."""
    job = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp(prefix="reel_"))

    try:
        job["status"] = "processing"
        job["step"] = "preparing"

        # Step 1: Get audio file
        audio_path = None

        if source_type == "url":
            job["step"] = "downloading"
            logger.info(f"[{job_id}] Downloading from URL: {url}")
            video_path = download_instagram_reel(url, tmp_dir)
            job["step"] = "extracting_audio"
            audio_path = extract_audio(video_path, tmp_dir)

        elif source_type == "video":
            job["step"] = "extracting_audio"
            video_path = Path(source_path)
            audio_path = extract_audio(video_path, tmp_dir)

        elif source_type == "audio":
            audio_path = Path(source_path)
            # Convert to optimal format for Whisper
            optimal = tmp_dir / "optimized.wav"
            run_cmd([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(optimal)
            ])
            audio_path = optimal

        if audio_path is None or not audio_path.exists():
            raise RuntimeError("No audio file available for transcription.")

        # Step 2: Transcribe
        job["step"] = "transcribing"
        logger.info(f"[{job_id}] Transcribing...")
        whisper_result = transcribe_with_whisper(audio_path)

        # Step 3: Generate outputs
        job["step"] = "formatting"
        verbatim = whisper_result["text"]
        clean = generate_clean_text(verbatim)

        # Build word-level transcript
        word_transcript = ""
        if whisper_result["words"]:
            word_lines = []
            for w in whisper_result["words"]:
                ts = f"[{w['start']:.2f}s]"
                word_lines.append(f"{ts} {w['word']}")
            word_transcript = "\n".join(word_lines)

        # Save outputs
        output_data = {
            "verbatim_transcript": verbatim,
            "clean_text": clean,
            "word_level_transcript": word_transcript,
            "language": whisper_result["language"],
            "duration_seconds": whisper_result["duration"],
            "segments": whisper_result["segments"],
            "word_count": len(verbatim.split()),
        }

        output_file = OUTPUT_DIR / f"{job_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        job["status"] = "completed"
        job["step"] = "done"
        job["result"] = output_data
        logger.info(f"[{job_id}] Transcription complete. Words: {output_data['word_count']}")

    except HTTPException as e:
        job["status"] = "failed"
        job["error"] = e.detail
        logger.error(f"[{job_id}] Failed: {e.detail}")
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:500]
        logger.error(f"[{job_id}] Failed: {e}")
    finally:
        # Cleanup temp files
        cleanup_files(tmp_dir)
        if source_type in ("video", "audio") and source_path:
            cleanup_files(source_path)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/transcribe/url")
async def transcribe_url(background_tasks: BackgroundTasks, url: str = Form(...)):
    """Transcribe from Instagram Reel URL."""
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL is required.")

    job_id = str(uuid.uuid4())[:12]
    jobs[job_id] = {"status": "queued", "step": "initializing", "result": None, "error": None}

    background_tasks.add_task(process_job, job_id, "url", None, url.strip())
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/transcribe/upload")
async def transcribe_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Transcribe from uploaded video or audio file."""
    ext = Path(file.filename).suffix.lower()
    is_video = ext in ALLOWED_VIDEO_EXT
    is_audio = ext in ALLOWED_AUDIO_EXT

    if not is_video and not is_audio:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {list(ALLOWED_VIDEO_EXT | ALLOWED_AUDIO_EXT)}"
        )

    # Save upload
    job_id = str(uuid.uuid4())[:12]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.0f} MB). Max: {MAX_FILE_SIZE_MB} MB.")
        with open(save_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    source_type = "video" if is_video else "audio"
    jobs[job_id] = {"status": "queued", "step": "initializing", "result": None, "error": None}

    background_tasks.add_task(process_job, job_id, source_type, str(save_path))
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Check transcription job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "step": job["step"],
    }
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
    return response


@app.get("/api/health")
async def health():
    """Health check."""
    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp": shutil.which("yt-dlp") is not None,
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }
    all_ok = all(checks.values())
    return {"healthy": all_ok, "checks": checks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
