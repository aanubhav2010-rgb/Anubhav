"""
ReelTranscribe — Production-grade Instagram Reel / Video / Audio transcription tool.
Backend: FastAPI  |  STT: OpenAI Whisper whisper-1  |  Video: yt-dlp (Python API) + instaloader fallback
"""

import os
import re
import time
import uuid
import json
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import openai
import yt_dlp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reeltranscribe")

UPLOAD_DIR   = Path("uploads")
OUTPUT_DIR   = Path("outputs")
DOWNLOAD_DIR = Path("downloads")   # persistent store — instagram_<unix_ts>.mp4

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
MAX_FILE_SIZE_MB  = 500

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
    description="Verbatim transcription from Instagram Reels, videos, and audio files.",
    version="3.0.0",
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


def _find_video_in_dir(directory: Path) -> Optional[Path]:
    """Return the first video file found inside *directory*, or None."""
    return next(
        (f for f in directory.glob("*.*") if f.suffix.lower() in ALLOWED_VIDEO_EXT),
        None,
    )


# ---------------------------------------------------------------------------
# NEW: download_instagram_video
# ---------------------------------------------------------------------------
def _ytdlp_download(url: str, out_path: Path) -> bool:
    """
    Use yt_dlp Python library to download *url* directly to *out_path*.
    Returns True on success, False on failure.
    """
    ydl_opts = {
        "outtmpl":              str(out_path),          # exact output path, no extension substitution
        "format":               "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format":  "mp4",
        "noplaylist":           True,
        "no_warnings":          False,
        "quiet":                False,
        "no_check_certificate": True,
    }

    # Attempt 1 — no cookies
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"yt-dlp (no cookies) → {out_path}")
            return True
    except Exception as e:
        logger.warning(f"yt-dlp (no cookies) failed: {e}")

    # Attempt 2 — Chrome cookies
    ydl_opts_cookies = {**ydl_opts, "cookiesfrombrowser": ("chrome",)}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_cookies) as ydl:
            ydl.download([url])
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"yt-dlp (chrome cookies) → {out_path}")
            return True
    except Exception as e:
        logger.warning(f"yt-dlp (chrome cookies) failed: {e}")

    return False


def _instaloader_download(url: str, tmp_dir: Path) -> Optional[Path]:
    """
    Fallback: use instaloader CLI to download a single public Instagram post/reel.
    Returns the downloaded video Path, or None on failure.
    """
    match = re.search(r"/(?:reel|p)/([A-Za-z0-9_-]+)", url)
    if not match:
        logger.error(f"Cannot parse Instagram shortcode from URL: {url}")
        return None

    shortcode = match.group(1)
    logger.info(f"instaloader shortcode: {shortcode}")

    cmd = [
        "instaloader",
        "--no-captions",
        "--no-metadata-json",
        "--no-compress-json",
        "--dirname-pattern", str(tmp_dir),
        "--filename-pattern", shortcode,
        "--", shortcode,
    ]

    try:
        run_cmd(cmd, timeout=180)
    except RuntimeError as e:
        logger.error(f"instaloader failed: {e}")
        return None

    return _find_video_in_dir(tmp_dir)


def download_instagram_video(url: str) -> Tuple[Optional[Path], Optional[str]]:
    """
    Download an Instagram Reel / Post and save it to:
        downloads/instagram_<unix_timestamp>.mp4

    Strategy:
        1. yt_dlp Python library  (no cookies)
        2. yt_dlp Python library  (chrome cookies)
        3. instaloader CLI        (public posts, no login)

    Returns:
        (saved_path, None)           — on success
        (None, error_message)        — if all methods fail
    """
    unix_ts   = int(time.time())
    filename  = f"instagram_{unix_ts}.mp4"
    dest_path = DOWNLOAD_DIR / filename

    # ── Primary: yt_dlp ──────────────────────────────────────────────────
    if _ytdlp_download(url, dest_path):
        return dest_path, None

    logger.warning("yt-dlp failed on both attempts — trying instaloader fallback...")

    # ── Fallback: instaloader ─────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="il_"))
    try:
        raw_video = _instaloader_download(url, tmp_dir)
        if raw_video:
            shutil.copy2(raw_video, dest_path)
            logger.info(f"instaloader → {dest_path}")
            return dest_path, None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── All methods failed ────────────────────────────────────────────────
    error_msg = (
        "All download methods failed (yt-dlp × 2, instaloader × 1). "
        "The post may be private or require login."
    )
    logger.error(error_msg)
    return None, error_msg


# ---------------------------------------------------------------------------
# Unchanged core helpers
# ---------------------------------------------------------------------------
def extract_audio(video_path: Path, dest_dir: Path) -> Path:
    """Extract audio from video using FFmpeg → 16 kHz mono WAV (optimal for Whisper)."""
    audio_path = dest_dir / f"{video_path.stem}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    run_cmd(cmd, timeout=120)
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Audio extraction failed.")
    return audio_path


def transcribe_with_whisper(audio_path: Path) -> dict:
    """
    Transcribe audio using OpenAI Whisper API (whisper-1).
    Returns {"text": ..., "segments": [...], "words": [...]}.
    """
    client = get_openai_client()

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    logger.info(f"Transcribing {audio_path.name} ({file_size_mb:.1f} MB)")

    actual_path = audio_path
    if file_size_mb > 24:
        compressed = audio_path.parent / f"{audio_path.stem}_compressed.mp3"
        run_cmd([
            "ffmpeg", "-y", "-i", str(audio_path),
            "-b:a", "64k", "-ar", "16000", "-ac", "1",
            str(compressed),
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
        "text":     response.text,
        "language": getattr(response, "language", "unknown"),
        "duration": getattr(response, "duration", None),
        "segments": [],
        "words":    [],
    }

    if hasattr(response, "segments") and response.segments:
        result["segments"] = [
            {
                "id":    seg.id if hasattr(seg, "id") else i,
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text,
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

    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        pattern = r'\b' + re.escape(filler) + r'\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
    text = text.strip()

    sentences  = re.split(r'(?<=[.!?])\s+', text)
    paragraphs = []
    chunk      = []
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
    job     = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp(prefix="reel_"))

    try:
        job["status"] = "processing"
        job["step"]   = "preparing"

        audio_path       = None
        saved_video_path = None
        download_error   = None

        # ------------------------------------------------------------------
        # Step 1: Acquire source
        # ------------------------------------------------------------------
        if source_type == "url":
            job["step"] = "downloading"
            logger.info(f"[{job_id}] Downloading from URL: {url}")

            # ── call the new dedicated function ───────────────────────────
            saved_video_path, download_error = download_instagram_video(url)
            # ─────────────────────────────────────────────────────────────

            if saved_video_path:
                job["video_file"] = str(saved_video_path)
                logger.info(f"[{job_id}] Video saved → {saved_video_path}")
            else:
                # download failed; pipeline cannot continue without video
                raise HTTPException(
                    status_code=422,
                    detail=download_error or "Video download failed.",
                )

            job["step"] = "extracting_audio"
            audio_path  = extract_audio(saved_video_path, tmp_dir)

        elif source_type == "video":
            job["step"] = "extracting_audio"
            video_path  = Path(source_path)
            audio_path  = extract_audio(video_path, tmp_dir)

        elif source_type == "audio":
            audio_path = Path(source_path)
            optimal    = tmp_dir / "optimized.wav"
            run_cmd([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(optimal),
            ])
            audio_path = optimal

        if audio_path is None or not audio_path.exists():
            raise RuntimeError("No audio file available for transcription.")

        # ------------------------------------------------------------------
        # Step 2: Transcribe
        # ------------------------------------------------------------------
        job["step"] = "transcribing"
        logger.info(f"[{job_id}] Transcribing...")
        whisper_result = transcribe_with_whisper(audio_path)

        # ------------------------------------------------------------------
        # Step 3: Format outputs
        # ------------------------------------------------------------------
        job["step"] = "formatting"
        verbatim = whisper_result["text"]
        clean    = generate_clean_text(verbatim)

        word_transcript = ""
        if whisper_result["words"]:
            word_transcript = "\n".join(
                f"[{w['start']:.2f}s] {w['word']}"
                for w in whisper_result["words"]
            )

        # ------------------------------------------------------------------
        # Build output payload
        # ------------------------------------------------------------------
        output_data: dict = {
            "verbatim_transcript":   verbatim,
            "clean_text":            clean,
            "word_level_transcript": word_transcript,
            "language":              whisper_result["language"],
            "duration_seconds":      whisper_result["duration"],
            "segments":              whisper_result["segments"],
            "word_count":            len(verbatim.split()),
        }

        # URL-job extras — matches required output format:
        #
        #   --------------------------------
        #   TRANSCRIPTION
        #   --------------------------------
        #   Hello everyone welcome to my channel...
        #
        #   --------------------------------
        #   VIDEO FILE
        #   --------------------------------
        #   downloads/instagram_1710602345.mp4
        #
        if source_type == "url" and saved_video_path:
            output_data["video_saved_at"]     = str(saved_video_path)
            output_data["video_download_url"] = f"/api/download/{job_id}"
            output_data["video_filename"]     = saved_video_path.name

        # Persist JSON result
        output_file = OUTPUT_DIR / f"{job_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        job["status"] = "completed"
        job["step"]   = "done"
        job["result"] = output_data
        logger.info(f"[{job_id}] Done. Words: {output_data['word_count']}")

    except HTTPException as e:
        job["status"] = "failed"
        job["error"]  = e.detail
        logger.error(f"[{job_id}] Failed: {e.detail}")
    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)[:500]
        logger.error(f"[{job_id}] Failed: {e}")
    finally:
        cleanup_files(tmp_dir)
        if source_type in ("video", "audio") and source_path:
            cleanup_files(source_path)


# ---------------------------------------------------------------------------
# Download-Only Pipeline
# ---------------------------------------------------------------------------
async def process_download_job(job_id: str, url: str):
    """
    Background task: download Instagram video WITHOUT transcription.
    Saves to downloads/instagram_<unix_ts>.mp4 and marks job completed.
    """
    job = jobs[job_id]
    try:
        job["status"] = "processing"
        job["step"]   = "downloading"
        logger.info(f"[{job_id}] Download-only job started for: {url}")

        saved_video_path, error = download_instagram_video(url)

        if saved_video_path:
            job["video_file"] = str(saved_video_path)
            job["status"]     = "completed"
            job["step"]       = "done"
            job["result"]     = {
                "video_filename":     saved_video_path.name,
                "video_download_url": f"/api/serve-download/{job_id}",
            }
            logger.info(f"[{job_id}] Download complete → {saved_video_path}")
        else:
            raise RuntimeError(error or "Video download failed.")

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)[:500]
        logger.error(f"[{job_id}] Download job failed: {e}")


# ---------------------------------------------------------------------------
# API Routes  (unchanged)
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/transcribe/url")
async def transcribe_url(background_tasks: BackgroundTasks, url: str = Form(...)):
    """Transcribe from Instagram Reel URL (downloads & saves video automatically)."""
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL is required.")

    job_id = str(uuid.uuid4())[:12]
    jobs[job_id] = {"status": "queued", "step": "initializing", "result": None, "error": None}

    background_tasks.add_task(process_job, job_id, "url", None, url.strip())
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/download-only")
async def download_only(background_tasks: BackgroundTasks, url: str = Form(...)):
    """
    Download an Instagram Reel video WITHOUT transcription.
    Returns a job_id to poll via /api/status/{job_id}.
    Once completed, fetch video from /api/serve-download/{job_id}.
    """
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL is required.")

    job_id = str(uuid.uuid4())[:12]
    jobs[job_id] = {
        "status":     "queued",
        "step":       "initializing",
        "result":     None,
        "error":      None,
        "video_file": None,
    }

    background_tasks.add_task(process_download_job, job_id, url.strip())
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/serve-download/{job_id}")
async def serve_download(job_id: str):
    """
    Serve the downloaded Instagram video as a file attachment.
    Only usable after /api/download-only job reaches status=completed.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]

    if job["status"] == "processing" or job["status"] == "queued":
        raise HTTPException(
            status_code=202,
            detail=f"Download still in progress. Step: {job['step']}",
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail=job.get("error", "Download failed."),
        )

    video_file = job.get("video_file")
    if not video_file or not Path(video_file).exists():
        raise HTTPException(
            status_code=404,
            detail="Video file not found on server.",
        )

    filename = job.get("result", {}).get("video_filename", Path(video_file).name)
    return FileResponse(
        path=video_file,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/transcribe/upload")
async def transcribe_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Transcribe from uploaded video or audio file."""
    ext      = Path(file.filename).suffix.lower()
    is_video = ext in ALLOWED_VIDEO_EXT
    is_audio = ext in ALLOWED_AUDIO_EXT

    if not is_video and not is_audio:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {list(ALLOWED_VIDEO_EXT | ALLOWED_AUDIO_EXT)}",
        )

    job_id    = str(uuid.uuid4())[:12]
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.0f} MB). Max: {MAX_FILE_SIZE_MB} MB.",
            )
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
        "step":   job["step"],
    }
    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]
    return response


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Serve the downloaded Instagram video as a file attachment."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {job['status']}",
        )

    video_file = job.get("video_file")
    if not video_file or not Path(video_file).exists():
        raise HTTPException(
            status_code=404,
            detail="No video file for this job. Video download is only available for URL-based jobs.",
        )

    filename = job.get("result", {}).get("video_filename", Path(video_file).name)
    return FileResponse(
        path=video_file,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
async def health():
    """Health check."""
    checks = {
        "ffmpeg":      shutil.which("ffmpeg") is not None,
        "yt_dlp":      True,                                 # Python library, always importable
        "instaloader": shutil.which("instaloader") is not None,
        "openai_key":  bool(os.getenv("OPENAI_API_KEY")),
    }
    try:
        import yt_dlp as _yt  # noqa: F401
    except ImportError:
        checks["yt_dlp"] = False

    all_ok = checks["ffmpeg"] and checks["yt_dlp"] and checks["openai_key"]
    return {"healthy": all_ok, "checks": checks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)