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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        # Common on Windows when an executable (ffmpeg/yt-dlp/instaloader) isn't on PATH.
        raise RuntimeError(
            f"Executable not found: {e.filename}. Ensure it is installed and on your PATH. "
            "If you're using the bundled version, install the required Python packages: "
            "pip install -r requirements.txt"
        ) from e

    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(result.stderr)
    return result


def get_ffmpeg_executable() -> str:
    """Return a usable ffmpeg executable path.

    The application needs ffmpeg for audio extraction and other operations.
    Prefer a system-installed ffmpeg, but fall back to the bundled ffmpeg from
    the `imageio-ffmpeg` package if available.
    """
    # 1) Prefer system ffmpeg in PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 2) Fallback to imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            return str(ffmpeg_exe)
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg executable not found. Install ffmpeg and ensure it is on your PATH, "
        "or install the Python package 'imageio-ffmpeg' (e.g. pip install imageio-ffmpeg)."
    )


def _find_video_in_dir(directory: Path) -> Optional[Path]:
    """Return the first video file found inside *directory* (recursively), or None."""
    return next(
        (f for f in directory.rglob("*.*") if f.suffix.lower() in ALLOWED_VIDEO_EXT),
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

    try:
        ffmpeg_path = get_ffmpeg_executable()
        ydl_opts["ffmpeg_location"] = ffmpeg_path
    except Exception as e:
        logger.warning(f"ffmpeg not found; yt-dlp may still work for some URLs (error: {e})")

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
    """Fallback: use Instaloader to download a single public Instagram post/reel.

    Returns the downloaded video Path, or None on failure.
    """
    match = re.search(r"/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", url)
    if not match:
        logger.error(f"Cannot parse Instagram shortcode from URL: {url}")
        return None

    shortcode = match.group(1)
    logger.info(f"instaloader shortcode: {shortcode}")

    # Prefer using the installed `instaloader` CLI; fall back to the Python module if
    # the CLI is not available.
    try:
        cmd = [
            "instaloader",
            "--no-captions",
            "--no-metadata-json",
            "--no-compress-json",
            "--dirname-pattern", str(tmp_dir),
            "--filename-pattern", shortcode,
            "--", shortcode,
        ]
        run_cmd(cmd, timeout=180)
    except FileNotFoundError:
        logger.warning("'instaloader' executable not found. Falling back to Python instaloader module.")
        try:
            import instaloader

            loader = instaloader.Instaloader(
                dirname_pattern=str(tmp_dir),
                filename_pattern=shortcode,
                download_comments=False,
                save_metadata=False,
                post_metadata_txt_pattern="",
            )
            loader.context.log.setLevel(logging.ERROR)

            post = instaloader.Post.from_shortcode(loader.context, shortcode)
            loader.download_post(post, target=str(tmp_dir))
        except Exception as e:
            logger.error(f"instaloader (module) failed: {e}")
            return None
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
    ffmpeg_exe = get_ffmpeg_executable()
    cmd = [
        ffmpeg_exe, "-y",
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
        ffmpeg_exe = get_ffmpeg_executable()
        run_cmd([
            ffmpeg_exe, "-y", "-i", str(audio_path),
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
            ffmpeg_exe = get_ffmpeg_executable()
            run_cmd([
                ffmpeg_exe, "-y", "-i", str(audio_path),
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
# Story Generation Pipeline
# ---------------------------------------------------------------------------

# Word targets per explicit duration — based on ~150 words/min natural speaking pace
STORY_WORD_TARGETS = {30: 75, 45: 115, 60: 155}

# Speaking pace used for auto-duration calculation (words per second)
WORDS_PER_SECOND = 150 / 60   # ≈ 2.5

STORY_SYSTEM_PROMPT = (
    "You are a creative social media story writer who specialises in Instagram and WhatsApp Stories. "
    "You write punchy, engaging, first-person narrative scripts that sound natural when spoken aloud. "
    "Your stories hook the viewer in the first sentence and leave a strong impression at the end. "
    "Respond ONLY with the story text — no labels, headers, explanations, or markdown."
)


def resolve_story_word_count(duration_seconds: int, actual_video_duration: Optional[float]) -> tuple[int, int]:
    """
    Return (target_word_count, effective_duration_seconds) for story generation.

    Rules:
      • duration_seconds == 0  →  AUTO mode: derive word count from actual video length.
        Word count = round(actual_video_duration * WORDS_PER_SECOND), capped at 400 words.
        If actual_video_duration is unavailable, fall back to 30 s target.
      • duration_seconds in {30, 45, 60}  →  use the fixed lookup table.
    """
    if duration_seconds == 0:
        # Auto mode — match actual video duration
        if actual_video_duration and actual_video_duration > 0:
            target_words   = min(round(actual_video_duration * WORDS_PER_SECOND), 400)
            eff_duration   = round(actual_video_duration)
        else:
            # Fallback: no duration info → 30 s default
            target_words   = STORY_WORD_TARGETS[30]
            eff_duration   = 30
        return target_words, eff_duration

    # Explicit duration chosen by user
    return STORY_WORD_TARGETS.get(duration_seconds, STORY_WORD_TARGETS[30]), duration_seconds


def generate_story_with_gpt(
    transcription: str,
    duration_seconds: int,
    actual_video_duration: Optional[float] = None,
) -> tuple[str, int]:
    """
    Use OpenAI GPT to craft a Story script from *transcription*.

    Args:
        transcription         : Verbatim speech text from Whisper.
        duration_seconds      : 0 = auto (use actual video length), or 30 / 45 / 60.
        actual_video_duration : Whisper-reported video duration in seconds (used in auto mode).

    Returns:
        (story_text, effective_duration_seconds)
    """
    client                        = get_openai_client()
    target_words, eff_duration    = resolve_story_word_count(duration_seconds, actual_video_duration)

    if duration_seconds == 0:
        duration_label = f"match the video length (~{eff_duration} seconds)"
    else:
        duration_label = f"fit a {eff_duration}-second Story"

    user_prompt = (
        f"Below is a verbatim transcription of an Instagram Reel.\n\n"
        f"---\n{transcription}\n---\n\n"
        f"Write a captivating Story script based on this content. "
        f"The script must be EXACTLY around {target_words} words so it can {duration_label} "
        f"when read aloud at a natural pace. "
        f"Make it engaging, punchy, and ready to post as-is."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": STORY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip(), eff_duration


async def process_story_job(job_id: str, url: str, duration_seconds: int):
    """
    Background pipeline for Instagram URL Story generation.
      1. Download Instagram video
      2. Extract audio & transcribe (Whisper)
      3. Generate Story script (GPT-4o-mini)

    duration_seconds == 0  →  AUTO: story length matches actual video duration.
    duration_seconds in {30,45,60}  →  explicit target chosen by user.
    """
    job     = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp(prefix="story_"))

    try:
        job["status"] = "processing"

        # ── Step 1: Download ──────────────────────────────────────────────
        job["step"] = "downloading"
        logger.info(f"[{job_id}] Story job — downloading: {url}")
        saved_video_path, dl_error = download_instagram_video(url)

        if not saved_video_path:
            raise RuntimeError(dl_error or "Video download failed.")

        job["video_file"] = str(saved_video_path)

        # ── Step 2: Extract audio + Transcribe ───────────────────────────
        job["step"] = "extracting_audio"
        audio_path  = extract_audio(saved_video_path, tmp_dir)

        job["step"] = "transcribing"
        logger.info(f"[{job_id}] Transcribing for story...")
        whisper_result        = transcribe_with_whisper(audio_path)
        transcription         = whisper_result["text"]
        actual_video_duration = whisper_result.get("duration")   # float seconds from Whisper

        if not transcription.strip():
            raise RuntimeError("Transcription returned empty — cannot generate story.")

        # ── Step 3: Generate Story ─────────────────────────────────────────
        mode_label = "auto" if duration_seconds == 0 else f"{duration_seconds}s"
        job["step"] = "generating_story"
        logger.info(f"[{job_id}] Generating story (mode={mode_label}, actual={actual_video_duration}s)...")

        story, eff_duration = generate_story_with_gpt(
            transcription,
            duration_seconds,
            actual_video_duration,
        )

        job["status"] = "completed"
        job["step"]   = "done"
        job["result"] = {
            "story":                  story,
            "duration_seconds":       eff_duration,
            "duration_mode":          "auto" if duration_seconds == 0 else "manual",
            "word_count":             len(story.split()),
            "source_transcription":   transcription,
        }
        logger.info(f"[{job_id}] Story complete — {len(story.split())} words, ~{eff_duration}s.")

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)[:500]
        logger.error(f"[{job_id}] Story job failed: {e}")
    finally:
        cleanup_files(tmp_dir)


# ---------------------------------------------------------------------------
# Story Generation Pipeline — Uploaded File (video or audio)
# ---------------------------------------------------------------------------
async def process_story_upload_job(
    job_id: str,
    source_path: str,
    source_type: str,          # "video" | "audio"
    duration_seconds: int,
):
    """
    Background pipeline for Story generation from an uploaded file.
    No download step — file is already saved on disk.

      Video path:  extract_audio → transcribe (Whisper) → generate_story (GPT)
      Audio path:  optimise_audio → transcribe (Whisper) → generate_story (GPT)
    """
    job     = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp(prefix="story_up_"))

    try:
        job["status"] = "processing"
        ffmpeg_exe    = get_ffmpeg_executable()

        # ── Step 1: Get audio ─────────────────────────────────────────────
        if source_type == "video":
            job["step"] = "extracting_audio"
            logger.info(f"[{job_id}] Story-upload: extracting audio from video")
            audio_path = extract_audio(Path(source_path), tmp_dir)

        else:  # audio
            job["step"] = "optimising_audio"
            logger.info(f"[{job_id}] Story-upload: optimising audio file")
            optimal = tmp_dir / "optimised.wav"
            run_cmd([
                ffmpeg_exe, "-y", "-i", source_path,
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(optimal),
            ])
            audio_path = optimal

        if not audio_path.exists():
            raise RuntimeError("Audio preparation failed — file not found after ffmpeg.")

        # ── Step 2: Transcribe ─────────────────────────────────────────────
        job["step"] = "transcribing"
        logger.info(f"[{job_id}] Story-upload: transcribing...")
        whisper_result        = transcribe_with_whisper(audio_path)
        transcription         = whisper_result["text"]
        actual_media_duration = whisper_result.get("duration")   # float seconds from Whisper

        if not transcription.strip():
            raise RuntimeError("Transcription returned empty — cannot generate story.")

        # ── Step 3: Generate Story ─────────────────────────────────────────
        job["step"] = "generating_story"
        logger.info(f"[{job_id}] Story-upload: generating {duration_seconds}s story...")
        story, eff_duration = generate_story_with_gpt(
            transcription,
            duration_seconds,
            actual_media_duration,   # passed through for consistent word-count calc
        )

        job["status"] = "completed"
        job["step"]   = "done"
        job["result"] = {
            "story":                story,
            "duration_seconds":     eff_duration,
            "duration_mode":        "manual",
            "word_count":           len(story.split()),
            "source_transcription": transcription,
        }
        logger.info(f"[{job_id}] Story-upload complete — {len(story.split())} words.")

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)[:500]
        logger.error(f"[{job_id}] Story-upload job failed: {e}")
    finally:
        cleanup_files(tmp_dir)
        cleanup_files(source_path)   # remove uploaded file after processing


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


@app.post("/api/generate-story")
async def generate_story(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    duration_seconds: int = Form(0),
):
    """
    Generate a social-media Story script from an Instagram Reel URL.

    Steps (all in background):
      1. Download the Reel
      2. Transcribe audio (Whisper)
      3. Generate Story text (GPT-4o-mini)

    Args:
        url              : Instagram Reel / Post URL
        duration_seconds : 0 = AUTO (story matches actual video length) [DEFAULT]
                           30 / 45 / 60 = explicit target chosen by user

    Poll /api/status/{job_id} until status == "completed".
    Result keys: story, duration_seconds, duration_mode, word_count, source_transcription
    """
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL is required.")

    if duration_seconds not in (0, 30, 45, 60):
        raise HTTPException(
            status_code=400,
            detail="duration_seconds must be 0 (auto), 30, 45, or 60.",
        )

    job_id = str(uuid.uuid4())[:12]
    jobs[job_id] = {
        "status":     "queued",
        "step":       "initializing",
        "result":     None,
        "error":      None,
        "video_file": None,
        "type":       "story",
    }

    background_tasks.add_task(process_story_job, job_id, url.strip(), duration_seconds)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/generate-story/upload")
async def generate_story_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    duration_seconds: int = Form(30),
):
    """
    Generate a Story script from an uploaded video or audio file.

    Steps (all in background):
      1. Save uploaded file
      2. Extract / optimise audio (FFmpeg)
      3. Transcribe (Whisper)
      4. Generate Story (GPT-4o-mini)

    Args:
        file             : Uploaded video (mp4/mov/mkv/webm/avi) or audio (mp3/wav/m4a/ogg/flac/aac)
        duration_seconds : Target story length — 30, 45, or 60

    Poll /api/status/{job_id} until status == "completed".
    Result keys: story, duration_seconds, word_count, source_transcription
    """
    ext      = Path(file.filename).suffix.lower()
    is_video = ext in ALLOWED_VIDEO_EXT
    is_audio = ext in ALLOWED_AUDIO_EXT

    if not is_video and not is_audio:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {list(ALLOWED_VIDEO_EXT | ALLOWED_AUDIO_EXT)}",
        )

    if duration_seconds not in (30, 45, 60):
        raise HTTPException(
            status_code=400,
            detail="duration_seconds must be 30, 45, or 60.",
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
    jobs[job_id] = {
        "status": "queued",
        "step":   "initializing",
        "result": None,
        "error":  None,
        "type":   "story_upload",
    }

    background_tasks.add_task(
        process_story_upload_job,
        job_id,
        str(save_path),
        source_type,
        duration_seconds,
    )
    return {"job_id": job_id, "status": "queued"}


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
        "ffmpeg":      True,
        "yt_dlp":      True,  # Python library, always importable if requirements installed
        "instaloader": True,
        "openai_key":  bool(os.getenv("OPENAI_API_KEY")),
    }

    try:
        get_ffmpeg_executable()
    except Exception:
        checks["ffmpeg"] = False

    try:
        import yt_dlp as _yt  # noqa: F401
    except ImportError:
        checks["yt_dlp"] = False

    try:
        import instaloader  # noqa: F401
    except ImportError:
        checks["instaloader"] = False

    all_ok = checks["ffmpeg"] and checks["yt_dlp"] and checks["openai_key"]
    return {"healthy": all_ok, "checks": checks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)