from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import asr_common as ac
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

log = logging.getLogger("voxis")
logging.basicConfig(level=logging.INFO)

BASE = Path(__file__).parent
MODELS_DIR = Path(os.getenv("ASR_MODELS_DIR", BASE / "models"))
DIST = Path(os.getenv("VOXIS_FRONTEND_DIR", BASE.parent / "frontend" / "dist"))
HUB_REPO = os.getenv("ASR_HUB_REPO", "")
MAX_UPLOAD_MB = int(os.getenv("ASR_MAX_UPLOAD_MB", "50"))
MAX_SECONDS = int(os.getenv("ASR_MAX_SECONDS", "900"))
DUMMY = os.getenv("ASR_DUMMY") == "1"
LANGS = {"en": "English", "hi": "Hindi"}

PIPES: dict = {}
LOCKS = {k: threading.Lock() for k in LANGS}          # a pipeline is not thread-safe
STATE = {"ready": False, "error": None, "device": "cpu"}


def ensure_models() -> None:
    if all((MODELS_DIR / k / "config.json").exists() for k in LANGS):
        return
    if not HUB_REPO:
        raise RuntimeError(f"Models not found in {MODELS_DIR}. Set ASR_HUB_REPO (and HF_TOKEN).")
    from huggingface_hub import snapshot_download
    log.info("Downloading models from %s ...", HUB_REPO)
    snapshot_download(HUB_REPO, local_dir=MODELS_DIR, token=os.getenv("HF_TOKEN"))


def load_models() -> None:
    """Runs in a background thread so the web server answers /api/health immediately."""
    try:
        import torch
        ensure_models()
        use_gpu = torch.cuda.is_available() and os.getenv("ASR_DEVICE", "auto") != "cpu"
        device, STATE["device"] = (0, "cuda") if use_gpu else (-1, "cpu")
        for lang in LANGS:
            log.info("Loading %s model on %s ...", lang, STATE["device"])
            PIPES[lang] = ac.load_ctc_pipeline(str(MODELS_DIR / lang), device=device, batch_size=8)
            ac.transcribe_ctc(PIPES[lang], np.zeros(16000, dtype=np.float32))   # warm-up
        STATE["ready"] = True
        log.info("All models ready.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Model loading failed")
        STATE["error"] = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DUMMY:
        STATE["ready"] = True
    else:
        threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Voxis ASR", lifespan=lifespan)
api = APIRouter(prefix="/api")


def decode_audio(raw: bytes, filename: str) -> np.ndarray:
    """Any audio container -> mono float32 @ 16 kHz via ffmpeg (temp file: mp4/m4a need seeking)."""
    suffix = Path(filename or "").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le",
                            "-ac", "1", "-ar", "16000", "pipe:1"], capture_output=True, timeout=180)
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg is not installed on the server.")
    finally:
        os.unlink(path)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, "Could not decode this audio file.")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def prettify(text: str, lang: str) -> str:
    """Cosmetic only: the English model emits UPPERCASE without punctuation."""
    text = " ".join(text.split())
    if lang == "en" and text:
        text = re.sub(r"\bi\b", "I", text.lower())
        text = text[0].upper() + text[1:]
    return text


@api.get("/health")
def health():
    return {"status": "ok", "ready": STATE["ready"], "error": STATE["error"],
            "device": STATE["device"], "languages": LANGS}


@api.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: str = Form("en")):
    if STATE["error"]:
        raise HTTPException(500, f"Models failed to load: {STATE['error']}")
    if not STATE["ready"]:
        raise HTTPException(503, "Models are still loading, please retry in a minute.")
    if language not in LANGS:
        raise HTTPException(400, f"language must be one of {list(LANGS)}")
    raw = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB.")
    if not raw:
        raise HTTPException(400, "Empty file.")
    audio = decode_audio(raw, file.filename)
    duration = len(audio) / ac.TARGET_SR
    if duration < 0.3:
        raise HTTPException(400, "Audio is too short.")
    if duration > MAX_SECONDS:
        raise HTTPException(413, f"Audio longer than {MAX_SECONDS} s.")
    t0 = time.time()
    if DUMMY:
        text = "dummy transcript for UI testing"
    else:
        with LOCKS[language]:
            text = ac.transcribe_ctc(PIPES[language], audio)
    took = time.time() - t0
    return {"language": language, "text": prettify(text, language), "audio_seconds": round(duration, 2),
            "processing_seconds": round(took, 2), "rtf": round(took / duration, 3), "device": STATE["device"]}


app.include_router(api)

# The built React app is served from "/" (same origin as the API -> no CORS). Must be mounted LAST.
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_ui():
        return {"message": "Voxis API is running. Frontend not built: run `npm run build` in frontend/."}
