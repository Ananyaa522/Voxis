---
title: Voxis
sdk: docker
app_port: 7860
pinned: false
---

# Voxis

**English and Hindi speech recognition, from audio to a reviewable transcript.**

Voxis is a web application for transcribing uploaded audio clips or speech recorded in the browser. It combines language-specific wav2vec 2.0 models with a FastAPI backend and a React frontend, packaged as a Docker container for deployment.

> **Scope:** Voxis processes submitted clips; it is not a real-time streaming service. Transcripts are drafts and should be reviewed, especially for noisy recordings, accents, and code-switched speech.

## What it does

- Accepts audio uploads or browser microphone recordings.
- Lets the user select English or Hindi and routes the request to the corresponding model.
- Decodes supported audio formats to mono, 16 kHz audio with FFmpeg.
- Applies Silero voice activity detection (VAD) before recognition by default. VAD identifies likely speech regions; it does not remove noise that overlaps speech.
- Returns a transcript and processing metadata through a JSON API.
- Serves the built React application and API from one origin.

## Models and evaluation

Voxis uses separate checkpoints for the two languages. The selected English model is a pretrained checkpoint used as-is; the Hindi Vakyansh checkpoint was fine-tuned for this project on Google FLEURS `hi_in`. English fine-tuning experiments are included in the notebook, but the selected English model was chosen by evaluation rather than produced by those experiments.

The following results are reported by the project’s final evaluation workflow. Lower WER and CER are better.

| Language | Model | Evaluation set | WER | CER |
|---|---|---|---:|---:|
| English | wav2vec2-base-960h | LibriSpeech test-clean (2,620 utterances) | 3.26% | 0.96% |
| English | wav2vec2-base-960h | LibriSpeech test-other (2,939 utterances) | 8.43% | 3.11% |
| English | wav2vec2-large-960h-lv60-self, pretrained | LibriSpeech test-clean (2,620 utterances) | 1.75% | 0.50% |
| English | wav2vec2-large-960h-lv60-self, pretrained | LibriSpeech test-other (2,939 utterances) | 3.73% | 1.27% |
| English | wav2vec2-base-960h / wav2vec2-large-960h-lv60-self | AppTek call-centre sample (20 calls) | 30.05% / 18.61% | 13.94% / 8.14% |
| Hindi | Fine-tuned Vakyansh wav2vec2 | FLEURS `hi_in` test (418 clips) | 19.33% | 7.20% |
| Hindi | Whisper large-v3 comparator | FLEURS `hi_in` test (418 clips) | 28.75% | 10.82% |

The Hindi comparison is on read speech from FLEURS, the dataset used for fine-tuning, so it is an in-domain result—not a guarantee for noisy or spontaneous speech. The AppTek sample is a small cross-domain stress test. Older Whisper and NVIDIA NeMo AppTek scores are omitted because the project notes identify them as results from an earlier, non-comparable evaluation protocol.

## How it works

```text
Browser upload or recording
          |
          v
FastAPI receives audio and language
          |
          v
FFmpeg decodes to mono 16 kHz audio
          |
          v
Silero VAD (enabled by default)
          |
          v
Language-specific wav2vec 2.0 CTC model
          |
          v
Transcript and timing metadata
```

Long inputs are transcribed in overlapping chunks. During evaluation, text normalization is applied consistently to predictions and references before WER/CER scoring. The English base checkpoint is run with batch size 1 because padding shorter examples in a batch corrupted its predictions in the original evaluation setup.

## Run locally

### Requirements

- Python 3.11 recommended
- Node.js 20 or later
- FFmpeg installed and available on `PATH`
- Access to the model repository configured for the backend

### 1. Configure the backend

From the repository root, create a Python environment and install the backend dependencies:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `backend/.env` with the model repository and a Hugging Face access token that can read it:

```dotenv
ASR_HUB_REPO=Ananyaa05/asr-final
HF_TOKEN=hf_your_token_here
ASR_DEVICE=auto
ASR_VAD_ENABLED=1
```

Keep the token private. `.env` files and downloaded model weights are excluded from Git.

Start the API from the `backend` directory:

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

On first start, the backend downloads the required checkpoints if they are not already present. Model loading happens in the background; check `http://localhost:8000/api/health` and wait for `"ready": true`.

### 2. Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`). In development, Vite proxies `/api` requests to the backend on port 8000.

### Test the UI without loading ASR models

For a frontend smoke test, start the backend with dummy mode enabled:

```powershell
cd backend
$env:ASR_DUMMY = "1"
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

Dummy mode returns a fixed sample transcript. It does **not** run the ASR models or VAD and cannot be used to evaluate transcription quality.

## Run with Docker

Build and run the combined frontend/backend image:

```powershell
docker build -t voxis .
docker run --env-file backend/.env -p 7860:7860 voxis
```

Open `http://localhost:7860`. For Hugging Face Spaces, configure `HF_TOKEN` as a Space secret rather than committing it to the repository. The Docker image exposes port 7860.

## API

### `GET /api/health`

Reports API/model readiness, the selected device, supported languages, and whether VAD is enabled and ready.

### `POST /api/transcribe`

Submit `multipart/form-data` containing:

- `file`: audio file
- `language`: `en` or `hi` (defaults to `en`)

The JSON response includes `text`, `language`, `audio_seconds`, `speech_seconds`, `vad_applied`, `processing_seconds`, `rtf`, and `device`. `speech_seconds` reflects the audio retained after VAD when VAD is applied; `audio_seconds` is the original decoded clip duration.

The backend enforces a 50 MB upload limit, a 0.3-second minimum, and a 900-second maximum. Set `ASR_VAD_ENABLED=0` to disable VAD. VAD is implemented in inference, but its effect on WER/CER has not yet been established by a controlled with/without comparison.

## Fine-tuning and evaluation notebook

The Colab workflow is in [notebooks/ASR_Finetuning_Final_clean.ipynb](./notebooks/ASR_Finetuning_Final_clean.ipynb). It covers data preparation, evaluation, Hindi fine-tuning, and checkpoint export. Upload `backend/asr_common.py` to `/content` in Colab when prompted by the notebook. Use Colab Secrets for Hugging Face credentials.

## Repository layout

```text
backend/
  app.py                 FastAPI endpoints, audio decoding, VAD and inference
  asr_common.py          Shared audio, CTC inference and evaluation helpers
  requirements.txt       Python dependencies
frontend/
  src/                   React application
  package.json           Frontend scripts and dependencies
notebooks/
  ASR_Finetuning_Final_clean.ipynb
Dockerfile               Multi-stage frontend/backend image
```

Model weights, local credentials, virtual environments, and generated build output should not be committed.

## Known limitations and next steps

- Reported benchmarks are primarily read speech; real microphone performance depends on noise, accent, distance, and recording quality.
- VAD detects speech regions but does not denoise overlapping background sounds. Its effect on WER/CER still needs a controlled evaluation.
- The English model is a pretrained checkpoint used as-is; English fine-tuning on already-seen LibriSpeech data did not show a reliable gain.
- The Hindi FLEURS result is in-domain and should not be generalized to all Hindi speech conditions.
- Code-switching, punctuation restoration, broader noisy-speech evaluation, and language-model decoding are potential areas for future work.
