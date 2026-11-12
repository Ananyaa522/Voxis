---
title: Voxis
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Voxis - English & Hindi speech to text

Fine-tuned wav2vec2 (CTC) models behind a FastAPI backend with a React UI: upload an audio clip or record with the microphone and get a transcript.

```
Voxis/
  backend/     FastAPI app (app.py), shared ASR helpers (asr_common.py), requirements.txt
  frontend/    React (Vite) UI, built into frontend/dist and served by the backend
  notebooks/   Colab notebooks: evaluation harness, English study, Hindi fine-tune, export
  Dockerfile   one container = API + UI (Hugging Face Spaces, Docker SDK)
```

## Run locally (two terminals)
```
cd backend && pip install -r requirements.txt
export ASR_HUB_REPO=An05/asr-final HF_TOKEN=hf_xxx        # models download once into backend/models
uvicorn app:app --reload --port 8000

cd frontend && npm install && npm run dev                  # http://localhost:5173 (proxies /api to :8000)
```
UI-only test without ASR models: `ASR_DUMMY=1 uvicorn app:app --port 8000`. Dummy mode returns a fixed transcript and skips VAD; it does not validate audio preprocessing. Needs ffmpeg installed.

## Run as the production container
```
docker build -t voxis .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx voxis           # http://localhost:7860
```

## Fine-tuning notebook

The curated Colab workflow for English/Hindi training experiments, evaluation, and model export is in [notebooks/ASR_Finetuning_Final.ipynb](./notebooks/ASR_Finetuning_Final.ipynb). Upload `backend/asr_common.py` to `/content` in Colab before running the shared-helper cell, and provide Hugging Face credentials through Colab Secrets. Model files and notebook outputs are intentionally not committed.

The backend applies Silero VAD to uploaded audio before ASR by default. Set `ASR_VAD_ENABLED=0` to disable it. The notebook does not yet include VAD in its training/evaluation pipeline, so it must be added there separately before comparing WER/CER with and without VAD.

## API
`GET /api/health` · `POST /api/transcribe` (multipart `file`, `language=en|hi`)
