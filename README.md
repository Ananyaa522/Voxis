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
UI-only test without models: `ASR_DUMMY=1 uvicorn app:app --port 8000`. Needs ffmpeg installed.

## Run as the production container
```
docker build -t voxis .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx voxis           # http://localhost:7860
```

## API
`GET /api/health` · `POST /api/transcribe` (multipart `file`, `language=en|hi`)
