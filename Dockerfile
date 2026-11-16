# ---------- stage 1: build the React UI ----------
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY frontend/ ./
RUN npm run build

# ---------- stage 2: Python API + built UI ----------
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user            # Hugging Face Spaces runs containers as uid 1000
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only torch keeps the image small. For a GPU Space build with:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url $TORCH_INDEX
COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user backend/ backend/
COPY --chown=user --from=ui /ui/dist frontend/dist

ENV ASR_HUB_REPO=Ananyaa05/asr-final ASR_MODELS_DIR=/home/user/app/models ASR_DEVICE=auto
WORKDIR $HOME/app/backend
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
