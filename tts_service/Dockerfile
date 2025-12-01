# ./tts_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    g++ \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install requirements.txt first (your FastAPI, etc.)
COPY requirements.txt .
RUN sed -i '/chatterbox-tts/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install ALL dependencies from chatterbox's pyproject.toml (except spacy-pkuseg)
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<1.26.0" \
    "librosa==0.11.0" \
    "s3tokenizer" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    "transformers==4.46.3" \
    "diffusers==0.29.0" \
    "resemble-perth==1.0.1" \
    "conformer==0.3.2" \
    "safetensors==0.5.3" \
    "pykakasi==2.3.0" \
    "gradio==5.44.1"

# Install russian-text-stresser from git
RUN pip install --no-cache-dir git+https://github.com/Vuizur/add-stress-to-epub

# NOW install chatterbox without dependencies
RUN pip install --no-cache-dir --no-deps chatterbox-tts>=0.1.2

# Copy app directory with package structure
COPY . /app/tts_service

# Set the Python path to include the parent directory of tts_service
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Create directories
RUN mkdir -p /data/tts_cache /shared/audio /shared/reference_voices

EXPOSE 8002

CMD ["uvicorn", "tts_service.main:app", "--host", "0.0.0.0", "--port", "8002"]