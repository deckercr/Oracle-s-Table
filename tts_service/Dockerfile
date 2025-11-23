# ./tts_service/Dockerfile

# tts_service/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing and Coqui TTS
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create cache directories
RUN mkdir -p /data/tts_cache /shared/audio

# Set environment variables for Coqui TTS
ENV COQUI_TTS_CACHE=/data/tts_cache
ENV TTS_CACHE=/data/tts_cache

EXPOSE 8002

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]