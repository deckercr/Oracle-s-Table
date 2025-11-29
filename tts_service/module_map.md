# Module Map

## Root Directory (`./tts_service/`)

### Application Logic
*   **`main.py`**: The core entry point.
    *   Initializes the `ChatterboxTTS` model.
    *   Defines FastAPI endpoints (`/speak`, `/voices`, `/health`).
    *   Contains the **Streaming Logic** (`stream_audio_generator`) which yields JSON-encoded SSE events.
    *   Contains **Text Cleaning** logic (`clean_text_for_tts`).

### Configuration & Infrastructure
*   **`Dockerfile`**: 
    *   Multi-step build process.
    *   **Crucial Note**: Manually handles complex dependency constraints for `chatterbox-tts` (numpy, transformers, diffusers) to prevent version conflicts.
    *   Installs system dependencies (`ffmpeg`, `espeak-ng`).
*   **`requirements.txt`**: 
    *   Lists base Python dependencies (FastAPI, Uvicorn, standard audio libs).
    *   *Note*: The Dockerfile modifies how these are installed to accommodate the specific needs of the ML models.