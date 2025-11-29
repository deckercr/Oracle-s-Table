# Project Context: D&D TTS Service (Chatterbox)

## Overview
This project is a Text-to-Speech (TTS) microservice designed for Dungeons & Dragons sessions. It utilizes the **Chatterbox TTS** model to generate speech from text, supporting both zero-shot voice cloning (using reference audio) and standard synthesis. The service is built with **FastAPI**.

## Core Functionality
1.  **Text-to-Speech Generation**: Converts text input into `.wav` audio.
2.  **Streaming Support**: Implements Server-Sent Events (SSE) to stream audio sentence-by-sentence for low-latency playback.
3.  **Voice Cloning**: accepts reference audio files (uploaded to a shared directory) to mimic specific voices.
4.  **Text Preprocessing**: Cleans markdown and formatting artifacts common in LLM-generated text (e.g., removing asterisks, underscores, HTML tags).

## Tech Stack
*   **Language**: Python 3.10
*   **Web Framework**: FastAPI / Uvicorn
*   **ML Engine**: PyTorch / Torchaudio
*   **TTS Model**: Chatterbox TTS (HuggingFace)
*   **Infrastructure**: Docker (Debian-based slim image)
*   **Audio Processing**: FFmpeg, Espeak-NG, Libsndfile

## Operational Context
*   **Environment**: Designed to run on CUDA (Nvidia GPU) for speed, with CPU fallback.
*   **Storage**: Relies on a shared volume system (`/shared/audio` for outputs, `/shared/reference_voices` for inputs).
*   **Network**: Exposes HTTP on port **8002**.

## Key Workflows
1.  **Startup**: The application downloads/loads the Chatterbox model into memory immediately upon container start.
2.  **Standard `/speak`**: Generates the full audio file, saves it to disk, and returns a URL.
3.  **Streaming `/speak`**: Splits text into sentences, generates audio for each sentence, and yields JSON-wrapped Base64 audio chunks via SSE.