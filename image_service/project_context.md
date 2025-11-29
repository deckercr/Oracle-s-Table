# Project Context: Image Generation Service

## Overview
The **Image Generation Service** is a Python-based microservice designed to generate images from text prompts using the Stable Diffusion v1.5 model. It exposes a REST API via FastAPI and is containerized using Docker.

The service handles prompt engineering, style application, model inference (supporting both CPU and GPU), and local file storage for generated assets.

## Architecture
*   **Service Type:** REST API (FastAPI).
*   **Containerization:** Docker (Python 3.10 Slim).
*   **ML Backend:** PyTorch + Diffusers (Hugging Face).
*   **Model:** `runwayml/stable-diffusion-v1-5`.
*   **Storage:** Local file system (writes to `/shared/images`).

## Tech Stack
*   **Language:** Python 3.10
*   **Web Framework:** FastAPI / Uvicorn
*   **ML Framework:** PyTorch, Diffusers, Transformers, Accelerate
*   **Infrastructure:** Docker

## Key Features
1.  **Text-to-Image:** Generates 512x512 images based on user descriptions.
2.  **Style Presets:** Includes predefined prompt templates (Fantasy, Dark, Epic, Cinematic, Painterly) to enhance output quality without complex user input.
3.  **Hardware Awareness:** Automatically detects CUDA capability; runs on GPU if available, falls back to CPU (with logging).
4.  **Static Serving:** Endpoints to retrieve generated images and list available styles.
5.  **Health Monitoring:** Simple health check endpoint exposing GPU status.

## Configuration & Environment
*   **Port:** Exposes port `8001`.
*   **Volume Mounts:**
    *   `/shared/images`: Destination for generated PNG files.
    *   `/data/model_cache`: Created in Dockerfile (implied usage for HuggingFace cache).

## Data Flow
1.  Client sends POST request with description and style.
2.  Server constructs full prompt using style templates and negative prompts.
3.  Model generates image (blocking operation).
4.  Image is saved to disk with a UUID.
5.  Server returns metadata and a URL to retrieve the image.