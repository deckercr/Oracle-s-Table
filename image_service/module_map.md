# Module Map

## Directory: ./image_service

### 1. `main.py`
The entry point for the FastAPI application. It handles model initialization, image generation logic, and static file serving.

*   **Initialization**
    *   `app`: FastAPI instance.
    *   `output_dir`: Sets up the `/shared/images` directory for local storage.
    *   `pipe`: Loads the Stable Diffusion v1.5 pipeline (`runwayml/stable-diffusion-v1-5`) and configures hardware (GPU/CPU).
    *   `STYLE_PROMPTS`: Dictionary mapping short style keys (e.g., "fantasy") to detailed artistic prompt suffixes.
    *   `DEFAULT_NEGATIVE_PROMPT`: A predefined string of negative embeddings to filter out common quality issues.
*   **Data Models**
    *   `class ImageRequest(BaseModel)`: Defines the JSON payload for generation (description, style, optional negative prompt).
*   **Endpoints**
    *   `POST /generate_image`: **Core Functionality**. Creates unique ID, merges user prompt with style templates, runs inference (50 steps), and saves the resulting PNG.
    *   `GET /images/{image_id}`: Retrieves the generated image file from the disk.
    *   `GET /styles`: Lists available art styles and their human-readable descriptions.
    *   `GET /health`: Service status check returning overall health and GPU availability.

### 2. `Dockerfile`
Configuration for building the service image.
*   Base: `python:3.10-slim`.
*   Installs system dependencies (`curl`).
*   Creates `/data/model_cache` and `/shared/images` directories.
*   Exposes port `8001`.
*   Entrypoint: `uvicorn`.

### 3. `requirements.txt`
Python dependencies.
*   **Web/API:** `fastapi`, `uvicorn`.
*   **ML/AI:** `diffusers`, `transformers`, `accelerate`, `torch`.