# LLM Notes & Developer Guide

## Codebase Characteristics
*   **Monolithic Logic:** All logic (API, Model Loading, Inference) is contained within `main.py`.
*   **Startup Time:** The application will take significant time to start because it downloads/loads the Stable Diffusion model into memory at the global scope before the API accepts requests.
*   **Stateful Storage:** Images are written to the local file system. In a production Docker environment, `/shared/images` must be mounted as a volume, or data will be lost on container restart.

## Important Architectural Caveats
1.  **Blocking Inference:**
    *   The `generate` function is defined as `async def`, but the model inference call `pipe(...)` is a synchronous, CPU/GPU-bound blocking operation.
    *   **Impact:** While an image is generating, the Python Event Loop is blocked. The service cannot answer `/health` or other requests until generation finishes.
    *   **Fix Strategy:** For production, inference should run in a `ThreadPoolExecutor` or a separate worker process (e.g., Celery/Redis) to keep the API responsive.

2.  **Memory Usage:**
    *   The code loads the model in `float16`. This requires roughly 4GB-6GB of VRAM (GPU) or significant RAM (CPU).
    *   If running on CPU, generation will be extremely slow (minutes per image).

## Prompt Engineering Context
*   **Style Dictionary:** The `STYLE_PROMPTS` dictionary is the primary mechanism for quality control. It prepends artistic keywords to the user's raw input.
*   **Negative Prompts:** A heavy `DEFAULT_NEGATIVE_PROMPT` is hardcoded. This effectively filters out "bad anatomy" and "blurry" results common in SD v1.5.

## Docker/Infrastructure Notes
*   **Cache:** The Dockerfile creates `/data/model_cache`, but the code doesn't explicitly set the `HF_HOME` environment variable to use it. Hugging Face libraries usually default to `~/.cache/huggingface`. This might result in re-downloading models if the home directory isn't persisted.
*   **Port:** Service runs on **8001**, not the standard 8000 or 80.

## Potential Refactoring Targets
*   **Async/Await:** Wrap the `pipe()` call in `fastapi.concurrency.run_in_threadpool`.
*   **Config Management:** Move `model_id` and paths to environment variables.
*   **Input Validation:** `style` in `ImageRequest` defaults to "fantasy" but does not strictly validate against keys in `STYLE_PROMPTS` (though `dict.get` handles the fallback gracefully).