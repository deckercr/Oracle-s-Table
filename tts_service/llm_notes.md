# LLM Notes & Developer Guide

## Dependency Management "Gotchas"
*   **Manual Dependency Resolution**: The `Dockerfile` does **not** simply install `requirements.txt`. It performs a specific dance to install `chatterbox-tts`.
    *   It intentionally excludes `chatterbox-tts` from the initial pip install.
    *   It manually installs specific versions of `numpy`, `torch`, `transformers`, and `diffusers` before installing `chatterbox-tts` with the `--no-deps` flag.
    *   **Reason**: Automatic dependency resolution for this specific TTS model often pulls in conflicting versions of CUDA libraries or NumPy. Any changes to dependencies should be made in the `Dockerfile` run commands, not just `requirements.txt`.

## Streaming Implementation Details
*   **Protocol**: The streaming implementation is **not** a raw byte stream. It uses **Server-Sent Events (SSE)**.
*   **Data Format**: Each event yields a JSON object.
    *   Metadata events contain text progress.
    *   Audio events contain a key `"audio_data"` with a **Base64 encoded** string of the WAV bytes.
*   **Client Handling**: A standard HTML5 `<audio src="...">` tag **cannot** play this stream directly. The client must:
    1.  Open an `EventSource`.
    2.  Decode Base64 chunks.
    3.  Feed them into an AudioContext or queue them for playback.

## Audio Generation Logic
*   **Sentence Splitting**: For streaming, text is split strictly by punctuation (`.!?`).
*   **Truncation**: To prevent memory overflow or timeouts, text input is hard-truncated at 1000 characters.
*   **Reference Voices**: The system looks for voice cloning files in `/shared/reference_voices`. These must be placed there by an external process or volume mount; there is no upload endpoint in this specific service.

## Performance Considerations
*   **Model Loading**: The model is loaded into the global scope on startup. This means the container startup time is slow (model download + load), but inference is faster.
*   **Concurrency**: PyTorch models on GPU are generally not thread-safe for concurrent inference. While FastAPI is async, the actual generation (`model.generate`) is synchronous/blocking code. Heavy concurrent load might require a semaphore or queue system not currently implemented here.