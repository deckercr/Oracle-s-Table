# LLM & Developer Notes

## Prompt Engineering Strategy
The service uses a specific prompt construction strategy found in `main.py` -> `dm_turn`:

1.  **System Prompt:** Defines the persona (Dungeon Master), brevity constraints (2-4 sentences), and tone.
2.  **Context Injection:**
    *   **Campaign Info:** Name and DM style.
    *   **Story Summary:** Last 3 narrative segments.
    *   **Conversation History:** Last 3 turns (User/Assistant pairs).
    *   **Active Entities:** Characters and Quests currently active in the DB.
    *   **Rules:** JSON data fetched from the 5e API based on keywords in the prompt.
3.  **Format:** Uses Llama 3 specific special tokens: `<|begin_of_text|>`, `<|start_header_id|>`, etc.

## Data Flow & State
*   **Stateful Database, Stateless Service:** The Python service itself is stateless, but the "Context Window" is manually reconstructed from Postgres for every request.
*   **Latency Considerations:**
    *   Every `dm_turn` request involves:
        1.  DB Reads (Campaign, History, Story, Chars, Quests).
        2.  External HTTP Request (D&D API).
        3.  LLM Inference (Heaviest operation).
        4.  DB Writes (Conversation log, optional Story segment).
        5.  *Async/Parallel* HTTP Requests to Image/TTS services (non-blocking for text return, but logic is sequential in code).

## Technical Implementation Details
*   **Model Caching:** The `Dockerfile` creates `/data/model_cache` and maps it to `HF_HOME`. This is critical to prevent re-downloading the 3B model on every container restart.
*   **Database Pooling:** `SimpleConnectionPool` is used (min 1, max 10). This is essential because `dm_turn` opens multiple context managers.
*   **Error Handling:**
    *   If `image_gen` or `tts_voice` are down, the service logs the error but still returns the text response to the user (`check_service_health` checks).
    *   If the DB is down, the service starts but enters a "limited functionality" mode (warned in startup logs).

## Potential Gotchas
1.  **Memory Usage:** Loading `Llama-3.2-3B-Instruct` requires decent RAM/VRAM. The `low_cpu_mem_usage=True` and `torch.float16` flags in `main.py` are critical optimization attempts.
2.  **Sentence Parsing:** The code manually cleans up incomplete sentences in `dm_turn` by looking for the last punctuation mark. This assumes the model might hit `max_new_tokens` (256) mid-sentence.
3.  **Context Window Limits:** The context builder (`build_context_prompt`) limits history to 5 turns and story to 3 segments. If the prompts become very long, they might exceed the model's context window, though Llama 3.2 has a generous window, 256 output tokens is relatively short.