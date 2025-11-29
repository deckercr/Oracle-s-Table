# Module Map

## Directory: ./llm_service

### 1. `main.py`
The entry point for the FastAPI application. It handles initialization, request routing, and orchestration.

*   **Initialization**
    *   `app`: FastAPI instance.
    *   `pipe`: Loads the Llama 3.2 text-generation pipeline (GPU/CPU config).
    *   `logger`: Service-level logging.
*   **Helper Functions**
    *   `check_service_health(url)`: Pings internal microservices (Image/TTS).
    *   `get_dnd_context(query)`: Fetches specific rule data from `dnd5eapi.co`.
    *   `build_context_prompt(campaign_id, user_prompt)`: Aggregates DB data (history, story, chars) into a string for the LLM.
*   **Endpoints**
    *   `POST /dm_turn`: **Core Loop**. Receives player input, builds context, prompts LLM, saves result, and triggers media generation.
    *   `POST /campaign/create`: Creates a new campaign.
    *   `GET /campaign/{id}`: Retrieves campaign metadata and summaries.
    *   `GET /campaign/{id}/history`: Retrieves raw conversation logs.
    *   `GET /campaign/{id}/story`: Retrieves narrative story segments.
    *   `DELETE /campaign/{id}/history`: Wipes context.
    *   `GET /health` & `GET /`: Service status checks.

### 2. `db_helper.py`
A database abstraction layer managing connections and SQL operations.

*   **Class: `DatabaseManager`**
    *   **Connection Management**
        *   `__init__`: Configures `SimpleConnectionPool` based on env vars.
        *   `get_connection()`: Context manager for yielding cursors and handling commits/rollbacks.
        *   `test_connection()`: Verifies DB connectivity.
    *   **Campaign Operations**
        *   `create_campaign`, `get_active_campaign`, `get_campaign_by_id`.
    *   **Conversation History**
        *   `save_conversation`: Logs prompt, response, and metadata (tokens, media flags).
        *   `get_recent_conversations`: Fetches raw rows.
        *   `get_conversation_context`: Formats history specifically for LLM prompt injection.
    *   **Story Segments**
        *   `save_story_segment`: Saves high-level narrative summaries.
        *   `get_story_so_far`: Retrieves recent narrative context.
    *   **World Building (CRUD)**
        *   `add_character`, `get_active_characters`.
        *   `add_location`, `visit_location`.
        *   `add_quest`, `complete_quest`, `get_active_quests`.
    *   **Utility**
        *   `get_campaign_summary`, `clear_campaign_history`.

### 3. `Dockerfile`
Configuration for building the service image.
*   Base: `python:3.10-slim`.
*   Installs system dependencies (`curl`).
*   Sets up `/data/model_cache` for persistent model storage.
*   Exposes port `8000`.

### 4. `requirements.txt`
Python dependencies.
*   **Web/API:** `fastapi`, `uvicorn`, `requests`.
*   **ML/AI:** `transformers`, `torch`, `accelerate`.
*   **Database:** `psycopg2-binary`.