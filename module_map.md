# Master Module Map - D&D AI Game System

## System Overview

A microservices-based AI Dungeon Master system that orchestrates text generation, image generation, text-to-speech, and persistent world state management through containerized services.

**Core Technologies:** FastAPI, PostgreSQL (pgvector), Llama 3.2, Stable Diffusion, ChatterboxTTS, Django

**Network:** `dm_network` (172.25.0.0/16 bridge)

---

## Service Architecture

### 1. **Database Service** (`database`)
**Container:** `dnd_postgres` | **Port:** 5432 | **Image:** `pgvector/pgvector:pg16`

The persistent data layer storing campaign state, conversation history, world entities, and vector embeddings.

#### Key Components
- **`init_db.sql`**: Initialization script that runs on first container start
  - **Extensions**: Enables `pgvector` for 384-dimensional embedding storage
  - **User Management**: Creates `llama_user` (configurable via env vars) with schema permissions
  - **Core Tables**:
    - `campaigns`: Game instances with DM style and active status
    - `conversations`: Raw user/AI logs with token counts and media flags
    - `story_segments`: Summarized narrative beats for context management
    - `story_embeddings`: Vector storage with `ivfflat` index for cosine similarity search
  - **World Building Tables**:
    - `characters`: NPCs/PCs with stats, race, class, status
    - `locations`: Discovery tracking and visit counts
    - `quests`: Objectives, rewards, completion status
  - **Views**:
    - `recent_conversations`: Readable campaign/chat join
    - `active_campaigns_summary`: Aggregated statistics for dashboards
  - **Seed Data**: "The Lost Mines of Phandelver" starter campaign

#### Health Check
`pg_isready -U dm_admin -d dungeon_data` (5s interval, 5s timeout, 5 retries)

---

### 2. **LLM Service** (`llm_brain`)
**Container:** `dnd_llm` | **Port:** 8000 | **GPU:** Required

The intelligence layer running Llama 3.2 for narrative generation and game orchestration.

#### Key Components

**`main.py`**: FastAPI application entry point
- **Initialization**:
  - Loads Llama 3.2 text-generation pipeline (GPU/CPU auto-config)
  - Service-level logging
- **Helper Functions**:
  - `check_service_health(url)`: Pings Image/TTS microservices
  - `get_dnd_context(query)`: Fetches D&D 5e rules from `dnd5eapi.co`
  - `build_context_prompt(campaign_id, user_prompt)`: Aggregates DB history, story segments, and characters into LLM context
- **Endpoints**:
  - `POST /dm_turn`: **Core game loop** - receives player input, builds context, generates response, saves to DB, triggers media generation
  - `POST /campaign/create`: Initialize new campaign
  - `GET /campaign/{id}`: Retrieve campaign metadata
  - `GET /campaign/{id}/history`: Fetch conversation logs
  - `GET /campaign/{id}/story`: Retrieve narrative segments
  - `DELETE /campaign/{id}/history`: Clear context
  - `GET /health`, `GET /`: Status checks

**`db_helper.py`**: Database abstraction layer
- **Class: `DatabaseManager`**
  - **Connection**: `SimpleConnectionPool` with env-based config
  - **Campaign Operations**: `create_campaign`, `get_active_campaign`, `get_campaign_by_id`
  - **Conversation History**: `save_conversation`, `get_recent_conversations`, `get_conversation_context`
  - **Story Management**: `save_story_segment`, `get_story_so_far`
  - **World Building CRUD**: 
    - Characters: `add_character`, `get_active_characters`
    - Locations: `add_location`, `visit_location`
    - Quests: `add_quest`, `complete_quest`, `get_active_quests`
  - **Utility**: `get_campaign_summary`, `clear_campaign_history`

**`Dockerfile`**: Python 3.10-slim with model cache at `/data/model_cache`

**`requirements.txt`**: fastapi, uvicorn, requests, transformers, torch, accelerate, psycopg2-binary

#### Dependencies
Database (healthy), Image Gen (healthy), TTS (healthy)

#### Health Check
`curl -f http://localhost:8000/health` (15s interval, 240s start period)

---

### 3. **Image Generation Service** (`image_gen`)
**Container:** `dnd_artist` | **Port:** 8001 | **GPU:** Required

Stable Diffusion v1.5 service for generating scene illustrations.

#### Key Components

**`main.py`**: FastAPI application for image generation
- **Initialization**:
  - Loads `runwayml/stable-diffusion-v1-5` pipeline
  - GPU/CPU auto-detection
  - Output directory: `/shared/images`
  - `STYLE_PROMPTS`: Maps style keys (e.g., "fantasy") to detailed artistic suffixes
  - `DEFAULT_NEGATIVE_PROMPT`: Quality filter embeddings
- **Data Models**:
  - `class ImageRequest(BaseModel)`: JSON payload schema (description, style, negative_prompt)
- **Endpoints**:
  - `POST /generate_image`: **Core functionality** - creates unique ID, merges prompts with style templates, runs 50-step inference, saves PNG
  - `GET /images/{image_id}`: Retrieves generated image file
  - `GET /styles`: Lists available art styles with descriptions
  - `GET /health`: Service status and GPU availability

**`Dockerfile`**: Python 3.10-slim with `/data/model_cache` and `/shared/images`

**`requirements.txt`**: fastapi, uvicorn, diffusers, transformers, accelerate, torch

#### Health Check
`curl -f http://localhost:8001/health` (15s interval, 180s start period)

---

### 4. **Text-to-Speech Service** (`tts_voice`)
**Container:** `dnd_voice` | **Port:** 8002 | **GPU:** Required

ChatterboxTTS service for streaming narration audio.

#### Key Components

**`main.py`**: FastAPI application with streaming TTS
- **Initialization**: Loads ChatterboxTTS model
- **Functions**:
  - `stream_audio_generator`: Yields JSON-encoded SSE events
  - `clean_text_for_tts`: Text preprocessing
- **Endpoints**:
  - `POST /speak`: Streams audio via Server-Sent Events
  - `GET /voices`: Lists available voice profiles
  - `GET /health`: Service status

**`Dockerfile`**: Multi-stage build with complex dependency management
- Handles version constraints for `chatterbox-tts` (numpy, transformers, diffusers)
- Installs `ffmpeg`, `espeak-ng` system dependencies

**`requirements.txt`**: fastapi, uvicorn, audio libraries (modified install in Dockerfile)

#### Health Check
`curl -f http://localhost:8002/health` (15s interval, 180s start period)

---

### 5. **Web Application** (`webapp`)
**Container:** `dnd_website` | **Port:** 8080 (maps to internal 8000)

Django-based user interface for interacting with the game system.

#### Key Features
- Interfaces with all backend services via REST APIs
- Accesses shared volumes for images and audio
- Direct database connection for queries
- Docker socket access (likely for container management)

#### Environment Configuration
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`: Database credentials
- `LLM_API`: http://llm_brain:8000
- `IMAGE_API`: http://image_gen:8001
- `TTS_API`: http://tts_voice:8002

#### Dependencies
Database (healthy), LLM Brain (healthy)

---

## Shared Resources

### Volumes
- `postgres_data`: Persistent database storage
- `shared_images`: Image generation output (accessible by webapp and image_gen)
- `shared_audio`: TTS output (accessible by webapp and tts_voice)

### Model Caches
- **LLM**: `${MODEL_CACHE_PATH}/huggingface` → `/data/model_cache`
- **Image Gen**: `${MODEL_CACHE_PATH}/diffusers` → `/data/model_cache`
- **TTS**: `~/dnd_models/tts_cache` → `/root/.cache/huggingface`
- **Reference Voices**: `~/dnd_models/reference_voices` → `/shared/reference_voices`

---

## Startup Sequence

1. **Database** starts and runs `init_db.sql`
2. **Image Gen** and **TTS** start in parallel (independent of each other)
3. **LLM Brain** starts after all support services are healthy
4. **Web App** starts after database and LLM are healthy

---

## Environment Variables

### Required
- `HF_TOKEN`: Hugging Face API token for model downloads
- `DB_PASS`: Database password (default: `secretpassword`)

### Optional
- `MODEL_CACHE_PATH`: Host path for model storage (default: `./model_cache`)

---

## API Flow Example

**Player Turn Execution:**
1. User submits input via Django webapp
2. Webapp calls `POST /dm_turn` on LLM service
3. LLM service:
   - Builds context via `build_context_prompt()` (queries database)
   - Fetches D&D rules via `get_dnd_context()` if needed
   - Generates narrative response via Llama 3.2
   - Saves conversation to database via `db_helper.save_conversation()`
   - Triggers `POST /generate_image` to Image Gen service
   - Returns response to webapp
4. Webapp displays text and polls for generated image
5. Optional: Webapp calls `POST /speak` on TTS service for audio narration

---

## Health Monitoring

All services expose `/health` endpoints with progressive startup periods:
- Database: 5s checks, no startup period
- Image Gen: 15s checks, 180s startup period
- TTS: 15s checks, 180s startup period
- LLM: 15s checks, 240s startup period

Services wait for dependencies to report healthy before starting.