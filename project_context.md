# Project Context: D&D AI Dungeon Master System

## Overview
The **D&D AI Dungeon Master System** is a fully containerized, AI-powered application that serves as an intelligent Dungeon Master for Dungeons & Dragons 5th Edition campaigns. It combines natural language processing, image generation, text-to-speech synthesis, and persistent game state management to create an immersive, interactive tabletop RPG experience.

The system orchestrates multiple specialized microservices working in concert to generate narrative content, maintain campaign continuity, produce visual assets, and deliver voiced responses in real-time.

## System Architecture

### High-Level Design
The application follows a microservices architecture with five primary containers:

1. **Database Service** (PostgreSQL + pgvector): Persistence layer for game state and AI memory
2. **LLM Brain Service** (Llama 3.2): Core narrative generation and game logic
3. **Image Generation Service** (Stable Diffusion): Visual asset creation
4. **TTS Voice Service** (Chatterbox): Audio narration synthesis
5. **Web Application** (Django): User interface and orchestration layer

### Network Architecture
- **Network**: Bridge network (`dm_network`) with subnet `172.25.0.0/16`
- **Service Discovery**: Container names used for internal DNS resolution
- **Health Checks**: Cascading healthcheck dependencies ensure services start in correct order
- **Startup Sequence**: Database → Media Services (Image/TTS) → LLM Brain → Web App

## Component Details

### 1. Database Service (`database`)
**Technology**: PostgreSQL 16 + pgvector extension  
**Port**: 5432  
**Container**: `dnd_postgres`

**Purpose**: Central persistence layer handling both relational game data and vector embeddings for semantic search.

**Key Features**:
- Automatic schema initialization via SQL script
- Vector similarity search for narrative memory retrieval (384-dimensional embeddings)
- Cascading deletes for data integrity
- Dynamic user authentication via environment variables

**Data Models**:
- **Core State**: `campaigns` (root entity)
- **Narrative History**: `conversations` (raw chat logs), `story_segments` (summarized beats)
- **World Building**: `characters`, `locations`, `quests`
- **AI Memory**: `story_embeddings` (vector representations for RAG)

**Environment Variables**:
- `POSTGRES_USER`: dm_admin
- `POSTGRES_PASSWORD`: Configurable via `DB_PASS`
- `POSTGRES_DB`: dungeon_data

### 2. LLM Brain Service (`llm_brain`)
**Technology**: Python 3.10, FastAPI, PyTorch, Hugging Face Transformers  
**Model**: meta-llama/Llama-3.2-3B-Instruct  
**Port**: 8000  
**Container**: `dnd_llm`

**Purpose**: Core intelligence of the system, generating contextual narrative responses and coordinating media generation.

**Key Features**:
- Context-aware narrative generation with full campaign history retrieval
- D&D 5e rule integration via `dnd5eapi.co`
- Media orchestration (triggers image generation and TTS)
- Campaign state management and persistence
- Connection pooling for database efficiency

**Dependencies**:
- Database (must be healthy)
- Image Generation Service (must be healthy)
- TTS Service (must be healthy)

**Environment Variables**:
- `HF_TOKEN`: Hugging Face authentication token
- `HF_HOME`: Model cache directory
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`: Database credentials

### 3. Image Generation Service (`image_gen`)
**Technology**: Python 3.10, FastAPI, PyTorch, Diffusers  
**Model**: runwayml/stable-diffusion-v1-5  
**Port**: 8001  
**Container**: `dnd_artist`

**Purpose**: Generates scene illustrations and character portraits from text descriptions.

**Key Features**:
- Text-to-image generation (512x512 resolution)
- Style presets: Fantasy, Dark, Epic, Cinematic, Painterly
- GPU acceleration with CPU fallback
- Static file serving for generated assets
- Prompt engineering with negative prompts

**Storage**:
- Generated images: `/shared/images` (shared volume)
- Model cache: `/data/model_cache`

**Environment Variables**:
- `HF_HOME`, `TRANSFORMERS_CACHE`, `DIFFUSERS_CACHE`: Model caching

### 4. TTS Voice Service (`tts_voice`)
**Technology**: Python 3.10, FastAPI, PyTorch, Chatterbox TTS  
**Port**: 8002  
**Container**: `dnd_voice`

**Purpose**: Converts narrative text into spoken audio with voice cloning capabilities.

**Key Features**:
- Text-to-speech synthesis with Chatterbox model
- Server-Sent Events (SSE) streaming for low-latency playback
- Zero-shot voice cloning from reference audio
- Text preprocessing (removes markdown/formatting artifacts)
- Sentence-by-sentence streaming

**Storage**:
- Generated audio: `/shared/audio` (shared volume)
- Reference voices: `/shared/reference_voices`
- Model cache: `/root/.cache/huggingface`

**Environment Variables**:
- `COQUI_TTS_CACHE`, `TTS_CACHE`: Model caching

### 5. Web Application (`webapp`)
**Technology**: Django, Python  
**Port**: 8080 (mapped to internal 8000)  
**Container**: `dnd_website`

**Purpose**: User interface for campaign management, chat interaction, and media presentation.

**Key Features**:
- Campaign creation and management
- Real-time chat interface with the AI DM
- Image and audio playback
- Character and quest tracking
- Direct access to backend services via API clients

**Dependencies**:
- Database (must be healthy)
- LLM Brain (must be healthy)

**Environment Variables**:
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`: Database credentials
- `LLM_API`: http://llm_brain:8000
- `IMAGE_API`: http://image_gen:8001
- `TTS_API`: http://tts_voice:8002

## Infrastructure

### Docker Configuration
**File**: `docker-compose.yml`

**Shared Volumes**:
- `postgres_data`: Database persistence
- `shared_images`: Generated images accessible by web app and image service
- `shared_audio`: Generated audio accessible by web app and TTS service

**Model Caching**:
- `${MODEL_CACHE_PATH:-./model_cache}/huggingface`: LLM models
- `${MODEL_CACHE_PATH:-./model_cache}/diffusers`: Diffusion models
- `~/dnd_models/tts_cache`: TTS models

**GPU Support**:
All AI services (llm_brain, image_gen, tts_voice) are configured for NVIDIA GPU acceleration via Docker deployment resources.

### Health Checks
Each service implements health endpoints with graduated timeouts:
- **Database**: 5s intervals, 5s timeout, 5 retries
- **Image Generation**: 15s intervals, 10s timeout, 20 retries, 180s start period
- **TTS Service**: 15s intervals, 10s timeout, 20 retries, 180s start period
- **LLM Brain**: 15s intervals, 10s timeout, 20 retries, 240s start period

## Data Flow

### Typical User Interaction
1. **User Input**: Player sends message via web interface
2. **Context Retrieval**: LLM service queries database for:
   - Conversation history
   - Story segments
   - Active quests
   - Character details
   - Related embeddings (vector search)
3. **Rule Integration**: Fetches relevant D&D 5e rules from external API
4. **Narrative Generation**: Llama 3.2 generates response
5. **Media Generation** (conditional):
   - Triggers image generation for scene descriptions
   - Triggers TTS for DM narration
6. **Persistence**: Stores conversation, updates embeddings, tracks token usage
7. **Response Delivery**: Web app presents text, images, and audio to user

### Vector Memory (RAG)
The system maintains long-term narrative memory through:
1. Story segments are embedded into 384-dimensional vectors
2. Vectors stored in `story_embeddings` table with pgvector
3. Semantic similarity search retrieves relevant past events
4. Retrieved context included in LLM prompts for continuity

## Configuration

### Required Environment Variables
Create a `.env` file in the project root:

```env
# Database
DB_PASS=your_secure_password

# Hugging Face (for model downloads)
HF_TOKEN=your_huggingface_token

# Model Cache (optional, defaults to ./model_cache)
MODEL_CACHE_PATH=/path/to/model/cache
```

### Port Mapping
- **5432**: PostgreSQL database
- **8000**: LLM Brain API
- **8001**: Image Generation API
- **8002**: TTS Voice API
- **8080**: Web Application

## Deployment

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit (for GPU access)
- Minimum 16GB RAM
- 50GB free disk space (for models)

### Startup Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Full cleanup (removes volumes)
docker-compose down -v
```

### First-Time Setup
1. Services will download models on first startup (~15-20GB total)
2. Database schema initializes automatically
3. Health checks ensure proper startup sequence
4. Web app becomes available at `http://localhost:8080`

## Technology Stack

### Languages & Frameworks
- **Python 3.10**: All backend services
- **FastAPI**: REST APIs for AI services
- **Django**: Web application framework
- **SQL**: Database schema and queries

### AI/ML Frameworks
- **PyTorch**: Deep learning runtime
- **Hugging Face Transformers**: LLM inference
- **Hugging Face Diffusers**: Image generation
- **Torchaudio**: Audio processing

### Infrastructure
- **PostgreSQL 16**: Relational database
- **pgvector**: Vector similarity extension
- **Docker**: Containerization
- **CUDA**: GPU acceleration

### External APIs
- **dnd5eapi.co**: D&D 5th Edition rule reference

## Security Considerations
- Database credentials injected via environment variables
- No hardcoded passwords in codebase
- Internal service communication over private Docker network
- Only web interface exposed to host network

## Performance Characteristics
- **GPU Acceleration**: All AI services utilize CUDA when available
- **Connection Pooling**: Database connections reused for efficiency
- **Streaming**: TTS supports streaming for low-latency audio
- **Model Caching**: Pre-downloaded models for faster subsequent runs

## Future Extensibility
The modular architecture supports:
- Swapping AI models (different LLMs, image generators, TTS engines)
- Adding new media types (music generation, 3D assets)
- Scaling services independently
- Integrating additional D&D tools (dice rollers, character sheets)
- Multi-user campaign support

## Troubleshooting

### Common Issues
1. **Services fail to start**: Check GPU availability and CUDA installation
2. **Out of memory**: Reduce batch sizes or use CPU fallback
3. **Slow startup**: Models downloading on first run (patience required)
4. **Connection refused**: Verify health checks pass before accessing services

### Logs
```bash
# View specific service logs
docker-compose logs -f llm_brain
docker-compose logs -f database
docker-compose logs -f image_gen
```

## Project Structure
```
dnd-ai-dungeon-master/
├── docker-compose.yml
├── .env
├── database/
│   ├── init_db.sql
│   └── project_context.md
├── llm_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── project_context.md
├── image_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── project_context.md
├── tts_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── project_context.md
└── web_service/
    ├── Dockerfile
    ├── requirements.txt
    └── project_context.md
```