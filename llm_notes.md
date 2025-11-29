# Master LLM Notes & Developer Guide
## D&D AI Dungeon Master Application

---

## System Architecture Overview

### Service Topology
The application consists of 5 containerized services orchestrated via Docker Compose:

1. **Database (PostgreSQL + pgvector)** - Port 5432
   - Primary data store with vector search capabilities
   - First to start, required by all other services

2. **LLM Brain (Llama 3.2-3B)** - Port 8000
   - Core narrative generation and game logic
   - Depends on: Database, Image Gen, TTS Voice

3. **Image Gen (Stable Diffusion v1.5)** - Port 8001
   - Scene and character visualization
   - Independent service, starts before LLM

4. **TTS Voice (Chatterbox TTS)** - Port 8002
   - Audio narration with voice cloning
   - Independent service, starts before LLM

5. **Web App (Django)** - Port 8080
   - User interface and orchestration layer
   - Starts last, depends on all other services

### Network Configuration
- **Network:** `dm_network` (bridge driver)
- **Subnet:** 172.25.0.0/16
- **Internal DNS:** Services communicate via container names (e.g., `http://llm_brain:8000`)

### Data Flow Pattern
```
User Request (Django)
    ↓
LLM Brain (generates narrative + makes decisions)
    ↓ (parallel, non-blocking)
    ├→ Image Gen (scene visualization)
    └→ TTS Voice (audio narration)
    ↓
Response returned to user (text first, media async)
```

---

## Database Service (PostgreSQL + pgvector)

### Core Specifications
- **Image:** `pgvector/pgvector:pg16`
- **Container:** `dnd_postgres`
- **Credentials:** `dm_admin` / `${DB_PASS:-secretpassword}`
- **Database:** `dungeon_data`

### Vector Search Implementation
- **Extension:** `pgvector` for similarity search
- **Dimensions:** `vector(384)` in `story_embeddings` table
- **Critical Constraint:** Python embedding model MUST output exactly 384 dimensions
  - ✅ Recommended: `sentence-transformers/all-MiniLM-L6-v2`
  - ❌ Incompatible: `text-embedding-ada-002` (1536 dims)

### Indexing Strategy
- **Type:** `ivfflat` with `lists = 100`
- **Caveat:** Index created at initialization (empty DB)
  - Works but suboptimal for large datasets
  - Consider `REINDEX` after significant data growth for better cluster centers

### RAG (Retrieval-Augmented Generation) Flow
1. Search `story_embeddings` by vector similarity
2. Join with `story_segments` to retrieve `content` text
3. Inject into LLM context window

### Schema Design Patterns
- **Cascading Deletes:** All gameplay tables use `ON DELETE CASCADE` from `campaigns(id)`
  - **Implication:** `DELETE FROM campaigns WHERE id = X` wipes entire game save
- **Media Flags:** `conversations` table has `image_generated` and `audio_generated` booleans
  - Purpose: Signal frontend to fetch assets from file server
  - Does NOT store binary data in DB

### Docker Integration
- **Custom Config:** Uses `current_setting('env.APP_DB_USER')` pattern
- **Setup Required:** Variables must be passed via `docker-compose.yml` or config mounts
- **Fallback:** Defaults to `llama_user` / `change_me_default_pass` if not set

### Development Tools
- **Debug View:** `SELECT * FROM recent_conversations` for chat flow inspection
- **Volume:** `postgres_data` persists across container restarts

---

## LLM Brain Service (Llama 3.2-3B)

### Core Specifications
- **Model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Container:** `dnd_llm`
- **Port:** 8000
- **Cache:** `/data/model_cache` (mapped to host `${MODEL_CACHE_PATH}/huggingface`)

### Architectural Characteristics
- **Monolithic Design:** All logic (API, Model Loading, Inference) in `main.py`
- **Startup Time:** Significant delay due to model download/loading at global scope
- **State Management:** Stateless service, context reconstructed from Postgres per request

### Prompt Engineering Strategy
Located in `main.py` → `dm_turn` function:

1. **System Prompt:** DM persona, brevity (2-4 sentences), tone
2. **Context Injection (ordered):**
   - Campaign name and DM style
   - Last 3 story segments
   - Last 3 conversation turns (User/Assistant pairs)
   - Active characters and quests from DB
   - D&D 5e rules (fetched from API based on keywords)
3. **Format:** Llama 3 special tokens: `<|begin_of_text|>`, `<|start_header_id|>`, etc.

### Context Window Management
- **History Limit:** 5 turns
- **Story Limit:** 3 segments
- **Risk:** Very long prompts might exceed context window (though Llama 3.2 has generous limits)
- **Output Truncation:** 256 `max_new_tokens` (relatively short)

### Request Latency Breakdown
Every `dm_turn` request involves:
1. **DB Reads:** Campaign, History, Story, Characters, Quests
2. **External HTTP:** D&D 5e API call
3. **LLM Inference:** Heaviest operation (blocking)
4. **DB Writes:** Conversation log, optional story segment
5. **Parallel HTTP:** Image/TTS services (async/non-blocking for text return)

### Memory & Performance
- **Optimization Flags:** `low_cpu_mem_usage=True`, `torch.float16`
- **RAM/VRAM:** Requires decent resources for 3B model
- **Database Pooling:** `SimpleConnectionPool` (min 1, max 10 connections)

### Error Handling & Resilience
- **Service Failures:** If `image_gen` or `tts_voice` are down, text response still returns (errors logged)
- **DB Failures:** Service enters "limited functionality" mode (warnings in startup logs)
- **Sentence Cleanup:** Manually truncates at last punctuation if `max_new_tokens` hit mid-sentence

### Critical Environment Variables
- `HF_TOKEN`: Required for model download
- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`: Database connection
- `HF_HOME`, `TRANSFORMERS_CACHE`: Model caching paths

---

## Image Generation Service (Stable Diffusion)

### Core Specifications
- **Model:** Stable Diffusion v1.5
- **Container:** `dnd_artist`
- **Port:** 8001
- **Precision:** `float16` (4-6GB VRAM/RAM required)

### Architectural Caveats
1. **Blocking Inference:**
   - `generate` is `async def` but `pipe(...)` is synchronous/blocking
   - **Impact:** Event loop blocked during generation (cannot answer `/health` during inference)
   - **Production Fix:** Use `ThreadPoolExecutor` or separate worker process (Celery/Redis)

2. **Startup Blocker:**
   - Model loads at global scope before API accepts requests
   - Container startup is slow (download + load into memory)

3. **CPU vs GPU:**
   - **GPU:** 4-6GB VRAM, reasonable speed
   - **CPU:** Works but extremely slow (minutes per image)

### Prompt Engineering
- **Style Dictionary:** `STYLE_PROMPTS` prepends artistic keywords to user input
- **Negative Prompt:** Heavy `DEFAULT_NEGATIVE_PROMPT` filters "bad anatomy" and "blurry" results

### Storage & Persistence
- **Output Path:** `/shared/images`
- **Critical:** Must mount as volume or data lost on restart
- **Volume Mapping:** `shared_images:/shared/images` in docker-compose

### Docker/Infrastructure Issues
- **Cache Location:** Dockerfile creates `/data/model_cache` but code may not explicitly use it
  - Hugging Face defaults to `~/.cache/huggingface`
  - Risk: Re-downloading models if home directory not persisted
  - **Fix:** Ensure `HF_HOME=/data/model_cache` in environment

### Potential Refactoring Targets
- Wrap `pipe()` in `fastapi.concurrency.run_in_threadpool`
- Move `model_id` and paths to environment variables
- Strict validation of `style` parameter against `STYLE_PROMPTS` keys

---

## TTS Voice Service (Chatterbox TTS)

### Core Specifications
- **Container:** `dnd_voice`
- **Port:** 8002
- **Reference Voices:** `/shared/reference_voices` (must be externally provided)

### Dependency Management (Critical)
**DO NOT simply install `requirements.txt`** - use Dockerfile's specific sequence:

1. Install all deps EXCEPT `chatterbox-tts`
2. Manually install: `numpy`, `torch`, `transformers`, `diffusers` (specific versions)
3. Install `chatterbox-tts` with `--no-deps` flag
4. **Reason:** Automatic resolution pulls conflicting CUDA/NumPy versions

### Streaming Implementation
- **Protocol:** Server-Sent Events (SSE), **not** raw byte stream
- **Data Format:** JSON objects per event
  - Metadata events: Text progress
  - Audio events: `"audio_data"` key with **Base64 encoded** WAV bytes
- **Client Requirements:**
  - Cannot use `<audio src="...">` directly
  - Must: Open `EventSource` → Decode Base64 → Feed to AudioContext

### Audio Generation Logic
- **Sentence Splitting:** Strictly by punctuation (`.!?`)
- **Input Truncation:** Hard limit at 1000 characters (prevents memory overflow/timeouts)
- **Voice Cloning:** Looks for files in `/shared/reference_voices` (no upload endpoint)

### Performance Characteristics
- **Model Loading:** Global scope (slow startup, fast inference)
- **Concurrency:** PyTorch models generally not thread-safe
  - FastAPI is async but `model.generate` is blocking
  - Heavy load may require semaphore/queue (not currently implemented)

### Volume Mappings
- `shared_audio:/shared/audio` - Generated audio output
- `~/dnd_models/tts_cache:/root/.cache/huggingface` - Model cache
- `~/dnd_models/reference_voices:/shared/reference_voices` - Voice cloning samples

---

## Web Application Service (Django)

### Core Specifications
- **Container:** `dnd_website`
- **Port:** 8080 (mapped to internal 8000)
- **Role:** User interface and orchestration layer

### Service Dependencies
- **Required:** Database (healthy), LLM Brain (healthy)
- **API Endpoints:**
  - `LLM_API=http://llm_brain:8000`
  - `IMAGE_API=http://image_gen:8001`
  - `TTS_API=http://tts_voice:8002`

### Volume Mounts
- `./web_service:/app` - Code (for development hot-reload)
- `/var/run/docker.sock:/var/run/docker.sock` - Docker API access
- `shared_images:/shared/images` - Image assets
- `shared_audio:/shared/audio` - Audio assets

---

## Shared Infrastructure Patterns

### GPU Resource Allocation
All AI services (LLM, Image Gen, TTS) configured with:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

### Health Check Strategy
- **Database:** `pg_isready` every 5s
- **AI Services:** HTTP `/health` endpoint
  - Interval: 15s
  - Timeout: 10s
  - Retries: 20
  - Start period: 180-240s (allows for model loading)

### Model Caching Best Practices
1. **Separate cache volumes per service type:**
   - Diffusers: `${MODEL_CACHE_PATH}/diffusers`
   - Hugging Face: `${MODEL_CACHE_PATH}/huggingface`
   - TTS: `~/dnd_models/tts_cache`

2. **Environment variables (set consistently):**
   - `HF_HOME`
   - `TRANSFORMERS_CACHE`
   - `DIFFUSERS_CACHE` / `TTS_CACHE`

3. **Host path mapping:** Use `${MODEL_CACHE_PATH:-./model_cache}` with fallback

---

## Common Gotchas & Production Considerations

### 1. Startup Order
- Database must be healthy first
- Image/TTS start in parallel, must be healthy before LLM
- Web app starts last
- **Risk:** If health checks fail, dependent services won't start

### 2. Memory Requirements (Approximate)
- **Database:** 512MB-1GB
- **LLM Brain:** 6-8GB RAM + 4-6GB VRAM
- **Image Gen:** 4-6GB VRAM (or 8GB+ RAM if CPU-only)
- **TTS Voice:** 2-4GB VRAM
- **Web App:** 512MB-1GB
- **Total (GPU):** ~16GB VRAM + 8-12GB RAM
- **Total (CPU):** ~24-32GB RAM (painfully slow inference)

### 3. Concurrent Request Handling
- All AI services use blocking inference in async contexts
- Single request blocks entire service during generation
- **Production fixes:**
  - Request queuing (Redis/Celery)
  - ThreadPoolExecutor wrapping
  - Multiple worker processes
  - Semaphores to limit concurrency

### 4. Data Persistence Points
- ✅ **Persisted:** `postgres_data`, `shared_images`, `shared_audio`, model caches
- ❌ **Not Persisted:** Anything in container filesystem outside volumes
- **Risk:** Container recreate loses non-volume data

### 5. Network Communication
- Services use internal DNS (container names)
- External access only via exposed ports
- **Debug Tip:** `docker exec -it dnd_llm curl http://image_gen:8001/health`

---

## Environment Variables Reference

### Required
- `HF_TOKEN` - Hugging Face API token (for Llama model download)
- `DB_PASS` - Database password (defaults to `secretpassword`)

### Optional
- `MODEL_CACHE_PATH` - Host path for model caching (defaults to `./model_cache`)

### Auto-Set by Compose
- `DB_HOST`, `DB_NAME`, `DB_USER` - Database connection
- `LLM_API`, `IMAGE_API`, `TTS_API` - Internal service URLs
- `HF_HOME`, `TRANSFORMERS_CACHE`, etc. - Cache directories

---

## Debugging & Development Tips

### Service Logs
```bash
docker-compose logs -f [service_name]
docker-compose logs -f llm_brain
```

### Database Inspection
```bash
docker exec -it dnd_postgres psql -U dm_admin -d dungeon_data
```

### Restart Single Service
```bash
docker-compose restart llm_brain
```

### Rebuild After Code Changes
```bash
docker-compose up --build [service_name]
```

### Check Service Health
```bash
curl http://localhost:8000/health  # LLM
curl http://localhost:8001/health  # Image Gen
curl http://localhost:8002/health  # TTS
```

### Database Debugging Queries
```sql
-- View recent conversations
SELECT * FROM recent_conversations;

-- Check vector embeddings
SELECT id, left(content, 50), embedding <-> '[0.1,0.2,...]' AS distance
FROM story_embeddings
ORDER BY distance LIMIT 5;

-- Wipe game save (cascades to all related tables)
DELETE FROM campaigns WHERE id = X;
```

---

## Performance Optimization Checklist

### For Development
- [ ] Use CPU-only mode (comment out GPU reservations)
- [ ] Reduce `max_new_tokens` in LLM service
- [ ] Use smaller models (consider DistilBERT for embeddings)
- [ ] Mount code directories for hot-reload

### For Production
- [ ] Implement request queuing (Redis/Celery)
- [ ] Add horizontal scaling for LLM service
- [ ] Use CDN for static assets (images/audio)
- [ ] Implement result caching (Redis)
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Set up log aggregation (ELK stack)
- [ ] Configure backup strategy for `postgres_data`
- [ ] Implement rate limiting on API endpoints
- [ ] Add authentication/authorization layer
- [ ] Use production WSGI server for Django (Gunicorn/uWSGI)

---

## Security Considerations

1. **Database Credentials:** Use Docker secrets instead of environment variables
2. **API Tokens:** `HF_TOKEN` should be in `.env` file (not committed to git)
3. **Network Isolation:** Consider using separate networks for DB access
4. **File Uploads:** Web service has Docker socket access (high privilege)
5. **Input Validation:** All user inputs should be sanitized before DB queries
6. **Rate Limiting:** Prevent abuse of expensive AI operations

---

## Troubleshooting Common Issues

### "CUDA out of memory"
- Reduce batch sizes in model configs
- Use `torch.float16` or `torch.bfloat16`
- Close other GPU applications
- Restart Docker daemon

### "Model not found" errors
- Check `HF_TOKEN` is set correctly
- Verify model cache volumes are mounted
- Check disk space on host
- Ensure model IDs are correct

### "Service unhealthy" on startup
- Increase `start_period` in health checks
- Check service logs for specific errors
- Verify GPU drivers installed (nvidia-docker2)
- Ensure sufficient memory available

### Database connection refused
- Wait for database health check to pass
- Verify network connectivity between containers
- Check `DB_PASS` matches in all services
- Inspect database logs for startup errors

### Inference is extremely slow
- Confirm GPU is being used (check `nvidia-smi`)
- Verify CUDA libraries are compatible
- Consider reducing model sizes
- Check for CPU throttling or thermal issues

---

## Future Improvement Ideas

1. **Microservices Split:** Separate Django into frontend/backend
2. **Async Workers:** Celery for long-running tasks
3. **API Gateway:** Kong or Traefik for routing/rate limiting
4. **Model Versioning:** Track and rollback model updates
5. **A/B Testing:** Compare different prompt strategies
6. **User Analytics:** Track engagement and performance metrics
7. **Backup Automation:** Scheduled PostgreSQL dumps
8. **Distributed Tracing:** OpenTelemetry integration
9. **Load Testing:** K6 or Locust for performance validation
10. **CI/CD Pipeline:** Automated testing and deployment