# ./tts_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import uuid
import logging
import torch
import torchaudio as ta  # Import as 'ta' like in their docs
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="D&D TTS Service - Chatterbox")

# Directories
output_dir = Path("/shared/audio")
output_dir.mkdir(parents=True, exist_ok=True)

reference_dir = Path("/shared/reference_voices")
reference_dir.mkdir(parents=True, exist_ok=True)

# Initialize Chatterbox
logger.info("=" * 60)
logger.info("Loading Chatterbox TTS model...")
logger.info("=" * 60)

model = None
try:
    # Import the English-only model (simpler and more stable)
    from chatterbox.tts import ChatterboxTTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info("Downloading/loading model from HuggingFace...")
    
    # Load model exactly as shown in their docs
    model = ChatterboxTTS.from_pretrained(device=device)
    
    logger.info("✓ Chatterbox loaded successfully!")
    logger.info(f"Sample rate: {model.sr}")
    
except Exception as e:
    logger.error("✗ Failed to load Chatterbox!")
    logger.error(f"Error: {e}")
    logger.error(traceback.format_exc())

logger.info("=" * 60)

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    exaggeration: float = 0.5  # 0.0 to 1.0
    cfg_weight: float = 0.5    # 0.0 to 1.0
    temperature: float = 0.7   # Not in their API, removing

@app.post("/speak")
async def speak(req: TTSRequest):
    logger.info("=" * 60)
    logger.info("NEW TTS REQUEST")
    logger.info(f"Text: {req.text[:100]}...")
    logger.info(f"Voice: {req.voice}")
    logger.info(f"Exaggeration: {req.exaggeration}")
    logger.info(f"CFG Weight: {req.cfg_weight}")
    
    if model is None:
        logger.error("Model is None - cannot generate audio")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Model not loaded - check container logs"}
        )

    try:
        # Sanitize text
        text = req.text.strip()
        if not text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "Empty text"}
            )
        
        # Limit text length (TTS models have limits)
        if len(text) > 500:
            logger.warning(f"Truncating text from {len(text)} to 500 chars")
            text = text[:500]
        
        audio_id = str(uuid.uuid4())
        filename = output_dir / f"{audio_id}.wav"

        # Find reference voice
        ref_path = None
        logger.info(f"Looking for voice '{req.voice}' in {reference_dir}")
        
        if req.voice != "default":
            for ext in [".wav", ".mp3"]:
                p = reference_dir / f"{req.voice}{ext}"
                if p.exists():
                    ref_path = str(p)
                    logger.info(f"✓ Found reference: {ref_path}")
                    break
            
            if not ref_path:
                logger.warning(f"Voice '{req.voice}' not found")
                available = list(reference_dir.glob("*.wav")) + list(reference_dir.glob("*.mp3"))
                logger.info(f"Available voices: {[f.stem for f in available]}")
        
        # Generate audio using their exact API
        logger.info("Generating audio...")
        
        if ref_path:
            logger.info(f"Using voice clone with exaggeration={req.exaggeration}, cfg_weight={req.cfg_weight}")
            wav = model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight
            )
            cloned = True
        else:
            logger.info(f"Using default voice with exaggeration={req.exaggeration}")
            wav = model.generate(
                text,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight
            )
            cloned = False

        # Save using torchaudio as 'ta' like in their examples
        logger.info(f"Saving audio to {filename}")
        ta.save(str(filename), wav, model.sr)
        
        logger.info("✓ Audio generated successfully!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "audio_url": f"/audio/{audio_id}",
            "cloned": cloned,
            "voice_used": req.voice if cloned else "default",
            "emotion": req.exaggeration
        }

    except Exception as e:
        logger.error("✗ TTS GENERATION FAILED")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        logger.info("=" * 60)
        
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/voices")
async def get_voices():
    """List available reference voices"""
    files = list(reference_dir.glob("*.wav")) + list(reference_dir.glob("*.mp3"))
    voices = [f.stem for f in files]
    logger.info(f"Available voices: {voices}")
    return {"voices": voices, "default": "Built-in default voice"}

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Retrieve generated audio file"""
    filename = output_dir / f"{audio_id}.wav"
    if not filename.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(filename, media_type="audio/wav")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model": "Chatterbox English TTS",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "Chatterbox TTS",
        "version": "0.1.2",
        "model": "English only",
        "endpoints": {
            "speak": "POST /speak",
            "voices": "GET /voices",
            "audio": "GET /audio/{id}",
            "health": "GET /health"
        }
    }