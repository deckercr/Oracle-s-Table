# ./tts_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import uuid
import logging
import torch
import torchaudio as ta
import traceback
import re

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
    from chatterbox.tts import ChatterboxTTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info("Downloading/loading model from HuggingFace...")
    
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
    exaggeration: float = 0.5
    cfg_weight: float = 0.5

def clean_text_for_tts(text: str) -> str:
    """Clean text to prevent TTS issues"""
    # Remove any special tokens or formatting
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/XML tags
    text = re.sub(r'\|[^|]+\|', '', text)  # Remove pipe-delimited tokens
    
    # Remove multiple spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', text)  # Bold/italic
    text = re.sub(r'__?([^_]+)__?', r'\1', text)  # Underline
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

@app.post("/speak")
async def speak(req: TTSRequest):
    logger.info("=" * 60)
    logger.info("NEW TTS REQUEST")
    logger.info(f"Text length: {len(req.text)} chars")
    logger.info(f"Text preview: {req.text[:100]}...")
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
        # Clean and sanitize text
        text = clean_text_for_tts(req.text)
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "Empty text after cleaning"}
            )
        
        # INCREASED: Raise limit to 1000 characters to handle longer narrations
        original_length = len(text)
        if len(text) > 1000:
            logger.warning(f"Truncating text from {len(text)} to 1000 chars")
            # Try to truncate at sentence boundary
            text = text[:1000]
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > 500:  # Only truncate at sentence if we have enough
                text = text[:last_period + 1]
        
        logger.info(f"Final text length: {len(text)} chars (original: {original_length})")
        logger.info(f"Clean text: {text[:150]}...")
        
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
        
        # Generate audio
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

        # Save audio
        logger.info(f"Saving audio to {filename}")
        logger.info(f"Audio tensor shape: {wav.shape}")
        ta.save(str(filename), wav, model.sr)
        
        logger.info("✓ Audio generated successfully!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "audio_url": f"/audio/{audio_id}",
            "cloned": cloned,
            "voice_used": req.voice if cloned else "default",
            "emotion": req.exaggeration,
            "text_length": len(text),
            "truncated": len(text) < original_length
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