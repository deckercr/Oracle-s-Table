# ./tts_service/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from TTS.api import TTS
import os
from pathlib import Path
import uuid
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="D&D TTS Service - Coqui TTS")

# Create audio output directory
output_dir = Path("/shared/audio")
output_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Audio output directory: {output_dir}")

# Create cache directory for TTS models
cache_dir = Path(os.getenv("TTS_CACHE", "/data/tts_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"TTS cache directory: {cache_dir}")

# Initialize Coqui TTS
logger.info("Loading Coqui TTS model...")

# Option 1: Fast, good quality English model (recommended for D&D)
# model_name = "tts_models/en/ljspeech/tacotron2-DDC"

# Option 2: Better quality, multi-speaker (can choose different voices)
model_name = "tts_models/en/vctk/vits"  # Multi-speaker, high quality

# Option 3: XTTS v2 - Most advanced, can clone voices (larger model ~2GB)
# model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

try:
    tts = TTS(
        model_name=model_name,
        progress_bar=True,
        gpu=torch.cuda.is_available()
    )
    
    # Set cache directory
    os.environ["COQUI_TTS_CACHE"] = str(cache_dir)
    
    if torch.cuda.is_available():
        tts = tts.to("cuda")
        logger.info("Coqui TTS loaded on GPU!")
    else:
        logger.info("Coqui TTS loaded on CPU")
        
    # Get available speakers for multi-speaker models
    available_speakers = tts.speakers if hasattr(tts, 'speakers') else []
    logger.info(f"Available speakers: {available_speakers[:5] if available_speakers else 'Single speaker model'}")
    
except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")
    raise

class TTSRequest(BaseModel):
    text: str
    speaker: str = None  # Optional: for multi-speaker models (e.g., "p230", "p231")
    speed: float = 1.0   # Optional: speech speed multiplier

@app.post("/speak")
async def speak(req: TTSRequest):
    """Convert text to speech using Coqui TTS"""
    try:
        # Create unique filename
        audio_id = str(uuid.uuid4())
        filename = output_dir / f"{audio_id}.wav"
        
        logger.info(f"Generating audio: {req.text[:50]}...")
        
        # Limit text length to prevent extremely long generation
        text = req.text[:1000]
        
        # Generate speech with optional speaker selection
        if hasattr(tts, 'speakers') and req.speaker and req.speaker in tts.speakers:
            logger.info(f"Using speaker: {req.speaker}")
            tts.tts_to_file(
                text=text,
                speaker=req.speaker,
                file_path=str(filename)
            )
        else:
            # Single speaker or default
            tts.tts_to_file(
                text=text,
                file_path=str(filename)
            )
        
        logger.info(f"Audio saved: {audio_id}")
        
        return {
            "status": "success",
            "audio_id": audio_id,
            "audio_url": f"/audio/{audio_id}",
            "speaker": req.speaker,
            "available_speakers": available_speakers[:10] if available_speakers else None
        }
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Retrieve generated audio"""
    filename = output_dir / f"{audio_id}.wav"
    if not filename.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(filename, media_type="audio/wav")

@app.get("/speakers")
async def get_speakers():
    """Get list of available speakers (for multi-speaker models)"""
    if hasattr(tts, 'speakers') and tts.speakers:
        return {
            "speakers": tts.speakers,
            "count": len(tts.speakers)
        }
    return {"speakers": [], "message": "Single speaker model"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tts": "Coqui TTS",
        "model": model_name,
        "gpu_available": torch.cuda.is_available(),
        "multi_speaker": hasattr(tts, 'speakers') and bool(tts.speakers)
    }