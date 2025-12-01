# ./tts_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import uuid
import logging
import torch
import torchaudio as ta
import traceback
from .schemas import TTSRequest
from .tts import TTSManager
from .utils import clean_text_for_tts, split_into_sentences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="D&D TTS Service - Chatterbox Streaming")

tts_manager = TTSManager()

@app.post("/speak")
async def speak(req: TTSRequest):
    logger.info("=" * 60)
    logger.info("NEW TTS REQUEST")
    logger.info(f"Text length: {len(req.text)} chars")
    logger.info(f"Stream mode: {req.stream}")
    logger.info(f"Voice: {req.voice}")
    
    if tts_manager.model is None:
        logger.error("Model is None - cannot generate audio")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Model not loaded"}
        )

    try:
        # Clean text
        text = clean_text_for_tts(req.text)
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "Empty text after cleaning"}
            )
        
        # Truncate if too long
        original_length = len(text)
        if len(text) > 1000:
            logger.warning(f"Truncating text from {len(text)} to 1000 chars")
            text = text[:1000]
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > 500:
                text = text[:last_period + 1]
        
        # Find reference voice
        ref_path = tts_manager.find_reference_voice(req.voice)
        
        # STREAMING MODE
        if req.stream:
            logger.info("Starting streaming mode...")
            sentences = split_into_sentences(text)
            logger.info(f"Split into {len(sentences)} sentences")
            
            return StreamingResponse(
                tts_manager.stream_audio_generator(sentences, ref_path, req.exaggeration, req.cfg_weight),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # STANDARD MODE (non-streaming)
        else:
            audio_id = str(uuid.uuid4())
            filename = tts_manager.output_dir / f"{audio_id}.wav"
            
            logger.info("Generating audio (standard mode)...")
            
            audio_chunk = tts_manager.generate_audio_chunk(
                text,
                ref_path=ref_path,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight
            )

            if audio_chunk:
                with open(filename, "wb") as f:
                    f.write(audio_chunk)
                logger.info("✓ Audio generated successfully!")
                logger.info("=" * 60)
                
                return {
                    "status": "success",
                    "audio_url": f"/audio/{audio_id}",
                    "cloned": ref_path is not None,
                    "voice_used": req.voice if ref_path else "default",
                    "emotion": req.exaggeration,
                    "text_length": len(text),
                    "truncated": len(text) < original_length
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to generate audio")


    except Exception as e:
        logger.error("✗ TTS GENERATION FAILED")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/voices")
async def get_voices():
    """List available reference voices"""
    files = list(tts_manager.reference_dir.glob("*.wav")) + list(tts_manager.reference_dir.glob("*.mp3"))
    voices = [f.stem for f in files]
    logger.info(f"Available voices: {voices}")
    return {"voices": voices, "default": "Built-in default voice"}

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Retrieve generated audio file"""
    filename = tts_manager.output_dir / f"{audio_id}.wav"
    if not filename.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(filename, media_type="audio/wav")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if tts_manager.model is not None else "degraded",
        "model": "Chatterbox English TTS",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": tts_manager.model is not None,
        "streaming_enabled": True
    }

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "Chatterbox TTS with Streaming",
        "version": "0.2.0",
        "model": "English only",
        "features": ["Standard mode", "Streaming mode (sentence-by-sentence)"],
        "endpoints": {
            "speak": "POST /speak (set stream=true for streaming)",
            "voices": "GET /voices",
            "audio": "GET /audio/{id}",
            "health": "GET /health"
        }
    }