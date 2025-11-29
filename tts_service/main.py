# ./tts_service/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import uuid
import logging
import torch
import torchaudio as ta
import traceback
import re
import io
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="D&D TTS Service - Chatterbox Streaming")

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
    stream: bool = False  # NEW: Enable streaming mode

def clean_text_for_tts(text: str) -> str:
    """Clean text to prevent TTS issues"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\|[^|]+\|', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', text)
    text = re.sub(r'__?([^_]+)__?', r'\1', text)
    text = text.strip()
    return text

def split_into_sentences(text: str) -> list:
    """Split text into sentences for streaming"""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_audio_chunk(text: str, ref_path: str = None, 
                         exaggeration: float = 0.5, cfg_weight: float = 0.5):
    """Generate audio for a single text chunk"""
    try:
        if ref_path:
            wav = model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()
    
    except Exception as e:
        logger.error(f"Error generating chunk: {e}")
        return None

async def stream_audio_generator(sentences: list, ref_path: str = None,
                                  exaggeration: float = 0.5, cfg_weight: float = 0.5):
    """Generator function that yields audio chunks"""
    for i, sentence in enumerate(sentences):
        logger.info(f"Generating sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
        
        audio_bytes = generate_audio_chunk(sentence, ref_path, exaggeration, cfg_weight)
        
        if audio_bytes:
            # Yield metadata + audio in JSON streaming format
            chunk_data = {
                "chunk_index": i,
                "total_chunks": len(sentences),
                "text": sentence,
                "audio_size": len(audio_bytes),
                "status": "success"
            }
            
            # Send metadata
            yield f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
            
            # Send audio data (base64 encoded for JSON transport)
            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_packet = {
                "audio_data": audio_b64,
                "chunk_index": i
            }
            yield f"data: {json.dumps(audio_packet)}\n\n".encode('utf-8')
            
            logger.info(f"✓ Sent chunk {i+1}/{len(sentences)}")
        else:
            error_packet = {
                "chunk_index": i,
                "status": "error",
                "message": "Failed to generate audio"
            }
            yield f"data: {json.dumps(error_packet)}\n\n".encode('utf-8')
    
    # Send completion signal
    completion = {"status": "complete", "total_chunks": len(sentences)}
    yield f"data: {json.dumps(completion)}\n\n".encode('utf-8')

@app.post("/speak")
async def speak(req: TTSRequest):
    logger.info("=" * 60)
    logger.info("NEW TTS REQUEST")
    logger.info(f"Text length: {len(req.text)} chars")
    logger.info(f"Stream mode: {req.stream}")
    logger.info(f"Voice: {req.voice}")
    
    if model is None:
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
        ref_path = None
        if req.voice != "default":
            for ext in [".wav", ".mp3"]:
                p = reference_dir / f"{req.voice}{ext}"
                if p.exists():
                    ref_path = str(p)
                    logger.info(f"✓ Found reference: {ref_path}")
                    break
        
        # STREAMING MODE
        if req.stream:
            logger.info("Starting streaming mode...")
            sentences = split_into_sentences(text)
            logger.info(f"Split into {len(sentences)} sentences")
            
            return StreamingResponse(
                stream_audio_generator(sentences, ref_path, req.exaggeration, req.cfg_weight),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # STANDARD MODE (non-streaming)
        else:
            audio_id = str(uuid.uuid4())
            filename = output_dir / f"{audio_id}.wav"
            
            logger.info("Generating audio (standard mode)...")
            
            if ref_path:
                wav = model.generate(
                    text,
                    audio_prompt_path=ref_path,
                    exaggeration=req.exaggeration,
                    cfg_weight=req.cfg_weight
                )
                cloned = True
            else:
                wav = model.generate(
                    text,
                    exaggeration=req.exaggeration,
                    cfg_weight=req.cfg_weight
                )
                cloned = False
            
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
        "model_loaded": model is not None,
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