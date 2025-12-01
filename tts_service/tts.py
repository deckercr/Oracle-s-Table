# ./tts_service/tts.py
import torch
import torchaudio as ta
import logging
import traceback
import io
import json
import hashlib
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

from .utils import clean_text_for_tts, split_into_sentences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, output_dir="/shared/audio", reference_dir="/shared/reference_voices"):
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()
        self.cache = {}

    def _load_model(self):
        logger.info("=" * 60)
        logger.info("Loading Chatterbox TTS model...")
        logger.info("=" * 60)
        model = None
        try:
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
        return model

    def _get_cache_key(self, text: str, ref_path: str, exaggeration: float, cfg_weight: float):
        key_str = f"{text}-{ref_path}-{exaggeration}-{cfg_weight}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def generate_audio_chunk(self, text: str, ref_path: str = None, 
                             exaggeration: float = 0.5, cfg_weight: float = 0.5):
        """Generate audio for a single text chunk, using a cache."""
        cache_key = self._get_cache_key(text, ref_path, exaggeration, cfg_weight)
        if cache_key in self.cache:
            logger.info(f"✓ Cache hit for: {text[:30]}...")
            return self.cache[cache_key]

        logger.info(f"Cache miss for: {text[:30]}...")
        try:
            if ref_path:
                wav = self.model.generate(
                    text,
                    audio_prompt_path=ref_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                wav = self.model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            
            # Convert to bytes
            buffer = io.BytesIO()
            ta.save(buffer, wav, self.model.sr, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            self.cache[cache_key] = audio_bytes
            return audio_bytes
        
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            return None

    async def stream_audio_generator(self, sentences: list, ref_path: str = None,
                                      exaggeration: float = 0.5, cfg_weight: float = 0.5):
        """Generator function that yields audio chunks"""
        for i, sentence in enumerate(sentences):
            logger.info(f"Generating sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            audio_bytes = self.generate_audio_chunk(sentence, ref_path, exaggeration, cfg_weight)
            
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

    def find_reference_voice(self, voice: str):
        if voice == "default":
            return None
        for ext in [".wav", ".mp3"]:
            p = self.reference_dir / f"{voice}{ext}"
            if p.exists():
                logger.info(f"✓ Found reference: {p}")
                return str(p)
        return None
