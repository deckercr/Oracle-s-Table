# ./llm_service/main.py
"""
LLM service updated to work with Chatterbox TTS.
Uses 'exaggeration' for emotion control.
Optimized for shorter, punchier narrations that work well with TTS.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from pathlib import Path
import requests
import torch
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# INTERNAL NETWORK ADDRESSES
IMAGE_API = "http://image_gen:8001/generate_image"
TTS_API = "http://tts_voice:8002/speak"
DND_API_BASE = "https://www.dnd5eapi.co/api"

# Ensure cache directory exists
cache_dir = Path(os.getenv("HF_HOME", "/data/model_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

logger.info("Loading Llama 3.2...")

try:
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        model_kwargs={
            "low_cpu_mem_usage": True,
            "cache_dir": str(cache_dir)
        },
        token=os.getenv("HF_TOKEN")
    )
    logger.info("✓ Llama model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load Llama model: {e}")
    raise

class GameRequest(BaseModel):
    prompt: str
    generate_image: bool = False
    image_style: str = "fantasy"
    voice: str = "gandalf" 

def check_service_health(url, timeout=2):
    """Quick non-blocking check if service is available"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_dnd_context(query):
    endpoints = ["spells", "monsters", "classes", "races"]
    query_term = query.split()[-1].lower()
    for endpoint in endpoints:
        try:
            url = f"{DND_API_BASE}/{endpoint}?name={query_term}"
            resp = requests.get(url, timeout=3).json()
            if resp.get('count', 0) > 0:
                index = resp['results'][0]['index']
                details = requests.get(f"{DND_API_BASE}/{endpoint}/{index}", timeout=3).json()
                return f"RULE ({endpoint}): {details.get('name')}. {str(details.get('desc', ''))[:200]}..."
        except Exception as e:
            logger.debug(f"No {endpoint} found: {e}")
            continue
    return None

@app.post("/dm_turn")
async def dm_turn(req: GameRequest):
    # 1. Get Rules
    rule_context = get_dnd_context(req.prompt)

    # 2. Build Prompt - UPDATED for shorter, punchier responses
    system_msg = """You are a Dungeon Master. Be vivid but concise. 
    Keep responses to 2-4 sentences maximum - roughly 150-200 words.
    Focus on the most important and dramatic details.
    Always complete your thoughts fully but keep it brief and impactful.
    Think of it like narrating a crucial moment, not describing everything."""

    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}"
    if rule_context:
        full_prompt += f" Rules: {rule_context}"
    full_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{req.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # 3. Generate Text - REDUCED from 512 to 256 tokens (~150-200 words)
    logger.info(f"Generating response for: {req.prompt[:50]}...")
    outputs = pipe(
        full_prompt,
        max_new_tokens=256,  # Reduced from 512
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=pipe.tokenizer.eos_token_id
    )
    response_text = outputs[0]["generated_text"].split("assistant<|end_header_id|>\n\n")[-1]

    # Clean up any incomplete sentences
    if response_text and not response_text.rstrip().endswith(('.', '!', '?', '"', "'")):
        last_period = max(
            response_text.rfind('.'),
            response_text.rfind('!'),
            response_text.rfind('?')
        )
        if last_period > 0:
            response_text = response_text[:last_period + 1]

    logger.info(f"✓ Generated {len(response_text)} characters")

    image_url = None
    audio_status = "skipped"
    audio_url = None

    # 4. Optional: Call Image Generator
    if req.generate_image:
        try:
            if check_service_health("http://image_gen:8001/health", timeout=1):
                logger.info("Requesting image generation...")
                img_req = requests.post(
                    IMAGE_API,
                    json={"description": response_text[:100], "style": req.image_style},
                    timeout=60
                )
                if img_req.status_code == 200:
                    img_data = img_req.json()
                    if img_data.get("status") == "success":
                        image_url = f"http://image_gen:8001{img_data['image_url']}"
                        logger.info(f"✓ Image generated: {image_url}")
            else:
                logger.warning("Image service not available")
                image_url = "Service unavailable"
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            image_url = f"Error: {str(e)}"

    # 5. Generate Audio with Chatterbox
    try:
        if check_service_health("http://tts_voice:8002/health", timeout=1):
            logger.info(f"Requesting TTS with voice '{req.voice}'...")
            
            tts_req = requests.post(
                TTS_API,
                json={
                    "text": response_text,
                    "voice": req.voice,
                    "exaggeration": 0.6,  # Emotion intensity (0.0 - 1.0)
                    "cfg_weight": 0.3     # Generation creativity
                },
                timeout=60
            )
            
            if tts_req.status_code == 200:
                tts_data = tts_req.json()
                if tts_data.get("status") == "success":
                    # The TTS service returns "/audio/{uuid}", we prepend host for webapp
                    audio_url = f"http://tts_voice:8002{tts_data['audio_url']}"
                    audio_status = "generated"
                    cloned = tts_data.get("cloned", False)
                    truncated = tts_data.get("truncated", False)
                    
                    if truncated:
                        logger.warning("⚠️ TTS text was truncated!")
                    
                    logger.info(f"✓ Audio generated (cloned={cloned}, truncated={truncated}): {audio_url}")
                else:
                    logger.warning(f"TTS returned non-success: {tts_data}")
                    audio_status = "TTS failed"
            else:
                logger.warning(f"TTS request failed: HTTP {tts_req.status_code}")
                audio_status = f"HTTP {tts_req.status_code}"
        else:
            logger.warning("TTS service not available")
            audio_status = "Service unavailable"
    except Exception as e:
        logger.error(f"TTS error: {e}")
        audio_status = f"Error: {str(e)}"

    return {
        "text": response_text,
        "rule_ref": rule_context,
        "image_url": image_url,
        "audio_url": audio_url,
        "audio_status": audio_status,
        "voice_used": req.voice
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "llm_brain",
        "model_loaded": pipe is not None
    }

@app.get("/")
async def root():
    return {
        "service": "D&D LLM Brain",
        "status": "running",
        "tts": "Chatterbox (Emotion Enabled)",
        "max_output": "~150-200 words (optimized for TTS)",
        "endpoints": {
            "health": "/health",
            "dm_turn": "/dm_turn (POST)"
        }
    }

logger.info("🎲 LLM Service ready with Chatterbox support!")