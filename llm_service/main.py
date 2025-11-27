# ./llm_service/main.py
"""
LLM service with database integration for campaign memory.
Stores conversations, story segments, and retrieves context.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from pathlib import Path
import requests
import torch
import os
import logging

# Import our database helper
from db_helper import db

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

# Test database connection on startup
logger.info("Testing database connection...")
if db.test_connection():
    logger.info("✓ Database ready!")
else:
    logger.warning("⚠️ Database not available - service will have limited functionality")

class GameRequest(BaseModel):
    prompt: str
    campaign_id: int = None  # Optional: specify campaign ID, or use active one
    generate_image: bool = False
    image_style: str = "fantasy"
    voice: str = "gandalf"
    save_to_db: bool = True  # Whether to save this interaction

def check_service_health(url, timeout=2):
    """Quick non-blocking check if service is available"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_dnd_context(query):
    """Fetch D&D rules from API"""
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

def build_context_prompt(campaign_id, user_prompt):
    """Build a prompt with campaign context from database"""
    context_parts = []
    
    try:
        # Get campaign info
        campaign = db.get_campaign_by_id(campaign_id)
        if campaign:
            context_parts.append(f"Campaign: {campaign['campaign_name']}")
            context_parts.append(f"Style: {campaign['dm_style']}")
        
        # Get recent story
        story = db.get_story_so_far(campaign_id, limit=3)
        if story:
            context_parts.append("\nRecent Story:")
            for segment in story:
                context_parts.append(f"- {segment['content'][:150]}...")
        
        # Get conversation history
        history = db.get_recent_conversations(campaign_id, limit=5)
        if history:
            context_parts.append("\nRecent Conversation:")
            for conv in history[-3:]:  # Last 3 turns
                context_parts.append(f"Player: {conv['user_prompt'][:100]}")
                context_parts.append(f"DM: {conv['ai_response'][:100]}...")
        
        # Get active characters
        characters = db.get_active_characters(campaign_id)
        if characters:
            char_names = [f"{c['name']} ({c['character_type']})" for c in characters[:5]]
            context_parts.append(f"\nActive Characters: {', '.join(char_names)}")
        
        # Get active quests
        quests = db.get_active_quests(campaign_id)
        if quests:
            quest_titles = [q['title'] for q in quests[:3]]
            context_parts.append(f"\nActive Quests: {', '.join(quest_titles)}")
        
    except Exception as e:
        logger.warning(f"Error building context: {e}")
    
    return "\n".join(context_parts) if context_parts else ""

@app.post("/dm_turn")
async def dm_turn(req: GameRequest):
    try:
        # 1. Determine which campaign to use
        campaign_id = req.campaign_id
        if not campaign_id:
            # Try to get active campaign
            active = db.get_active_campaign()
            if active:
                campaign_id = active['id']
                logger.info(f"Using active campaign: {active['campaign_name']} (ID: {campaign_id})")
            else:
                # Create a new campaign
                new_campaign = db.create_campaign(
                    "Untitled Adventure",
                    "A new D&D campaign",
                    "balanced"
                )
                campaign_id = new_campaign['id']
                logger.info(f"Created new campaign (ID: {campaign_id})")
        
        # 2. Get D&D rules context
        rule_context = get_dnd_context(req.prompt)
        
        # 3. Get campaign context from database
        db_context = build_context_prompt(campaign_id, req.prompt)
        
        # 4. Build enhanced prompt with context
        system_msg = """You are a Dungeon Master. Be vivid but concise. 
        Keep responses to 2-4 sentences maximum - roughly 150-200 words.
        Focus on the most important and dramatic details.
        Always complete your thoughts fully but keep it brief and impactful.
        Use the campaign context to maintain continuity."""

        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}"
        
        if db_context:
            full_prompt += f"\n\nCampaign Context:\n{db_context}"
        
        if rule_context:
            full_prompt += f"\n\nRules: {rule_context}"
        
        full_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{req.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # 5. Generate response
        logger.info(f"Generating response for campaign {campaign_id}...")
        outputs = pipe(
            full_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=pipe.tokenizer.eos_token_id
        )
        response_text = outputs[0]["generated_text"].split("assistant<|end_header_id|>\n\n")[-1]

        # Clean up incomplete sentences
        if response_text and not response_text.rstrip().endswith(('.', '!', '?', '"', "'")):
            last_period = max(
                response_text.rfind('.'),
                response_text.rfind('!'),
                response_text.rfind('?')
            )
            if last_period > 0:
                response_text = response_text[:last_period + 1]

        logger.info(f"✓ Generated {len(response_text)} characters")

        # 6. Save to database
        if req.save_to_db:
            try:
                db.save_conversation(
                    campaign_id=campaign_id,
                    user_prompt=req.prompt,
                    ai_response=response_text,
                    response_tokens=len(response_text.split()),
                    image_generated=req.generate_image,
                    audio_generated=True  # Always try to generate audio
                )
                
                # Also save as story segment if it's narrative content
                if len(response_text) > 50:  # Only save substantial responses
                    db.save_story_segment(
                        campaign_id=campaign_id,
                        content=response_text,
                        scene_type="narrative"
                    )
                    
                logger.info("✓ Saved to database")
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")

        image_url = None
        audio_status = "skipped"
        audio_url = None

        # 7. Optional: Generate image
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

        # 8. Generate audio
        try:
            if check_service_health("http://tts_voice:8002/health", timeout=1):
                logger.info(f"Requesting TTS with voice '{req.voice}'...")
                
                tts_req = requests.post(
                    TTS_API,
                    json={
                        "text": response_text,
                        "voice": req.voice,
                        "exaggeration": 0.6,
                        "cfg_weight": 0.3
                    },
                    timeout=60
                )
                
                if tts_req.status_code == 200:
                    tts_data = tts_req.json()
                    if tts_data.get("status") == "success":
                        audio_url = f"http://tts_voice:8002{tts_data['audio_url']}"
                        audio_status = "generated"
                        logger.info(f"✓ Audio generated: {audio_url}")
                    else:
                        audio_status = "TTS failed"
                else:
                    audio_status = f"HTTP {tts_req.status_code}"
            else:
                audio_status = "Service unavailable"
        except Exception as e:
            logger.error(f"TTS error: {e}")
            audio_status = f"Error: {str(e)}"

        return {
            "text": response_text,
            "campaign_id": campaign_id,
            "rule_ref": rule_context,
            "image_url": image_url,
            "audio_url": audio_url,
            "audio_status": audio_status,
            "voice_used": req.voice,
            "context_used": bool(db_context)
        }
        
    except Exception as e:
        logger.error(f"Error in dm_turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# DATABASE MANAGEMENT ENDPOINTS
# ========================================

@app.post("/campaign/create")
async def create_campaign(name: str, description: str = "", dm_style: str = "balanced"):
    """Create a new campaign"""
    try:
        result = db.create_campaign(name, description, dm_style)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaign/{campaign_id}")
async def get_campaign(campaign_id: int):
    """Get campaign details"""
    try:
        campaign = db.get_campaign_by_id(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Add summary stats
        summary = db.get_campaign_summary(campaign_id)
        campaign['stats'] = summary
        
        return campaign
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaign/{campaign_id}/history")
async def get_history(campaign_id: int, limit: int = 20):
    """Get conversation history"""
    try:
        history = db.get_recent_conversations(campaign_id, limit)
        return {"campaign_id": campaign_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaign/{campaign_id}/story")
async def get_story(campaign_id: int, limit: int = 10):
    """Get story segments"""
    try:
        story = db.get_story_so_far(campaign_id, limit)
        return {"campaign_id": campaign_id, "story": story}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/campaign/{campaign_id}/history")
async def clear_history(campaign_id: int):
    """Clear conversation history"""
    try:
        db.clear_campaign_history(campaign_id)
        return {"status": "success", "message": f"Cleared history for campaign {campaign_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    db_status = "connected" if db.test_connection() else "disconnected"
    return {
        "status": "healthy",
        "service": "llm_brain",
        "model_loaded": pipe is not None,
        "database": db_status
    }

@app.get("/")
async def root():
    return {
        "service": "D&D LLM Brain with Campaign Memory",
        "status": "running",
        "database": "PostgreSQL + pgvector",
        "features": [
            "Campaign context retention",
            "Conversation history",
            "Story segment tracking",
            "Character & quest management"
        ],
        "endpoints": {
            "health": "GET /health",
            "dm_turn": "POST /dm_turn",
            "create_campaign": "POST /campaign/create",
            "get_campaign": "GET /campaign/{id}",
            "get_history": "GET /campaign/{id}/history",
            "get_story": "GET /campaign/{id}/story",
            "clear_history": "DELETE /campaign/{id}/history"
        }
    }

logger.info("🎲 LLM Service ready with database integration!")