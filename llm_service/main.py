# ./llm_service/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from pathlib import Path
import requests
import torch
import psycopg2

app = FastAPI()

# INTERNAL NETWORK ADDRESSES (Defined in docker-compose)
IMAGE_API = "http://image_gen:8001/generate_image"
TTS_API = "http://tts_voice:8002/speak"
DND_API_BASE = "https://www.dnd5eapi.co/api"

# Ensure cache directory exists
cache_dir = Path(os.getenv("HF_HOME", "/data/model_cache"))
cache_dir.mkdir(parents=True, exist_ok=True)  # Creates if doesn't exist

print("Loading Llama...")
pipe = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-3B-Instruct", 
    device_map="auto",
    torch_dtype=torch.float16,  # Use half precision
    model_kwargs={"low_cpu_mem_usage": True}
    cache_dir=str(cache_dir),  # Tell HF where to cache
    token=os.getenv("HF_TOKEN")
    )

class GameRequest(BaseModel):
    prompt: str
    generate_image: bool = False

def get_dnd_context(query):
    # Expanded to check classes and races
    endpoints = ["spells", "monsters", "classes", "races"]
    query_term = query.split()[-1].lower() # simplistic keyword extraction
    
    for endpoint in endpoints:
        try:
            url = f"{DND_API_BASE}/{endpoint}?name={query_term}"
            resp = requests.get(url).json()
            if resp['count'] > 0:
                index = resp['results'][0]['index']
                details = requests.get(f"{DND_API_BASE}/{endpoint}/{index}").json()
                # Return a summarized rule snippet
                return f"RULE ({endpoint}): {details.get('name')}. {str(details.get('desc', ''))[:200]}..."
        except:
            continue
    return None

@app.post("/dm_turn")
async def dm_turn(req: GameRequest):
    # 1. Get Rules
    rule_context = get_dnd_context(req.prompt)
    
    # 2. Build Prompt
    system_msg = "You are a DM. Be vivid. Use the rules provided."
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}"
    if rule_context:
        full_prompt += f" Rules: {rule_context}"
    full_prompt += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{req.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # 3. Generate Text
    outputs = pipe(full_prompt, max_new_tokens=250)
    response_text = outputs[0]["generated_text"].split("assistant<|end_header_id|>\n\n")[-1]

    image_url = None
    audio_status = "skipped"

    # 4. Optional: Call Image Generator
    if req.generate_image:
        try:
            # We ask the image container to make art based on the AI's description
            img_req = requests.post(IMAGE_API, json={"description": response_text[:100]})
            if img_req.status_code == 200:
                image_url = img_req.json()["image_path"]
        except:
            image_url = "Error generating image"

    # 5. Always generate Audio (TTS)
    try:
        requests.post(TTS_API, json={"text": response_text})
        audio_status = "sent to speakers"
    except:
        audio_status = "tts failed"

    return {
        "text": response_text,
        "rule_ref": rule_context,
        "image": image_url,
        "audio": audio_status
    }