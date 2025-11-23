# ./image_service/main.py

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uuid
from pathlib import Path

app = FastAPI()

# Create output directory
output_dir = Path("/shared/images")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Stable Diffusion...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Stable Diffusion loaded on GPU!")
else:
    print("Stable Diffusion loaded on CPU (will be slow)")

class ImageRequest(BaseModel):
    description: str
    negative_prompt: str = None
    style: str = "fantasy"  # fantasy, dark, epic, cinematic

# Enhanced prompt templates for different styles
STYLE_PROMPTS = {
    "fantasy": "high quality fantasy art, detailed, dungeons and dragons style, artstation, concept art, highly detailed, sharp focus, digital painting",
    "dark": "dark fantasy art, moody atmosphere, dramatic lighting, ominous, gothic, detailed, professional illustration",
    "epic": "epic fantasy scene, grand scale, cinematic, dramatic lighting, highly detailed, masterpiece, 8k",
    "cinematic": "cinematic composition, dramatic lighting, film still, high detail, professional photography, depth of field",
    "painterly": "oil painting, brushstrokes, artistic, fantasy art, detailed, masterwork, museum quality"
}

# Default negative prompt to avoid common issues
DEFAULT_NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, pixelated, jpeg artifacts, watermark, "
    "text, signature, username, out of frame, cropped, worst quality, "
    "low resolution, distorted, deformed, disfigured, bad anatomy, "
    "mutation, mutated, extra limbs, missing limbs, floating limbs, "
    "disconnected limbs, malformed hands, poorly drawn hands, "
    "poorly drawn face, poorly rendered face, amateur, "
    "low detail, oversaturated"
)

@app.post("/generate_image")
async def generate(req: ImageRequest):
    try:
        # Create unique filename
        image_id = str(uuid.uuid4())
        filename = output_dir / f"{image_id}.png"
        
        # Build enhanced prompt
        style_prefix = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["fantasy"])
        full_prompt = f"{style_prefix}, {req.description}"
        
        # Use custom negative prompt or default
        negative_prompt = req.negative_prompt if req.negative_prompt else DEFAULT_NEGATIVE_PROMPT
        
        print(f"Generating image...")
        print(f"  Prompt: {full_prompt[:100]}...")
        print(f"  Negative: {negative_prompt[:80]}...")
        
        # Generate with enhanced settings
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,  # More steps = better quality (default: 50)
            guidance_scale=7.5,      # How closely to follow prompt (7-9 is good)
            height=512,              # Standard SD 1.5 resolution
            width=512
        ).images[0]
        
        # Save image
        image.save(str(filename))
        print(f"✓ Image saved: {image_id}")
        
        return {
            "status": "success",
            "image_id": image_id,
            "image_url": f"/images/{image_id}",
            "prompt_used": full_prompt[:200],
            "style": req.style
        }
    except Exception as e:
        print(f"✗ Error generating image: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    """Retrieve generated image"""
    filename = output_dir / f"{image_id}.png"
    if not filename.exists():
        return {"error": "Image not found"}
    return FileResponse(filename, media_type="image/png")

@app.get("/styles")
async def get_styles():
    """Get available art styles"""
    return {
        "styles": list(STYLE_PROMPTS.keys()),
        "descriptions": {
            "fantasy": "Classic D&D fantasy art style",
            "dark": "Moody, gothic atmosphere",
            "epic": "Grand, cinematic scenes",
            "cinematic": "Movie-like composition",
            "painterly": "Traditional oil painting look"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "image_gen",
        "gpu_available": torch.cuda.is_available()
    }