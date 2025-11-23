from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

print("Loading Stable Diffusion...")
# Using a fast, smaller model version suitable for home use
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

class ImageRequest(BaseModel):
    description: str

@app.post("/generate_image")
async def generate(req: ImageRequest):
    prompt = f"fantasy art, dungeons and dragons style, {req.description}"
    image = pipe(prompt).images[0]
    
    # Save to a shared volume or return base64 (saving locally for now)
    filename = f"/tmp/generated.png"
    image.save(filename)
    return {"status": "success", "image_path": filename}