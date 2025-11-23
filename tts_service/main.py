# ./tts_service/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from gtts import gTTS
import os

app = FastAPI()

class TTSRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak(req: TTSRequest):
    tts = gTTS(req.text, lang='en')
    tts.save("output.mp3")
    # Here you would stream the audio back or save it
    # Since this is in Docker, hearing it on the server requires passing audio devices
    # Usually, you send the FILE back to the client computer to play.
    return {"status": "audio_generated", "file": "output.mp3"}