from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    stream: bool = False  # NEW: Enable streaming mode
