import re

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
