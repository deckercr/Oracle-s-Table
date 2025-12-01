# Gemini CLI Client Instructions for Streaming Audio & Images

This document outlines how to interact with the new streaming pipeline of the D&D Dungeon Master application. The `llm_service` now provides a streaming endpoint that allows the client to receive and play audio sentence by sentence, and to receive a generated image at the end of the stream.

## 1. The Streaming Endpoint

A new endpoint has been added to the `llm_service`:

- **URL**: `http://10.0.0.5:8000/dm_turn_stream`
- **Method**: `POST`
- **Request Body**: To generate an image, set `"generate_image": true`. You can also specify an `"image_style"`.

```json
{
  "prompt": "I open the creaky door and peer inside...",
  "campaign_id": 1,
  "voice": "gandalf",
  "generate_image": true,
  "image_style": "dark",
  "save_to_db": true
}
```

## 2. Server Response

The server responds with a `StreamingResponse` using the `application/x-ndjson` (newline-delimited JSON) media type. As each part of the response is ready, the server sends a JSON object followed by a newline.

You will receive multiple objects for audio, and one at the end for the image.

### Stream Object Types

**Audio Clip (Success):**
```json
{"text": "A chilling wind blows from the darkness beyond.", "audio_url": "http://tts_voice:8002/static/audio/some_unique_id.wav"}
```

**Image URL (Success):**
```json
{"image_url": "http://10.0.0.5:8001/images/some_unique_id.png"}
```

**Error:**
```json
{"text": "The air grows heavy with an ancient evil.", "error": "TTS service returned status 500"}
```
```json
{"error": "Image service not available"}
```

## 3. How to Process the Stream

The client should read the response from the server line by line. Each line is a complete JSON object that needs to be parsed to determine its type (audio, image, or error).

### Example using `curl`, `jq`, and `ffplay`

This example script demonstrates how to handle both audio and image objects from the stream.

```bash
#!/bin/bash

# Create a directory for downloaded images
mkdir -p ./generated_images

# Make the streaming request
curl -N -X POST http://10.0.0.5:8000/dm_turn_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A huge, ancient red dragon sleeps on a mountain of gold coins inside a volcanic cavern. I want to try and sneak past it.",
    "voice": "gandalf",
    "generate_image": true,
    "image_style": "epic"
  }' \
| while read -r line; do
    # 1. Parse the JSON and check for an audio URL
    audio_url=$(echo "$line" | jq -r '.audio_url')
    text=$(echo "$line" | jq -r '.text')

    if [ -n "$audio_url" ] && [ "$audio_url" != "null" ]; then
      echo "Playing audio for: \"$text\""
      
      # Play the audio clip.
      # You may need to install ffplay: `sudo apt-get install ffmpeg`
      ffplay -nodisp -autoexit "$audio_url" > /dev/null 2>&1
      continue # Move to the next line
    fi

    # 2. Check for an image URL
    image_url=$(echo "$line" | jq -r '.image_url')
    if [ -n "$image_url" ] && [ "$image_url" != "null" ]; then
      filename=$(basename "$image_url")
      echo "Downloading image: $filename"
      
      # Download the image using curl
      curl -s -o "./generated_images/$filename" "$image_url"
      echo "Image saved to ./generated_images/$filename"
      continue # Move to the next line
    fi

    # 3. Handle any errors
    error=$(echo "$line" | jq -r '.error')
    if [ -n "$error" ] && [ "$error" != "null" ]; then
      echo "Received an error: $error"
    fi
  done

echo "Stream finished."
```

### Explanation of the Script:

1.  **`curl -N`**: The `-N` flag disables buffering, crucial for processing the stream as it arrives.
2.  **`while read -r line`**: Reads the response from `curl` one line at a time.
3.  **`jq -r '.key'`**: `jq` extracts the value of a given key. The script checks for `.audio_url` first, then `.image_url`.
4.  **`ffplay`**: Plays the audio from the received URL. The output is redirected to `/dev/null` to keep the console clean.
5.  **`curl -s -o ...`**: If an `image_url` is found, `curl` is used again to download the image and save it locally.

By following this model, your client can create a rich, interactive experience, playing audio clips as they arrive and displaying a scene-setting image at the end of the Dungeon Master's narration.
