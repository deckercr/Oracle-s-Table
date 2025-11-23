

import requests
from django.shortcuts import render

def play_turn(request):
    player_input = request.POST.get("input")
    
    # NOTICE: We use "http://llm_brain:8000" 
    # Docker automatically routes this to the container named 'dnd_llm'
    
    try:
        ai_response = requests.post(
            "http://llm_brain:8000/dm_turn", 
            json={"prompt": player_input, "generate_image": True}
        )
        data = ai_response.json()
    except:
        data = {"text": "The Dungeon Master is sleeping (Service Offline)."}

    # The image URL returned will be a path inside the container.
    # You might need to handle serving that file, but for now:
    return render(request, "game.html", {"response": data})
