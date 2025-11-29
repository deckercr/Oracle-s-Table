# Project Context: D&D AI Dungeon Master Service

## Overview
This project is a Python-based microservice designed to act as the "Brain" of an AI-powered Dungeons & Dragons campaign manager. It utilizes a local Large Language Model (Llama 3.2) to act as the Dungeon Master (DM), generating narrative responses, tracking game state, and coordinating with other media services (Image Generation and Text-to-Speech).

## Core Functionality
1.  **Narrative Generation:** Uses `meta-llama/Llama-3.2-3B-Instruct` to generate immersive roleplay responses based on user input.
2.  **Context Management:** Maintains campaign continuity by retrieving conversation history, story segments, active quests, and character details from a PostgreSQL database before every generation.
3.  **Rule Integration:** Fetches real D&D 5e rules via the external `dnd5eapi.co` to ensure rule accuracy in responses.
4.  **Media Orchestration:**
    *   Triggers image generation for scenes.
    *   Triggers speech synthesis (TTS) for the DM's voice.
5.  **Campaign Persistence:** Stores all interactions, campaigns, and world-building elements in a structured relational database.

## Architecture
*   **Service Type:** FastAPI REST API.
*   **Database:** PostgreSQL (using `psycopg2` driver with connection pooling).
*   **ML Engine:** Hugging Face Transformers & PyTorch.
*   **Infrastructure:** Dockerized container, intended to run within a docker-compose network.

## External Dependencies
*   **Internal Microservices:**
    *   `database`: PostgreSQL container.
    *   `image_gen`: Service on port 8001.
    *   `tts_voice`: Service on port 8002.
*   **External APIs:**
    *   `dnd5eapi.co`: For D&D 5th Edition rules lookup.

## Environment Variables
*   **Database:** `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_PORT`.
*   **Model:** `HF_TOKEN` (Hugging Face token), `HF_HOME` (Cache directory).