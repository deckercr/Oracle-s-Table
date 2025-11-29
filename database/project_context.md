# Project Context: Database Service

## Overview
The **Database Service** is the persistence layer for the D&D AI Dungeon Master application. It uses **PostgreSQL** extended with **pgvector** to handle both relational game data (campaigns, characters, quests) and semantic vector embeddings (for RAG/memory).

This container initializes the schema automatically upon startup, ensuring the application user has the correct permissions and the database structure supports the application's narrative flow.

## Architecture
*   **Database Engine:** PostgreSQL 15+
*   **Extensions:** `pgvector` (Vector similarity search).
*   **Initialization:** SQL script runs via Docker entrypoint (`/docker-entrypoint-initdb.d/`).
*   **Security:** User credentials injected via Environment Variables (`APP_DB_USER`, `APP_DB_PASSWORD`).

## Data Models
1.  **Core State:** `campaigns` act as the root object. All other data is linked to a campaign ID.
2.  **Narrative History:**
    *   `conversations`: Raw chat logs between User and AI.
    *   `story_segments`: Summarized story beats used for long-term context.
3.  **World Building:** `characters`, `locations`, `quests`.
4.  **AI Memory:** `story_embeddings` stores 384-dimensional vectors representing narrative segments for retrieval.

## Key Features
*   **Vector Search:** Native support for storing and searching embeddings to give the LLM "long-term memory."
*   **Cascading Deletes:** Deleting a campaign automatically cleans up all associated history, characters, and embeddings.
*   **Dynamic Auth:** The initialization script dynamically creates or updates the application user based on Docker environment variables, removing hardcoded credentials from the SQL.
*   **Audit/Metrics:** Tracks token usage and media generation flags per conversation turn.