# Module Map

## Directory: ./database

### 1. `init_db.sql`
The primary initialization script executed when the PostgreSQL container starts. It sets up the environment, schema, and security.

*   **System Setup**
    *   `Extension`: Enables `pgvector` for embedding storage.
    *   `User Config`: Dynamically creates/updates the application user (`llama_user` default) using `docker-compose` env vars.
    *   `Permissions`: Grants schema usage and default privileges for future table creation.
*   **Core Tables**
    *   `campaigns`: The parent table for all game instances. Tracks DM style and activity status.
    *   `conversations`: Stores raw User/AI interaction logs, token usage, and media generation flags.
    *   `story_segments`: Stores summarized narrative blocks (beats/scenes) for context management.
*   **Vector Storage**
    *   `story_embeddings`: Stores 384-dimensional vectors linked to story segments.
    *   `Index`: Creates an `ivfflat` index for efficient cosine similarity search.
*   **World Building Tables**
    *   `characters`: Tracks NPCs/PCs, including stats, race, class, and status (active/dead).
    *   `locations`: Tracks discovery status and visitation counts of in-game places.
    *   `quests`: Tracks objectives, rewards, and completion status.
*   **Views**
    *   `recent_conversations`: readable join of campaigns and chat logs.
    *   `active_campaigns_summary`: aggregated statistics (counts of chars, quests, segments) for dashboarding.
*   **Seed Data**
    *   `Insertion`: Adds "The Lost Mines of Phandelver" as a default starter campaign.