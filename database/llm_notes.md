# LLM Notes & Developer Guide

## Database Logic & Constraints
*   **Vector Dimensions:** The `story_embeddings` table uses `vector(384)`.
    *   **Crucial:** The Python application *must* use an embedding model that outputs exactly 384 dimensions (e.g., `sentence-transformers/all-MiniLM-L6-v2`). Using `text-embedding-ada-002` (1536 dims) or others will cause SQL insertion errors.
*   **Foreign Keys:** All gameplay tables (`characters`, `quests`, `conversations`, etc.) utilize `ON DELETE CASCADE` referencing `campaigns(id)`.
    *   **Implication:** To wipe a game save, you only need to run `DELETE FROM campaigns WHERE id = X`. The DB handles the cleanup.

## Environment Dependencies
*   **Docker Integration:** This script uses PostgreSQL `current_setting` to read custom configuration variables (`env.APP_DB_USER`, `env.APP_DB_PASSWORD`).
*   **Setup:** These variables must be passed in the `docker-compose.yml` under the `db` service, usually via the `POSTGRES_INITDB_ARGS` or specific config file mounts, otherwise the `DO $$` block for user creation will default to `llama_user` / `change_me_default_pass`.

## RAG Implementation Details
*   **Indexing strategy:** The script creates an `ivfflat` index (`lists = 100`).
    *   **Note:** In a production scenario, `ivfflat` indexes are best built *after* some data exists. Since this runs at init (empty DB), the index is created empty. It will still work, but if the dataset grows massive, a `REINDEX` might be required later to optimize the cluster centers.
*   **Retrieval:** The `story_embeddings` table links to `story_segments`. The intended RAG flow is:
    1.  Search `story_embeddings` by vector similarity.
    2.  Join with `story_segments` to get the readable text (`content`).

## Development Tips
*   **Views:** Use `SELECT * FROM recent_conversations` for debugging chat flows rather than joining tables manually.
*   **Media Flags:** The `conversations` table has `image_generated` and `audio_generated` booleans. These are intended for the frontend to know if it should fetch assets from the file server, not for storing the actual binary data.