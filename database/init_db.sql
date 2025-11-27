-- ./database/init_db.sql
-- PostgreSQL initialization script for D&D Campaign Database
-- Uses environment variables for passwords (set in docker-compose.yml)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Get app user credentials from environment variables
-- These are passed from docker-compose.yml which reads from .env
DO $$
DECLARE
    app_user TEXT := COALESCE(current_setting('env.APP_DB_USER', true), 'llama_user');
    app_pass TEXT := COALESCE(current_setting('env.APP_DB_PASSWORD', true), 'change_me_default_pass');
BEGIN
    -- Create the application user if it doesn't exist
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = app_user) THEN
        EXECUTE format('CREATE USER %I WITH PASSWORD %L', app_user, app_pass);
        RAISE NOTICE 'Created user: %', app_user;
    ELSE
        -- Update password if user exists
        EXECUTE format('ALTER USER %I WITH PASSWORD %L', app_user, app_pass);
        RAISE NOTICE 'Updated password for user: %', app_user;
    END IF;
    
    -- Grant connect privilege
    EXECUTE format('GRANT CONNECT ON DATABASE dungeon_data TO %I', app_user);
    
    -- Grant schema usage
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', app_user);
    
    -- Grant table privileges (for future tables)
    EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I', app_user);
    
    -- Grant sequence privileges (for auto-increment IDs)
    EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO %I', app_user);
END $$;

-- ============================================
-- CAMPAIGN STORIES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS campaigns (
    id SERIAL PRIMARY KEY,
    campaign_name VARCHAR(255) NOT NULL,
    campaign_description TEXT,
    dm_style VARCHAR(50) DEFAULT 'balanced', -- 'combat', 'roleplay', 'exploration', 'balanced'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- ============================================
-- STORY SEGMENTS TABLE
-- Store individual story beats/scenes
-- ============================================
CREATE TABLE IF NOT EXISTS story_segments (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    segment_order INTEGER NOT NULL, -- Track the order of story segments
    title VARCHAR(255),
    content TEXT NOT NULL,
    scene_type VARCHAR(50), -- 'combat', 'dialogue', 'exploration', 'puzzle', 'rest'
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- CONVERSATION HISTORY TABLE
-- Store user prompts and AI responses
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    user_prompt TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    response_tokens INTEGER, -- Track token usage
    image_generated BOOLEAN DEFAULT false,
    audio_generated BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- VECTOR EMBEDDINGS TABLE
-- Store semantic embeddings for RAG (Retrieval Augmented Generation)
-- ============================================
CREATE TABLE IF NOT EXISTS story_embeddings (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    segment_id INTEGER REFERENCES story_segments(id) ON DELETE CASCADE,
    content_text TEXT NOT NULL,
    embedding vector(384), -- 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster vector similarity search
CREATE INDEX IF NOT EXISTS story_embeddings_vector_idx 
ON story_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ============================================
-- CHARACTERS TABLE
-- Track NPCs and player characters
-- ============================================
CREATE TABLE IF NOT EXISTS characters (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    character_type VARCHAR(20) NOT NULL, -- 'pc' or 'npc'
    race VARCHAR(100),
    class VARCHAR(100),
    level INTEGER DEFAULT 1,
    description TEXT,
    personality_traits TEXT,
    backstory TEXT,
    current_status VARCHAR(50) DEFAULT 'active', -- 'active', 'injured', 'dead', 'absent'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- LOCATIONS TABLE
-- Track important places in the campaign
-- ============================================
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    location_type VARCHAR(50), -- 'city', 'dungeon', 'wilderness', 'tavern', etc.
    description TEXT,
    discovered BOOLEAN DEFAULT false,
    visited_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- QUEST LOG TABLE
-- Track active and completed quests
-- ============================================
CREATE TABLE IF NOT EXISTS quests (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    quest_giver VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'failed', 'abandoned'
    reward TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- ============================================
-- GRANT PERMISSIONS TO APPLICATION USER
-- ============================================
DO $$
DECLARE
    app_user TEXT := COALESCE(current_setting('env.APP_DB_USER', true), 'llama_user');
BEGIN
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I', app_user);
    EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', app_user);
    RAISE NOTICE 'Granted permissions to: %', app_user;
END $$;

-- ============================================
-- CREATE USEFUL VIEWS
-- ============================================

-- View: Recent conversation history with campaign context
CREATE OR REPLACE VIEW recent_conversations AS
SELECT 
    c.id,
    c.campaign_id,
    cam.campaign_name,
    c.user_prompt,
    c.ai_response,
    c.created_at
FROM conversations c
JOIN campaigns cam ON c.campaign_id = cam.id
ORDER BY c.created_at DESC
LIMIT 100;

-- View: Active campaign summary
CREATE OR REPLACE VIEW active_campaigns_summary AS
SELECT 
    c.id,
    c.campaign_name,
    c.dm_style,
    COUNT(DISTINCT s.id) as story_segments,
    COUNT(DISTINCT ch.id) as characters,
    COUNT(DISTINCT q.id) as active_quests,
    c.created_at,
    c.updated_at
FROM campaigns c
LEFT JOIN story_segments s ON c.id = s.campaign_id
LEFT JOIN characters ch ON c.id = ch.campaign_id
LEFT JOIN quests q ON c.id = q.campaign_id AND q.status = 'active'
WHERE c.is_active = true
GROUP BY c.id, c.campaign_name, c.dm_style, c.created_at, c.updated_at;

-- Grant view permissions
DO $$
DECLARE
    app_user TEXT := COALESCE(current_setting('env.APP_DB_USER', true), 'llama_user');
BEGIN
    EXECUTE format('GRANT SELECT ON recent_conversations TO %I', app_user);
    EXECUTE format('GRANT SELECT ON active_campaigns_summary TO %I', app_user);
END $$;

-- ============================================
-- INSERT SAMPLE DATA (OPTIONAL)
-- ============================================

-- Create a starter campaign
INSERT INTO campaigns (campaign_name, campaign_description, dm_style) 
VALUES (
    'The Lost Mines of Phandelver',
    'A classic adventure of goblins, dragons, and hidden treasure in the Sword Coast.',
    'balanced'
)
ON CONFLICT DO NOTHING;

-- Welcome message
DO $$
BEGIN
    RAISE NOTICE '==============================================';
    RAISE NOTICE 'âœ" Database initialized successfully!';
    RAISE NOTICE 'âœ" pgvector extension enabled';
    RAISE NOTICE 'âœ" Application user created with CRUD permissions';
    RAISE NOTICE 'âœ" Tables: campaigns, story_segments, conversations,';
    RAISE NOTICE '           story_embeddings, characters, locations, quests';
    RAISE NOTICE 'âœ" Sample campaign created';
    RAISE NOTICE 'âœ" Passwords loaded from environment variables';
    RAISE NOTICE '==============================================';
END $$;