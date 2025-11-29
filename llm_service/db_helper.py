# ./llm_service/db_helper.py
"""
Database helper module for D&D Campaign management.
FIXED: Now matches docker-compose.yml credentials
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # FIXED: Match docker-compose.yml environment variables
        self.config = {
            'host': os.getenv('DB_HOST', 'database'),
            'database': os.getenv('DB_NAME', 'dungeon_data'),
            'user': os.getenv('DB_USER', 'dm_admin'),  # FIXED: was 'llama_user'
            'password': os.getenv('DB_PASS', 'secretpassword'),  # FIXED: was 'llama_secret_pass'
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Connection pool for better performance
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **self.config
            )
            logger.info(f"✓ Database pool initialized: {self.config['user']}@{self.config['host']}/{self.config['database']}")
        except Exception as e:
            logger.error(f"✗ Failed to create connection pool: {e}")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with pooling"""
        conn = None
        try:
            if self.pool:
                conn = self.pool.getconn()
            else:
                # Fallback to direct connection if pool failed
                conn = psycopg2.connect(**self.config)
            
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                if self.pool:
                    self.pool.putconn(conn)
                else:
                    conn.close()
    
    def test_connection(self):
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"✓ Database connected: {version[:50]}...")
                    return True
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
    
    def close_pool(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("✓ Database connection pool closed")
    
    # ========================================
    # CAMPAIGN OPERATIONS
    # ========================================
    
    def create_campaign(self, name, description="", dm_style="balanced"):
        """Create a new campaign"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO campaigns (campaign_name, campaign_description, dm_style)
                    VALUES (%s, %s, %s)
                    RETURNING id, campaign_name, created_at
                """, (name, description, dm_style))
                result = cur.fetchone()
                logger.info(f"✓ Created campaign: {result['campaign_name']} (ID: {result['id']})")
                return dict(result)
    
    def get_active_campaign(self):
        """Get the currently active campaign"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM campaigns 
                    WHERE is_active = true 
                    ORDER BY updated_at DESC 
                    LIMIT 1
                """)
                result = cur.fetchone()
                return dict(result) if result else None
    
    def get_campaign_by_id(self, campaign_id):
        """Get campaign details by ID"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    
    # ========================================
    # CONVERSATION HISTORY
    # ========================================
    
    def save_conversation(self, campaign_id, user_prompt, ai_response, 
                         response_tokens=0, image_generated=False, audio_generated=False):
        """Save a conversation turn to history"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO conversations 
                    (campaign_id, user_prompt, ai_response, response_tokens, 
                     image_generated, audio_generated)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                """, (campaign_id, user_prompt, ai_response, response_tokens, 
                      image_generated, audio_generated))
                result = cur.fetchone()
                logger.info(f"✓ Saved conversation (ID: {result['id']})")
                return dict(result)
    
    def get_recent_conversations(self, campaign_id, limit=10):
        """Get recent conversation history for context"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_prompt, ai_response, created_at
                    FROM conversations
                    WHERE campaign_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (campaign_id, limit))
                results = cur.fetchall()
                return [dict(row) for row in reversed(results)]
    
    def get_conversation_context(self, campaign_id, limit=5):
        """Get formatted conversation context for prompt building"""
        history = self.get_recent_conversations(campaign_id, limit)
        context = []
        for conv in history:
            context.append(f"Player: {conv['user_prompt']}")
            context.append(f"DM: {conv['ai_response']}")
        return "\n".join(context)
    
    # ========================================
    # STORY SEGMENTS
    # ========================================
    
    def save_story_segment(self, campaign_id, content, title=None, 
                          scene_type=None, location=None):
        """Save a story segment/scene"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT COALESCE(MAX(segment_order), 0) + 1 as next_order
                    FROM story_segments
                    WHERE campaign_id = %s
                """, (campaign_id,))
                next_order = cur.fetchone()['next_order']
                
                cur.execute("""
                    INSERT INTO story_segments 
                    (campaign_id, segment_order, title, content, scene_type, location)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, segment_order
                """, (campaign_id, next_order, title, content, scene_type, location))
                result = cur.fetchone()
                logger.info(f"✓ Saved story segment #{result['segment_order']}")
                return dict(result)
    
    def get_story_so_far(self, campaign_id, limit=10):
        """Get recent story segments for context"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT title, content, scene_type, location, created_at
                    FROM story_segments
                    WHERE campaign_id = %s
                    ORDER BY segment_order DESC
                    LIMIT %s
                """, (campaign_id, limit))
                results = cur.fetchall()
                return [dict(row) for row in reversed(results)]
    
    # ========================================
    # CHARACTERS
    # ========================================
    
    def add_character(self, campaign_id, name, character_type, 
                     race=None, char_class=None, description=None):
        """Add a character (PC or NPC) to the campaign"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO characters 
                    (campaign_id, name, character_type, race, class, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, name
                """, (campaign_id, name, character_type, race, char_class, description))
                result = cur.fetchone()
                logger.info(f"✓ Added character: {result['name']}")
                return dict(result)
    
    def get_active_characters(self, campaign_id):
        """Get all active characters in the campaign"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM characters
                    WHERE campaign_id = %s AND current_status = 'active'
                    ORDER BY character_type, name
                """, (campaign_id,))
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    # ========================================
    # LOCATIONS
    # ========================================
    
    def add_location(self, campaign_id, name, location_type=None, description=None):
        """Add a location to the campaign"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO locations 
                    (campaign_id, name, location_type, description, discovered)
                    VALUES (%s, %s, %s, %s, true)
                    RETURNING id, name
                """, (campaign_id, name, location_type, description))
                result = cur.fetchone()
                logger.info(f"✓ Added location: {result['name']}")
                return dict(result)
    
    def visit_location(self, location_id):
        """Mark a location as visited (increment counter)"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE locations 
                    SET visited_count = visited_count + 1
                    WHERE id = %s
                """, (location_id,))
    
    # ========================================
    # QUESTS
    # ========================================
    
    def add_quest(self, campaign_id, title, description=None, quest_giver=None):
        """Add a new quest"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO quests 
                    (campaign_id, title, description, quest_giver)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, title
                """, (campaign_id, title, description, quest_giver))
                result = cur.fetchone()
                logger.info(f"✓ Added quest: {result['title']}")
                return dict(result)
    
    def complete_quest(self, quest_id):
        """Mark a quest as completed"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE quests 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (quest_id,))
    
    def get_active_quests(self, campaign_id):
        """Get all active quests"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM quests
                    WHERE campaign_id = %s AND status = 'active'
                    ORDER BY created_at
                """, (campaign_id,))
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    # ========================================
    # UTILITY FUNCTIONS
    # ========================================
    
    def get_campaign_summary(self, campaign_id):
        """Get a comprehensive summary of the campaign state"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM active_campaigns_summary
                    WHERE id = %s
                """, (campaign_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    
    def delete_conversation(self, conversation_id):
        """Delete a specific conversation"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
                logger.info(f"✓ Deleted conversation {conversation_id}")
    
    def clear_campaign_history(self, campaign_id):
        """Clear all conversation history for a campaign"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations WHERE campaign_id = %s", (campaign_id,))
                logger.info(f"✓ Cleared history for campaign {campaign_id}")


# Singleton instance
db = DatabaseManager()