"""
Database helper functions for Supabase integration
"""
import os
import json
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(override=True)

def get_supabase_config():
    """Get Supabase configuration from Streamlit secrets or .env file"""
    try:
        # Try Streamlit secrets first (for deployment)
        import streamlit as st
        if hasattr(st, 'secrets'):
            return {
                'url': st.secrets.get('SUPABASE_URL', os.getenv('SUPABASE_URL')),
                'key': st.secrets.get('SUPABASE_KEY', os.getenv('SUPABASE_KEY')),
                'db_url': st.secrets.get('SUPABASE_DB_URL', os.getenv('SUPABASE_DB_URL'))
            }
    except:
        pass
    
    # Fall back to environment variables
    return {
        'url': os.getenv('SUPABASE_URL'),
        'key': os.getenv('SUPABASE_KEY'),
        'db_url': os.getenv('SUPABASE_DB_URL')
    }

class SupabaseVectorStore:
    def __init__(self):
        config = get_supabase_config()
        self.url = config['url']
        self.key = config['key']
        self.supabase: Client = create_client(self.url, self.key)
    
    def store_embeddings(self, grade, subject, documents, embeddings):
        """Store documents and embeddings in Supabase"""
        try:
            # Prepare data for insertion
            data = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                data.append({
                    'grade': grade,
                    'subject': subject,
                    'chunk_id': f"{grade}_{subject}_{i}",
                    'page_content': doc.page_content,
                    'metadata': json.dumps(doc.metadata),
                    'embedding': embedding.tolist()  # Convert numpy array to list
                })
            
            # Insert into database
            result = self.supabase.table('embeddings').insert(data).execute()
            print(f"Stored {len(data)} embeddings for {grade}_{subject}")
            return True
            
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False
    
    def get_embeddings(self, grade, subject):
        """Retrieve embeddings for a specific grade and subject"""
        try:
            result = self.supabase.table('embeddings').select('*').eq('grade', grade).eq('subject', subject).execute()
            return result.data
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            return []
    
    def search_similar(self, query_embedding, grade, subject, limit=3):
        """Search for similar embeddings using vector similarity"""
        # Use fallback search since RPC function doesn't exist yet
        return self._fallback_search(query_embedding, grade, subject, limit)
    
    def _fallback_search(self, query_embedding, grade, subject, limit=3):
        """Fallback search method"""
        try:
            # Get all embeddings for the grade/subject
            all_data = self.get_embeddings(grade, subject)
            
            if not all_data:
                return []
            
            # Convert query embedding to numpy array if needed
            if not hasattr(query_embedding, 'shape'):
                query_embedding = np.array(query_embedding)
            
            # Calculate similarities
            similarities = []
            for item in all_data:
                # Convert string embedding to numpy array
                if isinstance(item['embedding'], str):
                    # Parse the string representation of the array
                    import json
                    stored_embedding = np.array(json.loads(item['embedding']), dtype=float)
                else:
                    stored_embedding = np.array(item['embedding'], dtype=float)
                
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append((similarity, item))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in similarities[:limit]]
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return []
    
    def clear_embeddings(self, grade=None, subject=None):
        """Clear embeddings from database"""
        try:
            if grade and subject:
                self.supabase.table('embeddings').delete().eq('grade', grade).eq('subject', subject).execute()
                print(f"Cleared embeddings for {grade}_{subject}")
            else:
                self.supabase.table('embeddings').delete().neq('id', 0).execute()
                print("Cleared all embeddings")
            return True
        except Exception as e:
            print(f"Error clearing embeddings: {e}")
            return False

def create_similarity_function():
    """Create the vector similarity function in Supabase"""
    sql = """
    CREATE OR REPLACE FUNCTION match_embeddings (
        query_embedding vector(768),
        match_threshold float,
        match_count int,
        grade_filter text,
        subject_filter text
    )
    RETURNS TABLE (
        id int,
        grade text,
        subject text,
        chunk_id text,
        page_content text,
        metadata jsonb,
        similarity float
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            embeddings.id,
            embeddings.grade,
            embeddings.subject,
            embeddings.chunk_id,
            embeddings.page_content,
            embeddings.metadata,
            1 - (embeddings.embedding <=> query_embedding) as similarity
        FROM embeddings
        WHERE embeddings.grade = grade_filter
        AND embeddings.subject = subject_filter
        AND 1 - (embeddings.embedding <=> query_embedding) > match_threshold
        ORDER BY embeddings.embedding <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    """
    return sql
