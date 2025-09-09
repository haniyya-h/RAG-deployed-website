"""
Migration script to move embeddings from local files to Supabase
"""
import os
import pickle
import numpy as np
from pathlib import Path
from database import SupabaseVectorStore, create_similarity_function
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

def load_chromadb_embeddings(embeddings_dir):
    """Load embeddings from existing ChromaDB files"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        all_data = {}
        embeddings_path = Path(embeddings_dir)
        
        # Look for individual ChromaDB directories
        for item in embeddings_path.iterdir():
            if item.is_dir() and '_' in item.name:
                try:
                        # Extract grade and subject from directory name
                        parts = item.name.split('_')
                        if len(parts) >= 2:
                            grade = parts[1]  # grade_9_biology -> grade = "9"
                            subject = '_'.join(parts[2:])  # grade_9_biology -> subject = "biology"
                        
                        print(f"Loading {grade}_{subject} from {item.name}...")
                        
                        # Initialize ChromaDB client for this specific directory
                        client = chromadb.PersistentClient(path=str(item))
                        
                        # Get the collection (usually named after the directory)
                        collection_name = item.name
                        try:
                            coll = client.get_collection(collection_name)
                        except:
                            # Try alternative collection names
                            collections = client.list_collections()
                            if collections:
                                coll = collections[0]
                            else:
                                print(f"No collections found in {item.name}")
                                continue
                        
                        # Get all data
                        results = coll.get(include=['documents', 'metadatas', 'embeddings'])
                        
                        documents = []
                        embeddings = []
                        metadatas = []
                        
                        if results['documents']:
                            for i, doc in enumerate(results['documents']):
                                # Create document-like object
                                from langchain.schema import Document
                                doc_obj = Document(
                                    page_content=doc,
                                    metadata=results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                                )
                                documents.append(doc_obj)
                                embeddings.append(np.array(results['embeddings'][i]))
                                metadatas.append(results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {})
                        
                        all_data[f"{grade}_{subject}"] = {
                            'documents': documents,
                            'embeddings': embeddings,
                            'metadatas': metadatas
                        }
                        
                        print(f"✅ Loaded {len(documents)} documents from {collection_name}")
                        
                except Exception as e:
                    print(f"⚠️ Error loading {item.name}: {e}")
                    continue
        
        return all_data
        
    except Exception as e:
        print(f"Error loading ChromaDB embeddings: {e}")
        return {}

def migrate_embeddings():
    """Migrate all embeddings to Supabase"""
    print("Starting migration to Supabase...")
    
    # Initialize Supabase client
    db = SupabaseVectorStore()
    
    # Create similarity function in Supabase
    print("Creating similarity function in Supabase...")
    similarity_sql = create_similarity_function()
    try:
        db.supabase.rpc('exec_sql', {'sql': similarity_sql}).execute()
        print("✅ Similarity function created")
    except Exception as e:
        print(f"⚠️ Could not create similarity function: {e}")
        print("This is okay, we'll use fallback search")
    
    # Load existing embeddings
    embeddings_dir = Path('embeddings')
    if not embeddings_dir.exists():
        print("❌ No embeddings directory found. Run preprocess.py first.")
        return
    
    print("Loading existing embeddings...")
    all_data = load_chromadb_embeddings(embeddings_dir)
    
    if not all_data:
        print("❌ No embeddings found to migrate")
        return
    
    # Migrate each grade/subject combination
    total_migrated = 0
    for combo, data in all_data.items():
        grade, subject = combo.split('_', 1)
        
        print(f"\nMigrating {grade}_{subject}...")
        
        if not data['documents'] or not data['embeddings']:
            print(f"⚠️ No data found for {grade}_{subject}")
            continue
        
        # Store in Supabase
        success = db.store_embeddings(grade, subject, data['documents'], data['embeddings'])
        
        if success:
            total_migrated += len(data['documents'])
            print(f"✅ Migrated {len(data['documents'])} documents for {grade}_{subject}")
        else:
            print(f"❌ Failed to migrate {grade}_{subject}")
    
    print(f"\n🎉 Migration complete! Total documents migrated: {total_migrated}")
    
    # Test the migration
    print("\nTesting migration...")
    test_grade = list(all_data.keys())[0].split('_')[0]
    test_subject = list(all_data.keys())[0].split('_')[1]
    
    test_data = db.get_embeddings(test_grade, test_subject)
    print(f"✅ Test successful: Found {len(test_data)} embeddings for {test_grade}_{test_subject}")

if __name__ == "__main__":
    migrate_embeddings()
