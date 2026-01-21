"""
Pinecone vector database integration
Enhanced with document management capabilities
"""
import time
from typing import List, Dict, Optional, Set
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from src.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION
)


class VectorStore:
    """Manage Pinecone vector database with document management"""
    
    def __init__(self):
        print(f"🔄 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.dimension = EMBEDDING_DIMENSION
        self.index = None
    
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            print(f"✅ Index '{self.index_name}' already exists")
            self.index = self.pc.Index(self.index_name)
        else:
            print(f"🔄 Creating new index: {self.index_name}...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(5)
            
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Index created successfully")
    
    def connect_index(self):
        """Connect to existing index"""
        self.index = self.pc.Index(self.index_name)
        stats = self.index.describe_index_stats()
        print(f"✅ Connected to index: {self.index_name}")
        print(f"   Total vectors: {stats['total_vector_count']}")
    
    def upsert_embeddings(
        self,
        chunks: List[Dict[str, str]],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """Upload embeddings to Pinecone"""
        if not self.index:
            self.connect_index()
        
        print(f"\n🔄 Uploading {len(embeddings)} vectors to Pinecone...")
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"][:1000],  # Limit metadata size
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"]
                }
            })
        
        # Upload in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Upload complete!")
        
        # Show stats
        time.sleep(2)  # Wait for stats to update
        stats = self.index.describe_index_stats()
        print(f"   Total vectors in index: {stats['total_vector_count']}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar vectors"""
        if not self.index:
            self.connect_index()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        matches = []
        for match in results['matches']:
            matches.append({
                "id": match['id'],
                "score": match['score'],
                "text": match['metadata']['text'],
                "source": match['metadata']['source']
            })
        
        return matches
    
    def get_all_sources(self) -> List[Dict]:
        """Get all unique document sources with their chunk counts"""
        if not self.index:
            self.connect_index()
        
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                return []
            
            # Query a sample to get sources (Pinecone doesn't have direct list metadata)
            # We'll use a zero vector to get all documents
            zero_vector = [0.0] * self.dimension
            
            # Get up to 10000 results to find all sources
            results = self.index.query(
                vector=zero_vector,
                top_k=min(10000, total_vectors),
                include_metadata=True
            )
            
            # Count chunks per source
            source_counts = {}
            for match in results.get('matches', []):
                source = match['metadata'].get('source', 'Unknown')
                if source not in source_counts:
                    source_counts[source] = {
                        'name': source,
                        'chunks': 0,
                        'ids': []
                    }
                source_counts[source]['chunks'] += 1
                source_counts[source]['ids'].append(match['id'])
            
            return list(source_counts.values())
        
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    def delete_by_source(self, source_name: str) -> bool:
        """Delete all vectors for a specific source document"""
        if not self.index:
            self.connect_index()
        
        try:
            # Get all IDs for this source
            sources = self.get_all_sources()
            source_info = next((s for s in sources if s['name'] == source_name), None)
            
            if not source_info:
                print(f"⚠️ Source '{source_name}' not found")
                return False
            
            # Delete by IDs
            ids_to_delete = source_info['ids']
            
            # Delete in batches of 100
            for i in range(0, len(ids_to_delete), 100):
                batch = ids_to_delete[i:i + 100]
                self.index.delete(ids=batch)
            
            print(f"✅ Deleted {len(ids_to_delete)} chunks for '{source_name}'")
            return True
        
        except Exception as e:
            print(f"❌ Error deleting source: {e}")
            return False
    
    def delete_all(self):
        """Delete all vectors from index"""
        if not self.index:
            self.connect_index()
        
        print(f"⚠️  Deleting all vectors from {self.index_name}...")
        self.index.delete(delete_all=True)
        print(f"✅ All vectors deleted")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.index:
            self.connect_index()
        
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', self.dimension),
            'index_name': self.index_name
        }


# Standalone testing
if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.create_index()
    vector_store.connect_index()
    
    # Get all sources
    sources = vector_store.get_all_sources()
    print(f"\n📚 Documents in index:")
    for source in sources:
        print(f"   - {source['name']}: {source['chunks']} chunks")
