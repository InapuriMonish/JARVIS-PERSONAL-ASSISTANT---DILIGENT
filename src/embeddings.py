"""
Embedding Generator for Enterprise JARVIS
Uses SentenceTransformers to convert text into semantic vector embeddings
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingGenerator:
    """
    Generate semantic vector embeddings from text using SentenceTransformers.
    
    Embeddings capture the meaning of text, enabling semantic similarity search.
    Uses all-MiniLM-L6-v2 model (384 dimensions) - fast and accurate.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: SentenceTransformer model name
        """
        if EmbeddingGenerator._model is None:
            print(f"🔄 Loading embedding model: {model_name}...")
            EmbeddingGenerator._model = SentenceTransformer(model_name)
            print(f"✅ Embedding model loaded successfully")
            print(f"   Model: {model_name}")
            print(f"   Dimension: {EMBEDDING_DIMENSION}")
        
        self.model = EmbeddingGenerator._model
        self.dimension = EMBEDDING_DIMENSION
        self.model_name = model_name
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        print(f"\n🔄 Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="🧠 Embedding")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        return self.dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)


# Standalone testing
if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    
    # Test single embedding
    test_text = "What is the company vacation policy?"
    embedding = embedder.embed_text(test_text)
    
    print(f"\n📊 Single Embedding Test:")
    print(f"   Text: '{test_text}'")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    test_texts = [
        "Employee vacation policy information",
        "How many days off do employees get?",
        "The weather is sunny today",
        "Company holiday schedule"
    ]
    
    batch_embeddings = embedder.embed_batch(test_texts)
    
    # Test similarity
    print(f"\n📊 Similarity Test:")
    for i, text in enumerate(test_texts):
        similarity = embedder.compute_similarity(embedding, batch_embeddings[i])
        print(f"   '{test_text[:30]}...' vs '{text[:30]}...': {similarity:.3f}")
