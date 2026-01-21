"""
RAG (Retrieval Augmented Generation) Engine
Orchestrates the entire query pipeline
"""
from typing import List, Dict, Tuple
import time

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler


class RAGEngine:
    """Main RAG orchestration engine"""
    
    def __init__(self):
        print("\n🚀 Initializing RAG Engine...\n")
        
        # Initialize components
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm = LLMHandler()
        
        # Connect to existing index
        try:
            self.vector_store.connect_index()
        except Exception as e:
            print(f"⚠️  Could not connect to index: {e}")
            print(f"   Run setup_vectordb.py first to create and populate the index")
        
        print("\n✅ RAG Engine initialized successfully!\n")
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        return_sources: bool = True
    ) -> Dict:
        """
        Process a query through the full RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            return_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        print(f"\n🔍 Processing query: '{question}'\n")
        start_time = time.time()
        
        # Step 1: Generate query embedding
        print("⏳ Step 1/3: Generating query embedding...")
        query_embedding = self.embedder.embed_text(question)
        print("   ✅ Query embedding generated")
        
        # Step 2: Search vector database
        print(f"⏳ Step 2/3: Searching vector database (top {top_k})...")
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question. Please check if the relevant documents have been uploaded to the collection.",
                "sources": [],
                "retrieved_chunks": 0,
                "processing_time": time.time() - start_time,
                "no_results": True
            }
        
        print(f"   ✅ Found {len(search_results)} relevant chunks")
        
        # Display relevance scores
        for i, result in enumerate(search_results, 1):
            print(f"      Chunk {i}: {result['source']} (score: {result['score']:.3f})")
        
        # Step 3: Generate answer using LLM
        print("⏳ Step 3/3: Generating answer with LLM...")
        
        # Extract context texts
        contexts = [result['text'] for result in search_results]
        
        # Generate response
        answer = self.llm.generate_response(
            context=contexts,
            question=question
        )
        
        print("   ✅ Answer generated")
        
        processing_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {processing_time:.2f}s\n")
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": search_results if return_sources else [],
            "retrieved_chunks": len(search_results),
            "processing_time": processing_time,
            "no_results": False
        }
        
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 3
    ) -> Dict:
        """
        Handle multi-turn conversations
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            top_k: Number of chunks to retrieve
            
        Returns:
            Response dictionary
        """
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return {
                "answer": "No user message found.",
                "sources": [],
                "retrieved_chunks": 0,
                "processing_time": 0
            }
        
        # Process as regular query
        return self.query(last_user_message, top_k=top_k)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            stats = self.vector_store.index.describe_index_stats()
            return {
                "total_vectors": stats['total_vector_count'],
                "index_name": self.vector_store.index_name,
                "dimension": self.vector_store.dimension,
                "embedding_model": self.embedder.model.get_sentence_embedding_dimension(),
                "llm_model": self.llm.model
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_documents(self) -> List[str]:
        """Get list of all unique document sources in the index"""
        try:
            return self.vector_store.get_all_sources()
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    def delete_document(self, source_name: str) -> bool:
        """Delete a document from the vector store"""
        try:
            return self.vector_store.delete_by_source(source_name)
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False


# Standalone testing
if __name__ == "__main__":
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Test queries
    test_queries = [
        "What is the company's vacation policy?",
        "How do I submit a leave request?",
        "What are the working hours?"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING RAG ENGINE")
    print("="*60)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        result = rag.query(query, top_k=3)
        
        print(f"\n📝 ANSWER:")
        print(result['answer'])
        
        print(f"\n📚 SOURCES ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n   {i}. {source['source']} (relevance: {source['score']:.3f})")
            print(f"      {source['text'][:150]}...")
        
        print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
        print("="*60)
