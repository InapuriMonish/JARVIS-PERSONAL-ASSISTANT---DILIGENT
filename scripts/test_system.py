"""
Test the RAG system with sample queries
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine


def main():
    """Run test queries"""
    
    print("\n" + "="*70)
    print("🧪 TESTING ENTERPRISE JARVIS")
    print("="*70 + "\n")
    
    # Initialize RAG engine
    print("🔄 Initializing RAG engine...")
    rag = RAGEngine()
    
    # Get statistics
    print("\n📊 System Statistics:")
    stats = rag.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test queries
    test_queries = [
        "What is the company vacation policy?",
        "How many sick leaves am I entitled to?",
        "What is the dress code?",
        "How do I submit an expense report?",
        "What are the working hours?",
    ]
    
    print("\n" + "="*70)
    print("🔍 RUNNING TEST QUERIES")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('='*70)
        
        result = rag.query(query, top_k=3)
        
        print(f"\n✨ ANSWER:")
        print(f"{result['answer']}\n")
        
        print(f"📚 SOURCES:")
        for j, source in enumerate(result['sources'], 1):
            print(f"   {j}. {source['source']} (relevance: {source['score']:.3f})")
        
        print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
        print('='*70)
        
        if i < len(test_queries):
            input("\nPress Enter to continue to next query...")
    
    print("\n✅ Testing complete!")
    print("\n🚀 To launch the UI, run:")
    print("   streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
