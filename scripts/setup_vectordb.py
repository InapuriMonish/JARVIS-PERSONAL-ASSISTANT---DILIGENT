"""
One-time setup script to process documents and populate Pinecone
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import validate_config, RAW_DOCS_DIR
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


def main():
    """Main setup pipeline"""
    
    print("\n" + "="*70)
    print("🚀 ENTERPRISE JARVIS - VECTOR DATABASE SETUP")
    print("="*70 + "\n")
    
    # Step 1: Validate configuration
    print("📋 Step 1/5: Validating configuration...")
    try:
        validate_config()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return
    
    # Check for documents
    doc_files = list(RAW_DOCS_DIR.glob('*'))
    doc_files = [f for f in doc_files if f.suffix.lower() in ['.pdf', '.txt', '.docx', '.md']]
    
    if not doc_files:
        print(f"\n⚠️  ERROR: No documents found in {RAW_DOCS_DIR}")
        print(f"\n📝 To continue:")
        print(f"   1. Add your company documents (PDF, TXT, DOCX) to:")
        print(f"      {RAW_DOCS_DIR}")
        print(f"   2. Run this script again")
        return
    
    print(f"   ✅ Found {len(doc_files)} documents to process")
    
    # Step 2: Process documents
    print("\n📄 Step 2/5: Processing documents into chunks...")
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    
    if not chunks:
        print("\n❌ No chunks created. Please check your documents.")
        return
    
    # Step 3: Generate embeddings
    print("\n🧠 Step 3/5: Generating embeddings...")
    embedder = EmbeddingGenerator()
    
    # Extract texts from chunks
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32)
    
    # Step 4: Create/connect to Pinecone index
    print("\n🗄️  Step 4/5: Setting up Pinecone index...")
    vector_store = VectorStore()
    vector_store.create_index()
    
    # Step 5: Upload to Pinecone
    print("\n☁️  Step 5/5: Uploading to Pinecone...")
    vector_store.upsert_embeddings(chunks, embeddings, batch_size=100)
    
    # Summary
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   - Documents processed: {len(doc_files)}")
    print(f"   - Total chunks created: {len(chunks)}")
    print(f"   - Embeddings generated: {len(embeddings)}")
    print(f"   - Vectors uploaded to Pinecone: {len(embeddings)}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Test the system: python scripts/test_system.py")
    print(f"   2. Launch the UI: streamlit run app/streamlit_app.py")
    print("\n")


if __name__ == "__main__":
    main()
