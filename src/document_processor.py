"""
Document Processing Module for Enterprise JARVIS
Handles loading, parsing, and chunking of various document formats
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Document loaders
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import RAW_DOCS_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """
    Process documents into optimized chunks for embedding and retrieval.
    
    Supports: PDF, DOCX, TXT, MD files
    Uses RecursiveCharacterTextSplitter for intelligent chunking.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 500)
            chunk_overlap: Overlap between chunks for context continuity (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with optimal separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        print(f"📄 Document Processor initialized")
        print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    def load_pdf(self, file_path: Path) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            print(f"❌ Error loading PDF {file_path.name}: {e}")
            return ""
    
    def load_docx(self, file_path: Path) -> str:
        """
        Extract text content from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        
        except Exception as e:
            print(f"❌ Error loading DOCX {file_path.name}: {e}")
            return ""
    
    def load_txt(self, file_path: Path) -> str:
        """
        Load text content from TXT or MD file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        except Exception as e:
            print(f"❌ Error loading TXT {file_path.name}: {e}")
            return ""
    
    def load_document(self, file_path: Path) -> str:
        """
        Load document based on file extension.
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted text content
        """
        extension = file_path.suffix.lower()
        
        loaders = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.txt': self.load_txt,
            '.md': self.load_txt,
            '.text': self.load_txt
        }
        
        loader = loaders.get(extension)
        if loader:
            return loader(file_path)
        else:
            print(f"⚠️  Unsupported format: {extension} ({file_path.name})")
            return ""
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Full text content
            source: Source filename
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        return [
            {
                "text": chunk.strip(),
                "source": source,
                "chunk_id": f"{source.replace('.', '_').replace(' ', '_')}__chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i, chunk in enumerate(chunks)
            if chunk.strip()  # Skip empty chunks
        ]
    
    def process_all_documents(self, docs_dir: Optional[Path] = None) -> List[Dict[str, str]]:
        """
        Process all documents in the specified directory.
        
        Args:
            docs_dir: Directory containing documents (default: RAW_DOCS_DIR)
            
        Returns:
            List of all processed chunks
        """
        docs_dir = docs_dir or RAW_DOCS_DIR
        all_chunks = []
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.text'}
        
        # Find all supported files
        files = [
            f for f in docs_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files:
            print(f"⚠️  No documents found in: {docs_dir}")
            print(f"   Supported formats: {', '.join(supported_extensions)}")
            return []
        
        print(f"\n📚 Processing {len(files)} document(s)...\n")
        
        for file_path in tqdm(files, desc="📄 Processing"):
            # Load document
            text = self.load_document(file_path)
            
            if text.strip():
                # Chunk document
                chunks = self.chunk_text(text, file_path.name)
                all_chunks.extend(chunks)
                print(f"  ✅ {file_path.name}: {len(chunks)} chunks")
            else:
                print(f"  ⚠️  {file_path.name}: No text extracted")
        
        print(f"\n{'='*50}")
        print(f"✅ Total documents processed: {len(files)}")
        print(f"✅ Total chunks created: {len(all_chunks)}")
        
        # Save chunks to JSON
        self._save_chunks(all_chunks)
        
        return all_chunks
    
    def _save_chunks(self, chunks: List[Dict[str, str]]):
        """Save processed chunks to JSON file"""
        output_file = PROCESSED_DIR / "chunks.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Chunks saved to: {output_file}")
    
    def load_processed_chunks(self) -> List[Dict[str, str]]:
        """
        Load previously processed chunks from JSON.
        
        Returns:
            List of chunk dictionaries
        """
        chunks_file = PROCESSED_DIR / "chunks.json"
        
        if not chunks_file.exists():
            print("⚠️  No processed chunks found.")
            print("   Run process_all_documents() first.")
            return []
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✅ Loaded {len(chunks)} chunks from cache")
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, str]]) -> Dict:
        """Get statistics about processed chunks"""
        if not chunks:
            return {"error": "No chunks provided"}
        
        text_lengths = [len(c["text"]) for c in chunks]
        sources = set(c["source"] for c in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_sources": len(sources),
            "sources": list(sources),
            "avg_chunk_length": sum(text_lengths) / len(text_lengths),
            "min_chunk_length": min(text_lengths),
            "max_chunk_length": max(text_lengths)
        }


# Standalone testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    
    if chunks:
        stats = processor.get_chunk_stats(chunks)
        print(f"\n📊 Chunk Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n📝 Sample chunk:")
        print(f"   Source: {chunks[0]['source']}")
        print(f"   ID: {chunks[0]['chunk_id']}")
        print(f"   Text: {chunks[0]['text'][:200]}...")
