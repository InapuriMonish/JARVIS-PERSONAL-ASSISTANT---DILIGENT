"""
Configuration management for Enterprise JARVIS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "enterprise-jarvis")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

# Document Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# LLM Configuration - Using Qwen 2.5 (optimized for RAG)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "Enterprise JARVIS")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

# Validation
if not PINECONE_API_KEY:
    print("⚠️  WARNING: PINECONE_API_KEY not found in .env file")


def validate_config():
    """Validate all required configurations"""
    errors = []
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is not set")
    
    if not RAW_DOCS_DIR.exists():
        errors.append(f"Raw documents directory does not exist: {RAW_DOCS_DIR}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
    
    print("✅ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print(f"\n📊 Current Configuration:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Raw Docs: {RAW_DOCS_DIR}")
    print(f"  Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  LLM Model: {LLM_MODEL}")
