# 🤖 Enterprise JARVIS - AI Knowledge Assistant

> **RAG-powered Enterprise Knowledge Assistant with Document Management**  
> Built with Qwen 2.5 + Pinecone + Streamlit

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Qwen](https://img.shields.io/badge/LLM-Qwen_2.5-green.svg)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

---

## 🌟 Features

- ✅ **Document Upload** - Upload PDF, DOCX, TXT, MD files through the UI
- ✅ **Store to RAG** - Process and store documents in Pinecone with one click
- ✅ **Smart Chat** - Ask questions and get answers ONLY from your documents
- ✅ **No Hallucination** - If info isn't in documents, JARVIS tells you to upload them
- ✅ **Document Collection** - View all documents in the RAG system
- ✅ **Delete Documents** - Remove individual documents or clear all
- ✅ **Source Citations** - Every answer shows which documents were used
- ✅ **Premium UI** - Modern dark theme with beautiful design

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   📤 Upload  │ │  💬 Chat     │ │ 📚 Collection │           │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG ENGINE                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Document    │ │  Embedding   │ │   Vector     │            │
│  │  Processor   │ │  Generator   │ │   Store      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Pinecone      │ │  SentenceTrans  │ │   Qwen 2.5      │
│   Vector DB     │ │  formers        │ │   via Ollama    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 📁 Project Structure

```
JARVIS PERSONAL ASSISTANT/
│
├── data/
│   ├── raw_documents/          # Upload documents here
│   │   ├── employee_handbook.txt
│   │   ├── hr_policies.txt
│   │   ├── company_benefits.txt
│   │   ├── company_policies.txt
│   │   └── faq_document.txt
│   └── processed/              # Auto-generated chunks
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── document_processor.py   # PDF/DOCX/TXT processing
│   ├── embeddings.py           # SentenceTransformers
│   ├── vector_store.py         # Pinecone (with delete support)
│   ├── llm_handler.py          # Qwen 2.5 via Ollama
│   └── rag_engine.py           # RAG orchestration
│
├── app/
│   └── streamlit_app.py        # Full-featured UI
│
├── scripts/
│   ├── setup_vectordb.py       # Initial document ingestion
│   └── test_system.py          # Testing script
│
├── requirements.txt
├── .env.example
├── .env                        # Your API keys
└── README.md
```

---

## 🚀 Quick Start (20-30 Minutes)

### Step 1: Prerequisites

```bash
# Check Python version (need 3.9+)
python3 --version

# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull Qwen 2.5 model (in new terminal)
ollama pull qwen2.5:7b
```

### Step 2: Get Pinecone API Key

1. Go to https://www.pinecone.io/
2. Sign up for free account
3. Create an API key
4. Copy the key

### Step 3: Setup Project

```bash
# Navigate to project
cd "/Users/akshayk/JARVIS PERSONAL ASSISTANT"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Edit .env and add your Pinecone API key
nano .env
# Change: PINECONE_API_KEY=your_actual_key_here
```

### Step 4: Launch the App

```bash
# Start Streamlit
streamlit run app/streamlit_app.py
```

Open: **http://localhost:8501** 🎉

---

## 📱 Using the App

### Tab 1: 💬 Chat
- Ask questions about your documents
- Get answers with source citations
- Uses ONLY information from uploaded documents
- If info not found, asks you to upload relevant docs

### Tab 2: 📤 Upload Documents
- Drag & drop or click to upload files
- Supports: PDF, DOCX, TXT, MD
- Click "Store to RAG" to process and embed
- Wait for confirmation

### Tab 3: 📚 Document Collection
- View all documents in the system
- See chunk counts per document
- Delete individual documents
- Delete all documents

---

## 🧪 Test Questions

After uploading documents, try:

1. "What is the vacation policy?"
2. "How many sick days do I get?"
3. "What's the dress code on Fridays?"
4. "How do I submit expenses?"
5. "What are the working hours?"

---

## ⚙️ Configuration

Edit `.env` file:

```bash
# Pinecone
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=enterprise-jarvis

# LLM (Qwen 2.5 via Ollama)
LLM_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=512

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

---

## 🔧 Troubleshooting

### "Ollama connection refused"
```bash
ollama serve
```

### "Model qwen2.5:7b not found"
```bash
ollama pull qwen2.5:7b
```

### "PINECONE_API_KEY not found"
```bash
# Edit .env file
nano .env
# Add your key
```

### "No documents found"
- Upload documents in the "Upload Documents" tab
- Or add files to `data/raw_documents/`

---

## 🎯 Key Features Explained

### No Hallucination
The LLM is strictly instructed to:
- ONLY use information from provided documents
- Say "I don't have this information" if not found
- Ask users to upload relevant documents

### Document Management
- Upload via UI (drag & drop)
- View all documents in collection
- Delete individual documents
- Clear entire collection

### Source Citations
Every answer includes:
- Source document name
- Relevance score
- Text snippet used

---

## 📝 License

MIT License

---

**Built with ❤️ for Enterprise Knowledge Management**
