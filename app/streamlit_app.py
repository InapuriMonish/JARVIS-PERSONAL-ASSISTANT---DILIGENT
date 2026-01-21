"""
Enterprise JARVIS - Comprehensive Streamlit Chat Interface
Features:
- Document Upload & Management
- Text Input for Direct RAG Storage
- Personalized Chat with User Name
- Store to RAG functionality
- Chat with AI
- View all documents in collection
- Delete documents
"""
import sys
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.config import APP_NAME, APP_VERSION, RAW_DOCS_DIR, PROCESSED_DIR


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=f"{APP_NAME} - Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - Premium Dark Theme
# ============================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Upload area */
    .upload-area {
        background: rgba(102, 126, 234, 0.1);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.15);
    }
    
    /* Document list */
    .doc-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    
    .doc-item:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .doc-name {
        color: #fff;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .doc-meta {
        color: #718096;
        font-size: 0.85rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: white;
        font-size: 0.95rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat styling */
    .chat-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        max-height: 500px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.75rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.75rem 0;
        max-width: 85%;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Welcome message */
    .welcome-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .user-greeting {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Source card */
    .source-card {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .source-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .source-text {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #718096;
        font-size: 0.85rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(102, 126, 234, 0.1);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: white;
        padding: 0.875rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Divider */
    .section-divider {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
        position: relative;
    }
    
    .section-divider span {
        background: #1a1a2e;
        padding: 0 1rem;
        color: #718096;
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #e2e8f0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None

if 'uploaded_files_pending' not in st.session_state:
    st.session_state.uploaded_files_pending = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'user_name' not in st.session_state:
    st.session_state.user_name = "Monish"  # Default name


# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def get_rag_engine():
    """Initialize RAG engine (cached)"""
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        return None


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to raw_documents directory"""
    file_path = RAW_DOCS_DIR / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def save_text_as_document(text: str, title: str) -> Path:
    """Save text content as a document file"""
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    filename = f"{safe_title}_{timestamp}.txt"
    
    file_path = RAW_DOCS_DIR / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return file_path


def process_and_store_documents(file_paths: list):
    """Process documents and store in vector database"""
    try:
        # Initialize components
        processor = DocumentProcessor()
        embedder = EmbeddingGenerator()
        vector_store = VectorStore()
        
        # Ensure index exists
        vector_store.create_index()
        
        all_chunks = []
        
        # Process each file
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.exists():
                text = processor.load_document(file_path)
                if text.strip():
                    chunks = processor.chunk_text(text, file_path.name)
                    all_chunks.extend(chunks)
        
        if not all_chunks:
            return False, "No text could be extracted from the documents"
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = embedder.embed_batch(texts, show_progress=False)
        
        # Store in Pinecone
        vector_store.upsert_embeddings(all_chunks, embeddings)
        
        return True, f"Successfully processed {len(all_chunks)} chunks"
    
    except Exception as e:
        return False, str(e)


def process_text_to_rag(text: str, title: str):
    """Process raw text and store directly in RAG"""
    try:
        # Save as document first
        file_path = save_text_as_document(text, title)
        
        # Process and store
        success, message = process_and_store_documents([file_path])
        
        return success, message, file_path.name
    
    except Exception as e:
        return False, str(e), None


def get_document_list():
    """Get list of documents in the RAG system"""
    try:
        vector_store = VectorStore()
        vector_store.connect_index()
        return vector_store.get_all_sources()
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []


def delete_document_from_rag(doc_name: str):
    """Delete a document from the RAG system"""
    try:
        vector_store = VectorStore()
        vector_store.connect_index()
        success = vector_store.delete_by_source(doc_name)
        
        # Also delete from raw_documents if exists
        file_path = RAW_DOCS_DIR / doc_name
        if file_path.exists():
            file_path.unlink()
        
        return success
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False


def generate_personalized_not_found_message(user_name: str) -> str:
    """Generate personalized message when info not found"""
    return f"""Hey **{user_name}**! 👋

I couldn't find any information related to your question in the private knowledge base.

**Here's what you can do:**
1. 📚 Go to the **"Document Collection"** tab to check what documents are currently available
2. 📤 Head to the **"Upload Documents"** tab to add the relevant documents or text
3. 🔄 Once uploaded, come back and ask me again!

I can only provide information from documents that have been uploaded to my knowledge base. This ensures accuracy and prevents me from making things up! 

Would you like me to help you with something else that might be covered in the existing documents?"""


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 JARVIS</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Enterprise Knowledge Assistant | Powered by RAG + Qwen 2.5 | v{APP_VERSION}</p>', unsafe_allow_html=True)
    
    # Sidebar for user settings
    with st.sidebar:
        st.markdown("## 👤 User Settings")
        
        # User name input
        user_name = st.text_input(
            "Your Name",
            value=st.session_state.user_name,
            placeholder="Enter your name",
            help="This will be used to personalize your experience"
        )
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("## 📊 Quick Stats")
        
        try:
            docs = get_document_list()
            total_docs = len(docs)
            total_chunks = sum(d.get('chunks', 0) for d in docs)
        except:
            total_docs = 0
            total_chunks = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", total_docs)
        with col2:
            st.metric("Chunks", total_chunks)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload Documents", "📚 Document Collection"])
    
    # ============================================================
    # TAB 1: CHAT INTERFACE
    # ============================================================
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Welcome message with personalization
            st.markdown(f"""
            <div class="welcome-card">
                <span class="user-greeting">👋 Welcome, {st.session_state.user_name}!</span>
                <p style="color: #a0aec0; margin-top: 0.5rem; margin-bottom: 0;">
                    I'm JARVIS, your Enterprise Knowledge Assistant. Ask me anything about your uploaded documents!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize RAG engine
            rag_engine = get_rag_engine()
            
            # Chat messages container
            chat_container = st.container()
            
            with chat_container:
                # Display welcome message if no messages
                if not st.session_state.messages:
                    st.markdown(f"""
                    <div class="assistant-message">
                        Hello <strong>{st.session_state.user_name}</strong>! 🤖<br><br>
                        I can help you find information from your uploaded documents. I will <strong>only</strong> answer based on the documents in my knowledge base - no hallucination, just facts!<br><br>
                        <strong>Try asking:</strong><br>
                        • "What is the vacation policy?"<br>
                        • "How do I submit expenses?"<br>
                        • "What are the working hours?"
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display chat history
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
                        
                        # Show sources if available
                        if "sources" in msg and msg["sources"]:
                            with st.expander("📚 View Sources", expanded=False):
                                for source in msg["sources"]:
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <div class="source-title">📄 {source.get('source', 'Unknown')} 
                                        <span style="color: #48bb78;">({source.get('score', 0):.1%} relevance)</span></div>
                                        <div class="source-text">{source.get('text', '')[:200]}...</div>
                                    </div>
                                    """, unsafe_allow_html=True)
            
            # Chat input
            st.markdown("---")
            
            col_input, col_btn = st.columns([5, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Ask a question",
                    placeholder=f"Ask me anything, {st.session_state.user_name}...",
                    key="chat_input",
                    label_visibility="collapsed"
                )
            
            with col_btn:
                send_clicked = st.button("Send 🚀", use_container_width=True)
            
            # Process user input
            if (send_clicked or user_input) and user_input and rag_engine:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                with st.spinner("🔍 Searching knowledge base..."):
                    try:
                        result = rag_engine.query(user_input, top_k=3)
                        
                        # Check if no results found or low relevance
                        if result.get('no_results', False) or (result.get('sources') and all(s.get('score', 0) < 0.3 for s in result.get('sources', []))):
                            answer = generate_personalized_not_found_message(st.session_state.user_name)
                            sources = []
                        else:
                            answer = result['answer']
                            sources = result.get('sources', [])
                            
                            # Check if answer indicates no info found
                            no_info_phrases = [
                                "couldn't find",
                                "don't have",
                                "no information",
                                "not found",
                                "not available",
                                "please upload"
                            ]
                            if any(phrase in answer.lower() for phrase in no_info_phrases):
                                answer = generate_personalized_not_found_message(st.session_state.user_name)
                                sources = []
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Sorry {st.session_state.user_name}, I encountered an error: {str(e)}. Please make sure documents have been uploaded and processed.",
                            "sources": []
                        })
                
                st.rerun()
        
        with col2:
            # Quick questions
            st.markdown("**💡 Quick Questions**")
            
            quick_qs = [
                "What is the vacation policy?",
                "How many sick days?",
                "What's the dress code?",
                "Working hours?",
                "How to submit expenses?"
            ]
            
            for q in quick_qs:
                if st.button(q, use_container_width=True, key=f"quick_{q}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    rag_engine = get_rag_engine()
                    if rag_engine:
                        result = rag_engine.query(q, top_k=3)
                        if result.get('no_results', False):
                            answer = generate_personalized_not_found_message(st.session_state.user_name)
                            sources = []
                        else:
                            answer = result['answer']
                            sources = result.get('sources', [])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    st.rerun()
    
    # ============================================================
    # TAB 2: UPLOAD DOCUMENTS
    # ============================================================
    with tab2:
        st.markdown('<div class="card-header">📤 Add Knowledge to RAG System</div>', unsafe_allow_html=True)
        
        # Create two sections: File Upload and Text Input
        upload_col, text_col = st.columns(2)
        
        # ---- SECTION 1: FILE UPLOAD ----
        with upload_col:
            st.markdown("""
            <div class="card">
                <div class="card-header">📁 Upload Files</div>
                <p style="color: #a0aec0; margin-bottom: 1rem;">
                    Upload documents to add to the knowledge base.<br>
                    Supports: <strong>PDF, DOCX, TXT, MD</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Drop files here or click to upload",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                key="file_uploader"
            )
            
            if uploaded_files:
                st.markdown("**📁 Files Ready:**")
                for file in uploaded_files:
                    st.markdown(f"• {file.name} ({file.size / 1024:.1f} KB)")
                
                # Store to RAG button
                if st.button("🚀 Store Files to RAG", use_container_width=True, key="store_files"):
                    with st.spinner("Processing files..."):
                        progress_bar = st.progress(0)
                        
                        # Save files
                        progress_bar.progress(25)
                        saved_paths = []
                        for file in uploaded_files:
                            path = save_uploaded_file(file)
                            saved_paths.append(path)
                        
                        # Process and store
                        progress_bar.progress(50)
                        success, message = process_and_store_documents(saved_paths)
                        progress_bar.progress(100)
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.cache_resource.clear()
                        else:
                            st.error(f"❌ Error: {message}")
        
        # ---- SECTION 2: TEXT INPUT ----
        with text_col:
            st.markdown("""
            <div class="card">
                <div class="card-header">📝 Paste Text Directly</div>
                <p style="color: #a0aec0; margin-bottom: 1rem;">
                    Paste or type text content directly.<br>
                    Great for quick notes, policies, or copied content.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Document title
            text_title = st.text_input(
                "Document Title",
                placeholder="e.g., Company Leave Policy",
                key="text_title",
                help="Give your text content a descriptive title"
            )
            
            # Text area for content
            text_content = st.text_area(
                "Paste or Type Content",
                placeholder="Paste your text content here...\n\nFor example:\n- Company policies\n- Meeting notes\n- Product documentation\n- FAQs\n- Any text-based information",
                height=200,
                key="text_content"
            )
            
            # Character count
            if text_content:
                char_count = len(text_content)
                word_count = len(text_content.split())
                st.caption(f"📊 {char_count} characters | {word_count} words")
            
            # Store text to RAG button
            if st.button("🚀 Store Text to RAG", use_container_width=True, key="store_text"):
                if not text_title:
                    st.error("Please provide a title for your text content")
                elif not text_content or len(text_content.strip()) < 10:
                    st.error("Please provide some text content (at least 10 characters)")
                else:
                    with st.spinner("Processing text..."):
                        progress_bar = st.progress(0)
                        progress_bar.progress(30)
                        
                        success, message, filename = process_text_to_rag(text_content, text_title)
                        progress_bar.progress(100)
                        
                        if success:
                            st.success(f"✅ Text stored as '{filename}'! {message}")
                            st.balloons()
                            st.cache_resource.clear()
                        else:
                            st.error(f"❌ Error: {message}")
        
        # Instructions
        st.markdown("---")
        st.markdown("""
        <div class="card">
            <div class="card-header">ℹ️ How it works</div>
            <ol style="color: #a0aec0; padding-left: 1.5rem;">
                <li>Upload files <strong>OR</strong> paste text directly</li>
                <li>Click the "Store to RAG" button</li>
                <li>Wait for processing (documents are chunked and embedded)</li>
                <li>Go to <strong>Chat tab</strong> to ask questions about your content!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # TAB 3: DOCUMENT COLLECTION
    # ============================================================
    with tab3:
        st.markdown('<div class="card-header">📚 Document Collection</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete All", use_container_width=True):
                st.session_state.confirm_delete_all = True
        
        # Confirm delete all
        if st.session_state.get('confirm_delete_all', False):
            st.warning(f"⚠️ {st.session_state.user_name}, are you sure you want to delete ALL documents?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Delete All", type="primary"):
                    try:
                        vector_store = VectorStore()
                        vector_store.connect_index()
                        vector_store.delete_all()
                        st.success("✅ All documents deleted")
                        st.session_state.confirm_delete_all = False
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            with col_no:
                if st.button("Cancel"):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
        
        st.markdown("---")
        
        # Get documents
        docs = get_document_list()
        
        if not docs:
            st.markdown(f"""
            <div class="card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📭</div>
                <div style="color: #fff; font-size: 1.2rem;">No documents in the collection, {st.session_state.user_name}</div>
                <div style="color: #718096; margin-top: 0.5rem;">Upload documents or paste text in the "Upload Documents" tab to get started</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Summary stats
            total_chunks = sum(d.get('chunks', 0) for d in docs)
            
            st.markdown(f"""
            <div class="card">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div class="stat-value">{len(docs)}</div>
                        <div class="stat-label">Total Documents</div>
                    </div>
                    <div>
                        <div class="stat-value">{total_chunks}</div>
                        <div class="stat-label">Total Chunks</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**📄 Documents in Knowledge Base:**")
            
            # Document list
            for doc in docs:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**📄 {doc['name']}**")
                
                with col2:
                    st.caption(f"{doc.get('chunks', 0)} chunks")
                
                with col3:
                    if st.button("🗑️", key=f"del_{doc['name']}", help=f"Delete {doc['name']}"):
                        if delete_document_from_rag(doc['name']):
                            st.success(f"Deleted {doc['name']}")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error("Failed to delete")
                
                st.markdown("---")
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #4a5568; padding: 1rem; font-size: 0.85rem;">
        🤖 <strong>JARVIS</strong> v{APP_VERSION} | Hello, {st.session_state.user_name}!<br>
        Powered by <strong>RAG</strong> + <strong>Qwen 2.5</strong> + <strong>Pinecone</strong>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    main()
