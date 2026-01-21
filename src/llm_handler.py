"""
LLM interaction using Ollama with Qwen 2.5
Qwen 2.5 is optimized for RAG, document understanding, and complex queries
"""
import ollama
from typing import List, Dict

from src.config import LLM_TEMPERATURE, LLM_MAX_TOKENS


class LLMHandler:
    """Handle LLM interactions using Ollama with Qwen 2.5"""
    
    def __init__(self, model: str = "qwen2.5:7b"):
        """
        Initialize LLM Handler with Qwen 2.5
        
        Qwen 2.5 is chosen for:
        - Superior RAG performance
        - Better document understanding
        - Faster inference
        - More accurate context following
        - Better handling of complex queries
        """
        self.model = model
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        print(f"✅ LLM Handler initialized (model: {self.model})")
        print(f"   🎯 Optimized for: RAG, Document QA, Complex Retrieval")
    
    def generate_response(
        self,
        context: List[str],
        question: str
    ) -> str:
        """Generate response using context and question"""
        
        # Build optimized prompt for Qwen
        prompt = self._build_qwen_prompt(context, question)
        
        try:
            # Call Ollama with Qwen-optimized parameters
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'top_k': 40,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            answer = response['response'].strip()
            
            # Clean up any artifacts
            answer = self._clean_response(answer)
            
            return answer
        
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"❌ {error_msg}")
            return f"I apologize, but I encountered an error while processing your question. Please try again."
    
    def _build_qwen_prompt(self, context: List[str], question: str) -> str:
        """
        Build optimized prompt for Qwen 2.5
        Qwen responds well to structured, clear instructions
        """
        
        # Format context with clear separation
        context_text = "\n\n---\n\n".join([
            f"[Document {i+1}]\n{ctx}"
            for i, ctx in enumerate(context)
        ])
        
        # Optimized prompt structure for Qwen - STRICT document-only answers
        prompt = f"""You are an intelligent enterprise AI assistant. Your task is to answer questions accurately based ONLY on the provided company documents.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the documents below - DO NOT use any external knowledge
2. If the information is NOT found in the documents, you MUST respond: "I couldn't find information about this in the uploaded documents. Please upload the relevant documents containing this information."
3. NEVER make up or hallucinate information
4. Be specific and cite relevant details from the documents
5. Keep answers concise but complete
6. Use professional, clear language
7. If partially relevant information exists, provide what you can and indicate what's missing

COMPANY DOCUMENTS:
{context_text}

QUESTION: {question}

ANSWER (based ONLY on the documents above):"""
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up response artifacts"""
        # Remove common artifacts
        response = response.replace("ANSWER:", "").strip()
        response = response.replace("Answer:", "").strip()
        response = response.replace("ANSWER (based ONLY on the documents above):", "").strip()
        
        # Remove excessive newlines
        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")
        
        return response
    
    def check_model_availability(self) -> bool:
        """Check if Qwen model is available in Ollama"""
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            
            # Check for any Qwen variant
            qwen_models = [m for m in available_models if 'qwen' in m.lower()]
            
            if qwen_models:
                print(f"✅ Qwen models available: {qwen_models}")
                return True
            else:
                print(f"⚠️  Qwen model not found")
                print(f"   Available models: {available_models}")
                print(f"\n   📥 To install Qwen 2.5 (RECOMMENDED):")
                print(f"   ollama pull qwen2.5:7b")
                print(f"\n   Alternative options:")
                print(f"   - ollama pull qwen2.5:3b (smaller, faster)")
                print(f"   - ollama pull qwen2.5:14b (larger, more capable)")
                return False
        
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
            print(f"\n   📋 Setup Steps:")
            print(f"   1. Install Ollama from: https://ollama.ai")
            print(f"   2. Start Ollama: ollama serve")
            print(f"   3. Pull Qwen model: ollama pull qwen2.5:7b")
            return False


# Standalone testing
if __name__ == "__main__":
    llm = LLMHandler()
    
    # Check model availability
    if not llm.check_model_availability():
        print("\n❌ Qwen model not available. Please install it first.")
        exit(1)
    
    # Test generation
    test_context = [
        "The company offers 15 days of paid vacation for full-time employees. Part-time employees receive pro-rated vacation days.",
        "Vacation requests must be submitted at least 2 weeks in advance through the HR portal.",
        "Unused vacation days can be carried over to the next year, up to a maximum of 5 days."
    ]
    
    test_questions = [
        "How many vacation days do employees get?",
        "How far in advance should I submit vacation requests?",
        "Can I carry over unused vacation days?"
    ]
    
    print(f"\n{'='*70}")
    print("🧪 TESTING QWEN 2.5 LLM")
    print('='*70)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print(f"⏳ Generating response...")
        
        response = llm.generate_response(test_context, question)
        
        print(f"\n✨ Response:\n{response}")
        print('='*70)
