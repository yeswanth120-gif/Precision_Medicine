# File: backend/app_chatbot_smart.py

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import re
import time
from datetime import datetime
import hashlib

# --- Page Configuration ---
st.set_page_config(page_title="Smart Q&A Bot", page_icon="🤖")
st.title("🤖 Smart Q&A Bot")

# --- Smart Response System ---
class SmartChatbot:
    def __init__(self):
        self.greetings = {
            'hi': "Hi there! 👋 I'm here to help you with health-related questions. What would you like to know?",
            'hello': "Hello! 😊 I'm your health assistant. Feel free to ask me anything about health topics!",
            'hey': "Hey! 👋 How can I help you today?",
            'good morning': "Good morning! ☀️ Hope you're having a great day. What can I help you with?",
            'good afternoon': "Good afternoon! 🌤️ How can I assist you today?",
            'good evening': "Good evening! 🌙 What would you like to know?",
            'how are you': "I'm doing great, thanks for asking! 😊 I'm here and ready to help with your health questions.",
            'what is your name': "I'm your friendly health assistant chatbot! 🤖 You can ask me questions about various health topics.",
            'who are you': "I'm a health-focused chatbot designed to help answer your medical and health-related questions! 🏥"
        }
        
        self.thanks_responses = [
            "You're welcome! 😊 Happy to help!",
            "Glad I could help! 👍 Feel free to ask more questions.",
            "You're very welcome! 🙂 Anything else you'd like to know?",
            "My pleasure! 😊 I'm here whenever you need help."
        ]
        
        self.unknown_responses = [
            "I don't have information about that topic in my knowledge base yet. 🤔 Try asking about health-related topics!",
            "I'm not sure about that one! 😅 I specialize in health and medical questions. Got any health queries?",
            "That's not in my knowledge base currently. 📚 I'm great with health and medical questions though!",
            "I don't know about that topic yet! 🤷‍♀️ But I'm here to help with health-related questions!"
        ]
        
        # Minimum similarity threshold for valid responses
        self.similarity_threshold = 0.5
        
    def is_greeting(self, text):
        """Check if the input is a greeting"""
        text_lower = text.lower().strip()
        for greeting in self.greetings.keys():
            if greeting in text_lower:
                return greeting
        return None
    
    def is_thanks(self, text):
        """Check if the input is a thank you"""
        thanks_words = ['thank', 'thanks', 'appreciate', 'grateful']
        text_lower = text.lower()
        return any(word in text_lower for word in thanks_words)
    
    def is_goodbye(self, text):
        """Check if the input is a goodbye"""
        goodbye_words = ['bye', 'goodbye', 'see you', 'take care', 'farewell']
        text_lower = text.lower()
        return any(word in text_lower for word in goodbye_words)
    
    def get_response_type(self, text, retrieved_nodes):
        """Determine the type of response needed"""
        # Check for greetings first
        greeting = self.is_greeting(text)
        if greeting:
            return 'greeting', greeting
        
        # Check for thanks
        if self.is_thanks(text):
            return 'thanks', None
        
        # Check for goodbye
        if self.is_goodbye(text):
            return 'goodbye', None
        
        # Check if we have good retrieval results
        if retrieved_nodes and len(retrieved_nodes) > 0:
            best_score = retrieved_nodes[0].score
            if best_score >= self.similarity_threshold:
                return 'knowledge', None
            else:
                return 'unknown', None
        
        return 'unknown', None

# Initialize smart chatbot
smart_bot = SmartChatbot()

# --- Caching System ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_response(question_hash, question):
    """Cache responses to avoid recomputation"""
    return None  # This will be populated by actual responses

def generate_question_hash(question):
    """Generate a hash for the question for caching"""
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

# Custom parser for Q&A format (same as before)
def parse_qa_file(file_path):
    """Parse Q&A formatted files into separate documents"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    qa_pairs = re.split(r'\n\s*Q:', content)
    documents = []
    
    for pair in qa_pairs:
        if not pair.strip():
            continue
            
        if not pair.startswith('Q:'):
            pair = 'Q:' + pair
            
        lines = pair.strip().split('\n')
        if len(lines) >= 2:
            question = lines[0].replace('Q:', '').strip()
            answer_lines = []
            
            for line in lines[1:]:
                if line.startswith('A:'):
                    answer_lines.append(line.replace('A:', '').strip())
                elif line.strip() and not line.startswith('Q:'):
                    answer_lines.append(line.strip())
                else:
                    break
            
            if question and answer_lines:
                answer = ' '.join(answer_lines)
                doc_text = f"Question: {question}\nAnswer: {answer}"
                doc = Document(
                    text=doc_text,
                    metadata={
                        'question': question,
                        'answer': answer,
                        'file_name': os.path.basename(file_path)
                    }
                )
                documents.append(doc)
    
    return documents

@st.cache_resource
def setup_pipeline():
    """Setup the knowledge base pipeline with caching"""
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    try:
        context_dir = "./knowledgeBase"
        if not os.path.exists(context_dir):
            st.error(f"The '{context_dir}' directory is missing.")
            return None

        all_documents = []
        txt_files = [f for f in os.listdir(context_dir) if f.endswith('.txt')]
        
        if not txt_files:
            st.error(f"No .txt files found in '{context_dir}'.")
            return None
            
        for filename in txt_files:
            file_path = os.path.join(context_dir, filename)
            try:
                qa_docs = parse_qa_file(file_path)
                if qa_docs:
                    all_documents.extend(qa_docs)
                else:
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    all_documents.extend(docs)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
        
        if not all_documents:
            st.error("No documents were loaded successfully.")
            return None

        with st.spinner(f"Building smart knowledge base from {len(all_documents)} documents..."):
            index = VectorStoreIndex.from_documents(all_documents)
        
        retriever = index.as_retriever(
            similarity_top_k=3,  # Get more options for better filtering
            verbose=False
        )
        
        return retriever, len(all_documents)
        
    except Exception as e:
        st.error(f"Failed to build the index. Error: {e}")
        return None, 0

# Initialize pipeline
result = setup_pipeline()
if result:
    retriever, doc_count = result
else:
    retriever, doc_count = None, 0

# --- Session State for Conversation History ---
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

st.divider()

# --- UI for Smart Chatbot ---
if retriever:
    st.success(f"✅ Smart knowledge base ready with {doc_count} documents")
    
    # Show conversation history
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation")
        for i, (user_msg, bot_msg, timestamp) in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5
            with st.container():
                st.write(f"**You ({timestamp}):** {user_msg}")
                st.write(f"**Bot:** {bot_msg}")
                st.write("---")
    
    st.header("Ask me anything!")
    
    user_question = st.text_input("Type your message here:", key="user_input")

    if user_question:
        start_time = time.time()
        
        # Generate hash for caching
        question_hash = generate_question_hash(user_question)
        
        # Check cache first
        if question_hash in st.session_state.response_cache:
            response_text = st.session_state.response_cache[question_hash]
            st.info("🚀 Retrieved from cache (super fast!)")
        else:
            with st.spinner("🤖 Thinking..."):
                try:
                    # Get retrieval results
                    retrieved_nodes = retriever.retrieve(user_question)
                    
                    # Determine response type
                    response_type, greeting_key = smart_bot.get_response_type(user_question, retrieved_nodes)
                    
                    if response_type == 'greeting':
                        response_text = smart_bot.greetings[greeting_key]
                    elif response_type == 'thanks':
                        response_text = smart_bot.thanks_responses[hash(user_question) % len(smart_bot.thanks_responses)]
                    elif response_type == 'goodbye':
                        response_text = "Goodbye! 👋 Take care and feel free to come back anytime you have health questions!"
                    elif response_type == 'knowledge' and retrieved_nodes:
                        # Extract the best answer
                        best_node = retrieved_nodes[0]
                        text = best_node.get_text()
                        
                        if "Answer:" in text:
                            response_text = text.split("Answer:", 1)[1].strip()
                        else:
                            response_text = text
                        
                        # Add confidence indicator
                        confidence = best_node.score
                        if confidence > 0.8:
                            response_text += "\n\n✅ High confidence answer"
                        elif confidence > 0.6:
                            response_text += "\n\n⚠️ Medium confidence answer"
                    else:
                        # Unknown topic
                        response_text = smart_bot.unknown_responses[hash(user_question) % len(smart_bot.unknown_responses)]
                    
                    # Cache the response
                    st.session_state.response_cache[question_hash] = response_text
                    
                except Exception as e:
                    response_text = f"Sorry, I encountered an error: {str(e)} 😅"
        
        # Display response
        st.subheader("🤖 Response:")
        st.success(response_text)
        
        # Show processing time
        processing_time = time.time() - start_time
        st.caption(f"⚡ Response time: {processing_time:.2f} seconds")
        
        # Add to conversation history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.conversation_history.append((user_question, response_text, timestamp))
        
        # Keep only last 10 conversations to save memory
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        
        # Show additional info for knowledge-based responses
        if 'retrieved_nodes' in locals() and retrieved_nodes and len(retrieved_nodes) > 0:
            if retrieved_nodes[0].score >= smart_bot.similarity_threshold:
                with st.expander("📚 Source Information"):
                    best_node = retrieved_nodes[0]
                    st.write(f"**Source:** {best_node.metadata.get('file_name', 'N/A')}")
                    st.write(f"**Confidence Score:** {best_node.score:.3f}")
                    if 'question' in best_node.metadata:
                        st.write(f"**Matched Question:** {best_node.metadata['question']}")

else:
    st.error("❌ Smart knowledge base could not be initialized.")

# --- Sidebar with Cache Stats ---
with st.sidebar:
    st.header("🚀 Performance Stats")
    st.metric("Cached Responses", len(st.session_state.response_cache))
    st.metric("Conversation History", len(st.session_state.conversation_history))
    
    if st.button("🗑️ Clear Cache"):
        st.session_state.response_cache.clear()
        st.session_state.conversation_history.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    st.header("💡 Tips")
    st.info("Try saying:\n- Hi, Hello\n- Thank you\n- Health questions\n- Goodbye")
    
    st.header("⚡ Speed Optimizations")
    st.success("✅ Response caching\n✅ Conversation memory\n✅ Smart filtering\n✅ Lightweight embeddings")