import os
import sys
from dotenv import load_dotenv # Import this to read .env
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from scripts.llm_config import chat_with_fallback

# --- 1. LOAD SECRETS ---
# This forces the script to look for .env in the main folder (one level up)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)

# Get Key from .env
api_key = os.getenv("GROQ_API_KEY")

# Safety Check
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found!")
    print(f"   Searching for .env at: {dotenv_path}")
    sys.exit(1)

# --- 2. CONFIGURATION ---
DB_PATH = os.path.join(parent_dir, "qdrant_db")
COLLECTION_NAME = "textbook_knowledge"

# --- 3. INITIALIZE ---
print("🚀 Loading AI Brain...")
try:
    # Initialize Clients
    client = QdrantClient(path=DB_PATH)
    encoder = SentenceTransformer('clip-ViT-B-32')
    groq_client = Groq(api_key=api_key)
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    print("TIP: Check if your 'qdrant_db' folder exists in the project root.")
    sys.exit(1)

def get_answer(query):
    print(f"\n🔎 Analyzing: '{query}'...")

    # A. RETRIEVE (Search Qdrant for Book Context)
    try:
        vector = encoder.encode(query).tolist()
        
        # Search Qdrant
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=3,
            with_payload=True
        )
        results = search_result.points
            
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return

    # B. PREPARE CONTEXT
    context_text = ""
    sources = []
    
    for res in results:
        if res.payload.get('type') == 'text':
            context_text += res.payload['content'] + "\n\n"
            page_num = res.payload.get('page', 'Unknown')
            sources.append(f"Page {page_num}")
    
    if not context_text:
        context_text = "No specific context found in the textbook."
        print("⚠️ Note: No relevant text found in the book. AI will use general knowledge.")

    # C. GENERATE (Ask Groq AI)
    print("🤖 Thinking...")
    
    system_prompt = f"""
    You are an expert Biology Tutor. 
    Use the following TEXTBOOK CONTEXT to answer the student's question.
    
    RULES:
    1. Explain the concept clearly using the Context.
    2. If the Context is hard to understand, simplify it.
    3. If the answer is NOT in the context, use your own knowledge but mention that.
    
    TEXTBOOK CONTEXT:
    {context_text}
    """

    try:
        ai_answer = chat_with_fallback(
            groq_client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=1024,
        )

        # D. SHOW RESULT
        print("\n" + "="*40)
        print(f"🎓 AI ANSWER:\n{ai_answer}")
        print("="*40)
        unique_sources = sorted(list(set(sources)))
        print(f"📚 Sources: {', '.join(unique_sources)}")
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")

if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk a Question (or 'q'): ")
            if q.lower() == 'q': break
            get_answer(q)
        except KeyboardInterrupt:
            print("\nExiting...")
            break