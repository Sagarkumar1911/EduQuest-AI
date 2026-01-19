import uuid
import datetime
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DB_PATH = "qdrant_db"
HISTORY_COLLECTION = "student_history"

# ⚠️ GLOBAL VARIABLES (Initially None)
client = None
encoder = None

def init_client():
    """Initializes the Database connection safely."""
    global client, encoder
    if client is None:
        print("🔌 Connecting to Qdrant Database...")
        client = QdrantClient(path=DB_PATH)
        encoder = SentenceTransformer('clip-ViT-B-32')
        init_history_db()

def close_client():
    """Closes the connection safely."""
    global client
    if client:
        print("🔌 Closing Qdrant Connection...")
        client.close()
        client = None

def init_history_db():
    """Creates the 'student_history' collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(col.name == HISTORY_COLLECTION for col in collections)
    
    if not exists:
        print(f"📂 Creating new collection: {HISTORY_COLLECTION}")
        client.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )

def log_activity_to_qdrant(query, answer):
    """Saves the student's interaction."""
    if not client: return # Safety check
    
    vector = encoder.encode(query).tolist()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    point_id = str(uuid.uuid4())
    
    payload = {
        "type": "history_record",
        "topic": query,
        "summary": answer[:150] + "...",
        "full_answer": answer,
        "timestamp": timestamp
    }

    client.upsert(
        collection_name=HISTORY_COLLECTION,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)]
    )

def get_qdrant_history():
    """Fetches the recent history."""
    if not client: return []
    
    try:
        result, _ = client.scroll(
            collection_name=HISTORY_COLLECTION,
            limit=10,
            with_payload=True
        )
        
        history_data = []
        for point in result:
            history_data.append({
                "topic": point.payload['topic'],
                "summary": point.payload['summary'],
                "date": point.payload['timestamp']
            })
            
        history_data.sort(key=lambda x: x['date'], reverse=True)
        return history_data
    except Exception as e:
        print(f"⚠️ Error fetching history: {e}")
        return []

def suggest_next_topic(groq_api_key):
    """Analyzes history to suggest next topic."""
    try:
        recent_history = get_qdrant_history()[:3] 
        if not recent_history:
            return "Start by asking about 'Cell Structure' or 'DNA'!"

        topics_str = ", ".join([item['topic'] for item in recent_history])
        client_groq = Groq(api_key=groq_api_key)
        
        prompt = f"""
        The student has recently studied: {topics_str}.
        Suggest ONE next logical topic in Biology.
        Return ONLY the topic name and a 1-sentence reason.
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception:
        return "Try studying 'Photosynthesis' next!"