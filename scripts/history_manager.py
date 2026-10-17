import uuid
import datetime
import os
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from scripts.llm_config import chat_with_fallback

# --- CONFIGURATION ---
# Get the project root directory (one level up from scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
DB_PATH = os.path.join(parent_dir, "qdrant_db")
HISTORY_COLLECTION = "student_history"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ⚠️ GLOBAL VARIABLES (Initially None)
client = None
encoder = None
text_encoder = None

def init_client():
    """Initializes the Database connection safely."""
    global client, encoder, text_encoder
    if client is None:
        try:
            print(f"Connecting to Qdrant Database at: {os.path.abspath(DB_PATH)}")
            client = QdrantClient(path=DB_PATH)
            print("Qdrant client connected")
            
            print("Loading AI encoder model...")
            encoder = SentenceTransformer('clip-ViT-B-32')
            print("Encoder loaded")
            print("Loading text embedding model...")
            text_encoder = SentenceTransformer(TEXT_MODEL)
            print("Text encoder loaded")
            
            init_history_db()
        except Exception as e:
            print(f"ERROR: Initialization Error: {e}")
            print(f"   Database path: {os.path.abspath(DB_PATH)}")
            raise

def close_client():
    """Closes the connection safely."""
    global client, encoder, text_encoder
    try:
        if client:
            print("Closing Qdrant Connection...")
            # Suppress errors during shutdown - these are harmless
            try:
                client.close()
            except Exception as e:
                # During Python shutdown, some cleanup errors are expected
                if "sys.meta_path" in str(e) or "shutting down" in str(e).lower():
                    pass  # Ignore shutdown-related errors
                else:
                    print(f"Warning: Error closing client: {e}")
            finally:
                client = None
        
        # Clean up encoder if needed
        if encoder is not None:
            encoder = None
        if text_encoder is not None:
            text_encoder = None
            
    except Exception as e:
        # Ignore all errors during shutdown
        pass

def init_history_db():
    """Creates the 'student_history' collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(col.name == HISTORY_COLLECTION for col in collections)
    
    if not exists:
        print(f"Creating new collection: {HISTORY_COLLECTION}")
        client.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )

def check_collection_exists(collection_name):
    """Check if a collection exists in the database."""
    if not client:
        return False
    try:
        collections = client.get_collections().collections
        return any(col.name == collection_name for col in collections)
    except Exception as e:
        print(f"WARNING: Error checking collection: {e}")
        return False

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

def get_qdrant_history(limit=10):
    """Fetches the recent history."""
    if not client: return []
    
    try:
        history_data = []
        offset = None
        while True:
            result, offset = client.scroll(
                collection_name=HISTORY_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True
            )
            for point in result:
                history_data.append({
                    "id": point.id,
                    "topic": point.payload['topic'],
                    "summary": point.payload['summary'],
                    "date": point.payload['timestamp']
                })
            if offset is None:
                break
            
        history_data.sort(key=lambda x: x['date'], reverse=True)
        return history_data[:limit]
    except Exception as e:
        print(f"WARNING: Error fetching history: {e}")
        return []

def get_all_history():
    """Fetches all history records for analytics."""
    if not client: return []
    
    try:
        history_data = []
        offset = None
        while True:
            result, offset = client.scroll(
                collection_name=HISTORY_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True
            )
            for point in result:
                history_data.append({
                    "id": point.id,
                    "topic": point.payload['topic'],
                    "summary": point.payload.get('summary', ''),
                    "date": point.payload['timestamp']
                })
            if offset is None:
                break
            
        history_data.sort(key=lambda x: x['date'], reverse=True)
        return history_data
    except Exception as e:
        print(f"WARNING: Error fetching all history: {e}")
        return []

def analyze_learning_patterns(groq_api_key):
    """Analyzes student's learning patterns and identifies weak topics."""
    try:
        all_history = get_all_history()
        
        if not all_history:
            return {
                "total_topics": 0,
                "topic_frequency": {},
                "activity_by_date": {},
                "weak_topics": [],
                "strong_topics": [],
                "recommendations": []
            }
        
        from collections import Counter, defaultdict
        from datetime import datetime, timedelta
        
        # 1. Count topic frequency
        topic_counts = Counter()
        topic_last_studied = {}  # Track when each topic was last studied
        
        for item in all_history:
            topic = item['topic'].strip()
            # Clean topic (remove difficulty level suffix if present)
            clean_topic = topic.split('(')[0].strip()
            topic_counts[clean_topic] += 1
            
            # Track last studied date
            try:
                item_date = datetime.strptime(item['date'], "%Y-%m-%d %H:%M:%S")
                if clean_topic not in topic_last_studied or item_date > topic_last_studied[clean_topic]:
                    topic_last_studied[clean_topic] = item_date
            except:
                pass
        
        # 2. Activity by date
        activity_by_date = defaultdict(int)
        for item in all_history:
            try:
                date_str = item['date'].split(' ')[0]  # Get just the date part
                activity_by_date[date_str] += 1
            except:
                pass
        
        # 3. Identify weak topics (studied once or not in last 7 days)
        weak_topics = []
        strong_topics = []
        today = datetime.now()
        seven_days_ago = today - timedelta(days=7)
        
        for topic, count in topic_counts.items():
            last_studied = topic_last_studied.get(topic)
            days_since = None
            
            if last_studied:
                days_since = (today - last_studied).days
            
            # Weak topic if: studied only once OR not studied in last 7 days
            if count == 1 or (days_since and days_since > 7):
                weak_topics.append({
                    "topic": topic,
                    "count": count,
                    "last_studied": last_studied.strftime("%Y-%m-%d") if last_studied else "Never",
                    "days_ago": days_since if days_since else "N/A"
                })
            elif count >= 3 and (not days_since or days_since <= 7):
                strong_topics.append({
                    "topic": topic,
                    "count": count,
                    "last_studied": last_studied.strftime("%Y-%m-%d") if last_studied else "Never"
                })
        
        # Sort weak topics by priority (older and fewer studies first)
        weak_topics.sort(key=lambda x: (x['count'], x['days_ago'] if isinstance(x['days_ago'], int) else 999))
        
        # 4. Generate AI recommendations for weak topics
        recommendations = []
        if weak_topics and groq_api_key:
            try:
                weak_topic_names = [t['topic'] for t in weak_topics[:5]]  # Top 5 weak topics
                groq_client = Groq(api_key=groq_api_key)
                
                prompt = f"""Based on the student's learning history, they have weak understanding in these topics: {', '.join(weak_topic_names)}.

Provide 2-3 specific, actionable recommendations to help them improve. Focus on:
1. Which topics to review first
2. Learning strategies for these topics
3. Related concepts to study together

Keep it brief and motivational."""
                
                recommendations_text = chat_with_fallback(
                    groq_client,
                    messages=[{"role": "user", "content": prompt}],
                )
                recommendations = [r.strip() for r in recommendations_text.split('\n') if r.strip() and not r.strip().startswith('#')]
            except Exception as e:
                print(f"Warning: Could not generate AI recommendations: {e}")
                recommendations = ["Review your weak topics regularly", "Practice active recall"]
        
        # 5. Prepare data for charts
        sorted_activity = sorted(activity_by_date.items())
        activity_labels = [item[0] for item in sorted_activity]
        activity_values = [item[1] for item in sorted_activity]
        
        # Top topics for chart
        top_topics = topic_counts.most_common(10)
        topic_labels = [t[0] for t in top_topics]
        topic_values = [t[1] for t in top_topics]
        
        return {
            "total_topics": len(topic_counts),
            "total_sessions": len(all_history),
            "topic_frequency": dict(topic_counts),
            "top_topics": {
                "labels": topic_labels,
                "values": topic_values
            },
            "activity_by_date": {
                "labels": activity_labels,
                "values": activity_values
            },
            "weak_topics": weak_topics[:10],  # Top 10 weak topics
            "strong_topics": strong_topics[:10],  # Top 10 strong topics
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"ERROR in analyze_learning_patterns: {e}")
        return {
            "total_topics": 0,
            "topic_frequency": {},
            "activity_by_date": {},
            "weak_topics": [],
            "strong_topics": [],
            "recommendations": []
        }

def delete_history_record(point_id):
    """Deletes a specific history record by ID."""
    if not client: return False
    try:
        client.delete(
            collection_name=HISTORY_COLLECTION,
            points_selector=[point_id]
        )
        return True
    except Exception as e:
        print(f"WARNING: Error deleting history: {e}")
        return False

def search_images(query_vector, limit=2):
    """Searches the image collection using a CLIP query vector."""
    if not client: return []
    
    try:
        # Define Filter for type="image"
        img_filter = Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="image")
                )
            ]
        )
        
        search_results = client.query_points(
            collection_name="textbook_images",
            query=query_vector,
            query_filter=img_filter,
            limit=limit,
            with_payload=True
        ).points
        
        return [
            {**(res.payload or {}), "score": res.score}
            for res in search_results
        ]
    except Exception as e:
        print(f"WARNING: Image search failed: {e}")
        return []

def search_text(query_vector, limit):
    """Searches the text collection using a MiniLM query vector."""
    if not client: return []

    try:
        text_filter = Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="text")
                )
            ]
        )

        search_results = client.query_points(
            collection_name="textbook_text",
            query=query_vector,
            query_filter=text_filter,
            limit=limit,
            with_payload=True
        ).points
        return [
            {**(result.payload or {}), "score": result.score}
            for result in search_results
        ]
    except Exception as e:
        print(f"WARNING: Text search failed: {e}")
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

        return chat_with_fallback(
            client_groq,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return "Try studying 'Photosynthesis' next!"