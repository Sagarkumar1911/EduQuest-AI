import sys
import os
import base64
from typing import List
from contextlib import asynccontextmanager # <--- NEW IMPORT
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# IMPORTS
from scripts.youtube_engine import get_relevant_video
import scripts.history_manager as hm # Import the whole module
from groq import Groq

# CONFIGURATION
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY or not YOUTUBE_API_KEY:
    print("❌ ERROR: Keys missing! Check .env")
    sys.exit(1)

# --- 🚀 LIFESPAN MANAGER (THE FIX) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Connect to DB
    print("🚀 Starting up... Connecting to Database.")
    hm.init_client()
    yield
    # SHUTDOWN: Close DB
    print("🛑 Shutting down... Closing Database.")
    hm.close_client()

# INITIALIZE APP WITH LIFESPAN
app = FastAPI(lifespan=lifespan)
groq_client = Groq(api_key=GROQ_API_KEY)

# ALLOW FRONTEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DATA MODELS
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message] = []
    language: str = "English"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# --- ENDPOINTS ---

@app.post("/get-lesson")
async def get_lesson(request: QueryRequest):
    user_query = request.query
    print(f"\n📩 Request: '{user_query}'")

    # 1. RAG SEARCH
    # Use the client inside history_manager
    try:
        vector = hm.encoder.encode(user_query).tolist()
        search_results = hm.client.query_points(
            collection_name="textbook_knowledge", query=vector, limit=5, with_payload=True
        ).points
    except:
        search_results = hm.client.search(
            collection_name="textbook_knowledge", query_vector=vector, limit=5, with_payload=True
        )

    context_text = ""
    images = []
    for res in search_results:
        payload = res.payload
        if payload['type'] == 'text':
            context_text += payload['content'] + "\n\n"
        elif payload['type'] == 'image':
            if payload['image_path'] not in [img['path'] for img in images]:
                images.append({"path": payload['image_path'], "description": payload['content']})

    # 2. AI GENERATION
    system_prompt = f"""
    You are a Tutor. Explain "{user_query}" in {request.language}.
    Context: {context_text[:3000]}
    """
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            model="llama-3.3-70b-versatile",
        )
        answer = chat.choices[0].message.content
    except:
        answer = "I'm having trouble thinking."

    # 3. VIDEO & HISTORY
    video_data = get_relevant_video(user_query, YOUTUBE_API_KEY)
    hm.log_activity_to_qdrant(user_query, answer)

    return {"answer": answer, "images": images[:2], "video": video_data}

@app.get("/get-student-history")
async def get_history():
    return {"history": hm.get_qdrant_history()}

@app.get("/get-recommendation")
async def get_recommendation():
    return {"suggestion": hm.suggest_next_topic(GROQ_API_KEY)}

@app.post("/get-quiz")
async def get_quiz(request: QueryRequest):
    prompt = f"Create a JSON mini-quiz (3 MCQs) for: {request.query}"
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile"
        )
        return {"quiz_data": chat.choices[0].message.content}
    except:
        return {"error": "Quiz failed."}

@app.post("/explain-image")
async def explain_image(file: UploadFile = File(...), user_query: str = Form("Explain this.")):
    img_bytes = await file.read()
    b64_img = encode_image(img_bytes)
    try:
        chat = groq_client.chat.completions.create(
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }],
            model="llama-3.2-11b-vision-preview",
        )
        return {"explanation": chat.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}