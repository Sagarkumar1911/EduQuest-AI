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
from scripts.web_image_engine import get_google_images
from groq import Groq

# CONFIGURATION
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY or not YOUTUBE_API_KEY:
    print("ERROR: Keys missing! Check .env")
    sys.exit(1)

# --- 🚀 LIFESPAN MANAGER (THE FIX) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Connect to DB
    print("Starting up... Connecting to Database.")
    hm.init_client()
    yield
    # SHUTDOWN: Close DB
    print("Shutting down... Closing Database.")
    try:
        hm.close_client()
    except Exception as e:
        # Suppress errors during shutdown - these are harmless
        # The "sys.meta_path is None" error during Python shutdown is expected
        if "sys.meta_path" not in str(e) and "shutting down" not in str(e).lower():
            print(f"Warning during shutdown: {e}")

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
    print(f"\nRequest: '{user_query}'")

    # Check if client and encoder are initialized
    if not hm.client or not hm.encoder:
        error_msg = "ERROR: Database not initialized. Please restart the server."
        print(error_msg)
        return {"answer": error_msg, "images": [], "video": None, "error": error_msg}

    # Check if the textbook_knowledge collection exists
    if not hm.check_collection_exists("textbook_knowledge"):
        error_msg = "ERROR: Collection 'textbook_knowledge' not found. Please run the ingest script first to populate the database."
        print(error_msg)
        return {"answer": error_msg, "images": [], "video": None, "error": error_msg}

    # 1. RAG SEARCH
    context_text = ""
    images = []
    
    try:
        vector = hm.encoder.encode(user_query).tolist()
        print(f"Encoded query vector (length: {len(vector)})")
        
        # A. Text Context Search
        try:
            # We search for everything, then filter for text
            search_results = hm.client.search(
                collection_name="textbook_knowledge", query_vector=vector, limit=5, with_payload=True
            )
            for res in search_results:
                if res.payload.get('type') == 'text':
                    context_text += res.payload.get('content', '') + "\n\n"
        except Exception as e:
            print(f"WARNING: Text search failed: {e}")

        # B. Image Search (Explicit)
        try:
            img_payloads = hm.search_images(vector, limit=2)
            print(f"Found {len(img_payloads)} images via explicit search")
            
            for payload in img_payloads:
                img_path = payload.get('image_path')
                if img_path and img_path not in [img['path'] for img in images]:
                    images.append({"path": img_path, "description": payload.get('content', '')})
            
            # C. Web Image Fallback (If not enough local images)
            if len(images) < 2:
                print("Not enough local images, searching web...")
                web_images = get_google_images(user_query)
                for img in web_images:
                    if len(images) >= 3: break # Limit total images
                    images.append(img)
                    
        except Exception as e:
            print(f"WARNING: Image specific search failed: {e}")

    except Exception as e:
        error_msg = f"ERROR: Encoding or DB error: {str(e)}"
        print(error_msg)
        return {"answer": f"Failed to process query: {str(e)}", "images": [], "video": None, "error": error_msg}

    if not context_text:
        context_text = "No specific context found in the textbook."
        print("WARNING: No relevant text found in the book. AI will use general knowledge.")

    # 2. AI GENERATION
    system_prompt = f"""
    You are an expert, encouraging AI Tutor. 
    Explain "{user_query}" in {request.language}.
    
    GUIDELINES:
    1. Tone: Professional, clear, and educational.
    2. Structure: Use bold headings, bullet points, and numbered lists for clarity.
    3. Context-Aware: Use the provided textbook context where relevant.
    4. Language: Respond ENTIRELY in {request.language}.
    
    Context from Textbook:
    {context_text[:3000]}
    """
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            model="llama-3.3-70b-versatile",
        )
        answer = chat.choices[0].message.content
        print("AI response generated successfully")
    except Exception as e:
        error_msg = f"ERROR: AI generation failed: {str(e)}"
        print(error_msg)
        answer = f"I'm having trouble thinking right now. Error: {str(e)}"

    # 3. VIDEO & HISTORY
    try:
        video_data = get_relevant_video(user_query, YOUTUBE_API_KEY)
        hm.log_activity_to_qdrant(user_query, answer)
    except Exception as e:
        print(f"WARNING: Error in video/history: {e}")
        video_data = None

    return {"answer": answer, "images": images[:2], "video": video_data}

@app.get("/get-student-history")
async def get_history():
    return {"history": hm.get_qdrant_history()}

@app.get("/get-student-profile")
async def get_student_profile():
    """Get comprehensive learning analytics and profile data."""
    try:
        analytics = hm.analyze_learning_patterns(GROQ_API_KEY)
        return analytics
    except Exception as e:
        print(f"ERROR in get_student_profile: {e}")
        return {"error": str(e)}

@app.get("/get-recommendation")
async def get_recommendation():
    return {"suggestion": hm.suggest_next_topic(GROQ_API_KEY)}

@app.delete("/delete-history/{point_id}")
async def delete_history(point_id: str):
    success = hm.delete_history_record(point_id)
    return {"success": success}

@app.post("/get-quiz")
async def get_quiz(request: QueryRequest):
    """Generate a quiz based on the given topic."""
    prompt = f"""Create a JSON mini-quiz with 3 multiple choice questions for the topic: {request.query}

Return the response in the following JSON format:
{{
    "questions": [
        {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Correct Option"
        }}
    ]
}}

Make sure the JSON is valid and all questions relate to the topic."""
    
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], 
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        quiz_response = chat.choices[0].message.content
        return {"quiz_data": quiz_response}
    except Exception as e:
        error_msg = f"Quiz generation failed: {str(e)}"
        print(f"ERROR in get-quiz: {error_msg}")
        return {"error": error_msg, "quiz_data": None}

@app.post("/explain-image")
async def explain_image(file: UploadFile = File(...), user_query: str = Form("Analyze this image in detail.")):
    """Analyze an uploaded image using AI vision model."""
    try:
        # 1. Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return {"error": "Invalid file type. Please upload an image file (JPEG, PNG, GIF, WebP)."}
        
        # 2. Read and encode image
        img_bytes = await file.read()
        b64_img = encode_image(img_bytes)
        
        # 3. Define Valid Vision Models (Updated for 2026)
        # Llama 3.2 and Pixtral are deprecated. We use Llama 4 Multimodal models.
        models_to_try = [
            "meta-llama/llama-4-maverick-17b-128e-instruct",  # High capability multimodal
            "meta-llama/llama-4-scout-17b-16e-instruct",     # Fast multimodal
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                print(f"📷 Analyzing image with model: {model_name}...")
                
                chat = groq_client.chat.completions.create(
                    messages=[{
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }],
                    model=model_name,
                    max_tokens=800
                )
                
                # If successful, return immediately
                return {"explanation": chat.choices[0].message.content}
                
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                last_error = e
                continue # Immediately try the next model

        # If ALL models fail
        return {
            "error": "Image analysis failed. The vision models are currently offline or incompatible.",
            "details": str(last_error)
        }

    except Exception as e:
        return {"error": f"Server error processing image: {str(e)}"}