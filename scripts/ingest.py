import os
import json
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from PIL import Image



# --- CONFIGURATION ---
# Get the project root directory (one level up from scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Use "Data" (capital D) as that's the actual folder name
DATA_PATH = os.path.join(parent_dir, "Data")
PDF_FOLDER = os.path.join(DATA_PATH, "pdf")
IMAGE_FOLDER = os.path.join(DATA_PATH, "images")
METADATA_FILE = os.path.join(DATA_PATH, "image_metadata.json")
DB_PATH = os.path.join(parent_dir, "qdrant_db")  # This folder will be created automatically

KNOWLEDGE_COLLECTION = "textbook_knowledge"
MEMORY_COLLECTION = "student_memory"

# --- 1. INITIALIZE RESOURCES ---
print("Initializing Qdrant and AI Model...")

# Initialize Qdrant (Local Persistence)
client = QdrantClient(path=DB_PATH)

# Load CLIP Model (Multimodal)
# This downloads about ~600MB the first time you run it
encoder = SentenceTransformer('clip-ViT-B-32')

def create_collections():
    """Creates the empty database buckets."""
    # 1. Knowledge Base
    print(f"   Creating collection: {KNOWLEDGE_COLLECTION}")
    client.recreate_collection(
        collection_name=KNOWLEDGE_COLLECTION,
        vectors_config=models.VectorParams(
            size=512,  # Matches CLIP model size
            distance=models.Distance.COSINE
        )
    )
    # 2. Memory (Chat History)
    print(f"   Creating collection: {MEMORY_COLLECTION}")
    client.recreate_collection(
        collection_name=MEMORY_COLLECTION,
        vectors_config=models.VectorParams(
            size=512,
            distance=models.Distance.COSINE
        )
    )

# --- 2. PROCESS PDFS ---
def process_pdfs():
    print("Processing PDFs...")
    points = []
    chunk_id = 0

    if not os.path.exists(PDF_FOLDER):
        print(f"ERROR: PDF Folder not found at {PDF_FOLDER}")
        return [], 0

    # Find all PDFs
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("WARNING: No PDF files found in data/pdfs/")

    for pdf_file in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"   Processing: {pdf_file}")
        
        try:
            doc = fitz.open(path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Split by paragraphs (double newline)
                chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
                
                for chunk in chunks:
                    vector = encoder.encode(chunk).tolist()
                    points.append(models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "type": "text",
                            "content": chunk,
                            "page": page_num + 1,
                            "source": pdf_file
                        }
                    ))
                    chunk_id += 1
        except Exception as e:
            print(f"ERROR: Error reading {pdf_file}: {e}")

    print(f"   Extracted {len(points)} text chunks.")
    return points, chunk_id

# --- 3. PROCESS IMAGES ---
def process_images(start_id):
    print("Processing Images...")
    points = []
    current_id = start_id

    if not os.path.exists(METADATA_FILE):
        print(f"WARNING: Metadata file missing: {METADATA_FILE}")
        return []

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    for item in metadata:
        filename = item["filename"]
        description = item["description"]
        img_path = os.path.join(IMAGE_FOLDER, filename)

        if os.path.exists(img_path):
            try:
                # Open Image
                image = Image.open(img_path)
                # Convert Image to Vector (The Magic Part)
                vector = encoder.encode(image).tolist()
                
                points.append(models.PointStruct(
                    id=current_id,
                    vector=vector,
                    payload={
                        "type": "image",
                        "content": description,
                        "image_path": img_path, # Path for Streamlit to display
                        "source": filename,
                        "page": 0 
                    }
                ))
                current_id += 1
                print(f"   Ingested: {filename}")
            except Exception as e:
                print(f"ERROR: Error with {filename}: {e}")
        else:
            print(f"WARNING: Image not found: {filename}")

    return points

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    create_collections()
    
    # Run Processing
    text_data, last_id = process_pdfs()
    image_data = process_images(start_id=last_id)
    
    all_data = text_data + image_data

    # Upload to Qdrant
    if all_data:
        print(f"Uploading {len(all_data)} items to database...")
        client.upsert(
            collection_name=KNOWLEDGE_COLLECTION,
            points=all_data
        )
        print("SUCCESS! Database is ready.")
        print(f"   - Database location: {os.path.abspath(DB_PATH)}")
    else:
        print("WARNING: No data was found. Check your folders!")