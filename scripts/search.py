from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DB_PATH = "../qdrant_db"
COLLECTION_NAME = "textbook_knowledge"

# --- INITIALIZE ---
print("🚀 Loading Search Engine...")
try:
    client = QdrantClient(path=DB_PATH)
    encoder = SentenceTransformer('clip-ViT-B-32')
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    exit()

def search(query):
    print(f"\n🔎 Searching for: '{query}'...")
    
    vector = encoder.encode(query).tolist()
    
    # 1. Fetch MORE results (Limit 10) so we can filter
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=10,  # <--- Increased to 10
            with_payload=True
        )
        results = search_result.points
    except AttributeError:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=10,
            with_payload=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. SEPARATE Text and Images (The "Bucket Strategy")
    text_results = []
    image_results = []
    
    for res in results:
        if res.payload['type'] == 'text':
            text_results.append(res)
        elif res.payload['type'] == 'image':
            image_results.append(res)

    # 3. DISPLAY Top 3 Text + Top 2 Images
    print("\n--- 📄 RELEVANT TEXT (Top 3) ---")
    if not text_results:
        print("   No text found.")
    for res in text_results[:3]: # Take top 3 only
        print(f"   Score: {res.score:.2f} | Page {res.payload['page']}")
        print(f"   \"{res.payload['content'][:100]}...\"\n")

    print("--- 🖼️  RELEVANT DIAGRAMS (Top 2) ---")
    if not image_results:
        print("   No images found (Try a specific query like 'plant cell').")
    for res in image_results[:2]: # Take top 2 only
        print(f"   Score: {res.score:.2f} | File: {res.payload['source']}")
        print(f"   Description: {res.payload['content'][:100]}...\n")

if __name__ == "__main__":
    while True:
        try:
            user_input = input("Enter question (e.g. 'plant cell'): ")
            if user_input.lower() == 'q': break
            search(user_input)
        except KeyboardInterrupt: break