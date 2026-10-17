import os
import statistics

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "textbook_text"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    client = QdrantClient(path=DB_PATH)
    encoder = SentenceTransformer(MODEL_NAME)
    tokenizer = encoder.tokenizer
    text_filter = Filter(
        must=[
            FieldCondition(
                key="type",
                match=MatchValue(value="text"),
            )
        ]
    )

    chunks = []
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=text_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                point.payload.get("content", "")
                for point in points
                if point.payload and isinstance(point.payload.get("content"), str)
            )
            if offset is None:
                break
    finally:
        client.close()

    if not chunks:
        print("Number of text chunks: 0")
        print("No token counts to report.")
        return

    token_counts = [
        len(tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"])
        for chunk in chunks
    ]
    over_limit = sum(count > 200 for count in token_counts)

    print(f"Number of text chunks: {len(chunks)}")
    print(f"MiniLM token count min: {min(token_counts)}")
    print(f"MiniLM token count median: {statistics.median(token_counts)}")
    print(f"MiniLM token count max: {max(token_counts)}")
    print(f"Chunks exceeding 200 tokens: {over_limit}")


if __name__ == "__main__":
    main()
