import math
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "textbook_knowledge"
MODEL_NAME = "clip-ViT-B-32"
EXTRA_TEXT = " zebra rocket volcano banana" * 30


def cosine_similarity(first, second):
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cannot compute cosine similarity for a zero-length embedding.")
    return dot_product / (first_norm * second_norm)


def main():
    client = QdrantClient(path=DB_PATH)
    encoder = SentenceTransformer(MODEL_NAME)
    tokenizer = getattr(encoder.tokenizer, "tokenizer", encoder.tokenizer)
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

    longest_chunks = sorted(chunks, key=len, reverse=True)[:5]
    similarities = []
    for rank, chunk in enumerate(longest_chunks, start=1):
        token_count = len(
            tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"]
        )
        print(f"Chunk {rank} token count: {token_count}")
        original_vector = encoder.encode(chunk)
        extended_vector = encoder.encode(chunk + EXTRA_TEXT)
        similarity = cosine_similarity(original_vector, extended_vector)
        similarities.append(similarity)
        print(f"Chunk {rank} cosine similarity: {similarity:.6f}")

    if similarities and all(similarity > 0.9999 for similarity in similarities):
        print("CLIP is ignoring text after the first 77 tokens")
    else:
        print("Text beyond 77 tokens changes the embedding")


if __name__ == "__main__":
    main()
