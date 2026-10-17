import argparse
import json
import os
import sys

import fitz
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(BASE_DIR, "Data", "eval_questions.json")
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
PDF_FOLDER = os.path.join(BASE_DIR, "Data", "pdf")
COLLECTION_NAME = "textbook_knowledge"
INGEST_MODEL = "clip-ViT-B-32"
TEMP_COLLECTION = "evaluation_text"


def load_questions():
    with open(QUESTIONS_PATH, encoding="utf-8") as file:
        data = json.load(file)
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError("eval_questions.json must contain a list or an object with a 'questions' list.")
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Question {index} needs a non-empty 'question' field.")
        source = item.get("expected_source", item.get("source"))
        page = item.get("expected_page", item.get("page"))
        if source is None or page is None:
            raise ValueError(
                f"Question {index} needs 'expected_source'/'expected_page' "
                "(or 'source'/'page') expected-answer fields."
            )
    return questions


def normalize_source(source):
    return os.path.basename(str(source)).casefold()


def expected_match(point, expected_source, expected_page):
    payload = point.payload or {}
    try:
        page_matches = abs(int(payload.get("page")) - int(expected_page)) <= 1
    except (TypeError, ValueError):
        page_matches = False
    source_matches = normalize_source(payload.get("source", "")) == normalize_source(expected_source)
    return source_matches and page_matches


def pdf_text_chunks():
    pdf_paths = [
        os.path.join(PDF_FOLDER, filename)
        for filename in os.listdir(PDF_FOLDER)
        if filename.lower().endswith(".pdf")
    ]
    points = []
    point_id = 0
    for path in sorted(pdf_paths):
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                chunks = [
                    chunk.strip()
                    for chunk in page.get_text().split("\n\n")
                    if len(chunk.strip()) > 50
                ]
                for chunk in chunks:
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=[],
                            payload={
                                "type": "text",
                                "content": chunk,
                                "page": page_number,
                                "source": os.path.basename(path),
                            },
                        )
                    )
                    point_id += 1
    return points


def make_temporary_collection(encoder):
    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=TEMP_COLLECTION,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    pending = pdf_text_chunks()
    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        texts = [point.payload["content"] for point in batch]
        vectors = encoder.encode(texts).tolist()
        for point, vector in zip(batch, vectors):
            point.vector = vector
        client.upsert(collection_name=TEMP_COLLECTION, points=batch)
    return client


def search_text(client, collection_name, encoder, question):
    vector = encoder.encode(question).tolist()
    return client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="text"),
                )
            ]
        ),
        limit=5,
        with_payload=True,
    ).points


def main():
    parser = argparse.ArgumentParser(description="Evaluate textbook text retrieval.")
    parser.add_argument("--model", default=INGEST_MODEL, help="Sentence-transformers model name.")
    args = parser.parse_args()

    questions = load_questions()
    encoder = SentenceTransformer(args.model)
    use_temporary = args.model != INGEST_MODEL
    client = make_temporary_collection(encoder) if use_temporary else QdrantClient(path=DB_PATH)
    collection_name = TEMP_COLLECTION if use_temporary else COLLECTION_NAME

    hits = {1: 0, 3: 0, 5: 0}
    misses = []
    try:
        for item in questions:
            results = search_text(client, collection_name, encoder, item["question"])
            ranks = [
                rank
                for rank, result in enumerate(results, start=1)
                if expected_match(
                    result,
                    item.get("expected_source", item.get("source")),
                    item.get("expected_page", item.get("page")),
                )
            ]
            for cutoff in hits:
                if any(rank <= cutoff for rank in ranks):
                    hits[cutoff] += 1
            print(
                f"{item['question']}: "
                f"hit@1={any(rank <= 1 for rank in ranks)}, "
                f"hit@3={any(rank <= 3 for rank in ranks)}, "
                f"hit@5={any(rank <= 5 for rank in ranks)}"
            )
            if not ranks:
                returned = [
                    {
                        "source": (result.payload or {}).get("source"),
                        "page": (result.payload or {}).get("page"),
                    }
                    for result in results
                ]
                misses.append((item, returned))
    finally:
        client.close()

    total = len(questions)
    if not total:
        print("No evaluation questions found.")
        return

    for cutoff in hits:
        print(f"hit@{cutoff}: {hits[cutoff] / total:.2%} ({hits[cutoff]}/{total})")
    if misses:
        print("\nMissed questions and returned pages:")
        for item, returned in misses:
            expected_source = item.get("expected_source", item.get("source"))
            expected_page = item.get("expected_page", item.get("page"))
            print(f"- {item['question']} (expected {expected_source} page {expected_page})")
            print(f"  Returned: {returned}")
    else:
        print("\nNo missed questions.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Evaluation setup failed: {error}", file=sys.stderr)
        sys.exit(1)
