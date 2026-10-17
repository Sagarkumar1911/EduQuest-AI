import json
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
QUESTIONS_PATH = os.path.join(BASE_DIR, "Data", "probe_questions.json")
COLLECTION_NAME = "textbook_text"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_probe_questions():
    with open(QUESTIONS_PATH, encoding="utf-8") as probe_file:
        data = json.load(probe_file)

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    elif isinstance(data, dict) and "on_topic" in data and "off_topic" in data:
        data = [
            {"question": question, "on_topic": True}
            for question in data["on_topic"]
        ] + [
            {"question": question, "on_topic": False}
            for question in data["off_topic"]
        ]

    if not isinstance(data, list):
        raise ValueError(
            "probe_questions.json must be a list, an object with a 'questions' list, "
            "or an object with 'on_topic' and 'off_topic' lists."
        )

    questions = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Probe question {index} must be an object.")

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Probe question {index} needs a non-empty 'question'.")

        label = item.get("on_topic", item.get("label", item.get("category")))
        if isinstance(label, str):
            if label.casefold() in {"on_topic", "on-topic", "on topic"}:
                label = True
            elif label.casefold() in {"off_topic", "off-topic", "off topic"}:
                label = False
        if not isinstance(label, bool):
            raise ValueError(
                f"Probe question {index} needs a boolean 'on_topic' field "
                "or an 'on_topic'/'off_topic' label."
            )
        questions.append({"question": question.strip(), "on_topic": label})
    return questions


def search_text(client, encoder, question):
    vector = encoder.encode(question).tolist()
    return client.query_points(
        collection_name=COLLECTION_NAME,
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
    questions = load_probe_questions()
    encoder = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=DB_PATH)
    on_topic_top_scores = []
    off_topic_top_scores = []
    on_topic_ranks_2_to_5 = []

    try:
        for item in questions:
            results = search_text(client, encoder, item["question"])
            print(f"\nQuestion ({'on-topic' if item['on_topic'] else 'off-topic'}): {item['question']}")
            for rank, result in enumerate(results, start=1):
                payload = result.payload or {}
                print(
                    f"  Rank {rank}: score={result.score}, "
                    f"printed_page={payload.get('printed_page')}"
                )
            top_payload = results[0].payload or {} if results else {}
            print(f"  Top chunk: {top_payload.get('content', '')[:60]}")

            scores = [result.score for result in results if result.score is not None]
            if not scores:
                continue
            if item["on_topic"]:
                on_topic_top_scores.append(scores[0])
                on_topic_ranks_2_to_5.extend(scores[1:5])
            else:
                off_topic_top_scores.append(scores[0])
    finally:
        client.close()

    lowest_on_topic = min(on_topic_top_scores) if on_topic_top_scores else None
    highest_off_topic = max(off_topic_top_scores) if off_topic_top_scores else None
    average_on_topic_ranks_2_to_5 = (
        sum(on_topic_ranks_2_to_5) / len(on_topic_ranks_2_to_5)
        if on_topic_ranks_2_to_5 else None
    )

    print("\nSummary:")
    print(f"Lowest on-topic top-1 score: {lowest_on_topic}")
    print(f"Highest off-topic top-1 score: {highest_off_topic}")
    print(f"Average on-topic ranks 2 to 5 score: {average_on_topic_ranks_2_to_5}")

    if (
        lowest_on_topic is not None
        and highest_off_topic is not None
        and lowest_on_topic > highest_off_topic
    ):
        midpoint = (lowest_on_topic + highest_off_topic) / 2
        print("Separable: a threshold between these two values would separate them")
        print(f"Midpoint threshold: {midpoint}")
    else:
        print(
            "Not separable: on-topic and off-topic scores overlap, so a single "
            "threshold would reject real questions or accept unrelated ones"
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Score probe failed: {error}", file=sys.stderr)
        sys.exit(1)
