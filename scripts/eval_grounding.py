import csv
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
QUESTIONS_PATH = os.path.join(BASE_DIR, "Data", "eval_questions.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "eval", "answers.csv")
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "textbook_knowledge"
MODEL_NAME = "clip-ViT-B-32"
GROQ_MODEL = "llama-3.3-70b-versatile"


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
    return questions


def retrieve_context(client, encoder, question):
    vector = encoder.encode(question).tolist()
    results = client.query_points(
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
    return "\n\n".join(
        result.payload.get("content", "")
        for result in results
        if result.payload and isinstance(result.payload.get("content"), str)
    )[:3000]


def generate_answer(groq_client, prompt_builder, question, context):
    system_prompt = prompt_builder(context, "English")
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        model=GROQ_MODEL,
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def main():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or project .env file.")
    os.environ.setdefault("YOUTUBE_API_KEY", "evaluation-only")
    from main_api import build_system_prompt

    questions = load_questions()
    encoder = SentenceTransformer(MODEL_NAME)
    groq_client = Groq(api_key=api_key)
    qdrant_client = QdrantClient(path=DB_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "question",
                    "mode",
                    "answer",
                    "context",
                    "unsupported_claims",
                    "notes",
                ],
            )
            writer.writeheader()
            for item in questions:
                question = item["question"]
                context = retrieve_context(qdrant_client, encoder, question)
                for mode, prompt_context in (
                    ("retrieved_context", context),
                    ("no_context", ""),
                ):
                    answer = generate_answer(groq_client, build_system_prompt, question, prompt_context)
                    writer.writerow(
                        {
                            "question": question,
                            "mode": mode,
                            "answer": answer,
                            "context": context,
                            "unsupported_claims": "",
                            "notes": "",
                        }
                    )
                print(f"Generated both answers for: {question}")
    finally:
        qdrant_client.close()
    print(f"Answers written to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Grounding evaluation setup failed: {error}", file=sys.stderr)
        sys.exit(1)
