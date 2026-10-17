import json
import math
import os
import re

import fitz
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
DATA_PATH = os.path.join(parent_dir, "Data")
PDF_FOLDER = os.path.join(DATA_PATH, "pdf")
IMAGE_FOLDER = os.path.join(DATA_PATH, "images")
METADATA_FILE = os.path.join(DATA_PATH, "image_metadata.json")
DB_PATH = os.path.join(parent_dir, "qdrant_db")

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
TEXT_COLLECTION = "textbook_text"
IMAGE_COLLECTION = "textbook_images"
TEXT_VECTOR_SIZE = 384
IMAGE_VECTOR_SIZE = 512
BATCH_SIZE = 64

print("Initializing Qdrant and embedding models...")
client = QdrantClient(path=DB_PATH)
text_encoder = SentenceTransformer(TEXT_MODEL)
clip_encoder = SentenceTransformer(CLIP_MODEL)
text_tokenizer = text_encoder.tokenizer


def create_collections():
    """Rebuilds the separate text and image vector collections."""
    existing = {collection.name for collection in client.get_collections().collections}
    for collection_name, vector_size in (
        (TEXT_COLLECTION, TEXT_VECTOR_SIZE),
        (IMAGE_COLLECTION, IMAGE_VECTOR_SIZE),
    ):
        if collection_name in existing:
            client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def extract_text_blocks(page):
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) > 6 and block[6] == 0 and isinstance(block[4], str):
            text = block[4].strip()
            if text:
                blocks.append((round(block[1]), block[0], block[1], block[3], text))
    return sorted(blocks, key=lambda block: (block[0], block[1]))


def repeated_lines_by_pdf(pages):
    line_pages = {}
    for page_index, blocks in enumerate(pages):
        seen_on_page = set()
        for block in blocks:
            for line in block[4].splitlines():
                normalized = " ".join(line.split()).casefold()
                if normalized:
                    seen_on_page.add(normalized)
        for line in seen_on_page:
            line_pages.setdefault(line, set()).add(page_index)

    minimum_pages = max(2, math.ceil(len(pages) * 0.30))
    return {
        line
        for line, page_indices in line_pages.items()
        if len(page_indices) >= minimum_pages
    }


def printed_page_number(blocks, page_height):
    lines = []
    for block in blocks:
        for line in block[4].splitlines():
            stripped = line.strip()
            if stripped:
                lines.append((block[2], block[3], stripped))
    for y0, y1, line in lines:
        if line.isdigit() and (y0 <= page_height * 0.10 or y1 >= page_height * 0.90):
            return line
    return None


def page_text_without_headers(blocks, repeated_lines):
    remaining = []
    for block in blocks:
        block_text = block[4].strip()
        if block_text.isdigit() or "reprint" in block_text.casefold():
            continue
        lines = [
            line.strip()
            for line in block_text.splitlines()
            if " ".join(line.split()).casefold() not in repeated_lines
        ]
        if lines:
            remaining.extend(lines)
    return re.sub(r"\s+", " ", " ".join(remaining)).strip()


def split_word_windows(text):
    words = text.split()
    if len(words) < 30:
        return []
    spans = [
        (start, min(start + 120, len(words)))
        for start in range(0, len(words), 100)
    ]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < 30:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return [" ".join(words[start:end]) for start, end in spans]


def process_pdfs():
    if not os.path.isdir(PDF_FOLDER):
        print(f"ERROR: PDF folder not found: {PDF_FOLDER}")
        print("PDFs processed: 0 of 0")
        return 0, 0

    pdf_files = [
        filename
        for filename in os.listdir(PDF_FOLDER)
        if filename.lower().endswith(".pdf")
    ]
    text_chunks = []
    processed_count = 0

    for pdf_file in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            with fitz.open(path) as document:
                page_blocks = [extract_text_blocks(page) for page in document]
                repeated_lines = repeated_lines_by_pdf(page_blocks)
                file_chunks = []
                for page_index, (page, blocks) in enumerate(zip(document, page_blocks)):
                    printed_page = printed_page_number(blocks, page.rect.height)
                    page_text = page_text_without_headers(blocks, repeated_lines)
                    for chunk_index, content in enumerate(split_word_windows(page_text)):
                        file_chunks.append(
                            {
                                "type": "text",
                                "content": content,
                                "page": page_index + 1,
                                "printed_page": printed_page,
                                "source": pdf_file,
                                "chunk_index": chunk_index,
                            }
                        )
            text_chunks.extend(file_chunks)
            processed_count += 1
        except Exception as error:
            print(f"ERROR: Could not read PDF {pdf_file}: {error}")

    token_counts = [
        len(text_tokenizer(chunk["content"], add_special_tokens=True, truncation=False)["input_ids"])
        for chunk in text_chunks
    ]
    long_chunks = sum(token_count > 200 for token_count in token_counts)
    print(f"Text chunks longer than 200 MiniLM tokens: {long_chunks}")

    for start in tqdm(
        range(0, len(text_chunks), BATCH_SIZE),
        desc="Embedding text chunks",
        unit="batch",
    ):
        batch_payloads = text_chunks[start:start + BATCH_SIZE]
        vectors = text_encoder.encode(
            [item["content"] for item in batch_payloads],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()
        points = [
            models.PointStruct(id=start + index, vector=vector, payload=payload)
            for index, (payload, vector) in enumerate(zip(batch_payloads, vectors))
        ]
        client.upsert(collection_name=TEXT_COLLECTION, points=points)

    print(f"Text chunks stored: {len(text_chunks)}")
    print(f"PDFs processed: {processed_count} of {len(pdf_files)}")
    return len(text_chunks), processed_count


def process_images():
    if not os.path.exists(METADATA_FILE):
        print(f"WARNING: Metadata file missing: {METADATA_FILE}")
        return 0

    with open(METADATA_FILE, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    points = []
    for item in metadata:
        filename = item["filename"]
        description = item["description"]
        image_path = os.path.join(IMAGE_FOLDER, filename)
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {filename}")
            continue

        try:
            with Image.open(image_path) as image:
                vector = clip_encoder.encode(image.copy()).tolist()
            points.append(
                models.PointStruct(
                    id=len(points),
                    vector=vector,
                    payload={
                        "type": "image",
                        "content": description,
                        "image_path": image_path,
                        "source": filename,
                        "page": 0,
                    },
                )
            )
            print(f"   Ingested: {filename}")
        except Exception as error:
            print(f"ERROR: Error with {filename}: {error}")

    if points:
        client.upsert(collection_name=IMAGE_COLLECTION, points=points)
    print(f"Images stored: {len(points)}")
    return len(points)


if __name__ == "__main__":
    create_collections()
    process_pdfs()
    process_images()
    print(f"SUCCESS! Database is ready at: {os.path.abspath(DB_PATH)}")
