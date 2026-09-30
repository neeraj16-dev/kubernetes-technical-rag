import os
import pickle
import re
import chromadb
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

INDEX_DIR = "data/indices"
BM25_FILE = os.path.join(INDEX_DIR, "bm25_index.pkl")
CHROMA_DIR = os.path.join(INDEX_DIR, "chroma_db")
COLLECTION_NAME = "k8_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def tokenize_corpus(texts: List[str]) -> List[str]:
    return [re.findall(r"\b\w+\b", text.lower()) for text in texts]

def build_bm25_index(chunks: List[Document]):
    os.makedirs(INDEX_DIR, exist_ok=True)

    corpus_tokens = tokenize_corpus([c.page_content for c in chunks])
    bm25 = BM25Okapi(corpus_tokens)

    bm25_payload = {
        "index": bm25,
        "chunks": chunks
    }

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_payload, f)

    print(f"[BM25] Built and saved BM25 index with {len(chunks)} documents to {BM25_FILE}")
    return bm25, chunks

def build_dense_index(chunks, batch_size: int = 64):
    os.makedirs(INDEX_DIR, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )

    ids = [c.metadata["chunk_id"] for c in chunks]
    contents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    print(f"Generating dense embeddings for {len(chunks)} chunks...")
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i+batch_size, len(chunks))
        batch_ids = ids[i:batch_end]
        batch_texts = contents[i:batch_end]
        batch_meta = metadatas[i:batch_end]

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta
        )

    print(f"Dense index built in ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
    return collection

def load_indexes():
    if not os.path.exists(BM25_FILE):
        raise FileNotFoundError(f"BM25 file not found at {BM25_FILE}. Run build_bm25_index first.")

    with open(BM25_FILE, "rb") as f:
        bm25_data = pickle.load(f)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return {
        "bm25": bm25_data["index"],
        "chunks": bm25_data["chunks"],
        "chroma_collection": collection,
        "embedding_model": embedding_model
    }