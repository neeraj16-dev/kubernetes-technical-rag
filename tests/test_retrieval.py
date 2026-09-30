from rag.ingestion import load_pdf_with_layout
from rag.chunking import structured_chunks
from rag.indexing import build_bm25_index, build_dense_index, load_indexes
from rag.retrieval import hybrid_retrieve

if __name__ == "__main__":
    # 1. Ingest & Chunk (or load saved)
    print("Ingesting and chunking...")
    raw_docs = load_pdf_with_layout("corpus")
    chunks = structured_chunks(raw_docs)
    print(f"Total structured chunks: {len(chunks)}")

    # 2. Build Indexes
    build_bm25_index(chunks)
    build_dense_index(chunks)

    # 3. Load & Retrieve
    handles = load_indexes()
    
    test_queries = [
        "What happens when a Kubernetes API request cannot be authenticated?"
    ]

    for q in test_queries:
        print(f"\n================ Query: {q} ================")
        results = hybrid_retrieve(q, handles, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"\nResult {i} | ID: {res.metadata['chunk_id']} | Section: {res.metadata.get('section')}")
            print(f"Content: {res.page_content[:200]}...")