import json
from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.retrieval import hybrid_retrieve_and_rerank

DATASET_PATH = "tests/eval_dataset.json"

handles = load_indexes()
reranker = Reranker()

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Auto-discovering expected_chunk_ids for {len(dataset)} questions...")

for item in dataset:
    # Use hybrid retrieval + reranker on the ground truth answer to locate the gold chunk
    search_query = f"{item['question']} {item['ground_truth']}"
    reranked = hybrid_retrieve_and_rerank(
        query=search_query,
        handles=handles,
        reranker=reranker,
        fetch_k=15,
        top_n=2
    )
    
    top_cids = [doc.metadata.get("chunk_id") for doc, score in reranked if doc.metadata.get("chunk_id")]
    item["expected_chunk_ids"] = top_cids
    print(f"Q: {item['question'][:40]}... -> Mapped: {top_cids}")

with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("\nUpdated tests/eval_dataset.json successfully!")