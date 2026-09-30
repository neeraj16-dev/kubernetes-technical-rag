from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.retrieval import hybrid_retrieve_and_rerank


handles = load_indexes()

reranker = Reranker()

query = "how to ensure no downtime during crash"

results = hybrid_retrieve_and_rerank(
    query=query,
    handles=handles,
    reranker=reranker,
    fetch_k=15,
    top_n=3
)

print(
    f"\n================ Reranked Results: {query} ================"
)

for rank, (doc, score) in enumerate(results, 1):

    print(
        f"\n[Rank {rank}] "
        f"Score: {score:.4f} | "
        f"ID: {doc.metadata.get('chunk_id')} | "
        f"Section: {doc.metadata.get('section')}"
    )

    print(f"Content:\n{doc.page_content[:250]}...")