import os
from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.retrieval import hybrid_retrieve_and_rerank
from rag.generation import RagGenerator
from dotenv import load_dotenv

load_dotenv()

# 1. Setup handles and models
reranker = Reranker()
generator = RagGenerator(model_name="gemini-3.5-flash-lite")

handles = load_indexes()

# 2. Query Pipeline
query = "How does Kubernetes handle container failover in a deployment?"

# Fetch top 15 via Hybrid, narrow to top 3 via Cross-Encoder
reranked_results = hybrid_retrieve_and_rerank(
    query=query,
    handles=handles,
    reranker=reranker,
    fetch_k=15,
    top_n=3
)

# Extract only Document objects (drop the rerank score float)
final_docs = [doc for doc, score in reranked_results]

# 3. Generate Answer + Citations
output = generator.generate(question=query, docs=final_docs)

print("\n--- Model Output ---")
import json
print(json.dumps(output, indent=2))