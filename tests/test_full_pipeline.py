from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.retrieval import hybrid_retrieve_and_rerank
from rag.generation import RagGenerator
from rag.verification import GroundingVerifier
import json

# Initialize components
handles = load_indexes()
reranker = Reranker()
generator = RagGenerator()
verifier = GroundingVerifier()

# 1. Execute query
query = "How to ensure no downtime during crash in Kubernetes?"

# 2. Retrieve & Rerank
reranked_results = hybrid_retrieve_and_rerank(
    query=query,
    handles=handles,
    reranker=reranker,
    fetch_k=15,
    top_n=3
)
top_docs = [doc for doc, score in reranked_results]

# 3. Generate
gen_output = generator.generate(question=query, docs=top_docs)

# 4. Verify Grounding
verification = verifier.verify(generation_output=gen_output, retrieved_docs=top_docs)

print("\n--- Generation Output ---")
print(json.dumps(gen_output, indent=2))

print("\n--- Grounding Verification ---")
print(json.dumps(verification, indent=2))