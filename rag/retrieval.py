from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import re 
from rag.reranking import Reranker

def retrieve_bm25(query: str, bm25_index, chunks: List[Document], top_k: int = 20) -> List[Document]:
    tokenized_query = re.findall(r"\b\w+\b", query.lower())
    scores = bm25_index.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)[:top_k]
    return [chunks[idx] for idx in ranked_indices]

def retrieve_dense(query:str, collection, model, top_k: int = 20) -> List[Document]:
    query_vector = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )

    retrieved_docs = []
    if results and "documents" in results and results["documents"]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append(Document(page_content=text, metadata=meta))

    return retrieved_docs

def reciprocal_rank_fusion(ranked_lists: List[List[Document]], k:int = 60, top_k:int = 10) -> List[Document]:
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, Document] = {}

    for doc_list in ranked_lists:
        for rank,doc in enumerate(doc_list, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue

            doc_store[chunk_id] = doc
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0

            rrf_scores[chunk_id] += 1.0/(k+rank)

    sorted_chunk_ids = sorted(rrf_scores.keys(), key = lambda cid: rrf_scores[cid], reverse=True)

    return [doc_store[cid] for cid in sorted_chunk_ids[:top_k]]

def hybrid_retrieve(query: str, handles: Dict[str, Any], top_k:int = 5) -> List[Document]:
    bm25_docs = retrieve_bm25(query, handles["bm25"],handles["chunks"], top_k=20)
    dense_docs = retrieve_dense(query, handles["chroma_collection"], handles["embedding_model"], top_k=20)

    fused_docs = reciprocal_rank_fusion([bm25_docs, dense_docs], k=60, top_k=top_k)
    return fused_docs


def hybrid_retrieve_and_rerank(
        query: str,
        handles: Dict[str, Any],
        reranker: Reranker,
        fetch_k: int = 15,
        top_n: int = 4
) -> List[Tuple[Document, float]]:

    candidate_docs = hybrid_retrieve(
        query=query,
        handles=handles,
        top_k=fetch_k
    )

    reranked_docs = reranker.rerank(
        query=query,
        docs=candidate_docs,
        top_n=top_n
    )

    return reranked_docs