from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        print(f"Loading cross-encoder model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(self, query:str, docs: List[Document], top_n: int = 5) -> List[Tuple[Document, float]]:
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        doc_score_pairs = list(zip(docs, [float(s) for s in scores]))

        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_n]
    