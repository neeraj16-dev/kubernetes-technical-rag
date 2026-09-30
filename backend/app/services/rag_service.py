from typing import Optional

from rag.pipeline import RAGPipeline

class RAGService:
    def __init__(self):
        print("Initializing RAG service...")
        self.pipeline = RAGPipeline()
        print("RAG service ready.")

    def query(
        self,
        question: str,
        top_n: int = 5,
        verify_grounding: bool = True
    ):
        return self.pipeline.query(
            question=question,
            top_n=top_n,
            verify=verify_grounding
        )

