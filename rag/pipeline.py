import time
from typing import Dict, Any, List

from langchain_core.documents import Document

from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.retrieval import hybrid_retrieve_and_rerank
from rag.generation import RagGenerator
from rag.verification import GroundingVerifier


class RAGPipeline:

    def __init__(
        self,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        generator_model: str = "gemini-3.5-flash-lite",
        verifier_model: str = "gemini-3.5-flash-lite",
    ):
        print("Initializing RAG Pipeline components...")

        self.handles = load_indexes()

        self.reranker = Reranker(
            model_name=reranker_model
        )

        self.generator = RagGenerator(
            model_name=generator_model
        )

        self.verifier = GroundingVerifier(
            model=verifier_model
        )

        print("RAG Pipeline ready.")

    def query(
        self,
        question: str,
        fetch_k: int = 20,
        top_n: int = 5,
        verify: bool = True
    ) -> Dict[str, Any]:

        """
        Executes the complete RAG workflow:

        Hybrid Retrieval
            ↓
        Cross-Encoder Reranking
            ↓
        Gemini Generation
            ↓
        Grounding Verification
        """

        metrics: Dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()

        reranked_results = hybrid_retrieve_and_rerank(
            query=question,
            handles=self.handles,
            reranker=self.reranker,
            fetch_k=fetch_k,
            top_n=top_n
        )

        metrics["retrieval_rerank_ms"] = round(
            (time.perf_counter() - t0) * 1000,
            2
        )

        top_docs: List[Document] = [
            doc for doc, score in reranked_results
        ]

        t0 = time.perf_counter()

        gen_output = self.generator.generate(
            question=question,
            docs=top_docs
        )

        metrics["generation_ms"] = round(
            (time.perf_counter() - t0) * 1000,
            2
        )

        verification_result = None

        if verify:

            t0 = time.perf_counter()

            verification_result = self.verifier.verify(
                generation_output=gen_output,
                retrieved_docs=top_docs
            )

            metrics["verification_ms"] = round(
                (time.perf_counter() - t0) * 1000,
                2
            )

        metrics["total_latency_ms"] = round(
            (time.perf_counter() - total_start) * 1000,
            2
        )

        retrieved_contexts = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
                "content_preview": doc.page_content[:200] + "..."
            }
            for doc in top_docs
        ]

        return {
            "query": question,
            "answer": gen_output.get("answer", ""),
            "citations": gen_output.get("citations", []),
            "verification": verification_result,
            "retrieved_chunks": retrieved_contexts,
            "metrics": metrics
        }