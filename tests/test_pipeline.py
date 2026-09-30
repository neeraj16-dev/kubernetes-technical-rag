import json
from rag.pipeline import RAGPipeline

if __name__ == "__main__":
    pipeline = RAGPipeline()

    question = "How does Kubernetes handle container crash failover?"

    result = pipeline.query(
        question=question,
        fetch_k=20,
        top_n=5,
        verify=True,
    )

    print("\n" + "=" * 60)
    print("                    RAG ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("                    CITATIONS")
    print("=" * 60)
    print(json.dumps(result["citations"], indent=2))

    print("\n" + "=" * 60)
    print("              GROUNDING VERIFICATION")
    print("=" * 60)
    print(json.dumps(result["verification"], indent=2))

    print("\n" + "=" * 60)
    print("                RETRIEVED CHUNKS")
    print("=" * 60)
    print(json.dumps(result["retrieved_chunks"], indent=2))

    print("\n" + "=" * 60)
    print("                  LATENCY")
    print("=" * 60)
    print(json.dumps(result["metrics"], indent=2))

    print("\n" + "=" * 60)
    print("                 PIPELINE COMPLETE")
    print("=" * 60)