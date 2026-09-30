import json

from rag.indexing import load_indexes
from rag.reranking import Reranker
from rag.generation import RagGenerator
from rag.evaluation import RAGEvaluator


if __name__ == "__main__":
    # Load BM25, ChromaDB and embedding model
    handles = load_indexes()

    # Load reranker
    reranker = Reranker()

    # Load Gemini generator
    generator = RagGenerator()

    # Load evaluator / LLM judge
    evaluator = RAGEvaluator()

    # Run evaluation
    results = evaluator.evaluate_testset(
        testset_path="tests/eval_dataset.json",
        handles=handles,
        reranker=reranker,
        generator=generator,
        top_n=5
    )

    # Print benchmark scorecard
    print("\n" + "=" * 50)
    print("              BENCHMARK SCORECARD")
    print("=" * 50)

    print(f"Hit Rate @ 5         : {results['hit_rate_at_k'] * 100:.1f}%")
    print(f"Mean Reciprocal Rank : {results['mean_reciprocal_rank']:.3f}")
    print(f"Avg Faithfulness     : {results['avg_faithfulness'] * 100:.1f}%")
    print(f"Avg Answer Relevance : {results['avg_answer_relevance'] * 100:.1f}%")
    print(f"Avg Ground Truth     : {results['avg_ground_truth_score'] * 100:.1f}%")

    print("=" * 50)

    # Save detailed evaluation report
    with open("tests/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Detailed report saved to tests/eval_report.json")
