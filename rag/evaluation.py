import os
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag.retrieval import hybrid_retrieve_and_rerank
from rag.generation import RagGenerator

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# 1. LLM Judge Output Schema
# ---------------------------------------------------------

class GenerationEvalScores(BaseModel):
    faithfulness_score: float = Field(
        description="Score from 0.0 to 1.0 indicating how faithfully the answer is supported by the provided context."
    )

    answer_relevance_score: float = Field(
        description="Score from 0.0 to 1.0 indicating how directly the answer answers the question."
    )

    ground_truth_score: float = Field(
        description="Score from 0.0 to 1.0 indicating how accurately the generated answer matches the provided ground truth."
    )

    critique: str = Field(
        description="Brief explanation of the scores and any missing or incorrect information."
    )


# ---------------------------------------------------------
# 2. RAG Evaluator
# ---------------------------------------------------------

class RAGEvaluator:

    def __init__(self, model: str = "gemini-3.5-flash-lite"):

        self.judge = ChatGoogleGenerativeAI(
            model=model,
        ).with_structured_output(GenerationEvalScores, method="json_mode")

        self.judge_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an objective RAG benchmark evaluator.\n\n"

                "Evaluate the generated answer using the Question, "
                "Context, and Ground Truth provided below.\n\n"

                "Evaluate three dimensions from 0.0 to 1.0:\n\n"

                "1. Faithfulness:\n"
                "Determine whether the claims in the generated answer "
                "are supported by the provided Context. "
                "Do not give credit for information that comes from outside the Context.\n\n"

                "2. Answer Relevance:\n"
                "Determine whether the generated answer directly addresses "
                "the user's Question and avoids unnecessary information.\n\n"

                "3. Ground Truth Accuracy:\n"
                "Compare the generated answer with the Ground Truth. "
                "Give a high score when the generated answer correctly conveys "
                "the important information contained in the Ground Truth. "
                "Do not require identical wording.\n\n"

                "Important:\n"
                "- Do not judge based on your own outside Kubernetes knowledge.\n"
                "- Use the provided Context and Ground Truth for evaluation.\n"
                "- If the generated answer contains unsupported claims, reduce the faithfulness score.\n"
                "- If important information from the Ground Truth is missing or incorrect, reduce the ground truth score."
            ),
            (
                "human",
                "Question:\n{question}\n\n"

                "Context:\n{context}\n\n"

                "Ground Truth:\n{ground_truth}\n\n"

                "Generated Answer:\n{answer}"
            )
        ])

        self.judge_chain = self.judge_prompt | self.judge


    # ---------------------------------------------------------
    # 3. Retrieval Metrics
    # ---------------------------------------------------------

    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_chunk_ids: List[str],
        expected_ids: List[str]
    ) -> Dict[str, float]:

        """
        Calculates:
        - Hit: 1 if any expected chunk was retrieved, otherwise 0
        - Reciprocal Rank: 1/rank of the first expected chunk
        """

        hit = 0.0
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1
        ):

            if chunk_id in expected_ids:

                hit = 1.0
                reciprocal_rank = 1.0 / rank

                break

        return {
            "hit": hit,
            "reciprocal_rank": reciprocal_rank
        }


    # ---------------------------------------------------------
    # 4. Evaluate Complete Test Set
    # ---------------------------------------------------------

    def evaluate_testset(
        self,
        testset_path: str,
        handles: Dict[str, Any],
        reranker,
        generator: RagGenerator,
        top_n: int = 3
    ) -> Dict[str, Any]:

        # Load evals.json
        with open(
            testset_path,
            "r",
            encoding="utf-8"
        ) as f:

            testset = json.load(f)


        total_queries = len(testset)

        hits = 0.0
        mrr_total = 0.0

        total_faithfulness = 0.0
        total_relevance = 0.0
        total_ground_truth_score = 0.0

        detailed_results = []


        print(
            f"\n--- Starting RAG Evaluation on "
            f"{total_queries} queries ---"
        )


        # -----------------------------------------------------
        # Process each question
        # -----------------------------------------------------

        for i, item in enumerate(testset, 1):

            question = item["question"]

            expected_cids = item.get(
                "expected_chunk_ids",
                []
            )

            ground_truth = item.get(
                "ground_truth",
                "" 
            )


            print(
                f"\n[{i}/{total_queries}] "
                f"{question}"
            )


            # -------------------------------------------------
            # Step 1: Hybrid Retrieval + Reranking
            # -------------------------------------------------

            reranked = hybrid_retrieve_and_rerank(
                query=question,
                handles=handles,
                reranker=reranker,
                fetch_k=15,
                top_n=top_n
            )


            # Remove reranker scores
            top_docs = [
                doc
                for doc, score in reranked
            ]


            # Extract retrieved chunk IDs
            retrieved_cids = [
                doc.metadata.get(
                    "chunk_id",
                    ""
                )
                for doc in top_docs
            ]


            # -------------------------------------------------
            # Step 2: Retrieval Evaluation
            # -------------------------------------------------

            retrieval_metrics = self.calculate_retrieval_metrics(
                retrieved_chunk_ids=retrieved_cids,
                expected_ids=expected_cids
            )


            hits += retrieval_metrics["hit"]
            mrr_total += retrieval_metrics["reciprocal_rank"]


            # -------------------------------------------------
            # Step 3: Generate Answer
            # -------------------------------------------------

            gen_output = generator.generate(
                question=question,
                docs=top_docs
            )

            answer = gen_output.get(
                "answer",
                ""
            )


            # -------------------------------------------------
            # Step 4: Prepare Context for Judge
            # -------------------------------------------------

            context_str = "\n\n".join(
                [
                    doc.page_content
                    for doc in top_docs
                ]
            )


            # -------------------------------------------------
            # Step 5: LLM Judge
            # -------------------------------------------------

            eval_scores: GenerationEvalScores = (
                self.judge_chain.invoke({
                    "question": question,
                    "context": context_str,
                    "ground_truth": ground_truth,
                    "answer": answer
                })
            )


            # -------------------------------------------------
            # Step 6: Accumulate Generation Metrics
            # -------------------------------------------------

            total_faithfulness += (
                eval_scores.faithfulness_score
            )

            total_relevance += (
                eval_scores.answer_relevance_score
            )

            total_ground_truth_score += (
                eval_scores.ground_truth_score
            )


            # -------------------------------------------------
            # Step 7: Save Detailed Result
            # -------------------------------------------------

            detailed_results.append({

                "question": question,

                "expected_chunk_ids": expected_cids,

                "retrieved_chunk_ids": retrieved_cids,

                "ground_truth": ground_truth,

                "generated_answer": answer,

                "hit": retrieval_metrics["hit"],

                "reciprocal_rank":
                    retrieval_metrics["reciprocal_rank"],

                "faithfulness":
                    eval_scores.faithfulness_score,

                "answer_relevance":
                    eval_scores.answer_relevance_score,

                "ground_truth_score":
                    eval_scores.ground_truth_score,

                "critique":
                    eval_scores.critique
            })


            # -------------------------------------------------
            # Console Output
            # -------------------------------------------------

            print(
                f"Hit: {retrieval_metrics['hit']:.0f} | "
                f"RR: {retrieval_metrics['reciprocal_rank']:.2f} | "
                f"Faithfulness: {eval_scores.faithfulness_score:.2f} | "
                f"Relevance: {eval_scores.answer_relevance_score:.2f} | "
                f"Ground Truth: {eval_scores.ground_truth_score:.2f}"
            )


        # -----------------------------------------------------
        # 8. Final Evaluation Summary
        # -----------------------------------------------------

        summary = {

            "total_queries": total_queries,

            "hit_rate_at_k":
                hits / total_queries,

            "mean_reciprocal_rank":
                mrr_total / total_queries,

            "avg_faithfulness":
                total_faithfulness / total_queries,

            "avg_answer_relevance":
                total_relevance / total_queries,

            "avg_ground_truth_score":
                total_ground_truth_score / total_queries,

            "details":
                detailed_results
        }


        return summary