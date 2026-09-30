from typing import List, Dict, Any
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os


class GroundingVerdict(BaseModel):
    is_grounded: bool = Field(description="True if every claim in the answer is fully supported by the cited chunks.")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Any specific factual statements made in the answer that cannot be verified by the cited chunks."
    )
    hallucination_score: float = Field(
        description="0.0 for completely grounded, 1.0 for completely hallucinated."
    )
    explanation: str = Field(description="Brief reasoning behind the evaluation.")


class GroundingVerifier:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        self.eval_llm = ChatGoogleGenerativeAI(
            model=model,
        ).with_structured_output(GroundingVerdict, method="json_mode")

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an impartial RAG grounding evaluator.\n"
                "You will receive an Answer and the Cited Source Chunks.\n"
                "Verify whether the Answer is completely supported by the Cited Source Chunks.\n"
                "- If the Answer introduces facts, numbers, or configuration not found in the chunks, flag it.\n"
                "- Minor paraphrasing that maintains meaning is grounded."
            ),
            (
                "human",
                "Cited Chunks:\n{cited_text}\n\n"
                "Generated Answer:\n{answer}"
            )
        ])

        self.chain = self.prompt | self.eval_llm

    def verify(
        self, 
        generation_output: Dict[str, Any], 
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        answer = generation_output.get("answer", "")
        citations = generation_output.get("citations", [])

        # 1. Structural check: Validate cited chunk_ids against context
        valid_chunk_map = {doc.metadata.get("chunk_id"): doc for doc in retrieved_docs}
        valid_citations = []
        phantom_citations = []

        for c in citations:
            cid = c.get("chunk_id")
            if cid in valid_chunk_map:
                valid_citations.append(valid_chunk_map[cid])
            else:
                phantom_citations.append(cid)

        # 2. Textual entailment check: Did the LLM fabricate claims?
        cited_text = "\n\n".join([f"[{doc.metadata.get('chunk_id')}]: {doc.page_content}" for doc in valid_citations])
        
        if not cited_text:
            return {
                "is_grounded": False,
                "unsupported_claims": ["No valid cited chunks found in retrieved context."],
                "hallucination_score": 1.0,
                "phantom_citations": phantom_citations,
                "explanation": "The model produced citations that did not match any retrieved chunks."
            }

        eval_result = self.chain.invoke({
            "cited_text": cited_text,
            "answer": answer
        })

        result_dict = eval_result.model_dump()
        result_dict["phantom_citations"] = phantom_citations
        return result_dict