from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class Citation(BaseModel):
    chunk_id: str = Field(description="The exact chunk_id of the source used (e.g., 01_Overview_p0_c78)")
    source: str = Field(description="The source file path or document name")
    page: int = Field(description="The page number where this information is located")

class GroundedAnswer(BaseModel):
    answer: str = Field(description="The detailed and comprehensive answer grounded strictly in the provided context.")
    citations: List[Citation] = Field(
        default_factory=list,
        description="The list of unique chunks explicitly used to formulate the answer."
    ) 

def format_context_for_prompt(docs: List[Document]) -> str:

    context_blocks = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id","N/A")
        src = doc.metadata.get("source","N/A")
        page = doc.metadata.get("page",0)
        section = doc.metadata.get("section", "N/A")

        header = f"--- CHUNK ID: {cid} | SOURCE: {src} | PAGE: {page} | SECTION: {section} ---"
        context_blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_blocks)

class RagGenerator:
    def __init__(self, model_name:str = "gemini-3.5-flash-lite"):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.structured_llm = self.llm.with_structured_output(GroundedAnswer, method="json_mode")

        self.prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                "You are an expert Kubernetes technical assistant.\n"
                "Answer the user query strictly using only the technical context provided below.\n"
                "Rules:\n"
                "1. If the provided context does not contain sufficient details to answer, state clearly that you don't know.\n"
                "2. Do not invent commands, flags, or configuration keys not present in the context.\n"
                "3. In the 'citations' list, include only the chunk_id, source, and page of the specific chunks you actually used to formulate the answer."
            ),
            (
                'human',
                "Context:\n{context}\n\n"
                "Question: {question}"
            )
        ])

        self.chain = self.prompt | self.structured_llm

    def generate(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        formatted_context = format_context_for_prompt(docs)
        response: GroundedAnswer = self.chain.invoke({
            "context": formatted_context,
            "question": question
        })

        return response.model_dump()
