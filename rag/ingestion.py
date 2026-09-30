from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import pymupdf4llm
import os
import glob
from langchain_core.documents import Document

def load_documents():
    loader = DirectoryLoader(
        "corpus/",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    return documents

def load_pdf_with_layout(corpus_dir: str = "corpus"):
    documents = []
    pdf_paths = glob.glob(os.path.join(corpus_dir, "**/*.pdf"), recursive=True)

    print(f"Found {len(pdf_paths)} PDF(s) in '{corpus_dir}'. Starting layout extraction...")

    for i, pdf_path in enumerate(pdf_paths, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{i}/{len(pdf_paths)}] Processing: {filename} ...", end=" ", flush=True)

        try:
            # table_strategy=None disables deep layout heuristic loops that stall PyMuPDF
            pages_data = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                show_progress=False
            )

            for p in pages_data:
                doc = Document(
                    page_content=p.get("text", ""),
                    metadata={
                        "source": pdf_path,
                        "page": p.get("metadata", {}).get("page", 0),
                    },
                )
                documents.append(doc)

            print(f"Done ({len(pages_data)} pages)")

        except Exception as e:
            print(f"Failed! Error: {e}")

    print(f"Extraction complete. Total document pages loaded: {len(documents)}")
    return documents