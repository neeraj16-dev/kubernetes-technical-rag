from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import os
from langchain_core.documents import Document

def fixed_size_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(documents)

    page_chunk_counters = {}

    for chunk in chunks:
        source_path = chunk.metadata.get('source', '')
        filename = os.path.splitext(os.path.basename(source_path))[0]

        page = chunk.metadata.get('page',0)

        page_key = (filename, page)
        chunk_idx = page_chunk_counters.get(page_key, 0)
        page_chunk_counters[page_key] = chunk_idx + 1

        chunk_id = f"{filename}_p{page}_c{chunk_idx:02d}"

        chunk.metadata = {
            "chunk_id": chunk_id,
            "source": source_path,
            "page": page,
            "section": chunk.metadata.get("section", "N/A")
        }

    return chunks


def structured_chunks(documents):

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),            
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n","\n"," ", ""]
    )

    final_chunks = []
    page_chunk_counters = {}

    for doc in documents:
        source_path = doc.metadata.get("source", "")
        filename = os.path.splitext(os.path.basename(source_path))[0]
        page = doc.metadata.get("page", 0)

        header_splits = markdown_splitter.split_text(doc.page_content)

        for split in header_splits:
            section_title = (
                split.metadata.get("Header_3") 
                or split.metadata.get("Header_2")
                or split.metadata.get("Header_1")
                or "General"
            )

            sub_chunks = text_splitter.split_text(split.page_content)

            for text in sub_chunks:
                page_key = (filename, page)
                chunk_idx = page_chunk_counters.get(page_key, 0)
                page_chunk_counters[page_key] = chunk_idx + 1

                chunk_id = f"{filename}_p{page}_c{chunk_idx:02d}"

                final_chunks.append(
                    Document(
                        page_content = text.strip(),
                        metadata = {
                            "chunk_id": chunk_id,
                            "source": source_path,
                            "page": page,
                            "section": section_title,
                        }
                    )
                )

    return final_chunks 




