from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


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