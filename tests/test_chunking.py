from rag.chunking import fixed_size_chunks, structured_chunks
from rag.injestion import load_documents, load_pdf_with_layout

documents = load_pdf_with_layout("corpus")
chunks = structured_chunks(documents)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}:-")
    print(f"Chunk Metadata: {chunk.metadata}")
    print(f"Chunk Length: {len(chunk.page_content)}")
    print(f"Chunk Content: {chunk.page_content[:300]}")