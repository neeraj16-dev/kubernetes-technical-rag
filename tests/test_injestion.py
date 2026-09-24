from rag.injestion import load_documents

documents = load_documents()

for i, document in enumerate(documents[12:15]):
    print(f"\n--- Document {i + 1} ---")
    print(f"Metadata: {document.metadata}")
    print(f"Content:\n{document.page_content[:500]}")