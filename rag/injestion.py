from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader


def load_documents():
    loader = DirectoryLoader(
        "corpus/",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    return documents