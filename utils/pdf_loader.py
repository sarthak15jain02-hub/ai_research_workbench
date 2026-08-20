import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    paper_name = os.path.basename(pdf_path)
    for document in documents:
        document.metadata["paper_name"] = paper_name

    return documents