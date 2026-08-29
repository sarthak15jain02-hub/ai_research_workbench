import os

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path, paper_id, paper_name):
    """Load page documents and attach metadata needed for filtering/citations."""
    documents = PyPDFLoader(pdf_path).load()
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ValueError("No selectable text was found. This may be a scanned PDF and needs OCR.")

    for document in documents:
        # PyPDFLoader's `page` is zero-based; citations should be one-based.
        document.metadata.update({
            "paper_id": paper_id,
            "paper_name": paper_name,
            "page_number": document.metadata.get("page", 0) + 1,
            "source_file": os.path.basename(pdf_path),
        })
    return documents
