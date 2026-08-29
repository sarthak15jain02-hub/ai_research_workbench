from utils.embedding_model import load_embedding_model
from utils.file_handler import file_hash, save_uploaded_file
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store


def process_uploaded_files(uploaded_files, papers):
    """Ingest each PDF independently, so one failure never corrupts another paper."""
    results = []
    embedding_model = None

    for uploaded_file in uploaded_files:
        paper_id = file_hash(uploaded_file)
        if paper_id in papers:
            results.append({"name": uploaded_file.name, "status": "already_loaded"})
            continue

        try:
            if embedding_model is None:
                embedding_model = load_embedding_model()
            path = save_uploaded_file(uploaded_file, paper_id)
            documents = load_pdf(path, paper_id, uploaded_file.name)
            chunks = split_documents(documents)
            papers[paper_id] = {
                "id": paper_id,
                "name": uploaded_file.name,
                "path": path,
                "page_count": len(documents),
                "chunk_count": len(chunks),
                "documents": documents,
                "chunks": chunks,
                "vector_store": create_vector_store(chunks, embedding_model),
            }
            results.append({"name": uploaded_file.name, "status": "processed"})
        except Exception as error:
            results.append({"name": uploaded_file.name, "status": "failed", "error": str(error)})

    return papers, results
