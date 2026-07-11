from utils.file_handler import save_uploaded_file
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embedding_model import load_embedding_model
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever


def build_retriever(uploaded_file):

    # Save PDF
    file_path = save_uploaded_file(uploaded_file)

    # Load PDF
    documents = load_pdf(file_path)

    # Split Documents
    chunks = split_documents(documents)

    # Load Embedding Model
    embedding_model = load_embedding_model()

    # Create Vector Store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # Create Retriever
    retriever = get_retriever(vector_store)

    return retriever, documents, chunks, file_path