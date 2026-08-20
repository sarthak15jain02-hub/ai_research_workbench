from utils.file_handler import save_uploaded_file
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embedding_model import load_embedding_model
from utils.vector_store import create_vector_store, add_to_vector_store
from utils.retriever import get_retriever


def process_uploaded_files(uploaded_files, vector_store, processed_filenames, all_documents, all_chunks):
    embedding_model = None
    new_files_processed = False

    for uploaded_file in uploaded_files:
        if uploaded_file.name in processed_filenames:
            continue

        if embedding_model is None:
            embedding_model = load_embedding_model()

        file_path = save_uploaded_file(uploaded_file)
        documents = load_pdf(file_path)
        chunks = split_documents(documents)

        if vector_store is None:
            vector_store = create_vector_store(chunks, embedding_model)
        else:
            vector_store = add_to_vector_store(vector_store, chunks)

        all_documents.extend(documents)
        all_chunks.extend(chunks)
        processed_filenames.add(uploaded_file.name)
        new_files_processed = True

    retriever = get_retriever(vector_store) if vector_store else None

    return vector_store, retriever, all_documents, all_chunks, processed_filenames, new_files_processed
