from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")
    return FAISS.from_documents(chunks, embedding_model)
