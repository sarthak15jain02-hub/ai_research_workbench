from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store