import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
