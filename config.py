import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _setting(name, default=""):
    """Read local environment settings or hosted Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


GOOGLE_API_KEY = _setting("GOOGLE_API_KEY")
GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_SIZE = int(_setting("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_setting("CHUNK_OVERLAP", "150"))
