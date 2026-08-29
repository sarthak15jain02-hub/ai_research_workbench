import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


SECTION_PATTERNS = [
    ("abstract", r"^\s*(?:\d+(?:\.\d+)?\s+)?abstract\s*$"),
    ("introduction", r"^\s*(?:\d+(?:\.\d+)?\s+)?introduction\s*$"),
    ("background", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:background|motivation)\s*$"),
    ("related_work", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:related work|literature review)\s*$"),
    ("methods", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:method|methods|methodology|proposed method|approach)\s*$"),
    ("dataset", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:dataset|data|materials)\s*$"),
    ("experiments", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:experiments?|experimental setup|evaluation)\s*$"),
    ("results", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:results?|findings|discussion)\s*$"),
    ("conclusion", r"^\s*(?:\d+(?:\.\d+)?\s+)?(?:conclusion|conclusions|future work)\s*$"),
    ("references", r"^\s*(?:references|bibliography)\s*$"),
]


def _section_on_page(text, current_section):
    # Section headings are usually isolated lines.  Keep prior section across pages.
    for line in text.splitlines()[:25]:
        for section, pattern in SECTION_PATTERNS:
            if re.match(pattern, line, flags=re.IGNORECASE):
                return section
    return current_section


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    current_section = "front_matter"
    prepared = []
    for page in documents:
        current_section = _section_on_page(page.page_content, current_section)
        page.metadata["section"] = current_section
        prepared.append(page)

    chunks = splitter.split_documents(prepared)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['paper_id']}:{index}"
    return chunks
