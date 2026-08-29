import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


SECTION_PATTERNS = [
    (
        "abstract",
        r"\babstract\b",
    ),
    (
        "introduction",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?introduction\b",
    ),
    (
        "background",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:background|motivation)\b",
    ),
    (
        "related_work",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:related work|related works|literature review)\b",
    ),
    (
        "methods",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:method|methods|methodology|proposed method|proposed approach|approach)\b",
    ),
    (
        "dataset",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:dataset|data|materials)\b",
    ),
    (
        "experiments",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:experiment|experiments|experimental setup|evaluation)\b",
    ),
    (
        "results",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:result|results|findings|discussion)\b",
    ),
    (
        "conclusion",
        r"(?:^|\n)\s*(?:(?:\d+|[IVXLC]+)\.?\s+)?(?:conclusion|conclusions|future work)\b",
    ),
    (
        "references",
        r"(?:^|\n)\s*(?:references|bibliography)\b",
    ),
]


def _section_on_page(text, current_section):
    """
    Detect common research-paper headings from the start of each page.

    Uses search instead of exact line matching because PDF extraction often
    produces headings such as 'ABSTRACT Cricket...' or 'I. INTRODUCTION'.
    """

    page_start = "\n".join(
        text.splitlines()[:40]
    )

    for section, pattern in SECTION_PATTERNS:
        if re.search(
            pattern,
            page_start,
            flags=re.IGNORECASE,
        ):
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
