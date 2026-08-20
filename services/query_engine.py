from config import GOOGLE_API_KEY
from services.llm_service import load_llm
from services.prompt_template import get_prompt, get_general_prompt, get_rewrite_prompt
from langchain_core.output_parsers import StrOutputParser

NOT_FOUND_PHRASE = "i could not find this information in the uploaded research paper"

STRUCTURAL_KEYWORDS = [
    "problem statement",
    "research gap",
    "contribution",
    "motivation",
    "objective of the paper",
    "aim of the paper",
    "aim of this paper",
    "why this paper",
    "what problem does this paper solve",
    "background of the study",
    "purpose of the paper",
    "purpose of this study",
]


def is_structural_question(question):
    q = question.lower()
    return any(keyword in q for keyword in STRUCTURAL_KEYWORDS)


def get_opening_chunks(all_chunks, n=3):
    """
    Returns the first n chunks per paper (by paper_name metadata),
    preserving original document order. Abstract/Introduction is
    almost always at the start of a paper, so this guarantees
    inclusion even when semantic search misses it.
    """
    if not all_chunks:
        return []

    grouped = {}
    for chunk in all_chunks:
        name = chunk.metadata.get("paper_name", "unknown")
        grouped.setdefault(name, [])
        if len(grouped[name]) < n:
            grouped[name].append(chunk)

    opening = []
    for chunks in grouped.values():
        opening.extend(chunks)
    return opening


def format_history(history, max_turns=3):
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"Q: {turn['question']}\nA: {turn['answer']}")
    return "\n\n".join(lines)


def rewrite_query(question, history, llm):
    prompt = get_rewrite_prompt()
    chain = prompt | llm | StrOutputParser()
    try:
        conversation_history = format_history(history) if history else "None"
        rewritten = chain.invoke({
            "history": conversation_history,
            "question": question
        })
        return rewritten.strip()
    except Exception:
        return question


def ask_question(vector_store, question, history=None, all_chunks=None):
    llm = load_llm(GOOGLE_API_KEY)

    search_query = rewrite_query(question, history, llm)

    print("=" * 50)
    print("Original question:", question)
    print("Search query used:", search_query)
    print("=" * 50)

    results = vector_store.similarity_search_with_score(search_query, k=20)
    documents = [document for document, score in results]

    print("Retrieved Docs:", len(results))
    for i, (document, score) in enumerate(results):
        print(f"\nDocument {i+1} | Score: {score:.4f}")
        print(document.page_content[:200])
    print("=" * 50)

    if is_structural_question(question):
        opening_chunks = get_opening_chunks(all_chunks, n=3)
        existing_content = {d.page_content for d in documents}
        injected_count = 0
        for chunk in opening_chunks:
            if chunk.page_content not in existing_content:
                documents.append(chunk)
                existing_content.add(chunk.page_content)
                injected_count += 1
        print(f"Structural question detected — injected {injected_count} opening chunk(s).")
        print("=" * 50)

    context = "\n\n".join(document.page_content for document in documents) if documents else ""
    conversation_history = format_history(history)

    paper_answer = None
    if context.strip() != "":
        prompt = get_prompt()
        chain = prompt | llm | StrOutputParser()
        try:
            paper_answer = chain.invoke({
                "context": context,
                "question": question,
                "history": conversation_history
            })
        except Exception as e:
            print("PAPER PROMPT ERROR:", repr(e))
            return "Unable to generate an answer at the moment. Please try again."

    if paper_answer is None or NOT_FOUND_PHRASE in paper_answer.lower():
        general_prompt = get_general_prompt()
        chain = general_prompt | llm | StrOutputParser()
        try:
            fallback_answer = chain.invoke({"question": question})
        except Exception as e:
            print("FALLBACK ERROR:", repr(e))
            return "Unable to generate an answer at the moment. Please try again."
        return (
            "*(This response is based on Gemini's general knowledge, "
            "because the uploaded paper does not contain sufficient information.)*\n\n"
            + fallback_answer
        )

    return paper_answer