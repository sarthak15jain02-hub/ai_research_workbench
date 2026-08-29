import re

from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

from services.llm_service import load_llm
from services.prompt_template import get_general_prompt, get_paper_prompt


INTENT_RULES = {
    "problem_statement": {
        "phrases": ["problem statement", "problem does", "problem addressed", "why this paper", "motivation", "aim of the paper", "objective of the paper"],
        "query": "problem addressed limitation of existing methods motivation research gap objective",
        "sections": {"abstract", "introduction", "background", "related_work", "conclusion"},
    },
    "research_gap": {
        "phrases": ["research gap", "gap", "existing work", "previous work"],
        "query": "research gap limitation previous existing methods insufficient",
        "sections": {"abstract", "introduction", "background", "related_work"},
    },
    "methodology": {
        "phrases": ["methodology", "method", "approach", "architecture", "pipeline"],
        "query": "methodology proposed approach method architecture training pipeline",
        "sections": {"methods", "dataset", "experiments"},
    },
    "dataset": {
        "phrases": ["dataset", "data", "preprocessing", "augmentation", "train test"],
        "query": "dataset data source size classes training test preprocessing augmentation",
        "sections": {"dataset", "methods", "experiments"},
    },
    "results": {
        "phrases": ["result", "accuracy", "metric", "performance", "evaluation", "experiment"],
        "query": "experimental results evaluation metrics accuracy performance comparison ablation",
        "sections": {"experiments", "results", "conclusion"},
    },
    "limitations": {
        "phrases": ["limitation", "drawback", "weakness"],
        "query": "limitations drawbacks challenges constraints",
        "sections": {"results", "conclusion", "discussion"},
    },
    "future_work": {
        "phrases": ["future work", "future direction", "extension"],
        "query": "future work future directions extensions improvements",
        "sections": {"conclusion", "results"},
    },
}


def classify_intent(question, explicit_intent=None):
    if explicit_intent and explicit_intent in INTENT_RULES:
        return explicit_intent
    lowered = question.lower()
    for intent, rule in INTENT_RULES.items():
        if any(phrase in lowered for phrase in rule["phrases"]):
            return intent
    return "general"


def build_search_query(question, intent):
    expansion = INTENT_RULES.get(intent, {}).get("query", "")
    return f"{question}\n{expansion}".strip()


def tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())


def rrf_add(scores, ranked_ids, weight=1.0, constant=60):
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (constant + rank)


def retrieve_for_paper(paper, search_query, intent, dense_k=18, lexical_k=18):
    """Hybrid dense + BM25 retrieval, restricted to one paper."""
    chunks = paper["chunks"]
    dense = paper["vector_store"].similarity_search_with_score(search_query, k=min(dense_k, len(chunks)))
    by_id = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    scores = {}
    rrf_add(scores, [doc.metadata["chunk_id"] for doc, _ in dense], weight=1.0)

    bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])
    lexical_scores = bm25.get_scores(tokenize(search_query))
    lexical_indices = sorted(range(len(chunks)), key=lambda i: lexical_scores[i], reverse=True)[:lexical_k]
    rrf_add(scores, [chunks[index].metadata["chunk_id"] for index in lexical_indices if lexical_scores[index] > 0], weight=1.0)

    preferred_sections = INTENT_RULES.get(intent, {}).get("sections", set())
    for chunk_id, score in list(scores.items()):
        if by_id[chunk_id].metadata.get("section") in preferred_sections:
            scores[chunk_id] = score + 0.015
        if by_id[chunk_id].metadata.get("section") == "references":
            scores[chunk_id] -= 0.03

    return [(by_id[chunk_id], score) for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def retrieve(papers, selected_paper_ids, question, intent):
    search_query = build_search_query(question, intent)
    all_results = []
    for paper_id in selected_paper_ids:
        all_results.extend(retrieve_for_paper(papers[paper_id], search_query, intent)[:8])
    # At most six chunks per paper prevents one paper from monopolising multi-PDF context.
    per_paper = {}
    for document, score in sorted(all_results, key=lambda item: item[1], reverse=True):
        paper_id = document.metadata["paper_id"]
        if len(per_paper.setdefault(paper_id, [])) < 6:
            per_paper[paper_id].append((document, score))
    return [item for items in per_paper.values() for item in items]


def format_history(history, max_turns=3):
    recent = history[-max_turns:]
    return "\n\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent) or "None"


def evidence_context(results):
    return "\n\n".join(
        "[Paper: {name} | Page: {page} | Section: {section}]\n{text}".format(
            name=document.metadata["paper_name"],
            page=document.metadata["page_number"],
            section=document.metadata.get("section", "unknown"),
            text=document.page_content,
        )
        for document, _ in results
    )


def ask_question(papers, selected_paper_ids, question, history=None, intent=None, allow_general_knowledge=False):
    if not selected_paper_ids:
        return {"answer": "Select at least one paper first.", "sources": [], "used_general_knowledge": False}

    resolved_intent = classify_intent(question, intent)
    results = retrieve(papers, selected_paper_ids, question, resolved_intent)
    if not results:
        return {"answer": "I could not find relevant evidence in the selected paper(s).", "sources": [], "used_general_knowledge": False}

    llm = load_llm()
    paper_answer = (get_paper_prompt() | llm | StrOutputParser()).invoke({
        "context": evidence_context(results),
        "question": question,
        "history": format_history(history or []),
    }).strip()

    sources = [{
        "paper": document.metadata["paper_name"],
        "page": document.metadata["page_number"],
        "section": document.metadata.get("section", "unknown"),
        "excerpt": document.page_content[:400],
    } for document, _ in results]

    if paper_answer == "INSUFFICIENT_EVIDENCE":
        if not allow_general_knowledge:
            return {"answer": "I could not find sufficient evidence in the selected paper(s) to answer this.", "sources": sources, "used_general_knowledge": False}
        general = (get_general_prompt() | llm | StrOutputParser()).invoke({"question": question})
        return {"answer": "*General-knowledge answer; not supported by the uploaded paper(s).*\n\n" + general, "sources": sources, "used_general_knowledge": True}

    return {"answer": paper_answer, "sources": sources, "used_general_knowledge": False}
