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
STRUCTURAL_INTENTS = {
    "problem_statement",
    "research_gap",
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

def get_opening_chunks(paper, limit=8):
    """
    Return chunks from the beginning of one paper.

    Problem statements and research gaps are normally found in
    the Abstract, Introduction, Background, or Related Work.
    """

    return sorted(
        paper["chunks"],
        key=lambda chunk: (
            chunk.metadata.get("page_number", 999999),
            chunk.metadata.get("chunk_id", ""),
        ),
    )[:limit]

def retrieve_for_paper(paper, search_query, intent, dense_k=18, lexical_k=18):
    """
    Hybrid dense + BM25 retrieval for one paper.

    For structural questions such as Problem Statement and Research Gap,
    opening chunks are force-included because these answers are usually
    written near the start of an academic paper.
    """

    chunks = paper["chunks"]

    if not chunks:
        return []

    # Semantic retrieval using embeddings and FAISS.
    dense_results = paper["vector_store"].similarity_search_with_score(
        search_query,
        k=min(dense_k, len(chunks)),
    )

    chunk_by_id = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in chunks
    }

    scores = {}

    # Add FAISS semantic-search ranking.
    rrf_add(
        scores,
        [
            document.metadata["chunk_id"]
            for document, _ in dense_results
        ],
        weight=1.0,
    )

    # Add BM25 keyword-search ranking.
    bm25 = BM25Okapi(
        [
            tokenize(chunk.page_content)
            for chunk in chunks
        ]
    )

    bm25_scores = bm25.get_scores(
        tokenize(search_query)
    )

    bm25_indices = sorted(
        range(len(chunks)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )[:lexical_k]

    rrf_add(
        scores,
        [
            chunks[index].metadata["chunk_id"]
            for index in bm25_indices
            if bm25_scores[index] > 0
        ],
        weight=1.0,
    )

    # Important fix:
    # Problem statements and research gaps are mostly near the beginning
    # of a paper. Force page 1–3 chunks into the ranking.
    if intent in STRUCTURAL_INTENTS:
        opening_chunks = get_opening_chunks(
            paper,
            limit=8,
        )

        for position, chunk in enumerate(opening_chunks):
            chunk_id = chunk.metadata["chunk_id"]

            # This boost is intentionally stronger than normal RRF scores.
            # It ensures Abstract and Introduction evidence reaches Gemini.
            opening_boost = 0.10 - (position * 0.003)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + opening_boost
            )

    # Boost chunks from sections that match the question type.
    preferred_sections = INTENT_RULES.get(
        intent,
        {},
    ).get(
        "sections",
        set(),
    )

    for chunk_id in list(scores):
        chunk = chunk_by_id[chunk_id]
        section = chunk.metadata.get(
            "section",
            "unknown",
        )

        if section in preferred_sections:
            scores[chunk_id] += 0.015

        # Avoid using reference-list chunks as main answer evidence.
        if section == "references":
            scores[chunk_id] -= 0.03

    return [
        (chunk_by_id[chunk_id], score)
        for chunk_id, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

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


def ask_question(
    papers,
    selected_paper_ids,
    question,
    history=None,
    intent=None,
    allow_general_knowledge=False,
):
    if not selected_paper_ids:
        return {
            "answer": "Select at least one paper first.",
            "sources": [],
            "used_general_knowledge": False,
            "answer_type": "input_error",
        }

    resolved_intent = classify_intent(question, intent)
    results = retrieve(papers, selected_paper_ids, question, resolved_intent)

    if not results:
        return {
            "answer": (
                "I could not find relevant evidence in the selected paper(s) "
                "to answer this question."
            ),
            "sources": [],
            "used_general_knowledge": False,
            "answer_type": "insufficient_evidence",
        }

    llm = load_llm()

    paper_answer = (
        get_paper_prompt() | llm | StrOutputParser()
    ).invoke(
        {
            "context": evidence_context(results),
            "question": question,
            "history": format_history(history or []),
        }
    ).strip()

    sources = [
        {
            "paper": document.metadata["paper_name"],
            "page": document.metadata["page_number"],
            "section": document.metadata.get("section", "unknown"),
            "excerpt": document.page_content[:400],
        }
        for document, _ in results
    ]

    if paper_answer == "INSUFFICIENT_EVIDENCE":
        if not allow_general_knowledge:
            return {
                "answer": (
                    "I could not find sufficient evidence in the selected "
                    "paper(s) to answer this question."
                ),
                "sources": sources,
                "used_general_knowledge": False,
                "answer_type": "insufficient_evidence",
            }

        general_answer = (
            get_general_prompt() | llm | StrOutputParser()
        ).invoke(
            {
                "question": question,
            }
        )

        return {
            "answer": (
                "*This answer is based on Gemini's general knowledge. "
                "It is not supported by the uploaded paper(s).*"
                "\n\n"
                + general_answer
            ),
            "sources": sources,
            "used_general_knowledge": True,
            "answer_type": "general_knowledge",
        }

    return {
        "answer": paper_answer,
        "sources": sources,
        "used_general_knowledge": False,
        "answer_type": "paper_evidence",
    }