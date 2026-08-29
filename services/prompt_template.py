from langchain_core.prompts import PromptTemplate


def get_paper_prompt():
    return PromptTemplate(
        input_variables=["context", "question", "history"],
        template="""
You are a precise research-paper assistant. Answer only from the supplied evidence.

Rules:
1. Every factual statement about a paper must include one or more citations in the
   form [Paper name, p. N]. Use only citations that appear in the evidence.
2. Do not merge claims from different papers. Name each paper when comparing them.
3. If the evidence cannot answer the question, return exactly: INSUFFICIENT_EVIDENCE
4. Conversation history can resolve a pronoun but is not evidence.
5. Do not use outside knowledge, and do not invent a problem statement or result.

Conversation history:
{history}

Evidence:
{context}

Question:
{question}

Answer:
""",
    )


def get_general_prompt():
    return PromptTemplate(
        input_variables=["question"],
        template="""
Answer this question using general knowledge. Clearly state that this is not
evidence from an uploaded paper. Do not claim that any paper says it.

Question: {question}
Answer:
""",
    )
