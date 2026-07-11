from config import GOOGLE_API_KEY
from services.llm_service import load_llm
from services.prompt_template import get_prompt
from langchain_core.output_parsers import StrOutputParser


def ask_question(retriever, question):

    # Retrieve relevant chunks
    documents = retriever.invoke(question)
    print("=" * 50)
    print("Retrieved Docs:", len(documents))
    print("=" * 50)

    for i, document in enumerate(documents):
        print(f"\nDocument {i+1}")
        print(document.page_content[:500])
    
    # Load Gemini
    llm = load_llm(GOOGLE_API_KEY)

    # Retrieve relevant chunks
    documents = retriever.invoke(question)

    if len(documents) == 0:
        return "I could not find relevant information in the uploaded research paper."

    # Convert retrieved documents into one context string
    context = "\n\n".join(
    document.page_content
    for document in documents
)

    if context.strip() == "":
        return "I could not find relevant information in the uploaded research paper."

    # Load Prompt
    prompt = get_prompt()

    # Build LangChain pipeline
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # Generate answer
    try:
        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )
    except Exception:
        return "Unable to generate an answer at the moment. Please try again."
    return answer
