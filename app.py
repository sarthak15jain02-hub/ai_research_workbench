import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

from services.rag_pipeline import build_retriever
from services.query_engine import ask_question
from services.research_tasks import TASKS


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 Intelligent Research Paper Assistant")

st.markdown(
    "Upload a research paper and interact with it using **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini**."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

st.sidebar.info(
    "Upload a research paper and use Quick Research Actions or ask your own questions."
)

# --------------------------------------------------
# Upload Research Paper
# --------------------------------------------------

st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type="pdf",
    accept_multiple_files=False
)

retriever = None
documents = []
chunks = []

if uploaded_file:

    with st.spinner("Processing research paper..."):

        retriever, documents, chunks, file_path = build_retriever(uploaded_file)

    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Pages", len(documents))

    with col2:
        st.metric("Chunks", len(chunks))

st.divider()

# --------------------------------------------------
# Quick Research Actions
# --------------------------------------------------

st.header("📑 Quick Research Actions")

selected_task = st.selectbox(
    "Choose an action",
    list(TASKS.keys()),
    index=None,
    placeholder="Select a research action..."
)

generate_button = st.button(
    "Generate",
    use_container_width=True
)

if generate_button:
    if not uploaded_file:
        st.warning("Please upload a research paper first.")
    elif selected_task is None:
        st.warning("Please select a research action.")
    else:
        with st.spinner("Generating response..."):
            answer = ask_question(
                retriever,
                TASKS[selected_task]
            )

            st.session_state.chat_history.append(
                {
                    "type": "Quick Action",
                    "question": selected_task,
                    "answer": answer
                }
            )
            st.markdown(answer)
            
st.divider()

# --------------------------------------------------
# Ask Questions
# --------------------------------------------------

st.header("💬 Ask Questions")
question = st.text_input(
    "Ask anything about the uploaded research paper"
)
ask_button = st.button(
    "Ask",
    use_container_width=True
)
st.divider()

# --------------------------------------------------
# Answer
# --------------------------------------------------

st.header("🤖 Answer")
if ask_button:
    if not uploaded_file:
        st.warning("Please upload a research paper first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = ask_question(
                retriever,
                question
            )
        st.markdown(answer)
        