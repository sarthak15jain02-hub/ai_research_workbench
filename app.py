import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

from services.rag_pipeline import process_uploaded_files
from services.query_engine import ask_question
from services.research_tasks import TASKS

# Page Configuration

st.set_page_config(
    page_title="Intelligent Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Title

st.title("📚 Intelligent Research Paper Assistant")
st.markdown(
    "Upload one or more research papers and interact with them using **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini**."
)
st.divider()

# Sidebar

st.sidebar.title("⚙️ Settings")
st.sidebar.info(
    "Upload research papers and use Quick Research Actions or ask your own questions."
)

# Upload Research Papers

st.header("📄 Upload Research Papers")
uploaded_files = st.file_uploader(
    "Choose PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing research paper(s)..."):
        (
            vector_store,
            retriever,
            all_documents,
            all_chunks,
            processed_filenames,
            new_files_processed
        ) = process_uploaded_files(
            uploaded_files,
            st.session_state.vector_store,
            st.session_state.processed_filenames,
            st.session_state.all_documents,
            st.session_state.all_chunks
        )

        st.session_state.vector_store = vector_store
        st.session_state.all_documents = all_documents
        st.session_state.all_chunks = all_chunks
        st.session_state.processed_filenames = processed_filenames

    st.success(f"✅ {len(st.session_state.processed_filenames)} paper(s) loaded: {', '.join(st.session_state.processed_filenames)}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Pages", len(st.session_state.all_documents))
    with col2:
        st.metric("Total Chunks", len(st.session_state.all_chunks))

st.divider()

# Quick Research Actions

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
    if not st.session_state.vector_store:
        st.warning("Please upload a research paper first.")
    elif selected_task is None:
        st.warning("Please select a research action.")
    else:
        with st.spinner("Generating response..."):
            answer = ask_question(
                st.session_state.vector_store,
                TASKS[selected_task],
                all_chunks=st.session_state.all_chunks
            )
            st.session_state.chat_history.append(
                {
                    "type": "Quick Action",
                    "question": selected_task,
                    "answer": answer
                }
            )

st.divider()

# Ask Questions

st.header("💬 Ask Questions")
question = st.text_input(
    "Ask anything about the uploaded research paper(s)"
)
ask_button = st.button(
    "Ask",
    use_container_width=True
)

st.divider()

# Answer

st.header("🤖 Conversation")
if ask_button:
    if not st.session_state.vector_store:
        st.warning("Please upload a research paper first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = ask_question(
                st.session_state.vector_store,
                question,
                history=st.session_state.chat_history,
                all_chunks=st.session_state.all_chunks
            )
        st.session_state.chat_history.append(
            {
                "type": "Ask",
                "question": question,
                "answer": answer
            }
        )

for turn in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(f"**[{turn['type']}]** {turn['question']}")
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])