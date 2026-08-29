import streamlit as st

from services.query_engine import ask_question
from services.rag_pipeline import process_uploaded_files
from services.research_tasks import TASKS
from utils.file_handler import file_hash

st.set_page_config(page_title="AI Research Workbench", page_icon="📚", layout="wide")

for key, default in {
    "papers": {},
    "chat_history": [],
    "ingestion_results": [],
    "last_upload_signature": (),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📚 AI Research Workbench")
st.caption("Ask evidence-grounded questions across one or more research papers.")

with st.sidebar:
    st.header("Workspace")
    if st.button("Clear workspace", use_container_width=True):
        st.session_state.papers = {}
        st.session_state.chat_history = []
        st.session_state.ingestion_results = []
        st.session_state.last_upload_signature = ()
        st.rerun()
    allow_general_knowledge = st.toggle("Allow general-knowledge fallback", value=False)
    st.caption("Keep this off for strict paper-grounded answers.")

st.header("1. Upload PDFs")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="PDFs are processed automatically after upload."
)

if uploaded_files:
    current_upload_signature = tuple(
        sorted(file_hash(uploaded_file) for uploaded_file in uploaded_files)
    )

    if current_upload_signature != st.session_state.last_upload_signature:
        with st.spinner("Extracting text, creating embeddings, and building indexes..."):
            (
                st.session_state.papers,
                st.session_state.ingestion_results,
            ) = process_uploaded_files(
                uploaded_files,
                st.session_state.papers,
            )

        st.session_state.last_upload_signature = current_upload_signature

        for result in st.session_state.ingestion_results:
            if result["status"] == "processed":
                st.success(f"✅ Indexed: {result['name']}")
            elif result["status"] == "already_loaded":
                st.info(f"ℹ️ Already loaded: {result['name']}")
            else:
                st.error(f"❌ Could not process {result['name']}: {result['error']}")

if st.session_state.papers:
    st.subheader("Loaded papers")
    for paper in st.session_state.papers.values():
        st.caption(f"• {paper['name']} — {paper['page_count']} pages, {paper['chunk_count']} chunks")

st.header("2. Select evidence sources")
paper_ids = list(st.session_state.papers)
selected_paper_ids = st.multiselect(
    "Search these papers",
    options=paper_ids,
    default=paper_ids,
    format_func=lambda paper_id: st.session_state.papers[paper_id]["name"],
    disabled=not paper_ids,
)

st.header("3. Ask or run a research action")
action = st.selectbox("Quick research action (optional)", ["Custom question"] + list(TASKS))
default_question = "" if action == "Custom question" else TASKS[action]["question"]
question = st.text_area("Question", value=default_question, placeholder="Example: What problem does this paper solve?")

if st.button("Ask", use_container_width=True, type="primary"):
    if not selected_paper_ids:
        st.warning("Select at least one paper.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        explicit_intent = None if action == "Custom question" else TASKS[action]["intent"]
        with st.spinner("Retrieving evidence and generating a cited answer..."):
            response = ask_question(
                papers=st.session_state.papers,
                selected_paper_ids=selected_paper_ids,
                question=question.strip(),
                history=st.session_state.chat_history,
                intent=explicit_intent,
                allow_general_knowledge=allow_general_knowledge,
            )
        st.session_state.chat_history.append({"question": question.strip(), **response})

st.divider()
st.header("🤖 Answers")

for turn in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

        if turn["sources"]:
            with st.expander("Retrieved evidence"):
                for source in turn["sources"]:
                    st.markdown(
                        f"**{source['paper']} — "
                        f"p. {source['page']} "
                        f"({source['section']})**"
                    )
                    st.caption(source["excerpt"] + "…")
