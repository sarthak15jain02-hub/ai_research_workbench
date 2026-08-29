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
    "removed_paper_ids": set(),
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

uploaded_by_id = {
    file_hash(uploaded_file): uploaded_file
    for uploaded_file in uploaded_files
}

current_upload_ids = set(uploaded_by_id)
previous_upload_ids = set(st.session_state.last_upload_signature)

# If the user clicks × in the uploader, that file is no longer selected.
# Remove it from the RAG workspace as well.
paper_ids_removed_from_uploader = previous_upload_ids - current_upload_ids

for paper_id in paper_ids_removed_from_uploader:
    st.session_state.papers.pop(paper_id, None)

    # Allow the paper to be indexed again if the user uploads it later.
    st.session_state.removed_paper_ids.discard(paper_id)

# Find PDFs that were newly added to the uploader and are not manually removed.
new_uploaded_files = [
    uploaded_file
    for paper_id, uploaded_file in uploaded_by_id.items()
    if (
        paper_id not in st.session_state.papers
        and paper_id not in st.session_state.removed_paper_ids
    )
]

if new_uploaded_files:
    with st.spinner("Extracting text, creating embeddings, and building indexes..."):
        (
            st.session_state.papers,
            st.session_state.ingestion_results,
        ) = process_uploaded_files(
            new_uploaded_files,
            st.session_state.papers,
        )

    for result in st.session_state.ingestion_results:
        if result["status"] == "processed":
            st.success(f"✅ Indexed: {result['name']}")
        elif result["status"] == "already_loaded":
            st.info(f"ℹ️ Already loaded: {result['name']}")
        else:
            st.error(f"❌ Could not process {result['name']}: {result['error']}")

elif paper_ids_removed_from_uploader:
    st.session_state.ingestion_results = []
    st.info("Removed deselected PDF(s) from the current workspace.")

st.session_state.last_upload_signature = tuple(sorted(current_upload_ids))

if st.session_state.papers:
    st.subheader("Loaded papers")
    st.caption(
        "These papers are available for retrieval in the current session. "
        "Removing a paper stops future questions from searching it."
    )

    for paper_id, paper in list(st.session_state.papers.items()):
        name_column, details_column, remove_column = st.columns([6, 2, 1])

        with name_column:
            st.markdown(f"📄 **{paper['name']}**")

        with details_column:
            st.caption(
                f"{paper['page_count']} pages\n\n"
                f"{paper['chunk_count']} chunks"
            )

        with remove_column:
            if st.button(
                "Remove",
                key=f"remove_paper_{paper_id}",
                use_container_width=True,
                type="secondary",
            ):
                removed_paper_name = paper["name"]

                # Removes this paper's chunks and FAISS vector store
                # from the current Streamlit session.
                del st.session_state.papers[paper_id]

                # Prevents this paper from returning automatically while it
                # remains selected inside the uploader.
                st.session_state.removed_paper_ids.add(paper_id)

                # Clear only the latest upload status messages.
                # Existing chat answers are intentionally preserved.
                st.session_state.ingestion_results = []

                st.toast(
                    f"Removed {removed_paper_name} from this workspace.",
                    icon="🗑️",
                )
                st.rerun()

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
        answer_type = turn.get("answer_type", "paper_evidence")

        if answer_type == "paper_evidence":
            st.success("✅ Paper Evidence")

        elif answer_type == "general_knowledge":
            st.info("ℹ️ General Knowledge")

        elif answer_type == "insufficient_evidence":
            st.warning("⚠️ Insufficient Paper Evidence")

        elif answer_type == "input_error":
            st.error("❌ Input Error")

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
