import streamlit as st

from services.query_engine import ask_question
from services.rag_pipeline import process_uploaded_files
from services.research_tasks import TASKS
from utils.file_handler import file_hash


st.set_page_config(
    page_title="AI Research Workbench",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

DEFAULT_SESSION_STATE = {
    "papers": {},
    "chat_history": [],
    "ingestion_results": [],
    "last_upload_signature": (),
    "removed_paper_ids": set(),
    "selected_paper_ids": [],
    "selection_initialized": False,
    "selected_action": "Custom question",
    "question_draft": "",
    "current_page": "home",
    "allow_general_knowledge": False,
    "uploader_version": 0,
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --background: #07111f;
            --surface: #0d1a2a;
            --surface-light: #13263d;
            --border: #263f5b;
            --text: #f3f7ff;
            --muted: #9bacc1;
            --primary: #7161ef;
            --primary-hover: #8578ff;
            --accent: #26c8a7;
            --warning: #f1bb4b;
            --danger: #f35d75;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 8% 0%,
                    rgba(113, 97, 239, 0.16),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 92% 4%,
                    rgba(38, 200, 167, 0.10),
                    transparent 24%
                ),
                var(--background);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #091522;
            border-right: 1px solid rgba(155, 172, 193, 0.14);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
        }

        h1, h2, h3 {
            color: var(--text) !important;
            letter-spacing: -0.025em;
        }

        h2 {
            font-size: 1.45rem !important;
            margin-top: 1.15rem;
        }

        h3 {
            font-size: 1.05rem !important;
        }

        p, label, .stCaption {
            color: var(--muted);
        }

        .hero {
            padding: 1.65rem 1.8rem;
            margin-bottom: 1.3rem;
            border: 1px solid rgba(155, 172, 193, 0.17);
            border-radius: 18px;
            background:
                linear-gradient(
                    120deg,
                    rgba(113, 97, 239, 0.18),
                    rgba(13, 26, 42, 0.93) 48%,
                    rgba(38, 200, 167, 0.08)
                );
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.18);
        }

        .hero-label {
            color: #b9b1ff;
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: clamp(1.85rem, 3.3vw, 2.6rem);
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -0.048em;
            margin: 0;
        }

        .hero-subtitle {
            max-width: 690px;
            color: #b7c7d8;
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0.7rem 0 0;
        }

        .section-label {
            color: #a99fff;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin-bottom: -0.5rem;
        }

        .home-step {
            min-height: 164px;
            padding: 1.1rem;
            border: 1px solid rgba(155, 172, 193, 0.16);
            border-radius: 15px;
            background: rgba(13, 26, 42, 0.74);
            transition: transform 160ms ease, border-color 160ms ease;
        }

        .home-step:hover {
            transform: translateY(-3px);
            border-color: rgba(113, 97, 239, 0.56);
        }

        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 9px;
            background: rgba(113, 97, 239, 0.18);
            color: #c9c3ff;
            font-size: 0.75rem;
            font-weight: 800;
        }

        .step-title {
            color: #f2f6ff;
            font-size: 1rem;
            font-weight: 750;
            margin-top: 0.75rem;
        }

        .step-text {
            color: #9fb0c2;
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.35rem;
        }

        div[data-testid="stMetric"] {
            padding: 0.78rem 0.9rem;
            border: 1px solid rgba(155, 172, 193, 0.15);
            border-radius: 13px;
            background: rgba(13, 26, 42, 0.72);
        }

        div[data-testid="stMetricLabel"] {
            color: #9aacbf;
            font-size: 0.78rem;
        }

        div[data-testid="stMetricValue"] {
            color: #f3f7ff;
            font-size: 1.45rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 15px;
            border-color: rgba(155, 172, 193, 0.17);
            background: rgba(13, 26, 42, 0.62);
        }

        .paper-title {
            color: #f3f7ff;
            font-weight: 700;
            font-size: 0.93rem;
            overflow-wrap: anywhere;
        }

        .paper-meta {
            color: #9bacc1;
            font-size: 0.80rem;
            margin-top: 0.28rem;
        }

        .paper-stat {
            color: #b7c7d8;
            font-size: 0.84rem;
            line-height: 1.7;
        }

        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.24rem 0.66rem;
            margin-bottom: 0.7rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        .status-evidence {
            color: #8cf1da;
            background: rgba(38, 200, 167, 0.14);
            border: 1px solid rgba(38, 200, 167, 0.27);
        }

        .status-general {
            color: #bcb5ff;
            background: rgba(113, 97, 239, 0.17);
            border: 1px solid rgba(113, 97, 239, 0.30);
        }

        .status-warning {
            color: #ffda88;
            background: rgba(241, 187, 75, 0.12);
            border: 1px solid rgba(241, 187, 75, 0.30);
        }

        .status-error {
            color: #ff9cab;
            background: rgba(243, 93, 117, 0.12);
            border: 1px solid rgba(243, 93, 117, 0.28);
        }

        .empty-state {
            padding: 1rem 1.1rem;
            border: 1px dashed rgba(155, 172, 193, 0.30);
            border-radius: 14px;
            color: #9fb0c2;
            background: rgba(13, 26, 42, 0.42);
            font-size: 0.88rem;
        }

        div.stButton > button {
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 700;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.20);
        }

        div.stButton > button[kind="primary"] {
            border: none;
            background: linear-gradient(135deg, #7161ef, #9365f3);
        }

        .start-workbench-card {
            min-height: 252px;
        }

        .start-workbench-eyebrow {
            display: inline-block;
            color: #8cf1da;
            background: rgba(38, 200, 167, 0.12);
            border: 1px solid rgba(38, 200, 167, 0.24);
            border-radius: 999px;
            padding: 0.25rem 0.62rem;
            margin-bottom: 0.7rem;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .st-key-start-workbench button {
            min-height: 3.7rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(113, 97, 239, 0.25);
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #8374ff, #a376ff);
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(155, 172, 193, 0.37);
            border-radius: 14px;
            padding: 0.3rem;
            background: rgba(13, 26, 42, 0.48);
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(155, 172, 193, 0.15);
            border-radius: 14px;
            padding: 0.35rem 0.65rem;
            margin-bottom: 0.75rem;
            background: rgba(13, 26, 42, 0.72);
            animation: answerAppear 220ms ease-out;
        }

        div[data-testid="stExpander"] {
            border-radius: 12px;
            border-color: rgba(155, 172, 193, 0.20);
        }

        hr {
            border-color: rgba(155, 172, 193, 0.14);
        }

        @keyframes answerAppear {
            from {
                opacity: 0;
                transform: translateY(7px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def clear_workspace():
    """Clear current session papers, answers, settings, and uploader."""
    st.session_state.papers = {}
    st.session_state.chat_history = []
    st.session_state.ingestion_results = []
    st.session_state.last_upload_signature = ()
    st.session_state.removed_paper_ids = set()
    st.session_state.selected_paper_ids = []
    st.session_state.selection_initialized = False
    st.session_state.question_draft = ""

    # Changing this key clears Streamlit's file uploader.
    st.session_state.uploader_version += 1


def open_workbench():
    st.session_state.current_page = "workbench"


def open_home():
    st.session_state.current_page = "home"


def apply_quick_action():
    """Fill question area when the user selects a quick action."""
    selected_action = st.session_state.selected_action

    if selected_action != "Custom question":
        st.session_state.question_draft = TASKS[
            selected_action
        ]["question"]


def show_answer_status(answer_type):
    """Display the source/trust status above each response."""
    status_map = {
        "paper_evidence": (
            "status-evidence",
            "✓ Paper evidence",
        ),
        "general_knowledge": (
            "status-general",
            "ℹ General knowledge",
        ),
        "insufficient_evidence": (
            "status-warning",
            "⚠ Insufficient paper evidence",
        ),
        "input_error": (
            "status-error",
            "✕ Input error",
        ),
    }

    css_class, text = status_map.get(
        answer_type,
        status_map["paper_evidence"],
    )

    st.markdown(
        f'<span class="status-pill {css_class}">{text}</span>',
        unsafe_allow_html=True,
    )


def synchronize_uploaded_files(uploaded_files):
    """
    Process new uploads and remove papers when the user clicks ×
    in Streamlit's uploader.
    """
    uploaded_by_id = {
        file_hash(uploaded_file): uploaded_file
        for uploaded_file in uploaded_files
    }

    current_upload_ids = set(uploaded_by_id)
    previous_upload_ids = set(
        st.session_state.last_upload_signature
    )

    removed_from_uploader = (
        previous_upload_ids - current_upload_ids
    )

    for paper_id in removed_from_uploader:
        st.session_state.papers.pop(paper_id, None)
        st.session_state.removed_paper_ids.discard(paper_id)

    new_uploaded_files = [
        uploaded_file
        for paper_id, uploaded_file in uploaded_by_id.items()
        if (
            paper_id not in st.session_state.papers
            and paper_id not in st.session_state.removed_paper_ids
        )
    ]

    if new_uploaded_files:
        with st.spinner(
            "Extracting text, creating embeddings, and building indexes..."
        ):
            (
                st.session_state.papers,
                st.session_state.ingestion_results,
            ) = process_uploaded_files(
                new_uploaded_files,
                st.session_state.papers,
            )

        # Automatically include newly uploaded papers in selection.
        st.session_state.selection_initialized = False

        for result in st.session_state.ingestion_results:
            if result["status"] == "processed":
                st.success(
                    f"Indexed successfully: {result['name']}"
                )

            elif result["status"] == "already_loaded":
                st.info(
                    f"Already loaded: {result['name']}"
                )

            else:
                st.error(
                    f"Could not process {result['name']}: "
                    f"{result['error']}"
                )

    elif removed_from_uploader:
        st.session_state.ingestion_results = []
        st.info(
            "Deselected PDF(s) were removed from the workspace."
        )

    st.session_state.last_upload_signature = tuple(
        sorted(current_upload_ids)
    )


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## AI Research Workbench")
    st.caption("Evidence-first PDF research assistant")

    st.divider()

    if st.session_state.current_page == "workbench":
        if st.button(
            "← Back to overview",
            use_container_width=True,
        ):
            open_home()
            st.rerun()

        st.divider()

        st.markdown("#### Current workspace")

        st.metric(
            "Loaded papers",
            len(st.session_state.papers),
        )

        st.metric(
            "Searchable chunks",
            sum(
                paper["chunk_count"]
                for paper in st.session_state.papers.values()
            ),
        )

        fallback_status = (
            "Enabled"
            if st.session_state.allow_general_knowledge
            else "Disabled"
        )

        st.caption(
            f"General-knowledge fallback: **{fallback_status}**"
        )

        if st.button(
            "Manage workspace settings",
            use_container_width=True,
            type="secondary",
        ):
            open_home()
            st.rerun()

    else:
        st.caption(
            "Use the overview page to configure the workspace "
            "before opening the research workbench."
        )


# ---------------------------------------------------------
# Home / overview page
# ---------------------------------------------------------

if st.session_state.current_page == "home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Research, simplified</div>
            <h1 class="hero-title">AI Research Workbench</h1>
            <p class="hero-subtitle">
                Turn research PDFs into searchable evidence. Upload papers,
                ask focused questions, and inspect the exact passages used
                in every answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-label">How it works</p>',
        unsafe_allow_html=True,
    )
    st.header("Three steps to research faster")

    step_one, step_two, step_three = st.columns(3)

    with step_one:
        st.markdown(
            """
            <div class="home-step">
                <div class="step-number">01</div>
                <div class="step-title">Upload papers</div>
                <div class="step-text">
                    Add one or more searchable research PDFs.
                    The workbench automatically extracts text,
                    creates embeddings, and builds the search index.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_two:
        st.markdown(
            """
            <div class="home-step">
                <div class="step-number">02</div>
                <div class="step-title">Select evidence</div>
                <div class="step-text">
                    Choose exactly which papers should be used
                    for each question. This prevents unrelated
                    papers from affecting an answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_three:
        st.markdown(
            """
            <div class="home-step">
                <div class="step-number">03</div>
                <div class="step-title">Ask and verify</div>
                <div class="step-text">
                    Ask a custom question or use Quick Analysis.
                    Review page-level retrieved evidence under
                    every generated answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    settings_column, start_column = st.columns([1, 1.15])

    with settings_column:
        with st.container(border=True):
            st.markdown("### Workspace settings")

            st.toggle(
                "Allow general-knowledge fallback",
                key="allow_general_knowledge",
                help=(
                    "When disabled, answers stay strictly grounded "
                    "in selected research papers. When enabled, Gemini "
                    "can answer from general knowledge if paper evidence "
                    "is insufficient."
                ),
            )

            st.caption(
                "Recommended for academic work: keep this disabled "
                "so answers remain paper-grounded."
            )

            st.divider()

            if st.button(
                "Clear workspace",
                use_container_width=True,
                type="secondary",
            ):
                clear_workspace()
                st.toast(
                    "Workspace cleared successfully.",
                    icon="🗑️",
                )
                st.rerun()

    with start_column:
        with st.container(border=True, key="start-workbench-card"):
            st.markdown(
                '<div class="start-workbench-eyebrow">Recommended first step</div>',
                unsafe_allow_html=True,
            )
            st.markdown("### Start researching")

            current_paper_count = len(st.session_state.papers)
            current_answer_count = len(
                st.session_state.chat_history
            )

            st.caption(
                f"Current session: {current_paper_count} paper(s), "
                f"{current_answer_count} answer(s)."
            )

            st.write("")

            with st.container(key="start-workbench"):
                st.button(
                    "Open research workbench  →",
                    use_container_width=True,
                    type="primary",
                    on_click=open_workbench,
                )

            st.caption(
                "Open the workspace to upload PDFs, select evidence "
                "sources, and ask questions."
            )


# ---------------------------------------------------------
# Research workbench page
# ---------------------------------------------------------

else:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Research workspace</div>
            <h1 class="hero-title">Ask better questions. Verify every answer.</h1>
            <p class="hero-subtitle">
                Upload papers, select evidence sources, and generate
                cited answers grounded in the content you provide.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_pages = sum(
        paper["page_count"]
        for paper in st.session_state.papers.values()
    )

    total_chunks = sum(
        paper["chunk_count"]
        for paper in st.session_state.papers.values()
    )

    metric_one, metric_two, metric_three = st.columns(3)

    metric_one.metric(
        "Loaded papers",
        len(st.session_state.papers),
    )

    metric_two.metric(
        "Total pages",
        total_pages,
    )

    metric_three.metric(
        "Searchable chunks",
        total_chunks,
    )

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    st.markdown(
        '<p class="section-label">Step 01</p>',
        unsafe_allow_html=True,
    )
    st.header("Upload research papers")

    st.caption(
        "Upload searchable PDFs. New files are processed automatically."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_version}",
        help=(
            "Image-only or scanned PDFs require OCR before they "
            "can be searched."
        ),
    )

    synchronize_uploaded_files(uploaded_files)

    # -----------------------------------------------------
    # Loaded papers
    # -----------------------------------------------------

    if st.session_state.papers:
        st.markdown(
            '<p class="section-label">Workspace</p>',
            unsafe_allow_html=True,
        )
        st.header("Loaded papers")

        st.caption(
            "Use × in the uploader to remove a PDF completely. "
            "Remove from workspace stops future retrieval while "
            "keeping the upload visible above."
        )

        for paper_id, paper in list(
            st.session_state.papers.items()
        ):
            with st.container(border=True):
                paper_name_column, paper_info_column, paper_button_column = (
                    st.columns([6, 2, 2])
                )

                with paper_name_column:
                    st.markdown(
                        f'<div class="paper-title">📄 {paper["name"]}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        """
                        <div class="paper-meta">
                            Ready for semantic and keyword retrieval
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with paper_info_column:
                    st.markdown(
                        f"""
                        <div class="paper-stat">
                            <strong>{paper["page_count"]}</strong> pages<br>
                            <strong>{paper["chunk_count"]}</strong> chunks
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with paper_button_column:
                    if st.button(
                        "Remove from workspace",
                        key=f"remove_paper_{paper_id}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        removed_paper_name = paper["name"]

                        del st.session_state.papers[paper_id]

                        # Do not automatically re-add this paper while
                        # it remains selected in the uploader.
                        st.session_state.removed_paper_ids.add(paper_id)

                        st.session_state.selected_paper_ids = [
                            selected_id
                            for selected_id in (
                                st.session_state.selected_paper_ids
                            )
                            if selected_id != paper_id
                        ]

                        st.toast(
                            f"Removed {removed_paper_name} from workspace.",
                            icon="🗑️",
                        )
                        st.rerun()

    else:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No papers loaded yet.</strong><br>
                Upload one or more searchable PDF files above to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Select evidence
    # -----------------------------------------------------

    st.markdown(
        '<p class="section-label">Step 02</p>',
        unsafe_allow_html=True,
    )
    st.header("Choose evidence sources")

    paper_ids = list(st.session_state.papers)

    # Remove IDs that are no longer in the workspace.
    st.session_state.selected_paper_ids = [
        paper_id
        for paper_id in st.session_state.selected_paper_ids
        if paper_id in paper_ids
    ]

    # Select all papers after first upload or after new uploads.
    if (
        paper_ids
        and not st.session_state.selection_initialized
    ):
        st.session_state.selected_paper_ids = paper_ids.copy()
        st.session_state.selection_initialized = True

    selected_paper_ids = st.multiselect(
        "Search these papers",
        options=paper_ids,
        key="selected_paper_ids",
        format_func=lambda paper_id: st.session_state.papers[
            paper_id
        ]["name"],
        disabled=not paper_ids,
        placeholder="Upload a paper to begin selecting evidence sources.",
        help=(
            "Only selected PDFs are searched. Select one paper for "
            "focused answers or multiple papers for broader evidence."
        ),
    )

    # -----------------------------------------------------
    # Ask question
    # -----------------------------------------------------

    st.markdown(
        '<p class="section-label">Step 03</p>',
        unsafe_allow_html=True,
    )
    st.header("Ask a research question")

    question_column, action_column = st.columns([2, 1])

    with action_column:
        st.markdown("#### Quick analysis")

        st.selectbox(
            "Choose an action",
            ["Custom question"] + list(TASKS),
            key="selected_action",
            on_change=apply_quick_action,
            help=(
                "Select an action to insert a structured research question."
            ),
        )

        if st.session_state.selected_action != "Custom question":
            selected_task = TASKS[
                st.session_state.selected_action
            ]

            st.caption(
                "Retrieval intent: "
                f"{selected_task['intent'].replace('_', ' ').title()}"
            )

        st.caption(
            "Quick actions guide retrieval toward relevant paper sections."
        )

    with question_column:
        st.markdown("#### Your question")

        st.text_area(
            "Ask anything about the selected paper(s)",
            key="question_draft",
            height=130,
            placeholder=(
                "Example: What problem does this paper solve, "
                "and how does its approach address it?"
            ),
        )

        ask_clicked = st.button(
            "Generate grounded answer",
            use_container_width=True,
            type="primary",
            disabled=not paper_ids,
        )

    if ask_clicked:
        question = st.session_state.question_draft.strip()

        if not selected_paper_ids:
            st.warning(
                "Select at least one evidence source first."
            )

        elif not question:
            st.warning(
                "Enter a research question first."
            )

        else:
            explicit_intent = None

            if (
                st.session_state.selected_action
                != "Custom question"
            ):
                explicit_intent = TASKS[
                    st.session_state.selected_action
                ]["intent"]

            with st.spinner(
                "Retrieving evidence and generating a cited answer..."
            ):
                response = ask_question(
                    papers=st.session_state.papers,
                    selected_paper_ids=selected_paper_ids,
                    question=question,
                    history=st.session_state.chat_history,
                    intent=explicit_intent,
                    allow_general_knowledge=(
                        st.session_state.allow_general_knowledge
                    ),
                )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    **response,
                }
            )

    # -----------------------------------------------------
    # Answers
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<p class="section-label">Research conversation</p>',
        unsafe_allow_html=True,
    )
    st.header("🤖 Answers")

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Your evidence-grounded answers will appear here.</strong><br>
                Upload a paper, select it as an evidence source, and ask a question.
            </div>
            """,
            unsafe_allow_html=True,
        )

    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["question"])

        with st.chat_message("assistant"):
            show_answer_status(
                turn.get("answer_type", "paper_evidence")
            )

            st.markdown(turn["answer"])

            if turn.get("sources"):
                with st.expander(
                    "Retrieved evidence",
                    expanded=False,
                ):
                    for source in turn["sources"]:
                        st.markdown(
                            f"**{source['paper']} — "
                            f"p. {source['page']} "
                            f"({source['section']})**"
                        )

                        st.caption(
                            source["excerpt"] + "…"
                        )