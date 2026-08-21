# 📚 AI Research Workbench

An intelligent Retrieval-Augmented Generation (RAG) application for querying and analyzing research papers using semantic search and LLM-powered reasoning.

Upload a research paper (or multiple), ask questions in natural language, and get answers grounded in the actual paper content — with automatic fallback to general knowledge when the paper doesn't have the answer.

---

## 🚀 Features

### ✅ Completed

- **Core RAG Pipeline** — PDF upload → chunking → embedding → FAISS vector indexing → semantic retrieval → LLM-generated answer, grounded strictly in retrieved content.
- **Hybrid Mode** — If the uploaded paper doesn't contain the answer, the system automatically falls back to Gemini's general knowledge instead of returning a dead end. Fallback answers are clearly labeled in the UI so users always know the source.
- **Conversation Memory + Query Rewriting** — Tracks recent conversation turns and resolves follow-up questions that use pronouns ("it", "that", "this method") by rewriting them into standalone queries before retrieval — improving retrieval accuracy on multi-turn conversations.
- **Structural Question Handling** — Questions like "what is the problem statement" or "what is the research gap" often don't phrase-match well against chunked academic text. The system detects these and force-includes the paper's opening sections (Abstract/Introduction) alongside standard retrieval results.
- **Session-State Caching** — The vector store and retriever are built once per session and cached, avoiding redundant re-embedding on every interaction.
- **Multi-PDF Upload (Basic)** — Multiple papers can be uploaded and accumulated into a shared vector index without needing to restart the session.
- **Quick Research Actions** — Pre-built structured queries (Limitations, Future Work, Methodology, Dataset, Research Gap) with specialized formatting rules for each category.
- **Conversational UI** — Full chat-style interface showing the complete question/answer history for the session, not just the latest response.

### 🚧 In Progress

- **Cross-Paper Disambiguation** — When multiple papers are loaded, ambiguous questions currently resolve to whichever paper's content scores highest in similarity search, without explicitly notifying the user or searching across all loaded papers. Improving this is a prerequisite for reliable comparison features.
- **Research Comparison** — Explicit side-by-side comparison of methodology, results, or findings across multiple uploaded papers.
- **Literature Review Generator** — Auto-generated structured literature review summaries spanning multiple uploaded papers.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Embeddings | HuggingFace Sentence Transformers |
| LLM | Google Gemini API (`gemini-2.5-flash`) |
| PDF Parsing | PyPDF2 / LangChain document loaders |

---

## 🏗️ Architecture

```
Upload PDF(s)
      ↓
PDF parsed → split into chunks → embedded → indexed in FAISS
      ↓
User asks a question
      ↓
(If follow-up) Query rewritten into a standalone question using conversation history
      ↓
Standalone query → semantic search against FAISS → relevant chunks retrieved
      ↓
(If structural question) Paper's opening chunks force-included alongside retrieved results
      ↓
Chunks + question + history → Gemini (paper-grounded prompt)
      ↓
If answer found in paper → returned to user
If not found → same question re-sent to Gemini with no paper context,
answered from general knowledge, clearly labeled as such
```

---

## 📂 Project Structure

```
AI Research Workbench/
├── app.py                     # Streamlit UI and app entry point
├── config.py                  # Environment/config loading
├── services/
│   ├── rag_pipeline.py        # Multi-file processing orchestration
│   ├── query_engine.py        # Core RAG logic: retrieval, hybrid mode, memory
│   ├── llm_service.py         # Gemini LLM initialization
│   ├── prompt_template.py     # All prompt templates (paper, fallback, rewrite)
│   └── research_tasks.py      # Quick Action task definitions
├── utils/
│   ├── file_handler.py        # File save handling
│   ├── pdf_loader.py          # PDF parsing + metadata tagging
│   ├── text_splitter.py       # Chunking logic
│   ├── embedding_model.py     # Embedding model loading
│   ├── vector_store.py        # FAISS index creation/updates
│   └── retriever.py           # Retriever configuration
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/sarthak15jain02-hub/ai_research_workbench.git
cd ai_research_workbench
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 📝 Usage

1. Upload one or more research papers (PDF format).
2. Use **Quick Research Actions** for structured extraction (Limitations, Future Work, Methodology, etc.), or
3. Ask any question directly in natural language via **Ask Questions**.
4. Follow-up questions referencing prior turns ("what preprocessing was done on it?") are automatically resolved using conversation context.
5. If the paper doesn't contain the answer, the system clearly labels responses drawn from general knowledge instead.

---

## 📌 Notes

- Running on Google Gemini's API free tier, which has daily request limits. For heavier usage, enable Cloud Billing on your Google AI Studio project.
- This project prioritizes retrieval-based grounding (RAG) over full-document context stuffing, to remain scalable as multi-document support expands.

---

## 📄 License

This project is intended for educational and portfolio purposes.
