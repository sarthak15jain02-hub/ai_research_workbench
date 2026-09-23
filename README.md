# 📚 AI Research Workbench

AI Research Workbench is a Streamlit-based Retrieval-Augmented Generation (RAG) application for reading and querying one or more research-paper PDFs. Users upload PDFs, the application converts their content into searchable vectors, retrieves relevant evidence for a question, and asks Google Gemini to produce a grounded answer with source details.

The project is designed for research questions whose exact wording may not appear in the PDF. For example, a paper may not contain the literal phrase **“problem statement”**, but it commonly explains the problem in the Abstract, Introduction, Motivation, Background, or Related Work sections. The application detects these question types and adjusts retrieval to find the relevant evidence.

## Interview Demo

AI Research Workbench is a grounded AI research assistant. Users upload research papers, ask questions about them, and receive answers supported by paper and page citations instead of untraceable chatbot responses.

### Demo workflow

1. Upload one or more searchable research-paper PDFs.
2. Select the paper or papers to search.
3. Ask a question such as:
   - What problem does this paper address?
   - What methodology does the paper use?
   - What are the main experimental results?
   - What limitations do the authors identify?
4. Review the answer and expand the retrieved evidence to verify the sources.

### Technical highlights

- Retrieval-Augmented Generation using LangChain and Google Gemini.
- Hybrid retrieval using FAISS dense search and BM25 keyword search.
- Reciprocal Rank Fusion to combine semantic and lexical rankings.
- Per-paper indexing with SHA-256 duplicate detection.
- Intent-aware retrieval for methodology, datasets, results, limitations, and research gaps.
- Page- and section-level metadata for traceable citations.
- Explicit insufficient-evidence handling and optional general-knowledge fallback.

The hosted demo requires a Gemini API secret configured by the application owner. Interviewers only need a web browser; they do not need Python, dependencies, or an API key on their computer.

## What the application currently does

- Upload one or multiple text-based PDF papers.
- Uses a two-screen interface: a compact **Overview** page for service guidance and workspace settings, followed by a dedicated **Research Workbench** page for paper analysis.
- Provides an Overview → Workbench navigation flow, including a return-to-overview control from the workbench.
- Provides Clear Workspace on the Overview page; it clears session papers, chat history, selections, and the visible file uploader state.
- Automatically starts processing when the uploaded-file selection changes; there is no separate Process button.
- Extracts text page by page and creates chunks for retrieval.
- Builds a separate in-memory FAISS vector store for every uploaded paper.
- Lets the user choose which loaded paper(s) are searched.
- Prevents duplicate indexing by identifying PDF content with a SHA-256 hash, rather than only its filename.
- Uses hybrid retrieval:
  - dense semantic retrieval with Hugging Face embeddings and FAISS;
  - BM25 keyword retrieval for exact technical terms, acronyms, datasets, and metrics;
  - Reciprocal Rank Fusion (RRF) to combine the two rankings.
- Detects common research-question intents: Problem Statement, Research Gap, Methodology, Dataset, Results, Limitations, and Future Work.
- Boosts relevant sections for those intents, such as Abstract/Introduction for a problem-statement question.
- Uses Gemini (`gemini-2.5-flash` by default) to answer using retrieved evidence.
- Requires paper-grounded answers to include paper/page citations where available.
- Shows the retrieved evidence below every answer.
- Shows the newest answer first in the bottom **🤖 Answers** section; earlier answers are below it.
- Labels every response as **Paper Evidence**, **General Knowledge**, **Insufficient Paper Evidence**, or **Input Error**.
- Supports two removal behaviours:
  - use the `×` in the uploader to remove a PDF completely from the uploader and the current RAG workspace;
  - use **Remove from workspace** below Loaded papers to stop retrieval from that paper while keeping it visible in the uploader.
- Keeps general-knowledge fallback disabled by default. Users can enable it from the sidebar, and such answers are explicitly labelled.
- Detects PDFs with no selectable text and tells the user that OCR is required.

## How it works

### 1. Upload and ingestion

When PDFs are selected in the uploader, the application:

1. Calculates a SHA-256 hash of each PDF's bytes. This allows the same content to be recognised even if the filename differs.
2. Saves the file inside `data/` using a safe name.
3. Extracts text one page at a time with `PyPDFLoader`.
4. Adds metadata to each page: `paper_id`, `paper_name`, `page_number`, and `source_file`.
5. Detects common academic headings such as Abstract, Introduction, Methods, Dataset, Experiments, Results, Conclusion, and References.
6. Splits the page text into overlapping chunks. Every chunk keeps its source paper, page number, detected section, and unique `chunk_id`.
7. Creates one FAISS vector store for that paper.

If one PDF fails to process, the remaining PDFs can still be indexed and queried.

### 2. Question understanding

The application first identifies whether the question belongs to a recognised research intent. For example:

| Question type | Example question | Sections favoured during retrieval |
|---|---|---|
| Problem Statement | What problem does this paper solve? | Abstract, Introduction, Background, Related Work, Conclusion |
| Research Gap | What gap in existing work does this paper address? | Abstract, Introduction, Background, Related Work |
| Methodology | Explain the proposed methodology. | Methods, Dataset, Experiments |
| Dataset | Which dataset was used and how was it processed? | Dataset, Methods, Experiments |
| Results | What are the major experimental results? | Experiments, Results, Conclusion |
| Limitations | What limitations do the authors identify? | Results, Conclusion, Discussion |
| Future Work | What do the authors propose for future work? | Conclusion, Results |

For a recognised intent, the search query is expanded with related academic terms. The user's original question is still sent to Gemini unchanged; only retrieval uses the expanded query.

For **Problem Statement** and **Research Gap** questions, the retriever also force-includes early chunks from the paper. This is important because the actual motivation and limitations of earlier work are usually in the Abstract and Introduction, even when the phrase “problem statement” never appears.

### 3. Hybrid retrieval

For every selected paper, two retrieval methods run independently:

- **Dense semantic retrieval:** FAISS compares the question embedding with chunk embeddings. This finds conceptually related text even when exact words differ.
- **BM25 keyword retrieval:** BM25 ranks chunks containing important words, acronyms, dataset names, metrics, and method names.

The two result lists are joined with Reciprocal Rank Fusion. Matching academic sections receive a small score boost. Reference chunks are down-ranked so citations alone do not become the main answer evidence.

For multiple selected papers, each paper is searched separately. The application limits the number of chunks taken from each paper so a longer or more similar paper does not take all evidence slots.

### 4. Answer generation

Retrieved chunks are formatted with evidence metadata, for example:

```text
[Paper: example_paper.pdf | Page: 3 | Section: introduction]
Retrieved passage from the PDF...
```

Gemini receives this evidence, the original question, and limited recent conversation history. It is instructed to:

- use the supplied evidence rather than inventing facts;
- cite paper claims as `[Paper name, p. N]`;
- distinguish evidence from different papers;
- return `INSUFFICIENT_EVIDENCE` when the evidence cannot answer the question.

If general-knowledge fallback is disabled, insufficient paper evidence produces a paper-only message. If it is enabled, the fallback is visibly labelled as general knowledge and is not presented as a paper claim.

## Project structure

```text
AI Research Workbench/
├── app.py                         # Streamlit UI, session state, uploads, answers
├── config.py                      # Environment-based configuration values
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── .env                           # Local Gemini API key; never commit this file
├── data/                          # Runtime PDF copies; ignored by Git
├── services/
│   ├── llm_service.py             # Gemini client creation
│   ├── prompt_template.py         # Grounded-answer and fallback prompts
│   ├── query_engine.py            # Intent detection, hybrid search, RRF, answer flow
│   ├── rag_pipeline.py            # Per-PDF ingestion process
│   └── research_tasks.py          # Quick Research Action definitions
└── utils/
    ├── embedding_model.py         # Cached Hugging Face embedding model
    ├── file_handler.py            # PDF hashing and safe local save
    ├── pdf_loader.py              # PDF extraction and page metadata
    ├── text_splitter.py           # Section detection and chunk creation
    └── vector_store.py            # FAISS vector-store construction
```

## File-by-file guide

| File | Main responsibility | Update it when you want to... |
|---|---|---|
| `app.py` | Website layout, widgets, session state, answer order/display | Improve UI, add tabs, remove-paper controls, progress bars, export buttons, or chat behaviour. |
| `config.py` | Model names, chunk settings, filesystem paths, API-key loading | Change models, chunk size/overlap, or environment configuration. |
| `services/rag_pipeline.py` | Process and isolate uploaded PDFs | Add ingestion stages, persistent indexing, OCR, file validation, or per-file progress. |
| `services/query_engine.py` | Question intent, retrieval, ranking, and Gemini execution | Improve retrieval quality, add a new intent, comparison mode, reranking, or evaluation logs. |
| `services/prompt_template.py` | Gemini instructions | Change answer format, citation style, summary format, or multi-paper comparison rules. |
| `services/research_tasks.py` | Quick Action text and intent | Add actions such as Contributions, Authors, Paper Objective, or Literature Review. |
| `utils/file_handler.py` | PDF hash and saved filename | Change local upload-file handling. |
| `utils/pdf_loader.py` | Text extraction and metadata | Add OCR support, metadata extraction, or improved PDF error handling. |
| `utils/text_splitter.py` | Headings, sections, chunk size/overlap | Improve section detection for new document formats. |
| `utils/embedding_model.py` | Embedding model loading/caching | Trade retrieval quality for speed. Re-upload/re-index after changing this model. |
| `utils/vector_store.py` | Build FAISS store | Add FAISS persistence or migrate to Chroma/Qdrant. |

## Technology stack

| Layer | Technology |
|---|---|
| User interface | Streamlit |
| LLM | Google Gemini through `langchain-google-genai` |
| Embeddings | Hugging Face Sentence Transformers (`BAAI/bge-base-en-v1.5` by default) |
| Vector search | FAISS |
| Keyword search | BM25 via `rank-bm25` |
| PDF text extraction | PyPDF / LangChain `PyPDFLoader` |
| RAG framework utilities | LangChain |

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- An internet connection on the first run to download the embedding model
- Text-based/searchable PDFs. Image-only/scanned PDFs require OCR first.

## Installation and setup

### Hosted demo

The application can be deployed directly from this repository using Streamlit Community Cloud:

1. Create a new Streamlit Cloud app from this repository.
2. Select the `main` branch and `app.py` as the main file.
3. Add the following secret in the app settings:

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
```

Optional secrets can also be configured:

```toml
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "150"
```

Never commit the real API key to GitHub. Use `.env.example` as the local configuration template.

### 1. Open the project folder

Run terminal commands from the folder containing `app.py` and `requirements.txt`.

```bash
cd "/Users/sarthakjain/project/AI Research Workbench"
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
venv/bin/pip install -r requirements.txt
```

On Windows after activating the virtual environment:

```powershell
pip install -r requirements.txt
```

### 4. Create the `.env` file

Create `.env` in the project root, beside `app.py`.

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Optional settings:

```env
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Never commit `.env` to GitHub. It contains your secret API key.

### 5. Run the app

```bash
venv/bin/streamlit run app.py
```

Open the URL printed in the terminal, usually [http://localhost:8501](http://localhost:8501).

If `localhost` does not work in Brave, use [http://127.0.0.1:8501](http://127.0.0.1:8501), disable Brave Shields for that local page, or use another browser. The application itself is browser-independent.

## Using the application

### Overview page

The Overview is the application's entry page. It contains:

- A short explanation of the service and its three-step workflow.
- The **Allow general-knowledge fallback** toggle.
- The **Clear workspace** button.
- Current session counts for papers and answers.
- The **Open research workbench** button.

### Research Workbench page

1. Open the Research Workbench from the Overview page.
2. Upload one or more research PDFs under **Upload research papers**.
3. Wait while the application automatically extracts text, creates embeddings, and builds indexes. Each successful PDF displays `Indexed successfully`.
4. Check **Loaded papers** to see the PDF name, page count, chunk count, and workspace-removal control.
5. Under **Choose evidence sources**, select the paper(s) you want included in the answer.
6. Under **Ask a research question**, choose a Quick Analysis action or select `Custom question` and type your own question.
7. Click **Generate grounded answer**.
8. Find the result in the bottom **🤖 Answers** section. The newest answer is always first.
9. Open **Retrieved evidence** below an answer to inspect paper name, page, section, and excerpt.
10. Use **← Back to overview** in the sidebar to return to the home page.

To remove a PDF completely, click its `×` in the uploader. This automatically removes it from Loaded papers, paper selection, and future retrieval. The **Remove from workspace** button below Loaded papers removes only the in-memory RAG version; the file remains visible in the uploader until its `×` is clicked. Clear Workspace resets the entire current session and clears the uploader too.

For the most trustworthy research-paper responses, leave **Allow general-knowledge fallback** switched off.

## Quick Research Actions

| Action | What it requests |
|---|---|
| Problem Statement | The problem, prior limitation, need, and how the paper addresses it. |
| Paper Summary | Problem, objective, method, dataset, metrics, results, contributions, limitations, and future work. |
| Research Gap | Limitations in earlier work and the gap the paper attempts to solve. |
| Methodology | Step-by-step workflow, model/algorithm, inputs, training, and outputs. |
| Dataset Analysis | Dataset name/source/size/classes/split/preprocessing/augmentation. |
| Experimental Results | Datasets, metrics, numbers, comparisons, ablations, and conclusions. |
| Limitations | Limitations explicitly supported by paper evidence. |
| Future Work | Future directions explicitly proposed by the authors. |

## Example questions

```text
What is the problem statement of this paper?
```

```text
What research gap does this paper address?
```

```text
Explain the methodology step by step.
```

```text
What dataset was used, and how was it preprocessed?
```

```text
What are the main experimental results and evaluation metrics?
```

```text
For each selected paper, explain the problem it addresses under separate headings.
```

## How to test the current implementation

### Single-paper test

1. Upload one searchable PDF.
2. Confirm it displays `✅ Indexed`.
3. Select only that paper.
4. Ask: `What is the problem statement of this paper?`
5. Check that the answer is supported by the retrieved evidence and cites the correct paper/page.

### Multi-paper test

1. Upload two different PDFs.
2. Select only Paper A and ask a question. Evidence should name only Paper A.
3. Select only Paper B and ask the same question. Evidence should name only Paper B.
4. Select both papers and ask: `For each selected paper, explain the problem it addresses under separate headings.`
5. Check that the answer clearly separates the papers and does not mix their facts or citations.

### Problem-statement retrieval test

1. Upload one paper and select only that paper.
2. Ask: `What is the problem statement of this paper?`
3. Open **Retrieved evidence**.
4. Confirm that early pages are included, usually an `abstract` or `introduction` chunk on page 1–3.
5. Confirm that the response describes the paper's problem, prior-work limitation, and proposed direction rather than only repeating a generic definition.

### Duplicate-upload test

1. Upload a paper that is already loaded.
2. The app should show `ℹ️ Already loaded` rather than creating duplicate embeddings.

### Insufficient-evidence test

1. Keep general-knowledge fallback off.
2. Ask a question that the selected paper cannot answer.
3. The response should state that there is insufficient paper evidence.
4. Enable fallback and repeat. The response should be explicitly labelled as general knowledge.

## Current limitations

- Vector stores live only in Streamlit session memory. Restarting Streamlit requires PDF upload and indexing again.
- `data/` stores uploaded PDF copies but the FAISS indexes are not persisted yet.
- PDFs must contain selectable text. OCR is not implemented for scanned/image-only PDFs.
- Section detection is based on common English research-paper headings. It may be less accurate for unusual layouts, multi-column extraction issues, or other languages.
- Multi-paper retrieval is working, but there is no dedicated side-by-side comparison table yet.
- Users should verify important claims against the displayed evidence and original PDF, especially for academic writing.
- The current embedding import emits a LangChain deprecation warning. It does not prevent indexing or answering; migrating to `langchain-huggingface` is a maintenance improvement for a later update.

## Recommended next improvements

1. **Show detailed ingestion progress:** Display extracting, splitting, embedding, and indexed stages for every PDF.
2. **Build comparison mode:** Generate a structured comparison table with a separate evidence column for each selected paper.
3. **Persist indexes:** Save/load per-paper FAISS indexes in `vectorstore/` so re-uploading is not needed after restart.
4. **Add OCR:** Support scanned PDFs using OCRmyPDF or another OCR service.
5. **Add reranking:** Rerank retrieved chunks before Gemini to improve answer precision for complex papers.
6. **Create evaluation tests:** Keep real questions with expected source pages to measure changes in chunking, models, prompts, and ranking.
7. **Add answer export:** Let users download a conversation or report as Markdown, PDF, or JSON.
8. **Modernise embeddings dependency:** Replace deprecated `langchain_community.embeddings.HuggingFaceEmbeddings` with `langchain_huggingface.HuggingFaceEmbeddings` after installing `langchain-huggingface`.

## Troubleshooting

### Browser opens a blank or “Running…” page

Wait 5–10 seconds, then refresh once. If it remains blank, stop Streamlit with `Control + C` and run:

```bash
venv/bin/streamlit run app.py
```

Open the exact URL printed in the terminal.

### Brave cannot load `localhost:8501`

Try the following:

1. Open `http://127.0.0.1:8501`.
2. Disable Brave Shields for the local page.
3. Hard-refresh with `Cmd + Shift + R`.
4. Disable local ad-block/privacy extensions for this address.
5. Use Chrome, Safari, or Firefox while developing if needed.

### `requirements.txt` not found

You are probably in the wrong folder. Check:

```bash
pwd
ls app.py requirements.txt
```

Then run the installation command from this project root.

### `GOOGLE_API_KEY is missing`

Ensure `.env` is beside `app.py`, contains `GOOGLE_API_KEY=...`, and restart Streamlit after saving it.

### First PDF processing is slow

This is expected on first use. The Hugging Face embedding model must download and load once. Later indexing in the same session is faster because the model is cached.

### Hugging Face unauthenticated-request warning

This warning does not stop the app. It only means model downloads are not authenticated and may have lower Hub rate limits. You do not need an HF token for normal use unless downloads become rate-limited.

### `HuggingFaceEmbeddings` deprecation warning

This warning does not stop embeddings or answers. It is a future maintenance task: install `langchain-huggingface` and update the import in `utils/embedding_model.py` when you are ready.

### “No selectable text was found” error

The PDF is likely scanned/image-only. Use OCR to create a searchable PDF, then upload it again.

### Answer does not look correct

1. Check the selected paper(s); unselect irrelevant papers.
2. Open **Retrieved evidence** and verify the excerpt supports the answer.
3. Ask a narrower question with a specific topic, method, dataset, or result.
4. For comparisons, explicitly say `separate headings for each selected paper`.
5. Keep general-knowledge fallback disabled when you need a strictly paper-grounded answer.

## Security notes

- Keep `.env` private.
- Never paste your Gemini API key into screenshots, commits, public chats, or GitHub repositories.
- `data/`, `vectorstore/`, virtual environments, and `.env` should remain in `.gitignore`.

## License

This project is intended for educational and portfolio purposes.
