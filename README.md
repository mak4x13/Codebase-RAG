# Codebase RAG System

A Python-based Retrieval-Augmented Generation (RAG) application for chatting with GitHub repositories and profiles.

It indexes code files, builds embeddings, stores them in FAISS, and uses an LLM to answer questions with repository-grounded context.

## What Problem It Solves

Developers often spend a lot of time manually exploring unfamiliar repositories to:

- understand project structure
- locate entry points and important files
- explain functions/classes
- create summaries/documentation

This project solves that by letting users ask natural-language questions about a codebase and receive code-grounded answers through a chat interface.

## What It Does

- Accepts a GitHub repository URL or GitHub profile URL
- Clones and indexes one or multiple repositories
- Scans supported code files and chunks them for retrieval
- Builds repo map and symbol map metadata for better navigation
- Generates embeddings using `sentence-transformers`
- Stores vectors in `FAISS`
- Retrieves relevant code snippets for each question
- Uses a Groq-hosted LLM for answer generation
- Supports repo summarization, file-focused questions, and entry-point questions

## Tech Stack

- Python
- Gradio
- SentenceTransformers
- FAISS (CPU)
- Groq API (LLM inference)
- GitPython

## Project Workflow (How It Works)

### 1) Indexing Pipeline

1. User enters a GitHub repo/profile URL in the UI.
2. The app resolves the URL using the GitHub API.
3. Repository/repositories are cloned locally into `data/repos`.
4. Code files are scanned and filtered by supported extensions.
5. Files are chunked with overlap (and basic boundary-aware chunking).
6. Repo metadata is created:
   - repo map (file list)
   - symbol map (top symbols per file)
7. Chunks are embedded using a SentenceTransformer model.
8. Embeddings and metadata are stored in a FAISS index (`data/faiss_index`).

### 2) Question Answering Pipeline

1. User asks a question in the Gradio chat UI.
2. The app detects query intent (general question, file question, symbol/function, entry point, summary).
3. Relevant chunks are retrieved from FAISS (or directly from file/symbol-matched metadata).
4. The app builds a token-budgeted prompt to avoid oversized requests.
5. The LLM generates an answer using only the provided code context.
6. The response is shown in the chat UI.

## Clone and Run Locally

### Prerequisites

- Python 3.10+ (recommended)
- Git installed
- Internet access (for GitHub cloning and LLM API calls)

### 1) Clone the Repository

```bash
git clone https://github.com/mak4x13/Codebase-RAG.git
cd Codebase-RAG
```

### 2) Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure Environment Variables

Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token_optional
```

Notes:

- `GROQ_API_KEY` is required for LLM responses.
- `GITHUB_TOKEN` is optional but recommended to avoid GitHub API rate limits.

### 5) Run the Application

```bash
python main.py
```

The app will:

- clear previous temporary repo/index data
- preload the embedding model
- launch the Gradio UI

Then open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

## Example Questions

- `Summarize the repo`
- `Summarize each repo`
- `Explain main.py`
- `Where is the entry point?`
- `Explain function chunk_file`
- `Prepare detailed documentation`

## Folder Overview

- `main.py` - app entry point, startup cleanup, and UI launch
- `app/ui/` - Gradio UI and query routing logic
- `app/preprocessing/` - file scanning, chunking, repo preprocessing
- `app/embeddings/` - embedding model loader
- `app/vectorstore/` - FAISS storage and metadata handling
- `app/retrieval/` - vector retrieval logic
- `app/llm/` - Groq LLM client
- `app/github/` - GitHub URL/profile resolver
- `data/` - temporary cloned repositories and FAISS indices

## Notes / Current Limitations

- Very large repositories may still require narrower questions (e.g., specify a file or function).
- Symbol extraction is regex-based (lightweight) and not a full AST parser.
- Index data is temporary and reset on app startup/shutdown in the current workflow.
