import gradio as gr
from app.repo.clone_repo import clone_repo
from app.preprocessing.preprocess_repo import preprocess_repository
from app.embeddings.embedder import CodeEmbedder
from app.vectorstore.faiss_store import FaissStore
from app.retrieval.retriever import Retriever
from app.llm.groq_client import GroqLLM, AVAILABLE_MODELS
from app.utils.repo_id import generate_repo_id
from app.utils.session_state import SessionState

from app.github.github_resolver import resolve_github_url, GitHubResolverError


def _normalize_text(text: str) -> str:
    normalized = []
    for ch in text.lower():
        normalized.append(ch if ch.isalnum() else " ")
    return " ".join("".join(normalized).split())


def _register_repo_name(state, name: str, repo_id: str):
    key = _normalize_text(name)
    if not key:
        return
    state.repo_name_map.setdefault(key, [])
    if repo_id not in state.repo_name_map[key]:
        state.repo_name_map[key].append(repo_id)


def _select_repo_id(question: str, state):
    normalized = _normalize_text(question)
    candidates = []
    for name_key, repo_ids in state.repo_name_map.items():
        if name_key and name_key in normalized:
            candidates.extend(repo_ids)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, candidates
    return state.repo_id, None


def _build_repo_summary(repo_id: str, llm: GroqLLM, model_name: str):
    repo_chunks = Retriever(repo_id=repo_id, top_k=12).retrieve(
        "overview architecture main modules entry points data flow dependencies"
    )
    if not repo_chunks:
        return "No indexable code was found for this repository."

    context = "\n".join([c["content"] for c in repo_chunks])
    user_prompt = f"""
Summarize this repository based on the retrieved code context.
Include: purpose, key modules, data flow, entry points, and external dependencies.
Keep it concise and structured.

Code context:
{context}
"""
    return llm.generate(
        system_prompt="You are a senior software engineer summarizing a codebase.",
        user_prompt=user_prompt,
        model=model_name,
        temperature=0.2,
        top_p=0.9
    )


def _is_doc_request(question: str) -> bool:
    q = question.strip().lower()
    return any(
        phrase in q for phrase in (
            "prepare detailed documentation",
            "detailed documentation",
            "documentation",
            "document the repo",
            "document the repository",
            "explain the codebase in detail",
            "explain the repository in detail"
        )
    )


def _is_summary_request(question: str) -> bool:
    q = question.strip().lower()
    if _is_doc_request(question):
        return True
    return any(
        phrase in q for phrase in (
            "summary",
            "summarize",
            "overview",
            "explain",
            "describe",
            "documentation"
        )
    )


def _extract_filename(question: str):
    import re
    match = re.search(
        r"([A-Za-z0-9_\-./\\]+\.(py|js|ts|tsx|jsx|java|go|rs|cpp|c|cs|rb|kt|swift|scala|md|txt|json|yaml|yml|toml))",
        question,
        flags=re.IGNORECASE
    )
    if match:
        return match.group(1).replace("\\", "/")
    return None


def _guess_filename(question: str, repo_id: str):
    import re
    tokens = re.findall(r"[A-Za-z0-9_\-]+", question.lower())
    stop = {
        "file", "files", "code", "repo", "repository", "project", "app",
        "in", "the", "this", "that", "where", "which", "what", "does", "do",
        "about", "explain", "summary", "summarize", "overview"
    }
    tokens = [t for t in tokens if len(t) >= 4 and t not in stop]
    if not tokens:
        return None
    candidates = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        base = path.split("/")[-1].lower()
        for t in tokens:
            if t in base:
                candidates.append(base)
                break
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_symbol_name(question: str):
    import re
    match = re.search(r"`([A-Za-z_]\w*)`", question)
    if match:
        return match.group(1)
    for pattern in (
        r"\bfunction\s+([A-Za-z_]\w*)",
        r"\bclass\s+([A-Za-z_]\w*)",
        r"\bmethod\s+([A-Za-z_]\w*)",
        r"\bdef\s+([A-Za-z_]\w*)",
    ):
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _is_entry_point_request(question: str) -> bool:
    q = question.strip().lower()
    return any(
        phrase in q for phrase in (
            "entry point",
            "entrypoint",
            "where does it start",
            "where to start",
            "main file",
            "starting point"
        )
    )


def _find_file_chunks(repo_id: str, filename: str) -> list[dict]:
    candidates = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        if path.endswith(filename):
            candidates.append(c)
    return candidates[:8]


def _find_all_file_chunks(repo_id: str, filename: str) -> list[dict]:
    candidates = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        if path.endswith(filename):
            candidates.append(c)
    candidates.sort(key=lambda c: c.get("start_line", 0))
    return candidates


def _find_chunks_by_paths(repo_id: str, paths: list[str]) -> list[dict]:
    if not paths:
        return []
    candidates = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        if any(path.endswith(p) for p in paths):
            candidates.append(c)
    return candidates[:10]


def _find_entry_point_chunks(repo_id: str) -> list[dict]:
    entry_files = [
        "main.py", "app.py", "__main__.py", "index.js", "server.js", "app.js",
        "main.ts", "index.ts", "main.go", "cmd/main.go", "src/main.rs",
        "Program.cs"
    ]
    candidates = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        if any(path.endswith(p) for p in entry_files):
            candidates.append(c)
    return candidates[:8]


def _find_similar_files(repo_id: str, filename: str) -> list[str]:
    base = filename.split("/")[-1]
    matches = []
    for c in FaissStore.load_metadata(repo_id):
        path = c.get("file_path", "").replace("\\", "/")
        if base and base in path:
            matches.append(path)
    return matches[:5]


def _find_symbol_files(repo_id: str, symbol: str) -> list[str]:
    files = []
    for c in FaissStore.load_metadata(repo_id):
        if c.get("chunk_type") != "symbol_map":
            continue
        for line in c.get("content", "").splitlines():
            if not line.startswith("- "):
                continue
            if ":" not in line:
                continue
            path, symbols = line[2:].split(":", 1)
            symbol_list = [s.strip() for s in symbols.split(",")]
            if symbol in symbol_list:
                files.append(path.strip())
    return list(dict.fromkeys(files))[:3]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _max_context_tokens(model_name: str, doc_request: bool) -> int:
    model = (model_name or "").lower()
    if "8b" in model:
        return 2000 if doc_request else 1500
    return 3500 if doc_request else 2500


def _build_context_blocks(chunks: list[dict], max_tokens: int) -> list[str]:
    blocks = []
    used = 0
    for c in chunks:
        header = f"\nRepo: {c.get('repo_name', 'unknown')}\nFile: {c['file_path']}\nLines: {c['start_line']}-{c['end_line']}\nCode:\n"
        header_tokens = _estimate_tokens(header)
        if used + header_tokens >= max_tokens:
            break
        content = c["content"]
        content_tokens = _estimate_tokens(content)
        available_tokens = max_tokens - used - header_tokens
        if content_tokens > available_tokens:
            available_chars = max(0, available_tokens * 4 - 3)
            content = content[:available_chars] + "..."
        block = header + content + "\n"
        blocks.append(block)
        used += _estimate_tokens(block)
        if used >= max_tokens:
            break
    return blocks


def _build_file_summary(chunks: list[dict], llm: GroqLLM, model_name: str) -> str:
    if not chunks:
        return "No indexable code was found for this file."
    max_context_tokens = _max_context_tokens(model_name, doc_request=True)
    context_blocks = _build_context_blocks(chunks, max_context_tokens)
    user_prompt = f"""
Summarize this file at a high level including its purpose, main components, and flow of control/data.
Include: what it does, main components, and flow of control/data.
Keep it concise and structured. Avoid line-by-line explanation.

Code context:
{''.join(context_blocks)}
"""
    return llm.generate(
        system_prompt="You are a senior software engineer summarizing a single source file.",
        user_prompt=user_prompt,
        model=model_name,
        temperature=0.2,
        top_p=0.9
    )


def index_repository(repo_url, state):
    try:
        state.repo_id = None
        state.repo_url = None
        state.repo_ids = []
        state.repo_name_map = {}
        state.repo_summaries = {}
        state.repo_id_to_name = {}

        repos = resolve_github_url(repo_url)

        if len(repos) > 1:  # profile with multiple repos
            first_repo_url = repos[0]["clone_url"]
            # extract owner/username from clone_url
            profile_name = first_repo_url.split("/")[3]

            metadata_content = "Profile: {}\nRepositories:\n- {}".format(
                profile_name,
                "\n- ".join([r["name"] for r in repos])
            )

            # Convert to chunk format
            metadata_chunk = [{
                "file_path": f"{profile_name}_metadata",
                "start_line": 1,
                "end_line": 1,
                "content": metadata_content,
                "chunk_type": "metadata"
            }]

            # We'll append this chunk to the repo chunks later
        else:
            metadata_chunk = []

        embedder = CodeEmbedder()  # load once (FAST)
        state.profile_metadata_id = f"profile_metadata__{state.session_id}"
        if not metadata_chunk:
            metadata_chunk = [{
                "file_path": "profile_metadata",
                "start_line": 0,
                "end_line": 0,
                "content": "",
                "chunk_type": "metadata"
            }]

        metadata_embeddings = embedder.embed_chunks(metadata_chunk)
        FaissStore(state.profile_metadata_id).build(metadata_embeddings, metadata_chunk)

        for repo in repos:
            repo_id = generate_repo_id(repo["clone_url"])
            local_path = clone_repo(repo["clone_url"])

            chunks = preprocess_repository(
                repo_path=local_path,
                repo_id=repo_id,
                repo_url=repo["clone_url"],
                repo_name=repo["name"]
            )

            if not chunks:
                print(f"Skipping {repo['name']} - no indexable files found.")
                continue

            embeddings = embedder.embed_chunks(chunks)
            FaissStore(repo_id).build(embeddings, chunks)

            state.repo_id = repo_id  # last repo active
            state.repo_url = repo_url
            state.repo_ids.append(repo_id)
            state.repo_id_to_name[repo_id] = repo["name"]
            _register_repo_name(state, repo["name"], repo_id)
            _register_repo_name(state, repo["name"].split("/")[-1], repo_id)

        return f"Indexed {len(repos)} repository(ies) successfully.", True

    except GitHubResolverError as e:
        return f"{str(e)}", False

    except Exception as e:
        return f"Unexpected error: {str(e)}", False


def answer_question(question, model_name, temperature, top_p, chat_history, state):
    # Prevent NoneType errors
    if not question or not question.strip():
        return "", chat_history
    if chat_history is None:
        chat_history = []
    if not state.repo_id:
        chat_history.append((question, "Please index a repository first."))
        yield "", chat_history
        return

    TOP_K = 5
    target_repo_id, ambiguous = _select_repo_id(question, state)
    if ambiguous:
        names = [state.repo_id_to_name.get(rid, rid) for rid in ambiguous]
        prompt = "Multiple repositories match your question. Please specify one:\n- "
        prompt += "\n- ".join(names)
        chat_history.append((question, prompt))
        yield "", chat_history
        return

    chat_history.append((question, ""))
    yield "", chat_history

    doc_request = _is_doc_request(question)
    filename = _extract_filename(question)
    symbol_name = _extract_symbol_name(question)
    entry_request = _is_entry_point_request(question)
    doc_top_k = 12 if doc_request else TOP_K
    summary_request = _is_summary_request(question)

    question_lower = question.strip().lower()
    summarize_all = summary_request and (
        "each repo" in question_lower
        or "each repository" in question_lower
        or "all repos" in question_lower
        or "all repositories" in question_lower
    )

    if not filename:
        filename = _guess_filename(question, target_repo_id)

    if summarize_all and state.repo_ids:
        summaries = []
        for repo_id in state.repo_ids:
            if repo_id not in state.repo_summaries:
                llm = GroqLLM()
                state.repo_summaries[repo_id] = _build_repo_summary(
                    repo_id=repo_id,
                    llm=llm,
                    model_name=model_name
                )
            answer = state.repo_summaries[repo_id]
            repo_name = state.repo_id_to_name.get(repo_id, repo_id)
            summaries.append(f"Repo {repo_name}:\n{answer}")

        final_answer = "\n\n".join(summaries)
        chat_history[-1] = (question, final_answer)
        yield "", chat_history
        return

    if filename:
        file_chunks = _find_all_file_chunks(target_repo_id, filename)
        if not file_chunks:
            suggestions = _find_similar_files(target_repo_id, filename)
            hint = ""
            if suggestions:
                hint = "\nSimilar files:\n- " + "\n- ".join(suggestions)
            chat_history[-1] = (question, f"File '{filename}' was not found in the indexed repo.{hint}")
            yield "", chat_history
            return
        llm = GroqLLM()
        answer = _build_file_summary(
            chunks=file_chunks,
            llm=llm,
            model_name=model_name
        )
        chat_history[-1] = (question, answer)
        yield "", chat_history
        return

    if summary_request:
        if target_repo_id not in state.repo_summaries:
            llm = GroqLLM()
            state.repo_summaries[target_repo_id] = _build_repo_summary(
                repo_id=target_repo_id,
                llm=llm,
                model_name=model_name
            )
        answer = state.repo_summaries[target_repo_id]
        chat_history[-1] = (question, answer)
        yield "", chat_history
        return

    repo_chunks = []
    if filename:
        repo_chunks = _find_file_chunks(target_repo_id, filename)
        if not repo_chunks:
            suggestions = _find_similar_files(target_repo_id, filename)
            hint = ""
            if suggestions:
                hint = "\nSimilar files:\n- " + "\n- ".join(suggestions)
            chat_history[-1] = (question, f"File '{filename}' was not found in the indexed repo.{hint}")
            yield "", chat_history
            return
    elif symbol_name:
        symbol_files = _find_symbol_files(target_repo_id, symbol_name)
        repo_chunks = _find_chunks_by_paths(target_repo_id, symbol_files)
        if not repo_chunks:
            chat_history[-1] = (question, f"I could not find '{symbol_name}' in the indexed code. Please specify a file.")
            yield "", chat_history
            return
    elif entry_request:
        repo_chunks = _find_entry_point_chunks(target_repo_id)
    if not repo_chunks:
        repo_chunks = Retriever(repo_id=target_repo_id, top_k=doc_top_k).retrieve(question)
    repo_map_chunks = [
        c for c in FaissStore.load_metadata(target_repo_id)
        if c.get("chunk_type") == "repo_map"
    ]
    symbol_map_chunks = []
    if doc_request or entry_request:
        symbol_map_chunks = [
            c for c in FaissStore.load_metadata(target_repo_id)
            if c.get("chunk_type") == "symbol_map"
        ]
    metadata_chunks = []
    if state.profile_metadata_id and FaissStore.exists(state.profile_metadata_id):
        metadata_chunks = Retriever(repo_id=state.profile_metadata_id, top_k=TOP_K).retrieve(question)

    if filename or symbol_name:
        retrieved_chunks = repo_map_chunks + repo_chunks + metadata_chunks
    elif entry_request:
        retrieved_chunks = repo_map_chunks + symbol_map_chunks + repo_chunks + metadata_chunks
    else:
        retrieved_chunks = repo_map_chunks + symbol_map_chunks + repo_chunks + metadata_chunks

    if not repo_chunks and not repo_map_chunks:
        chat_history[-1] = (question, "I could not find relevant code. Please specify a file or function name.")
        yield "", chat_history
        return

    max_context_tokens = _max_context_tokens(model_name, doc_request)
    context_blocks = _build_context_blocks(retrieved_chunks, max_context_tokens)

    user_prompt = f"""
Context:
{''.join(context_blocks)}

Question:
{question}
"""

    with open("app/prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm = GroqLLM()
    answer = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model_name,
        temperature=temperature,
        top_p=top_p
    )

    chat_history[-1] = (question, answer)
    yield "", chat_history

    # Clear input box after submission
    return


def launch_ui():
    with gr.Blocks(css="""
    html, body { height: 100%; margin: 0; }
    .gradio-container { height: 100vh; }
    #app { height: 100vh; display: flex; flex-direction: column; }
    #main-row { flex: 1; min-height: 0; }
    #chatbot { height: 100%; }
""", js="""
() => {
  const root = document.querySelector("#chatbot");
  if (!root) return;
  const target = root.querySelector(".wrap") || root;
  const observer = new MutationObserver(() => {
    target.scrollTop = target.scrollHeight;
  });
  observer.observe(root, { childList: true, subtree: true });
}
""", elem_id="app") as demo:

        gr.Markdown("## Chat with Your Codebases")

        state = gr.State(SessionState())
        index_ok = gr.State(False)

        # --- Repo input ---
        with gr.Row():
            repo_input = gr.Textbox(
                label="GitHub Repo or Profile",
                placeholder="https://github.com/username OR /repo",
                scale=6
            )
            repo_submit = gr.Button("Index", variant="primary", scale=1, interactive=False)
            index_status = gr.Textbox(label="Indexing Status", scale=3)

        # --- Model settings (below repo input) ---
        with gr.Row():
            with gr.Accordion("Model Settings", open=False):
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="llama-3.1-8b-instant",
                    label="LLM Model"
                )

                temperature = gr.Slider(0, 1, value=0.2, step=0.05, label="Temperature")
                top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="Top-P")

        # --- Main layout ---
        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=1, elem_id="chat-column"):
                chatbot = gr.Chatbot(
                    label="Repository Chat",
                    elem_id="chatbot",
                    height=None
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question about the codebase",
                        placeholder="Type your question here...",
                        interactive=False,
                        scale=8
                    )
                    question_submit = gr.Button(
                        "Send",
                        variant="primary",
                        interactive=False,
                        scale=1
                    )

        # --- Events ---
        repo_input.submit(
            index_repository,
            inputs=[repo_input, state],
            outputs=[index_status, index_ok],
            api_name=False
        )
        repo_submit.click(
            index_repository,
            inputs=[repo_input, state],
            outputs=[index_status, index_ok],
            api_name=False
        )
        repo_input.change(
            lambda q: gr.update(interactive=bool(q and q.strip())),
            inputs=[repo_input],
            outputs=[repo_submit]
        )
        index_ok.change(
            lambda ok: (gr.update(interactive=ok), gr.update(interactive=ok)),
            inputs=index_ok,
            outputs=[question_input, question_submit]
        )

        question_input.submit(
            answer_question,
            inputs=[
                question_input,
                model_dropdown,
                temperature,
                top_p,
                chatbot,
                state
            ],
            outputs=[question_input, chatbot],
            api_name=False
        )
        question_submit.click(
            answer_question,
            inputs=[
                question_input,
                model_dropdown,
                temperature,
                top_p,
                chatbot,
                state
            ],
            outputs=[question_input, chatbot],
            api_name=False
        )

        question_input.change(
            lambda q, ok: gr.update(interactive=bool(q and q.strip()) and ok),
            inputs=[question_input, index_ok],
            outputs=[question_submit]
        )

        demo.unload(on_exit)

    demo.launch(share=True, show_api=False)


def on_exit():
    from app.utils.cleanup import reset_storage
    reset_storage()
    print("Cleaned repos and FAISS indexes on shutdown.")
