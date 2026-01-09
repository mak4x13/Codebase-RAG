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
from app.utils.cleanup import reset_storage


def index_repository(repo_url, state):
    try:
        # Hard reset when new input comes
        reset_storage()
        state.repo_id = None
        state.repo_url = None

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
        if not metadata_chunk:
            metadata_chunk = [{
                "file_path": "profile_metadata",
                "start_line": 0,
                "end_line": 0,
                "content": "",
                "chunk_type": "metadata"
            }]

        metadata_embeddings = embedder.embed_chunks(metadata_chunk)
        FaissStore("profile_metadata").build(metadata_embeddings, metadata_chunk)

        for repo in repos:
            repo_id = generate_repo_id(repo["clone_url"])
            local_path = clone_repo(repo["clone_url"])

            chunks = preprocess_repository(
                repo_path=local_path,
                repo_id=repo_id,
                repo_url=repo["clone_url"]
            )

            if not chunks:
                print(f"Skipping {repo['name']} — no indexable files found.")
                continue

            embeddings = embedder.embed_chunks(chunks)
            FaissStore(repo_id).build(embeddings, chunks)

            state.repo_id = repo_id  # last repo active
            state.repo_url = repo_url

        return f"Indexed {len(repos)} repository(ies) successfully.", True

    except GitHubResolverError as e:
        return f"{str(e)}", False

    except Exception as e:
        return f"Unexpected error: {str(e)}", False



def answer_question(question, model_name, temperature, top_p, chat_history, state):
    # Prevent NoneType errors
    if not state.repo_id:
        return "Please index a repository first.", chat_history

    TOP_K = 5

    repo_chunks = Retriever(repo_id=state.repo_id, top_k=TOP_K).retrieve(question)
    metadata_chunks = []
    if FaissStore.exists("profile_metadata"):
        metadata_chunks = Retriever(repo_id="profile_metadata", top_k=TOP_K).retrieve(question)

    retrieved_chunks = repo_chunks + metadata_chunks

    context_blocks = [
        f"""
File: {c['file_path']}
Lines: {c['start_line']}-{c['end_line']}
Code:
{c['content']}
""" for c in retrieved_chunks
    ]

    user_prompt = f"""
Repository code context:
{''.join(context_blocks)}
Profile metadata:
{''.join([c['content'] for c in retrieved_chunks if c.get('chunk_type') == 'metadata'])}

Code context:
{''.join([c['content'] for c in retrieved_chunks if c.get('chunk_type') != 'metadata'])}

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

    chat_history.append((question, answer))

    # Clear input box after submission
    return "", chat_history

def launch_ui():
    with gr.Blocks(css="""
    #chatbot { height: 520px; overflow: auto; }
    #model-settings { height: 520px; overflow: auto; }
""") as demo:

        gr.Markdown("## 💬 Chat with Your Codebases")

        state = gr.State(SessionState())
        index_ok = gr.State(False)

        # --- Repo input ---
        with gr.Row():
            repo_input = gr.Textbox(
                label="GitHub Repo or Profile",
                placeholder="https://github.com/username OR /repo"
            )
            index_status = gr.Textbox(label="Indexing Status")

        # --- Main layout ---
        with gr.Row():
            # Left: controls
            with gr.Column(scale=1, min_width=260, elem_id="model-settings"):
                gr.Markdown("### ⚙️ Model Settings")

                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="llama-3.1-8b-instant",
                    label="LLM Model"
                )

                temperature = gr.Slider(0, 1, value=0.2, step=0.05, label="Temperature")
                top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="Top-P")

            # Right: chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Repository Chat",
                    elem_id="chatbot",
                    height=520
                )

                question_input = gr.Textbox(
                    label="Ask a question about the codebase",
                    placeholder="Type your question here...",
                    interactive=False
                )

        # --- Events ---
        repo_input.submit(
            index_repository,
            inputs=[repo_input, state],
            outputs=[index_status, index_ok],
            api_name=False
        ).then(
            lambda ok: gr.update(interactive=ok),
            inputs=index_ok,
            outputs=question_input
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

        demo.unload(on_exit)

    demo.launch(share=True, show_api=False)


def on_exit():
    from app.utils.cleanup import reset_storage
    reset_storage()
    print("Cleaned repos and FAISS indexes on shutdown.")

