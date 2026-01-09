import os
import shutil
import stat
from dotenv import load_dotenv
from app.embeddings.embedder import CodeEmbedder

# Load environment variables
load_dotenv()

REPO_DIR = "data/repos"
FAISS_DIR = "data/faiss_index"

def remove_readonly(func, path, _):
    """Handler for readonly files on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Clean old repos and FAISS indexes safely
for path in [REPO_DIR, FAISS_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path, onerror=remove_readonly)
    os.makedirs(path, exist_ok=True)

print("Cleaned old repos and FAISS indexes safely.")

# Preload CodeBERT model once
print("Loading SentenceTransformer model...")
try:
    embedder = CodeEmbedder()
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
else:
    print("SentenceTransformer model loaded successfully.")

# Launch Gradio app
from app.ui.gradio_app import launch_ui
launch_ui()
