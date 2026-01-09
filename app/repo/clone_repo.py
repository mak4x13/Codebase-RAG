import os
import shutil
import stat
from git import Repo
from pathlib import Path

def remove_readonly(func, path, _):
    """Fix Windows PermissionError for readonly files"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(repo_url: str, base_path="data/repos") -> str:
    """
    Clones a public GitHub repo into a temporary folder.
    Returns the local path of the cloned repo.
    """
    from app.utils.repo_id import generate_repo_id

    repo_id = generate_repo_id(repo_url)
    local_path = Path(base_path) / repo_id

    # Remove old folder if exists
    if local_path.exists():
        shutil.rmtree(local_path, onerror=remove_readonly)

    # Clone repo
    try:
        Repo.clone_from(repo_url, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to clone {repo_url}: {e}")

    return str(local_path)

def cleanup_repo(repo_path: str):
    """Delete cloned repo folder and its FAISS index"""
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)

    # Delete corresponding FAISS folder
    faiss_path = repo_path.replace("repos", "faiss")
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path, onerror=remove_readonly)
