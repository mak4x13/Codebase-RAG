from app.preprocessing.file_scanner import get_code_files
from app.preprocessing.chunker import chunk_file

def preprocess_repository(repo_path: str, repo_id: str, repo_url: str):
    """
    Full preprocessing pipeline for a repository.
    Returns a list of chunks with repo-aware metadata.
    """
    all_chunks = []

    code_files = get_code_files(repo_path)

    for file_path in code_files:
        chunks = chunk_file(file_path)

        for chunk in chunks:
            chunk["repo_id"] = repo_id
            chunk["repo_url"] = repo_url
            all_chunks.append(chunk)

    return all_chunks
