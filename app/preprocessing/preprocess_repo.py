import os

import re

from app.preprocessing.file_scanner import get_code_files
from app.preprocessing.chunker import chunk_file

MAX_REPO_MAP_FILES = 200
MAX_REPO_MAP_CHARS = 8000
MAX_SYMBOL_FILES = 120
MAX_SYMBOLS_PER_FILE = 20
MAX_SYMBOL_MAP_CHARS = 8000

SYMBOL_PATTERN = re.compile(
    r"^\s*(def|class|function|interface|struct|enum)\s+([A-Za-z_]\w*)"
)
ASSIGN_FUNC_PATTERN = re.compile(
    r"^\s*(const|let|var)\s+([A-Za-z_]\w*)\s*=\s*\("
)


def _build_repo_map(repo_path: str, repo_name: str, code_files: list[str]) -> str:
    rel_paths = [os.path.relpath(p, repo_path) for p in code_files]
    rel_paths.sort()

    total = len(rel_paths)
    shown = rel_paths[:MAX_REPO_MAP_FILES]
    suffix = "" if total <= MAX_REPO_MAP_FILES else "\n... (truncated)"

    content = [
        f"Repository: {repo_name}",
        f"Files: {total}",
        "File list:"
    ]
    content.extend([f"- {p}" for p in shown])
    content.append(suffix)

    text = "\n".join([line for line in content if line])
    return text[:MAX_REPO_MAP_CHARS]


def _read_text_file(file_path: str):
    for enc in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def _extract_symbols(lines: list[str]) -> list[str]:
    symbols = []
    for line in lines:
        match = SYMBOL_PATTERN.match(line)
        if match:
            symbols.append(match.group(2))
            if len(symbols) >= MAX_SYMBOLS_PER_FILE:
                break
            continue
        match = ASSIGN_FUNC_PATTERN.match(line)
        if match:
            symbols.append(match.group(2))
            if len(symbols) >= MAX_SYMBOLS_PER_FILE:
                break
    return symbols


def _build_symbol_map(repo_path: str, repo_name: str, code_files: list[str]) -> str:
    rel_paths = [os.path.relpath(p, repo_path) for p in code_files]
    rel_paths.sort()

    entries = []
    for rel_path in rel_paths[:MAX_SYMBOL_FILES]:
        full_path = os.path.join(repo_path, rel_path)
        lines = _read_text_file(full_path)
        if not lines:
            continue
        symbols = _extract_symbols(lines)
        if symbols:
            entries.append(f"- {rel_path}: {', '.join(symbols)}")

    if not entries:
        return ""

    content = [
        f"Repository: {repo_name}",
        "Symbol map (top definitions per file):"
    ]
    content.extend(entries)

    text = "\n".join(content)
    return text[:MAX_SYMBOL_MAP_CHARS]


def preprocess_repository(repo_path: str, repo_id: str, repo_url: str, repo_name: str):
    """
    Full preprocessing pipeline for a repository.
    Returns a list of chunks with repo-aware metadata.
    """
    all_chunks = []

    code_files = get_code_files(repo_path)
    repo_map = _build_repo_map(repo_path, repo_name, code_files)
    if repo_map:
        all_chunks.append({
            "file_path": f"{repo_name}_repo_map",
            "start_line": 1,
            "end_line": 1,
            "content": repo_map,
            "chunk_type": "repo_map",
            "repo_id": repo_id,
            "repo_name": repo_name,
            "repo_url": repo_url
        })

    symbol_map = _build_symbol_map(repo_path, repo_name, code_files)
    if symbol_map:
        all_chunks.append({
            "file_path": f"{repo_name}_symbol_map",
            "start_line": 1,
            "end_line": 1,
            "content": symbol_map,
            "chunk_type": "symbol_map",
            "repo_id": repo_id,
            "repo_name": repo_name,
            "repo_url": repo_url
        })

    for file_path in code_files:
        chunks = chunk_file(file_path)

        for chunk in chunks:
            chunk["repo_id"] = repo_id
            chunk["repo_name"] = repo_name
            chunk["repo_url"] = repo_url
            all_chunks.append(chunk)

    return all_chunks
