from pathlib import Path
from typing import List, Dict

CHUNK_SIZE = 50     # max lines per chunk
CHUNK_OVERLAP = 10  # lines overlap between chunks


def read_text_file(file_path: str):
    for enc in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def chunk_file(file_path: str) -> List[Dict]:
    """
    Reads a file and splits it into chunks preserving function/class boundaries.
    Returns a list of chunks with metadata: start_line, end_line, content.
    """
    chunks = []
    lines = read_text_file(file_path)
    if not lines:
        print(f"⚠️ Skipped unreadable file: {file_path}")
        return []

    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        content = "".join(lines[start:end])
        chunks.append({
            "file_path": str(file_path),
            "start_line": start + 1,
            "end_line": end,
            "content": content
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP  # move forward with overlap

    return chunks
