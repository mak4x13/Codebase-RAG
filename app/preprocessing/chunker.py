import re
from typing import Dict, List

CHUNK_SIZE = 50     # max lines per chunk
CHUNK_OVERLAP = 10  # lines overlap between chunks
MIN_CHUNK_LINES = 10

BOUNDARY_PATTERN = re.compile(r"^\s*(def|class|function|interface|struct|enum)\s+\w+")


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
        print(f"Skipped unreadable file: {file_path}")
        return []

    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        if end < len(lines):
            # Try to end chunk at the last boundary inside the window.
            candidate = None
            for i in range(start + 1, end):
                if BOUNDARY_PATTERN.match(lines[i]):
                    candidate = i
            if candidate is not None and (candidate - start) >= MIN_CHUNK_LINES:
                end = candidate

        content = "".join(lines[start:end])
        chunks.append({
            "file_path": str(file_path),
            "start_line": start + 1,
            "end_line": end,
            "content": content
        })

        if end >= len(lines):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)  # move forward with overlap

    return chunks
