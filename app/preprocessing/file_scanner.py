import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".go", ".rs",
    ".rb", ".cs", ".kt", ".swift", ".scala",
    ".md", ".txt",
    ".json", ".yaml", ".yml", ".toml",
    ".asm"
    }

BINARY_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".exe", ".zip", ".bin", ".dll"
}

IGNORE_DIRS = {".git", "node_modules", "venv", "dist", "build"}

def get_code_files(repo_path: str):
    """
    Recursively scans the repo folder and returns a list of file paths
    that match supported extensions.
    """
    repo_path = Path(repo_path)
    code_files = []

    for root, dirs, files in os.walk(repo_path):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            suffix = Path(file).suffix.lower()

            if suffix in BINARY_EXTENSIONS:
                continue

            if suffix in SUPPORTED_EXTENSIONS:
                code_files.append(str(Path(root) / file))

    return code_files
