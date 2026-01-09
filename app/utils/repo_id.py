import hashlib
from urllib.parse import urlparse

def generate_repo_id(repo_url: str) -> str:
    """
    Generates a deterministic repo ID from a GitHub URL.
    Format: <repo_name>__<short_hash>
    """
    parsed = urlparse(repo_url)
    repo_name = parsed.path.rstrip('/').split('/')[-1]
    hash_digest = hashlib.sha1(repo_url.encode()).hexdigest()[:6]
    return f"{repo_name}__{hash_digest}"
