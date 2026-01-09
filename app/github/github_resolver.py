import requests
from urllib.parse import urlparse
import os

GITHUB_API = "https://api.github.com"

class GitHubResolverError(Exception):
    pass


def _headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise GitHubResolverError("GitHub token not found in environment.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }


def resolve_github_url(url: str) -> list[dict]:
    """
    Returns a list of valid repos with:
    { name, clone_url }
    """
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        raise GitHubResolverError("Not a valid GitHub URL.")

    parts = parsed.path.strip("/").split("/")

    # PROFILE: github.com/username
    if len(parts) == 1:
        return _resolve_profile(parts[0])

    # REPO: github.com/username/repo
    if len(parts) == 2:
        return [_resolve_repo(parts[0], parts[1])]

    raise GitHubResolverError("Invalid GitHub URL format.")


def _resolve_repo(owner: str, repo: str) -> dict:
    r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}",
        headers=_headers()
    )

    if r.status_code == 404:
        raise GitHubResolverError("Repository does not exist.")

    data = r.json()

    if data.get("fork"):
        raise GitHubResolverError("Forked repositories are not supported.")

    if data.get("size", 0) == 0:
        raise GitHubResolverError("Repository is empty.")

    if data.get("archived"):
        raise GitHubResolverError("Repository is archived.")

    return {
        "name": data["full_name"],
        "clone_url": data["clone_url"]
    }


def _resolve_profile(username: str) -> list[dict]:
    r = requests.get(
        f"{GITHUB_API}/users/{username}/repos",
        headers=_headers(),
        params={"per_page": 100}
    )

    if r.status_code == 404:
        raise GitHubResolverError("GitHub profile not found.")

    repos = []
    for repo in r.json():
        if repo["fork"]:
            continue
        if repo["size"] == 0:
            continue
        if repo["archived"]:
            continue

        repos.append({
            "name": repo["full_name"],
            "clone_url": repo["clone_url"]
        })

    if not repos:
        raise GitHubResolverError("No valid repositories found in profile.")

    return repos
