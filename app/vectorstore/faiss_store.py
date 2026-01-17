import faiss
import json
import os
import numpy as np

class FaissStore:
    def __init__(self, repo_id, base_path="data/faiss_index"):
        self.repo_id = repo_id
        self.index_path = os.path.join(base_path, repo_id)
        os.makedirs(self.index_path, exist_ok=True)

        self.index_file = os.path.join(self.index_path, "index.faiss")
        self.meta_file = os.path.join(self.index_path, "metadata.json")

        self.index = None
        self.metadata = []

    def build(self, embeddings: np.ndarray, chunks: list):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.metadata = chunks

        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self):
        if not os.path.exists(self.index_file) or not os.path.exists(self.meta_file):
            raise FileNotFoundError("FAISS index or metadata not found.")
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    @staticmethod
    def exists(repo_id: str) -> bool:
        return os.path.exists(f"data/faiss_index/{repo_id}/index.faiss")

    @staticmethod
    def load_metadata(repo_id: str) -> list:
        meta_file = os.path.join("data/faiss_index", repo_id, "metadata.json")
        if not os.path.exists(meta_file):
            return []
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
