import numpy as np
from app.embeddings.embedder import CodeEmbedder
from app.vectorstore.faiss_store import FaissStore

class Retriever:
    def __init__(self, repo_id: str, top_k: int = 6):
        self.repo_id = repo_id
        self.top_k = top_k
        self.embedder = CodeEmbedder()
        self.store = FaissStore(repo_id)
        if FaissStore.exists(repo_id):
            self.store.load()
        else:
            self.store.index = None
            self.store.metadata = []

    def retrieve(self, query: str):
        """
        Returns top-K relevant chunks for a query.
        """
        if self.store.index is None:
            return []
        query_embedding = self.embedder.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.store.index.search(
            np.array(query_embedding), self.top_k 
            # Returns similarity scores and their positions 
        )

        results = []
        for idx in indices[0]:
            # Ensures the index exists
            # Prevents “index out of range” errors
            if idx < len(self.store.metadata):
                results.append(self.store.metadata[idx])
                # Fetches the actual code chunk and adds it to results.

        return results
