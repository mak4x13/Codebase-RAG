from sentence_transformers import SentenceTransformer
import numpy as np

_CACHED_MODEL = None


class CodeEmbedder:
    def __init__(self, model_name="BAAI/bge-code-v1"):
        global _CACHED_MODEL
        if _CACHED_MODEL is None:
            _CACHED_MODEL = SentenceTransformer(model_name)
        self.model = _CACHED_MODEL

    def embed_chunks(self, chunks):
        """
        Takes list of chunks and returns numpy array of embeddings.
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.array(embeddings)
