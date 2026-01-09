from sentence_transformers import SentenceTransformer
import numpy as np

class CodeEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
