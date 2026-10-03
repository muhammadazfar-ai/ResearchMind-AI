import numpy as np
import faiss

# Cache model in memory globally so it only loads once
_MODEL_CACHE = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Loads and caches the SentenceTransformer model (imported lazily to keep startup fast)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model '{model_name}'...")
        _MODEL_CACHE = SentenceTransformer(model_name)
    return _MODEL_CACHE


def embed_texts(texts: list, model=None) -> np.ndarray:
    """Encodes texts into L2-normalized float32 vectors (so inner product == cosine similarity)."""
    model = model or get_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def build_cosine_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Builds an in-memory FAISS inner-product index over normalized vectors."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
