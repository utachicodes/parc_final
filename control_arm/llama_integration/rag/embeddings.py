"""
Embedding Manager - Lightweight embeddings for Jetson Nano
Uses sentence-transformers with all-MiniLM-L6-v2 (fast, low memory)
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Model: all-MiniLM-L6-v2 - 384 dimensions, 80MB, very fast on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingManager:
    """
    Manages sentence embeddings for RAG.
    Uses sentence-transformers with lightweight model optimized for edge devices.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",  # cpu for Jetson, cuda if available
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension = EMBEDDING_DIM

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def encode(self, texts: str | List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (lower on Jetson for memory)

        Returns:
            List of embedding vectors
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # Convert to list of floats
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            return [[0.0] * self._dimension for _ in texts]

    def encode_query(self, query: str) -> List[float]:
        """
        Encode a search query (same as encode but always single query).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.encode(query, batch_size=1)[0]

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (-1 to 1)
        """
        import numpy as np
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Singleton instance
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get singleton embedding manager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
