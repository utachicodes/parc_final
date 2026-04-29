"""
Vector Store - ChromaDB for lightweight vector storage
Optimized for Jetson Nano Orin
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default persist directory
DEFAULT_PERSIST_DIR = os.path.expanduser("~/models/rag")


class VectorStore:
    """
    ChromaDB-backed vector store for embeddings.
    Persistent storage for RAG knowledge base.
    """

    def __init__(
        self,
        collection_name: str = "parc_robotics",
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_dim: int = 384
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                os.makedirs(self.persist_dir, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(f"ChromaDB client initialized at {self.persist_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise

        return self._client

    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We provide embeddings manually
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create new
                self._collection = client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "PARC Robotics SO-101 knowledge base"},
                    embedding_function=None
                )
                logger.info(f"Created new collection: {self.collection_name}")

        return self._collection

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add documents to the vector store.

        Args:
            ids: Unique IDs for each document
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: Optional metadata for each document

        Returns:
            True if successful
        """
        try:
            collection = self._get_collection()

            # Ensure embeddings are list of lists of floats
            embeddings = [[float(x) for x in emb] for emb in embeddings]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{} for _ in ids]
            )
            logger.info(f"Added {len(ids)} documents to collection")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        where_document_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Metadata filter (ChromaDB where clause)
            where_document_filter: Document content filter

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        try:
            collection = self._get_collection()

            # Ensure embedding is list of floats
            query_embedding = [float(x) for x in query_embedding]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                where_document=where_document_filter
            )

            # Format results
            if results and results.get('ids'):
                return {
                    'ids': results['ids'][0],
                    'documents': results['documents'][0],
                    'metadatas': results['metadatas'][0] if results.get('metadatas') else [{}],
                    'distances': results['distances'][0] if results.get('distances') else []
                }
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

    def search_text(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search by text (requires external embedding).

        Args:
            query: Query text
            n_results: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            Search results
        """
        # This would need embedding from external source
        # For now, use search with pre-computed embedding
        logger.warning("search_text requires external embedding. Use search() instead.")
        return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

    def get(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents by IDs.

        Args:
            ids: Document IDs

        Returns:
            Dict with documents and metadata
        """
        try:
            collection = self._get_collection()
            results = collection.get(ids=ids)
            return {
                'ids': results.get('ids', []),
                'documents': results.get('documents', []),
                'metadatas': results.get('metadatas', [])
            }
        except Exception as e:
            logger.error(f"Get failed: {e}")
            return {'ids': [], 'documents': [], 'metadatas': []}

    def count(self) -> int:
        """Get number of documents in collection."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            collection = self._get_collection()
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def reset(self) -> bool:
        """Reset (delete all) the collection."""
        try:
            client = self._get_client()
            client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.warning(f"Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
