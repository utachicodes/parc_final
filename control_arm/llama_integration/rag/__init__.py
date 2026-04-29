"""
RAG System for PARC Robotics SO-101
===================================
Lightweight Retrieval-Augmented Generation using ChromaDB + Sentence Transformers
Optimized for Jetson Nano Orin
"""

from .embeddings import EmbeddingManager, get_embedding_manager
from .vector_store import VectorStore, get_vector_store
from .document_processor import DocumentProcessor, RobotDocsProcessor
from .rag_system import RAGSystem, get_rag_system, RetrievalResult

__all__ = [
    'EmbeddingManager',
    'get_embedding_manager',
    'VectorStore',
    'get_vector_store',
    'DocumentProcessor',
    'RobotDocsProcessor',
    'RAGSystem',
    'get_rag_system',
    'RetrievalResult'
]
