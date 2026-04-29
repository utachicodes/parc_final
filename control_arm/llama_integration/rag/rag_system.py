"""
RAG System - Main interface for Retrieval-Augmented Generation
Integrates embeddings, vector store, and document processing
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default RAG data directory
DEFAULT_RAG_DIR = os.path.expanduser("~/models/rag")


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    distance: float = 0.0


class RAGSystem:
    """
    Complete RAG system for PARC Robotics SO-101.
    Provides retrieval-augmented generation with robot knowledge base.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_RAG_DIR,
        collection_name: str = "parc_robotics",
        embedding_model: str = "all-MiniLM-L6-v2",
        n_results: int = 5,
        similarity_threshold: float = 0.3
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.n_results = n_results
        self.similarity_threshold = similarity_threshold

        # Lazy load components
        self._embedding_manager = None
        self._vector_store = None
        self._doc_processor = None

    @property
    def embedding_manager(self):
        """Lazy load embedding manager."""
        if self._embedding_manager is None:
            from .embeddings import EmbeddingManager, get_embedding_manager
            self._embedding_manager = get_embedding_manager()
        return self._embedding_manager

    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from .vector_store import VectorStore, get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def doc_processor(self):
        """Lazy load document processor."""
        if self._doc_processor is None:
            from .document_processor import RobotDocsProcessor
            self._doc_processor = RobotDocsProcessor()
        return self._doc_processor

    def add_documents(
        self,
        texts: List[str],
        sources: List[str],
        categories: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add documents to the knowledge base.

        Args:
            texts: List of document texts
            sources: List of source identifiers
            categories: List of categories
            metadatas: Optional additional metadata

        Returns:
            True if successful
        """
        try:
            # Process documents into chunks
            all_chunks = []
            for i, (text, source, category) in enumerate(zip(texts, sources, categories)):
                chunks = self.doc_processor.process_text(
                    text=text,
                    source=source,
                    category=category,
                    metadata=metadatas[i] if metadatas else None
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning("No chunks generated from documents")
                return False

            # Generate embeddings
            texts_for_embedding = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_manager.encode(texts_for_embedding)

            # Add embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding

            # Add to vector store
            ids = [chunk.id for chunk in all_chunks]
            embeddings_list = [chunk.embedding for chunk in all_chunks]
            documents = [chunk.text for chunk in all_chunks]
            chunk_metadatas = [chunk.metadata for chunk in all_chunks]

            success = self.vector_store.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=chunk_metadatas
            )

            logger.info(f"Added {len(all_chunks)} chunks to knowledge base")
            return success

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            n_results: Number of results (default: self.n_results)
            category: Optional category filter

        Returns:
            List of RetrievalResult objects
        """
        try:
            n = n_results or self.n_results

            # Encode query
            query_embedding = self.embedding_manager.encode_query(query)

            # Build filter if category specified
            where_filter = None
            if category:
                where_filter = {"category": category}

            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n * 2,  # Over-fetch for filtering
                where_filter=where_filter
            )

            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, doc_id in enumerate(results.get('ids', [])):
                distance = results['distances'][i] if i < len(results['distances']) else 0.0
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                score = 1.0 / (1.0 + distance)

                if score >= self.similarity_threshold:
                    retrieval_results.append(RetrievalResult(
                        id=doc_id,
                        text=results['documents'][i] if i < len(results['documents']) else "",
                        metadata=results['metadatas'][i] if i < len(results['metadatas']) else {},
                        score=score,
                        distance=distance
                    ))

            # Sort by score descending
            retrieval_results.sort(key=lambda x: x.score, reverse=True)

            # Limit to n results
            return retrieval_results[:n]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_with_context(
        self,
        query: str,
        n_results: Optional[int] = None,
        category: Optional[str] = None,
        max_context_length: int = 2000
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve documents and format as context string.

        Args:
            query: Search query
            n_results: Number of results
            category: Optional category filter
            max_context_length: Maximum context length in characters

        Returns:
            Tuple of (context_string, retrieval_results)
        """
        results = self.retrieve(query, n_results, category)

        if not results:
            return "", []

        # Build context string
        context_parts = []
        current_length = 0

        for result in results:
            # Add separator if not first
            if context_parts:
                context_parts.append("\n---\n")

            # Format as markdown
            source = result.metadata.get('source', 'unknown')
            header = result.metadata.get('header', '')
            category = result.metadata.get('category', '')

            part = f"[{category.upper()}] {source}"
            if header:
                part += f" - {header}"
            part += f"\n{result.text}"

            # Check if adding this would exceed limit
            if current_length + len(part) > max_context_length:
                break

            context_parts.append(part)
            current_length += len(part)

        context = "".join(context_parts)
        return context, results

    def query(
        self,
        question: str,
        llm_chat_func=None,  # Function to call LLM chat
        include_context: bool = True,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User question
            llm_chat_func: Function to call LLM (receives messages list)
            include_context: Whether to include retrieved context
            system_prompt: Optional system prompt override

        Returns:
            Dict with answer, sources, and context
        """
        default_system = """You are an AI assistant for PARC Robotics SO-101 robot arm.
Answer questions based on the provided context. If the context doesn't contain
relevant information, say so. Be technical and precise, especially when discussing
robotics concepts, kinematics, or code."""

        result = {
            'question': question,
            'answer': None,
            'sources': [],
            'context_used': False,
            'error': None
        }

        if include_context:
            # Retrieve relevant context
            context, sources = self.retrieve_with_context(
                question,
                n_results=5,
                max_context_length=1500
            )

            result['sources'] = [
                {
                    'id': s.id,
                    'text': s.text[:200] + "..." if len(s.text) > 200 else s.text,
                    'score': s.score,
                    'metadata': s.metadata
                }
                for s in sources
            ]

            if context:
                # Build messages with context
                messages = [
                    {"role": "system", "content": system_prompt or default_system},
                    {"role": "system", "content": f"Context:\n{context}"},
                    {"role": "user", "content": question}
                ]
                result['context_used'] = True
            else:
                # No context found
                messages = [
                    {"role": "system", "content": system_prompt or default_system},
                    {"role": "user", "content": question}
                ]
        else:
            messages = [
                {"role": "system", "content": system_prompt or default_system},
                {"role": "user", "content": question}
            ]

        # Call LLM if function provided
        if llm_chat_func:
            try:
                response = llm_chat_func(messages)
                result['answer'] = response
            except Exception as e:
                result['error'] = f"LLM call failed: {e}"
        else:
            result['answer'] = "LLM not configured. Install llama-server for AI responses."

        return result

    def count_documents(self) -> int:
        """Get number of documents in knowledge base."""
        return self.vector_store.count()

    def reset(self) -> bool:
        """Reset (clear) the knowledge base."""
        success = self.vector_store.reset()
        if success:
            logger.warning("Knowledge base reset")
        return success


# Singleton instance
_rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """Get singleton RAG system."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system
