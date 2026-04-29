"""
Document Processor - Semantic chunking for robot documentation
Respects document structure for quality retrieval
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class DocumentProcessor:
    """
    Processes documents into chunks for RAG.
    Uses semantic chunking - respects sentences, paragraphs, and headers.
    """

    def __init__(
        self,
        chunk_size: int = 500,  # Target characters per chunk
        overlap: int = 100,  # Overlap between chunks for context
        min_chunk_size: int = 100,  # Minimum chunk size
        max_chunk_size: int = 1000,  # Maximum chunk size
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def process_text(
        self,
        text: str,
        source: str = "unknown",
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Process raw text into chunks.

        Args:
            text: Raw document text
            source: Document source (URL, filename, etc.)
            category: Document category (kinematics, programming, etc.)
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        # Clean text
        text = self._clean_text(text)

        # Split into semantic units
        chunks_text = self._semantic_chunk(text)

        # Create chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk_id = self._generate_id(f"{source}:{i}")

            chunk_metadata = {
                'source': source,
                'category': category,
                'chunk_index': i,
                'total_chunks': len(chunks_text),
                **(metadata or {})
            }

            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text.strip(),
                metadata=chunk_metadata
            ))

        logger.info(f"Processed {len(chunks)} chunks from {source}")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        # Remove special chars but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)\[\]{}"\']+', '', text)
        return text.strip()

    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        Respects sentence and paragraph boundaries.
        """
        if not text:
            return []

        chunks = []

        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\n+', text)

        current_chunk = ""
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_size = len(paragraph)

            # If single paragraph exceeds max, split by sentences
            if para_size > self.max_chunk_size:
                sentences = self._split_sentences(paragraph)
                for sentence in sentences:
                    if current_size + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = sentence[self.overlap:] if self.overlap else sentence
                        current_size = len(current_chunk)
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_size += len(sentence) + 1
                continue

            # Check if adding paragraph exceeds chunk size
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk[-self.overlap:]:
                    overlap_text = current_chunk[-self.overlap:]
                    # Don't start mid-sentence if possible
                    overlap_text = self._trim_to_sentence_boundary(overlap_text)
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += para_size + 2

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _trim_to_sentence_boundary(self, text: str) -> str:
        """Trim text to sentence boundary."""
        # Find last sentence end
        match = re.search(r'[.!?]\s*[A-Z]', text[-50:])
        if match:
            return text[match.end():]
        return text

    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text."""
        hash_obj = hashlib.md5(text.encode())
        return f"chunk_{hash_obj.hexdigest()[:12]}"

    def process_markdown(
        self,
        markdown: str,
        source: str,
        category: str = "documentation"
    ) -> List[Chunk]:
        """
        Process markdown document, preserving structure.

        Args:
            markdown: Markdown text
            source: Document source
            category: Category

        Returns:
            List of chunks with markdown preserved
        """
        # Keep markdown syntax for code blocks and headers
        chunks = []
        current_section = ""
        current_size = 0

        lines = markdown.split('\n')
        current_header = ""

        for line in lines:
            # Track headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                current_header = header_match.group(2).strip()

            line_size = len(line)

            # Check chunk size
            if current_size + line_size > self.chunk_size and current_section:
                chunk_id = self._generate_id(f"{source}:{len(chunks)}")
                chunks.append(Chunk(
                    id=chunk_id,
                    text=current_section.strip(),
                    metadata={
                        'source': source,
                        'category': category,
                        'header': current_header,
                        'chunk_index': len(chunks)
                    }
                ))

                # Overlap for context
                if self.overlap > 0:
                    overlap_text = current_section[-self.overlap:]
                    current_section = overlap_text + "\n" + line
                    current_size = len(current_section)
                else:
                    current_section = line
                    current_size = line_size
            else:
                if current_section:
                    current_section += "\n" + line
                else:
                    current_section = line
                current_size += line_size + 1

        # Add final chunk
        if current_section.strip():
            chunk_id = self._generate_id(f"{source}:{len(chunks)}")
            chunks.append(Chunk(
                id=chunk_id,
                text=current_section.strip(),
                metadata={
                    'source': source,
                    'category': category,
                    'header': current_header,
                    'chunk_index': len(chunks)
                }
            ))

        return chunks


class RobotDocsProcessor(DocumentProcessor):
    """
    Specialized processor for PARC Robotics documentation.
    Handles code examples, kinematics formulas, and API docs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_categories = [
            'scservo_sdk', 'sms_sts', 'kinematics', 'calibration',
            'examples', 'api', 'tutorials'
        ]

    def process_code_example(
        self,
        code: str,
        filename: str,
        description: str = "",
        language: str = "python"
    ) -> Chunk:
        """Process a code example as a single chunk."""
        chunk_id = self._generate_id(f"code:{filename}")

        # Wrap code in markdown
        full_text = f"```{language}\n{code}\n```"
        if description:
            full_text = f"{description}\n\n{full_text}"

        return Chunk(
            id=chunk_id,
            text=full_text,
            metadata={
                'source': filename,
                'category': 'code_example',
                'language': language,
                'type': 'code'
            }
        )

    def process_api_doc(
        self,
        name: str,
        description: str,
        parameters: List[Dict[str, str]],
        returns: str = "",
        example: str = ""
    ) -> Chunk:
        """Process an API documentation entry."""
        chunk_id = self._generate_id(f"api:{name}")

        text = f"## {name}\n\n{description}\n\n"
        if parameters:
            text += "### Parameters\n\n"
            for p in parameters:
                text += f"- `{p['name']}` ({p.get('type', 'any')}): {p.get('desc', '')}\n"
        if returns:
            text += f"\n### Returns\n\n{returns}\n"
        if example:
            text += f"\n### Example\n\n```python\n{example}\n```\n"

        return Chunk(
            id=chunk_id,
            text=text,
            metadata={
                'source': f"api:{name}",
                'category': 'api_documentation',
                'type': 'api'
            }
        )

    def process_kinematics(
        self,
        name: str,
        description: str,
        formula: str,
        parameters: List[str],
        code_example: str = ""
    ) -> Chunk:
        """Process a kinematics concept."""
        chunk_id = self._generate_id(f"kinematics:{name}")

        text = f"## {name}\n\n{description}\n\n"
        text += f"### Formula\n\n$${formula}$$\n\n"
        text += f"### Parameters\n\n" + ", ".join(parameters) + "\n\n"
        if code_example:
            text += f"### Implementation\n\n```python\n{code_example}\n```\n"

        return Chunk(
            id=chunk_id,
            text=text,
            metadata={
                'source': f"kinematics:{name}",
                'category': 'kinematics',
                'type': 'concept'
            }
        )
