"""Memory management core logic."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config, get_config_manager
from .database import ChromaDBManager, Memory, MemoryWithVector, get_db_manager
from .embeddings import EmbeddingProvider, get_embedding_provider
from .llm import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

# Default chunk settings
DEFAULT_CHUNK_SIZE = 1000  # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # Overlap between chunks


def distance_to_similarity(distance: float) -> float:
    """Convert distance to similarity score (0-100%)."""
    # For cosine distance: similarity = 1 - distance/2
    # Distance range is typically 0-2 for cosine
    similarity = max(0.0, min(1.0, 1.0 - distance / 2.0))
    return round(similarity * 100, 1)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split long text into overlapping chunks.

    Uses sentence boundaries when possible for cleaner splits.

    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    # Split by sentences first
    sentence_pattern = r'(?<=[。！？.!?])\s*'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence is longer than chunk_size, split by chunk_size
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Split long sentence
            for i in range(0, len(sentence), chunk_size - overlap):
                chunk = sentence[i : i + chunk_size]
                if chunk:
                    chunks.append(chunk)
        elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Start new chunk with overlap
            if current_chunk:
                chunks.append(current_chunk)
                # Create overlap from last part of current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class ConfigurationError(Exception):
    """Raised when the system is not properly configured."""

    pass


class MemoryManager:
    """Core memory management class."""

    def __init__(
        self,
        db_manager: Optional[ChromaDBManager] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self._db_manager = db_manager
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._initialized = False

    def _check_configuration(self) -> None:
        """Check if the system is properly configured."""
        # Force reload config to get latest settings
        config_manager = get_config_manager()
        config_manager._config = None  # Clear cache
        config = config_manager.load()

        if not config.is_configured():
            raise ConfigurationError(
                "系统未配置。请通过 Web 界面配置 LLM 和嵌入模型设置：http://localhost:8765/settings"
            )

    @property
    def db(self) -> ChromaDBManager:
        """Get the database manager."""
        if self._db_manager is None:
            self._db_manager = get_db_manager()
        return self._db_manager

    @property
    def embeddings(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    @property
    def llm(self) -> LLMProvider:
        """Get the LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return

        self._check_configuration()

        # Get vector dimensions from embedding provider
        vector_dim = self.embeddings.get_dimensions()

        # Initialize database
        self.db.initialize(vector_dim)
        self._initialized = True

        logger.info(f"Memory system initialized with vector dimension: {vector_dim}")

    async def add_memory(
        self,
        content: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        process_with_llm: bool = True,
        chunk_long_text: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> List[str]:
        """Add a new memory (or multiple if content is long and chunking is enabled).

        Args:
            content: The raw content to store
            user_id: User identifier
            metadata: Additional metadata
            process_with_llm: Whether to process content with LLM
            chunk_long_text: Whether to split long text into chunks
            chunk_size: Maximum characters per chunk (if chunking)

        Returns:
            List of memory IDs (one if not chunked, multiple if chunked)
        """
        await self.initialize()

        # Check if we need to chunk the content
        if chunk_long_text and len(content) > chunk_size:
            chunks = chunk_text(content, chunk_size)
            logger.info(f"Content chunked into {len(chunks)} parts")

            memory_ids = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["is_chunk"] = True

                ids = await self._add_single_memory(
                    content=chunk,
                    user_id=user_id,
                    metadata=chunk_metadata,
                    process_with_llm=process_with_llm,
                )
                memory_ids.append(ids)

            return memory_ids
        else:
            memory_id = await self._add_single_memory(
                content=content,
                user_id=user_id,
                metadata=metadata,
                process_with_llm=process_with_llm,
            )
            return [memory_id]

    async def _add_single_memory(
        self,
        content: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        process_with_llm: bool = True,
    ) -> str:
        """Add a single memory entry.

        Args:
            content: The raw content to store
            user_id: User identifier
            metadata: Additional metadata
            process_with_llm: Whether to process content with LLM

        Returns:
            The memory ID
        """
        # Process with LLM if enabled
        processed_content = content
        tags = []

        if process_with_llm:
            try:
                result = await self.llm.process_memory(content)
                processed_content = result.get("processed_content", content)
                tags = result.get("tags", [])
            except Exception as e:
                logger.warning(f"LLM processing failed, using raw content: {e}")
                processed_content = content

        # Generate embeddings for both processed and original content
        processed_embedding = await self.embeddings.embed(processed_content)
        content_embedding = await self.embeddings.embed(content)

        # Create memory object with both vectors
        memory = MemoryWithVector(
            content=content,
            processed_content=processed_content,
            metadata=metadata or {},
            tags=tags,
            user_id=user_id,
            vector=processed_embedding,
            content_vector=content_embedding,
        )

        # Store in database
        memory_id = self.db.add_memory(memory)

        logger.info(f"Added memory: {memory_id}")
        return memory_id

    async def add_memories(
        self,
        contents: List[str],
        user_id: str = "default",
        process_with_llm: bool = True,
    ) -> List[str]:
        """Add multiple memories.

        Args:
            contents: List of raw contents to store
            user_id: User identifier
            process_with_llm: Whether to process content with LLM

        Returns:
            List of memory IDs
        """
        await self.initialize()

        memories = []
        processed_contents = []
        raw_contents = []

        for content in contents:
            processed_content = content
            tags = []

            if process_with_llm:
                try:
                    result = await self.llm.process_memory(content)
                    processed_content = result.get("processed_content", content)
                    tags = result.get("tags", [])
                except Exception as e:
                    logger.warning(f"LLM processing failed for content, using raw: {e}")

            processed_contents.append(processed_content)
            raw_contents.append(content)
            memories.append(
                {
                    "content": content,
                    "processed_content": processed_content,
                    "tags": tags,
                    "user_id": user_id,
                }
            )

        # Generate embeddings in batch for both processed and raw content
        processed_embeddings = await self.embeddings.embed_batch(processed_contents)
        content_embeddings = await self.embeddings.embed_batch(raw_contents)

        # Create memory objects with both vectors
        memory_objects = []
        for i, mem_data in enumerate(memories):
            memory = MemoryWithVector(
                content=mem_data["content"],
                processed_content=mem_data["processed_content"],
                metadata={},
                tags=mem_data["tags"],
                user_id=mem_data["user_id"],
                vector=processed_embeddings[i],
                content_vector=content_embeddings[i],
            )
            memory_objects.append(memory)

        # Store in database
        memory_ids = self.db.add_memories(memory_objects)

        logger.info(f"Added {len(memory_ids)} memories")
        return memory_ids

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        min_similarity: Optional[float] = None,
        fusion_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using dual-vector fusion.

        Searches both processed content and original content vectors,
        then fuses the results by averaging similarity scores.

        Args:
            query: Search query
            limit: Maximum number of results
            user_id: Filter by user ID
            tags: Filter by tags
            distance_threshold: Maximum distance threshold (0.0-2.0, lower = more similar)
                               Uses config value if not specified
            min_similarity: Minimum similarity percentage (0-100) to include in results
                           Uses config value if not specified
            fusion_weight: Weight for processed content similarity (0-1).
                          0 = only use original content, 1 = only use processed content
                          Default 0.5 = equal weight for both

        Returns:
            List of matching memories with fused similarity scores
        """
        await self.initialize()

        # Use config defaults if not specified
        config = get_config()
        if distance_threshold is None:
            distance_threshold = config.search.distance_threshold
        if min_similarity is None:
            min_similarity = config.search.min_similarity

        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)

        # Search both vector columns with more results for fusion
        search_limit = limit * 3  # Get more results for better fusion

        # Search processed content vector
        processed_results = self.db.search(
            query_vector=query_embedding,
            limit=search_limit,
            distance_threshold=distance_threshold,
            user_id=user_id,
            tags=tags,
            vector_column="vector",
        )

        # Search original content vector
        content_results = self.db.search(
            query_vector=query_embedding,
            limit=search_limit,
            distance_threshold=distance_threshold,
            user_id=user_id,
            tags=tags,
            vector_column="content_vector",
        )

        # Build similarity maps by ID
        processed_scores = {}
        for result in processed_results:
            mem_id = result.get("id")
            distance = result.get("_distance", 2.0)
            similarity = distance_to_similarity(distance)
            processed_scores[mem_id] = {
                "similarity": similarity,
                "data": result,
            }

        content_scores = {}
        for result in content_results:
            mem_id = result.get("id")
            distance = result.get("_distance", 2.0)
            similarity = distance_to_similarity(distance)
            content_scores[mem_id] = {
                "similarity": similarity,
                "data": result,
            }

        # Fuse results: combine both sets of IDs
        all_ids = set(processed_scores.keys()) | set(content_scores.keys())

        fused_results = []
        for mem_id in all_ids:
            # Get scores from both searches (default to 0 if not found)
            proc_info = processed_scores.get(mem_id, {"similarity": 0, "data": None})
            cont_info = content_scores.get(mem_id, {"similarity": 0, "data": None})

            proc_sim = proc_info["similarity"]
            cont_sim = cont_info["similarity"]

            # Calculate fused similarity score (weighted average)
            fused_similarity = (proc_sim * fusion_weight) + (cont_sim * (1 - fusion_weight))

            # Use data from whichever search found it (prefer processed if both)
            result_data = proc_info["data"] or cont_info["data"]
            if result_data:
                result_data = result_data.copy()
                result_data["similarity"] = round(fused_similarity, 1)
                result_data["processed_similarity"] = proc_sim
                result_data["content_similarity"] = cont_sim

                # Only include if fused similarity is above threshold
                if fused_similarity >= min_similarity:
                    fused_results.append(result_data)
                else:
                    logger.debug(
                        f"Filtered out result with fused similarity {fused_similarity:.1f}% < {min_similarity}%"
                    )

        # Sort by fused similarity (highest first)
        fused_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return fused_results[:limit]

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        await self.initialize()
        return self.db.get_memory(memory_id)

    async def get_all_memories(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all memories with optional filtering."""
        await self.initialize()
        return self.db.get_all_memories(user_id=user_id, limit=limit, offset=offset)

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update an existing memory."""
        await self.initialize()

        updates = {}
        if content is not None:
            updates["content"] = content
            # Re-process and re-embed
            try:
                result = await self.llm.process_memory(content)
                updates["processed_content"] = result.get("processed_content", content)
                updates["tags"] = result.get("tags", [])
            except Exception as e:
                logger.warning(f"LLM processing failed: {e}")
                updates["processed_content"] = content

            # Re-embed both vectors
            processed_embedding = await self.embeddings.embed(updates["processed_content"])
            content_embedding = await self.embeddings.embed(content)
            updates["vector"] = processed_embedding
            updates["content_vector"] = content_embedding

        if metadata is not None:
            updates["metadata"] = metadata

        if tags is not None:
            updates["tags"] = tags

        return self.db.update_memory(memory_id, updates)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        await self.initialize()
        return self.db.delete_memory(memory_id)

    async def count_memories(self, user_id: Optional[str] = None) -> int:
        """Count total memories."""
        await self.initialize()
        return self.db.count_memories(user_id)

    async def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        await self.initialize()
        return self.db.get_all_tags()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager instance."""
    global _memory_manager
    _memory_manager = None
