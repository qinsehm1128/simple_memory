"""Memory management core logic."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_config, get_config_manager
from .database import LanceDBManager, Memory, MemoryWithVector, get_db_manager
from .embeddings import EmbeddingProvider, get_embedding_provider
from .llm import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


def distance_to_similarity(distance: float) -> float:
    """Convert distance to similarity score (0-100%)."""
    # For cosine distance: similarity = 1 - distance/2
    # Distance range is typically 0-2 for cosine
    similarity = max(0.0, min(1.0, 1.0 - distance / 2.0))
    return round(similarity * 100, 1)


class ConfigurationError(Exception):
    """Raised when the system is not properly configured."""

    pass


class MemoryManager:
    """Core memory management class."""

    def __init__(
        self,
        db_manager: Optional[LanceDBManager] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self._db_manager = db_manager
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._initialized = False

    def _check_configuration(self) -> None:
        """Check if the system is properly configured."""
        config = get_config()
        if not config.is_configured():
            raise ConfigurationError(
                "System is not configured. Please configure the LLM and embedding settings "
                "via the web interface at http://localhost:8765/settings before using the MCP."
            )

    @property
    def db(self) -> LanceDBManager:
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
    ) -> str:
        """Add a new memory.

        Args:
            content: The raw content to store
            user_id: User identifier
            metadata: Additional metadata
            process_with_llm: Whether to process content with LLM

        Returns:
            The memory ID
        """
        await self.initialize()

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

        # Generate embedding
        embedding = await self.embeddings.embed(processed_content)

        # Create memory object
        memory = MemoryWithVector(
            content=content,
            processed_content=processed_content,
            metadata=metadata or {},
            tags=tags,
            user_id=user_id,
            vector=embedding,
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
            memories.append(
                {
                    "content": content,
                    "processed_content": processed_content,
                    "tags": tags,
                    "user_id": user_id,
                }
            )

        # Generate embeddings in batch
        embeddings = await self.embeddings.embed_batch(processed_contents)

        # Create memory objects with vectors
        memory_objects = []
        for i, mem_data in enumerate(memories):
            memory = MemoryWithVector(
                content=mem_data["content"],
                processed_content=mem_data["processed_content"],
                metadata={},
                tags=mem_data["tags"],
                user_id=mem_data["user_id"],
                vector=embeddings[i],
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
    ) -> List[Dict[str, Any]]:
        """Search for similar memories.

        Args:
            query: Search query
            limit: Maximum number of results
            user_id: Filter by user ID
            tags: Filter by tags
            distance_threshold: Maximum distance threshold (0.0-2.0, lower = more similar)
                               Uses config value if not specified
            min_similarity: Minimum similarity percentage (0-100) to include in results
                           Uses config value if not specified

        Returns:
            List of matching memories with similarity scores
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

        # Search database with distance threshold
        results = self.db.search(
            query_vector=query_embedding,
            limit=limit,
            distance_threshold=distance_threshold,
            user_id=user_id,
            tags=tags,
        )

        # Add similarity scores and filter by minimum similarity
        filtered_results = []
        for result in results:
            distance = result.get("_distance", 0.0)
            similarity = distance_to_similarity(distance)
            result["similarity"] = similarity

            # Only include results above minimum similarity threshold
            if similarity >= min_similarity:
                filtered_results.append(result)
            else:
                logger.debug(f"Filtered out result with similarity {similarity}% < {min_similarity}%")

        results = filtered_results

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return results[:limit]

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

            # Re-embed
            embedding = await self.embeddings.embed(updates["processed_content"])
            updates["vector"] = embedding

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
