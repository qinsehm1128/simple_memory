"""Memory management core logic."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config, get_config_manager
from .database import SatoriDBManager, Memory, MemoryWithVector, get_db_manager
from .embeddings import EmbeddingProvider, get_embedding_provider
from .llm import LLMProvider, get_llm_provider
from .rerank import RerankProvider, get_rerank_provider
from .code_index import (
    CodeIndexer,
    CodeChunk,
    create_code_memory_content,
    create_code_metadata,
)

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
        db_manager: Optional[SatoriDBManager] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        rerank_provider: Optional[RerankProvider] = None,
        code_indexer: Optional[CodeIndexer] = None,
    ):
        self._db_manager = db_manager
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._rerank_provider = rerank_provider
        self._code_indexer = code_indexer
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
    def db(self) -> SatoriDBManager:
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

    @property
    def reranker(self) -> Optional[RerankProvider]:
        """Get the rerank provider (if configured)."""
        if self._rerank_provider is None:
            self._rerank_provider = get_rerank_provider()
        return self._rerank_provider

    @property
    def code_indexer(self) -> CodeIndexer:
        """Get the code indexer."""
        if self._code_indexer is None:
            self._code_indexer = CodeIndexer()
        return self._code_indexer

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
        use_hybrid: Optional[bool] = None,
        hybrid_alpha: Optional[float] = None,
        use_rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using dual-vector fusion.

        Searches both processed content and original content vectors,
        then fuses the results by averaging similarity scores. Optionally
        uses hybrid search (vector + keyword) and reranking.

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
            use_hybrid: Enable hybrid search (vector + keyword). Uses config value if not specified
            hybrid_alpha: Weight for vector search in hybrid mode (0-1). Uses config value if not specified
            use_rerank: Enable reranking of results. Uses config value if not specified

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
        if use_hybrid is None:
            use_hybrid = config.search.hybrid_enabled
        if hybrid_alpha is None:
            hybrid_alpha = config.search.hybrid_alpha
        if use_rerank is None:
            use_rerank = config.rerank.enabled

        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)

        # Determine initial search limit
        initial_limit = config.search.initial_limit if use_rerank else limit * 3

        if use_hybrid:
            # Use hybrid search combining vector and keyword
            fused_results = await self._hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=initial_limit,
                user_id=user_id,
                tags=tags,
                distance_threshold=distance_threshold,
                min_similarity=min_similarity,
                hybrid_alpha=hybrid_alpha,
            )
        else:
            # Use dual-vector fusion search
            fused_results = await self._dual_vector_search(
                query_embedding=query_embedding,
                limit=initial_limit,
                user_id=user_id,
                tags=tags,
                distance_threshold=distance_threshold,
                min_similarity=min_similarity,
                fusion_weight=fusion_weight,
            )

        # Apply reranking if enabled and reranker is available
        if use_rerank and self.reranker and fused_results:
            fused_results = await self._rerank_results(query, fused_results, limit)
        else:
            fused_results = fused_results[:limit]

        return fused_results

    async def _dual_vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        user_id: Optional[str],
        tags: Optional[List[str]],
        distance_threshold: float,
        min_similarity: float,
        fusion_weight: float,
    ) -> List[Dict[str, Any]]:
        """Perform dual-vector fusion search."""
        # Search processed content vector
        processed_results = self.db.search(
            query_vector=query_embedding,
            limit=limit,
            distance_threshold=distance_threshold,
            user_id=user_id,
            tags=tags,
            vector_column="vector",
        )

        # Search original content vector
        content_results = self.db.search(
            query_vector=query_embedding,
            limit=limit,
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

        return fused_results

    async def _hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int,
        user_id: Optional[str],
        tags: Optional[List[str]],
        distance_threshold: float,
        min_similarity: float,
        hybrid_alpha: float,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword matching."""
        # Use database hybrid search
        results = self.db.hybrid_search(
            query_vector=query_embedding,
            query_text=query,
            limit=limit,
            user_id=user_id,
            tags=tags,
            distance_threshold=distance_threshold,
            alpha=hybrid_alpha,
            initial_limit=limit * 2,
        )

        # Convert to similarity format
        fused_results = []
        for result in results:
            result_copy = result.copy()
            # Calculate similarity from hybrid score
            hybrid_score = result.get("_hybrid_score", 0)
            similarity = round(hybrid_score * 100, 1)
            result_copy["similarity"] = similarity
            result_copy["vector_similarity"] = round(result.get("_vector_similarity", 0) * 100, 1)
            result_copy["keyword_score"] = round(result.get("_keyword_score", 0) * 100, 1)

            if similarity >= min_similarity:
                fused_results.append(result_copy)

        return fused_results

    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Rerank search results using the reranker."""
        if not results:
            return results

        try:
            # Extract document texts for reranking
            documents = [
                r.get("processed_content") or r.get("content", "")
                for r in results
            ]

            # Get rerank scores
            config = get_config()
            top_k = min(limit, config.rerank.top_k)
            rerank_results = await self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Reorder results based on rerank scores
            reranked = []
            for rr in rerank_results:
                if rr.index < len(results):
                    result = results[rr.index].copy()
                    result["rerank_score"] = round(rr.score * 100, 1)
                    reranked.append(result)

            logger.info(f"Reranked {len(results)} results to {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed, returning original results: {e}")
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

    # ========== Code Indexing Methods ==========

    async def index_code_file(
        self,
        file_path: str,
        user_id: str = "default",
        force: bool = False,
    ) -> List[str]:
        """Index a code file and store its chunks as memories.

        Args:
            file_path: Path to the code file
            user_id: User identifier
            force: Force re-indexing even if file hasn't changed

        Returns:
            List of memory IDs for the indexed chunks
        """
        await self.initialize()

        # Check if file should be indexed
        if not self.code_indexer.should_index_file(file_path):
            logger.info(f"Skipping file (not indexable): {file_path}")
            return []

        # Check if re-indexing is needed
        if not force and not self.code_indexer.needs_reindex(file_path):
            logger.info(f"Skipping file (not changed): {file_path}")
            return []

        # Parse and index the file
        chunks = self.code_indexer.index_file(file_path)
        if not chunks:
            return []

        # Delete existing memories for this file
        await self._delete_code_memories(file_path)

        # Create memories from chunks
        memory_ids = []
        for chunk in chunks:
            content = create_code_memory_content(chunk)
            metadata = create_code_metadata(chunk)

            # Add memory without LLM processing (code should be indexed as-is)
            ids = await self.add_memory(
                content=content,
                user_id=user_id,
                metadata=metadata,
                process_with_llm=False,  # Don't process code with LLM
                chunk_long_text=False,  # Already chunked
            )
            memory_ids.extend(ids)

        logger.info(f"Indexed file {file_path}: {len(memory_ids)} chunks")
        return memory_ids

    async def index_code_directory(
        self,
        directory: str,
        user_id: str = "default",
        recursive: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Index all code files in a directory.

        Args:
            directory: Path to the directory
            user_id: User identifier
            recursive: Whether to recursively index subdirectories
            force: Force re-indexing even if files haven't changed

        Returns:
            Dictionary with indexing statistics
        """
        await self.initialize()

        stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "errors": [],
        }

        for file_path, chunks in self.code_indexer.index_directory(directory, recursive):
            try:
                if not force and not self.code_indexer.needs_reindex(file_path):
                    stats["files_skipped"] += 1
                    continue

                # Delete existing memories for this file
                await self._delete_code_memories(file_path)

                # Store chunks as memories
                for chunk in chunks:
                    content = create_code_memory_content(chunk)
                    metadata = create_code_metadata(chunk)

                    await self.add_memory(
                        content=content,
                        user_id=user_id,
                        metadata=metadata,
                        process_with_llm=False,
                        chunk_long_text=False,
                    )
                    stats["total_chunks"] += 1

                stats["files_indexed"] += 1

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                stats["errors"].append({"file": file_path, "error": str(e)})

        logger.info(
            f"Directory indexing complete: {stats['files_indexed']} files, "
            f"{stats['total_chunks']} chunks, {len(stats['errors'])} errors"
        )
        return stats

    async def update_code_files(
        self,
        directory: str,
        user_id: str = "default",
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """Update only changed code files in a directory.

        This performs incremental updates - only re-indexes files that have
        changed since the last indexing.

        Args:
            directory: Path to the directory
            user_id: User identifier
            recursive: Whether to recursively check subdirectories

        Returns:
            Dictionary with update statistics
        """
        await self.initialize()

        changed_files = self.code_indexer.get_changed_files(directory)

        stats = {
            "files_checked": 0,
            "files_updated": 0,
            "total_chunks": 0,
            "errors": [],
        }

        for file_path in changed_files:
            try:
                stats["files_checked"] += 1

                # Delete existing memories for this file
                deleted_count = await self._delete_code_memories(file_path)

                # Re-index the file
                chunks = self.code_indexer.index_file(file_path)

                for chunk in chunks:
                    content = create_code_memory_content(chunk)
                    metadata = create_code_metadata(chunk)

                    await self.add_memory(
                        content=content,
                        user_id=user_id,
                        metadata=metadata,
                        process_with_llm=False,
                        chunk_long_text=False,
                    )
                    stats["total_chunks"] += 1

                stats["files_updated"] += 1
                logger.info(f"Updated file: {file_path} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Error updating {file_path}: {e}")
                stats["errors"].append({"file": file_path, "error": str(e)})

        logger.info(
            f"Update complete: {stats['files_updated']} files updated, "
            f"{stats['total_chunks']} chunks"
        )
        return stats

    async def _delete_code_memories(self, file_path: str) -> int:
        """Delete all memories associated with a code file.

        Args:
            file_path: Path to the code file

        Returns:
            Number of memories deleted
        """
        # Get all memories and filter by file_path in metadata
        memories = await self.get_all_memories(limit=10000)
        deleted_count = 0

        for memory in memories:
            metadata = memory.get("metadata", {})
            if metadata.get("type") == "code" and metadata.get("file_path") == file_path:
                if await self.delete_memory(memory["id"]):
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing chunks for {file_path}")

        return deleted_count

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_path_pattern: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for code in indexed files.

        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language
            file_path_pattern: Filter by file path pattern
            chunk_type: Filter by chunk type (function, class, etc.)

        Returns:
            List of matching code chunks
        """
        await self.initialize()

        # Perform regular search
        results = await self.search(query, limit=limit * 3)

        # Filter for code type
        code_results = []
        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("type") != "code":
                continue

            # Filter by language
            if language and metadata.get("language") != language:
                continue

            # Filter by file path pattern
            if file_path_pattern:
                import fnmatch
                if not fnmatch.fnmatch(metadata.get("file_path", ""), file_path_pattern):
                    continue

            # Filter by chunk type
            if chunk_type and metadata.get("chunk_type") != chunk_type:
                continue

            code_results.append(result)

            if len(code_results) >= limit:
                break

        return code_results

    def get_indexed_files(self) -> Dict[str, Any]:
        """Get information about indexed code files.

        Returns:
            Dictionary with file path -> file info
        """
        return {
            path: {
                "hash": info.hash,
                "size": info.size,
                "language": info.language,
                "chunk_count": info.chunk_count,
                "indexed_at": info.indexed_at,
            }
            for path, info in self.code_indexer.get_indexed_files().items()
        }


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
