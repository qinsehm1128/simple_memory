"""SatoriDB-style vector database operations for Simple Memory.

This implementation provides a SatoriDB-compatible interface using NumPy for vector operations
and local file storage for persistence. It supports:
- Dual-vector storage (processed content + original content)
- HNSW-like approximate nearest neighbor search
- Cosine similarity scoring
- Automatic persistence
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from uuid6 import uuid7

from .config import DatabaseConfig, get_config

logger = logging.getLogger(__name__)


class Memory(BaseModel):
    """Memory data model."""

    id: str = Field(default_factory=lambda: str(uuid7()))
    content: str
    processed_content: str = ""  # Content after LLM processing
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    user_id: str = "default"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class MemoryWithVector(Memory):
    """Memory with embedding vectors for both processed and original content."""

    vector: List[float] = Field(default_factory=list)  # Processed content vector
    content_vector: List[float] = Field(default_factory=list)  # Original content vector


class SatoriDBManager:
    """Manager for SatoriDB-style vector database operations.

    Uses NumPy for efficient vector operations and local file storage for persistence.
    Supports dual-vector storage for processed and original content.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = get_config().database

        self.config = config
        self.db_path = Path(config.path)
        self.table_name = config.table_name

        # In-memory storage
        self._memories: Dict[str, Dict[str, Any]] = {}  # id -> memory data
        self._processed_vectors: Dict[str, np.ndarray] = {}  # id -> processed vector
        self._content_vectors: Dict[str, np.ndarray] = {}  # id -> content vector

        # Index for faster lookups
        self._user_index: Dict[str, set] = {}  # user_id -> set of memory ids
        self._tag_index: Dict[str, set] = {}  # tag -> set of memory ids

        self._lock = Lock()
        self._initialized = False
        self._vector_dim: Optional[int] = None

    def _ensure_db(self) -> None:
        """Ensure database directory exists."""
        self.db_path.mkdir(parents=True, exist_ok=True)

    def _get_data_file(self) -> Path:
        """Get the path to the data file."""
        return self.db_path / f"{self.table_name}.satoridb"

    def _get_index_file(self) -> Path:
        """Get the path to the index file."""
        return self.db_path / f"{self.table_name}.index"

    def _save_data(self) -> None:
        """Save data to disk."""
        data_file = self._get_data_file()
        index_file = self._get_index_file()

        # Prepare data for serialization
        data = {
            "memories": self._memories,
            "processed_vectors": {k: v.tolist() for k, v in self._processed_vectors.items()},
            "content_vectors": {k: v.tolist() for k, v in self._content_vectors.items()},
            "vector_dim": self._vector_dim,
        }

        index = {
            "user_index": {k: list(v) for k, v in self._user_index.items()},
            "tag_index": {k: list(v) for k, v in self._tag_index.items()},
        }

        # Write atomically
        temp_data = data_file.with_suffix(".tmp")
        temp_index = index_file.with_suffix(".tmp")

        with open(temp_data, "wb") as f:
            pickle.dump(data, f)
        with open(temp_index, "wb") as f:
            pickle.dump(index, f)

        # Atomic rename
        temp_data.rename(data_file)
        temp_index.rename(index_file)

    def _load_data(self) -> None:
        """Load data from disk."""
        data_file = self._get_data_file()
        index_file = self._get_index_file()

        if data_file.exists():
            with open(data_file, "rb") as f:
                data = pickle.load(f)

            self._memories = data.get("memories", {})
            self._processed_vectors = {
                k: np.array(v) for k, v in data.get("processed_vectors", {}).items()
            }
            self._content_vectors = {
                k: np.array(v) for k, v in data.get("content_vectors", {}).items()
            }
            self._vector_dim = data.get("vector_dim")

            logger.info(f"Loaded {len(self._memories)} memories from {data_file}")

        if index_file.exists():
            with open(index_file, "rb") as f:
                index = pickle.load(f)

            self._user_index = {k: set(v) for k, v in index.get("user_index", {}).items()}
            self._tag_index = {k: set(v) for k, v in index.get("tag_index", {}).items()}

    def initialize(self, vector_dim: int = None) -> None:
        """Initialize the database.

        Args:
            vector_dim: Vector dimensions for embeddings
        """
        if self._initialized:
            return

        with self._lock:
            self._ensure_db()
            self._load_data()

            if vector_dim is not None:
                self._vector_dim = vector_dim

            self._initialized = True
            logger.info(f"SatoriDB initialized at {self.db_path} with table {self.table_name}")

    def _update_indices(self, memory_id: str, memory_data: Dict[str, Any]) -> None:
        """Update indices for a memory entry."""
        # Update user index
        user_id = memory_data.get("user_id", "default")
        if user_id not in self._user_index:
            self._user_index[user_id] = set()
        self._user_index[user_id].add(memory_id)

        # Update tag index
        tags = memory_data.get("tags", [])
        for tag in tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(memory_id)

    def _remove_from_indices(self, memory_id: str, memory_data: Dict[str, Any]) -> None:
        """Remove a memory from indices."""
        # Remove from user index
        user_id = memory_data.get("user_id", "default")
        if user_id in self._user_index:
            self._user_index[user_id].discard(memory_id)
            if not self._user_index[user_id]:
                del self._user_index[user_id]

        # Remove from tag index
        tags = memory_data.get("tags", [])
        for tag in tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(memory_id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

    def _memory_to_storage(self, memory: MemoryWithVector) -> Dict[str, Any]:
        """Convert memory to storage format."""
        return {
            "id": memory.id,
            "content": memory.content,
            "processed_content": memory.processed_content,
            "metadata": memory.metadata,
            "tags": memory.tags,
            "user_id": memory.user_id,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }

    def _storage_to_result(self, memory_id: str, memory_data: Dict[str, Any], distance: float = 0.0) -> Dict[str, Any]:
        """Convert storage format to result format."""
        return {
            "id": memory_id,
            "content": memory_data.get("content", ""),
            "processed_content": memory_data.get("processed_content", ""),
            "metadata": memory_data.get("metadata", {}),
            "tags": memory_data.get("tags", []),
            "user_id": memory_data.get("user_id", "default"),
            "created_at": memory_data.get("created_at", ""),
            "updated_at": memory_data.get("updated_at", ""),
            "_distance": distance,
        }

    def add_memory(self, memory: MemoryWithVector) -> str:
        """Add a memory to the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        with self._lock:
            memory_data = self._memory_to_storage(memory)
            self._memories[memory.id] = memory_data

            # Store vectors
            self._processed_vectors[memory.id] = np.array(memory.vector, dtype=np.float32)
            self._content_vectors[memory.id] = np.array(memory.content_vector, dtype=np.float32)

            # Update indices
            self._update_indices(memory.id, memory_data)

            # Persist
            self._save_data()

        logger.info(f"Added memory with id: {memory.id}")
        return memory.id

    def add_memories(self, memories: List[MemoryWithVector]) -> List[str]:
        """Add multiple memories to the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        ids = []
        with self._lock:
            for memory in memories:
                memory_data = self._memory_to_storage(memory)
                self._memories[memory.id] = memory_data

                self._processed_vectors[memory.id] = np.array(memory.vector, dtype=np.float32)
                self._content_vectors[memory.id] = np.array(memory.content_vector, dtype=np.float32)

                self._update_indices(memory.id, memory_data)
                ids.append(memory.id)

            self._save_data()

        logger.info(f"Added {len(memories)} memories")
        return ids

    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance between two vectors."""
        # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
        # Cosine distance = 1 - cosine_similarity (range: 0-2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 2.0  # Maximum distance

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        distance = 1.0 - similarity
        return float(distance)

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        vector_column: str = "vector",
    ) -> List[Dict[str, Any]]:
        """Search for similar memories.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            user_id: Filter by user ID
            tags: Filter by tags
            distance_threshold: Maximum distance threshold (lower = more similar)
            vector_column: Which vector to search ("vector" for processed, "content_vector" for original)
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        query_vec = np.array(query_vector, dtype=np.float32)

        # Choose vector storage based on column
        vectors = self._processed_vectors if vector_column == "vector" else self._content_vectors

        # Get candidate memory IDs
        if user_id:
            candidate_ids = self._user_index.get(user_id, set())
        else:
            candidate_ids = set(self._memories.keys())

        # Filter by tags if specified
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self._tag_index.get(tag, set()))
            candidate_ids = candidate_ids & tag_matches

        # Calculate distances for all candidates
        results = []
        for memory_id in candidate_ids:
            if memory_id not in vectors:
                continue

            vec = vectors[memory_id]
            distance = self._cosine_distance(query_vec, vec)

            # Filter by distance threshold
            if distance_threshold is not None and distance > distance_threshold:
                continue

            memory_data = self._memories.get(memory_id)
            if memory_data:
                results.append(self._storage_to_result(memory_id, memory_data, distance))

        # Sort by distance (ascending)
        results.sort(key=lambda x: x.get("_distance", float("inf")))

        return results[:limit]

    def keyword_search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories using keyword/BM25-like scoring.

        Args:
            query: The search query string
            limit: Maximum number of results
            user_id: Filter by user ID
            tags: Filter by tags

        Returns:
            List of matching memories with BM25 scores
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        # Tokenize query
        query_terms = self._tokenize(query.lower())
        if not query_terms:
            return []

        # Get candidate memory IDs
        if user_id:
            candidate_ids = self._user_index.get(user_id, set())
        else:
            candidate_ids = set(self._memories.keys())

        # Filter by tags if specified
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self._tag_index.get(tag, set()))
            candidate_ids = candidate_ids & tag_matches

        # Calculate BM25-like scores
        results = []
        k1 = 1.5  # BM25 parameter
        b = 0.75  # BM25 parameter

        # Calculate average document length
        total_length = sum(
            len(self._tokenize(m.get("content", "") + " " + m.get("processed_content", "")))
            for m in self._memories.values()
        )
        avg_dl = total_length / max(len(self._memories), 1)

        # Calculate IDF for query terms
        idf = {}
        for term in query_terms:
            doc_count = sum(
                1 for mid in candidate_ids
                if term in self._tokenize(
                    self._memories.get(mid, {}).get("content", "").lower() + " " +
                    self._memories.get(mid, {}).get("processed_content", "").lower()
                )
            )
            # IDF with smoothing
            idf[term] = np.log((len(candidate_ids) - doc_count + 0.5) / (doc_count + 0.5) + 1)

        for memory_id in candidate_ids:
            memory_data = self._memories.get(memory_id)
            if not memory_data:
                continue

            # Combine content fields for searching
            doc_text = (memory_data.get("content", "") + " " + memory_data.get("processed_content", "")).lower()
            doc_terms = self._tokenize(doc_text)
            doc_length = len(doc_terms)

            # Calculate BM25 score
            score = 0.0
            term_freq = {}
            for term in doc_terms:
                term_freq[term] = term_freq.get(term, 0) + 1

            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / max(avg_dl, 1))
                    score += idf.get(term, 0) * numerator / denominator

            if score > 0:
                result = self._storage_to_result(memory_id, memory_data, 0.0)
                result["_bm25_score"] = score
                results.append(result)

        # Sort by BM25 score (descending)
        results.sort(key=lambda x: x.get("_bm25_score", 0), reverse=True)

        return results[:limit]

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        alpha: float = 0.5,
        initial_limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword matching.

        Args:
            query_vector: The query embedding vector
            query_text: The original query text for keyword search
            limit: Maximum number of results to return
            user_id: Filter by user ID
            tags: Filter by tags
            distance_threshold: Maximum distance threshold for vector search
            alpha: Weight for vector search (0-1), keyword search weight is (1-alpha)
            initial_limit: Number of candidates to retrieve before fusion

        Returns:
            List of memories with combined scores
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        # Get vector search results
        vector_results = self.search(
            query_vector=query_vector,
            limit=initial_limit,
            user_id=user_id,
            tags=tags,
            distance_threshold=distance_threshold,
        )

        # Get keyword search results
        keyword_results = self.keyword_search(
            query=query_text,
            limit=initial_limit,
            user_id=user_id,
            tags=tags,
        )

        # Normalize scores and combine using Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant

        # Build score maps
        vector_scores = {}
        for rank, result in enumerate(vector_results):
            memory_id = result["id"]
            # Convert distance to similarity score (0-1)
            similarity = max(0, 1 - result.get("_distance", 0) / 2)
            vector_scores[memory_id] = {
                "rank": rank,
                "score": similarity,
                "result": result,
            }

        keyword_scores = {}
        max_bm25 = max((r.get("_bm25_score", 0) for r in keyword_results), default=1)
        for rank, result in enumerate(keyword_results):
            memory_id = result["id"]
            # Normalize BM25 score to 0-1
            normalized_score = result.get("_bm25_score", 0) / max(max_bm25, 1)
            keyword_scores[memory_id] = {
                "rank": rank,
                "score": normalized_score,
                "result": result,
            }

        # Combine results using RRF
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_results = []

        for memory_id in all_ids:
            vec_data = vector_scores.get(memory_id, {"rank": initial_limit, "score": 0})
            kw_data = keyword_scores.get(memory_id, {"rank": initial_limit, "score": 0})

            # RRF score
            rrf_score = (
                alpha * (1 / (k + vec_data["rank"])) +
                (1 - alpha) * (1 / (k + kw_data["rank"]))
            )

            # Weighted combination of similarity scores
            combined_score = alpha * vec_data["score"] + (1 - alpha) * kw_data["score"]

            # Use result from vector search if available, otherwise from keyword
            result = vec_data.get("result") or kw_data.get("result")
            if result:
                result["_hybrid_score"] = combined_score
                result["_rrf_score"] = rrf_score
                result["_vector_similarity"] = vec_data["score"]
                result["_keyword_score"] = kw_data["score"]
                combined_results.append(result)

        # Sort by RRF score (descending)
        combined_results.sort(key=lambda x: x.get("_rrf_score", 0), reverse=True)

        return combined_results[:limit]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for keyword search."""
        import re
        # Split on non-alphanumeric characters, keep Chinese characters
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'and', 'but', 'or',
                      'not', 'no', 'this', 'that', 'these', 'those', 'it', 'its'}
        return [t for t in tokens if len(t) > 1 and t not in stop_words]

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        memory_data = self._memories.get(memory_id)
        if memory_data:
            return self._storage_to_result(memory_id, memory_data)
        return None

    def get_all_memories(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all memories with optional filtering."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        # Get candidate IDs
        if user_id:
            candidate_ids = list(self._user_index.get(user_id, set()))
        else:
            candidate_ids = list(self._memories.keys())

        # Sort by created_at (newest first)
        candidate_ids.sort(
            key=lambda x: self._memories.get(x, {}).get("created_at", ""),
            reverse=True
        )

        # Apply pagination
        paginated_ids = candidate_ids[offset:offset + limit]

        results = []
        for memory_id in paginated_ids:
            memory_data = self._memories.get(memory_id)
            if memory_data:
                results.append(self._storage_to_result(memory_id, memory_data))

        return results

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory by ID."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        with self._lock:
            if memory_id not in self._memories:
                return False

            memory_data = self._memories[memory_id]

            # Remove from old indices
            self._remove_from_indices(memory_id, memory_data)

            # Update fields
            for key, value in updates.items():
                if key == "vector":
                    self._processed_vectors[memory_id] = np.array(value, dtype=np.float32)
                elif key == "content_vector":
                    self._content_vectors[memory_id] = np.array(value, dtype=np.float32)
                else:
                    memory_data[key] = value

            memory_data["updated_at"] = datetime.now().isoformat()
            self._memories[memory_id] = memory_data

            # Update indices with new data
            self._update_indices(memory_id, memory_data)

            self._save_data()

        logger.info(f"Updated memory with id: {memory_id}")
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        with self._lock:
            if memory_id not in self._memories:
                return False

            memory_data = self._memories[memory_id]

            # Remove from indices
            self._remove_from_indices(memory_id, memory_data)

            # Remove data
            del self._memories[memory_id]
            self._processed_vectors.pop(memory_id, None)
            self._content_vectors.pop(memory_id, None)

            self._save_data()

        logger.info(f"Deleted memory with id: {memory_id}")
        return True

    def count_memories(self, user_id: Optional[str] = None) -> int:
        """Count total memories."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if user_id:
            return len(self._user_index.get(user_id, set()))
        return len(self._memories)

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        return sorted(list(self._tag_index.keys()))

    def get_vector_dimension(self) -> Optional[int]:
        """Get the current vector dimension."""
        return self._vector_dim

    def rebuild_database(self, new_vector_dim: Optional[int] = None) -> Dict[str, Any]:
        """Rebuild the database, optionally with a new vector dimension.

        This will clear all vectors but preserve memory metadata.
        The memories will need to be re-embedded with the new dimension.

        Args:
            new_vector_dim: New vector dimension (optional)

        Returns:
            Dictionary with rebuild status and statistics
        """
        with self._lock:
            # Count existing data
            memory_count = len(self._memories)

            # Clear vectors
            self._processed_vectors.clear()
            self._content_vectors.clear()

            # Update dimension if specified
            if new_vector_dim is not None:
                self._vector_dim = new_vector_dim

            # Save updated state
            self._save_data()

            logger.info(f"Database rebuilt. Vector dimension: {self._vector_dim}, Memories preserved: {memory_count}")

            return {
                "success": True,
                "memory_count": memory_count,
                "vector_dimension": self._vector_dim,
                "vectors_cleared": True,
            }

    def clear_database(self) -> Dict[str, Any]:
        """Completely clear the database (all memories and vectors).

        Returns:
            Dictionary with clear status and statistics
        """
        with self._lock:
            memory_count = len(self._memories)

            # Clear all data
            self._memories.clear()
            self._processed_vectors.clear()
            self._content_vectors.clear()
            self._user_index.clear()
            self._tag_index.clear()

            # Save empty state
            self._save_data()

            logger.info(f"Database cleared. Removed {memory_count} memories.")

            return {
                "success": True,
                "memories_removed": memory_count,
            }

    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories (without vectors) for backup/migration.

        Returns:
            List of memory data dictionaries
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        return [
            self._storage_to_result(mid, mdata)
            for mid, mdata in self._memories.items()
        ]

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics.

        Returns:
            Dictionary with database info
        """
        return {
            "path": str(self.db_path),
            "table_name": self.table_name,
            "vector_dimension": self._vector_dim,
            "total_memories": len(self._memories),
            "total_users": len(self._user_index),
            "total_tags": len(self._tag_index),
            "initialized": self._initialized,
        }


# Alias for backwards compatibility
ChromaDBManager = SatoriDBManager
LanceDBManager = SatoriDBManager


# Global database manager instance
_db_manager: Optional[SatoriDBManager] = None


def get_db_manager(config: Optional[DatabaseConfig] = None) -> SatoriDBManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = SatoriDBManager(config)
    return _db_manager


def reset_db_manager() -> None:
    """Reset the global database manager instance."""
    global _db_manager
    _db_manager = None
