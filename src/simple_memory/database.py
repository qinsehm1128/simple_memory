"""ChromaDB database operations for Simple Memory."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
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


class ChromaDBManager:
    """Manager for ChromaDB operations.

    Uses two collections for dual-vector storage:
    - {table_name}_processed: for processed content embeddings
    - {table_name}_content: for original content embeddings
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = get_config().database

        self.config = config
        self.db_path = Path(config.path)
        self.table_name = config.table_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._processed_collection = None
        self._content_collection = None
        self._initialized = False

    def _ensure_db(self) -> chromadb.ClientAPI:
        """Ensure database connection is established."""
        if self._client is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        return self._client

    def initialize(self, vector_dim: int = None) -> None:
        """Initialize the database collections.

        Args:
            vector_dim: Vector dimensions (not strictly required for ChromaDB)
        """
        if self._initialized:
            return

        client = self._ensure_db()

        # Create/get two collections for dual-vector storage
        # Using cosine distance for similarity search
        self._processed_collection = client.get_or_create_collection(
            name=f"{self.table_name}_processed",
            metadata={"hnsw:space": "cosine"}
        )

        self._content_collection = client.get_or_create_collection(
            name=f"{self.table_name}_content",
            metadata={"hnsw:space": "cosine"}
        )

        self._initialized = True
        logger.info(f"ChromaDB initialized with collections: {self.table_name}_processed, {self.table_name}_content")

    def _get_collections(self):
        """Get both collections."""
        if not self._initialized:
            raise RuntimeError(
                "Database not initialized. Call initialize() first."
            )
        return self._processed_collection, self._content_collection

    def _memory_to_metadata(self, memory: MemoryWithVector) -> Dict[str, Any]:
        """Convert memory to ChromaDB metadata format."""
        return {
            "content": memory.content,
            "processed_content": memory.processed_content,
            "metadata_json": json.dumps(memory.metadata, ensure_ascii=False),
            "tags_json": json.dumps(memory.tags, ensure_ascii=False),
            "user_id": memory.user_id,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }

    def _metadata_to_result(self, id: str, metadata: Dict[str, Any], distance: float = 0.0) -> Dict[str, Any]:
        """Convert ChromaDB metadata back to result format."""
        return {
            "id": id,
            "content": metadata.get("content", ""),
            "processed_content": metadata.get("processed_content", ""),
            "metadata": json.loads(metadata.get("metadata_json", "{}")),
            "tags": json.loads(metadata.get("tags_json", "[]")),
            "user_id": metadata.get("user_id", "default"),
            "created_at": metadata.get("created_at", ""),
            "updated_at": metadata.get("updated_at", ""),
            "_distance": distance,
        }

    def add_memory(self, memory: MemoryWithVector) -> str:
        """Add a memory to the database."""
        processed_col, content_col = self._get_collections()

        metadata = self._memory_to_metadata(memory)

        # Add to processed collection (with processed content embedding)
        processed_col.add(
            ids=[memory.id],
            embeddings=[memory.vector],
            metadatas=[metadata],
            documents=[memory.processed_content],
        )

        # Add to content collection (with original content embedding)
        content_col.add(
            ids=[memory.id],
            embeddings=[memory.content_vector],
            metadatas=[metadata],
            documents=[memory.content],
        )

        logger.info(f"Added memory with id: {memory.id}")
        return memory.id

    def add_memories(self, memories: List[MemoryWithVector]) -> List[str]:
        """Add multiple memories to the database."""
        processed_col, content_col = self._get_collections()

        ids = [m.id for m in memories]
        processed_embeddings = [m.vector for m in memories]
        content_embeddings = [m.content_vector for m in memories]
        metadatas = [self._memory_to_metadata(m) for m in memories]
        processed_documents = [m.processed_content for m in memories]
        content_documents = [m.content for m in memories]

        # Add to processed collection
        processed_col.add(
            ids=ids,
            embeddings=processed_embeddings,
            metadatas=metadatas,
            documents=processed_documents,
        )

        # Add to content collection
        content_col.add(
            ids=ids,
            embeddings=content_embeddings,
            metadatas=metadatas,
            documents=content_documents,
        )

        logger.info(f"Added {len(memories)} memories")
        return ids

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
            tags: Filter by tags (not directly supported, filtered post-query)
            distance_threshold: Maximum distance threshold (lower = more similar)
            vector_column: Which vector to search ("vector" for processed, "content_vector" for original)
        """
        processed_col, content_col = self._get_collections()

        # Choose collection based on vector_column
        collection = processed_col if vector_column == "vector" else content_col

        # Build where clause for filtering
        where = None
        if user_id:
            where = {"user_id": user_id}

        # Query the collection
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
            include=["metadatas", "distances"],
        )

        # Process results
        output = []
        if results and results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0

                # Filter by distance threshold
                if distance_threshold is not None and distance > distance_threshold:
                    continue

                result = self._metadata_to_result(id, metadata, distance)

                # Filter by tags if specified
                if tags:
                    result_tags = result.get("tags", [])
                    if not any(tag in result_tags for tag in tags):
                        continue

                output.append(result)

        return output

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        processed_col, _ = self._get_collections()

        results = processed_col.get(
            ids=[memory_id],
            include=["metadatas"],
        )

        if results and results["ids"] and len(results["ids"]) > 0:
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            return self._metadata_to_result(memory_id, metadata)

        return None

    def get_all_memories(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all memories with optional filtering."""
        processed_col, _ = self._get_collections()

        # Build where clause
        where = None
        if user_id:
            where = {"user_id": user_id}

        # ChromaDB doesn't have direct offset support, so we get more and slice
        results = processed_col.get(
            where=where,
            limit=limit + offset,
            include=["metadatas"],
        )

        output = []
        if results and results["ids"]:
            ids = results["ids"]
            metadatas = results["metadatas"] if results["metadatas"] else []

            # Apply offset
            start_idx = offset
            end_idx = min(offset + limit, len(ids))

            for i in range(start_idx, end_idx):
                if i < len(ids):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    output.append(self._metadata_to_result(ids[i], metadata))

        return output

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory by ID."""
        processed_col, content_col = self._get_collections()

        # Get existing memory
        existing = self.get_memory(memory_id)
        if not existing:
            return False

        # Merge updates
        for key, value in updates.items():
            existing[key] = value

        existing["updated_at"] = datetime.now().isoformat()

        # Prepare metadata
        metadata = {
            "content": existing.get("content", ""),
            "processed_content": existing.get("processed_content", ""),
            "metadata_json": json.dumps(existing.get("metadata", {}), ensure_ascii=False),
            "tags_json": json.dumps(existing.get("tags", []), ensure_ascii=False),
            "user_id": existing.get("user_id", "default"),
            "created_at": existing.get("created_at", ""),
            "updated_at": existing.get("updated_at", ""),
        }

        # Update both collections
        if "vector" in updates:
            processed_col.update(
                ids=[memory_id],
                embeddings=[updates["vector"]],
                metadatas=[metadata],
                documents=[existing.get("processed_content", "")],
            )
        else:
            processed_col.update(
                ids=[memory_id],
                metadatas=[metadata],
                documents=[existing.get("processed_content", "")],
            )

        if "content_vector" in updates:
            content_col.update(
                ids=[memory_id],
                embeddings=[updates["content_vector"]],
                metadatas=[metadata],
                documents=[existing.get("content", "")],
            )
        else:
            content_col.update(
                ids=[memory_id],
                metadatas=[metadata],
                documents=[existing.get("content", "")],
            )

        logger.info(f"Updated memory with id: {memory_id}")
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        processed_col, content_col = self._get_collections()

        # Delete from both collections
        processed_col.delete(ids=[memory_id])
        content_col.delete(ids=[memory_id])

        logger.info(f"Deleted memory with id: {memory_id}")
        return True

    def count_memories(self, user_id: Optional[str] = None) -> int:
        """Count total memories."""
        processed_col, _ = self._get_collections()

        if user_id:
            results = processed_col.get(
                where={"user_id": user_id},
                include=[],
            )
            return len(results["ids"]) if results["ids"] else 0
        else:
            return processed_col.count()

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        processed_col, _ = self._get_collections()

        results = processed_col.get(include=["metadatas"])

        all_tags = set()
        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                tags_json = metadata.get("tags_json", "[]")
                try:
                    tags = json.loads(tags_json)
                    all_tags.update(tags)
                except json.JSONDecodeError:
                    pass

        return sorted(list(all_tags))


# Aliases for backwards compatibility
LanceDBManager = ChromaDBManager


# Global database manager instance
_db_manager: Optional[ChromaDBManager] = None


def get_db_manager(config: Optional[DatabaseConfig] = None) -> ChromaDBManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = ChromaDBManager(config)
    return _db_manager
