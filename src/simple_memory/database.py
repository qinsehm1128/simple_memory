"""LanceDB database operations for Simple Memory."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa
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


class LanceDBManager:
    """Manager for LanceDB operations."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = get_config().database

        self.config = config
        self.db_path = Path(config.path)
        self.table_name = config.table_name
        self._db: Optional[lancedb.DBConnection] = None
        self._table = None
        self._vector_dim: Optional[int] = None

    def _ensure_db(self) -> lancedb.DBConnection:
        """Ensure database connection is established."""
        if self._db is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_schema(self, vector_dim: int) -> pa.Schema:
        """Get PyArrow schema for the memories table."""
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("processed_content", pa.string()),
                pa.field("metadata", pa.string()),  # JSON string
                pa.field("tags", pa.list_(pa.string())),
                pa.field("user_id", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # Processed content vector
                pa.field("content_vector", pa.list_(pa.float32(), vector_dim)),  # Original content vector
            ]
        )

    def initialize(self, vector_dim: int) -> None:
        """Initialize the database with the given vector dimensions."""
        self._vector_dim = vector_dim
        db = self._ensure_db()

        # Check if table exists
        existing_tables = db.table_names()
        if self.table_name not in existing_tables:
            logger.info(f"Creating table {self.table_name} with vector dimension {vector_dim}")
            # Create empty table with schema
            schema = self._get_schema(vector_dim)
            self._table = db.create_table(self.table_name, schema=schema)
        else:
            self._table = db.open_table(self.table_name)

    def _get_table(self):
        """Get the memories table."""
        if self._table is None:
            db = self._ensure_db()
            if self.table_name in db.table_names():
                self._table = db.open_table(self.table_name)
            else:
                raise RuntimeError(
                    "Database not initialized. Call initialize() first with vector dimensions."
                )
        return self._table

    def add_memory(self, memory: MemoryWithVector) -> str:
        """Add a memory to the database."""
        import json

        table = self._get_table()

        data = {
            "id": memory.id,
            "content": memory.content,
            "processed_content": memory.processed_content,
            "metadata": json.dumps(memory.metadata, ensure_ascii=False),
            "tags": memory.tags,
            "user_id": memory.user_id,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "vector": memory.vector,
            "content_vector": memory.content_vector,
        }

        table.add([data])
        logger.info(f"Added memory with id: {memory.id}")
        return memory.id

    def add_memories(self, memories: List[MemoryWithVector]) -> List[str]:
        """Add multiple memories to the database."""
        import json

        table = self._get_table()

        data = []
        for memory in memories:
            data.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "processed_content": memory.processed_content,
                    "metadata": json.dumps(memory.metadata, ensure_ascii=False),
                    "tags": memory.tags,
                    "user_id": memory.user_id,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "vector": memory.vector,
                    "content_vector": memory.content_vector,
                }
            )

        table.add(data)
        ids = [m.id for m in memories]
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
            tags: Filter by tags
            distance_threshold: Maximum distance threshold (lower = more similar)
                               Default is 1.0 for cosine distance
            vector_column: Which vector column to search ("vector" or "content_vector")
        """
        import json

        table = self._get_table()

        # Build query - search on specified vector column
        query = table.search(query_vector, vector_column_name=vector_column).limit(limit)

        # Apply filters
        filters = []
        if user_id:
            filters.append(f"user_id = '{user_id}'")
        if tags:
            # Filter by any of the tags
            tag_filters = [f"array_contains(tags, '{tag}')" for tag in tags]
            filters.append(f"({' OR '.join(tag_filters)})")

        if filters:
            query = query.where(" AND ".join(filters))

        results = query.to_list()

        # Filter by distance threshold if specified
        if distance_threshold is not None:
            results = [r for r in results if r.get("_distance", float("inf")) <= distance_threshold]

        # Parse metadata JSON
        for result in results:
            if isinstance(result.get("metadata"), str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    result["metadata"] = {}

        return results

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        import json

        table = self._get_table()
        results = table.search().where(f"id = '{memory_id}'").limit(1).to_list()

        if results:
            result = results[0]
            if isinstance(result.get("metadata"), str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    result["metadata"] = {}
            return result
        return None

    def get_all_memories(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all memories with optional filtering."""
        import json

        table = self._get_table()

        # LanceDB doesn't support offset directly, we'll handle it differently
        query = table.search()

        if user_id:
            query = query.where(f"user_id = '{user_id}'")

        # Get more results and slice
        results = query.limit(limit + offset).to_list()

        # Apply offset
        results = results[offset : offset + limit]

        # Parse metadata JSON
        for result in results:
            if isinstance(result.get("metadata"), str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    result["metadata"] = {}

        return results

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory by ID."""
        import json

        table = self._get_table()

        # Get existing memory
        existing = self.get_memory(memory_id)
        if not existing:
            return False

        # Merge updates
        for key, value in updates.items():
            existing[key] = value

        existing["updated_at"] = datetime.now().isoformat()

        # Convert metadata back to JSON string
        if isinstance(existing.get("metadata"), dict):
            existing["metadata"] = json.dumps(existing["metadata"], ensure_ascii=False)

        # Delete old and add new (LanceDB doesn't support in-place updates well)
        self.delete_memory(memory_id)
        table.add([existing])

        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        table = self._get_table()
        table.delete(f"id = '{memory_id}'")
        logger.info(f"Deleted memory with id: {memory_id}")
        return True

    def count_memories(self, user_id: Optional[str] = None) -> int:
        """Count total memories."""
        table = self._get_table()

        if user_id:
            results = table.search().where(f"user_id = '{user_id}'").to_list()
        else:
            results = table.search().to_list()

        return len(results)

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        table = self._get_table()
        results = table.search().to_list()

        all_tags = set()
        for result in results:
            tags = result.get("tags", [])
            if tags:
                all_tags.update(tags)

        return sorted(list(all_tags))


# Global database manager instance
_db_manager: Optional[LanceDBManager] = None


def get_db_manager(config: Optional[DatabaseConfig] = None) -> LanceDBManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = LanceDBManager(config)
    return _db_manager
