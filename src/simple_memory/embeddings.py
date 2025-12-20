"""Embedding model implementations for Simple Memory."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
import numpy as np

from .config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.api_url = config.api_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.dimensions = config.dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embeddings]

    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.dimensions


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding provider with auto-download support."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self.model = config.ollama_model
        self._dimensions: Optional[int] = None
        self._model_ready = False

    async def _ensure_model(self) -> None:
        """Ensure the model is available, download if necessary."""
        if self._model_ready:
            return

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Check if model exists
            try:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                # Check if our model is in the list (with or without :latest tag)
                model_exists = (
                    self.model in model_names
                    or f"{self.model}:latest" in model_names
                    or any(m.startswith(f"{self.model}:") for m in model_names)
                )

                if not model_exists:
                    logger.info(f"Model {self.model} not found, downloading...")
                    await self._pull_model(client)

                self._model_ready = True

            except httpx.RequestError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Please make sure Ollama is running."
                ) from e

    async def _pull_model(self, client: httpx.AsyncClient) -> None:
        """Pull (download) the model from Ollama."""
        logger.info(f"Pulling model {self.model}...")

        async with client.stream(
            "POST",
            f"{self.host}/api/pull",
            json={"name": self.model},
            timeout=600.0,  # 10 minutes timeout for model download
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        if total > 0:
                            progress = (completed / total) * 100
                            logger.info(f"Download progress: {progress:.1f}%")

        logger.info(f"Model {self.model} downloaded successfully")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        await self._ensure_model()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.host}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                },
            )
            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings", [])
            if embeddings:
                embedding = embeddings[0]
                # Cache dimensions
                if self._dimensions is None:
                    self._dimensions = len(embedding)
                return embedding

            raise RuntimeError("No embedding returned from Ollama")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self._ensure_model()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings", [])
            if embeddings:
                # Cache dimensions
                if self._dimensions is None and len(embeddings) > 0:
                    self._dimensions = len(embeddings[0])
                return embeddings

            raise RuntimeError("No embeddings returned from Ollama")

    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Default dimensions for common models
        model_dimensions = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }

        for model_name, dims in model_dimensions.items():
            if model_name in self.model:
                return dims

        # Default fallback
        return 768


def get_embedding_provider(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
    """Get the appropriate embedding provider based on configuration."""
    if config is None:
        config = get_config().embedding

    if config.provider == "openai":
        return OpenAIEmbedding(config)
    elif config.provider == "ollama":
        return OllamaEmbedding(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
