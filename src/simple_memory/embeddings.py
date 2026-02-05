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
        self.auth_type = getattr(config, 'auth_type', 'bearer')
        self.auth_header = getattr(config, 'auth_header', 'Authorization')

    def _get_headers(self) -> dict:
        """Get request headers based on auth type."""
        headers = {"Content-Type": "application/json"}

        if not self.api_key or self.auth_type == "none":
            return headers

        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type == "direct":
            # 直接携带密钥，不带 Bearer 前缀
            headers["Authorization"] = self.api_key
        elif self.auth_type == "api_key":
            headers["api-key"] = self.api_key
        elif self.auth_type == "custom":
            headers[self.auth_header] = self.api_key

        return headers

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_url}/embeddings",
                headers=self._get_headers(),
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
        # Use configured dimensions if available, otherwise auto-detect
        self._dimensions: Optional[int] = config.dimensions if config.dimensions > 0 else None
        self._model_ready = False
        self._use_legacy_api: Optional[bool] = None  # None = not determined yet

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

    async def _embed_with_api(
        self, client: httpx.AsyncClient, text_or_texts, is_batch: bool = False
    ) -> List[float] | List[List[float]]:
        """Try embedding with both new and legacy Ollama API."""
        # Try new API first if we haven't determined which to use
        if self._use_legacy_api is None or self._use_legacy_api is False:
            try:
                response = await client.post(
                    f"{self.host}/api/embed",
                    json={"model": self.model, "input": text_or_texts},
                )
                if response.status_code == 200:
                    self._use_legacy_api = False
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        if self._dimensions is None:
                            self._dimensions = len(embeddings[0]) if embeddings else None
                        return embeddings if is_batch else embeddings[0]
            except Exception:
                pass

        # Fallback to legacy API
        self._use_legacy_api = True
        if is_batch:
            # Legacy API doesn't support batch, so we need to do one at a time
            results = []
            for text in text_or_texts:
                response = await client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])
                if embedding:
                    if self._dimensions is None:
                        self._dimensions = len(embedding)
                    results.append(embedding)
            return results
        else:
            response = await client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text_or_texts},
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if embedding:
                if self._dimensions is None:
                    self._dimensions = len(embedding)
                return embedding

        raise RuntimeError("No embedding returned from Ollama")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        await self._ensure_model()

        async with httpx.AsyncClient(timeout=60.0) as client:
            return await self._embed_with_api(client, text, is_batch=False)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self._ensure_model()

        async with httpx.AsyncClient(timeout=120.0) as client:
            return await self._embed_with_api(client, texts, is_batch=True)

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


class RemarkEmbedding(EmbeddingProvider):
    """Remark embedding provider.

    Supports Remark's embedding API for generating text embeddings.
    The API is compatible with OpenAI-style embedding endpoints.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.api_url = config.remark_api_url.rstrip("/")
        self.api_key = config.remark_api_key
        self.model = config.remark_model
        self._dimensions: Optional[int] = config.dimensions if config.dimensions > 0 else None

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Supports multiple API formats:
        1. OpenAI-compatible: POST /embeddings with {"input": [...], "model": "..."}
        2. Simple format: POST /embed with {"texts": [...]}
        3. Alternative: POST /v1/embeddings (OpenAI v1 API)
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Try OpenAI-compatible format first
            endpoints_to_try = [
                (f"{self.api_url}/v1/embeddings", {"input": texts, "model": self.model}),
                (f"{self.api_url}/embeddings", {"input": texts, "model": self.model}),
                (f"{self.api_url}/embed", {"texts": texts, "model": self.model}),
                (f"{self.api_url}/api/embed", {"input": texts, "model": self.model}),
            ]

            last_error = None
            for endpoint, payload in endpoints_to_try:
                try:
                    response = await client.post(endpoint, headers=headers, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        embeddings = self._parse_response(data)
                        if embeddings:
                            # Auto-detect dimensions
                            if self._dimensions is None and embeddings:
                                self._dimensions = len(embeddings[0])
                            return embeddings

                except httpx.RequestError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue

            # If all attempts failed, raise the last error
            if last_error:
                raise RuntimeError(f"Failed to get embeddings from Remark API: {last_error}")
            raise RuntimeError("Failed to get embeddings from Remark API: No valid response")

    def _parse_response(self, data: dict) -> List[List[float]]:
        """Parse embedding response from various API formats."""
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in data:
            embeddings = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [e.get("embedding", e.get("values", [])) for e in embeddings]

        # Simple format: {"embeddings": [[...], [...]]}
        if "embeddings" in data:
            return data["embeddings"]

        # Alternative: {"vectors": [[...], [...]]}
        if "vectors" in data:
            return data["vectors"]

        # Single embedding: {"embedding": [...]}
        if "embedding" in data:
            return [data["embedding"]]

        return []

    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is not None:
            return self._dimensions

        # Default dimensions for common Remark models
        model_dimensions = {
            "remark-embed": 1024,
            "remark-embed-small": 512,
            "remark-embed-large": 2048,
        }

        for model_name, dims in model_dimensions.items():
            if model_name in self.model:
                return dims

        # Default fallback
        return 1024


def get_embedding_provider(config: Optional[EmbeddingConfig] = None) -> EmbeddingProvider:
    """Get the appropriate embedding provider based on configuration."""
    if config is None:
        config = get_config().embedding

    if config.provider == "openai":
        return OpenAIEmbedding(config)
    elif config.provider == "ollama":
        return OllamaEmbedding(config)
    elif config.provider == "remark":
        return RemarkEmbedding(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
