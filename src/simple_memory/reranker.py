"""Reranker model implementations for Simple Memory."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import httpx

from .config import RerankConfig, get_config

logger = logging.getLogger(__name__)


class RerankProvider(ABC):
    """Abstract base class for rerank providers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank documents based on query relevance.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by score descending
        """
        pass


class APIReranker(RerankProvider):
    """API-based reranker (Cohere, Jina, etc.)."""

    def __init__(self, config: RerankConfig):
        self.config = config
        self.api_url = config.api_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.top_k = config.top_k

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using API."""
        if not documents:
            return []

        top_k = top_k or self.top_k

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Detect API type based on URL
            if "cohere" in self.api_url.lower():
                return await self._rerank_cohere(client, query, documents, top_k)
            elif "jina" in self.api_url.lower():
                return await self._rerank_jina(client, query, documents, top_k)
            else:
                # Try generic OpenAI-compatible format
                return await self._rerank_generic(client, query, documents, top_k)

    async def _rerank_cohere(
        self, client: httpx.AsyncClient, query: str, documents: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank using Cohere API."""
        response = await client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append((item["index"], item["relevance_score"]))

        return sorted(results, key=lambda x: x[1], reverse=True)

    async def _rerank_jina(
        self, client: httpx.AsyncClient, query: str, documents: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank using Jina API."""
        response = await client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append((item["index"], item["relevance_score"]))

        return sorted(results, key=lambda x: x[1], reverse=True)

    async def _rerank_generic(
        self, client: httpx.AsyncClient, query: str, documents: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank using generic API format."""
        response = await client.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        data = response.json()

        results = []
        # Try different response formats
        items = data.get("results", data.get("data", []))
        for item in items:
            idx = item.get("index", item.get("idx", 0))
            score = item.get("relevance_score", item.get("score", 0.0))
            results.append((idx, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]


class OllamaReranker(RerankProvider):
    """Ollama-based reranker with auto-download support."""

    def __init__(self, config: RerankConfig):
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self.model = config.ollama_model
        self.top_k = config.top_k
        self._model_ready = False

    async def _ensure_model(self) -> None:
        """Ensure the model is available, download if necessary."""
        if self._model_ready:
            return

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                model_exists = (
                    self.model in model_names
                    or f"{self.model}:latest" in model_names
                    or any(m.startswith(f"{self.model}:") for m in model_names)
                )

                if not model_exists:
                    logger.info(f"Reranker model {self.model} not found, downloading...")
                    await self._pull_model(client)

                self._model_ready = True

            except httpx.RequestError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Please make sure Ollama is running."
                ) from e

    async def _pull_model(self, client: httpx.AsyncClient) -> None:
        """Pull (download) the model from Ollama."""
        import json

        logger.info(f"Pulling reranker model {self.model}...")

        async with client.stream(
            "POST",
            f"{self.host}/api/pull",
            json={"name": self.model},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        if total > 0:
                            progress = (completed / total) * 100
                            logger.info(f"Download progress: {progress:.1f}%")

        logger.info(f"Reranker model {self.model} downloaded successfully")

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using Ollama.

        Note: Ollama doesn't have native rerank support, so we use a workaround
        by generating relevance scores using the model.
        """
        if not documents:
            return []

        await self._ensure_model()
        top_k = top_k or self.top_k

        results = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, doc in enumerate(documents):
                # Use the model to score relevance
                prompt = f"""Rate the relevance of the following document to the query on a scale of 0 to 1.
Only output a single number between 0 and 1, nothing else.

Query: {query}

Document: {doc[:500]}

Relevance score:"""

                try:
                    response = await client.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0},
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Parse score from response
                    score_text = data.get("response", "0").strip()
                    try:
                        score = float(score_text.split()[0])
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    except (ValueError, IndexError):
                        score = 0.0

                    results.append((idx, score))

                except Exception as e:
                    logger.warning(f"Error scoring document {idx}: {e}")
                    results.append((idx, 0.0))

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class DummyReranker(RerankProvider):
    """Dummy reranker that returns documents in original order."""

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Return documents in original order with dummy scores."""
        top_k = top_k or len(documents)
        return [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]


def get_rerank_provider(config: Optional[RerankConfig] = None) -> Optional[RerankProvider]:
    """Get the appropriate rerank provider based on configuration."""
    if config is None:
        config = get_config().rerank

    if not config.enabled:
        return None

    if config.provider == "api":
        if not config.api_key:
            logger.warning("Rerank API key not configured, disabling reranker")
            return None
        return APIReranker(config)
    elif config.provider == "ollama":
        return OllamaReranker(config)
    else:
        return None
