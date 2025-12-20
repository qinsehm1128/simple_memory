"""Reranker model implementations for Simple Memory."""

import asyncio
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


class LocalReranker(RerankProvider):
    """Local cross-encoder reranker using sentence-transformers.

    Downloads and runs cross-encoder models locally for reranking.
    Supports models like BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2, etc.
    """

    def __init__(self, config: RerankConfig):
        self.config = config
        self.model_name = config.local_model
        self.device = config.device
        self.top_k = config.top_k
        self._model = None
        self._model_loading = False

    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _load_model(self):
        """Load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for local reranking. "
                "Install it with: pip install sentence-transformers"
            )

        device = self._get_device()
        logger.info(f"Loading reranker model '{self.model_name}' on device '{device}'...")

        try:
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Reranker model '{self.model_name}' loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reranker model '{self.model_name}': {e}. "
                "Make sure the model name is correct and you have internet access "
                "for the first download."
            ) from e

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Rerank using local cross-encoder model."""
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        # Prepare query-document pairs
        pairs = [[query, doc[:1000]] for doc in documents]  # Limit doc length

        # Run prediction in thread pool
        def predict():
            scores = self._model.predict(pairs)
            return scores

        scores = await loop.run_in_executor(None, predict)

        # Create results with indices and scores
        results = [(i, float(score)) for i, score in enumerate(scores)]

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
    elif config.provider == "local":
        return LocalReranker(config)
    else:
        return None
