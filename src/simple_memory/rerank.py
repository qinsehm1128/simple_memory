"""Rerank model implementations for Simple Memory.

Supports various rerank providers for improving search result relevance:
- Cohere Rerank
- Jina Rerank
- BGE Rerank (via API)
- Custom OpenAI-compatible rerank APIs
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import httpx

from .config import get_config

logger = logging.getLogger(__name__)


class RerankResult:
    """Result from reranking operation."""

    def __init__(self, index: int, score: float, text: str = ""):
        self.index = index
        self.score = score
        self.text = text

    def __repr__(self):
        return f"RerankResult(index={self.index}, score={self.score:.4f})"


class RerankProvider(ABC):
    """Abstract base class for rerank providers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            List of RerankResult sorted by relevance score (descending)
        """
        pass

    def get_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__


class CohereRerank(RerankProvider):
    """Cohere Rerank API provider."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.cohere.ai/v1/rerank"

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "return_documents": False,
            }
            if top_k:
                payload["top_n"] = top_k

            response = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(RerankResult(
                    index=item["index"],
                    score=item["relevance_score"],
                    text=documents[item["index"]] if item["index"] < len(documents) else "",
                ))

            return results


class JinaRerank(RerankProvider):
    """Jina AI Rerank API provider."""

    def __init__(self, api_key: str, model: str = "jina-reranker-v2-base-multilingual"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.jina.ai/v1/rerank"

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }
            if top_k:
                payload["top_n"] = top_k

            response = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(RerankResult(
                    index=item["index"],
                    score=item["relevance_score"],
                    text=documents[item["index"]] if item["index"] < len(documents) else "",
                ))

            return results


class BGERerank(RerankProvider):
    """BGE Rerank via API (self-hosted or cloud)."""

    def __init__(self, api_url: str, api_key: str = "", model: str = "bge-reranker-v2-m3"):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Try different API formats
            endpoints = [
                (f"{self.api_url}/rerank", {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(documents),
                }),
                (f"{self.api_url}/v1/rerank", {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(documents),
                }),
                (f"{self.api_url}/api/rerank", {
                    "query": query,
                    "passages": documents,
                    "top_k": top_k or len(documents),
                }),
            ]

            for endpoint, payload in endpoints:
                try:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return self._parse_response(data, documents)
                except Exception:
                    continue

            raise RuntimeError(f"Failed to connect to BGE Rerank API at {self.api_url}")

    def _parse_response(self, data: dict, documents: List[str]) -> List[RerankResult]:
        """Parse various BGE API response formats."""
        results = []

        # Format 1: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        if "results" in data:
            for item in data["results"]:
                idx = item.get("index", 0)
                score = item.get("relevance_score", item.get("score", 0))
                results.append(RerankResult(
                    index=idx,
                    score=score,
                    text=documents[idx] if idx < len(documents) else "",
                ))
            return results

        # Format 2: {"scores": [[0.9, 0.8, 0.7, ...]]}
        if "scores" in data:
            scores = data["scores"]
            if isinstance(scores[0], list):
                scores = scores[0]
            for idx, score in enumerate(scores):
                results.append(RerankResult(
                    index=idx,
                    score=float(score),
                    text=documents[idx] if idx < len(documents) else "",
                ))
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        # Format 3: List of scores directly
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (int, float)):
                    score = float(item)
                else:
                    score = item.get("score", item.get("relevance_score", 0))
                results.append(RerankResult(
                    index=idx,
                    score=score,
                    text=documents[idx] if idx < len(documents) else "",
                ))
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        return results


class CustomRerank(RerankProvider):
    """Custom rerank provider for OpenAI-compatible or custom APIs."""

    def __init__(
        self,
        api_url: str,
        api_key: str = "",
        model: str = "rerank",
        request_format: str = "openai",
    ):
        """
        Args:
            api_url: Base URL for the rerank API
            api_key: API key for authentication
            model: Model name to use
            request_format: Request format ("openai", "cohere", "jina", "simple")
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.request_format = request_format

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Build request based on format
            if self.request_format == "cohere":
                endpoint = f"{self.api_url}/rerank"
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                }
                if top_k:
                    payload["top_n"] = top_k
            elif self.request_format == "jina":
                endpoint = f"{self.api_url}/rerank"
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                }
                if top_k:
                    payload["top_n"] = top_k
            elif self.request_format == "simple":
                endpoint = f"{self.api_url}/rerank"
                payload = {
                    "query": query,
                    "texts": documents,
                    "top_k": top_k or len(documents),
                }
            else:  # openai-like format
                endpoint = f"{self.api_url}/v1/rerank"
                payload = {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(documents),
                }

            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            return self._parse_response(data, documents)

    def _parse_response(self, data: dict, documents: List[str]) -> List[RerankResult]:
        """Parse various response formats."""
        results = []

        # Try different response formats
        items = data.get("results", data.get("data", data.get("rankings", [])))

        if items:
            for item in items:
                if isinstance(item, dict):
                    idx = item.get("index", item.get("corpus_id", 0))
                    score = item.get("relevance_score", item.get("score", 0))
                else:
                    continue
                results.append(RerankResult(
                    index=idx,
                    score=score,
                    text=documents[idx] if idx < len(documents) else "",
                ))
            return results

        # Try scores array format
        scores = data.get("scores", [])
        if scores:
            if isinstance(scores[0], list):
                scores = scores[0]
            for idx, score in enumerate(scores):
                results.append(RerankResult(
                    index=idx,
                    score=float(score),
                    text=documents[idx] if idx < len(documents) else "",
                ))
            results.sort(key=lambda x: x.score, reverse=True)

        return results


class NoOpRerank(RerankProvider):
    """No-op reranker that returns documents in original order."""

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        results = []
        for idx, doc in enumerate(documents):
            results.append(RerankResult(
                index=idx,
                score=1.0 - (idx * 0.01),  # Slightly decreasing scores
                text=doc,
            ))
        if top_k:
            results = results[:top_k]
        return results


def get_rerank_provider(config: Optional[dict] = None) -> Optional[RerankProvider]:
    """Get the appropriate rerank provider based on configuration.

    Args:
        config: Rerank configuration dict, or None to use global config

    Returns:
        RerankProvider instance, or None if reranking is disabled
    """
    if config is None:
        app_config = get_config()
        if not hasattr(app_config, "rerank"):
            return None
        config = app_config.rerank.model_dump() if hasattr(app_config.rerank, "model_dump") else {}

    if not config or not config.get("enabled", False):
        return None

    provider = config.get("provider", "custom")

    if provider == "cohere":
        return CohereRerank(
            api_key=config.get("api_key", ""),
            model=config.get("model", "rerank-english-v3.0"),
        )
    elif provider == "jina":
        return JinaRerank(
            api_key=config.get("api_key", ""),
            model=config.get("model", "jina-reranker-v2-base-multilingual"),
        )
    elif provider == "bge":
        return BGERerank(
            api_url=config.get("api_url", "http://localhost:8080"),
            api_key=config.get("api_key", ""),
            model=config.get("model", "bge-reranker-v2-m3"),
        )
    elif provider == "custom":
        return CustomRerank(
            api_url=config.get("api_url", "http://localhost:8080"),
            api_key=config.get("api_key", ""),
            model=config.get("model", "rerank"),
            request_format=config.get("request_format", "openai"),
        )
    elif provider == "none":
        return NoOpRerank()
    else:
        logger.warning(f"Unknown rerank provider: {provider}")
        return None
