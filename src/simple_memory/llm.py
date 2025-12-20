"""LLM interface for memory processing."""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from .config import LLMConfig, get_config

logger = logging.getLogger(__name__)

# System prompt for memory processing
MEMORY_PROCESSING_PROMPT = """You are a memory processing assistant. Your task is to:
1. Extract key information from the given content
2. Summarize and organize the content into clear, concise memory entries
3. Extract relevant tags that describe the content

Return your response as a JSON object with the following structure:
{
    "processed_content": "A clear, concise summary of the key information",
    "tags": ["tag1", "tag2", "tag3"]
}

Guidelines:
- Keep the processed_content focused and factual
- Extract 3-5 relevant tags that describe the main topics
- Preserve important details, names, dates, and specific information
- Make the content easy to search and retrieve later
- Use consistent tagging conventions (lowercase, singular form)
"""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def process_memory(self, content: str) -> dict:
        """Process memory content and return structured data."""
        pass

    @abstractmethod
    async def chat(self, messages: List[dict]) -> str:
        """Send chat messages and get response."""
        pass


class OpenAILLM(LLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_url = config.api_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model

    async def chat(self, messages: List[dict]) -> str:
        """Send chat messages and get response."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def process_memory(self, content: str) -> dict:
        """Process memory content and return structured data."""
        messages = [
            {"role": "system", "content": MEMORY_PROCESSING_PROMPT},
            {"role": "user", "content": f"Process the following content:\n\n{content}"},
        ]

        response = await self.chat(messages)

        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)
            return {
                "processed_content": result.get("processed_content", content),
                "tags": result.get("tags", []),
            }
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse LLM response as JSON: {response}")
            return {
                "processed_content": content,
                "tags": [],
            }


class OllamaLLM(LLMProvider):
    """Ollama LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self.model = config.model

    async def chat(self, messages: List[dict]) -> str:
        """Send chat messages and get response."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    async def process_memory(self, content: str) -> dict:
        """Process memory content and return structured data."""
        messages = [
            {"role": "system", "content": MEMORY_PROCESSING_PROMPT},
            {"role": "user", "content": f"Process the following content:\n\n{content}"},
        ]

        response = await self.chat(messages)

        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)
            return {
                "processed_content": result.get("processed_content", content),
                "tags": result.get("tags", []),
            }
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse LLM response as JSON: {response}")
            return {
                "processed_content": content,
                "tags": [],
            }


def get_llm_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Get the appropriate LLM provider based on configuration."""
    if config is None:
        config = get_config().llm

    if config.provider == "openai":
        return OpenAILLM(config)
    elif config.provider == "ollama":
        return OllamaLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
