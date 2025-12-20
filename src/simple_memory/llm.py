"""LLM interface for memory processing."""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from .config import LLMConfig, get_config

logger = logging.getLogger(__name__)

# System prompt for memory processing (Chinese)
MEMORY_PROCESSING_PROMPT = """你是一个记忆处理助手。你的任务是：
1. 从给定内容中提取关键信息
2. 将内容整理总结成清晰、简洁的记忆条目
3. 提取描述内容的相关标签

请以JSON格式返回响应，结构如下：
{
    "processed_content": "对关键信息的清晰、简洁的中文总结",
    "tags": ["标签1", "标签2", "标签3"]
}

处理指南：
- processed_content 应聚焦于事实，保持简洁
- 提取3-5个描述主要主题的相关标签
- 保留重要细节，如人名、日期、具体信息等
- 使内容便于后续搜索和检索
- 标签使用中文，保持简短（2-4个字）
- 如果原文是英文，总结和标签也应使用中文
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
            {"role": "user", "content": f"请处理以下内容：\n\n{content}"},
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
            {"role": "user", "content": f"请处理以下内容：\n\n{content}"},
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
