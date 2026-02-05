"""Configuration management for Simple Memory."""

import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration for memory processing."""

    provider: Literal["openai", "ollama"] = "openai"
    api_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    # Authentication type: bearer, api_key, custom, none
    auth_type: Literal["bearer", "api_key", "custom", "none"] = "bearer"
    # Custom header name (used when auth_type is "custom")
    auth_header: str = "Authorization"
    # Ollama specific
    ollama_host: str = "http://localhost:11434"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: Literal["openai", "ollama", "remark"] = "openai"
    api_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    # Authentication type: bearer, api_key, custom, none
    auth_type: Literal["bearer", "api_key", "custom", "none"] = "bearer"
    # Custom header name (used when auth_type is "custom")
    auth_header: str = "Authorization"
    # Ollama specific
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    # Remark specific
    remark_api_url: str = "http://localhost:8080"
    remark_api_key: str = ""
    remark_model: str = "remark-embed"
    # Embedding dimensions
    dimensions: int = 1536


class RerankConfig(BaseModel):
    """Rerank model configuration."""

    enabled: bool = False
    provider: Literal["cohere", "jina", "bge", "custom", "none"] = "custom"
    api_url: str = "http://localhost:8080"
    api_key: str = ""
    model: str = "rerank"
    # For custom provider
    request_format: Literal["openai", "cohere", "jina", "simple"] = "openai"
    # Rerank settings
    top_k: int = 10  # Number of documents to return after reranking


class SearchConfig(BaseModel):
    """Search configuration."""

    min_similarity: float = 50.0  # Minimum similarity percentage (0-100)
    distance_threshold: float = 1.0  # Maximum distance threshold
    # Hybrid search settings
    hybrid_enabled: bool = False
    hybrid_alpha: float = 0.5  # Weight for vector search (0-1), keyword is 1-alpha
    # Initial retrieval count before reranking
    initial_limit: int = 50


class CodeIndexConfig(BaseModel):
    """Code indexing configuration."""

    enabled: bool = False
    # Supported file extensions
    extensions: List[str] = Field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".vue", ".svelte", ".md", ".txt", ".json",
        ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss",
    ])
    # Directories to ignore
    ignore_dirs: List[str] = Field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "dist", "build", ".next", ".nuxt", "target", "bin", "obj",
        ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    ])
    # Max file size in bytes (default 1MB)
    max_file_size: int = 1048576
    # Chunk settings for code
    chunk_size: int = 1500
    chunk_overlap: int = 200
    # Whether to extract code structure (functions, classes, etc.)
    extract_structure: bool = True


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = "./data/satoridb"
    table_name: str = "memories"


class WebConfig(BaseModel):
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 8765
    debug: bool = False


class AppConfig(BaseModel):
    """Application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    code_index: CodeIndexConfig = Field(default_factory=CodeIndexConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    configured: bool = False

    def is_configured(self) -> bool:
        """Check if the application is properly configured."""
        # Check LLM configuration
        if self.llm.provider == "openai":
            if not self.llm.api_key:
                return False
        elif self.llm.provider == "ollama":
            if not self.llm.ollama_host:
                return False

        # Check embedding configuration
        if self.embedding.provider == "openai":
            if not self.embedding.api_key:
                return False
        elif self.embedding.provider == "ollama":
            if not self.embedding.ollama_host:
                return False
        elif self.embedding.provider == "remark":
            if not self.embedding.remark_api_url:
                return False

        return True


class ConfigManager:
    """Manage application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to user's home directory
            config_dir = Path.home() / ".simple_memory"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.config_path = config_dir / "config.json"
        else:
            self.config_path = Path(config_path)

        self._config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        """Load configuration from file."""
        if self._config is not None:
            return self._config

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._config = AppConfig(**data)
            except (json.JSONDecodeError, ValueError):
                self._config = AppConfig()
        else:
            self._config = AppConfig()

        return self._config

    def save(self, config: AppConfig) -> None:
        """Save configuration to file."""
        self._config = config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)

    def update(self, **kwargs) -> AppConfig:
        """Update configuration with new values."""
        config = self.load()
        config_dict = config.model_dump()

        # Deep update
        for key, value in kwargs.items():
            if key in config_dict and isinstance(value, dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value

        new_config = AppConfig(**config_dict)
        new_config.configured = new_config.is_configured()
        self.save(new_config)
        return new_config

    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.load()

    def reset(self) -> AppConfig:
        """Reset configuration to defaults."""
        self._config = AppConfig()
        self.save(self._config)
        return self._config


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> AppConfig:
    """Get current application configuration."""
    return get_config_manager().get_config()
