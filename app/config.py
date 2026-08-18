"""Application configuration using pydantic-settings."""

from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DASHSCOPE_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    # Model Selection
    LLM_MODEL_PLANNER: str = "gpt-4o"
    LLM_MODEL_RESEARCHER: str = "gpt-4o"
    LLM_MODEL_WRITER: str = "gpt-4o"
    LLM_MODEL_REVIEWER: str = "gpt-4o"

    # Search API
    SEARCH_API_PROVIDER: str = "tavily"
    SEARCH_BACKENDS: str = "tavily,duckduckgo,github"
    TAVILY_API_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    GITHUB_TOKEN: Optional[str] = None
    SEARCH_MOCK_FALLBACK: bool = False

    # Storage
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_DB_PATH: str = "./data/chroma_db"

    # Tools
    ENABLE_PYTHON_TOOL: bool = False

    # Browser
    BROWSER_USE_PLAYWRIGHT: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Execution
    RESEARCH_PARALLELISM: int = 3
    CONTEXT_MAX_CHARS: int = 60000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/research.db"

    # Auth
    JWT_SECRET: str = "dev-secret-change-in-production"
    ENABLE_AUTH: bool = False

    # Hybrid Search
    HYBRID_SEARCH_ENABLED: bool = False
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    RERANKER_ENABLED: bool = False
    RERANKER_PROVIDER: str = "dashscope"
    RERANKER_API_KEY: Optional[str] = None
    RERANKER_MODEL: str = "gte-rerank"
    RERANKER_API_BASE: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    CANDIDATE_TOP_K: int = 20
    FINAL_TOP_K: int = 5

    # Observability (optional)
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = None

    # Workspace
    WORKSPACE_ROOT: str = "./data/workspaces"
    UPLOAD_MAX_FILES: int = 10
    UPLOAD_MAX_BYTES: int = 20 * 1024 * 1024
    UPLOAD_ALLOWED_EXTS: str = ".pdf,.md,.txt,.csv,.json,.docx"

    @model_validator(mode="after")
    def _check_production_security(self) -> "Settings":
        if self.ENABLE_AUTH and self.JWT_SECRET == "dev-secret-change-in-production":
            raise ValueError(
                "ENABLE_AUTH=true 时必须设置非默认的 JWT_SECRET（生产环境安全要求）"
            )
        return self


settings = Settings()
