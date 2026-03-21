"""
config_loader.py
----------------
Loads, validates, and exposes configuration for NSL-RAG.
Single access point for all config across the system.
Fails fast on missing or invalid configuration.

Usage:
    from nsl_rag.config.config_loader import config
    model = config.llm.model
    api_key = config.api_key
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nsl_rag.core.exceptions import ConfigError, MissingAPIKeyError
from nsl_rag.core.logger import get_logger

log = get_logger(__name__)


# ── Config Models ─────────────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    name: str
    version: str
    environment: str = "development"


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 1000
    max_retries: int = 3
    retry_delay_seconds: int = 5


class LatticeConfig(BaseModel):
    max_depth: int = 10
    max_nodes_per_query: int = 5
    confidence_threshold: float = 0.7


class RetrievalConfig(BaseModel):
    max_tags_per_query: int = 5
    min_tag_match: int = 1


class ProsecutorConfig(BaseModel):
    max_facts_per_node: int = 5


class JudgeConfig(BaseModel):
    strict_mode: bool = True


class AuditorConfig(BaseModel):
    contradiction_threshold: float = 0.8


class ReasoningConfig(BaseModel):
    prosecutor: ProsecutorConfig = Field(default_factory=ProsecutorConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    auditor: AuditorConfig = Field(default_factory=AuditorConfig)


class GenerationConfig(BaseModel):
    max_context_nodes: int = 5
    include_trace: bool = True
    include_confidence: bool = True


class NaiveRAGConfig(BaseModel):
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50


class BenchmarkConfig(BaseModel):
    test_queries_path: str = "tests/fixtures/sample_queries.py"
    output_path: str = "assets/benchmark_results.json"


class EvaluationConfig(BaseModel):
    naive_rag: NaiveRAGConfig = Field(default_factory=NaiveRAGConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class LogFileConfig(BaseModel):
    enabled: bool = False
    path: str = "logs/nsl_rag.log"


class LoggingConfig(BaseModel):
    level: str = "DEBUG"
    format: str = "console"
    file: LogFileConfig = Field(default_factory=LogFileConfig)


class Settings(BaseModel):
    """
    Root settings model.
    Mirrors the structure of settings.yaml exactly.
    Validated by Pydantic on load — fails fast on bad config.
    """

    project: ProjectConfig
    llm: LLMConfig
    lattice: LatticeConfig
    retrieval: RetrievalConfig
    reasoning: ReasoningConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


# ── Loader ────────────────────────────────────────────────────────────────────


class ConfigLoader:
    """
    Loads settings.yaml and .env, validates them, and exposes
    a single config object to the rest of the system.
    """

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._api_key: str | None = None

    def load(
        self,
        settings_path: Path | None = None,
        env_path: Path | None = None,
    ) -> None:
        """
        Load and validate all configuration.
        Call once at application startup.

        Args:
            settings_path: Path to settings.yaml. Defaults to package location.
            env_path:      Path to .env file. Defaults to project root.
        """
        self._load_env(env_path)
        self._load_settings(settings_path)
        log.info("Configuration loaded successfully")

    def _load_env(self, env_path: Path | None) -> None:
        """Load environment variables from .env file."""
        path = env_path or Path(__file__).parents[2] / ".env"

        if not path.exists():
            log.warning(".env file not found at: %s", path)
            log.warning("Falling back to system environment variables")
        else:
            load_dotenv(path)
            log.debug(".env loaded from: %s", path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not set", details={"expected_in": str(path)}
            )

        self._api_key = api_key
        log.debug("GEMINI_API_KEY loaded successfully")

    def _load_settings(self, settings_path: Path | None) -> None:
        """Load and validate settings.yaml."""
        path = settings_path or Path(__file__).parent / "settings.yaml"

        if not path.exists():
            raise ConfigError(
                "settings.yaml not found", details={"expected_at": str(path)}
            )

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        try:
            self._settings = Settings(**raw)
            log.debug("settings.yaml validated successfully")
        except Exception as e:
            raise ConfigError(
                "settings.yaml validation failed", details={"error": str(e)}
            ) from e

    # ── Public Properties ─────────────────────────────────────────────────────

    @property
    def settings(self) -> Settings:
        """Return validated settings. Raises if not loaded yet."""
        if self._settings is None:
            raise ConfigError("Config not loaded. Call config.load() first.")
        return self._settings

    @property
    def api_key(self) -> str:
        """Return Gemini API key. Raises if not loaded yet."""
        if self._api_key is None:
            raise MissingAPIKeyError("API key not loaded. Call config.load() first.")
        return self._api_key

    # ── Convenience Accessors ─────────────────────────────────────────────────

    @property
    def llm(self) -> LLMConfig:
        return self.settings.llm

    @property
    def lattice(self) -> LatticeConfig:
        return self.settings.lattice

    @property
    def retrieval(self) -> RetrievalConfig:
        return self.settings.retrieval

    @property
    def reasoning(self) -> ReasoningConfig:
        return self.settings.reasoning

    @property
    def generation(self) -> GenerationConfig:
        return self.settings.generation

    @property
    def evaluation(self) -> EvaluationConfig:
        return self.settings.evaluation

    @property
    def logging_config(self) -> LoggingConfig:
        return self.settings.logging


# ── Singleton ─────────────────────────────────────────────────────────────────

# Single shared instance used across the entire system.
# Import this directly — do not instantiate ConfigLoader yourself.
config = ConfigLoader()
