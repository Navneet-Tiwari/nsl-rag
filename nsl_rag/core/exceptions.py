"""
exceptions.py
-------------
Custom exception hierarchy for NSL-RAG.
Every module raises from this hierarchy — never raise generic exceptions.

Usage:
    from nsl_rag.core.exceptions import LatticeError, ConfigError
"""

# ── Base Exception ────────────────────────────────────────────────────────────


class NSLRAGError(Exception):
    """
    Base exception for all NSL-RAG errors.
    Every custom exception inherits from this.
    Allows callers to catch all NSL-RAG errors with a single except clause:
        except NSLRAGError as e: ...
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details: {self.details}"
        return self.message


# ── Config Exceptions ─────────────────────────────────────────────────────────


class ConfigError(NSLRAGError):
    """
    Raised when configuration is missing, invalid, or cannot be loaded.
    Examples:
        - settings.yaml not found
        - Required environment variable missing
        - Invalid config value type
    """


class MissingAPIKeyError(ConfigError):
    """
    Raised when a required API key is not set in .env.
    Subclass of ConfigError — catches with either.
    """


# ── Lattice Exceptions ────────────────────────────────────────────────────────


class LatticeError(NSLRAGError):
    """
    Raised when lattice construction or validation fails.
    Examples:
        - Duplicate node ID
        - Child node references non-existent parent
        - Circular dependency detected
    """


class NodeNotFoundError(LatticeError):
    """
    Raised when a requested node ID does not exist in the lattice index.
    """


class CircularDependencyError(LatticeError):
    """
    Raised when a circular dependency is detected during lattice construction.
    Example: Service A depends on Service B which depends on Service A.
    """


# ── Retrieval Exceptions ──────────────────────────────────────────────────────


class RetrievalError(NSLRAGError):
    """
    Raised when lattice retrieval fails.
    Examples:
        - Intent extraction returns no tags
        - Traversal finds no matching nodes
    """


class IntentExtractionError(RetrievalError):
    """
    Raised when the neural layer fails to extract intent from a query.
    """


class NoNodesFoundError(RetrievalError):
    """
    Raised when lattice traversal finds zero matching nodes for a query.
    """


# ── Reasoning Exceptions ──────────────────────────────────────────────────────


class ReasoningError(NSLRAGError):
    """
    Raised when the Prosecutor / Judge / Auditor pipeline fails.
    Examples:
        - Judge rejects all proposed facts
        - Auditor detects unresolvable contradictions
    """


class AllFactsRejectedError(ReasoningError):
    """
    Raised when the Judge rejects every fact proposed by the Prosecutor.
    System cannot generate a validated answer.
    """


class UnresolvableContradictionError(ReasoningError):
    """
    Raised when the Auditor detects contradictions that cannot be resolved.
    """


# ── Generation Exceptions ─────────────────────────────────────────────────────


class GenerationError(NSLRAGError):
    """
    Raised when LLM generation fails.
    Examples:
        - API rate limit exceeded after all retries
        - LLM returns empty response
        - Response cannot be parsed
    """


class RateLimitError(GenerationError):
    """
    Raised when Gemini API rate limit is exceeded after all retries.
    """


class EmptyResponseError(GenerationError):
    """
    Raised when the LLM returns an empty or unusable response.
    """
