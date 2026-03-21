"""
types.py
--------
Shared type definitions, enums, and constants for NSL-RAG.
Imported by every module that needs type safety.
Never import from other nsl_rag modules here — this is the base layer.

Usage:
    from nsl_rag.core.types import QueryType, FactStatus, NodeType
"""

from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────


class QueryType(str, Enum):
    """
    Classification of incoming user queries.
    Determined by IntentExtractor during the neural layer pass.
    Drives reasoning strictness and traversal strategy.
    """

    CAUSAL_TRACE = "causal_trace"  # Why did X fail?
    DEPENDENCY_MAP = "dependency_map"  # What does X depend on?
    STATUS_CHECK = "status_check"  # Is X healthy?
    IMPACT_ANALYSIS = "impact_analysis"  # What is affected by X failing?
    GENERAL = "general"  # Catch-all for unclassified queries


class NodeType(str, Enum):
    """
    Classification of nodes in the lattice.
    Determines what rules the Judge applies during validation.
    """

    SERVICE = "service"  # A microservice or application
    DATABASE = "database"  # A data store
    DEPLOYMENT = "deployment"  # A deployment event
    LOG = "log"  # A log entry or event
    DEPENDENCY = "dependency"  # A dependency relationship
    ROOT = "root"  # Top-level lattice root node


class FactStatus(str, Enum):
    """
    Status assigned to each fact by the Judge.
    Determines whether a fact reaches the LLM.
    """

    CONFIDENT = "confident"  # Fact is validated — passes to LLM
    UNCERTAIN = "uncertain"  # Fact exists but confidence is low — flagged
    INVALID = "invalid"  # Fact violates lattice constraints — rejected
    PENDING = "pending"  # Fact not yet evaluated


class Environment(str, Enum):
    """
    Deployment environment.
    Loaded from NSL_RAG_ENV in .env.
    """

    DEVELOPMENT = "development"
    PRODUCTION = "production"


# ── Output Format ─────────────────────────────────────────────────────────────


class OutputFormat(str, Enum):
    """
    Format of the final system output.
    Structured is default — returns full {answer, trace, confidence, flags}.
    Plain returns answer text only — for simple integrations.
    """

    STRUCTURED = "structured"
    PLAIN = "plain"


# ── Constants ─────────────────────────────────────────────────────────────────


class LatticeConstants:
    """
    Fixed constants for lattice operations.
    These are not tunable — they are structural invariants.
    """

    ROOT_NODE_ID = "root"
    MAX_TRAVERSAL_DEPTH = 20  # Hard ceiling — prevents infinite loops
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0


class SystemConstants:
    """
    System-wide constants.
    """

    PROJECT_NAME = "nsl-rag"
    VERSION = "0.1.0"
    DEFAULT_ENCODING = "utf-8"
