"""
node.py
-------
LatticeNode — the atomic unit of the NSL-RAG knowledge lattice.
Every service, database, deployment, and log entry is a LatticeNode.
Nodes are immutable after creation — use LatticeIndex to manage them.

Usage:
    from nsl_rag.lattice.node import LatticeNode
    from nsl_rag.core.types import NodeType
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from nsl_rag.core.types import NodeType, FactStatus


# ── LatticeNode ───────────────────────────────────────────────────────────────


class LatticeNode(BaseModel):
    """
    A single concept node in the NSL-RAG knowledge lattice.

    Every node represents one concept in the domain:
    a service, database, deployment, log entry, or dependency.

    Fields:
        node_id:     Unique identifier. Used for traversal and lookup.
        node_type:   What kind of entity this node represents.
        title:       Short human-readable name.
        summary:     One sentence description — sent to LLM in trace.
        content:     Full factual content — sent to LLM for generation.
        tags:        Symbolic labels for lattice-guided retrieval.
                     These are your FCA attributes.
        children:    List of child node IDs — defines the partial order.
        parents:     List of parent node IDs — reverse index for traversal.
        metadata:    Flexible key-value store for domain-specific facts.
                     Example: {"owner": "payments-team", "port": 8080}
        confidence:  How reliable is this node's content (0.0 to 1.0).
                     1.0 = ground truth, lower = inferred or uncertain.
        status:      Current fact status — set by Judge during reasoning.
        created_at:  Timestamp of node creation.
    """

    node_id: str
    node_type: NodeType
    title: str
    summary: str
    content: str
    tags: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: FactStatus = Field(default=FactStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"frozen": True}

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("node_id")
    @classmethod
    def node_id_must_be_valid(cls, v: str) -> str:
        """
        node_id must be non-empty, lowercase, and use only
        alphanumeric characters and underscores.
        Example valid IDs: payment_service, orders_db, flash_sale_v2
        """
        if not v:
            raise ValueError("node_id cannot be empty")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"node_id '{v}' must contain only alphanumeric characters, "
                "underscores, or hyphens"
            )
        return v.lower()

    @field_validator("tags")
    @classmethod
    def tags_must_be_lowercase(cls, v: list[str]) -> list[str]:
        """Tags are always lowercase for consistent matching."""
        return [tag.lower().strip() for tag in v]

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: float) -> float:
        """Confidence must be between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_root(self) -> bool:
        """A root node has no parents."""
        return len(self.parents) == 0

    @property
    def is_leaf(self) -> bool:
        """A leaf node has no children."""
        return len(self.children) == 0

    @property
    def is_confident(self) -> bool:
        """Returns True if this node's confidence meets the default threshold."""
        return self.confidence >= 0.7

    @property
    def has_tag(self) -> bool:
        """Returns True if this node has at least one tag."""
        return len(self.tags) > 0

    # ── Methods ───────────────────────────────────────────────────────────────

    def matches_tags(self, query_tags: list[str]) -> bool:
        """
        Returns True if ALL query tags are present in this node's tags.
        This is the symbolic AND filter — the core of lattice retrieval.
        A node must satisfy every tag constraint to be retrieved.

        Args:
            query_tags: Tags extracted from the user query.

        Example:
            node.tags = ["payment", "service", "critical"]
            query_tags = ["payment", "service"]
            → True  (all query tags present)

            query_tags = ["payment", "database"]
            → False (database tag not in node)
        """
        return all(tag in self.tags for tag in query_tags)

    def to_context_string(self) -> str:
        """
        Formats node content for LLM context window.
        Used by Generator when assembling the prompt.
        """
        return (
            f"[{self.node_type.value.upper()}] {self.title}\n"
            f"Summary: {self.summary}\n"
            f"Details: {self.content}\n"
            f"Confidence: {self.confidence}"
        )

    def to_trace_entry(self) -> str:
        """
        Formats node as a single line in the reasoning trace.
        Used by OutputFormatter when building {trace} in the output.
        """
        return f"{self.node_type.value.upper()} | {self.title} | {self.summary}"

    def __repr__(self) -> str:
        return (
            f"LatticeNode("
            f"id={self.node_id!r}, "
            f"type={self.node_type.value!r}, "
            f"tags={self.tags}, "
            f"children={self.children})"
        )
