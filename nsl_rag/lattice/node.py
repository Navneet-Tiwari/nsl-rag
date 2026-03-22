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

    def matches_tags(
        self,
        query_tags: list[str],
        min_match: int = 1,
    ) -> bool:
        """
        Returns True if at least min_match query tags are present
        in this node's tags.

        Default min_match=1 preserves backward compatibility.
        Navigator passes config.retrieval.min_tag_match for
        production retrieval.

        Args:
            query_tags: Tags extracted from the user query.
            min_match:  Minimum number of tags that must match.
                        1 = any tag matches (OR logic)
                        len(query_tags) = all tags must match (AND logic)

        Example:
            node.tags = ["payment", "service", "critical"]
            query_tags = ["payment", "service", "gateway"]
            min_match = 2
            → 2 tags match ("payment", "service") → True

            min_match = 3
            → only 2 match → False
        """
        if not query_tags:
            return True
        matches = sum(1 for tag in query_tags if tag in self.tags)
        return matches >= min_match

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
