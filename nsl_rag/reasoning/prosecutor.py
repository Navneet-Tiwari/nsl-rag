"""
prosecutor.py
-------------
Prosecutor — proposes candidate facts from retrieved lattice nodes.
First stage of the Prosecutor / Judge / Auditor reasoning pipeline.

The Prosecutor extracts factual claims from node content.
It does NOT validate — it only proposes.
Validation is the Judge's responsibility.

Usage:
    from nsl_rag.reasoning.prosecutor import Prosecutor

    prosecutor = Prosecutor()
    facts = prosecutor.propose(retrieval_result)
"""

from dataclasses import dataclass, field
from datetime import datetime

from nsl_rag.config.config_loader import config
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import FactStatus, NodeType
from nsl_rag.lattice.node import LatticeNode
from nsl_rag.retrieval.navigator import RetrievalResult

log = get_logger(__name__)


# ── Proposed Fact ─────────────────────────────────────────────────────────────


@dataclass
class ProposedFact:
    """
    A single factual claim extracted from a lattice node.
    Passed to the Judge for validation.

    Fields:
        fact_id:     Unique identifier for this fact
        claim:       The factual statement extracted from the node
        source_node: The node this fact was extracted from
        confidence:  Inherited from source node confidence
        status:      Starts PENDING — Judge sets final status
        metadata:    Additional context from node metadata
        created_at:  Timestamp of fact creation
    """

    fact_id: str
    claim: str
    source_node: LatticeNode
    confidence: float = 1.0
    status: FactStatus = FactStatus.PENDING
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def source_id(self) -> str:
        return self.source_node.node_id

    @property
    def source_type(self) -> NodeType:
        return self.source_node.node_type

    def to_context_string(self) -> str:
        """Format for LLM context window."""
        status_label = (
            ""
            if self.status == FactStatus.CONFIDENT
            else f" [{self.status.value.upper()}]"
        )
        return (
            f"[{self.source_type.value.upper()}] "
            f"{self.source_node.title}{status_label}: "
            f"{self.claim}"
        )

    def __repr__(self) -> str:
        return (
            f"ProposedFact("
            f"id={self.fact_id!r}, "
            f"source={self.source_id!r}, "
            f"status={self.status.value!r}, "
            f"claim={self.claim[:50]!r})"
        )


# ── Prosecutor ────────────────────────────────────────────────────────────────


class Prosecutor:
    """
    Extracts candidate facts from retrieved lattice nodes.

    For each node, the Prosecutor generates:
    - A primary fact from the node summary
    - Dependency facts from parent/child relationships
    - Metadata facts from structured node metadata
    - Temporal facts from deployment/log timestamps

    All facts start as PENDING — the Judge evaluates them.
    """

    def __init__(self) -> None:
        self._max_facts = config.reasoning.prosecutor.max_facts_per_node
        log.debug("Prosecutor initialised — max_facts_per_node: %d", self._max_facts)

    # ── Primary Interface ─────────────────────────────────────────────────────

    def propose(
        self,
        retrieval_result: RetrievalResult,
    ) -> list[ProposedFact]:
        """
        Extract candidate facts from all nodes in retrieval result.
        """
        log.info(
            "Prosecutor proposing facts from %d nodes", len(retrieval_result.all_nodes)
        )

        all_facts: list[ProposedFact] = []

        for node in retrieval_result.all_nodes:
            node_facts = self._propose_from_node(node, start_counter=len(all_facts))
            all_facts.extend(node_facts)
            log.debug("Node %s → %d facts proposed", node.node_id, len(node_facts))

        log.info(
            "Prosecutor proposed %d total facts from %d nodes",
            len(all_facts),
            len(retrieval_result.all_nodes),
        )
        return all_facts

    # ── Fact Extraction ───────────────────────────────────────────────────────

    def _propose_from_node(
        self,
        node: LatticeNode,
        start_counter: int,
    ) -> list[ProposedFact]:
        """
        Extract all facts from a single node.
        Each fact gets a globally unique ID based on start_counter.
        """
        facts: list[ProposedFact] = []

        extractors = [
            self._extract_primary_fact,
            self._extract_dependency_facts,
            self._extract_metadata_facts,
            self._extract_temporal_facts,
        ]

        for extractor in extractors:
            if len(facts) >= self._max_facts:
                break
            # Pass current total count as the counter
            # so each new fact gets a unique sequential ID
            new_facts = extractor(node, start_counter + len(facts))
            remaining = self._max_facts - len(facts)
            facts.extend(new_facts[:remaining])

        return facts

    def _extract_primary_fact(
        self,
        node: LatticeNode,
        counter: int,
    ) -> list[ProposedFact]:
        """
        Extract the primary fact — what this node IS and DOES.
        Always produces exactly one fact per node.
        """
        claim = f"{node.title}: {node.summary}. {node.content}"

        return [
            ProposedFact(
                fact_id=f"fact_{counter:04d}",
                claim=claim,
                source_node=node,
                confidence=node.confidence,
                metadata={"extraction_type": "primary"},
            )
        ]

    def _extract_dependency_facts(
        self,
        node: LatticeNode,
        counter: int,
    ) -> list[ProposedFact]:
        """
        Extract dependency relationship facts.
        Each fact gets its own unique sequential ID.
        """
        facts = []

        if node.children:
            children_str = ", ".join(node.children)
            facts.append(
                ProposedFact(
                    fact_id=f"fact_{counter + len(facts):04d}",
                    claim=(f"{node.title} depends on or manages: " f"{children_str}"),
                    source_node=node,
                    confidence=node.confidence,
                    metadata={"extraction_type": "dependency_downstream"},
                )
            )

        if node.parents:
            parents_str = ", ".join(node.parents)
            facts.append(
                ProposedFact(
                    fact_id=f"fact_{counter + len(facts):04d}",
                    claim=(f"{node.title} is depended on by: " f"{parents_str}"),
                    source_node=node,
                    confidence=node.confidence,
                    metadata={"extraction_type": "dependency_upstream"},
                )
            )

        return facts

    def _extract_metadata_facts(
        self,
        node: LatticeNode,
        counter: int,
    ) -> list[ProposedFact]:
        """
        Extract structured facts from node metadata.
        Converts key-value pairs into readable factual claims.
        """
        facts = []

        # Meaningful metadata keys that produce useful facts
        useful_keys = {
            "port": "runs on port",
            "owner": "is owned by team",
            "version": "is at version",
            "max_connections": "has max connections",
            "timeout_ms": "has timeout of",
            "traffic_multiplier": "increased traffic by",
            "cpu_increase_pct": "increased CPU usage by",
            "deployed_at": "was deployed at",
            "crashed_at": "crashed at",
            "model_version": "uses model version",
        }

        for key, description in useful_keys.items():
            if key in node.metadata and len(facts) < 2:
                value = node.metadata[key]
                facts.append(
                    ProposedFact(
                        fact_id=f"fact_{counter + len(facts):04d}",
                        claim=f"{node.title} {description} {value}",
                        source_node=node,
                        confidence=node.confidence,
                        metadata={
                            "extraction_type": "metadata",
                            "metadata_key": key,
                        },
                    )
                )

        return facts

    def _extract_temporal_facts(
        self,
        node: LatticeNode,
        counter: int,
    ) -> list[ProposedFact]:
        """
        Extract time-sensitive facts from deployment and log nodes.
        Temporal ordering is critical for causal reasoning in DevOps.
        """
        if node.node_type not in (NodeType.DEPLOYMENT, NodeType.LOG):
            return []

        facts = []

        if "deployed_at" in node.metadata:
            facts.append(
                ProposedFact(
                    fact_id=f"fact_{counter:04d}",
                    claim=(
                        f"{node.title} was deployed at "
                        f"{node.metadata['deployed_at']} — "
                        f"{node.summary}"
                    ),
                    source_node=node,
                    confidence=node.confidence,
                    metadata={
                        "extraction_type": "temporal",
                        "timestamp": node.metadata["deployed_at"],
                    },
                )
            )

        if "crashed_at" in node.metadata:
            facts.append(
                ProposedFact(
                    fact_id=f"fact_{counter + 1:04d}",
                    claim=(
                        f"{node.title} crashed at " f"{node.metadata['crashed_at']}"
                    ),
                    source_node=node,
                    confidence=node.confidence,
                    metadata={
                        "extraction_type": "temporal",
                        "timestamp": node.metadata["crashed_at"],
                    },
                )
            )

        return facts
