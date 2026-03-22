"""
navigator.py
------------
LatticeNavigator — orchestrates intent extraction and lattice traversal.
Single entry point for the full retrieval pipeline.

Takes a natural language query and returns relevant LatticeNodes.
Connects the neural layer (IntentExtractor) to the
symbolic layer (LatticeTraversal).

Usage:
    from nsl_rag.retrieval.navigator import LatticeNavigator

    navigator = LatticeNavigator(index)
    result = navigator.retrieve("Why is the payment service failing?")
    print(result.nodes)
    print(result.intent)
    print(result.trace)
"""

from dataclasses import dataclass, field

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import NoNodesFoundError, RetrievalError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import QueryType
from nsl_rag.lattice.index import LatticeIndex
from nsl_rag.lattice.node import LatticeNode
from nsl_rag.lattice.traversal import LatticeTraversal
from nsl_rag.retrieval.intent import IntentExtractor, IntentResult

log = get_logger(__name__)


# ── Retrieval Result ──────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """
    Complete result from the retrieval pipeline.
    Passed to the reasoning layer (Prosecutor/Judge/Auditor).

    Fields:
        nodes:            Retrieved lattice nodes — ordered general to specific
        intent:           Extracted intent from the query
        dependency_chain: Upward traversal from target entity if available
        trace:            Human readable retrieval trace for debugging
        query:            Original natural language query
    """

    nodes: list[LatticeNode]
    intent: IntentResult
    dependency_chain: list[LatticeNode] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    query: str = ""

    @property
    def all_nodes(self) -> list[LatticeNode]:
        """
        Returns combined unique nodes from both retrieval
        and dependency chain. Used by Prosecutor for fact extraction.
        """
        seen = set()
        combined = []
        for node in self.nodes + self.dependency_chain:
            if node.node_id not in seen:
                seen.add(node.node_id)
                combined.append(node)
        return combined

    @property
    def node_ids(self) -> list[str]:
        """Returns IDs of all retrieved nodes."""
        return [n.node_id for n in self.all_nodes]


# ── Lattice Navigator ─────────────────────────────────────────────────────────


class LatticeNavigator:
    """
    Orchestrates the full NSL-RAG retrieval pipeline.

    Pipeline:
        1. Extract intent from natural language query
        2. Retrieve nodes via lattice traversal
        3. Retrieve dependency chain if target entity known
        4. Build retrieval trace for explainability
        5. Return RetrievalResult

    This is the only class the reasoning layer needs to interact with.
    It hides the complexity of intent extraction and traversal.
    """

    def __init__(self, index: LatticeIndex) -> None:
        self._index = index
        self._traversal = LatticeTraversal(index)
        self._extractor = IntentExtractor()
        log.debug("LatticeNavigator initialised")

    # ── Primary Interface ─────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full retrieval pipeline — query in, RetrievalResult out.

        Args:
            query: Natural language query from the user.

        Returns:
            RetrievalResult containing nodes, intent, and trace.

        Raises:
            RetrievalError: If retrieval fails at any stage.
        """
        log.info("=" * 50)
        log.info("Navigator retrieving: %r", query)

        try:
            # Step 1 — Extract intent
            intent = self._extractor.extract(query)
            log.info(
                "Intent: tags=%s type=%s entity=%s",
                intent.tags,
                intent.query_type.value,
                intent.target_entity,
            )

            # Step 2 — Retrieve nodes via lattice traversal
            nodes = self._retrieve_nodes(intent)

            # Step 3 — Retrieve dependency chain
            chain = self._retrieve_chain(intent)

            # Step 4 — Build trace
            trace = self._build_trace(intent, nodes, chain)

            result = RetrievalResult(
                nodes=nodes,
                intent=intent,
                dependency_chain=chain,
                trace=trace,
                query=query,
            )

            log.info(
                "Retrieval complete — %d nodes, %d chain nodes, %d total unique",
                len(nodes),
                len(chain),
                len(result.all_nodes),
            )
            log.info("Retrieved: %s", result.node_ids)

            return result

        except NoNodesFoundError as e:
            raise RetrievalError(
                f"No nodes found for query: {query}",
                details={"query": query, "error": str(e)},
            ) from e

        except Exception as e:
            raise RetrievalError(
                f"Retrieval failed: {e}", details={"query": query, "error": str(e)}
            ) from e

    # ── Retrieval Steps ───────────────────────────────────────────────────────

    def _retrieve_nodes(self, intent: IntentResult) -> list[LatticeNode]:
        """
        Retrieve nodes from the lattice using extracted tags.
        Uses target entity as direct entry point if available.
        """
        return self._traversal.retrieve(
            query_tags=intent.tags,
            target_node_id=intent.target_entity,
        )

    def _retrieve_chain(
        self,
        intent: IntentResult,
    ) -> list[LatticeNode]:
        """
        Retrieve dependency chain for causal trace queries.
        Only runs when:
        - query_type is CAUSAL_TRACE or IMPACT_ANALYSIS
        - a target entity was identified
        """
        if intent.target_entity is None:
            return []

        if intent.query_type not in (
            QueryType.CAUSAL_TRACE,
            QueryType.IMPACT_ANALYSIS,
            QueryType.DEPENDENCY_MAP,
        ):
            return []

        log.debug("Retrieving dependency chain for: %s", intent.target_entity)
        return self._traversal.retrieve_dependency_chain(intent.target_entity)

    def _build_trace(
        self,
        intent: IntentResult,
        nodes: list[LatticeNode],
        chain: list[LatticeNode],
    ) -> list[str]:
        """
        Build a human-readable retrieval trace.
        This becomes the reasoning trace in the final output.
        """
        trace = []

        trace.append(f"Query: {intent.raw_query}")
        trace.append(
            f"Intent: tags={intent.tags}, "
            f"type={intent.query_type.value}, "
            f"entity={intent.target_entity}"
        )
        trace.append(f"Retrieved {len(nodes)} nodes via lattice traversal")

        for node in nodes:
            trace.append(node.to_trace_entry())

        if chain:
            trace.append(f"Dependency chain ({len(chain)} hops):")
            chain_ids = " → ".join(n.node_id for n in chain)
            trace.append(chain_ids)

        return trace

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns combined stats from intent extractor."""
        return self._extractor.get_stats()
