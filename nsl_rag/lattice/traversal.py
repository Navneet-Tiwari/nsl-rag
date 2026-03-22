"""
traversal.py
------------
LatticeTraversal — the core NSL-RAG retrieval engine.
Implements formally bounded, intent-driven lattice traversal.

This is the primary innovation of NSL-RAG.
Instead of vector similarity search, retrieval is performed by
navigating the concept lattice using symbolic tag constraints.

Three-step retrieval algorithm:
    1. Join Navigation  — find the least upper bound of query tags
    2. Downward Traversal — navigate from join to specific nodes
    3. Bounded Return — return only nodes satisfying all constraints

Usage:
    from nsl_rag.lattice.traversal import LatticeTraversal

    traversal = LatticeTraversal(index)
    nodes = traversal.retrieve(query_tags=["payment", "critical"])
"""

from nsl_rag.config.config_loader import config
from nsl_rag.core.exceptions import NoNodesFoundError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import NodeType
from nsl_rag.lattice.index import LatticeIndex
from nsl_rag.lattice.node import LatticeNode

log = get_logger(__name__)


class LatticeTraversal:
    """
    Formally bounded lattice traversal engine.

    Core properties:
    - Bounded: cannot return nodes outside the queried concept boundary
    - Deterministic: same query always returns same nodes
    - Explainable: every retrieval decision is traceable
    - Efficient: navigates directly to relevant nodes, skips irrelevant ones
    """

    def __init__(self, index: LatticeIndex) -> None:
        self._index = index
        self._max_depth = config.lattice.max_depth
        self._max_nodes = config.lattice.max_nodes_per_query
        self._threshold = config.lattice.confidence_threshold
        self._min_match = config.retrieval.min_tag_match
        log.debug(
            "LatticeTraversal initialised — max_depth: %d, max_nodes: %d",
            self._max_depth,
            self._max_nodes,
        )

    # ── Primary Retrieval Interface ───────────────────────────────────────────

    def retrieve(
        self,
        query_tags: list[str],
        target_node_id: str | None = None,
        max_nodes: int | None = None,
    ) -> list[LatticeNode]:
        """
        Primary retrieval method — the NSL-RAG core algorithm.

        Given symbolic tags extracted from a user query, navigates
        the lattice to find all relevant nodes. Retrieval is bounded
        by the lattice structure — irrelevant nodes are never visited.

        Algorithm:
            Step 1 — If target_node_id provided, start there directly.
                     Otherwise, find entry point via tag matching.
            Step 2 — From entry point, traverse downward collecting
                     all nodes that satisfy tag constraints.
            Step 3 — Filter by confidence threshold.
            Step 4 — Cap at max_nodes and return.

        Args:
            query_tags:     Symbolic tags extracted from user query.
            target_node_id: Optional — start traversal from specific node.
            max_nodes:      Override default max nodes cap.

        Returns:
            List of LatticeNode objects satisfying all constraints.
            Ordered from most general to most specific.

        Raises:
            NoNodesFoundError: If no nodes satisfy the constraints.
        """
        log.info("Retrieving nodes for tags: %s", query_tags)

        cap = max_nodes or self._max_nodes

        # Step 1 — Find entry point
        if target_node_id and self._index.exists(target_node_id):
            entry_nodes = [self._index.get(target_node_id)]
            log.debug("Entry point: direct → %s", target_node_id)
        else:
            entry_nodes = self._find_entry_points(query_tags)

        if not entry_nodes:
            raise NoNodesFoundError(
                f"No entry points found for tags: {query_tags}",
                details={"query_tags": query_tags},
            )

        # Step 2 — Traverse downward from each entry point
        visited: set[str] = set()
        results: list[LatticeNode] = []

        for entry in entry_nodes:
            self._traverse_down(
                node=entry,
                query_tags=query_tags,
                visited=visited,
                results=results,
                depth=0,
            )

        # Step 3 — Filter by confidence threshold
        results = [n for n in results if n.confidence >= self._threshold]

        if not results:
            raise NoNodesFoundError(
                f"No nodes met confidence threshold {self._threshold} "
                f"for tags: {query_tags}",
                details={"query_tags": query_tags, "threshold": self._threshold},
            )

        # Step 4 — Cap and return
        final = results[:cap]

        log.info(
            "Retrieval complete — %d nodes found, returning %d",
            len(results),
            len(final),
        )

        return final

    def retrieve_dependency_chain(self, node_id: str) -> list[LatticeNode]:
        """
        Retrieve the full dependency chain for a node.
        Traverses UPWARD from the given node to the root.
        Used for causal trace queries: "why is X failing?"

        This is the key method for DevOps reasoning —
        it follows the dependency path from a specific failure
        all the way up to the root cause entry point.

        Args:
            node_id: The node to start from (usually the failing service).

        Returns:
            List of nodes from given node up to root.
            Ordered from specific (bottom) to general (top).
        """
        log.info("Retrieving dependency chain for: %s", node_id)

        visited: set[str] = set()
        chain: list[LatticeNode] = []

        self._traverse_up(
            node_id=node_id,
            visited=visited,
            chain=chain,
            depth=0,
        )

        log.info("Dependency chain: %s", " → ".join(n.node_id for n in chain))

        return chain

    # ── Traversal Algorithms ──────────────────────────────────────────────────

    def _find_entry_points(self, query_tags: list[str]) -> list[LatticeNode]:
        """
        Find the best entry points for traversal given query tags.
        Uses min_match threshold from config for flexible matching.
        """
        if not query_tags:
            return self._index.root_nodes()

        matching = self._index.nodes_matching_tags(
            query_tags, min_match=self._min_match
        )

        if not matching:
            log.debug("No tag matches found — falling back to root nodes")
            return self._index.root_nodes()

        entry_points = []
        for node in matching:
            parents = self._index.get_parents(node.node_id)
            parent_matches = any(
                p.matches_tags(query_tags, min_match=self._min_match) for p in parents
            )
            if not parent_matches:
                entry_points.append(node)
                log.debug("Entry point identified: %s", node.node_id)

        return entry_points if entry_points else matching

    def _traverse_down(
        self,
        node: LatticeNode,
        query_tags: list[str],
        visited: set[str],
        results: list[LatticeNode],
        depth: int,
    ) -> None:
        """
        Recursively traverse downward from a node collecting matches.

        Bounded by:
        - max_depth: prevents infinite traversal
        - visited set: prevents revisiting nodes
        - tag matching: only follows nodes satisfying constraints

        Args:
            node:       Current node being visited.
            query_tags: Tags all results must satisfy.
            visited:    Set of already-visited node IDs.
            results:    Accumulator for matching nodes.
            depth:      Current traversal depth.
        """
        # Bounds check
        if depth > self._max_depth:
            log.debug("Max depth reached at: %s", node.node_id)
            return

        # Cycle prevention
        if node.node_id in visited:
            return

        visited.add(node.node_id)

        # Collect this node if it matches
        if node.matches_tags(query_tags, min_match=self._min_match):
            results.append(node)
            log.debug(
                "  depth=%d | collected: %s | tags: %s", depth, node.node_id, node.tags
            )

            # Continue traversal into children
            for child_id in node.children:
                if self._index.exists(child_id):
                    child = self._index.get(child_id)
                    self._traverse_down(
                        node=child,
                        query_tags=query_tags,
                        visited=visited,
                        results=results,
                        depth=depth + 1,
                    )

    def _traverse_up(
        self,
        node_id: str,
        visited: set[str],
        chain: list[LatticeNode],
        depth: int,
    ) -> None:
        """
        Recursively traverse upward from a node following parent links.
        Used for dependency chain retrieval.

        Args:
            node_id: Current node ID being visited.
            visited: Set of already-visited node IDs.
            chain:   Accumulator for the dependency chain.
            depth:   Current traversal depth.
        """
        if depth > self._max_depth:
            return

        if node_id in visited:
            return

        if not self._index.exists(node_id):
            return

        visited.add(node_id)
        node = self._index.get(node_id)
        chain.append(node)

        log.debug("  depth=%d | chain node: %s", depth, node.node_id)

        for parent_id in node.parents:
            self._traverse_up(
                node_id=parent_id,
                visited=visited,
                chain=chain,
                depth=depth + 1,
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def explain_retrieval(self, query_tags: list[str]) -> dict:
        """
        Returns a human-readable explanation of how retrieval
        would proceed for given tags. Used for debugging and
        paper demonstrations.

        Args:
            query_tags: Tags to explain retrieval for.

        Returns:
            Dict with entry_points, traversal_path, and result_nodes.
        """
        try:
            entry_points = self._find_entry_points(query_tags)
            results = self.retrieve(query_tags)

            return {
                "query_tags": query_tags,
                "entry_points": [n.node_id for n in entry_points],
                "result_nodes": [n.node_id for n in results],
                "result_count": len(results),
            }
        except NoNodesFoundError:
            return {
                "query_tags": query_tags,
                "entry_points": [],
                "result_nodes": [],
                "result_count": 0,
            }
