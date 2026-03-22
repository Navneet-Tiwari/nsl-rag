"""
index.py
--------
LatticeIndex — in-memory store for all LatticeNode objects.
Manages node lifecycle: add, retrieve, validate, and query.
Single source of truth for the lattice structure at runtime.

Usage:
    from nsl_rag.lattice.index import LatticeIndex
    index = LatticeIndex()
    index.add(node)
    node = index.get("payment_service")
"""

from nsl_rag.core.exceptions import (
    LatticeError,
    NodeNotFoundError,
    CircularDependencyError,
)
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import NodeType
from nsl_rag.lattice.node import LatticeNode

log = get_logger(__name__)


class LatticeIndex:
    """
    In-memory store and manager for the knowledge lattice.

    Responsibilities:
    - Store nodes by ID for O(1) lookup
    - Validate structural integrity on add
    - Detect circular dependencies
    - Provide query methods for traversal
    - Provide summary and debug utilities
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LatticeNode] = {}
        log.debug("LatticeIndex initialised")

    # ── Core Operations ───────────────────────────────────────────────────────

    def add(self, node: LatticeNode) -> None:
        """
        Add a node to the index.
        Validates uniqueness and structural integrity before adding.

        Raises:
            LatticeError: If node_id already exists in the index.
        """
        if node.node_id in self._nodes:
            raise LatticeError(
                f"Node already exists: {node.node_id}",
                details={"node_id": node.node_id},
            )

        self._nodes[node.node_id] = node
        log.debug("Node added: %s (%s)", node.node_id, node.node_type.value)

    def get(self, node_id: str) -> LatticeNode:
        """
        Retrieve a node by ID.

        Raises:
            NodeNotFoundError: If node_id does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(
                f"Node not found: {node_id}",
                details={"node_id": node_id, "available": list(self._nodes.keys())},
            )
        return self._nodes[node_id]

    def exists(self, node_id: str) -> bool:
        """Returns True if node_id exists in the index."""
        return node_id in self._nodes

    def remove(self, node_id: str) -> None:
        """
        Remove a node from the index.

        Raises:
            NodeNotFoundError: If node_id does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Cannot remove — node not found: {node_id}")
        del self._nodes[node_id]
        log.debug("Node removed: %s", node_id)

    # ── Query Methods ─────────────────────────────────────────────────────────

    def all_nodes(self) -> list[LatticeNode]:
        """Return all nodes in the index."""
        return list(self._nodes.values())

    def nodes_by_type(self, node_type: NodeType) -> list[LatticeNode]:
        """Return all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def nodes_matching_tags(self, tags: list[str]) -> list[LatticeNode]:
        """
        Return all nodes that match ALL provided tags.
        This is the symbolic AND filter applied across the full index.
        Used by LatticeNavigator during retrieval.

        Args:
            tags: List of tags that must ALL be present in matching nodes.

        Returns:
            List of nodes satisfying all tag constraints.
        """
        if not tags:
            return self.all_nodes()

        matched = [n for n in self._nodes.values() if n.matches_tags(tags)]
        log.debug(
            "Tag query %s → %d nodes matched from %d total",
            tags,
            len(matched),
            len(self._nodes),
        )
        return matched

    def root_nodes(self) -> list[LatticeNode]:
        """Return all nodes with no parents — the lattice entry points."""
        return [n for n in self._nodes.values() if n.is_root]

    def leaf_nodes(self) -> list[LatticeNode]:
        """Return all nodes with no children — the most specific concepts."""
        return [n for n in self._nodes.values() if n.is_leaf]

    def get_children(self, node_id: str) -> list[LatticeNode]:
        """
        Return all direct children of a node.

        Raises:
            NodeNotFoundError: If node_id does not exist.
        """
        node = self.get(node_id)
        return [
            self.get(child_id) for child_id in node.children if self.exists(child_id)
        ]

    def get_parents(self, node_id: str) -> list[LatticeNode]:
        """
        Return all direct parents of a node.

        Raises:
            NodeNotFoundError: If node_id does not exist.
        """
        node = self.get(node_id)
        return [
            self.get(parent_id) for parent_id in node.parents if self.exists(parent_id)
        ]

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """
        Validate structural integrity of the entire lattice.
        Call after all nodes have been added.

        Checks:
        - All child references point to existing nodes
        - All parent references point to existing nodes
        - No circular dependencies exist

        Raises:
            LatticeError: If any structural violation is found.
            CircularDependencyError: If a cycle is detected.
        """
        log.debug("Validating lattice integrity...")

        for node in self._nodes.values():
            # Validate child references
            for child_id in node.children:
                if not self.exists(child_id):
                    raise LatticeError(
                        f"Node '{node.node_id}' references non-existent child: '{child_id}'",
                        details={"node_id": node.node_id, "missing_child": child_id},
                    )

            # Validate parent references
            for parent_id in node.parents:
                if not self.exists(parent_id):
                    raise LatticeError(
                        f"Node '{node.node_id}' references non-existent parent: '{parent_id}'",
                        details={"node_id": node.node_id, "missing_parent": parent_id},
                    )

        # Check for circular dependencies
        self._detect_cycles()
        log.debug("Lattice validation passed — %d nodes", len(self._nodes))

    def _detect_cycles(self) -> None:
        """
        Detect circular dependencies using DFS with visited tracking.
        Raises CircularDependencyError if a cycle is found.
        """
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            in_stack.add(node_id)

            node = self._nodes[node_id]
            for child_id in node.children:
                if child_id not in self._nodes:
                    continue
                if child_id not in visited:
                    dfs(child_id)
                elif child_id in in_stack:
                    raise CircularDependencyError(
                        f"Circular dependency detected: '{node_id}' → '{child_id}'",
                        details={"from": node_id, "to": child_id},
                    )

            in_stack.discard(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

    # ── Stats and Debug ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Total number of nodes in the index."""
        return len(self._nodes)

    def summary(self) -> dict:
        """
        Returns a summary of the lattice for logging and debugging.
        """
        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            key = node.node_type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "total_nodes": self.size,
            "root_nodes": len(self.root_nodes()),
            "leaf_nodes": len(self.leaf_nodes()),
            "by_type": type_counts,
        }

    def print_tree(self, node_id: str, level: int = 0) -> None:
        """
        Print the lattice as an indented tree from a given root node.
        Used for debugging and demo visualisation.
        """
        if not self.exists(node_id):
            return
        node = self.get(node_id)
        indent = "  " * level
        prefix = "┗━ " if level > 0 else "🌐 "
        print(f"{indent}{prefix}[{node.node_id}] {node.title}")
        print(f"{indent}   tags: {node.tags}")
        for child_id in node.children:
            self.print_tree(child_id, level + 1)
