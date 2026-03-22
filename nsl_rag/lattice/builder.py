"""
builder.py
----------
LatticeBuilder — constructs a LatticeIndex from raw domain data.
Transforms a list of raw node definitions into a validated lattice.
Entry point for dataset → lattice transformation.

Usage:
    from nsl_rag.lattice.builder import LatticeBuilder

    builder = LatticeBuilder()
    index = builder.build(raw_nodes)
"""

from nsl_rag.core.exceptions import LatticeError
from nsl_rag.core.logger import get_logger
from nsl_rag.core.types import NodeType
from nsl_rag.lattice.index import LatticeIndex
from nsl_rag.lattice.node import LatticeNode

log = get_logger(__name__)


class LatticeBuilder:
    """
    Constructs a validated LatticeIndex from raw node definitions.

    Responsibilities:
    - Accept raw node data as a list of dicts
    - Create LatticeNode objects from raw data
    - Add nodes to LatticeIndex
    - Validate the complete lattice after construction
    - Log construction progress

    Does NOT:
    - Store the lattice — that is LatticeIndex's job
    - Traverse the lattice — that is LatticeTraversal's job
    - Define what nodes exist — that is the data module's job
    """

    def __init__(self) -> None:
        log.debug("LatticeBuilder initialised")

    def build(self, raw_nodes: list[dict]) -> LatticeIndex:
        """
        Build and return a validated LatticeIndex from raw node definitions.

        Args:
            raw_nodes: List of dicts, each defining one node.
                       Each dict must contain at minimum:
                       node_id, node_type, title, summary, content.

        Returns:
            A fully constructed and validated LatticeIndex.

        Raises:
            LatticeError: If any node is invalid or structure is broken.
        """
        log.info("Building lattice from %d raw nodes...", len(raw_nodes))

        index = LatticeIndex()

        for raw in raw_nodes:
            node = self._build_node(raw)
            index.add(node)

        index.validate()

        log.info("Lattice built successfully — %s", index.summary())

        return index

    def _build_node(self, raw: dict) -> LatticeNode:
        """
        Build a single LatticeNode from a raw dict.
        Handles type conversion and default values.

        Args:
            raw: Dict containing node field values.

        Returns:
            A validated LatticeNode instance.

        Raises:
            LatticeError: If required fields are missing or invalid.
        """
        required = ["node_id", "node_type", "title", "summary", "content"]

        missing = [field for field in required if field not in raw]
        if missing:
            raise LatticeError(
                f"Raw node missing required fields: {missing}",
                details={"raw": raw, "missing": missing},
            )

        try:
            # Convert node_type string to NodeType enum
            raw_copy = raw.copy()
            if isinstance(raw_copy["node_type"], str):
                raw_copy["node_type"] = NodeType(raw_copy["node_type"])

            node = LatticeNode(**raw_copy)
            log.debug("Node built: %s", node.node_id)
            return node

        except Exception as e:
            raise LatticeError(
                f"Failed to build node '{raw.get('node_id', 'unknown')}'",
                details={"error": str(e), "raw": raw},
            ) from e
