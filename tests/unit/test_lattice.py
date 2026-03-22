"""
test_lattice.py
---------------
Unit tests for LatticeNode.
Run: pytest tests/unit/test_lattice.py -v
"""

import pytest
from nsl_rag.lattice.node import LatticeNode
from nsl_rag.core.types import NodeType, FactStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def payment_node() -> LatticeNode:
    """A realistic payment service node for testing."""
    return LatticeNode(
        node_id="payment_service",
        node_type=NodeType.SERVICE,
        title="Payment Service",
        summary="Handles all payment processing for the e-commerce platform",
        content="Payment Service runs on port 8080. Depends on Fraud Detection and Payment DB.",
        tags=["payment", "service", "critical"],
        children=["fraud_detection", "payment_db"],
        parents=["api_gateway"],
        metadata={"owner": "payments-team", "port": 8080},
        confidence=1.0,
    )


# ── Creation Tests ────────────────────────────────────────────────────────────


class TestLatticeNodeCreation:

    def test_valid_node_creates_successfully(self, payment_node):
        assert payment_node.node_id == "payment_service"
        assert payment_node.node_type == NodeType.SERVICE
        assert payment_node.confidence == 1.0
        assert payment_node.status == FactStatus.PENDING

    def test_tags_are_lowercased_on_creation(self):
        node = LatticeNode(
            node_id="test_node",
            node_type=NodeType.SERVICE,
            title="Test",
            summary="Test node",
            content="Test content",
            tags=["PAYMENT", "Service", "CRITICAL"],
        )
        assert node.tags == ["payment", "service", "critical"]

    def test_node_id_is_lowercased(self):
        node = LatticeNode(
            node_id="Payment_Service",
            node_type=NodeType.SERVICE,
            title="Test",
            summary="Test",
            content="Test",
        )
        assert node.node_id == "payment_service"

    def test_default_status_is_pending(self, payment_node):
        assert payment_node.status == FactStatus.PENDING

    def test_confidence_is_rounded(self):
        node = LatticeNode(
            node_id="test_node",
            node_type=NodeType.SERVICE,
            title="Test",
            summary="Test",
            content="Test",
            confidence=0.123456789,
        )
        assert node.confidence == 0.1235


# ── Validation Tests ──────────────────────────────────────────────────────────


class TestLatticeNodeValidation:

    def test_empty_node_id_raises(self):
        with pytest.raises(Exception):
            LatticeNode(
                node_id="",
                node_type=NodeType.SERVICE,
                title="Test",
                summary="Test",
                content="Test",
            )

    def test_invalid_confidence_above_1_raises(self):
        with pytest.raises(Exception):
            LatticeNode(
                node_id="test_node",
                node_type=NodeType.SERVICE,
                title="Test",
                summary="Test",
                content="Test",
                confidence=1.5,
            )

    def test_invalid_confidence_below_0_raises(self):
        with pytest.raises(Exception):
            LatticeNode(
                node_id="test_node",
                node_type=NodeType.SERVICE,
                title="Test",
                summary="Test",
                content="Test",
                confidence=-0.1,
            )


# ── Property Tests ────────────────────────────────────────────────────────────


class TestLatticeNodeProperties:

    def test_is_leaf_when_no_children(self):
        node = LatticeNode(
            node_id="leaf_node",
            node_type=NodeType.DATABASE,
            title="Leaf",
            summary="A leaf node",
            content="No children",
            children=[],
        )
        assert node.is_leaf is True

    def test_is_not_leaf_when_has_children(self, payment_node):
        assert payment_node.is_leaf is False

    def test_is_root_when_no_parents(self):
        node = LatticeNode(
            node_id="root_node",
            node_type=NodeType.ROOT,
            title="Root",
            summary="Root node",
            content="Top level",
            parents=[],
        )
        assert node.is_root is True

    def test_is_not_root_when_has_parents(self, payment_node):
        assert payment_node.is_root is False

    def test_is_confident_above_threshold(self, payment_node):
        assert payment_node.is_confident is True

    def test_is_not_confident_below_threshold(self):
        node = LatticeNode(
            node_id="uncertain_node",
            node_type=NodeType.SERVICE,
            title="Uncertain",
            summary="Low confidence node",
            content="Inferred data",
            confidence=0.5,
        )
        assert node.is_confident is False


# ── Tag Matching Tests ────────────────────────────────────────────────────────


class TestTagMatching:
    """
    These tests verify the core symbolic retrieval logic.
    matches_tags() is the AND filter that powers lattice retrieval.
    """

    def test_matches_all_query_tags(self, payment_node):
        assert payment_node.matches_tags(["payment", "service"]) is True

    def test_matches_single_tag(self, payment_node):
        assert payment_node.matches_tags(["payment"]) is True

    def test_fails_when_one_tag_missing(self, payment_node):
        assert payment_node.matches_tags(["payment", "database"]) is False

    def test_fails_when_no_tags_match(self, payment_node):
        assert payment_node.matches_tags(["warehouse", "inventory"]) is False

    def test_empty_query_tags_matches_any_node(self, payment_node):
        assert payment_node.matches_tags([]) is True


# ── Output Format Tests ───────────────────────────────────────────────────────


class TestOutputFormats:

    def test_to_context_string_contains_title(self, payment_node):
        context = payment_node.to_context_string()
        assert "Payment Service" in context

    def test_to_context_string_contains_content(self, payment_node):
        context = payment_node.to_context_string()
        assert "port 8080" in context

    def test_to_trace_entry_contains_title(self, payment_node):
        trace = payment_node.to_trace_entry()
        assert "Payment Service" in trace

    def test_to_trace_entry_contains_node_type(self, payment_node):
        trace = payment_node.to_trace_entry()
        assert "SERVICE" in trace


# ── Integration Smoke Test ────────────────────────────────────────────────────


class TestLatticeIntegration:
    """
    Full stack test — Node + Index + Builder + Traversal together.
    These are not unit tests — they test the whole lattice pipeline.
    """

    @pytest.fixture
    def raw_nodes(self) -> list[dict]:
        return [
            {
                "node_id": "ecommerce_root",
                "node_type": "root",
                "title": "E-Commerce System",
                "summary": "Root node",
                "content": "Top level system",
                "tags": ["root", "system"],
                "children": ["payment_service"],
                "parents": [],
            },
            {
                "node_id": "payment_service",
                "node_type": "service",
                "title": "Payment Service",
                "summary": "Handles payments",
                "content": "Runs on port 8080",
                "tags": ["payment", "service", "critical"],
                "children": ["payment_db", "fraud_detection"],
                "parents": ["ecommerce_root"],
            },
            {
                "node_id": "fraud_detection",
                "node_type": "service",
                "title": "Fraud Detection",
                "summary": "Validates payments",
                "content": "ML fraud scoring",
                "tags": ["payment", "service", "fraud", "critical"],
                "children": [],
                "parents": ["payment_service"],
            },
            {
                "node_id": "payment_db",
                "node_type": "database",
                "title": "Payment Database",
                "summary": "Stores transactions",
                "content": "PostgreSQL port 5432",
                "tags": ["payment", "database", "critical"],
                "children": [],
                "parents": ["payment_service"],
            },
        ]

    @pytest.fixture
    def built_index(self, raw_nodes):
        from nsl_rag.lattice.builder import LatticeBuilder

        builder = LatticeBuilder()
        return builder.build(raw_nodes)

    @pytest.fixture
    def traversal(self, built_index):
        from nsl_rag.lattice.traversal import LatticeTraversal
        from nsl_rag.config.config_loader import config

        config.load()
        return LatticeTraversal(built_index)

    def test_builder_creates_correct_node_count(self, built_index):
        assert built_index.size == 4

    def test_retrieval_payment_critical(self, traversal):
        results = traversal.retrieve(["payment", "critical"])
        ids = [n.node_id for n in results]
        assert "payment_service" in ids
        assert "payment_db" in ids
        assert "fraud_detection" in ids

    def test_retrieval_excludes_unrelated_nodes(self, traversal):
        results = traversal.retrieve(["payment", "database"])
        ids = [n.node_id for n in results]
        assert "payment_db" in ids
        assert "fraud_detection" not in ids
        assert "ecommerce_root" not in ids

    def test_dependency_chain_traverses_upward(self, traversal):
        chain = traversal.retrieve_dependency_chain("payment_db")
        ids = [n.node_id for n in chain]
        assert "payment_db" in ids
        assert "payment_service" in ids
        assert "ecommerce_root" in ids

    def test_explain_retrieval_returns_correct_structure(self, traversal):
        explanation = traversal.explain_retrieval(["fraud", "critical"])
        assert "query_tags" in explanation
        assert "result_nodes" in explanation
        assert "fraud_detection" in explanation["result_nodes"]
