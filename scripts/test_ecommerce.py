"""
test_ecommerce.py
-----------------
Smoke test for the full e-commerce dataset.
Builds the lattice and runs all three failure scenarios.

Run: python scripts/test_ecommerce.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.lattice.traversal import LatticeTraversal

setup_logging()
config.load()

log = get_logger("nsl_rag.test.ecommerce")

# ── Build Lattice ─────────────────────────────────────────────────────────────

log.info("=" * 60)
log.info("Building e-commerce lattice...")

raw_nodes = EcommerceSystem.get_raw_nodes()
builder = LatticeBuilder()
index = builder.build(raw_nodes)

log.info("Summary: %s", index.summary())

# ── Print Tree ────────────────────────────────────────────────────────────────

log.info("=" * 60)
log.info("Lattice tree:")
print()
index.print_tree("ecommerce_root")
print()

# ── Run Failure Scenarios ─────────────────────────────────────────────────────

traversal = LatticeTraversal(index)
scenarios = EcommerceSystem.get_failure_scenarios()

for scenario in scenarios:
    log.info("=" * 60)
    log.info("SCENARIO: %s", scenario["name"])
    log.info("Query   : %s", scenario["query"])
    log.info("Tags    : %s", scenario["expected_tags"])

    results = traversal.retrieve(query_tags=scenario["expected_tags"])

    retrieved_ids = [n.node_id for n in results]
    log.info("Retrieved: %s", retrieved_ids)

    # Check expected nodes were found
    for expected in scenario["expected_nodes"]:
        if expected in retrieved_ids:
            log.info("  ✓ found: %s", expected)
        else:
            log.info("  ✗ missing: %s", expected)

    # Dependency chain
    target = scenario["expected_nodes"][0]
    chain = traversal.retrieve_dependency_chain(target)
    log.info(
        "Dependency chain from '%s': %s", target, " → ".join(n.node_id for n in chain)
    )

log.info("=" * 60)
log.info("E-commerce dataset test complete.")
