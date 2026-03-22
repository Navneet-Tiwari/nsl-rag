"""
test_navigator.py
-----------------
Tests the full retrieval pipeline.
Natural language query → intent → lattice → RetrievalResult.

Run: python scripts/test_navigator.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.retrieval.navigator import LatticeNavigator

setup_logging()
config.load()

log = get_logger("nsl_rag.test.navigator")

# ── Build Lattice ─────────────────────────────────────────────────────────────

log.info("Building e-commerce lattice...")
index = LatticeBuilder().build(EcommerceSystem.get_raw_nodes())
navigator = LatticeNavigator(index)

# ── Test Queries ──────────────────────────────────────────────────────────────

queries = [
    "Why is the payment service failing?",
    "Checkout is broken, customers can't pay",
    "Why are orders stuck in pending state?",
]

for query in queries:
    log.info("=" * 55)
    log.info("QUERY: %r", query)

    result = navigator.retrieve(query)

    log.info("Nodes retrieved: %s", result.node_ids)
    log.info("Trace:")
    for line in result.trace:
        log.info("  %s", line)

# ── Stats ─────────────────────────────────────────────────────────────────────

stats = navigator.get_stats()
log.info("=" * 55)
log.info("COST STATS: %s", stats)
