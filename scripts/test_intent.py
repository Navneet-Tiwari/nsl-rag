"""
test_intent.py
--------------
Tests IntentExtractor with real Gemini calls.
Verifies extraction, caching, and cost controls.

Run: python scripts/test_intent.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.retrieval.intent import IntentExtractor

setup_logging()
config.load()

log = get_logger("nsl_rag.test.intent")
extractor = IntentExtractor()

# ── Test Queries ──────────────────────────────────────────────────────────────
# Deliberately varied phrasing — not DevOps keywords
# This tests that Gemini handles natural language correctly

queries = [
    # Scenario 1 — Classic Cascade
    "Why is the payment service failing?",
    # Scenario 2 — Natural language variation
    "Checkout is broken, customers can't pay",
    # Scenario 3 — Business language
    "Sales are dropping, something is wrong with orders",
    # Cache test — same intent, different phrasing
    "Why is the payment service failing?",  # should be cache hit
    # Scenario 4 — Vague query
    "Something went wrong with the database",
]

log.info("=" * 60)
log.info("Testing IntentExtractor — %d queries", len(queries))
log.info("=" * 60)

for i, query in enumerate(queries, 1):
    log.info("Query %d: %r", i, query)
    try:
        result = extractor.extract(query)
        log.info("  tags          : %s", result.tags)
        log.info("  query_type    : %s", result.query_type.value)
        log.info("  target_entity : %s", result.target_entity)
        log.info("  from_cache    : %s", result.from_cache)
        log.info("  from_fallback : %s", result.from_fallback)
    except Exception as e:
        log.error("  FAILED: %s", e)
    log.info("-" * 40)

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = extractor.get_stats()
log.info("=" * 60)
log.info("COST STATS:")
log.info("  total extractions : %d", stats["total_extractions"])
log.info("  gemini calls      : %d", stats["gemini_calls"])
log.info("  cache hits        : %d", stats["cache_hits"])
log.info("  fallback calls    : %d", stats["fallback_calls"])
log.info("  cache hit rate    : %.1f%%", stats["cache_hit_rate"] * 100)
log.info("=" * 60)
