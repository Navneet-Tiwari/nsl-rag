"""
test_reasoning.py
-----------------
Tests the full Prosecutor / Judge / Auditor pipeline.
Retrieved nodes → proposed facts → validated facts → audit report.

Run: python scripts/test_reasoning.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.retrieval.navigator import LatticeNavigator
from nsl_rag.reasoning.prosecutor import Prosecutor
from nsl_rag.reasoning.judge import Judge
from nsl_rag.reasoning.auditor import Auditor

setup_logging()
config.load()

log = get_logger("nsl_rag.test.reasoning")

# ── Build System ──────────────────────────────────────────────────────────────

log.info("Building e-commerce lattice...")
index = LatticeBuilder().build(EcommerceSystem.get_raw_nodes())
navigator = LatticeNavigator(index)
prosecutor = Prosecutor()
judge = Judge(index)
auditor = Auditor()

# ── Test Query ────────────────────────────────────────────────────────────────

query = "Why is the payment service failing?"

log.info("=" * 60)
log.info("QUERY: %r", query)
log.info("=" * 60)

# Step 1 — Retrieve
log.info("STEP 1 — Retrieval")
retrieval = navigator.retrieve(query)
log.info("Retrieved %d unique nodes", len(retrieval.all_nodes))

# Step 2 — Prosecutor
log.info("=" * 60)
log.info("STEP 2 — Prosecutor")
proposed_facts = prosecutor.propose(retrieval)
log.info("Proposed %d facts", len(proposed_facts))
for fact in proposed_facts:
    log.info("  %s", fact)

# Step 3 — Judge
log.info("=" * 60)
log.info("STEP 3 — Judge")
validated_facts = judge.validate(proposed_facts)
summary = judge.summarise(validated_facts)
log.info("Validation summary: %s", summary)
for fact in validated_facts:
    log.info("  [%s] %s → %s", fact.status.value.upper(), fact.fact_id, fact.claim[:80])

# Step 4 — Auditor
log.info("=" * 60)
log.info("STEP 4 — Auditor")
report = auditor.audit(validated_facts)
log.info("Audit: %s", report.summary)

if not report.is_clean:
    log.warning("Contradictions found:")
    for flag in report.contradictions:
        log.warning("  [%s] %s", flag.severity.upper(), flag.reason)

log.info("Clean facts: %d", len(report.clean_facts))
log.info("Flagged facts: %d", len(report.flagged_facts))

# Step 5 — What goes to LLM
log.info("=" * 60)
log.info("STEP 5 — Context ready for LLM")
log.info("Facts reaching LLM: %d", len(report.all_facts))
for fact in report.all_facts:
    log.info("  %s", fact.to_context_string())

if report.to_context_flags():
    log.info("Contradiction flags for LLM:")
    for flag in report.to_context_flags():
        log.info("  %s", flag)

log.info("=" * 60)
log.info("Reasoning pipeline complete.")
