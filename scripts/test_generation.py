"""
test_generation.py
------------------
Tests the complete NSL-RAG pipeline end to end.
Query → Intent → Lattice → P/J/A → Generate → Format → Output

Run: python scripts/test_generation.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.retrieval.navigator import LatticeNavigator
from nsl_rag.reasoning.prosecutor import Prosecutor
from nsl_rag.reasoning.judge import Judge
from nsl_rag.reasoning.auditor import Auditor
from nsl_rag.generation.generator import Generator
from nsl_rag.generation.formatter import OutputFormatter

setup_logging()
config.load()

log = get_logger("nsl_rag.test.generation")

# ── Build System ──────────────────────────────────────────────────────────────

log.info("Building e-commerce lattice...")
index = LatticeBuilder().build(EcommerceSystem.get_raw_nodes())
navigator = LatticeNavigator(index)
prosecutor = Prosecutor()
judge = Judge(index)
auditor = Auditor()
generator = Generator()
formatter = OutputFormatter()

# ── Run Full Pipeline ─────────────────────────────────────────────────────────

queries = [
    "Why is the payment service failing?",
    "Why are orders stuck in pending state?",
]

for query in queries:
    log.info("=" * 60)
    log.info("RUNNING PIPELINE FOR: %r", query)
    log.info("=" * 60)

    # Step 1 — Retrieve
    retrieval = navigator.retrieve(query)

    # Step 2 — Prosecutor
    proposed = prosecutor.propose(retrieval)

    # Step 3 — Judge
    validated = judge.validate(proposed)

    # Step 4 — Auditor
    report = auditor.audit(validated)

    # Step 5 — Generate
    raw_response = generator.generate(query, report, retrieval)

    # Step 6 — Format
    response = formatter.format(raw_response, report, retrieval)

    # Display
    print(response.display())
    print()

# ── Stats ─────────────────────────────────────────────────────────────────────

nav_stats = navigator.get_stats()
gen_stats = generator.get_stats()

log.info("=" * 60)
log.info("PIPELINE STATS:")
log.info("  Intent extractions : %d", nav_stats["total_extractions"])
log.info("  Gemini intent calls: %d", nav_stats["gemini_calls"])
log.info("  Cache hits         : %d", nav_stats["cache_hits"])
log.info("  Generation calls   : %d", gen_stats["gemini_generation_calls"])
log.info(
    "  Total Gemini calls : %d",
    nav_stats["gemini_calls"] + gen_stats["gemini_generation_calls"],
)
