"""
pipeline.py
-----------
NSLRAGPipeline — single entry point for the complete NSL-RAG system.
Wires all layers together into one callable interface.

Usage:
    from nsl_rag import NSLRAGPipeline

    pipeline = NSLRAGPipeline()
    response = pipeline.query("Why is the payment service failing?")

    print(response.answer)
    print(response.confidence)
    print(response.trace)
    print(response.to_dict())
"""

from nsl_rag.config.config_loader import config
from nsl_rag.core.logger import get_logger, setup_logging
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.generation.formatter import NSLRAGResponse, OutputFormatter
from nsl_rag.generation.generator import Generator
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.lattice.index import LatticeIndex
from nsl_rag.reasoning.auditor import Auditor
from nsl_rag.reasoning.judge import Judge
from nsl_rag.reasoning.prosecutor import Prosecutor
from nsl_rag.retrieval.navigator import LatticeNavigator

log = get_logger(__name__)


class NSLRAGPipeline:
    """
    Complete NSL-RAG pipeline — single entry point.

    Initialises all layers once and keeps them in memory.
    Call query() for each question.

    Architecture:
        Query → Intent → Lattice → Prosecutor → Judge →
        Auditor → Generator → Formatter → NSLRAGResponse
    """

    def __init__(
        self,
        auto_setup: bool = True,
    ) -> None:
        """
        Initialise the NSL-RAG pipeline.

        Args:
            auto_setup: If True, loads config, sets up logging,
                        and builds the lattice automatically.
                        Set False if you manage setup yourself.
        """
        if auto_setup:
            setup_logging()
            config.load()

        self._index = self._build_index()
        self._navigator = LatticeNavigator(self._index)
        self._prosecutor = Prosecutor()
        self._judge = Judge(self._index)
        self._auditor = Auditor()
        self._generator = Generator()
        self._formatter = OutputFormatter()

        log.info("NSLRAGPipeline ready — %d nodes in lattice", self._index.size)

    # ── Primary Interface ─────────────────────────────────────────────────────

    def query(self, question: str) -> NSLRAGResponse:
        """
        Run the complete NSL-RAG pipeline for a question.

        Args:
            question: Natural language question from the user.

        Returns:
            NSLRAGResponse with answer, trace, confidence, and flags.
        """
        log.info("Pipeline query: %r", question)

        # Step 1 — Retrieve
        retrieval = self._navigator.retrieve(question)

        # Step 2 — Propose facts
        proposed = self._prosecutor.propose(retrieval)

        # Step 3 — Validate facts
        validated = self._judge.validate(proposed)

        # Step 4 — Audit for contradictions
        report = self._auditor.audit(validated)

        # Step 5 — Generate answer
        raw = self._generator.generate(question, report, retrieval)

        # Step 6 — Format structured response
        response = self._formatter.format(raw, report, retrieval)

        log.info(
            "Pipeline complete — confidence: %s, facts: %d, trace: %d steps",
            response.confidence,
            response.facts_used,
            len(response.trace),
        )

        return response

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns combined usage stats from all pipeline components."""
        nav_stats = self._navigator.get_stats()
        gen_stats = self._generator.get_stats()
        return {
            **nav_stats,
            **gen_stats,
            "lattice_nodes": self._index.size,
        }

    def print_lattice(self) -> None:
        """Print the full lattice tree — useful for demos."""
        self._index.print_tree("ecommerce_root")

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_index(self) -> LatticeIndex:
        """Build lattice index from e-commerce dataset."""
        log.info("Building lattice index...")
        raw_nodes = EcommerceSystem.get_raw_nodes()
        index = LatticeBuilder().build(raw_nodes)
        log.info("Lattice ready — %s", index.summary())
        return index
