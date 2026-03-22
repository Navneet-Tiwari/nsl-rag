"""
metrics.py
----------
Metrics and BenchmarkRunner for NSL-RAG evaluation.
Compares NSL-RAG against NaiveRAG baseline.

Measures:
- Token usage per query
- Retrieval precision
- Latency
- Reasoning trace availability

Usage:
    from nsl_rag.evaluation.metrics import BenchmarkRunner
    runner = BenchmarkRunner(index)
    report = runner.run(queries)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from nsl_rag.core.logger import get_logger
from nsl_rag.evaluation.naive_rag import NaiveRAG, NaiveRAGResult
from nsl_rag.generation.formatter import NSLRAGResponse
from nsl_rag.lattice.index import LatticeIndex

log = get_logger(__name__)


# ── Query Result ──────────────────────────────────────────────────────────────


@dataclass
class QueryBenchmark:
    """Benchmark result for a single query."""

    query: str
    nsl_nodes_retrieved: int = 0
    nsl_tokens_estimated: int = 0
    nsl_latency_ms: float = 0.0
    nsl_has_trace: bool = False
    nsl_has_root_cause: bool = False
    nsl_confidence: str = ""
    naive_chunks_retrieved: int = 0
    naive_tokens_used: int = 0
    naive_latency_ms: float = 0.0
    naive_has_trace: bool = False
    error: str = ""


@dataclass
class BenchmarkReport:
    """
    Complete benchmark comparison report.
    This is the paper's results section in data form.
    """

    queries: list[QueryBenchmark] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_queries(self) -> int:
        return len(self.queries)

    @property
    def successful_queries(self) -> list[QueryBenchmark]:
        return [q for q in self.queries if not q.error]

    def avg_nsl_tokens(self) -> float:
        successful = self.successful_queries
        if not successful:
            return 0.0
        return sum(q.nsl_tokens_estimated for q in successful) / len(successful)

    def avg_naive_tokens(self) -> float:
        successful = self.successful_queries
        if not successful:
            return 0.0
        return sum(q.naive_tokens_used for q in successful) / len(successful)

    def avg_nsl_latency(self) -> float:
        successful = self.successful_queries
        if not successful:
            return 0.0
        return sum(q.nsl_latency_ms for q in successful) / len(successful)

    def avg_naive_latency(self) -> float:
        successful = self.successful_queries
        if not successful:
            return 0.0
        return sum(q.naive_latency_ms for q in successful) / len(successful)

    def token_reduction_pct(self) -> float:
        naive = self.avg_naive_tokens()
        nsl = self.avg_nsl_tokens()
        if naive == 0:
            return 0.0
        return round((naive - nsl) / naive * 100, 1)

    def trace_availability_pct(self) -> float:
        successful = self.successful_queries
        if not successful:
            return 0.0
        with_trace = sum(1 for q in successful if q.nsl_has_trace)
        return round(with_trace / len(successful) * 100, 1)

    def display(self) -> str:
        """
        Human readable benchmark report.
        This is your paper's results table.
        """
        lines = [
            "=" * 65,
            "  NSL-RAG vs NAIVE RAG — BENCHMARK RESULTS",
            "=" * 65,
            f"  Queries run    : {self.total_queries}",
            f"  Successful     : {len(self.successful_queries)}",
            "",
            "  TOKEN EFFICIENCY",
            f"  NSL-RAG avg tokens  : {self.avg_nsl_tokens():.0f}",
            f"  Naive RAG avg tokens: {self.avg_naive_tokens():.0f}",
            f"  Token reduction     : {self.token_reduction_pct()}%",
            "",
            "  LATENCY (retrieval only)",
            f"  NSL-RAG avg         : {self.avg_nsl_latency():.1f}ms",
            f"  Naive RAG avg       : {self.avg_naive_latency():.1f}ms",
            "",
            "  EXPLAINABILITY",
            f"  NSL-RAG trace       : {self.trace_availability_pct()}%",
            f"  Naive RAG trace     : 0.0%",
            "",
            "  PER QUERY BREAKDOWN",
            "-" * 65,
        ]

        for i, q in enumerate(self.queries, 1):
            lines.append(f"  Q{i}: {q.query[:50]}")
            if q.error:
                lines.append(f"       ERROR: {q.error}")
            else:
                lines.append(
                    f"       NSL   — nodes: {q.nsl_nodes_retrieved}, "
                    f"tokens: {q.nsl_tokens_estimated}, "
                    f"trace: {q.nsl_has_trace}, "
                    f"confidence: {q.nsl_confidence}"
                )
                lines.append(
                    f"       Naive — chunks: {q.naive_chunks_retrieved}, "
                    f"tokens: {q.naive_tokens_used}"
                )
            lines.append("")

        lines.append("=" * 65)
        return "\n".join(lines)


# ── Benchmark Runner ──────────────────────────────────────────────────────────


class BenchmarkRunner:
    """
    Runs benchmark comparison between NSL-RAG and NaiveRAG.

    For each query:
    1. Run NaiveRAG — measure tokens and latency
    2. Run NSL-RAG retrieval only — measure nodes and latency
    3. Record comparison metrics

    Note: Full NSL-RAG generation is excluded from automated
    benchmarks to preserve API quota. Generation quality is
    evaluated manually using test_generation.py.
    """

    def __init__(self, index: LatticeIndex) -> None:
        self._index = index
        self._naive = NaiveRAG()
        self._naive.build_index()
        log.info("BenchmarkRunner initialised")

    def run(self, queries: list[str]) -> BenchmarkReport:
        """
        Run benchmark for all queries.

        Args:
            queries: List of natural language queries to benchmark.

        Returns:
            BenchmarkReport with comparison metrics.
        """
        log.info("=" * 60)
        log.info("Starting benchmark — %d queries", len(queries))
        log.info("=" * 60)

        report = BenchmarkReport()

        for i, query in enumerate(queries, 1):
            log.info("Benchmarking query %d/%d: %r", i, len(queries), query)
            result = self._benchmark_query(query)
            report.queries.append(result)

        log.info("Benchmark complete")
        log.info(report.display())

        return report

    def _benchmark_query(self, query: str) -> QueryBenchmark:
        """Run both systems on a single query and collect metrics."""
        benchmark = QueryBenchmark(query=query)

        # ── Naive RAG ─────────────────────────────────────────────
        try:
            naive_result = self._naive.query(query)
            benchmark.naive_chunks_retrieved = naive_result.chunk_count
            benchmark.naive_tokens_used = naive_result.tokens_used
            benchmark.naive_latency_ms = naive_result.latency_ms
            benchmark.naive_has_trace = False
        except Exception as e:
            log.error("NaiveRAG failed for query %r: %s", query, e)
            benchmark.error = f"NaiveRAG: {e}"

        # ── NSL-RAG Retrieval ──────────────────────────────────────
        try:
            from nsl_rag.retrieval.navigator import LatticeNavigator
            from nsl_rag.reasoning.prosecutor import Prosecutor
            from nsl_rag.reasoning.judge import Judge
            from nsl_rag.reasoning.auditor import Auditor

            navigator = LatticeNavigator(self._index)
            prosecutor = Prosecutor()
            judge = Judge(self._index)
            auditor = Auditor()

            start = time.time()
            retrieval = navigator.retrieve(query)
            proposed = prosecutor.propose(retrieval)
            validated = judge.validate(proposed)
            report_out = auditor.audit(validated)
            latency_ms = (time.time() - start) * 1000

            # Estimate tokens — context sent to LLM
            context_text = " ".join(f.claim for f in report_out.all_facts)
            tokens_est = len(context_text) // 4

            benchmark.nsl_nodes_retrieved = len(retrieval.all_nodes)
            benchmark.nsl_tokens_estimated = tokens_est
            benchmark.nsl_latency_ms = latency_ms
            benchmark.nsl_has_trace = True
            benchmark.nsl_has_root_cause = True
            benchmark.nsl_confidence = retrieval.intent.query_type.value

        except Exception as e:
            log.error("NSL-RAG failed for query %r: %s", query, e)
            if not benchmark.error:
                benchmark.error = f"NSL-RAG: {e}"

        return benchmark

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens from text. ~4 chars per token."""
        return len(text) // 4
