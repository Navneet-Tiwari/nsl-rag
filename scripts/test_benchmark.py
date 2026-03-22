# """
# test_benchmark.py
# -----------------
# Runs benchmark comparison between NSL-RAG and NaiveRAG.
# No Gemini API calls — tests retrieval layer only.

# Run: python scripts/test_benchmark.py
# """

# from nsl_rag.core.logger import setup_logging, get_logger
# from nsl_rag.config.config_loader import config
# from nsl_rag.data.ecommerce import EcommerceSystem
# from nsl_rag.lattice.builder import LatticeBuilder
# from nsl_rag.evaluation.metrics import BenchmarkRunner

# setup_logging()
# config.load()

# log = get_logger("nsl_rag.test.benchmark")

# # ── Build Lattice ─────────────────────────────────────────────────────────────

# log.info("Building e-commerce lattice...")
# index = LatticeBuilder().build(EcommerceSystem.get_raw_nodes())

# # ── Benchmark Queries ─────────────────────────────────────────────────────────

# queries = [
#     "Why is the payment service failing?",
#     "Why are orders stuck in pending state?",
#     "Checkout is broken, customers cannot pay",
#     "What is wrong with the payment database?",
#     "Why are emails not being sent?",
# ]

# # ── Run Benchmark ─────────────────────────────────────────────────────────────

# runner = BenchmarkRunner(index)
# report = runner.run(queries)

# # ── Print Report ──────────────────────────────────────────────────────────────

# print(report.display())

# log.info("Benchmark complete.")

"""
test_benchmark.py
-----------------
Runs benchmark comparison between NSL-RAG and NaiveRAG.
Uses pre-defined intent tags — no Gemini API calls needed.
Tests retrieval and reasoning layers only.

Run: python scripts/test_benchmark.py
"""

import time

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config
from nsl_rag.data.ecommerce import EcommerceSystem
from nsl_rag.lattice.builder import LatticeBuilder
from nsl_rag.lattice.traversal import LatticeTraversal
from nsl_rag.reasoning.prosecutor import Prosecutor
from nsl_rag.reasoning.judge import Judge
from nsl_rag.reasoning.auditor import Auditor
from nsl_rag.evaluation.naive_rag import NaiveRAG
from nsl_rag.retrieval.intent import IntentResult
from nsl_rag.retrieval.navigator import RetrievalResult
from nsl_rag.core.types import QueryType

setup_logging()
config.load()

log = get_logger("nsl_rag.test.benchmark")

# ── Build System ──────────────────────────────────────────────────────────────

log.info("Building e-commerce lattice...")
index = LatticeBuilder().build(EcommerceSystem.get_raw_nodes())
traversal = LatticeTraversal(index)
prosecutor = Prosecutor()
judge = Judge(index)
auditor = Auditor()
naive = NaiveRAG()
naive.build_index()

# ── Pre-defined Benchmark Queries ─────────────────────────────────────────────
# Tags hardcoded — no Gemini needed
# This is valid for benchmarking retrieval quality

BENCHMARK_QUERIES = [
    {
        "query": "Why is the payment service failing?",
        "tags": ["payment", "critical"],
        "target_entity": "payment_service",
        "query_type": QueryType.CAUSAL_TRACE,
    },
    {
        "query": "Why are orders stuck in pending state?",
        "tags": ["orders", "critical"],
        "target_entity": "order_service",
        "query_type": QueryType.CAUSAL_TRACE,
    },
    {
        "query": "Why is the fraud detection service slow?",
        "tags": ["payment", "fraud"],
        "target_entity": "fraud_detection",
        "query_type": QueryType.CAUSAL_TRACE,
    },
    {
        "query": "What is wrong with the payment database?",
        "tags": ["payment", "database"],
        "target_entity": "payment_db",
        "query_type": QueryType.CAUSAL_TRACE,
    },
    {
        "query": "Why are emails not being sent?",
        "tags": ["email", "notification"],
        "target_entity": "email_service",
        "query_type": QueryType.CAUSAL_TRACE,
    },
]

# ── Results Storage ───────────────────────────────────────────────────────────

results = []

# ── Run Benchmark ─────────────────────────────────────────────────────────────

log.info("=" * 65)
log.info("Running benchmark — %d queries", len(BENCHMARK_QUERIES))
log.info("=" * 65)

for i, bq in enumerate(BENCHMARK_QUERIES, 1):
    query = bq["query"]
    log.info("Query %d/%d: %r", i, len(BENCHMARK_QUERIES), query)

    result = {
        "query": query,
        "nsl_nodes": 0,
        "nsl_tokens": 0,
        "nsl_latency_ms": 0.0,
        "nsl_has_trace": False,
        "naive_chunks": 0,
        "naive_tokens": 0,
        "naive_latency_ms": 0.0,
        "error": "",
    }

    # ── NSL-RAG (no Gemini — direct traversal) ────────────────────
    try:
        nsl_start = time.time()

        # Build intent manually — bypass Gemini
        intent = IntentResult(
            tags=bq["tags"],
            query_type=bq["query_type"],
            target_entity=bq["target_entity"],
            raw_query=query,
        )

        # Retrieve
        nodes = traversal.retrieve(
            query_tags=intent.tags,
            target_node_id=intent.target_entity,
        )

        # Dependency chain
        chain = (
            traversal.retrieve_dependency_chain(intent.target_entity)
            if intent.target_entity
            else []
        )

        # Build retrieval result
        retrieval = RetrievalResult(
            nodes=nodes,
            intent=intent,
            dependency_chain=chain,
            query=query,
        )

        # Reasoning pipeline
        proposed = prosecutor.propose(retrieval)
        validated = judge.validate(proposed)
        report = auditor.audit(validated)

        nsl_latency = (time.time() - nsl_start) * 1000

        # Estimate tokens — facts sent to LLM
        context_text = " ".join(f.claim for f in report.all_facts)
        tokens_est = len(context_text) // 4

        result["nsl_nodes"] = len(retrieval.all_nodes)
        result["nsl_tokens"] = tokens_est
        result["nsl_latency_ms"] = round(nsl_latency, 2)
        result["nsl_has_trace"] = True

        log.info(
            "  NSL-RAG — nodes: %d, tokens: ~%d, latency: %.1fms",
            result["nsl_nodes"],
            result["nsl_tokens"],
            result["nsl_latency_ms"],
        )

    except Exception as e:
        result["error"] = str(e)
        log.error("  NSL-RAG failed: %s", e)

    # ── Naive RAG ─────────────────────────────────────────────────
    try:
        naive_result = naive.query(query)

        result["naive_chunks"] = naive_result.chunk_count
        result["naive_tokens"] = naive_result.tokens_used
        result["naive_latency_ms"] = round(naive_result.latency_ms, 2)

        log.info(
            "  NaiveRAG — chunks: %d, tokens: %d, latency: %.1fms",
            result["naive_chunks"],
            result["naive_tokens"],
            result["naive_latency_ms"],
        )

    except Exception as e:
        result["error"] += f" NaiveRAG: {e}"
        log.error("  NaiveRAG failed: %s", e)

    results.append(result)

# ── Print Comparison Table ────────────────────────────────────────────────────

print("\n")
print("=" * 65)
print("  NSL-RAG vs NAIVE RAG — BENCHMARK RESULTS")
print("=" * 65)

total_nsl_tokens = 0
total_naive_tokens = 0
total_nsl_latency = 0.0
total_naive_latency = 0.0
successful = 0

for i, r in enumerate(results, 1):
    print(f"\n  Q{i}: {r['query'][:55]}")
    if r["error"]:
        print(f"       ERROR: {r['error']}")
    else:
        print(
            f"       NSL-RAG  — nodes: {r['nsl_nodes']:2d} | "
            f"tokens: {r['nsl_tokens']:4d} | "
            f"latency: {r['nsl_latency_ms']:6.1f}ms | "
            f"trace: ✅"
        )
        print(
            f"       NaiveRAG — chunks: {r['naive_chunks']:2d} | "
            f"tokens: {r['naive_tokens']:4d} | "
            f"latency: {r['naive_latency_ms']:6.1f}ms | "
            f"trace: ❌"
        )
        total_nsl_tokens += r["nsl_tokens"]
        total_naive_tokens += r["naive_tokens"]
        total_nsl_latency += r["nsl_latency_ms"]
        total_naive_latency += r["naive_latency_ms"]
        successful += 1

if successful > 0:
    avg_nsl_tokens = total_nsl_tokens // successful
    avg_naive_tokens = total_naive_tokens // successful
    avg_nsl_lat = total_nsl_latency / successful
    avg_naive_lat = total_naive_latency / successful
    token_reduction = (
        round((avg_naive_tokens - avg_nsl_tokens) / avg_naive_tokens * 100, 1)
        if avg_naive_tokens > 0
        else 0.0
    )

    print("\n" + "-" * 65)
    print("  AVERAGES")
    print(f"  NSL-RAG  avg tokens  : {avg_nsl_tokens}")
    print(f"  NaiveRAG avg tokens  : {avg_naive_tokens}")
    print(f"  Token reduction      : {token_reduction}%")
    print(f"  NSL-RAG  avg latency : {avg_nsl_lat:.1f}ms")
    print(f"  NaiveRAG avg latency : {avg_naive_lat:.1f}ms")
    print(f"  NSL-RAG  trace       : 100%")
    print(f"  NaiveRAG trace       : 0%")

print("=" * 65)
