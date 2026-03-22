# NSL-RAG
### Neuro-Symbolic Lattice Retrieval-Augmented Generation

> *Standard RAG retrieves. It does not reason. NSL-RAG does both.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Preview](https://img.shields.io/badge/status-research%20preview-orange.svg)]()

---

## What Is NSL-RAG?

NSL-RAG replaces vector similarity search in RAG systems with a **formally ordered knowledge lattice** and a **symbolic reasoning pipeline**.

Instead of finding text that *looks similar* to a query, NSL-RAG navigates a concept lattice to find nodes that are *logically relevant* — then validates every retrieved fact through a Prosecutor / Judge / Auditor pipeline before generating an answer.

The result: causally traced, contradiction-checked, explainable answers with full reasoning audit trails.

---

## The Problem With Standard RAG
```
User: "Why is the payment service failing?"

Standard RAG:
→ Finds chunks that mention "payment" and "failing"
→ Returns top-5 similar chunks (may include unrelated content)
→ LLM generates answer from noisy context
→ No reasoning trace. No validation. No causal chain.

NSL-RAG:
→ Extracts intent: tags=[payment, critical], entity=payment_service
→ Navigates lattice: payment_service → fraud_detection → payment_db
→ Prosecutor extracts 21 structured facts
→ Judge validates all 21 facts (confident/uncertain/invalid)
→ Auditor checks for contradictions
→ LLM generates answer from validated facts only
→ Returns: {answer, root_cause, confidence, trace, flags}
```

---

## Architecture
```
Natural Language Query
        ↓
[NEURAL] Intent Extractor (Gemini)
        ↓ tags, query_type, target_entity
[SYMBOLIC] Lattice Traversal
        ↓ retrieved nodes
[PROSECUTOR] Fact Extraction
        ↓ proposed facts
[JUDGE] Symbolic Validation
        ↓ confident / uncertain / invalid
[AUDITOR] Contradiction Detection
        ↓ clean facts + flags
[NEURAL] LLM Generation (Gemini)
        ↓
{answer, root_cause, confidence, trace, flags}
```

---

## Key Properties

| Property | Standard RAG | NSL-RAG |
|---|---|---|
| Retrieval method | Vector similarity | Lattice traversal |
| Multi-hop reasoning | ❌ | ✅ |
| Reasoning trace | ❌ | ✅ 100% |
| Contradiction detection | ❌ | ✅ |
| Fact validation | ❌ | ✅ P/J/A pipeline |
| Bounded retrieval | ❌ | ✅ Formally provable |
| Explainability | Low | High |

---

## Benchmark Results

Evaluated on 5 DevOps incident queries — retrieval layer only (no LLM calls):

| Metric | NSL-RAG | Naive RAG |
|---|---|---|
| Reasoning trace | **100%** | 0% |
| Avg latency (retrieval) | 18.9ms | 0.8ms |
| Fact validation | **Yes** | No |
| Contradiction detection | **Yes** | No |
| Bounded retrieval proof | **Yes** | No |

*Full benchmark with LLM generation metrics coming in v0.2.0*

---

## Comparison With Graph-Based RAG Approaches

| Property | Standard RAG | GraphRAG | HippoRAG | NSL-RAG |
|---|---|---|---|---|
| Retrieval structure | None | Knowledge Graph | Knowledge Graph | Concept Lattice |
| Formal ordering | ❌ | ❌ | ❌ | ✅ |
| Bounded retrieval proof | ❌ | ❌ | ❌ | ✅ |
| Fact validation layer | ❌ | ❌ | ❌ | ✅ P/J/A |
| Contradiction detection | ❌ | ❌ | ❌ | ✅ |
| Causal chain traversal | ❌ | Partial | ❌ | ✅ |
| Reasoning trace | ❌ | Partial | ❌ | ✅ 100% |

### Why A Lattice Beats A Knowledge Graph For Reasoning

A knowledge graph stores connections. A lattice stores **ordered** 
connections — every concept has a provable position relative to 
every other concept.

This formal ordering enables **bounded retrieval** — the guarantee 
that irrelevant nodes are mathematically excluded from results. 
GraphRAG uses graph proximity heuristics. NSL-RAG uses formal 
lattice constraints. The difference is the difference between 
"probably relevant" and "provably relevant."

### Roadmap

- **v0.1.0** *(current)* — Proof of concept vs Standard RAG
- **v0.2.0** *(planned)* — Formal benchmark vs GraphRAG, HippoRAG, RAPTOR
- **v0.3.0** *(planned)* — Automatic lattice construction via FCA
- **v1.0.0** *(planned)* — Production release + arXiv paper

---

## Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/nsl-rag.git
cd nsl-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

Create `.env` from template:
```bash
copy .env.example .env
# Add your GEMINI_API_KEY to .env
```

Run the full pipeline:
```python
from nsl_rag import NSLRAGPipeline

pipeline = NSLRAGPipeline()
response = pipeline.query("Why is the payment service failing?")

print(response.answer)
print(response.root_cause)
print(response.confidence)
for step in response.trace:
    print(f"  → {step}")
```

---

## Example Output
```
QUERY: Why is the payment service failing?

ANSWER:
The Payment Service is failing due to a dependency cascade
triggered by the Flash Sale v2.1 deployment at 14:00, which
increased traffic 10x. This caused Fraud Detection timeouts
(threshold: 2 seconds) and Payment DB connection pool
exhaustion, resulting in 503 errors.

ROOT CAUSE: Flash Sale deployment caused 10x traffic spike
            overwhelming Fraud Detection and Payment DB.

CONFIDENCE: HIGH (0.90)

AFFECTED: payment_service, fraud_detection, payment_db

REASONING TRACE:
  → Intent: tags=[payment, critical], type=causal_trace
  → Lattice traversal retrieved 3 nodes
  →   ↳ SERVICE | Payment Service
  →   ↳ SERVICE | Fraud Detection Service
  →   ↳ DATABASE | Payment Database
  → Dependency chain: payment_service → api_gateway → ecommerce_root
  → 21 facts validated — clean audit
```

---

## Project Structure
```
nsl_rag/
├── lattice/          # Knowledge lattice — node, index, builder, traversal
├── retrieval/        # Intent extraction (Gemini) + lattice navigation
├── reasoning/        # Prosecutor → Judge → Auditor pipeline
├── generation/       # LLM generation + structured output formatter
├── evaluation/       # NaiveRAG baseline + benchmark metrics
├── data/             # Synthetic e-commerce dataset (16 nodes)
├── config/           # settings.yaml + config loader
└── core/             # Logger, exceptions, shared types
```

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Running The Benchmark
```bash
python scripts/test_benchmark.py
```

---

## Research Background

NSL-RAG is grounded in:
- **Formal Concept Analysis** (Wille, 1982) — mathematical basis for lattice construction
- **Neuro-Symbolic AI** (Garcez et al., 2019) — combining neural and symbolic reasoning
- **RAG** (Lewis et al., 2020) — retrieval-augmented generation foundation

A research paper formalising the NSL-RAG framework is in preparation.

---

## Status

- [x] Lattice layer — node, index, builder, traversal
- [x] Retrieval layer — Gemini intent extraction with caching
- [x] Reasoning layer — Prosecutor / Judge / Auditor pipeline
- [x] Generation layer — structured output with reasoning trace
- [x] Evaluation layer — NaiveRAG baseline + benchmark
- [ ] Automatic lattice construction via FCA
- [ ] REST API (FastAPI)
- [ ] pip package release
- [ ] arXiv paper submission

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of ongoing research into formally bounded retrieval systems.*